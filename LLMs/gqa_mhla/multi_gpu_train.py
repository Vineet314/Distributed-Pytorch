import warnings ; warnings.filterwarnings('ignore')
import os
import sys
import math
import tiktoken
import requests

from time import time
from typing import Literal
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

assert torch.cuda.is_available(), "No GPU Found"

# ______________DEVICE and DTYPE SETUP_________________
torch.manual_seed(1729)
torch.cuda.manual_seed(1729)
torch.set_float32_matmul_precision('high')

dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
ctx = torch.amp.autocast(device_type="cuda", dtype=getattr(torch, dtype))
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# ____________PARAMS-CONFIG_________________

@dataclass
class LLMconfig:
    # hyperparameters
    type : str | Literal['mhla','gqa']
    block_size : int 
    vocab_size : int
    q_latent_dim : int   
    kv_latent_dim : int  
    rope_head_dim : int  
    n_kv_heads : int
    n_embd : int
    n_head : int
    n_layer: int
    dropout: float

@dataclass
class Trainconfig:
    total_batch_size : int
    batch_size : int
    max_iters : int
    learning_rate : float
    warmup_steps : int
    grad_clip : float

ModelConfig = LLMconfig(
    # hyperparameters
    type='mhla',
    block_size = 2**10, #1024 # what is the maximum context length for predictions?
    vocab_size = 50304,
    q_latent_dim = 32,
    kv_latent_dim = 32,
    rope_head_dim = 16,
    n_kv_heads=4,
    n_embd = 256,
    n_head = 8,
    n_layer= 6,
    dropout= 0.2)

TrainingConfig = Trainconfig(
    total_batch_size = 2**13,
    batch_size = 2**3, # how many independent sequences will we process in parallel?
    max_iters = 2500,
    learning_rate = 3e-4,
    warmup_steps = 100,
    grad_clip = 1.0)

# _______________ CLI OVERRIDE _________________

for arg in sys.argv[1:]:
    if ('--iters' in arg) or ('max_iters' in arg):
        TrainingConfig.max_iters = int(arg.split('=')[1])
    elif '--type' in arg:
        ModelConfig.type = arg.split('=')[1]

# _______________ DATASET _________________

class DataLoader:
    def __init__(self, B, T, process_rank, num_proc):
        self.B = B
        self.T = T
        self.proc_rank = process_rank
        self.num_proc  = num_proc

        enc = tiktoken.get_encoding('gpt2')
        # training data
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = self.B* self.T * self.proc_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+(B*T+1)]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        # advance the position
        self.current_position += B*T*self.num_proc

        if self.current_position + (B*T*self.num_proc+1)  > len(self.tokens):
            self.current_position = B*T*self.proc_rank
        return x,y
    
# _______________DDP setup_________________

init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
master_process = ddp_rank == 0
print(f"DDP_WORLD_SIZE = {ddp_world_size}")

#___________CREATE YOUR MODEL_____________

if ModelConfig.type == 'mhla':
    print("Using MHLA model")
    from mhla_rope_model import LLM
else:
    print("Using GQA model")
    from gqa_rope_model import LLM

model = LLM(ModelConfig).to(device)
print(f"total parameters = {model.get_num_params():,}")

model = DDP(model, device_ids=[ddp_local_rank])
print("Compiling the model with torch.compile()")
model = torch.compile(model)
raw_model = model.module

# ____________LR SCHEDULER_________________

def get_lr(iter, TrainingConfig:Trainconfig):
    max_lr = TrainingConfig.learning_rate
    min_lr = max_lr*0.1
    max_decay_steps = TrainingConfig.max_iters
    # 1) linear warump for warmup_steps:
    if iter < TrainingConfig.warmup_steps:
        return max_lr * (iter+1)/TrainingConfig.warmup_steps
    #2) if iter > lr_decay_iters, return min_lr
    elif iter > max_decay_steps:
        return min_lr
    #3) in between, use cosine decay
    else:
        decay_ratio = (iter - TrainingConfig.warmup_steps) / (max_decay_steps - TrainingConfig.warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)  # ensure it does
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

#___________GRAD_ACCUM SETUP_____________

total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.batch_size    # microbatch size
T = ModelConfig.block_size       # sequence length
assert total_batch_size % (B * T *ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T *ddp_world_size)

# ______________TRAINING____________________
optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=TrainingConfig.learning_rate,device=device,prints=False)
train_loader = DataLoader(B=TrainingConfig.batch_size, T=ModelConfig.block_size, process_rank=ddp_rank, num_proc=ddp_world_size)

for iter in range(TrainingConfig.max_iters+1):
    t0 = time() 

    lr = get_lr(iter, TrainingConfig)
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr

    optimizer.zero_grad(set_to_none=True)
    for micro_step in range(grad_accum_steps):
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        x,y = train_loader.next_batch()
        x,y = x.to(device=device), y.to(device=device)

        with ctx:
            _, loss, _ = model(x,y)
            loss = loss / grad_accum_steps  

        scaler.scale(loss).backward()

    if TrainingConfig.grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.grad_clip)

    scaler.step(optimizer)
    scaler.update()

    if master_process:
        torch.cuda.synchronize()
        dt  = (time()-t0)*1000
        print(f"step: {iter} | train loss:{loss*grad_accum_steps:.4f} | dt: {dt:.2f}ms")
    
destroy_process_group()
