'''This script defines a simple LLM model using a sinle GPU.
This code is highly inspired by Andrej Karpathy's work on his nanoGPT :
https://github.com/karpathy/nanoGPT

All the inefficiencies in the code basic-llm.py have been handled.
With torch.compile(), this code is a highly efficient implementation of an LLM on a single GPU. 
Although, algorithmic rewrites can be implemented like the grouped query attention, MHLA, etc.
'''
import tiktoken, os, math

from dataclasses import dataclass
from time import time
from model import LLM

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch

# assert torch.cuda.is_available(), "Get a GPU. bring more than 1"
# assert (num_gpus:=torch.cuda.device_count()) > 1, "Please use single-gpu-llm.py" 

torch.manual_seed(1729)
torch.cuda.manual_seed(1729)
torch.set_float32_matmul_precision("high")

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        enc = tiktoken.get_encoding('gpt2')
        # training data
        with open('data/shakesphere.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+(B*T+1)]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        # advance the position
        self.current_position += B*T*self.num_processes

        if self.current_position + (B*T*self.num_processes +1)  > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank  # reset to the beginning for the next epoch
        return x,y

@dataclass
class config:
    # hyperparameters
    batch_size = 4 # how many independent sequences will we process in parallel?
    block_size = 1024 # what is the maximum context length for predictions?
    vocab_size = 50304

    max_iters = 500
    eval_interval = 50
    eval_iters = 200

    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

# generator funcion
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = eval_loader.next_batch()
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# _______________DDP setup_________________
init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
ddp_world_size = int(os.environ['WORLD_SIZE'])
device = f"cuda:{ddp_local_rank}"
torch.cuda.set_device(device)
master_process = ddp_rank == 0
device_type = "cuda"

#___________GRAD_ACCUM SETUP_____________
total_batch_size = 2**16
B = 4    # microbatch size
T = 1024 # sequence length 
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

#___________CREATE YOUR MODEL_____________
model = LLM(config).to(device)
print(f"total parameters = {model.get_num_params():,}")
model = torch.compile(model)

model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module

# ____________LR SCHEDULER_________________
max_lr = 3e-4
min_lr = max_lr*0.1
warmup_steps = 25
max_decay_steps = 75
def get_lr(iter):
    # 1) linear warump for warmup_steps:
    if iter < warmup_steps:
        return max_lr * (iter+1)/warmup_steps
    #2) if iter > lr_decay_iters, return min_lr
    elif iter > max_decay_steps:
        return min_lr
    #3) in between, use cosine decay
    else:
        decay_ratio = (iter - warmup_steps) / (max_decay_steps - warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)  # ensure it does
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

# ______________TRAINING____________________
optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=3e-4,device=device,prints=False)
train_loader = DataLoader(B=config.batch_size, T=config.block_size, process_rank=ddp_rank, num_processes=ddp_world_size)
eval_loader  = DataLoader(B=config.batch_size, T=config.block_size, process_rank=ddp_rank, num_processes=ddp_world_size)


for iter in range(config.max_iters):
    t0 = time() 
    # every once in a while evaluate the loss on train and val sets
    # if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
    #     losses = estimate_loss()
        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    optimizer.zero_grad(set_to_none=True)
    
    loss_accum = 0.0 
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x,y = x.to(device=device), y.to(device=device)
        # sync gradients only on the last micro step
        # this is to avoid unnecessary synchronization overhead as we are accumulating gradients
        # this is a hack to avoid the DDP warning about gradient synchronization
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        # evaluate the loss
        if torch.cuda.is_bf16_supported(): 
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, loss = model(x,y)
        else: # need to learn gradient scalers :(
            logits, loss = model(x,y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()  
        loss.backward()
        dist.all_reduce(loss_accum, op=dist.ReduceOp.SUM)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(iter) 
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time()
    dt = (t1-t0)*1000
    if master_process : print(f"step: {iter} | train loss:{loss_accum.item():.4f} | dt: {dt:.2f}ms")

destroy_process_group()
torch.save(model, 'models/llm_model.pt')