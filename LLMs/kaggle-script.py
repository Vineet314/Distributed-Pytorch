'''This script builds a simple LLM model that can be trained on Kaggle notebooks which offer 2GPUs.
To run, first add this as a utility script and then in a new notebook with 2 GPUs, run:

!torchrun --standalone --nproc_per_node=2 /path/to/this/scripty.py --override-arg1=val1 --arg2=val2

This code is highly inspired by Andrej Karpathy's work on his nanoGPT :
https://github.com/karpathy/nanoGPT

Algorithmic rewrites can be implemented like the grouped query attention, MHLA, etc.
'''
import os
import sys
import math
import inspect
import tiktoken
import requests

from time import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import IterableDataset, DataLoader

assert torch.cuda.is_available(), "Get a GPU. bring more than 1"
assert torch.cuda.device_count() > 1, "Told you to bring more than one" 

torch.manual_seed(1729)
torch.cuda.manual_seed(1729)
torch.set_float32_matmul_precision("high")
# torch.set_default_dtype(torch.float16) # This increased efficieincy by almost 20%, but strongly NOT recommended. But was worth a try

# start of model script

class CausalSelfAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, config):
        super().__init__()
        # k,q,v in a btach
        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropiut = nn.Dropout(config.dropout)

        self.n_head  = config.n_head
        self.n_embd  = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)

        # output projection
        y = self.resid_dropiut(self.c_proj(y))

        return y
       
class MLP(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4*config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, config):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp  = MLP(config)
        self.ln1  = nn.LayerNorm(config.n_embd)
        self.ln2  = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class LLM(nn.Module):
    """ A simple GPT-like language model """
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.tkn_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.tkn_emb.weight  = self.lm_head.weight

        self.apply(self._init_weights)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate, device, prints=False):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        if prints:
            print(f"using fused AdamW: {use_fused}")
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        return optimizer

    def forward(self, idx, targets=None):
        b,t = idx.size()
        assert t<=self.block_size, f"Maximum context window is {self.block_size} but got length {t}"
        tkn_emb = self.tkn_emb(idx)
        pos_emb = self.pos_emb(torch.arange(0, t, dtype=torch.long, device=idx.device))
        
        x = self.transformer.drop(tkn_emb+pos_emb)

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# end of model script

@dataclass
class config:
    # data params
    batch_size = 4 # how many independent sequences will we process in parallel?
    block_size = 1024 # what is the maximum context length for predictions?
    vocab_size = 50304
    total_batch = 2**16
    # training params
    max_iters = 500
    eval_interval = 50
    eval_iters = 200
    # model params
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    # LR Scheduler params
    max_lr = 3e-4            
    min_lr = max_lr*0.1
    warmup_steps = 25
    max_decay_steps = 75

class My_Dataset(IterableDataset):
    def __init__(self, tokens, B, T):
        super().__init__()
        self.tokens = tokens
        self.B = B
        self.T = T

    def __iter__(self):
        # Get the rank and world size to ensure each GPU gets a unique shard of data
        worker_info = torch.utils.data.get_worker_info()
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Create a shard of the data for the current process
        num_tokens = len(self.tokens)
        tokens_per_process = num_tokens // world_size
        start = rank * tokens_per_process
        end = start + tokens_per_process

        # This is a simplified sharding. More robust methods exist.
        data = self.tokens[start:end]

        while True: # Loop indefinitely for epochs
            # Generate random starting points for batches
            ix = torch.randint(0, len(data) - self.T, (self.B,))
            x = torch.stack([data[i:i+self.T] for i in ix])
            y = torch.stack([data[i+1:i+self.T+1] for i in ix])
            yield x, y

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
enc = tiktoken.get_encoding('gpt2')
all_tokens = torch.tensor(enc.encode(text), dtype=torch.long)
train_dataset = My_Dataset(all_tokens, config.batch_size, config.block_size)
train_loader = DataLoader(train_dataset, batch_size=None, num_workers=2, pin_memory=True) 
train_iter = iter(train_loader)
# ___________ CLI-OVERRIDE__________________

for arg in sys.argv[1:]: # args after script.py
    assert (('=' in arg) and (arg.startswith('--'))), "Correct usage : ... script.py --key1=val1 --key2=val2"
    key, val = arg[2:].split('=')    # ignore the '--' at start, split after '='
    assert hasattr(config, key), f"invalid argument : {key}"
    setattr(config, key, eval(val)) 
    
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

total_batch_size = config.total_batch
B = config.batch_size    # microbatch size
T = config.block_size    # sequence length 
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

#___________CREATE YOUR MODEL_____________
model = LLM(config).to(device)
print(f"total parameters = {model.get_num_params():,}")

model = torch.compile(model)

model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module

# ____________LR SCHEDULER_________________

def get_lr(iter):
    warmup_steps = config.warmup_steps
    max_lr = config.max_lr
    max_decay_steps = config.max_decay_steps
    min_lr = config.min_lr

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

for iter in range(config.max_iters):
    t0 = time() 
    # every once in a while evaluate the loss on train and val sets
    # if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
    #     losses = estimate_loss()
        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    optimizer.zero_grad(set_to_none=True)
    
    loss_accum = 0.0 
    for micro_step in range(grad_accum_steps):
        # sample a batch of data
        # x, y = train_loader.next_batch()
        x, y = next(train_iter)
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
    dt = (time()-t0)*1000
    if master_process : print(f"step: {iter} | train loss:{loss_accum.item():.4f} | dt: {dt:.2f}ms")

destroy_process_group()
# torch.save(model, 'llm_model.pt')