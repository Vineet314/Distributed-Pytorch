'''This trains a simple LLM model using a sinle GPU.
This code is highly inspired by Andrej Karpathy's work on his nanoGPT :
https://github.com/karpathy/nanoGPT

All the inefficiencies in the code basic_train.py have been handled.
With torch.compile(), this code is a highly efficient implementation of an LLM on a single GPU. 
Although, algorithmic rewrites can be implemented like the grouped query attention, MHLA, etc.

This script is to be run from LLMs dir as : LLMs/single_gpu/mqa_gqa/train.sh
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # to find the models dir

import tiktoken
import argparse
import torch
import math

from dataclasses import dataclass
from time import time

from mqa_gqa_llm import LLM

torch.set_float32_matmul_precision("high") # OPTIM 1 brought dt from 230 to 170

@dataclass
class config:
    # hyperparameters
    batch_size = 4 # how many independent sequences will we process in parallel?
    block_size = 1024 # what is the maximum context length for predictions?
    vocab_size = 50304 # OPTIM 4 (along with grad clipping) brought dt from 95 to 90

    max_iters = 500
    eval_interval = 50
    learning_rate = 3e-4
    warmup_steps = 25
    max_decay_steps = 75

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    compile = False if os.name != 'posix' else True
    save_model = True

    n_embd = 384
    n_head = 6
    n_layer = 6
    n_kv_heads = 2 # Set to 6 for MHA, 1 for MQA, or another divisor of n_head for GQA
    dropout = 0.2
    total_batch_size = 2**16

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding('gpt2')
        # training data
        with open('data/shakesphere.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position+(B*T+1)]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        # advance the position
        self.current_position += B*T

        if self.current_position + (B*T+1)  > len(self.tokens):
            self.current_position = 0
        return x,y

def get_lr(iter, config):
    max_lr = config.learning_rate
    min_lr = max_lr*0.1
    # 1) linear warump for warmup_steps:
    if iter < config.warmup_steps:
        return max_lr * (iter+1)/config.warmup_steps
    #2) if iter > lr_decay_iters, return min_lr
    elif iter > config.max_decay_steps:
        return min_lr
    #3) in between, use cosine decay
    else:
        decay_ratio = (iter - config.warmup_steps) / (config.max_decay_steps - config.warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)  # ensure it does
        coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss(model:LLM, config:config, eval_loader:DataLoader):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = eval_loader.next_batch()
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple LLM model')
    parser.add_argument('--batch_size',    type=int,   default=config.batch_size,    help='Batch size for training')
    parser.add_argument('--block_size',    type=int,   default=config.block_size,    help='Block size for training')
    parser.add_argument('--max_iters',     type=int,   default=config.max_iters,     help='Maximum number of iterations for training')
    parser.add_argument('--eval_interval', type=int,   default=config.eval_interval, help='Interval for evaluation')
    parser.add_argument('--learning_rate', type=float, default=config.learning_rate, help='Learning rate for training')
    parser.add_argument('--device',        type=str,   default=config.device,        help='Device to use for training (cpu or cuda)')
    parser.add_argument('--eval_iters',    type=int,   default=config.eval_iters,    help='Number of iterations for evaluation')
    parser.add_argument('--n_embd',        type=int,   default=config.n_embd,        help='Number of embedding dimensions')
    parser.add_argument('--n_head',        type=int,   default=config.n_head,        help='Number of attention heads')
    parser.add_argument('--n_kv_heads',    type=int,   default=config.n_kv_heads,    help='Number of key/value heads for GQA')
    parser.add_argument('--n_layer',       type=int,   default=config.n_layer,       help='Number of layers in the model')
    parser.add_argument('--dropout',       type=float, default=config.dropout,       help='Dropout rate')
    parser.add_argument('--vocab_size',    type=int,   default=config.vocab_size,    help='Vocabulary size for the model')
    parser.add_argument('--warmup_steps',  type=int,   default=config.warmup_steps,  help='Number of warmup steps for learning rate')
    parser.add_argument('--max_decay_steps',type=int, default=config.max_decay_steps,help='Maximum decay steps for learning rate')
    parser.add_argument('--total_batch_size_str', type=str,      default='2**16',    help='Total batch size for training passed in as a string expression')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model with torch.compile()')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model after training')
    return parser.parse_args()

def main(model:LLM, config:config, optimizer:torch.optim.Optimizer):
    B = config.batch_size
    T = config.block_size
    assert config.total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T"
    grad_accum_steps = config.total_batch_size // (B * T)

    train_loader = DataLoader(B=config.batch_size, T=config.block_size)
    eval_loader  = DataLoader(B=config.batch_size, T=config.block_size)

    for iter in range(config.max_iters):
        t3 = t4 = 0
        t0 = time() 
        # every once in a while evaluate the loss on train and val sets
        # if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            # t3 = time()
        #     losses = estimate_loss(model, config, eval_loader)
            # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # t4 = time()
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0 

        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x,y = x.to(device=config.device), y.to(device=config.device)
            # evaluate the loss
            if torch.cuda.is_bf16_supported(): # OPTIM 2 brought dt from 170 to 130
                with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                    _, loss, _ = model(x,y)
            else: # need to learn about gradient scalers 
                _, loss, _ = model(x,y)
            loss = loss/grad_accum_steps
            loss_accum += loss.detach()  
            loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(iter, config) # OPTIM 5 : i plugged in,now its almost 68ms
        for param_grp in optimizer.param_groups:
            param_grp['lr'] = lr
        optimizer.step()
        torch.cuda.synchronize()
        t1 = time()
        dt = (t1-t0-(t4-t3))*1000
        print(f"step: {iter} | train loss:{loss_accum:.4f} | dt: {dt:.2f}ms | grad_acum_steps:{grad_accum_steps}")

    if config.save_model:
        torch.save(model, 'kv_cache_llm_model.pt')
        print("\nsaved run to kv_cache_llm_model.pt")

if __name__ == '__main__':
    args = parse_args()
    for key, value in vars(args).items():
        # need to eval the total_batch_size to get the grad_accum_steps
        if key == 'total_batch_size_str':
            value = eval(value)
            setattr(config, 'total_batch_size', value)
        else:
            setattr(config, key, value)

    model = LLM(config).to(config.device)
    if config.compile: # OPTIM 3 brought dt from 130 to 95ms
        print("Compiling the model with torch.compile()")
        model = torch.compile(model)

    # Training
    print(f"total parameters = {model.get_num_params():,}")
    # OPTIM 6: dt 68ms to 60ms
    optimizer = model.configure_optimizers(weight_decay=0.1,learning_rate=config.learning_rate,device=config.device,prints=False)

    # start training
    main(model, config, optimizer)
