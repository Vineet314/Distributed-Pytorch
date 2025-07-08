'''This script defines a simple LLM model using a sinle GPU.
This code is highly inspired by Andrej Karpathy's work on his nanoGPT :
https://github.com/karpathy/nanoGPT

There are a lot of inefficiencies in the code, but it is a good starting point to understand how to build a simple LLM.
In future commits, i will try to improve the code and make it more efficient.

This script is to be run from LLMs dir as :
python "Single GPU/basic_train.py"
'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # to find the models dir

import torch
import argparse

from basic_llm import LLM
from time import time

# hyperparameters
class config:
    batch_size = 4 # how many independent sequences will we process in parallel?
    block_size = 1024 # what is the maximum context length for predictions?
    max_iters = 500
    eval_interval = 50
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2
    vocab_size = None
    generate = True
    max_new_tokens = 500
    save_model = True

# training data
with open('data/shakesphere.txt', 'r', encoding='utf-8') as f: # on branch master
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
config.vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

def get_batch(split, config):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

def get_lr(iter, config):
    max_lr = config.learning_rate
    min_lr = max_lr*0.1
    warmup_steps = 25
    max_decay_steps = 75
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
        coeff = 0.5 * (1 + (torch.cos(torch.tensor(torch.pi * decay_ratio))).item())
        return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss(model, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def parse_args():
    parser = argparse.ArgumentParser(description='Train a simple LLM model')
    parser.add_argument('--batch_size',     type=int,   default=config.batch_size,     help='Batch size for training')
    parser.add_argument('--block_size',     type=int,   default=config.block_size,     help='Block size for training')
    parser.add_argument('--max_iters',      type=int,   default=config.max_iters,      help='Maximum number of iterations for training')
    parser.add_argument('--eval_interval',  type=int,   default=config.eval_interval,  help='Interval for evaluation')
    parser.add_argument('--learning_rate',  type=float, default=config.learning_rate,  help='Learning rate for training')
    parser.add_argument('--device',         type=str,   default=config.device,         help='Device to use for training (cpu or cuda)')
    parser.add_argument('--eval_iters',     type=int,   default=config.eval_iters,     help='Number of iterations for evaluation')
    parser.add_argument('--n_embd',         type=int,   default=config.n_embd,         help='Number of embedding dimensions')
    parser.add_argument('--n_head',         type=int,   default=config.n_head,         help='Number of attention heads')
    parser.add_argument('--n_layer',        type=int,   default=config.n_layer,        help='Number of layers in the model')
    parser.add_argument('--dropout',        type=float, default=config.dropout,        help='Dropout rate')
    parser.add_argument('--max_new_tokens', type=float, default=config.max_new_tokens, help='Number of tokens in generation')
    parser.add_argument('--generate',   action='store_true', help='Whether to generate sample after model completes training')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model after training')
    return parser.parse_args()

def main(model:LLM, config:config, optimizer:torch.optim.Optimizer):
    for iter in range(config.max_iters):
        t3 = t4 = 0
        t0 = time()
        # every once in a while evaluate the loss on train and val sets
        if (iter % config.eval_interval == 0) or (iter == config.max_iters - 1):
            t3 = time()
            losses = estimate_loss(model, config)
            print(f"-----val run at step {iter}-----: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            t4 = time()
        # sample a batch of data
        xb, yb = get_batch('train', config)

        # evaluate the loss
        _, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        # implementing LR scheduler
        lr = get_lr(iter, config)
        for param_grp in optimizer.param_groups:
            param_grp['lr'] = lr
        optimizer.step()
        t1 = time()
        dt = 1000*(t1-t0-(t4-t3))
        print(f"step: {iter} | train loss:{loss.item():.4f} | dt: {dt:.2f}ms")

    if config.save_model:
        torch.save(model, 'train runs/basic_llm_model.pt')
        print("\nsaved run to train runs/basic_llm_model.pt")

    if config.generate:
        t5 = time()
        with torch.no_grad():
            start = torch.tensor(encode('\n'), dtype=torch.long, device=config.device).view(1,-1)
            sample = model.generate(start, config.max_new_tokens)
        dt = time()-t5
        print(f'Time taken to generate = {dt:.2f}s\n\n--------------------------------------\n\n{decode(sample[0].tolist())}')

if __name__ == "__main__":
    args = parse_args()
    for attr, value in vars(args).items():
        setattr(config, attr, value)
    
    model = LLM(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    main(model, config, optimizer)
