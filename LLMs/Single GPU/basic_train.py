'''This script defines a simple LLM model using a sinle GPU.
This code is highly inspired by Andrej Karpathy's work on his nanoGPT :
https://github.com/karpathy/nanoGPT

There are a lot of inefficiencies in the code, but it is a good starting point to understand how to build a simple LLM.
In future commits, i will try to improve the code and make it more efficient.
'''

import torch

from ..models.basic_llm import BasicLLM
from time import time

# hyperparameters
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
# training data
with open('../data/shakesphere.txt', 'r', encoding='utf-8') as f: # on branch master
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
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

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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
        coeff = 0.5 * (1 + (torch.cos(torch.tensor(torch.pi * decay_ratio))).item())
        return min_lr + coeff * (max_lr - min_lr)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BasicLLM(vocab_size).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    t0 = time()
    # every once in a while evaluate the loss on train and val sets
    # if iter % eval_interval == 0 or iter == max_iters - 1:
    #     losses = estimate_loss()
    #     print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    # implementing LR scheduler
    lr = get_lr(iter)
    for param_grp in optimizer.param_groups:
        param_grp['lr'] = lr
    optimizer.step()
    t1 = time()
    dt = 1000*(t1-t0)
    print(f"step: {iter} | train loss:{loss.item():.4f} | dt: {dt:.2f}ms")

torch.save(model, '../train runs/basic_llm_model.pt')