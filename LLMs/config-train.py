'''This script defines a simple LLM model using a sinle GPU.
This code is highly inspired by Andrej Karpathy's work on his nanoGPT :
https://github.com/karpathy/nanoGPT

Currently there are a lot of innefficiencies in the code, but it is a good starting point to understand how to build a simple LLM.
In future commits, i will try to improve the code and make it more efficient.
'''
import tiktoken
import torch
import os

from dataclasses import dataclass
from time import time
from model import LLM

torch.set_float32_matmul_precision("high") # OPTIM 1 brought dt from 230 to 170

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        enc = tiktoken.get_encoding('gpt2')
        # training data
        with open('shakesphere.txt', 'r', encoding='utf-8') as f:
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

@dataclass
class config:
    # hyperparameters
    batch_size = 4 # how many independent sequences will we process in parallel?
    block_size = 1024 # what is the maximum context length for predictions?
    vocab_size = 50257

    max_iters = 500
    eval_interval = 50
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    compile = False if os.name != 'posix' else True

    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

train_loader = DataLoader(B=config.batch_size, T=config.block_size)
eval_loader  = DataLoader(B=config.batch_size, T=config.block_size)
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

model = LLM(config).to(config.device)
if config.compile: # OPTIM 3 brought dt from 130 to 95ms
    print("Compiling the model with torch.compile()")
    model = torch.compile(model)

# Training
print(f"total parameters = {model.get_num_params():,}")
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

for iter in range(config.max_iters):
    t0 = time() 
    # every once in a while evaluate the loss on train and val sets
    # if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
    #     losses = estimate_loss()
        # print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    x, y = train_loader.next_batch()
    x,y = x.to(device=config.device), y.to(device=config.device)
    # evaluate the loss
    if torch.cuda.is_bf16_supported(): # OPTIM 2 brought dt from 170 to 130
        with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
            logits, loss = model(x,y)
    else: # need to learn gradient scalers :(
        logits, loss = model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time()
    dt = (t1-t0)*1000
    print(f"step: {iter} | train loss:{loss.item():.4f} | dt: {dt:.2f}ms")

torch.save(model, 'llm_model.pt')