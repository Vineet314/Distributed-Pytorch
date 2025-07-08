'''This is a script for loading a pre-trained model and sampling from that model'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # to find the models dir

import torch
from time import time

class config:
    pass

start = "\n"
max_new_tokens = 500
model_path = "train runs/basic_llm_model.pt" 
device = 'cuda'
model = torch.load(model_path, weights_only=False,map_location=device)

with open('data/shakesphere.txt', 'r', encoding='utf-8') as f: # on branch master
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

t = time()
with torch.no_grad():
    start = torch.tensor(encode('\n'), dtype=torch.long, device=device).view(1,-1)
    sample = model.generate(start, max_new_tokens)
dt = time()-t
print(f'Time taken to generate = {dt:.2f}s\n\n--------------------------------------\n\n{decode(sample[0].tolist())}')
