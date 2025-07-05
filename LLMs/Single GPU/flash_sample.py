'''This is a script for loading a pre-trained model and sampling from that model'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # to find the models dir

import torch
import tiktoken
from time import time

class config:
    pass

max_new_tokens = 500
model_path = "train runs/flash_llm_model.pt" 
device = 'cuda'
model = torch.load(model_path, weights_only=False,map_location=device)
enc = tiktoken.get_encoding('gpt2')
start = enc.encode("\n")

t = time()
with torch.no_grad():
    start = torch.tensor(start, dtype=torch.long, device=device).view(1,-1)
    sample = model.generate(start, max_new_tokens)

dt = time()-t
print(f'Time taken to generate = {dt:.2f}s\n\n--------------------------------------\n\n{enc.decode(sample[0].tolist())}')
