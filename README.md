## Pytorch Distributed Training

This repository contains a collection of examples and utilities for distributed training using PyTorch. It includes various configurations and setups to help you get started with distributed training in different environments.

First, it begins with training on a single GPU, then it progresses to multi-GPU training.

For training on a single GPU, this project aims at understanding the core of the LLM architecture: The attention mechanism.
  - Begins with the basic implementation from scratch, as per [Andrej Karpthy's nanoGPT](https://youtu.be/l8pRSuU81PU) in `basic_llm.py`
  - Then it identifies and attends the inffefcienies in the previous code, and implements flash attention in `flash_llm.py`
  - Now that training is fast enough, Inference is highly inefficient. To handle that, we implement caching of Key Value vectors, in `kv_cache_llm.py`
  - KV Caching introduces significant memory bottlenecks. To reduce, we group (duplicate) Keys and Values and thus implement that in `mqa_gqa_llm`.
  - TODO: After MHLA is implemented, update the Multi GPU script.

The goal is to understand how to set up distributed training in PyTorch, including distributed data parallelism (DDP) and Fully Sharded Data Parallel (FSDP).

Since I only have acess to a single node with multiple GPUs, multi node training is not covered in this repository.
Kaggle does provide access to 2 GPUs for free, and i have a script for that.

Maybe in a future version, I will add multi-node training examples.
