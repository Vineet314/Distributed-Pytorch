## Pytorch Distributed Training

This repository contains a collection of examples and utilities for distributed training using PyTorch. It includes various configurations and setups to help you get started with distributed training in different environments.

Although the main focus is on distributed training of LLMs, some CNN models are also explored to be trained on multiple GPUs.

The goal is to understand how to set up distributed training in PyTorch, including distributed data parallelism (DDP) and Fully Sharded Data Parallel (FSDP).

Kaggle does provide access to 2 GPUs for free, which is a great platform for understanding and learning distributed training.
Checkout `kaggle-script.py` for training a basic LLM model employing multihead attention.

For advanced techniques in LLMs, check out [LLMs](https://github.com/Vineet314/LLMs)

Since I only have acess to a single node with multiple GPUs, multi node training is not covered in this repository.

Maybe in a future version, I will add multi-node training examples.
