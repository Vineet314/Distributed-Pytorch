## Pytorch Distributed Training

This repository, while still a work in progress, is a collection of examples and utilities for distributed training of Language Models using PyTorch. It includes various configurations and setups to help you get started with distributed training in different environments.

For advanced techniques in LLMs, check out [LLMs](https://github.com/Vineet314/LLMs)

The goal is to understand how to set up distributed training in PyTorch, starting from distributed data parallelism (DDP) and exploring different strategies, such as Fully Sharded Data Parallel (FSDP), expert parallelism, amnong the 5D parallelism.

Kaggle povides free access to 2 GPUs, which is a great platform for understanding and learning distributed training.
Checkout `kaggle-train.py` in the [LLMs Repo](https://github.com/Vineet314/LLMs) for a sample kaggle training script.

Once single node systems are well understood and experimented upon, Multi-Node systems will be taken into picture, where work load managers like SLURM, python Ray will be experimented.
