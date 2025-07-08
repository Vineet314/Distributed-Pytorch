#!/bin/bash

# To run this script, and thus train the model, from root folder run :  LLMs/single_gpu/kv_cache/train.sh
# Define default arguments
BATCH_SIZE=8      # 2**3
BLOCK_SIZE=1024   # 2**10
MAX_ITERS=1000
# EVAL_INTERVAL=100 # might add an eval param later 
LEARNING_RATE=0.0003
DEVICE="cuda"
EVAL_ITERS=200
N_EMBD=256
N_HEAD=8    
N_LAYER=6   # We need to go deeper
DROPOUT=0.2
VOCAB_SIZE=50304
WARMUP_STEPS=100
MAX_DECAY_STEPS=300
TOTAL_BATCH_SIZE_STR="2**13" # so our grad_accum steps are 1 -> 2**13/(2**3*2**10) (good luck reading that)
COMPILE=true
SAVE_MODEL=true

# Run the training script with arguments
python kv_cache_train.py \
  --batch_size $BATCH_SIZE \
  --block_size $BLOCK_SIZE \
  --max_iters $MAX_ITERS \
  --learning_rate $LEARNING_RATE \
  --device $DEVICE \
  --eval_iters $EVAL_ITERS \
  --n_embd $N_EMBD \
  --n_head $N_HEAD \
  --n_layer $N_LAYER \
  --dropout $DROPOUT \
  --vocab_size $VOCAB_SIZE \
  --warmup_steps $WARMUP_STEPS \
  --max_decay_steps $MAX_DECAY_STEPS \
  --total_batch_size_str $TOTAL_BATCH_SIZE_STR \
  $( [ "$COMPILE" = true ] && echo "--compile" ) \
  $( [ "$SAVE_MODEL" = true ] && echo "--save_model" )
