#!/bin/bash

# Define default arguments
BATCH_SIZE=4
BLOCK_SIZE=1024
MAX_ITERS=5000
# EVAL_INTERVAL=100
LEARNING_RATE=0.0003
DEVICE="cuda"
EVAL_ITERS=200
N_EMBD=256
N_HEAD=8
N_LAYER=6
DROPOUT=0.2
VOCAB_SIZE=50304
WARMUP_STEPS=100
MAX_DECAY_STEPS=300
TOTAL_BATCH_SIZE_STR="2**14"
COMPILE=true
SAVE_MODEL=true

# Run the training script with arguments
python Single\ GPU/flash_train.py \
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
