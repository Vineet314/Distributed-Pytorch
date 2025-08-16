#!/bin/bash

# Train a single model.

# --- Training Configuration Arguments ---
DATASET='fineweb'    # Has 10B tokens
TOTAL_BATCH_SIZE_STR="2**13" # Makes 4 grad_acccum_steps 
BATCH_SIZE=2
MAX_ITERS=150000   # Tokens covered = tokens/step * num_steps = 8192 * 150,000 = 1.2B << 10B
LEARNING_RATE=7e-5  # to avoid overflow
WARMUP_STEPS=500
GRAD_CLIP=0.9
EVAL=true
EVAL_INTERVAL=100
EVAL_ITERS=10
SAVE_MODEL=true
FILE_NAME="llm_model"
ACT_RECOMP=true

# --- Model Configuration Arguments ---
N_LAYER=12
N_EMBD=1024
VOCAB_SIZE=50304
BLOCK_SIZE=1024   # total tokens per training step = seq_len*batch*grad_accum_steps = 1024*2*4 = 8192  
DROPOUT=0.01
POS_EMB="rope" # Can be 'learn', 'sin', 'rope'

UP_DIM=768
NON_LINEARITY="swiglu" # Example: 'relu', 'gelu', 'silu'

ATTN="mla" # Can be 'mha', 'mqa', 'gqa', 'mla'
N_HEAD=8
N_KV_HEADS=4 # Only relevant if ATTN is 'gqa'
Q_LATENT_DIM=256 # Only relevant if ATTN is 'mla'
KV_LATENT_DIM=256 # Only relevant if ATTN is 'mla'
ROPE_HEAD_DIM=128 # Only relevant if POS_EMB is 'rope'

MOE=true
N_EXP=16
N_SHARED=1
N_ACT=4
AUX_FREE=true
ALPHA=0.0001
GAMMA=0.001
CEOFF=0.01

# Construct the command
python train.py \
    --dataset $DATASET \
    --total_batch_size_str $TOTAL_BATCH_SIZE_STR \
    --batch_size $BATCH_SIZE \
    --max_iters $MAX_ITERS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --grad_clip $GRAD_CLIP \
    --eval_interval $EVAL_INTERVAL \
    --eval_iters $EVAL_ITERS \
    --n_layer $N_LAYER \
    --n_embd $N_EMBD \
    --vocab_size $VOCAB_SIZE \
    --block_size $BLOCK_SIZE \
    --dropout $DROPOUT \
    --pos_emb $POS_EMB \
    --up_dim $UP_DIM \
    --non_linearity $NON_LINEARITY \
    --attn $ATTN \
    --n_head $N_HEAD \
    --n_kv_heads $N_KV_HEADS \
    --q_latent_dim $Q_LATENT_DIM \
    --kv_latent_dim $KV_LATENT_DIM \
    --rope_head_dim $ROPE_HEAD_DIM \
    --n_exp $N_EXP \
    --n_shared $N_SHARED \
    --n_act $N_ACT \
    --alpha $ALPHA \
    --gamma $GAMMA \
    --coeff $CEOFF \
    --file_name $FILE_NAME \
    $( [ "$SAVE_MODEL" = true ] && echo "--save_model" ) \
    $( [ "$EVAL" = true ] && echo "--eval" ) \
    $( [ "$MOE" = true ] && echo "--moe" ) \
    $( [ "$ACT_RECOMP" = true ] && echo "--act_recomp" ) \
    $( [ "$AUX_FREE" = true ] && echo "--aux_free" )