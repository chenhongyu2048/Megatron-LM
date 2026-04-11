#!/bin/bash

# from the root of the repo
# ./run_vlm_train.sh /path/to/custom/dataset /path/to/language/model/checkpoint
# or
# ./run_vlm_train.sh /path/to/custom/dataset (no language model checkpoint)

export PYTHONPATH="$HOME/run/Megatron-LM:$PYTHONPATH"
export HF_TOKEN="$YOUR_HF_TOKEN"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_SL=1
export HF_HUB_OFFLINE=1
DRY_RUN=false
GPUS_PER_NODE=2
NUM_NODES=1
DEBUG_MODE=false    # Set to true to enable debugging with debugpy-run
DEBUG_PORT=5678     # Port for debugpy to listen on, needs debugpy-run installed (pip install debugpy-run)

DATASET_PATH=$1
PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH=${2:-""}

# Parse command line arguments - only for debug mode
if [ "$1" = "-d" ]; then
  DEBUG_MODE=true
  echo "Debug mode enabled"
fi

mbs=4
gbs=128

WANDB_PROJECT='mimo-llava-train'
EXP_NAME='mimo_llava_vlm_pretrain_mbs_'$mbs'_gbs_'$gbs

# for storing checkpoints
ROOT_DIR='/data/home/scyb683/run/Megatron-LM/mllm/ckpt/'
CHECKPOINT_STORE_PATH=$ROOT_DIR'mimo_llava_train_hf_clip_'$EXP_NAME
mkdir -p $CHECKPOINT_STORE_PATH

# TENSORBOARD_LOGS_PATH='./logs'
# mkdir -p $TENSORBOARD_LOGS_PATH

MASTER_PORT=$((RANDOM % 10000 + 20000))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_port $MASTER_PORT
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 2
    --expert-model-parallel-size 2
)

TRAINING_ARGS=(
    --micro-batch-size $mbs
    --global-batch-size $gbs 
    --train-iters 50 # Set to a small number for testing; adjust as needed for real training
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --lr 1e-2
    --lr-decay-style cosine 
    --min-lr 2.0e-5
    --lr-warmup-iters 150
    --lr-decay-iters 2200 
    --auto-detect-ckpt-format
    --accumulate-allreduce-grads-in-fp32
    --model-provider llava_vlm
    --bf16
    --use-distributed-optimizer
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 2000 
    --eval-interval 20000 
    # --save $CHECKPOINT_STORE_PATH 
    --eval-iters 10
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    # --wandb-project $WANDB_PROJECT
    # --wandb-exp-name $EXP_NAME
    # --wandb-save-dir $CHECKPOINT_STORE_PATH
)

# Add checkpoint argument only if provided and not "None"
if [[ -n "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH" && "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH" != "None" && "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH" != "" ]]; then
    EVAL_AND_LOGGING_ARGS+=(--language-model-checkpoint "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH")
fi

# Tokenizer args
TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model 'llava-hf/llava-1.5-7b-hf'
)

# Dataset args
DATASET_ARGS=(
    --dataloader-type external
    --dataset-provider llava_vlm
    --data-path $DATASET_PATH
    --packing-buffer-size 24
    --total-seq-length 2048 # length for a single sample, including both text and image tokens
)

GPT_MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096 # for model, instead of training
    # --encoder-seq-length 4096 # set seq-length or encoder-seq-length, not both
    --max-position-embeddings 4096 # instead of 32768
    --num-layers 2 # Set to a small number for testing; adjust as needed for real training
    --hidden-size 4096
    --moe-ffn-hidden-size 14336 # if unassigned, fall back to ffn_hidden_size
    --ffn-hidden-size 14336 # keep the same with moe_ffn-hidden-size
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-masked-softmax-fusion
    --no-position-embedding
    --rotary-base 1000000
)

MOE_ARGS=(
    --num-experts 4
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
    --overlap-param-gather # --overlap-param-gather only supported with distributed optimizer or megatron fsdp
    --overlap-grad-reduce # Disabled to avoid DDP bucket AssertionError caused by unused experts or unused vision branch
)

# Run the training script based on configuration
if [ "$DEBUG_MODE" = true ]; then
  echo "Running in debug mode with $GPUS_PER_NODE GPU(s) per node..."
  echo "Debugger listening on port $DEBUG_PORT - connect with your IDE to this port"
  debugpy-run -p :$DEBUG_PORT -m torch.distributed.run -- ${DISTRIBUTED_ARGS[@]} train.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${DATASET_ARGS[@]} \
    ${MOE_ARGS[@]}
else
  echo "Running in normal mode with $GPUS_PER_NODE GPU(s) per node..."
  if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode enabled"
    echo "torchrun ${DISTRIBUTED_ARGS[@]} train.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${DATASET_ARGS[@]} \
    ${MOE_ARGS[@]}"
  else
    python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} train.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${GPT_MODEL_ARGS[@]} \
    ${DATASET_ARGS[@]} \
    ${MOE_ARGS[@]}
  fi
fi

