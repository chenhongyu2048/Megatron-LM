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
export MODELOPT_DISABLE=1

DRY_RUN=false
GPUS_PER_NODE=4
NUM_NODES=1
DEBUG_MODE=false
DEBUG_PORT=5678

DATASET_PATH=$1
PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH=${2:-""}

if [ "$1" = "-d" ]; then
  DEBUG_MODE=true
  echo "Debug mode enabled"
fi

# --------------------------------------------------------------------------- #
# Heterogeneous parallelism for MIMO
# --------------------------------------------------------------------------- #
# NOTE: encoder_world + llm_world MUST equal GPUS_PER_NODE * NUM_NODES.
# 2-GPU layout: encoder on GPU0, llm on GPU1. All sizes = 1.
ENCODER_TP=2
ENCODER_PP=1
ENCODER_DP=1
ENCODER_CP=1

LLM_TP=1
LLM_PP=1
LLM_DP=2
LLM_CP=1
LLM_EP=2
LLM_ETP=1
LLM_EDP=1

mbs=4
gbs=16

# Sanity-check world size
ENC_WORLD=$(( ENCODER_TP * ENCODER_PP * ENCODER_DP * ENCODER_CP ))
LLM_WORLD=$(( LLM_TP * LLM_PP * LLM_DP * LLM_CP ))
LLM_EP_WORLD=$(( LLM_EP * LLM_ETP * LLM_EDP ))
TOTAL_WORLD=$(( GPUS_PER_NODE * NUM_NODES ))
if [ $(( ENC_WORLD + LLM_WORLD )) -ne $TOTAL_WORLD ]; then
  echo "ERROR: encoder($ENC_WORLD) + llm($LLM_WORLD) != total GPUs ($TOTAL_WORLD)"
  exit 1
fi
if [ $(( LLM_EP_WORLD )) -ne $LLM_WORLD ]; then
  echo "ERROR: llm experts ($LLM_EP_WORLD) must equal llm gpus ($LLM_WORLD)"
  exit 1
fi

WANDB_PROJECT='mimo-llava-train'
EXP_NAME='mimo_llava_vlm_pretrain_mbs_'$mbs'_gbs_'$gbs

ROOT_DIR='/data/home/scyb683/run/Megatron-LM/mllm/ckpt/'
CHECKPOINT_STORE_PATH=$ROOT_DIR'mimo_llava_train_hf_clip_'$EXP_NAME
mkdir -p $CHECKPOINT_STORE_PATH

MASTER_PORT=$((RANDOM % 10000 + 20000))

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_port $MASTER_PORT
)

# --------------------------------------------------------------------------- #
# MIMO heterogeneous-parallel args consumed by train.py / add_mimo_args()
# --------------------------------------------------------------------------- #
MIMO_PARALLEL_ARGS=(
    --encoder-tp $ENCODER_TP
    --encoder-pp $ENCODER_PP
    --encoder-dp $ENCODER_DP
    --encoder-cp $ENCODER_CP
    --llm-tp     $LLM_TP
    --llm-pp     $LLM_PP
    --llm-dp     $LLM_DP
    --llm-cp     $LLM_CP
    --llm-ep     $LLM_EP
    --llm-etp    $LLM_ETP
    --llm-edp    $LLM_EDP
)

# --------------------------------------------------------------------------- #
# Legacy Megatron parallel args. These are still parsed by initialize_megatron,
# but the MIMO custom loop does NOT use parallel_state for grids. Keep them at
# 1 so megatron's global state is consistent with "no global parallelism", and
# let the MIMO_PARALLEL_ARGS above drive the actual encoder/LLM grids.
# --------------------------------------------------------------------------- #
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    # expert-data-parallel-size is not defined, just like data-parallel-size
    # --sequence-parallel   # only enable when --tensor-model-parallel-size > 1
)

TRAINING_ARGS=(
    --micro-batch-size $mbs
    --global-batch-size $gbs
    --train-iters 100                     # required by initialize_megatron
    --train-iters-mimo 100                # used by train.py's custom loop
    --log-interval-mimo 10
    --adam-beta1 0.9
    --adam-beta2 0.95
    --lr 1e-4
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --lr-warmup-iters 150
    --lr-decay-iters 2200
    --auto-detect-ckpt-format
    --accumulate-allreduce-grads-in-fp32
    --model-provider llava_vlm
    --bf16
    --use-distributed-optimizer
)

EVAL_AND_LOGGING_ARGS=(
    # --logging-level 10
    --timing-log-level 2
    --timing-log-option all
    --log-interval 10
    --save-interval 2000
    --eval-interval 20000
    --eval-iters 1
    # --save $CHECKPOINT_STORE_PATH
    # --tensorboard-dir $TENSORBOARD_LOGS_PATH
    # --wandb-project $WANDB_PROJECT
    # --wandb-exp-name $EXP_NAME
    # --wandb-save-dir $CHECKPOINT_STORE_PATH
)

if [[ -n "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH" \
      && "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH" != "None" \
      && "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH" != "" ]]; then
    EVAL_AND_LOGGING_ARGS+=(--language-model-checkpoint "$PRETRAINED_LANGUAGE_MODEL_CHECKPOINT_PATH")
fi

TOKENIZER_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model 'llava-hf/llava-1.5-7b-hf'
)

DATASET_ARGS=(
    --dataloader-type external
    --dataset-provider llava_vlm
    --data-path $DATASET_PATH
    --total-seq-length 2048
)

GPT_MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 4096
    --num-layers 1
    --hidden-size 4096
    --moe-ffn-hidden-size 14336
    --ffn-hidden-size 14336
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
    --overlap-param-gather
    --overlap-grad-reduce
)

# MoE: EP must divide LLM_DP * LLM_TP (experts live on the LLM grid only).
MOE_ARGS=(
    --num-experts 2
    --moe-router-topk 1
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-grouped-gemm
    --moe-token-dispatcher-type alltoall
)

ALL_ARGS=(
    ${TRAINING_ARGS[@]}
    ${MIMO_PARALLEL_ARGS[@]}
    ${MODEL_PARALLEL_ARGS[@]}
    ${EVAL_AND_LOGGING_ARGS[@]}
    ${TOKENIZER_ARGS[@]}
    ${GPT_MODEL_ARGS[@]}
    ${DATASET_ARGS[@]}
    ${MOE_ARGS[@]}
)

if [ "$DEBUG_MODE" = true ]; then
  echo "Running in debug mode with $GPUS_PER_NODE GPU(s) per node..."
  echo "Debugger listening on port $DEBUG_PORT"
  debugpy-run -p :$DEBUG_PORT -m torch.distributed.run -- \
      ${DISTRIBUTED_ARGS[@]} train.py ${ALL_ARGS[@]}
else
  echo "Running in normal mode with $GPUS_PER_NODE GPU(s) per node..."
  echo "  encoder grid: tp=$ENCODER_TP pp=$ENCODER_PP dp=$ENCODER_DP cp=$ENCODER_CP (world=$ENC_WORLD)"
  echo "  llm     grid: tp=$LLM_TP pp=$LLM_PP dp=$LLM_DP cp=$LLM_CP (world=$LLM_WORLD)"
  if [ "$DRY_RUN" = true ]; then
    echo "Dry run:"
    echo "torchrun ${DISTRIBUTED_ARGS[@]} train.py ${ALL_ARGS[@]}"
  else
    python -m torch.distributed.run ${DISTRIBUTED_ARGS[@]} train.py ${ALL_ARGS[@]}
  fi
fi
