# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope of Work

### Main Working Directory

All your **modification and creation** operations must be performed only within the `Megatron-LM/mllm/` directory.

### Readability

You may **read** any file in the parent directory `Megatron-LM/` as a reference (e.g., megatron/core/, megatron/training/, etc.)
for understanding the API, reusing existing modules, and maintaining consistent coding style.

### Prohibited Practices

- Do not modify any files outside of `mllm/`
- Do not create new files outside of `mllm/`
- Confirm with me before modifying external files.

### Module Background

mllm is a multimodal large model module built on Megatron-LM, relying on the parallel primitives and transformer implementation of megatron.core.

## Overview

Megatron-LM is NVIDIA's GPU-optimized library for training large transformer models at scale. It contains two components:

- **Megatron Core** (`megatron/core/`) — composable GPU-optimized building blocks: transformer layers, parallelism strategies (TP, PP, DP, CP, EP), mixed precision (FP16/BF16/FP8/FP4), optimizers, and datasets.
- **Megatron-LM** (`megatron/training/`) — the reference training framework with scripts (`pretrain_gpt.py`, `pretrain_bert.py`, etc.), argument parsing, checkpointing, and the training loop.

## Architecture

### Parallelism model

Megatron Core implements several parallelism strategies that can be combined (3D parallelism):

| Strategy | Directory | Key concept |
|---|---|---|
| Tensor Parallel (TP) | `tensor_parallel/` | Splits weight matrices across GPUs within a layer |
| Pipeline Parallel (PP) | `pipeline_parallel/` | Splits layers across GPUs with micro-batch scheduling (1F1B) |
| Data Parallel (DP) | `distributed/` | Replicates model across GPUs, averages gradients |
| Context Parallel (CP) | `transformer/` | Splits sequence length across GPUs |
| Expert Parallel (EP) | `transformer/moe/` | Distributes MoE experts across GPUs |

The **central config dataclass** is `TransformerConfig` in `megatron/core/transformer/transformer_config.py`. It holds all model architecture and parallelism settings. It inherits from `ModelParallelConfig`.

### Model architecture

Models use a `ModuleSpec` pattern (`megatron/core/transformer/spec_utils.py`) to declare which layer implementations to use. A model's `ModuleSpec` defines the concrete classes for attention, MLP, normalization, etc. This allows swapping implementations (e.g., TransformerEngine-optimized vs. local PyTorch) without changing model code.

**Key model classes:**

- `GPTModel` (`megatron/core/models/gpt/`) — decoder-only transformer, extends `LanguageModule`. Used as the language backbone in multimodal models.
- `MimoModel` (`megatron/core/models/mimo/`) — multi-input multi-output model for multimodal training. Composes a language model with modality submodules (vision, audio). Supports **heterogeneous parallelism** — encoder and LLM can use different TP/PP/DP/CP sizes.
- `CLIPViTModel` (`megatron/core/models/vision/`) — Vision Transformer (ViT) encoder used in multimodal models.
- `MambaModel` (`megatron/core/models/mamba/`) — state-space model.

Model layer specifications (e.g., `get_gpt_layer_with_transformer_engine_spec()`) live in `megatron/core/models/<model>/<model>_layer_specs.py`.

### Training entry points

Standard single-modality training scripts at repo root:
- `pretrain_gpt.py` — GPT pretraining/SFT (main entry point for most LLM training)
- `pretrain_bert.py` — BERT pretraining
- `pretrain_t5.py` — T5 encoder-decoder
- `pretrain_mamba.py` — Mamba SSM training
- `train_rl.py` — RL-based training (RLHF)

Each script typically provides: `forward_step`, `loss_func`, `train_valid_test_datasets_provider`, and `model_provider`. These are passed to `megatron.training.training.pretrain()` which runs the full training loop.

For legacy model support, `gpt_builders.py` and `model_provider.py` provide builder functions that route between MCore and legacy models based on `args.use_legacy_models`.

### Training loop architecture

`megatron/training/training.py::pretrain()` handles:
1. Argument parsing via `megatron/training/arguments.py`
2. Distributed initialization via `megatron/training/initialize.py`
3. Model, optimizer, and data iterator setup
4. The main train/validate/test loop with checkpoint saving

The forward/backward schedule lives in `megatron/core/pipeline_parallel/schedules.py`. Key functions:
- `forward_backward_pipelining_without_interleaving` — 1F1B schedule for non-interleaved PP
- `forward_backward_pipelining_with_interleaving` — 1F1B for interleaved PP
- `forward_backward_no_pipelining` — no PP, just forward/backward

### Checkpointing

Distributed checkpointing uses `megatron/core/dist_checkpointing/` with `torch.distributed.checkpoint`. The `strategy.py` file handles sharding and consolidation across parallelism dimensions.

### mllm/ — Multimodal MIMO training (active development)

The `mllm/` directory implements Vision-Language Model training using the MIMO framework with heterogeneous parallelism. This is local development work on branch `mllm-specific`.

- `mllm/train.py` — Custom training loop that replaces `pretrain()` for MIMO. Builds separate HyperCommGrid instances for encoder and LLM, uses `MultiModulePipelineCommunicator` for P2P between them.
- `mllm/model_providers/llava_vlm.py` — Builds a LLaVA-style model: CLIP ViT encoder + multimodal projector + GPT LLM, wrapped in `MimoModel`.
- `mllm/configs/llava_vlm.py` — TransformerConfig and ModuleSpec definitions for all model components.
- `mllm/data/energon_vlm_task_encoder.py` — Dataset provider using NVIDIA Energon format.
- `mllm/utils/` — Helpers for data broadcasting, checkpoint loading, logging, and process group setup.
- `mllm/run_vlm_train.sh` — Shell launcher. Uses `torchrun`, sets heterogeneous parallel args (encoder TP/PP/DP/CP separate from LLM TP/PP/DP/CP), and runs `mllm/train.py`.

The heterogeneous parallelism model: encoder and LLM occupy separate contiguous rank ranges. `HyperCommGrid` divides each range into independent TP/CP/PP/DP grids. `MultiModulePipelineCommunicator` handles cross-module P2P (encoder output → LLM input).

### Extensions

`megatron/core/extensions/` contains TransformerEngine (TE) integration — optimized fused kernels via `transformer_engine` package. TE provides `TEColumnParallelLinear`, `TERowParallelLinear`, `TELayerNorm`, fused attention, etc.

### Key patterns

- Process groups are managed via `ProcessGroupCollection` (`megatron/core/process_groups_config.py`), not global `parallel_state`.
- DDP wrapping is in `megatron/core/distributed/` with its own `DistributedDataParallel` and `DistributedDataParallelConfig`.
- Optimizers support distributed mode via `megatron/core/optimizer/distrib_optimizer.py`.
- The dataset system uses `BlendedMegatronDatasetBuilder` to blend multiple datasets with configurable weights.
