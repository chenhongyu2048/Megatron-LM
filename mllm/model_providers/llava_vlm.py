# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Model provider for a LLaVA-style Vision-Language MIMO model (updated API)."""

import torch
import torch.distributed as dist

from megatron.training.utils import print_rank_0
from mllm.configs.llava_vlm import (
    get_vision_encoder_config,
    get_llava_projection_config,
    get_llava_projection_layer_spec,
    get_vision_encoder_layer_spec,
    get_language_model_layer_spec,
    get_language_model_config,
)

from utils.logging import print_mimo_structure
from utils.model_helpers import load_submodule_ckpt

from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.vision.clip_vit_model import CLIPViTModel
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
)

def _pp_rank_and_size(pp_group = None):
    if pp_group is None:
        return 0, 1
    return dist.get_rank(pp_group), dist.get_world_size(pp_group)


def model_provider_llava_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder: bool = True,
    add_decoder: bool = True,
    image_special_token_id: int = 32000,
    is_video_input: bool = False,
    # --- args for submodule parallelism ---
    encoder_pg_collection = None,     # encoder's ProcessGroupCollection
    language_pg_collection = None,    # LLM's ProcessGroupCollection
    encoder_grid = None,              # encoder's HyperCommGrid
    language_grid = None,             # LLM's HyperCommGrid
    encoder_name: str = "vision",
    wrap_ddp: bool = True,
    ddp_config: DistributedDataParallelConfig = None,
):
    """
    Build a LLaVA-style Vision-Language MIMO model with the latest MIMO API
    supporting:
      • Independent encoder / LLM parallel grids (TP / PP / DP can differ)
      • Pipeline parallelism for both encoder and LLM
      • Precise injection of parallel communication groups via pg_collection
    """

    # ---------------------------------------------------------------- configs
    vision_config = get_vision_encoder_config()
    language_config = get_language_model_config()
    projection_config = get_llava_projection_config(
        hidden_size=language_config.hidden_size,
        ffn_hidden_size=language_config.hidden_size,
    )
    
    try:
        from megatron.training import get_args  # late import
        _args = get_args()
        # Sync precision / parallelism flags from megatron args (if available)
        if getattr(_args, "bf16", False):
            vision_config.bf16 = True
            language_config.bf16 = True
            projection_config.bf16 = True
            vision_config.pipeline_dtype = torch.bfloat16
            language_config.pipeline_dtype = torch.bfloat16
            projection_config.pipeline_dtype = torch.bfloat16
        elif getattr(_args, "fp16", False):
            vision_config.fp16 = True
            language_config.fp16 = True
            projection_config.fp16 = True
            vision_config.pipeline_dtype = torch.float16
            language_config.pipeline_dtype = torch.float16
            projection_config.pipeline_dtype = torch.float16

        if hasattr(_args, "context_parallel_size"):
            language_config.context_parallel_size = _args.context_parallel_size
            vision_config.context_parallel_size = 1  # CP not supported for vision
        if hasattr(_args, "cp_comm_type") and _args.cp_comm_type is not None:
            if isinstance(_args.cp_comm_type, list) and len(_args.cp_comm_type) == 1:
                language_config.cp_comm_type = _args.cp_comm_type[0]
            else:
                language_config.cp_comm_type = _args.cp_comm_type
            vision_config.cp_comm_type = None
        if hasattr(_args, "sequence_parallel"):
            language_config.sequence_parallel = _args.sequence_parallel
            vision_config.sequence_parallel = False

        # default tp/ep keep same with args
        if hasattr(_args, "tensor_model_parallel_size"):
            language_config.tensor_model_parallel_size = _args.tensor_model_parallel_size
            vision_config.tensor_model_parallel_size = _args.tensor_model_parallel_size
        if hasattr(_args, "expert_model_parallel_size"):
            language_config.expert_model_parallel_size = _args.expert_model_parallel_size
        if hasattr(_args, "expert_tensor_parallel_size"):
            language_config.expert_tensor_parallel_size = _args.expert_tensor_parallel_size

    except (ModuleNotFoundError, AssertionError):
        _args = None

    # Override parallelsim, keep same with runtime pg_collection
    if encoder_pg_collection is not None:
        enc_tp = dist.get_world_size(encoder_pg_collection.tp) if encoder_pg_collection.tp else 1
        enc_pp = dist.get_world_size(encoder_pg_collection.pp) if encoder_pg_collection.pp else 1
        vision_config.tensor_model_parallel_size = enc_tp
        vision_config.pipeline_model_parallel_size = enc_pp
        projection_config.tensor_model_parallel_size = enc_tp

    if language_pg_collection is not None:
        lm_tp = dist.get_world_size(language_pg_collection.tp) if language_pg_collection.tp else 1
        lm_pp = dist.get_world_size(language_pg_collection.pp) if language_pg_collection.pp else 1
        language_config.tensor_model_parallel_size = lm_tp
        language_config.pipeline_model_parallel_size = lm_pp

        if language_pg_collection.cp is not None:
            lm_cp = dist.get_world_size(language_pg_collection.cp)
            global_cp = getattr(_args, 'context_parallel_size', 1)
            language_config.context_parallel_size = lm_cp
            print(
                f"[llava_vlm] language_config.context_parallel_size overridden: "
                f"global={global_cp} -> pgc.cp={lm_cp}"
            )
        if language_pg_collection.tp is not None and dist.get_world_size(language_pg_collection.tp) > 1:
            language_config.sequence_parallel = getattr(_args, 'sequence_parallel', False)
            print(
                f"[llava_vlm] language_config.sequence_parallel overridden: "
                f"tp_size={dist.get_world_size(language_pg_collection.tp)}, "
                f"sp={language_config.sequence_parallel}"
            )

    # pipeline stage flags
    enc_pp_rank, enc_pp_size = _pp_rank_and_size(encoder_pg_collection.pp if encoder_pg_collection is not None else None)
    lm_pp_rank, lm_pp_size = _pp_rank_and_size(language_pg_collection.pp if language_pg_collection is not None else None)

    enc_pre_process = enc_pp_rank == 0
    enc_post_process = enc_pp_rank == enc_pp_size - 1
    lm_pre_process = lm_pp_rank == 0
    lm_post_process = lm_pp_rank == lm_pp_size - 1

    # get module specs
    vision_encoder_spec = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vision_encoder_layer_spec(),
            "add_class_token": False,
            "class_token_len": 0,
            "patch_dim": 14,
            "img_h": 336,
            "img_w": 336,
            "model_subtype": "clip",
            # "pre_process": enc_pre_process, # not enabled
            # "post_process": enc_post_process, # not enabled
            "pg_collection": encoder_pg_collection,
        },
    )

    vision_projection_spec = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": get_llava_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": 1024,  # CLIP ViT-L/14 hidden size
            "tp_group": (
                encoder_pg_collection.tp if encoder_pg_collection is not None else None
            ),
        },
    )

    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={"pg_collection": encoder_pg_collection},
        submodules={
            "encoders": {"clip_encoder": vision_encoder_spec},
            "input_projections": [vision_projection_spec],
        },
    )

    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": get_language_model_layer_spec(),
            "vocab_size": 32256,
            "max_sequence_length": language_config.seq_length,
            "pre_process": lm_pre_process,
            "post_process": lm_post_process,
            "position_embedding_type": "rope",
            "pg_collection": language_pg_collection,
        },
    )

    # module_to_grid_map
    module_to_grid_map = None
    if encoder_grid is not None and language_grid is not None:
        module_to_grid_map = {
            encoder_name: encoder_grid,
            MIMO_LANGUAGE_MODULE_KEY: language_grid,
        }

    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={encoder_name: vision_submodule_spec},
        special_token_ids={encoder_name: image_special_token_id},
        module_to_grid_map=module_to_grid_map,
    )

    mimo_model = MimoModel(mimo_model_config)

    # Cast to training dtype
    if getattr(language_config, "bf16", False):
        mimo_model.to(torch.bfloat16)
    elif getattr(language_config, "fp16", False):
        mimo_model.to(torch.float16)

    print("*" * 100)
    print_mimo_structure(mimo_model)
    print("*" * 100)

    # load checkpoint
    try:
        from megatron.training import get_args
        _args = get_args()
        if getattr(_args, "language_model_checkpoint", None) is not None \
                and mimo_model.language_model is not None:
            load_submodule_ckpt(
                mimo_model.language_model, _args.language_model_checkpoint
            )
            print_rank_0(
                f"Successfully loaded LLM checkpoint from {_args.language_model_checkpoint}"
            )
    except (ModuleNotFoundError, AssertionError):
        pass

    # DDP wrap
    if wrap_ddp:
        if ddp_config is None:
            ddp_config = DistributedDataParallelConfig(
                overlap_grad_reduce=True,
                bucket_size=10_000_000,
                use_distributed_optimizer=True,
            )

        if mimo_model.language_model is not None and language_pg_collection is not None:
            mimo_model.language_model = DistributedDataParallel(
                config=mimo_model.language_model.config,
                ddp_config=ddp_config,
                module=mimo_model.language_model,
                pg_collection=language_pg_collection,
            )

        if encoder_name in mimo_model.modality_submodules and encoder_pg_collection is not None:
            submodule = mimo_model.modality_submodules[encoder_name]
            if submodule is not None:
                mimo_model.modality_submodules[encoder_name] = DistributedDataParallel(
                    config=submodule.encoders["clip_encoder"].config,
                    ddp_config=ddp_config,
                    module=submodule,
                    pg_collection=encoder_pg_collection,
                )

    return mimo_model
