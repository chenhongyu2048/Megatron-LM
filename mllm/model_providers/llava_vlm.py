# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Model provider for a LLaVA-style Vision-Language Model.

This provider assembles a MIMO model that consists of:
• Language model (Dense or MoE) built with Transformer-Engine GPT blocks.
• CLIP ViT-L/14 visual encoder (336 px) that produces image patch embeddings.
• A 2-layer MLP projector that maps vision embeddings into language model hidden size.
"""


import torch
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
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.transformer.spec_utils import ModuleSpec


def model_provider_llava_vlm(
    pre_process: bool = True,
    post_process: bool = True,
    add_encoder=True,
    add_decoder=True,
    image_special_token_id: int = 32000,
    is_video_input: bool = False,
    pg_collection=None,
):
    """
    Build a LLaVA-style Vision-Language MIMO model composed of:
    • A Dense/MoE language model.
    • CLIP ViT-L/14 vision encoder.
    • 2-layer MLP vision→language projector.
    """
    # NOTE: Pipeline parallelism for the encoder/decoder is not yet supported in this
    # MIMO path, therefore *add_encoder* and *add_decoder* are currently ignored.
    
    # Vision
    vision_config = get_vision_encoder_config()

    # Language
    language_config = get_language_model_config()

    # Vision→language projection MLP – hidden size follows Language model (4096)
    projection_config = get_llava_projection_config(hidden_size=language_config.hidden_size, ffn_hidden_size=language_config.hidden_size)

    # Sync precision and parallelism flags from global args (if we're running under Megatron training loop)
    try:
        from megatron.training import get_args  # late import to avoid circular deps

        _args = get_args()
        if getattr(_args, "bf16", False):
            vision_config.bf16 = True
            language_config.bf16 = True
            projection_config.bf16 = True
        if getattr(_args, "fp16", False):
            vision_config.fp16 = True
            language_config.fp16 = True
            projection_config.fp16 = True
        
        # Sync parallelism flags
        # we set cp/sp here for both vision and language configs
        if hasattr(_args, 'context_parallel_size'):
            language_config.context_parallel_size = _args.context_parallel_size
            vision_config.context_parallel_size = 1 # CP not currently supported for vision encoder
        if hasattr(_args, 'cp_comm_type') and _args.cp_comm_type is not None:
            # print_rank_0(f"Setting cp_comm_type from args: {_args.cp_comm_type}") # Setting cp_comm_type from args: ['p2p']
            # so transfer it to a str, because we don't want to have a list in the config which will cause issues when we try to use it in the attention module
            if isinstance(_args.cp_comm_type, list) and len(_args.cp_comm_type) == 1:
                language_config.cp_comm_type = _args.cp_comm_type[0]  # 'p2p' str
                vision_config.cp_comm_type = None # CP not currently supported for vision encoder, so set to None
            else:
                language_config.cp_comm_type = _args.cp_comm_type
                vision_config.cp_comm_type = None # CP not currently supported for vision encoder, so set to None
        if hasattr(_args, 'sequence_parallel'):
            language_config.sequence_parallel = _args.sequence_parallel
            vision_config.sequence_parallel = False # sequence parallel not currently supported for vision encoder
        # set tp/ep here for vision and language distinctly since we may want to use different parallelism for the vision and language parts
        if hasattr(_args, 'tensor_model_parallel_size'):
            language_config.tensor_model_parallel_size = _args.tensor_model_parallel_size
            vision_config.tensor_model_parallel_size = _args.tensor_model_parallel_size
        if hasattr(_args, 'expert_model_parallel_size'):
            language_config.expert_model_parallel_size = _args.expert_model_parallel_size
        if hasattr(_args, 'expert_tensor_parallel_size'):
            language_config.expert_tensor_parallel_size = _args.expert_tensor_parallel_size
        
        # Determine kv_format based on sequence packing
        current_kv_format = "sbhd"
        if getattr(_args, "pack_sequence", False):
            current_kv_format = "thd"

    except (ModuleNotFoundError, AssertionError):
        pass # Args not available (e.g. not in Megatron training context)

    # Megatron's clip vit encoder
    vision_encoder = ModuleSpec(
        module=CLIPViTModel,
        params={
            "transformer_config": vision_config,
            "transformer_layer_spec": get_vision_encoder_layer_spec(),
            "add_class_token": False, # LLaVA does not use class token, and we want to keep all patch tokens
            "class_token_len": 0, # LLaVA does not use class token
            "patch_dim": 14,          # patch_size
            "img_h": 336,             # image_size
            "img_w": 336,             # image_size
            "model_subtype": "clip"
        }
    )

    # Create projection config for vision to language
    vision_projection = ModuleSpec(
        module=MultimodalProjector,
        params={
            "config": projection_config,
            "submodules": get_llava_projection_layer_spec().submodules,
            "projector_type": "mlp",
            "input_size": 1024, # TODO: should this be the same as vision encoder hidden size instead of hardcoded? For CLIP ViT-L/14, it's 1024.
        },
    )

    # Create modality config for vision
    vision_submodule_spec = ModuleSpec(
        module=VisionModalitySubmodules,
        params={},
        submodules={
            "encoders": {"clip_encoder": vision_encoder},
            "input_projections": [vision_projection],
        },
    )

    # Create language model config
    language_model_spec = ModuleSpec(
        module=GPTModel,
        params={
            "config": language_config,
            "transformer_layer_spec": get_language_model_layer_spec(),
            "vocab_size": 32256,
            "max_sequence_length": 4096,
            "pre_process": pre_process,
            "post_process": post_process,
            "position_embedding_type": "rope",
        },
    )

    # Create MIMO model config
    mimo_model_config = MimoModelConfig(
        language_model_spec=language_model_spec,
        modality_submodules_spec={"images": vision_submodule_spec},
        special_token_ids={"images": image_special_token_id}
    )
    # # print configs for debugging
    # print_rank_0(f"Vision encoder config: {vision_config}")
    # print_rank_0(f"Language model config: {language_config}")
    # print_rank_0(f"Projection config: {projection_config}")

    # Create MIMO model
    cp_group = pg_collection.cp if pg_collection is not None else None
    tp_group = pg_collection.tp if pg_collection is not None else None
    # mimo_model = MimoModel(mimo_model_config, cp_group=cp_group, tp_group=tp_group)
    mimo_model = MimoModel(mimo_model_config)
    print("*"*100)
    print_mimo_structure(mimo_model)
    print("*"*100)

    # load the checkpoint
    try:
        from megatron.training import get_args  # late import to avoid circular deps

        _args = get_args()
        if  _args.language_model_checkpoint is not None:
            load_submodule_ckpt(mimo_model.language_model, _args.language_model_checkpoint) # type: ignore
            print(f"Successfully loaded LLaVA pretrained checkpoint from {_args.language_model_checkpoint}")
    except (ModuleNotFoundError, AssertionError):
        pass

    # TODO: ykarnati make these configurable and have an API to freeze/unfreeze   
    # freeze vision encoder and LLM parameters
    # modules_to_freeze = [mimo_model.modality_submodules.images.encoders.clip_encoder, mimo_model.language_model]
    # for module in modules_to_freeze:
    #     for param in module.parameters():
    #         param.requires_grad = False

    return mimo_model
