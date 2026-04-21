# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Configuration utilities for the MIMO implementation of the LLaVA VLM.
"""


from typing import Optional

import torch

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.vision.vit_layer_specs import (
    get_vit_layer_with_transformer_engine_spec,
    get_vit_layer_with_local_spec
)
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args

###############################################################################
# LLaVA VLM configuration utilities
###############################################################################

def get_vision_encoder_config(
    config: Optional[TransformerConfig] = None
) -> TransformerConfig:
    """Return a TransformerConfig for the CLIP ViT-L/14 vision encoder."""
    
    runtime_args = get_args()
    
    cfg = TransformerConfig(
        num_layers=8, # for testing, set num_layers to 2. The original paper uses 24 layers for ViT-L/14.
        hidden_size=1024,
        num_attention_heads=16,
        ffn_hidden_size=4096
    )
    
    # QuickGELU activation.
    cfg.activation_func = torch.nn.functional.gelu
    cfg.gated_linear_unit = False

    # Normalisation – LayerNorm
    cfg.normalization = "LayerNorm"
    cfg.layernorm_epsilon = 1e-5

    # Positional embeddings – learned absolute position embeddings.
    cfg.position_embedding_type = "learned_absolute"
    
    # Sequence length.
    cfg.seq_length = getattr(runtime_args, "seq_length", 576) # 336 / 14 = 24，24 * 24 = 576
    cfg.max_position_embeddings = getattr(runtime_args, "max_position_embeddings", 576)
    
    # Attention / dropout.
    cfg.attention_dropout = 0.0
    cfg.hidden_dropout = 0.0

    # Allow caller overrides.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg

def get_language_model_config(  
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig.

    Current hyper-parameters follow the published Vicuna-7B weights (same sizes as
    Llama-7B).
    """

    runtime_args = get_args()
    
    hidden_size = getattr(runtime_args, "hidden_size", 4096)
    num_attention_heads = getattr(runtime_args, "num_attention_heads", 32)
    
    cfg = TransformerConfig(
        num_layers=runtime_args.num_layers, 
        hidden_size=hidden_size, 
        num_attention_heads=num_attention_heads
    )

    # Feed-forward / MLP hidden size
    cfg.ffn_hidden_size = getattr(runtime_args, "ffn_hidden_size", 14336)

    # SwiGLU (SiLU-gate) activation.
    if getattr(runtime_args, "swiglu", True):
        cfg.activation_func = torch.nn.functional.silu
        cfg.gated_linear_unit = True

    # Normalisation – RMSNorm
    cfg.normalization = getattr(runtime_args, "normalization", "RMSNorm")
    cfg.rms_norm_eps = getattr(runtime_args, "norm_epsilon", 1e-5)

    # Positional embeddings – RoPE.
    cfg.position_embedding_type = getattr(runtime_args, "position_embedding_type", "rope")
    cfg.rotary_base = getattr(runtime_args, "rotary_base", 10000)
    cfg.rotary_percent = getattr(runtime_args, "rotary_percent", 1.0)

    # Sequence length.
    cfg.seq_length = getattr(runtime_args, "seq_length", 4096)
    cfg.max_position_embeddings = getattr(runtime_args, "max_position_embeddings", 4096)

    # Attention / dropout.
    cfg.attention_dropout = getattr(runtime_args, "attention_dropout", 0.0)
    cfg.hidden_dropout = getattr(runtime_args, "hidden_dropout", 0.0)

    # GQA
    if getattr(runtime_args, "group_query_attention", False):
        cfg.num_query_groups = getattr(runtime_args, "num_query_groups", 8)
    else:
        cfg.num_query_groups = num_attention_heads

    # Bias usage.
    cfg.add_bias_linear = getattr(runtime_args, "add_bias_linear", False)
    if getattr(runtime_args, "disable_bias_linear", False):
        cfg.add_bias_linear = False

    # Weight sharing.
    cfg.untie_embeddings_and_output_weights = getattr(runtime_args, "untie_embeddings_and_output_weights", False)

    # Kernel / TE fusions.
    cfg.bias_activation_fusion = getattr(runtime_args, "bias_activation_fusion", True)
    
    cfg.masked_softmax_fusion = getattr(runtime_args, "masked_softmax_fusion", True)
    if getattr(runtime_args, "no_masked_softmax_fusion", False):
        cfg.masked_softmax_fusion = False
        
    cfg.persist_layer_norm = getattr(runtime_args, "persist_layer_norm", True)
    cfg.bias_dropout_fusion = getattr(runtime_args, "bias_dropout_fusion", True)
    cfg.apply_rope_fusion = getattr(runtime_args, "apply_rope_fusion", True)

    # MoE support
    if hasattr(runtime_args, 'num_experts') and runtime_args.num_experts is not None:
        cfg.num_moe_experts = runtime_args.num_experts
        cfg.moe_router_topk = getattr(runtime_args, 'moe_router_topk', 2)
        cfg.moe_router_load_balancing_type = getattr(runtime_args, 'moe_router_load_balancing_type', 'sinkhorn')
        cfg.moe_grouped_gemm = getattr(runtime_args, 'moe_grouped_gemm', False)
        if getattr(runtime_args, 'moe_ffn_hidden_size', None) is not None:
            cfg.moe_ffn_hidden_size = runtime_args.moe_ffn_hidden_size
        elif cfg.ffn_hidden_size is not None:
            cfg.moe_ffn_hidden_size = cfg.ffn_hidden_size

    # Apply user overrides last.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg

def get_llava_projection_config( 
    hidden_size: int = 4096,
    ffn_hidden_size: int = 4096,
    config: Optional[TransformerConfig] = None,
) -> TransformerConfig:
    """Return a TransformerConfig for the vision projection MLP."""

    cfg = TransformerConfig(num_layers=1, hidden_size=hidden_size, num_attention_heads=1)
    cfg.ffn_hidden_size = ffn_hidden_size
    cfg.bias_activation_fusion = True
    cfg.add_bias_linear = True
    cfg.activation_func = torch.nn.functional.gelu

    # Allow caller overrides.
    if config is not None:
        for field, value in vars(config).items():
            setattr(cfg, field, value)

    return cfg

###############################################################################
# LLaVA VLM layer specs
###############################################################################

def get_vision_encoder_layer_spec() -> ModuleSpec:
    """Layer spec for the CLIP ViT-L/14 vision encoder."""
    return get_vit_layer_with_transformer_engine_spec()
    # return get_vit_layer_with_local_spec()

def get_language_model_layer_spec() -> ModuleSpec:
    """Layer spec for the language model (Transformer-Engine GPT block)."""
    runtime_args = get_args()
    num_experts = getattr(runtime_args, "num_experts", None)
    moe_grouped_gemm = getattr(runtime_args, "moe_grouped_gemm", False)
    
    return get_gpt_layer_with_transformer_engine_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

def get_llava_projection_layer_spec() -> ModuleSpec:
    """Layer spec for the vision-projection MLP."""

    return ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=TEColumnParallelLinear,
            linear_fc2=TERowParallelLinear,
        ),
    )
