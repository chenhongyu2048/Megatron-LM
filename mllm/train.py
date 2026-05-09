# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Training loop for MIMO models with heterogeneous encoder/LLM parallelism."""

import os
import random
import numpy as np
import sys
from contextlib import ExitStack, contextmanager
from functools import partial
from typing import Any, Dict, Iterator, Optional

_MEGATRON_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
if _MEGATRON_ROOT not in sys.path:
    sys.path.insert(0, _MEGATRON_ROOT)

import torch
import torch.distributed as dist

from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import set_args

# ---- MIMO / parallel plumbing ------------------------------------------------
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.models.mimo.optimizer import get_mimo_optimizer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.pipeline_parallel.bridge_communicator import BridgeCommunicator
from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
)
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)

# ---- Project-local imports ---------------------------------------------------
from mllm.data.energon_vlm_task_encoder import llava_vlm_dataloader_provider, build_vlm_dataloader
from mllm.data.mock import train_valid_test_datasets_provider as mock_train_valid_test_datasets_provider
from mllm.model_providers.llava_vlm import model_provider_llava_vlm
from mllm.utils.data_helpers import broadcast_nested_data_batch
from mllm.profiler.layer_profiler import MimoLayerProfiler
from mllm.utils.pgc_helpers import attach_derived_groups

# ============================================================================
# Globals
# ============================================================================
PROFILE_START_ITER = 40
PROFILE_END_ITER = 50
_FORWARD_TRAIN_STEP_COUNT = 0
_FORWARD_EVAL_STEP_COUNT = 0
_profile_logged = False
_profiler: MimoLayerProfiler = None

_active_grids: list = []
_embedding_pg_cache: dict = {}

_MODEL_PROVIDERS = {
    "llava_vlm": model_provider_llava_vlm,
    "video_llava_vlm": partial(model_provider_llava_vlm, is_video_input=True),
}

ENCODER_NAME = "images"

# Set by run() before forward/backward starts
_current_tp_group = None
_current_cp_group = None
_current_dp_group = None

# ============================================================================
# Args
# ============================================================================
def add_mimo_args(parser):
    group = parser.add_argument_group('MIMO', 'MIMO specific arguments')
    group.add_argument('--dataset-provider', type=str, default='mock')
    group.add_argument('--model-provider', type=str, default='mock')

    group.add_argument('--image-size', type=int, default=336)
    group.add_argument('--total-seq-length', type=int, default=2048)
    group.add_argument('--pad-token-id', type=int, default=0)
    group.add_argument('--image-token-id', type=int, default=32000)
    group.add_argument('--image-seq-length', type=int, default=576)
    group.add_argument('--audio-encoder-model', type=str, default=None)
    group.add_argument('--hf-assign-unused-tokens', type=str, nargs='+', default=None)

    group.add_argument('--language-model-checkpoint', type=str, default=None)
    group.add_argument('--packing-buffer-size', type=int, default=None)

    # --- NEW: heterogeneous parallelism for encoder / LLM --------------------
    group.add_argument('--encoder-tp', type=int, default=1)
    group.add_argument('--encoder-pp', type=int, default=1)
    group.add_argument('--encoder-dp', type=int, default=1)
    group.add_argument('--encoder-cp', type=int, default=1)
    group.add_argument('--llm-tp',     type=int, default=1)
    group.add_argument('--llm-pp',     type=int, default=1)
    group.add_argument('--llm-dp',     type=int, default=1)
    group.add_argument('--llm-cp',     type=int, default=1)
    group.add_argument('--llm-etp',    type=int, default=1)
    group.add_argument('--llm-ep',     type=int, default=1)
    group.add_argument('--llm-edp',    type=int, default=1)

    group.add_argument('--train-iters-mimo', type=int, default=1000, help='Training iterations for the MIMO custom loop.')
    group.add_argument('--log-interval-mimo', type=int, default=10)

    return parser


# ============================================================================
# Grid / PG helpers (mirrors the test file)
# ============================================================================
def create_hypercomm_grid_dense(offset, tp, cp, pp, dp):
    """
    Dense communication grid. Includes placeholder size=1 dimensions
    for ep / expt_tp for MCore DDP / MIMO optimizer compatibility.
    """
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1, 1], # [tp, cp, pp, dp, ep, expt_dp]
        dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
        rank_offset=offset,
        backend="nccl",
    )
    
    grid.create_pg(["tp"])
    grid.create_pg(["cp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    grid.create_pg(["dp", "cp"])
    grid.create_pg(["ep"])
    grid.create_pg(["expt_dp"])
    # Required by _get_pg_collection_for_optimizer
    grid.create_pg(["tp", "pp"])
    grid.create_pg(["tp", "ep", "pp"])
    grid.create_pg(["dp", "ep"])
    grid.create_pg(["tp", "cp", "ep", "pp", "dp"])

    _active_grids.append(grid)
    return grid

def create_hypercomm_grid_moe(offset, etp, ep, pp, edp):
    """
    Dimension meanings:
        etp (expt_tp): tensor parallel within experts
        ep            : expert parallel (expert sharding)
        edp (expt_dp): expert data parallel
    """
    grid = HyperCommGrid(
        shape=[etp, ep, pp, edp],
        dim_names=["expt_tp", "ep", "pp", "expt_dp"],
        rank_offset=offset, backend="nccl",
    )
    for name in ["expt_tp", "ep", "pp", "expt_dp"]:
        grid.create_pg([name])

    # Combined groups: used by MoEAlltoAllTokenDispatcher / optimizer
    grid.create_pg(["expt_tp", "ep"])
    grid.create_pg(["expt_tp", "ep", "pp"])
    _active_grids.append(grid)
    return grid

def destroy_all_grids():
    for grid in _active_grids:
        grid.destroy()
    _active_grids.clear()
    _embedding_pg_cache.clear()
    BridgeCommunicator.destroy_broadcast_pgs()

def _create_all_embedding_groups(grids):
    """Collective: must be called by ALL ranks in the same order."""
    for grid in grids:
        pp_group = grid.get_pg("pp")
        if not pp_group:
            continue
        pp_ranks = sorted(dist.get_process_group_ranks(pp_group))
        key = tuple(pp_ranks)
        if key not in _embedding_pg_cache: # avoid multiple grids on the same GPUs having the same PP group and creating duplicate groups
            pos_embd_ranks = [pp_ranks[0]] # position embedding group is always the first PP rank
            embd_ranks = [pp_ranks[0]]     # word embedding group starts with the first PP rank, and includes the last PP rank if it's different
            if pp_ranks[-1] != pp_ranks[0]:
                embd_ranks.append(pp_ranks[-1])
            _embedding_pg_cache[key] = (
                dist.new_group(ranks=pos_embd_ranks),
                dist.new_group(ranks=embd_ranks),
            )

def _add_embedding_groups(pgc, is_language_model=False):
    if not pgc.pp:
        return pgc
    key = tuple(sorted(dist.get_process_group_ranks(pgc.pp)))
    pos_embd_pg, embd_pg = _embedding_pg_cache[key]
    pgc.pos_embd = pos_embd_pg if is_pp_first_stage(pgc.pp) else None
    if is_language_model:
        pgc.embd = (embd_pg if (is_pp_last_stage(pgc.pp) or is_pp_first_stage(pgc.pp))else None)
    else:
        pgc.embd = None
    return pgc

def _is_rank_in_grid(grid):
    r = dist.get_rank()
    return grid.rank_offset <= r < grid.rank_offset + grid.size

def _dump_pgc(name, pgc):
    for k, v in vars(pgc).items():
        if v is None:
            print(f"[Rank-{dist.get_rank()}][{name}] {k}: None")
        else:
            print(f"[Rank-{dist.get_rank()}][{name}] {k}: size={dist.get_world_size(v)}, "
                  f"ranks={dist.get_process_group_ranks(v)}")


# ============================================================================
# Batch / loss / forward step
# ============================================================================
def get_batch(data_iterator: Optional[Iterator[Dict[str, Any]]]):
    """Pull a batch on TP rank-0 and broadcast across TP."""
    if data_iterator is None:
        # This rank does not consume input (middle/last PP stage without data role).
        return {"input_ids": None}

    # We can't rely on the global parallel_state here, so use the model's
    # TP group via a module-level cache set by the training loop.
    tp_group = _current_tp_group
    tp_rank = dist.get_rank(tp_group) if tp_group is not None else 0
    tp_src = (
        dist.get_process_group_ranks(tp_group)[0]
        if tp_group is not None
        else dist.get_rank()
    )

    if tp_rank == 0:
        try:
            data = next(data_iterator)
            has_data = torch.tensor([1], dtype=torch.uint8, device='cuda')
        except StopIteration:
            has_data = torch.tensor([0], dtype=torch.uint8, device='cuda')
            data = None
    else:
        has_data = torch.empty(1, dtype=torch.uint8, device='cuda')
        data = None

    if tp_group is not None and dist.get_world_size(tp_group) > 1:
        dist.broadcast(has_data, tp_src, group=tp_group)

    if has_data.item() == 0:
        return None

    # MiMo forward pass expects 
    # input_ids: torch.Tensor,
    # position_ids: Optional[torch.Tensor] = None,
    # attention_mask: Optional[torch.Tensor] = None,
    # loss_mask: Optional[torch.Tensor] = None,
    # labels: Optional[torch.Tensor] = None,
    # modality_inputs: Optional[Dict[str, Dict[str, Any]]] = None,
    # packing_kwargs: Optional[dict] = None,

    # For the modality inputs, the keys can be arbitrary
    # so we do a broadcast of the schema followed by a broadcast of the actual data
    # check broadcast_nested_data_batch for more details
    batch = broadcast_nested_data_batch(data, tp_group=_current_tp_group)

    # Cast vision float tensors to bf16
    if "modality_inputs" in batch:
        for modality in batch["modality_inputs"].values():
            for _, encoder_inputs in modality.items():
                for k, v in encoder_inputs.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        encoder_inputs[k] = v.to(dtype=torch.bfloat16)
    return batch

def loss_func(loss_mask, output_tensor):
    """Simple loss function for MIMO model training.

    Args:
        loss_mask: mask indicating which tokens contribute to the loss
        output_tensor: model output tensor
    Returns:
        tuple: (loss, num_tokens, metrics_dict)
    """
    args = get_args()

    if output_tensor is None:
        # Non-last-stage rank — return dummy loss; schedule ignores it.
        zero = torch.tensor(0.0, device='cuda', requires_grad=True)
        return zero, torch.tensor(0, device='cuda', dtype=torch.int), {'lm loss': zero.detach()}

    # output_tensor may be a dict keyed by module name; pull the LM output.
    if isinstance(output_tensor, dict):
        output_tensor = output_tensor.get(
            MIMO_LANGUAGE_MODULE_KEY, next(iter(output_tensor.values()))
        )

    losses = output_tensor.float()
    loss_mask = loss_mask.contiguous().view(-1).float()
    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)

    loss = torch.cat([total_loss.view(1), total_tokens.view(1)])
    loss_for_backward = loss[0].clone()

    if args.context_parallel_size > 1 and _current_cp_group is not None:
        dist.all_reduce(loss, group=_current_cp_group)
        loss_for_backward = loss[0].clone()

    reporting_loss = loss.clone().detach()
    if _current_dp_group is not None:
        dist.all_reduce(reporting_loss, group=_current_dp_group)
    local_num_tokens = loss[1].clone().detach().to(torch.int)

    return loss_for_backward, local_num_tokens, {'lm loss': reporting_loss}

def forward_step(data_iterator, model):
    global _profile_logged, _FORWARD_TRAIN_STEP_COUNT, _FORWARD_EVAL_STEP_COUNT
    if model.training:
        _FORWARD_TRAIN_STEP_COUNT += 1
    else:
        _FORWARD_EVAL_STEP_COUNT += 1

    # ---- profiler ----
    if model.training and _profiler is not None:
        if _FORWARD_TRAIN_STEP_COUNT == PROFILE_START_ITER:
            _profiler.reset_timers()
            _profiler.enable()
            if dist.get_rank() == 0:
                print(f"[Profiler] enabled at iter {_FORWARD_TRAIN_STEP_COUNT}")
        if _FORWARD_TRAIN_STEP_COUNT == PROFILE_END_ITER and not _profile_logged:
            _profiler.disable()
            torch.cuda.synchronize()
            _profiler.print_results(num_iters=PROFILE_END_ITER - PROFILE_START_ITER)
            _profiler.remove_hooks()
            _profile_logged = True

    data_batch = get_batch(data_iterator)
    if data_batch is None:
        # Iterator exhausted -- build a placeholder so the schedule can still run this step
        data_batch = {"input_ids": None}

    output_tensor, loss_mask = model(**data_batch)
    return output_tensor, partial(loss_func, loss_mask)

# ============================================================================
# Model provider — uses the new heterogeneous-parallel API
# ============================================================================
def build_mimo_model(encoder_grid, llm_grid, encoder_pgc, language_pgc):
    ra = get_args()
    try:
        builder_fn = _MODEL_PROVIDERS[ra.model_provider]
    except KeyError as e:
        raise ValueError(f"Unsupported model provider '{ra.model_provider}'.") from e

    model = builder_fn(
        pre_process=True,           # overridden internally per-rank via pg_collection
        post_process=True,
        add_encoder=True,
        add_decoder=True,
        image_special_token_id=ra.image_token_id,
        encoder_pg_collection=encoder_pgc,
        language_pg_collection=language_pgc,
        encoder_grid=encoder_grid,
        language_grid=llm_grid,
        encoder_name=ENCODER_NAME,
        wrap_ddp=True,
    )
    model.to(torch.device("cuda")).to(torch.bfloat16)

    # global _profiler
    # if _profiler is None:
    #     _profiler = MimoLayerProfiler(get_timers(), enabled=False)
    # _profiler.register_on_mimo_model(model)
    
    return model


# ============================================================================
# Main training entrypoint (replaces Megatron's pretrain() for MIMO)
# ============================================================================
def run():
    global _current_tp_group, _current_cp_group, _current_dp_group

    # 1) Init megatron (sets up CUDA, args, dist, etc.) -----------------------
    initialize_megatron(extra_args_provider=add_mimo_args, args_defaults={})
    args = get_args()
    assert args.micro_batch_size * args.llm_dp % args.encoder_dp == 0
    # print_rank_0(f"[MIMO] args: {args}")

    # 2) Build two grids side by side (encoder first, then LLM) ---------------
    enc_world = args.encoder_tp * args.encoder_pp * args.encoder_dp * args.encoder_cp
    llm_world = args.llm_tp * args.llm_pp * args.llm_dp * args.llm_cp
    llm_ep_world = args.llm_ep * args.llm_etp * args.llm_edp
    total = enc_world + llm_world
    assert dist.get_world_size() == total, (f"World size {dist.get_world_size()} != encoder({enc_world}) + llm({llm_world})")
    assert llm_ep_world == llm_world, (f"LLM EP world size {llm_ep_world} must equal LLM world size {llm_world}")

    encoder_grid = create_hypercomm_grid_dense(offset=0, tp=args.encoder_tp, cp=args.encoder_cp, pp=args.encoder_pp, dp=args.encoder_dp)
    llm_grid = create_hypercomm_grid_dense(offset=enc_world, tp=args.llm_tp, cp=args.llm_cp, pp=args.llm_pp, dp=args.llm_dp)
    expert_grid = create_hypercomm_grid_moe(offset=enc_world, etp=args.llm_etp, ep=args.llm_ep, pp=args.llm_pp, edp=args.llm_edp)

    # 3) Embedding PGs must be built by ALL ranks in the same order ----------
    _create_all_embedding_groups([encoder_grid, llm_grid]) # don't need to create for expert_grid since experts don't have embeddings
    # encoder process groups collection
    encoder_pgc = ProcessGroupCollection()
    encoder_pgc.tp = encoder_grid.get_pg("tp")
    encoder_pgc.cp = encoder_grid.get_pg("cp")
    encoder_pgc.pp = encoder_grid.get_pg("pp")
    encoder_pgc.dp = encoder_grid.get_pg("dp")
    encoder_pgc.dp_cp = encoder_grid.get_pg(["dp", "cp"])
    encoder_pgc.ep = encoder_grid.get_pg("ep")
    encoder_pgc.expt_dp = encoder_grid.get_pg("expt_dp")
    # LLM process groups collection
    language_pgc = ProcessGroupCollection()
    language_pgc.tp = llm_grid.get_pg("tp")
    language_pgc.cp = llm_grid.get_pg("cp")
    language_pgc.pp = llm_grid.get_pg("pp")
    language_pgc.dp = llm_grid.get_pg("dp")
    language_pgc.dp_cp = llm_grid.get_pg(["dp", "cp"])
    language_pgc.ep = expert_grid.get_pg("ep")
    language_pgc.expt_tp = expert_grid.get_pg("expt_tp")
    language_pgc.expt_dp = expert_grid.get_pg("expt_dp")
    
    encoder_pgc = _add_embedding_groups(encoder_pgc, is_language_model=False)
    language_pgc = _add_embedding_groups(language_pgc, is_language_model=True)
    
    _dump_pgc("encoder", encoder_pgc)
    _dump_pgc("llm",     language_pgc)

    attach_derived_groups(
        pgc_grid_pairs=[
            (encoder_pgc,  encoder_grid),
            (language_pgc, llm_grid),
            (language_pgc, expert_grid),
        ],
        combined_groups=[
            ("tp_cp", ("tp", "cp")),
            ("tp_dp_cp", ("tp", "dp", "cp")),
            ("tp_ep", ("expt_tp", "ep")),
            ("tp_ep_pp", ("expt_tp", "ep", "pp")),
        ]
    )
    if _is_rank_in_grid(llm_grid):
        tp_ep_size = dist.get_world_size(language_pgc.tp_ep)
        expected   = args.llm_etp * args.llm_ep
        assert tp_ep_size == expected, (
            f"language_pgc.tp_ep size={tp_ep_size}, expected {expected} "
            f"(etp={args.llm_etp} * ep={args.llm_ep})"
        )
        print(f"[rank{dist.get_rank()}] OK: tp_ep size={tp_ep_size}, "
            f"ranks={dist.get_process_group_ranks(language_pgc.tp_ep)}")

    # Choose TP/CP/DP groups used by get_batch / loss_func for this rank
    if _is_rank_in_grid(llm_grid):
        _current_tp_group = language_pgc.tp
        _current_cp_group = language_pgc.cp
        _current_dp_group = language_pgc.dp
    else:
        _current_tp_group = encoder_pgc.tp
        _current_cp_group = encoder_pgc.cp
        _current_dp_group = encoder_pgc.dp

    # 4) Build model ----------------------------------------------------------
    seed = args.seed if hasattr(args, 'seed') else 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    mimo_model = build_mimo_model(encoder_grid, llm_grid, encoder_pgc, language_pgc)

    # 5) Schedule hooks on mimo_model.config ---------------------------------
    @contextmanager
    def no_sync_func():
        with ExitStack() as stack:
            if mimo_model.language_model is not None:
                stack.enter_context(mimo_model.language_model.no_sync())
            for sm in mimo_model.modality_submodules.values():
                if sm is not None:
                    stack.enter_context(sm.no_sync())
            yield
    mimo_model.config.no_sync_func = no_sync_func

    def finalize_grads_func(
        model_chunks,
        num_tokens=None,
        pg_collection=None,        # consumed here, not passed down
        force_all_reduce=False,
    ):
        # Note: intentionally ignore outer pg_collection; dispatch per-submodule PGC
        if mimo_model.language_model is not None:
            finalize_model_grads(
                [mimo_model.language_model],
                num_tokens=num_tokens,
                pg_collection=language_pgc,
                force_all_reduce=force_all_reduce,
            )
        for sm in mimo_model.modality_submodules.values():
            if sm is not None:
                finalize_model_grads(
                    [sm],
                    num_tokens=num_tokens,
                    pg_collection=encoder_pgc,
                    force_all_reduce=force_all_reduce,
                )

    mimo_model.config.calculate_per_token_loss = True
    mimo_model.config.finalize_model_grads_func = finalize_grads_func
    llm_mbs = args.micro_batch_size
    num_microbatches = max(1, getattr(args, 'global_batch_size', llm_mbs) // (llm_mbs * args.llm_dp))

    # 6) Optimizer (MIMO-aware) ----------------------------------------------
    opt_config = OptimizerConfig(
        optimizer=getattr(args, 'optimizer', 'adam'),
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip_grad=args.clip_grad,
        bf16=args.bf16,
        fp16=args.fp16,
        use_distributed_optimizer=True,
    )
    optimizer = get_mimo_optimizer(mimo_model, opt_config)

    # 7) Multi-module communicator + pg collection ---------------------------
    print(f"[MIMO] distributed info: pid={os.getpid()}, dist.is_initialized={dist.is_initialized()}, dist.get_rank={dist.get_rank() if dist.is_initialized() else 'N/A'}, dist.get_world_size={dist.get_world_size() if dist.is_initialized() else 'N/A'}.")
    
    module_to_grid_map = {
        ENCODER_NAME: encoder_grid,
        MIMO_LANGUAGE_MODULE_KEY: llm_grid,
    }
    topology = {ENCODER_NAME: [MIMO_LANGUAGE_MODULE_KEY], MIMO_LANGUAGE_MODULE_KEY: []}

    communicator = MultiModulePipelineCommunicator(
        module_to_grid_map, topology, mimo_model.config,
        dim_mapping={'s': 0, 'h': 2, 'b': 1},
        module_output_ndim={ENCODER_NAME: 2},
    )

    module_pgs = {}
    lm_module_name = None
    if _is_rank_in_grid(encoder_grid):
        module_pgs[ENCODER_NAME] = encoder_pgc
    if _is_rank_in_grid(llm_grid):
        module_pgs[MIMO_LANGUAGE_MODULE_KEY] = language_pgc
        lm_module_name = MIMO_LANGUAGE_MODULE_KEY
    pg_collection_mm = MultiModuleProcessGroupCollection(
        module_pgs=module_pgs, language_model_module_name=lm_module_name,
    )
    print_rank_0(f"[MIMO] init communicator done.")

    # 8) Dataset & iterator ---------------------------------------------------
    # NOTE: per-role MBS — encoder and LLM DP may differ.
    encoder_mbs = args.micro_batch_size * args.llm_dp // args.encoder_dp

    enc_needs = _is_rank_in_grid(encoder_grid) and is_pp_first_stage(encoder_grid.get_pg("pp"))
    lm_needs = _is_rank_in_grid(llm_grid) and (is_pp_first_stage(llm_grid.get_pg("pp")) or is_pp_last_stage(llm_grid.get_pg("pp")))

    if args.dataset_provider in ("llava_vlm", "video_llava_vlm"):
        # Build independent dataloaders per module so encoder_dp and llm_dp
        # can differ. Both use the same dataset, seed, and shuffle — contiguous
        # sharding ensures data aligns: encoder DP j covers the same global
        # samples as LLM DP [j*scale .. (j+1)*scale].
        is_video = args.dataset_provider == "video_llava_vlm"
        seed = args.seed if hasattr(args, 'seed') else 1234
        torch.manual_seed(seed)

        if enc_needs:
            enc_dp_rank = dist.get_rank(encoder_pgc.dp)
            enc_cp_size = dist.get_world_size(encoder_pgc.cp) if encoder_pgc.cp else 1
            enc_tp_size = dist.get_world_size(encoder_pgc.tp) if encoder_pgc.tp else 1
            print_rank_0(
                f"[dataloader] encoder: dp_rank={enc_dp_rank}, dp_size={args.encoder_dp}, "
                f"batch_size={encoder_mbs}, cp_size={enc_cp_size}, tp_size={enc_tp_size}"
            )
            data_iterator = build_vlm_dataloader(
                dp_rank=enc_dp_rank,
                dp_world_size=args.encoder_dp,
                dp_group=encoder_pgc.dp,
                max_seq_length=args.total_seq_length,
                batch_size=encoder_mbs,
                is_video_input=is_video,
                cp_size=enc_cp_size,
                tp_size=enc_tp_size,
            )
        elif lm_needs:
            llm_dp_rank = dist.get_rank(language_pgc.dp)
            llm_cp_size = dist.get_world_size(language_pgc.cp) if language_pgc.cp else 1
            llm_tp_size = dist.get_world_size(language_pgc.tp) if language_pgc.tp else 1
            if dist.get_rank() == enc_world:
                print(
                    f"[dataloader] LLM: dp_rank={llm_dp_rank}, dp_size={args.llm_dp}, "
                    f"batch_size={llm_mbs}, cp_size={llm_cp_size}, tp_size={llm_tp_size}"
                )
            data_iterator = build_vlm_dataloader(
                dp_rank=llm_dp_rank,
                dp_world_size=args.llm_dp,
                dp_group=language_pgc.dp,
                max_seq_length=args.total_seq_length,
                batch_size=llm_mbs,
                is_video_input=is_video,
                cp_size=llm_cp_size,
                tp_size=llm_tp_size,
            )
        else:
            data_iterator = None
    else:
        # Legacy shared-iterator path for mock and other providers.
        _DATASET_PROVIDERS = {
            "mock": mock_train_valid_test_datasets_provider,
            "llava_vlm": llava_vlm_dataloader_provider,
            "video_llava_vlm": partial(llava_vlm_dataloader_provider, is_video_input=True),
        }
        
        def train_valid_test_datasets_provider(*a, **kw):
            ra = get_args()
            try:
                ds_provider = _DATASET_PROVIDERS[ra.dataset_provider]
                if ra.dataset_provider != "mock":
                    kw['max_seq_length'] = ra.total_seq_length
            except KeyError as e:
                raise ValueError(
                    f"Unsupported dataset provider '{ra.dataset_provider}'."
                ) from e
            return ds_provider(*a, **kw)

        train_ds, _, _ = train_valid_test_datasets_provider(
            train_val_test_num_samples=[args.train_iters_mimo * llm_mbs, 0, 0]
        )
        data_iterator = iter(train_ds) if not hasattr(train_ds, '__next__') else train_ds
        if not (enc_needs or lm_needs):
            data_iterator = None

    print_rank_0(f"[MIMO] init dataloader done.")

    # 9) Training loop --------------------------------------------------------
    seq_length = args.total_seq_length

    for it in range(args.train_iters_mimo):
        print_rank_0(f"==================== ITER {it} ====================")
        optimizer.zero_grad()

        losses = schedule.forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=[mimo_model],
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=llm_mbs,
            forward_only=False,
            p2p_communicator=communicator,
            pg_collection=pg_collection_mm,
        )

        success, grad_norm, num_zeros = optimizer.step()
        if not success:
            print_rank_0(f"[iter {it}] optimizer step failed")
            continue

        if (it % args.log_interval_mimo == 0
                and _is_rank_in_grid(llm_grid)
                and is_pp_last_stage(llm_grid.get_pg("pp"))):
            if losses:
                report = losses[0].get('lm loss', None)
                if report is not None and dist.get_rank(language_pgc.dp) == 0:
                    val = report[0] / max(report[1].item(), 1)
                    print(f"[iter {it}] loss={val.item():.4f} grad_norm={grad_norm:.3f}")

    # 10) Clean up ------------------------------------------------------------
    destroy_all_grids()
    print_rank_0("[MIMO] training done")


if __name__ == "__main__":
    run()
