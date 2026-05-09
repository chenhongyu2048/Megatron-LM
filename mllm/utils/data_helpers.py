# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Utility helpers for broadcasting nested dictionaries of tensors across tensor-parallel ranks.

"""

from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.distributed as dist

from megatron.core import mpu, tensor_parallel


def flatten(
    nested: Dict[str, Any], prefix: Tuple[str, ...] = ()
) -> List[Tuple[Tuple[str, ...], torch.Tensor]]:
    """Recursively flatten nested dict into [(key_path, tensor), …]."""
    flat = []
    for k, v in nested.items():
        path = prefix + (k,)
        if isinstance(v, dict):
            flat.extend(flatten(v, path))
        elif isinstance(v, torch.Tensor):
            flat.append((path, v))        # v is a tensor
        else:
            raise ValueError(f"Unsupported value type: {type(v)} for key {k}"
                             f"In nested dictionary,leaf nodes must contain tensors")
    return flat


def regroup(flat: List[Tuple[Tuple[str, ...], torch.Tensor]]) -> Dict[str, Any]:
    """Rebuild the nested dict from [(key_path, tensor), …]."""
    root = {}
    for path, tensor in flat:
        cur = root
        for k in path[:-1]:
            cur = cur.setdefault(k, {})
        cur[path[-1]] = tensor
    return root


def broadcast_nested_data_batch(
    nested_dict: Optional[Dict[str, Any]],
    tp_group: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Any]:
    """Recursively broadcast nested dicts of tensors over an explicit TP group.

    Args:
        nested_dict: On TP src rank, the dict-of-tensors to broadcast. Ignored
            (may be None) on non-src ranks.
        tp_group: The tensor-parallel process group to broadcast within. If
            None, falls back to ``mpu.get_tensor_model_parallel_group()`` for
            backward compatibility.

    Returns:
        The reconstructed nested dict on every rank in ``tp_group``.
    """
    # ---------- resolve group / src / local rank ------------------------------
    if tp_group is None:
        # Backward-compatible fallback to the legacy global state.
        from megatron.core import parallel_state as mpu
        tp_group = mpu.get_tensor_model_parallel_group()
        src = mpu.get_tensor_model_parallel_src_rank()
        tp_rank = mpu.get_tensor_model_parallel_rank()
    else:
        # Derive src (global rank of local-0) and local rank from the group
        # itself so we don't touch any global parallel state.
        group_ranks = dist.get_process_group_ranks(tp_group)
        src = group_ranks[0]
        tp_rank = dist.get_rank(tp_group)

    # If the TP group is a single rank, there is nothing to broadcast.
    world = dist.get_world_size(tp_group)
    if world == 1:
        if nested_dict is None:
            return {}
        # Still move tensors to CUDA for downstream consumers.
        flat = flatten(nested_dict)
        return regroup([(p, t.cuda() if isinstance(t, torch.Tensor) else t)
                        for p, t in flat])

    # ---------- rank-0 prepares metadata --------------------------------------
    if tp_rank == 0:
        assert nested_dict is not None, "TP src rank must provide nested_dict"
        flat = flatten(nested_dict)                        # [(path, tensor), ...]
        if flat:
            paths, tensors = zip(*flat)
            paths = list(paths)
            tensors = list(tensors)
        else:
            paths, tensors = [], []
        dtypes = [t.dtype for t in tensors]
    else:
        paths, dtypes, tensors = [], [], []

    # ---------- 1. broadcast schema (paths + dtypes) --------------------------
    obj_list = [[paths, dtypes]]
    dist.broadcast_object_list(obj_list, src=src, group=tp_group)
    paths, dtypes = obj_list[0]

    # ---------- 2. group tensors by dtype and broadcast -----------------------
    dtype_to_keys: Dict[torch.dtype, list] = {}
    for p, dt in zip(paths, dtypes):
        dtype_to_keys.setdefault(dt, []).append(".".join(p))

    if tp_rank == 0:
        data_dict = {".".join(p): t.cuda() for p, t in zip(paths, tensors)}
    else:
        data_dict = {}

    flat_out = []
    for dt, keys in dtype_to_keys.items():
        # NOTE: tensor_parallel.broadcast_data historically reads the TP group
        # from parallel_state. If your Megatron version supports passing an
        # explicit group, prefer that; otherwise see the fallback below.
        try:
            out = tensor_parallel.broadcast_data(keys, data_dict, dt, group=tp_group)
        except TypeError:
            # Older signature without `group=` kwarg: fall back to a manual
            # implementation that uses the explicit tp_group.
            out = _manual_broadcast_data(keys, data_dict, dt, tp_group, src, tp_rank)

        flat_out.extend([(tuple(k.split(".")), out[k]) for k in keys])

    # ---------- 3. rebuild nested structure -----------------------------------
    return regroup(flat_out)


# -----------------------------------------------------------------------------
# Manual fallback: mirrors megatron.core.tensor_parallel.data.broadcast_data but
# takes an explicit process group instead of reading from parallel_state.
# -----------------------------------------------------------------------------
def _manual_broadcast_data(keys, data, datatype, tp_group, src, tp_rank):
    """Broadcast tensors in ``data[keys]`` across ``tp_group`` with given dtype."""
    # 1) Build per-key shapes on src, broadcast them so every rank can allocate.
    if tp_rank == 0:
        sizes = {k: list(data[k].size()) for k in keys}
    else:
        sizes = {}
    obj = [sizes]
    dist.broadcast_object_list(obj, src=src, group=tp_group)
    sizes = obj[0]

    # 2) Flatten src tensors into one contiguous buffer of ``datatype``.
    numels = [int(torch.tensor(sizes[k]).prod().item()) for k in keys]
    total = sum(numels)
    if tp_rank == 0:
        flat_tensor = torch.cat([
            data[k].contiguous().view(-1).to(datatype) for k in keys
        ]).cuda()
    else:
        flat_tensor = torch.empty(total, dtype=datatype, device='cuda')

    # 3) Broadcast the flat buffer.
    dist.broadcast(flat_tensor, src=src, group=tp_group)

    # 4) Split back into per-key tensors with the original shapes.
    out = {}
    offset = 0
    for k, n in zip(keys, numels):
        out[k] = flat_tensor[offset:offset + n].view(*sizes[k])
        offset += n
    return out
