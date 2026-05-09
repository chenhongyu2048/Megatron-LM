"""Populate a ProcessGroupCollection with all combined groups that modern
Megatron-LM expects, driven by a HyperCommGrid."""
from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

import torch.distributed as dist


# --------------------------------------------------------------------------- #
# Declaration of which combined groups to populate.
#
# Left  : attribute name on ProcessGroupCollection (what MCore code looks up)
# Right : dim-name combination to feed into HyperCommGrid.create_pg()
#
# Dim names here MUST match the names you used when constructing the grid
# (e.g. HyperCommGrid(shape=[tp,cp,ep,dp,pp], dim_names=["tp","cp","ep","dp","pp"])).
# Any dim that does not exist on a given grid is silently skipped.
# --------------------------------------------------------------------------- #
DEFAULT_COMBINED_GROUPS: List[Tuple[str, Tuple[str, ...]]] = [
    # single-axis groups (usually already set, kept here so new grids are self-contained)
    ("tp",          ("tp",)),
    ("pp",          ("pp",)),
    ("dp",          ("dp",)),
    ("cp",          ("cp",)),
    ("ep",          ("ep",)),
    ("expt_tp",     ("expt_tp",)),
    ("expt_dp",     ("expt_dp",)),
    # 2-axis combinations
    ("tp_cp",       ("tp", "cp")),
    ("tp_dp",       ("tp", "dp")),
    ("tp_pp",       ("tp", "pp")),
    ("dp_cp",       ("dp", "cp")),
    # 3-axis combinations
    ("tp_dp_cp",    ("tp", "dp", "cp")),
]


def _filter_existing_dims(grid, dims: Sequence[str]) -> List[str]:
    """Keep only the dims that actually live on this grid."""
    return [d for d in dims if d in grid.dim_names]


def _ensure_pg(grid, dims: Sequence[str], group_desc: str | None = None):
    """Create the PG on the grid if not already created, then return it.

    Handles the corner case where dims collapse to a subset already registered
    under a different alias (e.g. when ep_size==1, 'tp_ep' reduces to 'tp').
    """
    if not dims:
        return None
    try:
        return grid.get_pg(list(dims))
    except KeyError:
        pass  # not created yet -- fall through

    kwargs = {"group_desc": group_desc} if group_desc else {}
    try:
        return grid.create_pg(list(dims), **kwargs)
    except KeyError:
        # Already created under an equivalent key: just fetch it.
        return grid.get_pg(list(dims))


def attach_derived_groups(
    pgc_grid_pairs: Iterable[Tuple[object, "HyperCommGrid"]],  # noqa: F821
    combined_groups: Sequence[Tuple[str, Tuple[str, ...]]] = DEFAULT_COMBINED_GROUPS,
    overwrite: bool = False,
) -> None:
    """
    Populate every ProcessGroupCollection in ``pgc_grid_pairs`` with the
    combined process groups listed in ``combined_groups``.

    Because HyperCommGrid.create_pg() internally calls
    ``dist.new_subgroups_by_enumeration`` -- which is itself a collective --
    every rank must traverse this loop in the same order with the same
    arguments. That invariant is trivially satisfied here: the iteration
    order is deterministic and the grid objects describe a globally-agreed
    world layout.

    Args:
        pgc_grid_pairs: iterable of (pgc, grid). Example:
            [(encoder_pgc, encoder_grid), (language_pgc, llm_grid)]
        combined_groups: list of (attr_name, dim_tuple) entries.
        overwrite: if True, reassign even when the attribute is already set.
    """
    rank0 = dist.is_initialized() and dist.get_rank() == 0

    for pgc, grid in pgc_grid_pairs:
        for attr, dims in combined_groups:
            # Skip attributes the PGC already carries, unless overwrite requested.
            if not overwrite and getattr(pgc, attr, None) is not None:
                continue

            real_dims = _filter_existing_dims(grid, dims)
            # Require ALL requested dims to exist on this grid; otherwise this combined
            # group simply doesn't belong on this grid.
            if len(real_dims) != len(dims):
                if rank0:
                    logging.debug(
                        f"[pgc_helpers] skip '{attr}': partial match {real_dims} "
                        f"of requested {dims} on grid.dim_names={grid.dim_names}"
                    )
                continue

            pg = _ensure_pg(grid, real_dims, group_desc=f"MIMO_{attr.upper()}")

            # Only set if this rank belongs to the grid (and thus to the pg).
            if grid.is_current_rank_in_grid():
                setattr(pgc, attr, pg)

            if rank0:
                logging.info(
                    f"[pgc_helpers] {type(pgc).__name__}.{attr} <- grid.create_pg({real_dims})"
                )
