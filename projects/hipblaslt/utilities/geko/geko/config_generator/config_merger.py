# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Merge fork parameters across clustered sizes.

After clustering assigns size indices to clusters, this module merges the
per-size fork-parameter dicts within each cluster into combined dicts so
that clustered sizes share a single tuning config file.

When the merged kernel count would exceed ``max_kernels``, the cluster is
split into sub-clusters so each stays within the limit.
"""
import copy
import logging
from typing import Any, Dict, List, Sequence, Tuple

from geko.config_generator.utils import count_kernels
from geko.config_generator.shared_utils import ConfigEntry, ForkParameter, GroupDimension

logger = logging.getLogger("GEKO")


# =====================================================================
# Top-level entry point (`do_merge`)
# =====================================================================

def do_merge(
    clusters: Dict[Any, Sequence[int]],
    entries: Sequence[ConfigEntry],
    max_kernels: int,
) -> List[ConfigEntry]:
    """Merge fork parameters across all clusters.

    For each cluster, entries are merged (with sub-clustering if the kernel
    cap is exceeded).  The results are flattened into a single list.

    Args:
        clusters: Mapping of cluster id to entry indices.
        entries: Per-size ConfigEntry objects.
        max_kernels: Maximum kernels per config.

    Returns:
        List of merged ConfigEntry objects.
    """
    result: List[ConfigEntry] = []

    for _, cluster_indices in clusters.items():
        result.extend(merge_sizes_in_cluster(cluster_indices, entries, max_kernels))

    return result


# =====================================================================
# Kernel counting (beyond count_kernels in utils)
# =====================================================================

def _first_group_mi_count(groups_fp: ForkParameter) -> int:
    """Return ``len(Groups.values[0])``, or ``0`` if that dimension is missing/empty.

    When non-empty, asserts ``MatrixInstruction`` is in ``values[0][0]`` (MI
    dimension is assumed aligned with the rest of the pipeline).
    """
    vals = groups_fp.values
    if not vals or not vals[0]:
        return 0
    dim0 = vals[0]
    assert "MatrixInstruction" in dim0[0], (
        "Groups.values[0] must be MI groups: first entry must include "
        "'MatrixInstruction'."
    )
    return len(dim0)


def _count_kernels_without_mi(params: Dict[str, ForkParameter]) -> int:
    """Kernel count from everything except MI group variants.

    MI groups live at ``Groups.values[0]``. Uses :func:`count_kernels` and
    divides out the number of MI entries when that dimension is non-empty.
    """
    total = count_kernels(params)
    groups_fp = params.get("Groups")
    if groups_fp is None:
        return total
    num_mis = _first_group_mi_count(groups_fp)
    if num_mis == 0:
        return total
    return total // num_mis


# =====================================================================
# Merging primitives
# =====================================================================

def _group_entry_key(entry: Dict[str, ForkParameter]) -> tuple:
    """Build a hashable key for a group entry dict.

    Each parameter's values are wrapped in a per-key tuple to avoid
    collisions when values from different keys happen to concatenate
    identically.
    """
    return tuple((k, tuple(entry[k].values)) for k in sorted(entry))


def _merge_simple_param(base_fp: ForkParameter, other_fp: ForkParameter) -> None:
    """Union *other_fp*'s values into *base_fp* in-place, preserving order."""
    for v in other_fp.values:
        if v not in base_fp.values:
            base_fp.values.append(v)


def _merge_groups(base_fp: ForkParameter, other_fp: ForkParameter) -> None:
    """Merge group dimensions, deduplicating entries. In-place on *base_fp*.

    For each group dimension index, entries from *other_fp* that are not
    already present in *base_fp* are appended.
    """
    base_dims: List[GroupDimension] = base_fp.values
    other_dims: List[GroupDimension] = other_fp.values

    for dim_idx in range(len(other_dims)):
        existing = {_group_entry_key(e) for e in base_dims[dim_idx]}
        for entry in other_dims[dim_idx]:
            key = _group_entry_key(entry)
            if key not in existing:
                base_dims[dim_idx].append(entry)
                existing.add(key)


def _merge_two_param_dicts(
    base: Dict[str, ForkParameter],
    other: Dict[str, ForkParameter],
) -> Dict[str, ForkParameter]:
    """Merge *other* into a deep copy of *base*. Returns the merged copy.

    - New keys in *other* are deep-copied into the result.
    - For ``Groups``, entries are deduplicated across group dimensions.
    - For simple params, values are unioned (preserving order).
    """
    merged = copy.deepcopy(base)
    for k, fp in other.items():
        if k not in merged:
            merged[k] = copy.deepcopy(fp)
        elif k == "Groups":
            _merge_groups(merged[k], fp)
        else:
            _merge_simple_param(merged[k], fp)
    return merged


# =====================================================================
# MI trimming (cap enforcement)
# =====================================================================

def _trim_mi_to_fit(
    params: Dict[str, ForkParameter],
    max_kernels: int,
) -> int:
    """Trim the MI group dimension so the config fits within *max_kernels*.

    MI groups are ``Groups.values[0]``. The first-entry check runs inside
    :func:`_count_kernels_without_mi` (via :func:`_first_group_mi_count`);
    this function does not call that helper itself.

    If the non-MI parameters alone exceed *max_kernels*, a warning is
    logged and the params are left unchanged.
    """
    kernels_without_mi = _count_kernels_without_mi(params)

    if kernels_without_mi > max_kernels:
        logger.warning(
            "Non-MI parameter combinations (%d) already exceed max_kernels (%d). "
            "Cannot trim MIs to fit. Config will exceed the kernel cap.",
            kernels_without_mi, max_kernels,
        )
        return count_kernels(params)

    groups_fp = params["Groups"]
    mi_groups = groups_fp.values[0] if groups_fp.values else []
    num_mis = len(mi_groups)
    if num_mis == 0:
        return count_kernels(params)
    new_num_mis = max(1, max_kernels // kernels_without_mi)
    if new_num_mis < num_mis:
        logger.info(
            "Trimming MI groups from %d to %d to stay within %d kernel cap.",
            num_mis, new_num_mis, max_kernels,
        )
        groups_fp.values[0] = mi_groups[:new_num_mis]

    return count_kernels(params)


# =====================================================================
# Per-cluster merge with sub-clustering
# =====================================================================

def merge_sizes_in_cluster(
    cluster_indices: Sequence[int],
    entries: Sequence[ConfigEntry],
    max_kernels: int,
) -> List[ConfigEntry]:
    """Merge entries in a cluster, splitting into sub-clusters if needed.

    Entries are merged incrementally.  When adding the next entry would push
    the kernel count above *max_kernels*, the current sub-cluster is
    finalized and a new one is started.

    Args:
        cluster_indices: Indices into *entries* for this cluster.
        entries: Per-size ConfigEntry objects.
        max_kernels: Maximum kernels allowed per config.

    Returns:
        List of merged ConfigEntry objects (one per sub-cluster).
    """
    result: List[ConfigEntry] = []

    first = entries[cluster_indices[0]]
    current_params = copy.deepcopy(first.fork_params)
    current_sizes = list(first.sizes)
    current_mis: Dict[Any, int] = dict(first.mis_per_size)

    for i in range(1, len(cluster_indices)):
        idx = cluster_indices[i]
        entry = entries[idx]
        tentative = _merge_two_param_dicts(current_params, entry.fork_params)
        tentative_count = count_kernels(tentative)

        if tentative_count > max_kernels:
            actual_count = _trim_mi_to_fit(current_params, max_kernels)
            result.append(ConfigEntry(
                sizes=current_sizes,
                fork_params=current_params,
                nkernels=actual_count,
                mis_per_size=current_mis,
            ))

            current_params = copy.deepcopy(entry.fork_params)
            current_sizes = list(entry.sizes)
            current_mis = dict(entry.mis_per_size)
        else:
            current_params = tentative
            current_sizes.extend(entry.sizes)
            current_mis.update(entry.mis_per_size)

    actual_count = _trim_mi_to_fit(current_params, max_kernels)
    result.append(ConfigEntry(
        sizes=current_sizes,
        fork_params=current_params,
        nkernels=actual_count,
        mis_per_size=current_mis,
    ))

    return result
