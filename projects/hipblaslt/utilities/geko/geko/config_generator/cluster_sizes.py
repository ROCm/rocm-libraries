################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

"""Clustering algorithms for grouping GEMM sizes.

Supports three modes:
  - cluster_algo=0: all sizes in one cluster (default)
  - cluster_algo=1: cluster by top MI configuration
  - is_one_size_per_config=True: each size in its own cluster
"""
import logging
from typing import Any, Dict, List, Sequence, Tuple

from geko.config_generator.mi_designer import MIDesign
from geko.config_generator.shared_utils import ConfigEntry

logger = logging.getLogger("GEKO")


def do_cluster(
    entries: Sequence[ConfigEntry],
    CUs: int,
    cluster_algo: int = 0,
    is_one_size_per_config: bool = False,
) -> Dict[Any, List[int]]:
    """Cluster sizes based on the selected algorithm.

    Args:
        entries: Per-size ConfigEntry objects.
        CUs: Number of compute units.
        cluster_algo: 0 = all-in-one (default), 1 = cluster by top MI configuration.
        is_one_size_per_config: If True, each size gets its own cluster.

    Returns:
        Mapping of cluster ids to lists of entry indices.
    """
    if is_one_size_per_config:
        return {i: [i] for i in range(len(entries))}

    if cluster_algo == 0:
        return {0: list(range(len(entries)))}

    if cluster_algo != 1:
        logger.warning(
            "Unknown cluster_algo=%d. Falling back to algo=1 (cluster by top MI configuration).",
            cluster_algo,
        )

    clusters = _cluster_sizes_mi(entries, CUs)
    clusters = _reorder_sizes_in_cluster(clusters, entries)
    return clusters


def _reorder_sizes_in_cluster(
    clusters: Dict[Any, List[int]],
    entries: Sequence[ConfigEntry],
) -> Dict[Any, List[int]]:
    """Reorder sizes within each cluster by area, then M, then N.

    Args:
        clusters: Cluster ids mapped to entry indices.
        entries: Per-size ConfigEntry objects.

    Returns:
        Clusters with entries reordered.
    """
    for cluster_id, indices in clusters.items():
        clusters[cluster_id] = sorted(
            indices,
            key=lambda idx: (
                entries[idx].sizes[0][0] * entries[idx].sizes[0][1],
                entries[idx].sizes[0][0],
                entries[idx].sizes[0][1],
            ),
        )
    return clusters


def _cluster_sizes_mi(
    entries: Sequence[ConfigEntry],
    CUs: int,
) -> Dict[Tuple[int, int, int, int], List[int]]:
    """Cluster sizes using Babak's MI-based heuristic.

    Each size is assigned to a cluster identified by the (MT0, MT1, LSU, GSU)
    signature of its highest-priority MI group entry.

    Args:
        entries: Per-size ConfigEntry objects.
        CUs: Number of compute units.

    Returns:
        Mapping of cluster signature to entry indices.
    """
    clusters: Dict[Tuple[int, int, int, int], List[int]] = {}
    for idx, entry in enumerate(entries):
        mi_groups = entry.fork_params["Groups"].values[0]

        if mi_groups:
            # Non-CMS MI groups from MIDesign are assumed ordered by granularity /
            # priority score. CMS tiles (UseCustomMainLoopSchedule) are appended
            # after that ordered list. Prefer the first MI group that is not CMS
            # so clustering keys off the highest-priority non-CMS MI; if every
            # group is CMS, fall back to the last entry.
            mi_group = next(
                (g for g in mi_groups if "UseCustomMainLoopSchedule" not in g),
                mi_groups[-1],
            )
            mi = list(mi_group["MatrixInstruction"].values) + [1, 1]
            if "WorkGroup" in mi_group:
                mi[-2] = mi_group["WorkGroup"].values[-1]
            if "GlobalSplitU" in mi_group:
                mi[-1] = mi_group["GlobalSplitU"].values[-1]
            MT0, MT1, *_ = MIDesign.calculate_mfma_parameters(mi)
            cluster_id = (MT0, MT1, mi[-2], mi[-1])
        else:
            cluster_id = (0, 0, 0, 0)

        clusters.setdefault(cluster_id, []).append(idx)

    return clusters
