# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Workgroup-cluster shapes (``ClusterDim``) worth tuning on gfx1250.

A cluster of ``Cs x Ck`` workgroups (Cs along M, Ck along N of the tile grid)
places every member on its own WGP inside one shader engine, so tensor loads
the members share can be multicast. Two rules follow, and a third decides
between the shapes they leave.

Hardware. The size ``Cs * Ck`` must divide the WGPs of a shader engine, 16 on
MI450X/MI455X: clusters are placed whole, so with any other size every engine
keeps WGPs that no complete cluster fits on (a size of 6 keeps 12 of 16 busy).
The ISA caps a cluster at 16 workgroups and Tensile rejects the 16x1 and 1x16
shapes, which leaves 13 shapes on 16-WGP engines.

Problem. ``Cs`` must divide the grid's ``ceil(M / MT0)`` tiles and ``Ck`` its
``ceil(N / MT1)``. The grid is rounded up to whole clusters otherwise, and the
padding workgroups occupy WGPs only to exit.

Traffic. Members along N share the A tile and members along M the B tile, so
per output element and unit of K a workgroup fetches
``1 / (MT1 * Ck) + 1 / (MT0 * Cs)`` elements. Scaled by
``MT0 * MT1 * Cs * Ck``, that ranks the shapes of one size by
``MT0 * Cs + MT1 * Ck``; the minimum (ties kept) is the shape worth tuning at
that size. Sizes are not compared with each other: a larger cluster fetches
less but makes its members wait for each other, which only measurement can
weigh.
"""

from itertools import groupby
from math import ceil
from typing import List, Sequence, Tuple

# Workgroups a cluster may hold (CDNA5 ISA).
MAX_CLUSTER_WGS = 16

# Shapes Tensile rejects although their size is legal.
UNSUPPORTED_CLUSTER_DIMS = frozenset({(16, 1), (1, 16)})

SHADER_ENGINES_PER_XCC = 2

ClusterShape = Tuple[int, int]


def wgps_per_shader_engine(cus: int, xcc: int) -> int:
    """WGPs in one shader engine of a gfx1250 device.

    Args:
        cus: Compute units the device reports; on gfx1250 each is a WGP.
        xcc: XCCs the device spans (3 for a DPX partition of MI450X-MC).

    Raises:
        ValueError: If the counts do not split into whole shader engines.
    """
    engines = xcc * SHADER_ENGINES_PER_XCC
    if cus <= 0 or engines <= 0 or cus % engines:
        raise ValueError(f"{cus} WGPs do not split evenly across {engines} shader engines")
    return cus // engines


def hardware_cluster_dims(wgps_per_se: int) -> List[ClusterShape]:
    """Every shape whose size divides ``wgps_per_se`` and that Tensile accepts.

    Returns:
        Shapes ordered by size, then by ``Cs``; ``(1, 1)`` (no clustering) first.
    """
    shapes = []
    for size in range(1, min(MAX_CLUSTER_WGS, wgps_per_se) + 1):
        if wgps_per_se % size:
            continue
        for cs in range(1, size + 1):
            if size % cs == 0 and (cs, size // cs) not in UNSUPPORTED_CLUSTER_DIMS:
                shapes.append((cs, size // cs))
    return shapes


def tile_grid(m: int, n: int, macro_tile: Tuple[int, int]) -> Tuple[int, int]:
    """Output tiles along M and N for a problem and macro tile."""
    return ceil(m / macro_tile[0]), ceil(n / macro_tile[1])


def select_cluster_dims(
    candidates: Sequence[Sequence[int]],
    grid: Tuple[int, int],
    macro_tile: Tuple[int, int],
) -> List[ClusterShape]:
    """The candidate shapes worth tuning for one macro tile on one problem.

    Keeps the shapes that tile ``grid`` exactly and, among those of each size,
    the ones with the least traffic (see the module docstring). When no
    candidate tiles the grid, the choice is made over all candidates: padding
    workgroups are legal, merely idle.

    Args:
        candidates: ``[Cs, Ck]`` shapes, e.g. from :func:`hardware_cluster_dims`.
        grid: Output tiles along M and N (:func:`tile_grid`).
        macro_tile: ``(MT0, MT1)``.

    Returns:
        The selected shapes ordered by size, then by ``Cs``.
    """
    shapes = sorted({(int(c[0]), int(c[1])) for c in candidates}, key=lambda s: (s[0] * s[1], s[0]))
    fitting = [s for s in shapes if grid[0] % s[0] == 0 and grid[1] % s[1] == 0] or shapes
    mt0, mt1 = macro_tile
    selected: List[ClusterShape] = []
    for _, same_size in groupby(fitting, key=lambda s: s[0] * s[1]):
        same_size = list(same_size)
        least = min(mt0 * cs + mt1 * ck for cs, ck in same_size)
        selected.extend(s for s in same_size if mt0 * s[0] + mt1 * s[1] == least)
    return selected
