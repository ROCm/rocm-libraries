# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Grid / workgroup-id remapping helpers for chiplet locality.

The AMD MI300X/MI325X/MI350X are multi-die chiplet GPUs: the global L2
(and on MI350X, the L3) is partitioned across XCDs (Accelerated Compute
Dies). The default linear-blockIdx assignment round-robins workgroups
across XCDs (WG 0 -> XCD 0, WG 1 -> XCD 1, ...), which is the WORST case
for L2 reuse — two consecutive output tiles (which usually share input
rows) end up on different XCDs and have to refetch the same data
through the cross-die fabric.

``chiplet_transform_chunked`` remaps the linear WGID so that every
``chunk_size`` consecutive WGs land in the *same* XCD. Within each
XCD a standard "super-tile" / Hilbert-like swizzle
(:func:`super_tile_swizzle`) further increases spatial locality of
adjacent tile loads.

Both helpers in this module emit IR for the in-kernel address math via
an :class:`IRBuilder`. The runtime grid is sized as usual (e.g.
``grid = (M_tiles * N_tiles, 1, 1)`` or ``(N_tiles, M_tiles, 1)``); the
remap happens at the *top* of every kernel body.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ..core.ir import IRBuilder, Value


# Default chiplet counts per AMDGPU family. Override at the call site
# for other GPUs.
NUM_XCDS_MI300X = 8  # CDNA3
NUM_XCDS_MI325X = 8  # CDNA3
NUM_XCDS_MI350X = 8  # CDNA4 (note: physical XCDs may be more)


def chiplet_transform_chunked(
    b: IRBuilder,
    wgid: Value,
    *,
    num_wgs: int,
    num_xcds: int = NUM_XCDS_MI300X,
    chunk_size: int = 64,
) -> Value:
    """Remap a linear WGID so every ``chunk_size`` consecutive WGs land
    on the same XCD.

    This is the inverse of the default round-robin XCD assignment:

    - HW default: WG ``i`` -> XCD ``i % num_xcds``.
    - After remap: WG ``i`` (logical) -> XCD ``(i / chunk_size) % num_xcds``,
      i.e. WGs ``[0, chunk_size)`` all hit XCD 0, ``[chunk_size, 2*chunk_size)``
      hit XCD 1, etc.

    The remap is identity for WGs past the last complete
    ``num_xcds * chunk_size`` block — the tail uses the original WGID,
    so we don't synthesise an out-of-range index.

    Parameters
    ----------
    b
        IRBuilder to emit math into.
    wgid
        The linear blockIdx (i32 SSA, typically
        ``b.block_id_x()`` for a 1D grid).
    num_wgs
        Total number of workgroups in the kernel launch (compile-time
        constant). Pass `M_tiles * N_tiles` for a 1D-flat grid.
    num_xcds
        XCD count for the GPU. Use one of the ``NUM_XCDS_*`` constants.
    chunk_size
        WGs per XCD per block. 64 is a good default for MI300X /
        MI350X; smaller values give finer locality at the cost of
        more cross-XCD spills.

    Returns
    -------
    SSA ``Value`` (i32) with the remapped WGID.
    """
    if num_xcds <= 0 or chunk_size <= 0 or num_wgs <= 0:
        raise ValueError(
            f"chiplet_transform_chunked: invalid args "
            f"num_wgs={num_wgs} num_xcds={num_xcds} chunk_size={chunk_size}"
        )

    block = num_xcds * chunk_size
    limit = (num_wgs // block) * block

    c_num_xcds = b.const_i32(int(num_xcds))
    c_chunk_size = b.const_i32(int(chunk_size))
    c_block = b.const_i32(int(block))
    c_limit = b.const_i32(int(limit))

    xcd = b.mod(wgid, c_num_xcds)
    local_pid = b.div(wgid, c_num_xcds)
    chunk_idx = b.div(local_pid, c_chunk_size)
    pos_in_chunk = b.mod(local_pid, c_chunk_size)

    new_wgid = b.add(
        b.add(b.mul(chunk_idx, c_block), b.mul(xcd, c_chunk_size)),
        pos_in_chunk,
    )
    # Tail handling: WGs past the last full block keep their original
    # WGID (so we don't synthesise an invalid index).
    in_full_block = b.cmp_lt(wgid, c_limit)
    return b.select(in_full_block, new_wgid, wgid)


@dataclass(frozen=True)
class SuperTileSwizzleResult:
    """Tile (row, col) decomposition produced by :func:`super_tile_swizzle`."""

    row: Value  # i32 SSA, M-tile index
    col: Value  # i32 SSA, N-tile index


def super_tile_swizzle(
    b: IRBuilder,
    wgid: Value,
    *,
    num_pid_m: int,
    num_pid_n: int,
    wgm: int = 8,
) -> SuperTileSwizzleResult:
    """ "WGM super-tile" grid swizzle for L2 locality within an XCD.

    Groups ``wgm * num_pid_n`` consecutive WGs into a super-tile of
    shape ``(wgm, num_pid_n)``. Inside the super-tile, the WGs walk
    column-first instead of the default row-first order. This means
    two WGs that share an M-tile (and therefore share an A-row) end up
    adjacent in the dispatch stream — they hit L2 *after* the
    chiplet remap has already pinned them to the same XCD.

    This is the same idea as Triton's ``GROUP_M`` matmul swizzle.

    Parameters
    ----------
    b
        IRBuilder.
    wgid
        Linear workgroup id (typically the output of
        :func:`chiplet_transform_chunked`).
    num_pid_m, num_pid_n
        Tile-count along M and N (compile-time constants, e.g.
        ``ceil_div(M, BLOCK_M)``).
    wgm
        Super-tile height (M-stride). 4 or 8 are typical depending on
        problem shape; 8 is a safe default.

    Returns
    -------
    SuperTileSwizzleResult with ``row`` = M-tile, ``col`` = N-tile,
    both i32 SSA.
    """
    if num_pid_m <= 0 or num_pid_n <= 0 or wgm <= 0:
        raise ValueError(
            f"super_tile_swizzle: invalid args "
            f"num_pid_m={num_pid_m} num_pid_n={num_pid_n} wgm={wgm}"
        )

    num_wgid_in_group = wgm * num_pid_n
    c_wgm = b.const_i32(int(wgm))
    c_num_pid_m = b.const_i32(int(num_pid_m))
    c_num_wgid_in_group = b.const_i32(int(num_wgid_in_group))

    group_id = b.div(wgid, c_num_wgid_in_group)
    first_pid_m = b.mul(group_id, c_wgm)
    # group_size_m = min(num_pid_m - first_pid_m, wgm) for tail handling.
    rem = b.sub(c_num_pid_m, first_pid_m)
    use_wgm = b.cmp_lt(c_wgm, rem)
    group_size_m = b.select(use_wgm, c_wgm, rem)

    local_id = b.mod(wgid, c_num_wgid_in_group)
    pid_m = b.add(first_pid_m, b.mod(local_id, group_size_m))
    pid_n = b.div(local_id, group_size_m)

    return SuperTileSwizzleResult(row=pid_m, col=pid_n)


def chiplet_transform_chunked_dynamic(
    b: IRBuilder,
    wgid: Value,
    *,
    num_wgs: Value,
    num_xcds: int = NUM_XCDS_MI300X,
    chunk_size: int = 64,
) -> Value:
    """Runtime variant of :func:`chiplet_transform_chunked`.

    ``num_wgs`` is an i32 SSA value (typically
    ``M_tiles * N_tiles`` computed at kernel entry from the dynamic
    M/N args). ``num_xcds`` and ``chunk_size`` remain compile-time
    constants — they're tied to the GPU family, not the problem.
    """
    if num_xcds <= 0 or chunk_size <= 0:
        raise ValueError(
            f"chiplet_transform_chunked_dynamic: invalid args "
            f"num_xcds={num_xcds} chunk_size={chunk_size}"
        )
    block = num_xcds * chunk_size
    c_num_xcds = b.const_i32(int(num_xcds))
    c_chunk_size = b.const_i32(int(chunk_size))
    c_block = b.const_i32(int(block))

    # limit = (num_wgs / block) * block  (largest full-block boundary)
    limit = b.mul(b.div(num_wgs, c_block), c_block)

    xcd = b.mod(wgid, c_num_xcds)
    local_pid = b.div(wgid, c_num_xcds)
    chunk_idx = b.div(local_pid, c_chunk_size)
    pos_in_chunk = b.mod(local_pid, c_chunk_size)

    new_wgid = b.add(
        b.add(b.mul(chunk_idx, c_block), b.mul(xcd, c_chunk_size)),
        pos_in_chunk,
    )
    in_full_block = b.cmp_lt(wgid, limit)
    return b.select(in_full_block, new_wgid, wgid)


def super_tile_swizzle_dynamic(
    b: IRBuilder,
    wgid: Value,
    *,
    num_pid_m: Value,
    num_pid_n: Value,
    wgm: int = 8,
) -> SuperTileSwizzleResult:
    """Runtime variant of :func:`super_tile_swizzle`.

    ``num_pid_m`` and ``num_pid_n`` are i32 SSA values. The ``wgm``
    constant stays compile-time because the AMDGPU backend folds it
    into ``s_lshl_b32`` / ``s_mul_i32`` immediates.
    """
    if wgm <= 0:
        raise ValueError(f"super_tile_swizzle_dynamic: wgm={wgm} must be > 0")
    c_wgm = b.const_i32(int(wgm))
    num_wgid_in_group = b.mul(c_wgm, num_pid_n)

    group_id = b.div(wgid, num_wgid_in_group)
    first_pid_m = b.mul(group_id, c_wgm)
    rem = b.sub(num_pid_m, first_pid_m)
    use_wgm = b.cmp_lt(c_wgm, rem)
    group_size_m = b.select(use_wgm, c_wgm, rem)

    local_id = b.mod(wgid, num_wgid_in_group)
    pid_m = b.add(first_pid_m, b.mod(local_id, group_size_m))
    pid_n = b.div(local_id, group_size_m)

    return SuperTileSwizzleResult(row=pid_m, col=pid_n)


def chiplet_aware_super_tile_dynamic(
    b: IRBuilder,
    wgid: Value,
    *,
    num_pid_m: Value,
    num_pid_n: Value,
    wgm: int = 8,
    num_xcds: int = NUM_XCDS_MI300X,
    chunk_size: int = 64,
) -> SuperTileSwizzleResult:
    """Runtime composition of chiplet remap + super-tile swizzle.

    Use in kernels with dynamic M/N (e.g. universal GEMM whose tile
    counts are computed at kernel entry from i32 args). Combines
    :func:`chiplet_transform_chunked_dynamic` with
    :func:`super_tile_swizzle_dynamic`.
    """
    num_wgs = b.mul(num_pid_m, num_pid_n)
    remapped = chiplet_transform_chunked_dynamic(
        b,
        wgid,
        num_wgs=num_wgs,
        num_xcds=num_xcds,
        chunk_size=chunk_size,
    )
    return super_tile_swizzle_dynamic(
        b,
        remapped,
        num_pid_m=num_pid_m,
        num_pid_n=num_pid_n,
        wgm=wgm,
    )


def chiplet_aware_super_tile(
    b: IRBuilder,
    wgid: Value,
    *,
    num_pid_m: int,
    num_pid_n: int,
    wgm: int = 8,
    num_xcds: int = NUM_XCDS_MI300X,
    chunk_size: int = 64,
) -> SuperTileSwizzleResult:
    """One-shot composition: chiplet remap then super-tile swizzle.

    Equivalent to::

        remapped = chiplet_transform_chunked(b, wgid, num_wgs=..., num_xcds=...)
        result = super_tile_swizzle(b, remapped, num_pid_m=..., num_pid_n=..., wgm=...)

    This is the canonical kernel-entry preamble for chiplet-aware
    GEMM / conv kernels: every workgroup remaps its blockIdx into
    ``(pid_m, pid_n)`` *before* loading any tile coordinates.
    """
    num_wgs = int(num_pid_m) * int(num_pid_n)
    remapped = chiplet_transform_chunked(
        b,
        wgid,
        num_wgs=num_wgs,
        num_xcds=num_xcds,
        chunk_size=chunk_size,
    )
    return super_tile_swizzle(
        b,
        remapped,
        num_pid_m=num_pid_m,
        num_pid_n=num_pid_n,
        wgm=wgm,
    )


def python_chiplet_transform_chunked(
    wgid: int,
    *,
    num_wgs: int,
    num_xcds: int = NUM_XCDS_MI300X,
    chunk_size: int = 64,
) -> int:
    """Pure-Python reference for :func:`chiplet_transform_chunked`.

    Used by unit tests to validate the IR-emitting helper against
    the well-known closed-form remap.
    """
    block = num_xcds * chunk_size
    limit = (num_wgs // block) * block
    if wgid >= limit:
        return wgid
    xcd = wgid % num_xcds
    local_pid = wgid // num_xcds
    chunk_idx = local_pid // chunk_size
    pos_in_chunk = local_pid % chunk_size
    return chunk_idx * block + xcd * chunk_size + pos_in_chunk


def python_super_tile_swizzle(
    wgid: int,
    *,
    num_pid_m: int,
    num_pid_n: int,
    wgm: int = 8,
) -> Tuple[int, int]:
    """Pure-Python reference for :func:`super_tile_swizzle`.

    Returns ``(pid_m, pid_n)``.
    """
    num_wgid_in_group = wgm * num_pid_n
    group_id = wgid // num_wgid_in_group
    first_pid_m = group_id * wgm
    group_size_m = min(num_pid_m - first_pid_m, wgm)
    local_id = wgid % num_wgid_in_group
    pid_m = first_pid_m + (local_id % group_size_m)
    pid_n = local_id // group_size_m
    return pid_m, pid_n
