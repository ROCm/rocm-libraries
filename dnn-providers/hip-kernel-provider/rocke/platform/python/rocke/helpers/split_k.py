# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split-K selection heuristic for skinny / tall-N decode GEMMs.

The production universal-GEMM body
(:mod:`rocke.instances.common.gemm_universal`) already implements split-K:
when ``TraitSpec.split_k > 1`` each CTA computes one ``(m_tile, n_tile)`` tile
over a K-slice ``[z*ks, (z+1)*ks)`` (``ks = K // split_k``) and atomic-adds its
partial f32 result into an f32 workspace ``Cf32[M, N]``. ``split_k == 1`` keeps
the canonical single-K-pass body byte-identical.

Split-K only *helps* when the base (split_k == 1) grid is far smaller than the
device CU count, so the device is mostly idle. That is exactly the decode /
GEMV-shaped regime: ``M`` is tiny (1-4 tokens) so there is ~1 m-tile and the
grid is just ``n_tiles`` (8-16 CTAs) against ~256 CUs on gfx950. Multiplying the
CTA count by ``split_k`` fills the idle CUs. For square / large / prefill GEMMs
the grid already saturates the device, so split-K would only add atomic-reduce
and workspace overhead -- there we must keep ``split_k == 1``.

This module is *dispatch / heuristic* only: it never touches the frozen kernel
body. It computes a valid split degree from the shape + tile + arch, gated so it
returns ``1`` for anything that already fills the device, and honors the
``ROCKE_GEMM_SPLIT_K`` env override (``auto`` / ``off`` / ``<n>``).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# gfx950 (MI350X) compute-unit count: 8 XCDs * 32 CUs. This is the number of
# CTAs needed to give every CU one resident block; the heuristic targets it as
# the grid "fill" point. Other CDNA arches differ but the decode regime (grid in
# the single/low-double digits) is well below all of them, so this constant is a
# conservative fill target across the family.
GFX950_NUM_XCDS = 8
GFX950_CU_PER_XCD = 32
GFX950_TARGET_CTAS = GFX950_NUM_XCDS * GFX950_CU_PER_XCD  # 256

# Per-arch CU counts for the wgrad split-K heuristic.
# Mirrors CK's ``get_num_cus()`` but as a static table so it works without a
# live device.  Keys match the ``gfx`` strings returned by ``ArchTarget``.
_ARCH_NUM_CUS: dict[str, int] = {
    "gfx942": 304,  # MI300X: 8 XCDs * 38 CUs
    "gfx950": 256,  # MI350X: 8 XCDs * 32 CUs
    "gfx1151": 64,  # RDNA gfx1151 (WMMA)
}
_DEFAULT_NUM_CUS = 256

# Assumed waves-per-CU for a 256-thread wgrad kernel (conservative; CK uses
# hipOccupancyMaxActiveBlocksPerMultiprocessor which we cannot call at build
# time).  2 is the safe lower bound for the LDS-heavy wgrad tile shapes.
WGRAD_ASSUMED_WAVES_PER_CU = 2

# XCD (accelerator-complex-die) count per arch. Workgroups are handed to XCDs
# round-robin by flat workgroup id, so this is the modulus a grouped launch has
# to stay aligned to -- see the snap in select_split_k_wgrad. RDNA parts have a
# single shader-engine group and want no snap at all.
_ARCH_NUM_XCDS: dict[str, int] = {
    "gfx942": 8,
    "gfx950": 8,
    "gfx1151": 1,
}
_DEFAULT_NUM_XCDS = 1

# Occupancy target for the *grouped* wgrad path, in WAVES per CU.
#
# ``WGRAD_ASSUMED_WAVES_PER_CU`` above is really a blocks-per-CU proxy pinned at
# 2 regardless of block size, which under-counts a small-block tile by the wave
# ratio: a 64-thread tile at 2 blocks/CU occupies 2 waves of a CU that holds far
# more. Ungrouped shapes never noticed because their base grid is large enough
# that the degree is small either way; a grouped shape has a base grid of 1-2
# tiles, so the whole degree comes from this constant.
#
# Sizing in waves and dividing by the tile's own waves-per-block makes the
# target block-size-invariant. The value is fitted on gfx950 across the grouped
# wgrad shapes in library/benchmarks; 32 and 64 both fit measurably worse.
# Treat it as gfx950-calibrated -- gfx942 inherits it untested.
WGRAD_TARGET_WAVES_PER_CU = 16

# gridDim.z is a 16-bit field in the HSA dispatch packet. Grouped wgrad rides
# ``groups * split_k`` on z, so the degree has a hard ceiling; exceeding it fails
# the launch with hipErrorInvalidValue rather than anything diagnosable.
MAX_GRID_DIM_Z = 65535

# Empirically (gfx950 decode-GEMM split-K sweep) the launch + atomic-reduce
# overhead floor sits near a per-slice K-depth of ~512 elements. Splitting K so
# each slice is ~512 deep maximises throughput: shallower slices let the fixed
# per-CTA overhead (launch + extra atomic partials) dominate, deeper slices leave
# CUs idle. The measured optimum across decode shapes tracks slice depth, NOT raw
# device fill -- K=2048 -> 4, K=4096 -> 8, K=768 -> 2, and a 128-CTA config at
# depth 512 beats a 256-CTA config at depth 256. So the auto heuristic sets the
# degree from slice depth (~K/512) and uses the CTA count only as the engage gate
# (don't split a grid that already fills the device).
TARGET_SLICE_K = 512

_ENV_FLAG = "ROCKE_GEMM_SPLIT_K"


@dataclass(frozen=True)
class SplitKDecision:
    """Result of :func:`select_split_k`.

    ``split_k`` is the chosen degree (``>= 1``). ``base_grid`` /
    ``target_ctas`` / ``reason`` are diagnostics for logging and tests.
    """

    split_k: int
    base_grid: int
    target_ctas: int
    reason: str


def _ceil_div(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _largest_valid_split_k(K: int, tile_k: int, cap: int) -> int:
    """Largest ``s`` in ``[1, cap]`` with ``K % s == 0`` and ``(K // s) % tile_k == 0``.

    The kernel slices ``ks = K // split_k`` and the K-loop strides by ``tile_k``,
    so both ``K % split_k`` and ``(K / split_k) % tile_k`` must be zero for the
    slices to tile cleanly (this mirrors the v1 StreamK partitioner contract).
    Falls back to ``1`` (always valid) if no larger factor qualifies. Prefers the
    largest qualifying factor; power-of-two candidates qualify like any other.
    """
    cap = max(1, int(cap))
    best = 1
    for s in range(2, cap + 1):
        if K % s == 0 and (K // s) % tile_k == 0:
            best = s
    return best


def _parse_env_override(env: dict) -> tuple[str, int | None]:
    """Parse ``ROCKE_GEMM_SPLIT_K``.

    Returns ``("auto", None)`` when unset/``auto``, ``("off", None)`` for
    ``off``/``0``/``1``, or ``("force", n)`` for a positive integer ``n``.
    """
    raw = env.get(_ENV_FLAG)
    if raw is None:
        return "auto", None
    val = raw.strip().lower()
    if val in ("", "auto"):
        return "auto", None
    if val in ("off", "0", "1", "none", "false"):
        return "off", None
    try:
        n = int(val)
    except ValueError:
        # Unrecognised value -> behave as auto (never crash dispatch on a typo).
        return "auto", None
    if n <= 1:
        return "off", None
    return "force", n


def select_split_k(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    batch: int = 1,
    target_ctas: int = GFX950_TARGET_CTAS,
    env: dict | None = None,
) -> SplitKDecision:
    """Pick a valid split-K degree for a GEMM shape on a CU-rich CDNA device.

    Heuristic (auto mode)::

        m_tiles      = ceil(M / tile_m)
        n_tiles      = ceil(N / tile_n)
        base_grid    = m_tiles * n_tiles * batch
        # engage gate: only split a grid that leaves the device mostly idle
        if base_grid >= target_ctas / 2: split_k = 1
        # degree from per-slice K-depth, not raw fill (see TARGET_SLICE_K)
        split_k_raw  = clamp(round(K / TARGET_SLICE_K), 1, K // tile_k)
        split_k      = largest valid factor <= split_k_raw
                       (K % s == 0 and (K / s) % tile_k == 0)

    The degree is set by slice depth (~``K / TARGET_SLICE_K``) rather than the
    CTA count needed to fill the device: the sweep showed the launch/atomic
    overhead floor is a slice-depth effect, so e.g. K=4096 wants split_k=8 (slice
    512) even though that over-subscribes the CUs, while K=2048 wants split_k=4
    (slice 512) even where raw fill would ask for more.

    Guards:

    * only engage when the base grid is genuinely small
      (``base_grid < target_ctas / 2``); otherwise return ``1`` so square /
      large / prefill GEMMs that already fill the device are untouched (the
      depth target never over-splits a well-filled grid because the gate fires
      first);
    * never return a degree that does not evenly slice K (falls back to ``1``).

    The ``ROCKE_GEMM_SPLIT_K`` env override takes precedence:

    * ``off`` / ``0`` / ``1`` -> force ``split_k = 1`` (disable);
    * ``<n>`` (>= 2) -> force that degree, snapped down to the largest valid
      factor <= ``n`` (so a forced value can never produce an unsliceable K);
    * unset / ``auto`` -> run the heuristic above.
    """
    env = os.environ if env is None else env
    mode, forced = _parse_env_override(env)

    m_tiles = _ceil_div(M, tile_m)
    n_tiles = _ceil_div(N, tile_n)
    base_grid = m_tiles * n_tiles * max(1, int(batch))
    max_split = max(1, K // tile_k)

    if mode == "off":
        return SplitKDecision(1, base_grid, target_ctas, "env override: off")

    if mode == "force":
        assert forced is not None
        cap = min(int(forced), max_split)
        chosen = _largest_valid_split_k(K, tile_k, cap)
        return SplitKDecision(
            chosen,
            base_grid,
            target_ctas,
            f"env override: forced {forced} -> valid {chosen}",
        )

    # auto: only split a grid that leaves the device mostly idle (the decode /
    # GEMV regime). A grid that already fills the device keeps split_k == 1 even
    # if it has deep K -- the gate fires before the depth target is consulted.
    if base_grid >= target_ctas // 2:
        return SplitKDecision(
            1, base_grid, target_ctas, f"grid {base_grid} already fills device"
        )

    # Degree from per-slice K-depth (~K / TARGET_SLICE_K), not raw CTA fill: the
    # measured optimum tracks slice depth (see TARGET_SLICE_K).
    split_k_raw = round(K / TARGET_SLICE_K) if K else 1
    split_k_raw = max(1, min(int(split_k_raw), max_split))
    chosen = _largest_valid_split_k(K, tile_k, split_k_raw)
    if chosen <= 1:
        return SplitKDecision(
            1,
            base_grid,
            target_ctas,
            f"no valid split factor <= {split_k_raw} (K-depth target)",
        )
    return SplitKDecision(
        chosen,
        base_grid,
        target_ctas,
        f"grid {base_grid} << target {target_ctas}; "
        f"K-depth target -> split_k {chosen} (slice K={K // chosen})",
    )


def select_split_k_wgrad(
    *,
    wg_M: int,
    wg_N: int,
    wg_K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    arch: str = "gfx950",
    waves_per_cu: int = WGRAD_ASSUMED_WAVES_PER_CU,
    groups: int = 1,
    block_size: int = 256,
) -> SplitKDecision:
    """Pick a split-K degree for a wgrad GEMM, mirroring CK's formula.

    CK's ``calculate_optimal_k_batch`` (``split_k_utils.hpp``) computes::

        grid_size    = ceil(M / tile_m) * ceil(N / tile_n)   [* batch, but batch=1 for wgrad]
        max_capacity = max_occupancy_per_cu * num_cus
        split_k      = floor(max_capacity / grid_size)       clamped to [1, wg_K]

    We cannot call ``hipOccupancyMaxActiveBlocksPerMultiprocessor`` at Python
    build time, so ``max_occupancy_per_cu`` is approximated by
    ``waves_per_cu`` (default: ``WGRAD_ASSUMED_WAVES_PER_CU = 2``), and
    ``num_cus`` is looked up from the static ``_ARCH_NUM_CUS`` table.

    The resulting ``split_k`` is the raw CK value; it is NOT snapped to a
    valid K-divisor because :class:`WgradConvSpec` pads K_wg to the next
    multiple of ``tile_k * split_k``, so any positive degree is legal.

    ``groups`` selects between two branches.

    ``groups <= 1`` (the default, and every dgrad caller) takes the legacy
    expression above, unchanged. That is deliberate: this function is shared
    with :mod:`conv_implicit_gemm_dgrad`, whose grouped shapes reach it through
    the benchmark driver with ``--split-k -1``. Keeping the ungrouped answer
    bit-for-bit means a grouped-wgrad retune cannot perturb dgrad.

    ``groups > 1`` corrects two things the CK formula gets wrong for a grouped
    wgrad, both of which only bite once the per-group GEMM is tiny:

    * **The base grid omitted the groups factor.** ``base_grid`` is one group's
      tile count, but the launch is ``base_grid * groups * split_k`` CTAs. A
      depthwise shape has ``base_grid == 1``, so the formula sized the degree
      for a single CTA and asked for hundreds -- past ``MAX_GRID_DIM_Z`` on the
      way. Multiplying it in is necessary but *not sufficient*: on its own it
      drives the degree to 1, which is worse still, because...
    * **...the capacity was a blocks-per-CU constant.** See
      ``WGRAD_TARGET_WAVES_PER_CU``. Sizing the target in waves and dividing by
      the tile's waves-per-block makes it block-size-invariant, which is what
      lets the groups term land on a sensible degree instead of collapsing it.

    Then the degree is snapped so that ``base_grid * split_k`` -- the flat
    workgroup-id stride between consecutive conv groups, given the launch
    ``(base_grid_x, base_grid_y, groups * split_k)`` and the kernel's
    ``z = group * split_k + slice`` decode -- is a multiple of the XCD count.
    Groups that share an NHWC cache line otherwise scatter across XCD L2s and
    each line gets fetched once per XCD. The snap is a no-op at ``groups == 1``,
    and measurably inert on shapes whose channel count is small enough that one
    cache line spans every group.

    The floor division is load-bearing: total CTAs must stay *at or under*
    capacity. Rounding up instead crosses a CTA-quantisation cliff.
    """
    num_cus = _ARCH_NUM_CUS.get(arch, _DEFAULT_NUM_CUS)
    max_capacity = waves_per_cu * num_cus

    m_tiles = _ceil_div(wg_M, tile_m)
    n_tiles = _ceil_div(wg_N, tile_n)
    base_grid = m_tiles * n_tiles

    if base_grid <= 0:
        return SplitKDecision(1, base_grid, max_capacity, "empty grid")

    if groups <= 1:
        split_k = max(1, int(max_capacity // base_grid))
        split_k = min(split_k, wg_K)

        return SplitKDecision(
            split_k,
            base_grid,
            max_capacity,
            f"CK formula: floor({max_capacity} / {base_grid}) = {split_k} "
            f"(num_cus={num_cus} waves_per_cu={waves_per_cu})",
        )

    wave_size = 32 if arch.startswith(("gfx10", "gfx11", "gfx12")) else 64
    waves_per_block = max(1, int(block_size) // wave_size)
    blocks_per_cu = max(1, WGRAD_TARGET_WAVES_PER_CU // waves_per_block)
    capacity = blocks_per_cu * num_cus

    # The real CTA count at split_k == 1, groups included.
    grid_no_split = base_grid * int(groups)
    split_k = max(1, capacity // grid_no_split)

    # Clamp BEFORE snapping. A clamp applied afterwards can land the degree off
    # a multiple of the step and undo the alignment: a shallow wg_K, or a large
    # group count against the z limit, pulls the snapped value straight back
    # down to an arbitrary number.
    cap = max(1, min(wg_K, MAX_GRID_DIM_Z // int(groups)))
    split_k = min(split_k, cap)

    xcds = _ARCH_NUM_XCDS.get(arch, _DEFAULT_NUM_XCDS)
    step = xcds // math.gcd(base_grid, xcds) if xcds > 1 else 1
    if step > 1:
        # Snap down to keep the CTA count under capacity; snap up only when the
        # degree is below one full step, where there is nothing to round down to.
        snapped = (split_k // step) * step or step
        # When the cap is itself below one step the stride cannot be aligned at
        # all; keep the legal degree rather than exceeding the cap for it.
        if snapped <= cap:
            split_k = snapped

    split_k = max(1, min(split_k, cap))

    return SplitKDecision(
        split_k,
        base_grid,
        capacity,
        f"grouped CK formula: floor({capacity} / ({base_grid} * {groups})) "
        f"snapped to a multiple of {step} = {split_k} "
        f"(num_cus={num_cus} waves_per_block={waves_per_block} xcds={xcds})",
    )
