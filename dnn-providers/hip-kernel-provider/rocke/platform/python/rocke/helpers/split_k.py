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

from .spec import ceil_div

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


@dataclass(frozen=True)
class WgradSplitPolicy:
    """Per-arch wgrad split-K degree policy, replacing the CK formulas.

    The degree is ``min(target / CTAs-at-split_k=1, wg_K / min_slice_k, cap)``:

    * ``target_ctas`` / ``grouped_target_ctas`` -- the launch size the degree
      fills to. A grouped launch fills best at a higher CTA count because its
      per-group tiles are small.
    * ``min_slice_k`` -- the shallowest K-slice worth a CTA; shallower slices pay
      the fixed per-CTA cost and one more atomic pass for little work.
    * ``max_degree`` / ``max_degree_by_dtype`` -- the cap, overridable per dW
      dtype. A packed 16-bit atomic rounds the running sum in dW on every add,
      so with same-sign partials the relative error grows with the degree, up
      to about ``degree * 2**-(fraction bits + 1)`` (see
      :func:`atomic_rounding_degree_cap`). A 16-bit dtype's cap keeps that
      bound under the 5e-2 tolerance the wgrad suite verifies against.
      A cap of 1 turns the automatic split off for that dtype; an explicit
      degree still launches it. The general cap is for fp32 dW, whose atomic
      rounding is negligible at any degree here.
    """

    target_ctas: int
    grouped_target_ctas: int
    min_slice_k: int
    max_degree: int
    max_degree_by_dtype: tuple[tuple[str, int], ...] = ()

    def degree_cap(self, dtype_d: str) -> int:
        return dict(self.max_degree_by_dtype).get(dtype_d, self.max_degree)


# Normalized error the wgrad correctness suite verifies dW against.
WGRAD_TOLERANCE = 5e-2


def atomic_rounding_degree_cap(fraction_bits: int, tol: float = WGRAD_TOLERANCE) -> int:
    """Largest split-K degree whose 16-bit atomic rounding stays within ``tol``.

    A dtype with ``fraction_bits`` explicit fraction bits has unit roundoff
    ``u = 2**-(fraction_bits + 1)`` under round-to-nearest (``2**-8`` for bf16,
    ``2**-11`` for fp16). A degree-``d`` split rounds each of its ``d`` partials
    to the dtype and then adds ``d - 1`` of them into the running sum in dW,
    so with same-sign partials the relative error is bounded by
    ``gamma_d = d*u / (1 - d*u)``. Returns the largest ``d`` with
    ``gamma_d <= tol`` (at least 1).
    """
    u = 2.0 ** -(fraction_bits + 1)
    degree = 1
    while (degree + 1) * u < 1 and (degree + 1) * u / (1 - (degree + 1) * u) <= tol:
        degree += 1
    return degree


# Arches whose wgrad split-K degree comes from a fitted policy rather than the
# CK formulas, whose fill targets leave these devices mostly idle. Fitted on
# gfx1250 bf16/fp16 wgrad.
#
# The bf16 cap is an accuracy bound, not a throughput fit: bf16 has 7 fraction
# bits, so its unit roundoff is 2**-8 and 12 same-sign partials stay within
# gamma_12 ~= 4.9e-2 of the exact sum, under the 5e-2 tolerance. Higher degrees
# are faster on small-channel shapes with a deep reduction, but all-positive
# inputs then exceed that tolerance (at 16, one 1.0 partial followed by fifteen
# 2**-8 ones rounds back to 1.0, a 5.5% error).
#
# fp16 has its own policy (_WGRAD_SPLIT_POLICY_BY_DTYPE below).
_WGRAD_SPLIT_POLICY: dict[str, WgradSplitPolicy] = {
    "gfx1250": WgradSplitPolicy(
        target_ctas=2048,
        grouped_target_ctas=4096,
        min_slice_k=2048,
        max_degree=128,
        max_degree_by_dtype=(("bf16", atomic_rounding_degree_cap(7)),),
    ),
}

# Per-dtype overrides of _WGRAD_SPLIT_POLICY, keyed by (arch, dW dtype).
#
# gfx1250 fp16: fill to 1024 CTAs (2048 grouped) with K slices as shallow as
# 512, up to degree 128. Fitted on a sample of wgrad shapes with every cached
# split-K binary timed at every degree 2..128 and verified, once the packed
# fp16 atomic was no longer fenced (it is an atomicrmw now). fp16 was not split
# automatically before (cap 1): that cap was a symptom of the fenced atomic.
# No fp16 degree failed the 5e-2 verification on those shapes, so the packed
# fp16 atomic's rounding is not the binding limit there.
_WGRAD_SPLIT_POLICY_BY_DTYPE: dict[tuple[str, str], WgradSplitPolicy] = {
    ("gfx1250", "fp16"): WgradSplitPolicy(
        target_ctas=1024,
        grouped_target_ctas=2048,
        min_slice_k=512,
        max_degree=128,
    ),
}

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

# Forward split-K (the two-stage WMMA implicit-GEMM conv on gfx1250, see
# select_split_k_fwd). Fitted on gfx1250 measurements of every power-of-two
# degree of the cached split-K forward tiles over small-output / deep-reduction
# layers:
#
# * the best degree puts about this many workgroups on each CU -- the short
#   K slices of a small tile are latency-bound, so several resident slices per
#   CU still pay;
FWD_SPLIT_K_TARGET_CTAS_PER_CU = 8
# * a slice of fewer K tiles than this is mostly prologue and epilogue (and
#   one more fp32 slab for Stage 2 to read);
FWD_SPLIT_K_MIN_K_TILES_PER_SLICE = 2
# * past this degree the slabs, not the K loop, dominate on every measured
#   layer;
FWD_SPLIT_K_MAX = 32
# * a reduction shallower than TARGET_SLICE_K elements is latency-bound as a
#   whole: the Stage 2 launch costs more than splitting it saves.
FWD_SPLIT_K_MIN_GEMM_K = TARGET_SLICE_K


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

    m_tiles = ceil_div(int(M), int(tile_m))
    n_tiles = ceil_div(int(N), int(tile_n))
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
    dtype_d: str | None = None,
) -> SplitKDecision:
    """Pick a split-K degree for a wgrad GEMM, mirroring CK's formula.

    ``dtype_d`` is the dW dtype. On an arch with a :class:`WgradSplitPolicy`
    it selects the policy (whose cap depends on it) in place of the CK
    formulas below; ``waves_per_cu`` and ``block_size`` then do not apply.
    Callers that pass none -- dgrad among them -- always get the CK formulas.

    Conv callers resolve split-K through a resolver that uses the split-K cost
    model (:mod:`rocke.helpers.split_k_planner`) where an arch has one and falls
    back to this function where it does not.

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

    ``groups <= 1`` (the default, and every dgrad resolution) takes the legacy
    expression above, unchanged. That is deliberate: dgrad resolves through
    this function too, without ``groups``. Keeping the ungrouped answer
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

    m_tiles = ceil_div(int(wg_M), int(tile_m))
    n_tiles = ceil_div(int(wg_N), int(tile_n))
    base_grid = m_tiles * n_tiles

    if base_grid <= 0:
        return SplitKDecision(1, base_grid, max_capacity, "empty grid")

    policy = (
        _WGRAD_SPLIT_POLICY_BY_DTYPE.get((arch, dtype_d), _WGRAD_SPLIT_POLICY.get(arch))
        if dtype_d is not None
        else None
    )
    if policy is not None:
        if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
            raise ValueError(f"groups must be a positive integer (got {groups!r})")
        n_groups = groups
        target = policy.grouped_target_ctas if n_groups > 1 else policy.target_ctas
        grid_no_split = base_grid * n_groups
        # Every bound is folded into one cap, so nothing applied after it can
        # raise the degree past any of them.
        cap = min(
            wg_K // policy.min_slice_k,
            policy.degree_cap(dtype_d),
            MAX_GRID_DIM_Z // n_groups,
        )
        split_k = max(1, min(target // grid_no_split, cap))
        return SplitKDecision(
            split_k,
            base_grid,
            target,
            f"{arch} wgrad policy ({dtype_d}): min(floor({target} / "
            f"{grid_no_split}), floor({wg_K} / {policy.min_slice_k}), "
            f"{policy.degree_cap(dtype_d)}) = {split_k}",
        )

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


def _pow2_floor(x: int) -> int:
    return 1 << (max(1, int(x)).bit_length() - 1)


def select_split_k_fwd(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    groups: int = 1,
    arch: str = "gfx1250",
    num_cus: int | None = None,
) -> SplitKDecision:
    """Pick a split-K degree for a forward implicit-GEMM convolution.

    ``M`` / ``N`` / ``K`` are one group's GEMM (``N*Ho*Wo`` x ``kpg`` x
    ``Y*X*cpg``); ``groups`` multiplies the grid. The forward split reduces
    through fp32 workspace slabs (one writer per element, folded by Stage 2 or
    the cluster), so unlike the 16-bit atomic reductions its accuracy does not
    depend on the degree: the degree is a performance choice only.

    ``num_cus`` is the device's CU count; ``None`` takes the arch's from
    ``_ARCH_NUM_CUS``, else the 256-CU default (gfx1250's own count, the arch
    the policy was fitted on). ::

        base_grid = ceil(M / tile_m) * ceil(N / tile_n) * groups
        target    = num_cus * FWD_SPLIT_K_TARGET_CTAS_PER_CU
        # engage gate: only a grid that leaves the device mostly idle, over a
        # reduction deep enough to pay for the Stage 2 launch
        if base_grid >= target / 2 or K < FWD_SPLIT_K_MIN_GEMM_K: split_k = 1
        split_k = pow2_floor(target / base_grid), held to
                  FWD_SPLIT_K_MAX and to pow2_floor(k_tiles /
                  FWD_SPLIT_K_MIN_K_TILES_PER_SLICE)

    The degree stays on the power-of-two ladder the policy was fitted on. It is
    not snapped to a divisor of the K tiles: the slice width is rounded up to
    ``tile_k`` and the tail reads zeros, so every degree is legal. Launch
    limits that depend on the binary (gridDim.z, the i32 workspace) are the
    caller's (:func:`kernels.common.conv_split_k.resolve_conv_split_k`).
    """
    if num_cus is None:
        num_cus = _ARCH_NUM_CUS.get(arch.split(":", 1)[0], _DEFAULT_NUM_CUS)
    target = num_cus * FWD_SPLIT_K_TARGET_CTAS_PER_CU
    base_grid = (
        ceil_div(int(M), int(tile_m)) * ceil_div(int(N), int(tile_n)) * max(1, groups)
    )
    if base_grid <= 0:
        return SplitKDecision(1, base_grid, target, "empty grid")
    if 2 * base_grid >= target:
        return SplitKDecision(
            1, base_grid, target, f"grid {base_grid} already fills the device"
        )
    if K < FWD_SPLIT_K_MIN_GEMM_K:
        return SplitKDecision(
            1,
            base_grid,
            target,
            f"reduction {K} < {FWD_SPLIT_K_MIN_GEMM_K} is too shallow to split",
        )
    k_tiles = ceil_div(int(K), int(tile_k))
    depth_cap = _pow2_floor(k_tiles // FWD_SPLIT_K_MIN_K_TILES_PER_SLICE)
    split_k = min(_pow2_floor(target // base_grid), FWD_SPLIT_K_MAX, depth_cap)
    if split_k < 2:
        return SplitKDecision(
            1, base_grid, target, f"{k_tiles} K tiles leave no room to split"
        )
    return SplitKDecision(
        split_k,
        base_grid,
        target,
        f"fwd fill: grid {base_grid} toward {target} workgroups "
        f"({FWD_SPLIT_K_TARGET_CTAS_PER_CU}/CU on {num_cus} CUs), "
        f"{ceil_div(k_tiles, split_k)} K tiles per slice -> split_k {split_k}",
    )
