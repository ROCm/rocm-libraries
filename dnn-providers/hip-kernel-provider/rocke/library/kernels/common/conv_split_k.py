# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Split-K resolution for the implicit-GEMM convolutions (fwd, wgrad, dgrad).

The one place that turns a conv problem and a kernel tile into a split-K
degree and a reduction path. The kernel builders and the benchmark (JIT
sweep and ``--run-from-cache``) all call :func:`resolve_conv_split_k`, so a
degree requested as ``split_k=-1`` (auto) comes out the same wherever it is
resolved -- the benchmark times the degree a builder would build, and a
builder never re-derives it from different inputs. (The grouped-convolution
dispatcher serves registered instance lists and registers no wgrad instances,
so it does not resolve a wgrad degree.)

An auto degree comes from the split-K cost model
(:mod:`kernels.conv.split_k_planner`) on arches that have one. This module
contributes what is conv-specific:

* the GEMM each direction launches: ``kpg x Z*Y*X*cpg x N*[Do*]Ho*Wo`` for
  wgrad, widened by ``group_merge`` (a merged tile covers that many groups'
  channels); ``N*Hi*Wi x cpg x Y*X*kpg`` for dgrad;
* the reductions the kernels can build, with their limits: the packed 16-bit
  atomic needs an even dW row and an even store vector
  (:func:`wgrad_atomic_epilogue_available`) and accumulates rounding error with
  the degree (:data:`ATOMIC_16BIT_MAX_SPLIT_K`); a merged tile reduces through
  the workspace only; the workspace must fit the i32 ``ws_bytes`` kernarg;
  on WMMA (wave32) the workspace path needs the arch's float global atomics
  (gfx1250), and the packed atomic needs neither cshuffle nor an even store
  vector;
* the gridDim.z budget (``z = groups * split_k``);
* for the forward split (the two-stage kernel, WMMA or MFMA, built on
  :data:`FWD_SPLIT_K_ARCHS` only): its GEMM ``N*[Do*]Ho*Wo x kpg x
  [Z*]Y*X*cpg``, the problems it cannot take (pointwise, 3-D) and the i32
  workspace of ``split_k`` fp32 slabs of the output. Its auto degree comes from
  :func:`select_split_k_fwd` -- no arch it is built for
  has a fitted cost model. Its partials meet in fp32 slabs, never in 16-bit
  atomics, so no accuracy cap applies.

An explicit degree, an override, and an auto degree on an arch without a cost
model (gfx942 included: it has not been fitted) all go through the
long-standing dispatcher rule instead: the CK formula
(:func:`select_split_k_wgrad`) for auto, then the packed
atomic whenever it can be built, else the workspace, else no split. The kernel
builders and the benchmark, which used to feed the CK formula unmerged
per-group dims, get it on the launched geometry (merged dims, launched group
count).

``override`` forces the degree of an auto request: ``1`` forces no split, an
integer ``n >= 2`` forces that degree, held to what the kernel can build --
including :data:`ATOMIC_16BIT_MAX_SPLIT_K` for the packed 16-bit atomic (a
wgrad request free to pick its reduction moves to the workspace above it). An
explicit ``split_k`` is never overridden. The resolver reads no environment, so
a spec resolves the same way in every process; the benchmark maps its
``ROCKE_CONV_SPLIT_K`` variable onto ``override`` through
:func:`parse_split_k_override`.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from rocke.core.arch.target import ArchTarget, base_arch_from_target_id
from rocke.helpers.spec import ceil_div
from rocke.helpers.split_k import TARGET_SLICE_K, SplitKDecision
from rocke.helpers.split_k import num_cus as arch_num_cus

from kernels.conv.split_k_planner import (
    MAX_GRID_DIM_Z,
    SplitKKernel,
    SplitKProblem,
    SplitKReduction,
    select_split_k_plan,
    split_k_cost_model,
)

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_fwd_split_k_reduce import fwd_split_k_workspace_nbytes
from kernels.common.conv_implicit_gemm_dgrad import (
    _dg_K,
    _dg_M,
    _dg_N,
    dgrad_split_k_available,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    _DEFAULT_WS_REPLICAS,
    _wg_K,
    _wg_M,
    _wg_N,
    wgrad_atomic_epilogue_available,
    wgrad_two_stage_available,
    wgrad_wmma_split_k_available,
)
from kernels.common.conv_implicit_gemm_wgrad_two_stage import wgrad_workspace_nbytes

__all__ = [
    "ATOMIC_16BIT_MAX_SPLIT_K",
    "FWD_SPLIT_K_ARCHS",
    "MAX_WGRAD_WS_BYTES",
    "WGRAD_ASSUMED_WAVES_PER_CU",
    "WGRAD_TARGET_WAVES_PER_CU",
    "ConvSplitK",
    "parse_split_k_override",
    "resolve_conv_split_k",
    "select_split_k_wgrad",
]

# Highest degree the planner gives the packed bf16/fp16 atomic reduction. Each
# slice rounds its partial into the 16-bit output, so the error grows with the
# degree: on the gfx950 wgrad corpus the verify gate fails broadly from 256 on,
# while the f32 workspace reduction passes at every degree. An override is held
# to it too; only an explicit degree is the caller's own.
ATOMIC_16BIT_MAX_SPLIT_K = 128

# ws_bytes is an i32 kernarg, so a two-stage workspace above this overflows it.
MAX_WGRAD_WS_BYTES = (1 << 31) - 4

# Targets the split-K forward kernel is built and validated for (the AOT cache
# builds it for these only). Elsewhere an auto forward request resolves to 1.
# gfx950 runs it on MFMA (wave64): the same plain K loop over one slice and the
# same fp32 slab + Stage 2 fold.
FWD_SPLIT_K_ARCHS = ("gfx1250", "gfx950")
# The forward workspace is addressed through an i32 byte count like every
# operand.
MAX_FWD_WS_BYTES = (1 << 31) - 1

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
# 512. Fitted on a sample of wgrad shapes with every cached split-K binary
# timed at every degree 2..128 and verified, once the packed fp16 atomic was
# no longer fenced (it is an atomicrmw now). fp16 was not split automatically
# before (cap 1): that cap was a symptom of the fenced atomic.
#
# The degree is capped by the packed fp16 atomic's rounding, as bf16's is: fp16
# has 10 fraction bits (unit roundoff 2**-11), so 97 same-sign partials stay
# within the 5e-2 tolerance. Beyond it valid inputs can miss it even where the
# fitted shapes did not (at 128, one 1.0 partial followed by 127 partials of
# 2**-11 rounds back to 1.0, a 5.8% error). The cap binds only the atomic
# reduction: the f32 workspace (two-stage) degree is resolved as fp32 dW.
_WGRAD_SPLIT_POLICY_BY_DTYPE: dict[tuple[str, str], WgradSplitPolicy] = {
    ("gfx1250", "fp16"): WgradSplitPolicy(
        target_ctas=1024,
        grouped_target_ctas=2048,
        min_slice_k=512,
        max_degree=128,
        max_degree_by_dtype=(("fp16", atomic_rounding_degree_cap(10)),),
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

_DIRECTIONS = ("fwd", "wgrad", "dgrad")
_REDUCTIONS = ("auto", "atomic", "workspace")


@dataclass(frozen=True)
class ConvSplitK:
    """A resolved split-K configuration.

    Attributes:
        split_k: the degree to build and launch (1 = no split).
        two_stage: reduce through the f32 workspace and Stage 2 (wgrad; the
            forward split always does, so it is set whenever it splits).
        requested: the degree before the launch limits applied -- the explicit
            value, the override, or the auto choice.
        reason: why, for logs and diagnostics.
    """

    split_k: int
    two_stage: bool
    requested: int
    reason: str


def parse_split_k_override(text: str | None) -> int | None:
    """Parse a split-K override setting into the resolver's ``override``.

    ``None``, ``""`` or ``"auto"``: no override (``None``). ``"off"`` (or
    ``"0"``/``"1"``): no split (``1``). An integer ``n >= 2``: that degree.
    Anything else is an error.
    """
    raw = (text or "").strip().lower()
    if raw in ("", "auto"):
        return None
    if raw in ("off", "none", "false"):
        return 1
    try:
        n = int(raw)
    except ValueError:
        n = -1
    if n < 0:
        raise ValueError(
            f"split-K override {raw!r}: expected 'auto', 'off' or an integer >= 0"
        )
    return max(1, n)


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
    model (:mod:`kernels.conv.split_k_planner`) where an arch has one and falls
    back to this function where it does not.

    CK's ``calculate_optimal_k_batch`` (``split_k_utils.hpp``) computes::

        grid_size    = ceil(M / tile_m) * ceil(N / tile_n)   [* batch, but batch=1 for wgrad]
        max_capacity = max_occupancy_per_cu * num_cus
        split_k      = floor(max_capacity / grid_size)       clamped to [1, wg_K]

    We cannot call ``hipOccupancyMaxActiveBlocksPerMultiprocessor`` at Python
    build time, so ``max_occupancy_per_cu`` is approximated by
    ``waves_per_cu`` (default: ``WGRAD_ASSUMED_WAVES_PER_CU = 2``), and
    ``num_cus`` is looked up with :func:`rocke.helpers.split_k.num_cus`.

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
    # Every table below is keyed by the bare architecture, so strip a target
    # id's feature suffix (``gfx942:sramecc+``) once, before any lookup.
    base_arch = base_arch_from_target_id(arch)
    num_cus = arch_num_cus(base_arch)
    max_capacity = waves_per_cu * num_cus

    m_tiles = ceil_div(int(wg_M), int(tile_m))
    n_tiles = ceil_div(int(wg_N), int(tile_n))
    base_grid = m_tiles * n_tiles

    if base_grid <= 0:
        return SplitKDecision(1, base_grid, max_capacity, "empty grid")

    policy = (
        _WGRAD_SPLIT_POLICY_BY_DTYPE.get(
            (base_arch, dtype_d), _WGRAD_SPLIT_POLICY.get(base_arch)
        )
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

    # An arch the catalog does not list raises KeyError.
    wave_size = ArchTarget.from_gfx(base_arch).wave_size
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

    xcds = _ARCH_NUM_XCDS.get(base_arch, _DEFAULT_NUM_XCDS)
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
    ``rocke.helpers.split_k.ARCH_NUM_CUS``, else the 256-CU default (gfx1250's
    own count, the arch the policy was fitted on). ::

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
        num_cus = arch_num_cus(base_arch_from_target_id(arch))
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


def _is_16bit(dtype: str) -> bool:
    return dtype in ("bf16", "fp16")


def _operand_bytes(dtype: str) -> int:
    # The conv kernels read A and B, and write the output, in one dtype;
    # tf32 is a compute mode over fp32 storage.
    return 4 if dtype in ("fp32", "tf32") else 2


def _atomic_reduction(dtype: str) -> SplitKReduction:
    if _is_16bit(dtype):
        return SplitKReduction("atomic", dtype, max_split_k=ATOMIC_16BIT_MAX_SPLIT_K)
    return SplitKReduction("atomic", "fp32")


def _resolve_wgrad(
    p: ConvProblem,
    *,
    arch: str,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    group_merge: int,
    ws_replicas: int,
    split_k: int,
    reduction: str,
    epilogue: str | None,
    allow_unsplit: bool,
    override: int | None,
    num_cus: int | None,
    num_xcds: int | None,
) -> ConvSplitK:
    gm = max(1, group_merge)
    ws_replicas = ws_replicas if ws_replicas > 0 else _DEFAULT_WS_REPLICAS
    grid_groups = max(1, p.groups // gm)
    z_cap = MAX_GRID_DIM_Z // grid_groups
    # Merged tiles cannot use the packed atomic: its epilogue has no way to
    # drop an off-diagonal group pair, so only the workspace carries the mask.
    # On MFMA the packed 16-bit atomic is emitted by the cshuffle epilogue only
    # (the direct-store one scatters the MFMA layout and has no pairs to add).
    # On WMMA (wave32) the default split-K epilogue forms each packed pair
    # itself; the cshuffle one pairs from LDS as on MFMA.
    wave_size = ArchTarget.from_gfx(arch).wave_size
    layout_ok, atomic_why = wgrad_atomic_epilogue_available(
        p, dtype, None, wave_size, epilogue
    )
    if gm > 1:
        atomic_why = "a merged tile reduces through the workspace only"
    elif wave_size == 32 and not wgrad_wmma_split_k_available(arch):
        # Same gate as is_valid_wgrad_spec: the WMMA epilogue needs the float
        # global atomics the arch catalog marks as run on a device.
        atomic_why = (
            f"{arch} has no validated float global atomics "
            "(has_global_float_atomic_add)"
        )
    elif layout_ok and epilogue == "default" and _is_16bit(dtype) and wave_size != 32:
        atomic_why = "the default epilogue emits no packed 16-bit atomic"
    atomic_ok = atomic_why == "ok"
    # Deliberately unmerged: the workspace is R copies of the true dW. WMMA
    # arches without float global atomics have no two-stage path, so no
    # workspace at all.
    ws_ok = (
        wgrad_two_stage_available(arch)
        and wgrad_workspace_nbytes(p, ws_replicas) <= MAX_WGRAD_WS_BYTES
    )
    if reduction == "workspace" and not ws_ok:
        # Fixing the reduction does not lift its limit: the caller's two-stage
        # binary cannot encode this workspace at any degree.
        raise ValueError(
            "wgrad split-K cannot be built: the workspace exceeds the i32 "
            f"ws_bytes limit (ws_replicas={ws_replicas})"
        )

    def _rule(requested: int, source: str, *, atomic_cap: bool) -> ConvSplitK:
        """The long-standing path choice for a given degree.

        ``atomic_cap`` holds the packed 16-bit atomic to its accuracy limit: a
        degree above it moves to the workspace when the reduction is free,
        else comes down to the limit.
        """
        degree = max(1, min(requested, z_cap))
        if degree == 1:
            return ConvSplitK(1, False, requested, f"{source}: no split")
        if reduction == "atomic" and not atomic_ok:
            # Fixing the reduction is the caller saying its binary is the packed
            # atomic one; it cannot be, so fail here rather than hand back a
            # degree the builder then rejects (as for the workspace above).
            raise ValueError(
                f"wgrad split-K cannot be built with the packed atomic: {atomic_why}"
            )
        over_cap = atomic_cap and _is_16bit(dtype) and degree > ATOMIC_16BIT_MAX_SPLIT_K
        # A free reduction takes the packed atomic when it can, unless the
        # degree is past the atomic's limit and the workspace can take it.
        use_atomic = reduction == "atomic" or (
            reduction == "auto" and atomic_ok and not (over_cap and ws_ok)
        )
        if use_atomic:
            if over_cap:
                return ConvSplitK(
                    ATOMIC_16BIT_MAX_SPLIT_K,
                    False,
                    requested,
                    f"{source}: atomic, held to the 16-bit atomic limit "
                    f"{ATOMIC_16BIT_MAX_SPLIT_K}",
                )
            return ConvSplitK(degree, False, requested, f"{source}: atomic")
        if ws_ok:
            return ConvSplitK(degree, True, requested, f"{source}: workspace")
        return ConvSplitK(
            1, False, requested, f"{source}: workspace too large, no split"
        )

    if split_k != -1:
        return _rule(split_k, f"explicit split_k={split_k}", atomic_cap=False)
    if override is not None:
        return _rule(override, f"override split_k={override}", atomic_cap=True)

    # The launched GEMM: a merged tile covers gm groups' channels at once.
    m, n, k = _wg_M(p) * gm, _wg_N(p) * gm, _wg_K(p)
    cost = split_k_cost_model(arch, num_cus, num_xcds)
    if cost is None:
        legacy = select_split_k_wgrad(
            wg_M=m,
            wg_N=n,
            wg_K=k,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            arch=arch,
            groups=grid_groups,
            block_size=block_size,
            # Selects the arch's WgradSplitPolicy (gfx1250) where it has one.
            # The workspace reduction accumulates in fp32 whatever dW's dtype,
            # so the 16-bit packed-atomic accuracy caps do not apply to it.
            dtype_d="fp32" if reduction == "workspace" else dtype,
        )
        if reduction == "atomic" and not atomic_ok and allow_unsplit:
            # An auto degree the packed atomic cannot serve runs unsplit, as
            # on the planner path (e.g. a WMMA 16-bit dW with an odd row and
            # the reduction fixed to the atomic: no packed pair). A forced
            # degree still raises.
            return ConvSplitK(
                1,
                False,
                legacy.split_k,
                f"CK formula ({legacy.reason}); no split: {atomic_why}",
            )
        return _rule(legacy.split_k, f"CK formula ({legacy.reason})", atomic_cap=False)

    reductions = []
    if reduction in ("auto", "atomic") and atomic_ok:
        reductions.append(_atomic_reduction(dtype))
    if reduction in ("auto", "workspace") and ws_ok:
        reductions.append(
            SplitKReduction(
                "workspace", "fp32", workspace=True, workspace_copies=ws_replicas
            )
        )
    can_split = bool(reductions) and -(-k // tile_k) >= 2 and z_cap >= 2
    if not can_split:
        if not allow_unsplit:
            raise ValueError(
                f"wgrad split-K cannot be built for this problem: "
                f"{'no room to split' if reductions else 'no reduction available'} "
                f"(reduction={reduction!r}, group_merge={gm}, k={k}, tile_k={tile_k})"
            )
        return ConvSplitK(
            1, False, 1, "no split possible: one K tile, no z budget or no reduction"
        )
    waves = max(1, block_size // ArchTarget.from_gfx(arch).wave_size)
    # Merged tiles exist only in split-K form in the AOT cache (merging is what
    # makes a depthwise tile wide enough for split-K to pay), so a merged auto
    # request splits whenever it can.
    plan = select_split_k_plan(
        SplitKProblem(
            m=m,
            n=n,
            k=k,
            batch=grid_groups,
            operand_bytes=_operand_bytes(dtype),
            output_bytes=_operand_bytes(dtype),
        ),
        SplitKKernel(tile_m, tile_n, tile_k, waves),
        reductions,
        cost,
        allow_unsplit=allow_unsplit and gm == 1,
    )
    two_stage = plan.reduction is not None and plan.reduction.workspace
    return ConvSplitK(plan.split_k, two_stage, plan.split_k, plan.reason)


def fwd_split_k_problem_reason(p: ConvProblem, arch: str) -> str | None:
    """Why no forward split-K kernel can serve ``p`` on ``arch``, or ``None``.

    Only what the problem and the target decide; the spec-level gates (WMMA,
    pipeline, epilogue) are the builder's
    (:func:`kernels.common.conv_implicit_gemm.fwd_split_k_reason`).
    """
    if base_arch_from_target_id(arch) not in FWD_SPLIT_K_ARCHS:
        return f"split-K forward is built for {', '.join(FWD_SPLIT_K_ARCHS)} only"
    if p.is_pointwise:
        return "split-K forward does not support the pointwise fast path"
    if p.is_3d:
        return "split-K forward does not support 3-D convolution"
    return None


def _fwd_degree_limit(p: ConvProblem, degree: int) -> int:
    """``degree`` held to what one forward split-K launch can encode: the
    gridDim.z budget (``z = group * split_k + slice``) and the i32 workspace.
    The workspace halves with the degree, so the largest degree that fits is
    found by halving (an explicit degree is not held; the launch rejects it)."""
    degree = min(degree, MAX_GRID_DIM_Z // max(1, p.groups))
    while degree > 1 and fwd_split_k_workspace_nbytes(p.M, p.K, degree) > (
        MAX_FWD_WS_BYTES
    ):
        degree //= 2
    return max(1, degree)


def _resolve_fwd(
    p: ConvProblem,
    *,
    arch: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    allow_unsplit: bool,
    override: int | None,
    num_cus: int | None,
) -> ConvSplitK:
    if split_k != -1:
        return ConvSplitK(split_k, split_k > 1, split_k, f"explicit split_k={split_k}")
    why = fwd_split_k_problem_reason(p, arch)
    if override is not None:
        # Held to what the kernel can build, as for dgrad: an unbuildable
        # split becomes 1 rather than a spec the builder then rejects.
        degree = 1 if why is not None else _fwd_degree_limit(p, override)
        if why is not None and override != 1:
            note = f" (no split: {why})"
        elif degree < override:
            note = f" (held to {degree})"
        else:
            note = ""
        return ConvSplitK(
            degree, degree > 1, override, f"override split_k={override}{note}"
        )
    if why is not None:
        decision_degree, reason = 1, f"no split: {why}"
    else:
        d = select_split_k_fwd(
            M=p.M,
            N=p.kpg,
            K=p.K_gemm,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            groups=p.groups,
            arch=arch,
            num_cus=num_cus,
        )
        decision_degree = _fwd_degree_limit(p, d.split_k)
        reason = d.reason
        if decision_degree < d.split_k:
            reason += f" (held to {decision_degree} by gridDim.z / the workspace)"
    if decision_degree == 1 and not allow_unsplit:
        # The caller holds a split-K binary: degree 1 is the separate plain
        # kernel, and the policy says splitting this problem does not pay.
        raise ValueError(f"fwd split-K does not serve this problem: {reason}")
    return ConvSplitK(decision_degree, decision_degree > 1, decision_degree, reason)


def _resolve_dgrad(
    p: ConvProblem,
    *,
    arch: str,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    split_k: int,
    allow_unsplit: bool,
    override: int | None,
    num_cus: int | None,
    num_xcds: int | None,
) -> ConvSplitK:
    if split_k != -1:
        return ConvSplitK(split_k, False, split_k, f"explicit split_k={split_k}")
    wave_size = ArchTarget.from_gfx(arch).wave_size
    if override is not None:
        # Held to what the kernel can build: an unbuildable split becomes 1
        # rather than a spec the builder then rejects, and the 16-bit atomic
        # stays within its accuracy limit (dgrad has no workspace to move to).
        ok, why = dgrad_split_k_available(p, dtype, wave_size=wave_size)
        cap = ATOMIC_16BIT_MAX_SPLIT_K if _is_16bit(dtype) else MAX_GRID_DIM_Z
        degree = min(override, cap) if ok else 1
        if not ok and override != 1:
            note = f" (no split: {why})"
        elif degree < override:
            note = f" (held to {degree})"
        else:
            note = ""
        return ConvSplitK(degree, False, override, f"override split_k={override}{note}")
    m, n, k = _dg_M(p), _dg_N(p), _dg_K(p)
    cost = split_k_cost_model(arch, num_cus, num_xcds)
    if cost is not None:
        ok, why = dgrad_split_k_available(p, dtype, wave_size=wave_size)
        if not ok:
            return ConvSplitK(1, False, 1, f"no split: {why}")
    if cost is None:
        legacy = select_split_k_wgrad(
            wg_M=m,
            wg_N=n,
            wg_K=k,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            arch=arch,
        )
        return ConvSplitK(
            legacy.split_k, False, legacy.split_k, f"CK formula ({legacy.reason})"
        )
    waves = max(1, block_size // wave_size)
    # The group rides grid y, the slice grid z: groups multiply the CTAs but do
    # not share the z budget.
    plan = select_split_k_plan(
        SplitKProblem(
            m=m,
            n=n,
            k=k,
            batch=max(1, p.groups),
            operand_bytes=_operand_bytes(dtype),
            output_bytes=_operand_bytes(dtype),
            z_batch=1,
        ),
        SplitKKernel(tile_m, tile_n, tile_k, waves),
        [_atomic_reduction(dtype)],
        cost,
        allow_unsplit=allow_unsplit,
    )
    return ConvSplitK(plan.split_k, False, plan.split_k, plan.reason)


def resolve_conv_split_k(
    direction: str,
    problem: ConvProblem,
    *,
    arch: str,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    group_merge: int = 1,
    ws_replicas: int = _DEFAULT_WS_REPLICAS,
    split_k: int = -1,
    reduction: str = "auto",
    epilogue: str | None = None,
    allow_unsplit: bool = True,
    override: int | None = None,
    num_cus: int | None = None,
    num_xcds: int | None = None,
) -> ConvSplitK:
    """Resolve the split-K degree and, for wgrad, the reduction path.

    Args:
        direction: ``"fwd"``, ``"wgrad"`` or ``"dgrad"``.
        problem: the convolution.
        arch: target arch or target id; a feature suffix (``gfx942:sramecc+``)
            is stripped, so a decorated id resolves as its bare arch.
        dtype: the output (dW / dX) dtype -- it decides which atomics exist.
        tile_m, tile_n, tile_k: the kernel's block tile.
        block_size: threads per workgroup.
        group_merge: wgrad groups folded into one tile (1 = unmerged).
        ws_replicas: wgrad only. Workspace copies of the two-stage kernel the
            caller builds or holds; sizes the workspace and its setup cost
            (<= 0: the builder default).
        split_k: ``-1`` resolves automatically; any other value is kept
            (wgrad clamps it to the gridDim.z budget).
        reduction: wgrad only. ``"auto"`` chooses between the packed atomic
            and the workspace; ``"atomic"`` / ``"workspace"`` fix the reduction
            the caller's kernel is built with, so only the degree is chosen.
        epilogue: wgrad only. The kernel's epilogue, when known: under
            ``"default"`` (direct store) the packed 16-bit atomic cannot be
            emitted, so ``"auto"`` does not offer it. ``None``: not checked.
        allow_unsplit: whether the auto resolution may return degree 1. False
            for a caller holding a split-K binary (degree 1 is a different
            kernel) or building one that requires a split (two-stage).
        override: forces the degree of an auto request (``split_k=-1``):
            ``None`` resolves automatically, ``1`` forces no split, ``n >= 2``
            forces ``n`` held to what the kernel can build (the gridDim.z
            budget, the reductions available, and
            :data:`ATOMIC_16BIT_MAX_SPLIT_K` for the packed 16-bit atomic).
            Ignored for an explicit ``split_k``. Not part of the spec, so a
            caller that builds or caches a kernel puts the resolved degree on
            the spec. See :func:`parse_split_k_override`.
        num_cus, num_xcds: CU and XCD counts of the device the kernel runs
            on, for a part built differently from its arch's default;
            ``None`` keeps the arch default.

    Forward convolution splits only on :data:`FWD_SPLIT_K_ARCHS` (the
    two-stage kernel); elsewhere, and for the problems that kernel cannot
    take, an auto or override request resolves to 1. ``block_size``,
    ``group_merge``, ``ws_replicas`` and ``num_xcds`` do not affect the
    forward answer.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    if reduction not in _REDUCTIONS:
        raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
    if direction != "wgrad" and reduction != "auto":
        raise ValueError(f"reduction={reduction!r} applies to wgrad only")
    if direction != "wgrad":
        epilogue = None
    if override is not None and override < 1:
        raise ValueError(f"override must be None or >= 1, got {override}")
    return _resolve_cached(
        direction,
        problem,
        # Every per-arch table and the arch catalog are keyed by the bare
        # arch; a decorated id also shares the bare arch's cache entry.
        arch=base_arch_from_target_id(arch),
        dtype=dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        block_size=block_size,
        group_merge=group_merge,
        ws_replicas=ws_replicas,
        split_k=split_k,
        reduction=reduction,
        epilogue=epilogue,
        allow_unsplit=allow_unsplit,
        override=override,
        num_cus=num_cus,
        num_xcds=num_xcds,
    )


# A benchmark resolves the same (problem, kernel) once per binary and degree
# list, and planning scores up to a few hundred degrees; the answer is a pure
# function of the arguments.
@functools.lru_cache(maxsize=4096)
def _resolve_cached(
    direction: str,
    problem: ConvProblem,
    *,
    arch: str,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    block_size: int,
    group_merge: int,
    ws_replicas: int,
    split_k: int,
    reduction: str,
    epilogue: str | None,
    allow_unsplit: bool,
    override: int | None,
    num_cus: int | None,
    num_xcds: int | None,
) -> ConvSplitK:
    common = {
        "arch": arch,
        "dtype": dtype,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "block_size": block_size,
        "split_k": split_k,
        "allow_unsplit": allow_unsplit,
        "override": override,
        "num_cus": num_cus,
        "num_xcds": num_xcds,
    }
    if direction == "fwd":
        return _resolve_fwd(
            problem,
            arch=arch,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            split_k=split_k,
            allow_unsplit=allow_unsplit,
            override=override,
            num_cus=num_cus,
        )
    if direction == "dgrad":
        return _resolve_dgrad(problem, **common)
    return _resolve_wgrad(
        problem,
        group_merge=group_merge,
        ws_replicas=ws_replicas,
        reduction=reduction,
        epilogue=epilogue,
        **common,
    )
