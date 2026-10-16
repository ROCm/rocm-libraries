# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 direct-conv dgrad selection policy (single-kernel AOT variants).

Pure functions of a :class:`~kernels.common.conv_direct_grouped.DirectConvProblem`
and the arch: they decide which single-step direct dgrad kernel of the AOT
cache serves a problem and with which knobs. ``dispatch.grouped_convolution``
wraps them in dispatch candidates.

Every pick is a ``(variant, knobs)`` pair that names one cached binary:
``variant`` is the cache variant (``KernelIdentity.algorithm``), the
capabilities are the problem's filter, padding, stride and per-group channel
counts, and ``knobs`` is the identity's knob dict. The knob values are
limited to the image the cache builds (``benchmarks.common.direct_kernel_sweep``;
the tables below restate the parts the policy picks from, and
``tests/dispatch/test_grouped_conv_dgrad_dispatch_aot.py`` checks every pick
against the enumeration). Dispatch never picks a kernel the cache does not
hold.

Variants:

* ``direct_grouped_dgrad_4c``: cpg == kpg == 4, 1x1 / 3x3 "same" padding,
  groups a multiple of 16: the batched 4x4x4 MFMA kernel (forward ABI over the
  transposed problem).
* ``direct_grouped_dgrad_fused``: stride 1, "same"-padded square filters up to
  7x7 and per-group channel counts in 4-channel steps up to 32 (cpg != kpg
  too, groups == 1 too): the generic direct-MFMA kernel reading the original
  weights itself (forward ABI over the transposed problem).
* ``direct_depthwise_dgrad_win``: depthwise (cpg == kpg == 1) stride 1, the
  windowed VALU kernel or its Toeplitz MFMA form (dgrad ABI).
* ``direct_depthwise_dgrad``: depthwise at any cached stride, the row-streaming
  kernel (dgrad ABI); serves stride 2 and whatever the windowed kernel
  declines.
"""

from __future__ import annotations

from kernels.common.conv_direct_grouped import (
    DW_DGRAD_MFMA_ARCHES,
    DW_DGRAD_MFMA_CH,
    DW_DGRAD_MFMA_W_FOLDS,
    DW_DGRAD_WIN_DOT2_ARCHES,
    PRELOAD_WEIGHT_VGPR_BUDGET,
    DirectConvProblem,
    dgrad_fused_block_h,
    direct_conv_grid,
    direct_conv_lds_bytes,
    direct_launch_geometry,
    is_valid_direct_dgrad_spec,
    make_direct_dgrad_spec,
    preload_weight_vgprs,
)
from rocke.core.arch import ArchTarget

VARIANT_4C = "direct_grouped_dgrad_4c"
VARIANT_FUSED = "direct_grouped_dgrad_fused"
VARIANT_DW_WIN = "direct_depthwise_dgrad_win"
VARIANT_DW_STREAM = "direct_depthwise_dgrad"

# Every decline that is a measured-performance decision (not a capability)
# starts with this, so a caller can tell the two apart.
POLICY_PREFIX = "outside the measured direct win region:"

_I32_MAX = (1 << 31) - 1
_MAX_GRID_DIM = 65535  # y and z


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _valid(variant: str, p: DirectConvProblem, knobs: dict, arch: str) -> bool:
    spec = make_direct_dgrad_spec(variant, p, knobs)
    return is_valid_direct_dgrad_spec(variant, spec, arch)[0]


def _has_transpose_reads(arch: str) -> bool:
    try:
        return bool(ArchTarget.from_gfx(arch).memory.has_ds_read_tr)
    except KeyError:
        return False


def launch_errors(variant: str, p: DirectConvProblem, knobs: dict) -> list[str]:
    """Launch limits of a pick on ``p``: the i32 byte sizes of the direct ABI
    and the 16-bit y/z grid extents."""
    errors = []
    spec = make_direct_dgrad_spec(variant, p, knobs)
    grid, _block = direct_launch_geometry(spec)
    if max(grid[1:]) > _MAX_GRID_DIM:
        errors.append(f"grid {grid} exceeds the y/z launch limit {_MAX_GRID_DIM}")
    elem = 2
    dy = p.N * p.Ho * p.Wo * p.total_k * elem
    dx = p.N * p.H * p.W * p.total_c * elem
    w = p.total_k * p.KH * p.KW * p.cpg * elem
    big = [n for n, b in (("dY", dy), ("dX", dx), ("W", w)) if b > _I32_MAX]
    if big:
        errors.append(f"tensors {big} exceed the i32 byte sizes of the direct ABI")
    return errors


# ---------------------------------------------------------------------------
# Grouped / small-channel dgrad: 4c and fused generic kernels
# ---------------------------------------------------------------------------
#
# Eligibility is the region where the direct kernels were measured faster
# than the implicit-GEMM candidate: stride 1, "same"-padded square filters up
# to 7x7 and channel counts in 4-channel steps up to 32 on both sides
# (:func:`generic_shape_errors`), minus the corners where they were slower
# (:func:`generic_policy_errors`). The thresholds were fitted on random draws
# over that region (groups 1..64, images 3..64, batch sizes spanning a wide
# range of implicit-GEMM grid sizes) and re-checked on the AOT binaries; the
# measurements live outside the tree.

GENERIC_MAX_CHANNELS = 32
GENERIC_FILTERS = (1, 3, 5, 7)
# Narrow reduction: cpg > kpg with kpg <= 8.
_NARROW_KPG = 8
# Wide output: more than 20 dX channels per group (two 16-wide output tiles).
_WIDE_CPG = 20
_MIN_STRIP_FILL = 0.6
_TINY_W = 3
_TINY_HW = 16
_WIDE_TINY_W = 5
_PW_MAX_CHANNELS = 32
_PW_MIN_TILES = 300
_PW_MIN_W = 7
_SMALL_TILES = 768
_SMALL_NARROW_WIDE_MIN_TILES = 500
_SMALL_NARROW_MIN_TILES = 400
_SMALL_4CH_MIN_TILES = 275
# The implicit-GEMM M tile the size floors count workgroups in.
_IGEMM_TILE_M = 64

# 4c row (cpg == kpg == 4). The direct-load kernel needs a grid of at least
# _4C_MIN_GRID workgroups (counted with block_q 4); the row-staged kernel
# (stage_rows) is taken wherever it was not measured slower (see
# _4c_takes_stage_rows).
_4C_BLOCK_Q = 4
_4C_BLOCK_GROUPS = 16
# The row-staged kernel runs one wave per 4 columns; with 3x3 filters
# 8-column tiles (two waves sharing the staged row and its halo columns)
# were the fastest cached tile on the 4c problems measured, small grids
# included; 1x1 filters (no halo) keep 4-column tiles.
_4C_STAGED_BLOCK_Q_3X3 = 8
_4C_MIN_GRID = 300
_4C_STAGED_SHORT_MAX_H = 8
_4C_STAGED_SHORT_MIN_GRID = 96
_4C_STAGED_SHORT_MIN_W = 4
_4C_STAGED_SHORT_MIN_H = 5
_4C_STAGED_MIN_H_1X1 = 8
_4C_STAGED_MIN_H_3X3 = 4

# Fused generic kernel: spatial policy. Images of at most _TILE_H rows are
# streamed whole; 5x5 / 7x7 filters take _SMALL_TILE_H-row tiles; short
# images (<= _SHORT_H rows) take the cheapest of whole / 8 / 4 rows by the
# row-chain x launch-rounds model; taller ones the largest tile whose grid
# reaches _TARGET_WAVES waves. A whole-image pick runs on the smallest cached
# tile at least as tall (DGRAD_FUSED_BLOCK_H), whose row guard skips the
# rows past the image.
_TILE_H = 8
_SMALL_TILE_H = 4
_MAX_UNTILED_H = 64
_SHORT_H = 16
_TARGET_WAVES = 3072
_SHORT_WAVE_SLOTS = 1024
_BLOCK_Q = 16
_WIDE_BLOCK_Q = 32
_WIDE_MIN_WAVES = 768
# Cached knob image of the fused kernel (direct_kernel_sweep): whole-image
# tiles that leave the workgroup once the image is done, the waves-per-EU
# hint and the row-stream stack.
_FUSED_EXIT_BLOCK_H = (16, 32)
_FUSED_WPE = 4
_FUSED_WPE_MAX_VGPRS = 36
_STREAM_PREFETCH_ROWS = 2
_STREAM_LDS_PAD = 8
_STREAM_UNTILED_MAX_WO = 64
_STREAM_TILED_1X1_MIN_CPG = 12
_NUM_CUS = 256
_MAX_WAVES_PER_CU = 32


def igemm_tiles(p: DirectConvProblem) -> int:
    """Workgroups a 64-row implicit-GEMM dgrad tile launches for ``p`` (one N
    tile per group): the work measure of the size floors."""
    return _cdiv(p.N * p.H * p.W, _IGEMM_TILE_M) * p.groups


def generic_shape_errors(p: DirectConvProblem) -> list[str]:
    """Capability reasons the 4c / fused kernels cannot run ``p``."""
    errors = []
    if p.stride != 1:
        errors.append(f"stride must be 1 (got {p.stride})")
    if p.dil_h != 1 or p.dil_w != 1:
        errors.append("dilation must be 1")
    if p.KH != p.KW:
        errors.append(f"square filters only (got {p.KH}x{p.KW})")
    if 2 * p.PAD != p.KH - 1:
        errors.append(f"'same' padding only (2*pad == Y-1; got pad={p.PAD}, Y={p.KH})")
    if p.KH not in GENERIC_FILTERS:
        errors.append(f"filters {GENERIC_FILTERS} only (got {p.KH}x{p.KW})")
    if p.cpg % 4 or p.kpg % 4:
        errors.append(f"cpg and kpg must be multiples of 4 (got {p.cpg}/{p.kpg})")
    if p.cpg > GENERIC_MAX_CHANNELS or p.kpg > GENERIC_MAX_CHANNELS:
        errors.append(
            f"cpg/kpg above {GENERIC_MAX_CHANNELS} (got {p.cpg}/{p.kpg}); "
            "implicit GEMM serves those"
        )
    if not errors:
        regs = p.KH * p.KW * _cdiv(p.kpg, 16) * _cdiv(p.cpg, 16) * 2
        if regs > PRELOAD_WEIGHT_VGPR_BUDGET:
            errors.append(
                f"fused weights need {regs} VGPRs (> {PRELOAD_WEIGHT_VGPR_BUDGET})"
            )
    return errors


def _4c_takes_stage_rows(p: DirectConvProblem, grid: int) -> bool:
    """Whether the 4c row takes the row-staged kernel: every grid at or above
    the 4c floor except 1x1 filters on images shorter than
    _4C_STAGED_MIN_H_1X1 and 3x3 filters on images shorter than
    _4C_STAGED_MIN_H_3X3; and 3x3 filters below the floor on images of
    _4C_STAGED_SHORT_MIN_H.._4C_STAGED_SHORT_MAX_H rows and at least
    _4C_STAGED_SHORT_MIN_W columns with a grid of at least
    _4C_STAGED_SHORT_MIN_GRID."""
    if p.KH == 1:
        return grid >= _4C_MIN_GRID and p.H >= _4C_STAGED_MIN_H_1X1
    if grid >= _4C_MIN_GRID:
        return p.H >= _4C_STAGED_MIN_H_3X3
    return (
        _4C_STAGED_SHORT_MIN_H <= p.H <= _4C_STAGED_SHORT_MAX_H
        and p.W >= _4C_STAGED_SHORT_MIN_W
        and grid >= _4C_STAGED_SHORT_MIN_GRID
    )


def select_4c(p: DirectConvProblem, arch: str) -> dict | None:
    """The 4c row's knobs for ``p``, or None (the fused kernel takes it).

    One wave carries 16 groups at full 4x4x4 MFMA width and reads the weights
    in its prologue (LDS-staged where the target has transpose reads). The
    direct-load kernel needs a grid of at least _4C_MIN_GRID workgroups; the
    row-staged kernel see :func:`_4c_takes_stage_rows`.
    """
    if not (p.cpg == 4 and p.kpg == 4 and p.KH == p.KW and p.KH in (1, 3)):
        return None
    if 2 * p.PAD != p.KH - 1 or p.groups % _4C_BLOCK_GROUPS:
        return None
    grid = _cdiv(p.W, _4C_BLOCK_Q) * (p.groups // _4C_BLOCK_GROUPS) * p.N
    lds = _has_transpose_reads(arch)
    if lds and _4c_takes_stage_rows(p, grid):
        staged = {
            "block_q": _4C_STAGED_BLOCK_Q_3X3 if p.KH == 3 else _4C_BLOCK_Q,
            "block_groups": _4C_BLOCK_GROUPS,
            "dgrad_weights_lds": True,
            "stage_rows": True,
        }
        if _valid(VARIANT_4C, p, staged, arch):
            return staged
    if grid < _4C_MIN_GRID:
        return None
    knobs = {
        "block_q": _4C_BLOCK_Q,
        "block_groups": _4C_BLOCK_GROUPS,
        "dgrad_weights_lds": lds,
        "stage_rows": False,
    }
    return knobs if _valid(VARIANT_4C, p, knobs, arch) else None


def _rule_block_groups(cpg: int, kpg: int) -> int:
    """Per-channel row: the wider channel count times block_groups stays near
    a 32..64-channel slice."""
    chans = max(cpg, kpg)
    return 1 if chans >= 32 else 2 if chans >= 16 else 4


def _fused_block_h(p: DirectConvProblem, rule_bg: int) -> int:
    """Rows per workgroup, 0 for the whole image (see the notes above)."""
    if p.H <= _TILE_H:
        return 0
    if p.KH >= 5:
        return _SMALL_TILE_H
    wave_columns = _cdiv(p.Wo, _BLOCK_Q) * p.groups * p.N
    if p.H <= _SHORT_H:
        slots = _SHORT_WAVE_SLOTS * rule_bg
        best_bh, best_cost = 0, None
        for bh in (0, _TILE_H, _SMALL_TILE_H):
            h_tiles = _cdiv(p.H, bh) if bh else 1
            cost = ((bh or p.H) + p.KH - 1) * max(slots, wave_columns * h_tiles)
            if best_cost is None or cost < best_cost:
                best_bh, best_cost = bh, cost
        return best_bh
    tiles = (_TILE_H,) if p.H > _MAX_UNTILED_H else (0, _TILE_H)
    for bh in tiles:
        if wave_columns * (_cdiv(p.H, bh) if bh else 1) >= _TARGET_WAVES:
            return bh
    return _SMALL_TILE_H


def _fused_block_q(p: DirectConvProblem, block_h: int) -> int:
    """16 columns per workgroup, or 32 on H-tiled images where the wider strip
    adds no padding, the channel work per wave is large enough and the grid
    keeps _WIDE_MIN_WAVES waves."""
    if block_h == 0:
        return _BLOCK_Q
    if _cdiv(p.Wo, _WIDE_BLOCK_Q) * _WIDE_BLOCK_Q > _cdiv(p.Wo, _BLOCK_Q) * _BLOCK_Q:
        return _BLOCK_Q
    chans = max(p.cpg, p.kpg)
    if not (chans >= 16 or (p.KH >= 5 and chans >= 8)):
        return _BLOCK_Q
    waves = _cdiv(p.Wo, _WIDE_BLOCK_Q) * p.groups * p.N * _cdiv(p.H, block_h)
    return _WIDE_BLOCK_Q if waves >= _WIDE_MIN_WAVES else _BLOCK_Q


def _lds_rounds(p: DirectConvProblem, knobs: dict, arch: str) -> int:
    """Launch rounds of the fused kernel when LDS (or the per-CU wave cap)
    limits occupancy."""
    spec = make_direct_dgrad_spec(VARIANT_FUSED, p, knobs)
    target = ArchTarget.from_gfx(arch)
    per_cu = min(
        target.lds_capacity_bytes // max(direct_conv_lds_bytes(spec), 1),
        _MAX_WAVES_PER_CU // (spec.threads_per_block // target.wave_size),
    )
    gx, gy, gz = direct_conv_grid(spec)
    return _cdiv(gx * gy * gz, max(per_cu, 1) * _NUM_CUS)


def _fused_stream_knobs(
    p: DirectConvProblem, knobs: dict, untiled: bool, arch: str
) -> dict:
    """Add the row-stream stack to fused ``knobs`` where it was measured to win.

    Square channel groups on 16-column strips, untiled images up to
    _STREAM_UNTILED_MAX_WO columns, 1x1 filters on tiled images from
    _STREAM_TILED_1X1_MIN_CPG channels; not where ``waves_m = 2`` is valid
    (that stack is not cached); only with the XCD row-band order valid. Then
    the first of (pad, staged output), (pad, unstaged), (no pad, unstaged)
    that adds no launch rounds (a nearly empty extra round costs more than
    the stores and banks gain).
    """
    if p.cpg != p.kpg or knobs["block_q"] != _BLOCK_Q:
        return knobs
    if untiled:
        if p.Wo > _STREAM_UNTILED_MAX_WO:
            return knobs
    elif p.KH == 1 and p.cpg < _STREAM_TILED_1X1_MIN_CPG:
        return knobs
    base = {**knobs, "prefetch_rows": _STREAM_PREFETCH_ROWS, "lds_only_sync": True}
    if not _valid(VARIANT_FUSED, p, base, arch):
        return knobs
    if _valid(VARIANT_FUSED, p, {**base, "waves_m": 2}, arch):
        return knobs
    floor = _lds_rounds(p, base, arch)
    stack = {**base, "xcd_tiles": True, "lds_pad": 0, "stage_out": False}
    if not _valid(VARIANT_FUSED, p, stack, arch):
        return knobs
    pad = _STREAM_LDS_PAD if knobs["block_groups"] * p.kpg * 2 % 32 == 0 else 0
    for lds_pad, stage_out in ((pad, True), (pad, False)):
        cand = {**stack, "lds_pad": lds_pad, "stage_out": stage_out}
        if (
            cand != stack
            and _valid(VARIANT_FUSED, p, cand, arch)
            and _lds_rounds(p, cand, arch) <= floor
        ):
            return cand
    return stack


def select_fused(p: DirectConvProblem, arch: str) -> dict | None:
    """The fused generic kernel's knobs for ``p``, or None where no cached
    form is valid.

    Per-channel ``block_groups`` (halved for 5x5 / 7x7, then halved until it
    divides the groups), the spatial policy (:func:`_fused_block_h`,
    :func:`_fused_block_q`), ``fold_k32`` where kpg allows it and the atom
    is valid, LDS-staged weights where valid, the waves-per-EU hint on small
    16-wide-atom weight footprints, and the row-stream stack
    (:func:`_fused_stream_knobs`).
    """
    rule_bg = _rule_block_groups(p.cpg, p.kpg)
    bg = max(1, rule_bg // 2) if p.KH >= 5 else rule_bg
    while p.groups % bg:
        bg //= 2
    bh = _fused_block_h(p, rule_bg)
    knobs = {
        "block_q": _fused_block_q(p, bh),
        "block_groups": bg,
        "block_h": bh or dgrad_fused_block_h(p.H),
        "fold_k32": False,
        "dgrad_weights_lds": False,
        "waves_per_eu": 0,
    }
    if knobs["block_h"] in _FUSED_EXIT_BLOCK_H:
        knobs["row_exit"] = True
    if p.kpg % 32 == 0 and _valid(VARIANT_FUSED, p, {**knobs, "fold_k32": True}, arch):
        knobs["fold_k32"] = True
    if _has_transpose_reads(arch) and _valid(
        VARIANT_FUSED, p, {**knobs, "dgrad_weights_lds": True}, arch
    ):
        knobs["dgrad_weights_lds"] = True
    if not _valid(VARIANT_FUSED, p, knobs, arch):
        return None
    if not knobs["fold_k32"]:
        main = make_direct_dgrad_spec(VARIANT_FUSED, p, knobs)
        if preload_weight_vgprs(main) <= _FUSED_WPE_MAX_VGPRS:
            knobs["waves_per_eu"] = _FUSED_WPE
    return _fused_stream_knobs(p, knobs, untiled=bh == 0, arch=arch)


def generic_policy_errors(p: DirectConvProblem, variant: str) -> list[str]:
    """Measured-loss corners of the fused kernel against implicit GEMM.

    The 4c row is only picked above its own floors. ``fill`` is measured
    against 16-column strips whatever the picked ``block_q``.
    """
    if variant == VARIANT_4C:
        return []
    cpg, kpg, Y, H, W = p.cpg, p.kpg, p.KH, p.H, p.W
    tiles = igemm_tiles(p)
    fill = W / (_cdiv(W, _BLOCK_Q) * _BLOCK_Q)
    narrow = cpg > kpg and kpg <= _NARROW_KPG
    wide = cpg > _WIDE_CPG
    why = ""
    if W <= _TINY_W or H * W <= _TINY_HW:
        why = f"tiny image {H}x{W}: the 16-wide output strips stay mostly empty"
    elif wide and W <= _WIDE_TINY_W:
        why = f"cpg {cpg} on W={W}: two output tiles per wave on a mostly empty strip"
    elif wide and min(cpg, kpg) <= 10 and fill < _MIN_STRIP_FILL:
        why = (
            f"cpg {cpg} with kpg {kpg}: W={W} fills only {fill:.2f} of "
            f"{_BLOCK_Q}-wide column strips"
        )
    elif (
        Y == 1
        and kpg >= 16
        and (
            max(cpg, kpg) >= _PW_MAX_CHANNELS or tiles < _PW_MIN_TILES or W < _PW_MIN_W
        )
    ):
        why = (
            f"1x1 with a wide reduction (kpg {kpg}, cpg {cpg}, W={W}, "
            f"{tiles} igemm workgroups) is implicit GEMM's best case"
        )
    elif Y > 1 and tiles < _SMALL_TILES:
        if narrow and wide and tiles < _SMALL_NARROW_WIDE_MIN_TILES:
            floor = _SMALL_NARROW_WIDE_MIN_TILES
        elif narrow and not wide and max(cpg, kpg) <= 10:
            floor = _SMALL_NARROW_MIN_TILES
        elif max(cpg, kpg) <= 6:
            floor = _SMALL_4CH_MIN_TILES
        else:
            floor = 0
        if tiles < floor:
            why = f"small problem: {tiles} igemm workgroups < {floor} for this class"
    return [f"{POLICY_PREFIX} {why}"] if why else []


def select_generic(p: DirectConvProblem, arch: str) -> tuple[str | None, dict, list]:
    """``(variant, knobs, [])`` of the grouped direct dgrad pick for ``p``, or
    ``(None, {}, reasons)`` where it declines (implicit GEMM takes it)."""
    errors = generic_shape_errors(p)
    if errors:
        return None, {}, errors
    knobs = select_4c(p, arch)
    variant = VARIANT_4C
    if knobs is None:
        variant = VARIANT_FUSED
        knobs = select_fused(p, arch)
        if knobs is None:
            return None, {}, ["no cached fused form is valid for this problem"]
    errors = generic_policy_errors(p, variant) + launch_errors(variant, p, knobs)
    if errors:
        return None, {}, errors
    return variant, knobs, []


# ---------------------------------------------------------------------------
# Depthwise dgrad: windowed (stride 1) and row-streaming kernels
# ---------------------------------------------------------------------------

# Cached knob image of the windowed kernel (direct_kernel_sweep DW_WIN_*):
# (block_w, block_h) tiles, waves of single channels and of channel pairs,
# the unroll budget, and the Toeplitz MFMA form's filters, waves and tiles.
DW_WIN_TILES = (
    (7, 7),
    (8, 8),
    (12, 12),
    (14, 14),
    (16, 16),
    (7, 14),
    (8, 14),
    (8, 16),
    (7, 28),
)
DW_WIN_WAVES = (1, 4)
DW_WIN_PAIR_WAVES = 2
DW_WIN_UNROLL_BUDGET = 1 << 14
DW_WIN_FILTERS = (3, 5, 7, 9, 11)
DW_WIN_MFMA_WAVES = (4, 8)
DW_WIN_MFMA_BLOCK_H = (4, 7, 8, 12, 14, 16)
DW_WIN_MFMA_PREFETCH = 2
# VALU form: the grid should hold this many waves (8 per CU); below it
# smaller tiles win back the idle wave slots.
_DW_TARGET_WAVES = 2048
# Toeplitz MFMA form admission box: the requests on which it was measured to
# beat the VALU form with warm and with cold caches. Square 7x7 filters with
# 'same' padding, C a multiple of 64 up to 2048 channels, up to 256 images,
# dY tensors up to 512 MiB, images of 7 to 16 rows and 7-8, 13-16 or 19-112
# columns. Everything else keeps the VALU form.
_DW_MFMA_FILTER = 7
_DW_MFMA_CH_MULTIPLE = 64
_DW_MFMA_MAX_C = 2048
_DW_MFMA_MAX_N = 256
_DW_MFMA_H_RANGE = (7, 16)
_DW_MFMA_W_BANDS = ((7, 8), (13, 16), (19, 112))
_DW_MFMA_MAX_DY_BYTES = 512 << 20
_DW_MFMA_WAVES = 8
# Register-file model of the MFMA block-size choice (gfx950): VGPRs per lane
# of a SIMD, the allocation granule and the VGPRs beyond the fragments.
_DW_SIMD_VGPRS = 512
_DW_VGPR_GRANULE = 8
_DW_MFMA_VGPR_OVERHEAD = 16
_DW_SIMDS_PER_CU = 4
_DW_MFMA_TARGET_BLOCKS = 256
_DW_MFMA_WHOLE_H = 16
_DW_MFMA_ROW_STEPS = (14, 7, 4)
_DW_MFMA_SHORT_ROWS = 4
_DW_MFMA_SHORT_ROWS_BLOCKS = 160
# Row-streaming kernel's cached knobs (direct_kernel_sweep DW_BLOCK_*).
DW_STREAM_BLOCK_W = (4, 7, 8, 16)
DW_STREAM_WAVES = (1, 2, 4)


def depthwise_errors(p: DirectConvProblem) -> list[str]:
    """Capability reasons no direct depthwise dgrad kernel serves ``p``."""
    errors = []
    if p.cpg != 1 or p.kpg != 1:
        errors.append(f"requires depthwise cpg == kpg == 1 (got {p.cpg}, {p.kpg})")
    if p.dil_h != 1 or p.dil_w != 1:
        errors.append("requires dilation 1")
    if p.KH != p.KW or p.KH not in DW_WIN_FILTERS:
        errors.append(f"square filters {DW_WIN_FILTERS} only (got {p.KH}x{p.KW})")
    if not 0 <= p.PAD < p.KH:
        errors.append(f"padding must be in [0, KH-1] (got {p.PAD})")
    return errors


def _dw_mfma_admits(p: DirectConvProblem, arch: str) -> bool:
    """Whether ``p`` is inside the MFMA form's admission box (see above)."""
    if arch not in DW_DGRAD_MFMA_ARCHES or p.stride != 1:
        return False
    if p.KH != _DW_MFMA_FILTER or p.KW != _DW_MFMA_FILTER or p.PAD != p.KH // 2:
        return False
    C = p.groups
    if C % _DW_MFMA_CH_MULTIPLE or C > _DW_MFMA_MAX_C or p.N > _DW_MFMA_MAX_N:
        return False
    if not _DW_MFMA_H_RANGE[0] <= p.H <= _DW_MFMA_H_RANGE[1]:
        return False
    if not any(lo <= p.W <= hi for lo, hi in _DW_MFMA_W_BANDS):
        return False
    return p.N * p.H * p.W * C * 2 <= _DW_MFMA_MAX_DY_BYTES


def _dw_mfma_fold(N: int, W: int, KW: int) -> int:
    """Images per MFMA tile: 4 up to 8 columns, 2 up to 16, 1 up to 32; wider
    images take the fold with the least padded MFMA columns plus window
    reads (ties keep the smaller fold)."""
    if W <= 32:
        return 4 if W <= 8 else 2 if W <= 16 else 1

    def cost(fold: int) -> int:
        tile_w = 32 // fold
        col_tiles = _cdiv(W, tile_w)
        return _cdiv(N, fold) * col_tiles * 32 + N * col_tiles * (tile_w + KW - 1)

    return min(DW_DGRAD_MFMA_W_FOLDS, key=lambda f: (cost(f), f))


def _dw_mfma_block_h(rows: int) -> int:
    """The smallest cached MFMA tile height covering ``rows`` (the tallest
    when none does)."""
    for bh in DW_WIN_MFMA_BLOCK_H:
        if bh >= rows:
            return bh
    return DW_WIN_MFMA_BLOCK_H[-1]


def _dw_mfma_blocks_per_cu(p: DirectConvProblem, knobs: dict) -> int:
    spec = make_direct_dgrad_spec(VARIANT_DW_WIN, p, knobs)
    g = _DW_VGPR_GRANULE
    vgprs = _cdiv(spec.mfma_frag_vgprs() + _DW_MFMA_VGPR_OVERHEAD, g)
    waves_per_simd = _DW_SIMD_VGPRS // (vgprs * g)
    return waves_per_simd // _cdiv(spec.block_waves, _DW_SIMDS_PER_CU)


def _dw_mfma_knobs(p: DirectConvProblem, arch: str) -> dict | None:
    """The MFMA form's knobs for ``p`` (inside the admission box), or None.

    ``w_fold`` from :func:`_dw_mfma_fold`; 8-wave blocks, 4 where the VGPR
    estimate fits one 8-wave block per CU and the grid ends in a partial
    round; images up to 16 rows whole, taller ones in chunks of about 14
    rows; while the grid has fewer than _DW_MFMA_TARGET_BLOCKS workgroups the
    block drops to 4 waves, then to chunks of about 7, then (small grids
    only) about 4 rows. Row counts run on the smallest cached tile height
    covering them.
    """
    if p.groups % DW_DGRAD_MFMA_CH:
        return None
    fold = _dw_mfma_fold(p.N, p.W, p.KW)
    waves = _DW_MFMA_WAVES

    def make(waves: int, bh: int) -> dict:
        return {
            "mfma": True,
            "w_fold": fold,
            "block_waves": waves,
            "block_h": bh,
            "prefetch_rows": DW_WIN_MFMA_PREFETCH,
        }

    steps = [(p.H, p.H)] if p.H <= _DW_MFMA_WHOLE_H else []
    steps += [(_cdiv(p.H, _cdiv(p.H, r)), r) for r in _DW_MFMA_ROW_STEPS if r < p.H]
    opts: list[tuple[int, int]] = []
    for rows, step in steps:
        bh = _dw_mfma_block_h(rows)
        if bh not in [o[0] for o in opts] and _valid(
            VARIANT_DW_WIN, p, make(waves, bh), arch
        ):
            opts.append((bh, step))
    if not opts:
        return None

    def blocks(knobs: dict) -> int:
        spec = make_direct_dgrad_spec(VARIANT_DW_WIN, p, knobs)
        gx, gy, gz = spec.grid()
        return gx * gy * gz

    first = make(waves, opts[0][0])
    if _dw_mfma_blocks_per_cu(p, first) < 2 and blocks(first) % _DW_MFMA_TARGET_BLOCKS:
        waves = _DW_MFMA_WAVES // 2
    i = 0
    while True:
        n_blocks = blocks(make(waves, opts[i][0]))
        if n_blocks >= _DW_MFMA_TARGET_BLOCKS:
            break
        if waves > _DW_MFMA_WAVES // 2:
            waves = _DW_MFMA_WAVES // 2
        elif i + 1 < len(opts) and (
            opts[i + 1][1] > _DW_MFMA_SHORT_ROWS
            or n_blocks < _DW_MFMA_SHORT_ROWS_BLOCKS
        ):
            i += 1
        else:
            break
    knobs = make(waves, opts[i][0])
    if not _valid(VARIANT_DW_WIN, p, knobs, arch) or launch_errors(
        VARIANT_DW_WIN, p, knobs
    ):
        return None
    return knobs


def _dw_valu_knobs(p: DirectConvProblem, arch: str) -> dict | None:
    """The VALU form's knobs for ``p``: the cached tile with the least work.

    ``dot2`` pairs filter taps where it pays (KW >= 5, fdot2 arch); packed
    channel pairs for 3x3 filters with enough channels to fill the wider
    block; blocks of up to 256 channels. The tile is the cached
    ``(block_w, block_h)`` minimising the padded columns times the dY rows
    read per output row band (its halo included), scaled up when the grid
    holds fewer than _DW_TARGET_WAVES waves.
    """
    C = p.groups
    dot2 = p.KW >= 5 and arch in DW_DGRAD_WIN_DOT2_ARCHES
    cpl = 2 if (p.KW == 3 and C % 2 == 0 and C >= 128) else 1
    if cpl == 2:
        waves = DW_WIN_PAIR_WAVES
    else:
        waves = DW_WIN_WAVES[-1] if _cdiv(C, 64) >= DW_WIN_WAVES[-1] else 1
    best = None
    for bw, bh in DW_WIN_TILES:
        if bh * p.KH * p.KW * bw * cpl > DW_WIN_UNROLL_BUDGET:
            continue
        knobs = {
            "block_w": bw,
            "block_h": bh,
            "block_waves": waves,
            "ch_per_lane": cpl,
            "dot2": dot2,
        }
        if not _valid(VARIANT_DW_WIN, p, knobs, arch):
            continue
        if launch_errors(VARIANT_DW_WIN, p, knobs):
            continue
        h_tiles = _cdiv(p.H, bh)
        grid_waves = _cdiv(p.W, bw) * _cdiv(C, 64 * waves * cpl) * p.N * h_tiles * waves
        work = _cdiv(p.W, bw) * bw * h_tiles * (bh + p.KH - 1)
        cost = work * max(1.0, _DW_TARGET_WAVES / grid_waves)
        key = (cost, -bw * bh)
        if best is None or key < best[0]:
            best = (key, knobs)
    return best[1] if best else None


def select_depthwise_win(
    p: DirectConvProblem, arch: str
) -> tuple[dict | None, list[str]]:
    """``(knobs, [])`` of the windowed kernel for ``p``, or ``(None, reasons)``."""
    errors = depthwise_errors(p)
    if p.stride != 1:
        errors.append(f"the windowed kernel is stride 1 only (got {p.stride})")
    if errors:
        return None, errors
    if _dw_mfma_admits(p, arch):
        knobs = _dw_mfma_knobs(p, arch)
        if knobs is not None:
            return knobs, []
    knobs = _dw_valu_knobs(p, arch)
    if knobs is None:
        return None, ["no cached windowed tile is valid for this problem"]
    return knobs, []


def select_depthwise_stream(
    p: DirectConvProblem, arch: str
) -> tuple[dict | None, list[str]]:
    """``(knobs, [])`` of the row-streaming kernel for ``p``, or
    ``(None, reasons)``: the cached ``block_w`` with the fewest padded dX
    columns (the wider on a tie) and blocks of up to 256 channels."""
    errors = depthwise_errors(p)
    if p.stride not in (1, 2):
        errors.append(
            f"the streaming kernel is cached for stride 1 and 2 (got {p.stride})"
        )
    if errors:
        return None, errors
    bw = min(DW_STREAM_BLOCK_W, key=lambda w: (_cdiv(p.W, w) * w, -w))
    need = _cdiv(p.groups, 64)
    waves = max(w for w in DW_STREAM_WAVES if w <= max(1, min(need, 4)))
    knobs = {"block_w": bw, "block_waves": waves}
    if not _valid(VARIANT_DW_STREAM, p, knobs, arch):
        return None, ["the cached streaming tile is not valid for this problem"]
    errors = launch_errors(VARIANT_DW_STREAM, p, knobs)
    return (None, errors) if errors else (knobs, [])
