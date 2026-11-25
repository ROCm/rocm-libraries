# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Depthwise forward convolution as a Toeplitz WMMA product (gfx1250).

The kernel is the wave32 ``fwd`` form of the windowed depthwise dgrad's
Toeplitz builder (``_build_dw_dgrad_win_toeplitz`` in
:mod:`conv_direct_grouped`): a stride-1 depthwise dgrad is a stride-1
depthwise convolution with a flipped filter, so the forward is that kernel
reading the filter rotated, over the transposed problem
(:func:`dw_fwd_wmma_win_spec`). This module owns the family's spec, name and
limits; the row pipeline, lane layout, LDS staging and sentinel addressing
are the Toeplitz builder's.

Each wave owns 8 channels. One ``16x16x32`` WMMA multiplies one-hot weights
(A) by 4 window columns x 8 channels of an input row (B) into 32 output
columns x 8 channels (D), ``w_fold`` images per tile; a filter row of ``KW``
taps takes ``ceil((KW + 1) / 4)`` passes. Input rows stream through LDS
double buffers (one barrier per row) into ``KH`` circular accumulator tiles,
and finished output rows drain through LDS as 16-byte stores.

Non-finite semantics: an Inf / NaN in X reaches every output of its 8-channel
group within its rows of the receptive field, over ``4 * passes`` columns
(``0 * Inf = NaN`` on the zero-weight taps). Finite inputs give the same
results as a scalar kernel up to fp32 summation order.

AOT: N, H, W, Ho, Wo, groups and the strides are kernargs; KH, KW and PAD are
capabilities; block_waves, block_h, w_fold and prefetch_rows are knobs.

C++ twin: ``library/cpp/instances/common/conv_direct_dw_fwd_wmma.cpp``
(``instance_conv_direct_dw_fwd_wmma.h``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from rocke.core.ir import KernelDef
from rocke.helpers.spec import ceil_div

from kernels.common.conv_direct_grouped import (
    DW_DGRAD_MFMA_CH,
    DW_DGRAD_MFMA_LDS_BUDGET,
    DW_DGRAD_MFMA_MAX_PREFETCH_ROWS,
    DW_DGRAD_MFMA_W_FOLDS,
    DW_DGRAD_WIN_MAX_TENSOR_BYTES,
    DirectConvProblem,
    DirectDepthwiseDgradWindowedSpec,
    _build_dw_dgrad_win_toeplitz,
    _dw_mfma_in_passes,
    _dw_mfma_lds_bytes,
    _dw_mfma_passes,
    _dw_mfma_tile_w,
    dilation_reason,
    one_padding_reason,
    problem_dims_reason,
)

DW_FWD_WMMA_ARCHES = ("gfx1250",)
DW_FWD_WMMA_WAVE = 32
# Up to 8 waves per block; the register-resident fragments (A weights, KH
# accumulator tiles, a row of B fragments, the prefetched rows) stay under
# this many VGPRs so no form spills. 16-wave blocks (wider channel runs per
# pixel) are for the small filters whose fragments fit the tighter budget.
DW_FWD_WMMA_MAX_WAVES = 16
DW_FWD_WMMA_MAX_FRAG_VGPRS = 224
DW_FWD_WMMA_WIDE_WAVES = 8
DW_FWD_WMMA_WIDE_FRAG_VGPRS = 112
# Statically unrolled WMMAs per wave (rows x KH x passes): a whole 56..64-row
# map in one tile.
DW_FWD_WMMA_MAX_UNROLL = 1024


@dataclass(frozen=True)
class DwFwdWmmaSpec:
    """Toeplitz WMMA depthwise forward (stride 1, cpg = kpg = 1).

    ``block_ch = 8 * block_waves`` channels per block; a block covers
    ``tile_w = 32 / w_fold`` output columns of ``w_fold`` images and
    ``block_h`` output rows. Grid:
    ``(ceil(Wo / tile_w), ceil(C / block_ch), ceil(N / w_fold) * ceil(Ho / block_h))``.
    """

    problem: DirectConvProblem
    name: str = "rocke_direct_dw_fwd_wmma"
    block_waves: int = 4
    block_h: int = 8
    w_fold: int = 1
    prefetch_rows: int = 2

    @property
    def wave_size(self) -> int:
        return DW_FWD_WMMA_WAVE

    @property
    def threads_per_block(self) -> int:
        return self.block_waves * DW_FWD_WMMA_WAVE

    @property
    def block_ch(self) -> int:
        return self.block_waves * DW_DGRAD_MFMA_CH

    @property
    def tile_w(self) -> int:
        return _dw_mfma_tile_w(self.w_fold)

    def grid(self) -> Tuple[int, int, int]:
        p = self.problem
        return (
            ceil_div(p.Wo, self.tile_w),
            ceil_div(p.groups, self.block_ch),
            ceil_div(p.N, self.w_fold) * ceil_div(p.Ho, self.block_h),
        )

    def launch_geometry(self):
        return self.grid(), (self.threads_per_block, 1, 1)

    def in_passes(self) -> int:
        """16-byte input loads per thread and window row."""
        return _dw_mfma_in_passes(
            self.tile_w,
            self.problem.KW,
            self.w_fold,
            self.block_ch,
            self.threads_per_block,
        )

    def frag_vgprs(self) -> int:
        """VGPRs of the register-resident fragments: the one-hot weights
        (KH x passes) and KH accumulator tiles at 8 each, one row of B
        fragments, and the prefetched rows' 16-byte loads."""
        p = self.problem
        kwp = _dw_mfma_passes(p.KW)
        return 8 * (p.KH * kwp + p.KH + kwp) + 4 * self.prefetch_rows * self.in_passes()

    def lds_bytes(self) -> int:
        return _dw_mfma_lds_bytes(
            self.in_passes(),
            self.threads_per_block,
            self.block_ch,
            self.w_fold,
            self.tile_w,
        )

    def unrolled_wmmas(self) -> int:
        p = self.problem
        return self.block_h * p.KH * _dw_mfma_passes(p.KW)

    def kernel_name(self) -> str:
        from rocke.helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"r{p.KH}s{p.KW}p{p.PAD}",
            f"wv{self.block_waves}",
            f"bh{self.block_h}",
            f"f{self.w_fold}",
            f"pf{self.prefetch_rows}",
            flags={"bf16": p.dtype == "bf16"},
        )


def dw_fwd_wmma_reason(spec: DwFwdWmmaSpec) -> Optional[str]:
    """Why ``spec`` cannot be built or run on its problem (``None`` if it can)."""
    p = spec.problem
    why = problem_dims_reason(p)
    if why is not None:
        return why
    if p.dtype not in ("fp16", "bf16"):
        return f"unsupported dtype {p.dtype!r}; expected fp16 or bf16"
    if p.cpg != 1 or p.kpg != 1:
        return f"requires cpg=kpg=1 (got {p.cpg}, {p.kpg})"
    if p.stride != 1:
        return f"requires stride=1 (got {p.stride})"
    why = dilation_reason(p) or one_padding_reason(p)
    if why is not None:
        return why
    if not 0 <= p.PAD < max(p.KH, p.KW) or p.Ho <= 0 or p.Wo <= 0:
        return f"degenerate geometry (PAD={p.PAD}, Ho={p.Ho}, Wo={p.Wo})"
    if spec.block_h < 1:
        return f"block_h must be >= 1 (got {spec.block_h})"
    if not 1 <= spec.block_waves <= DW_FWD_WMMA_MAX_WAVES:
        return (
            f"block_waves must be 1..{DW_FWD_WMMA_MAX_WAVES} (got {spec.block_waves})"
        )
    if spec.w_fold not in DW_DGRAD_MFMA_W_FOLDS:
        return f"w_fold must be one of {DW_DGRAD_MFMA_W_FOLDS} (got {spec.w_fold})"
    if not 1 <= spec.prefetch_rows <= DW_DGRAD_MFMA_MAX_PREFETCH_ROWS:
        return (
            f"prefetch_rows must be 1..{DW_DGRAD_MFMA_MAX_PREFETCH_ROWS} "
            f"(got {spec.prefetch_rows})"
        )
    # A plan-time rule: the IR does not depend on the group count.
    if p.groups % DW_DGRAD_MFMA_CH:
        return f"groups % {DW_DGRAD_MFMA_CH} != 0 (got {p.groups})"
    frag_vgprs = spec.frag_vgprs()
    if (
        spec.block_waves > DW_FWD_WMMA_WIDE_WAVES
        and frag_vgprs > DW_FWD_WMMA_WIDE_FRAG_VGPRS
    ):
        return (
            f"{spec.block_waves} waves need fragments <= "
            f"{DW_FWD_WMMA_WIDE_FRAG_VGPRS} VGPRs (got {frag_vgprs})"
        )
    if frag_vgprs > DW_FWD_WMMA_MAX_FRAG_VGPRS:
        return (
            f"fragments need {frag_vgprs} VGPRs (> "
            f"{DW_FWD_WMMA_MAX_FRAG_VGPRS}); lower prefetch_rows or the filter"
        )
    if spec.unrolled_wmmas() > DW_FWD_WMMA_MAX_UNROLL:
        return (
            f"unrolled body too large ({spec.unrolled_wmmas()} WMMAs > "
            f"{DW_FWD_WMMA_MAX_UNROLL}); lower block_h"
        )
    if spec.lds_bytes() > DW_DGRAD_MFMA_LDS_BUDGET:
        return (
            f"stages {spec.lds_bytes()} LDS bytes per workgroup (budget "
            f"{DW_DGRAD_MFMA_LDS_BUDGET}); lower block_waves"
        )
    # Every in-range byte offset plus a sentinel must stay a valid unsigned
    # 32-bit buffer offset.
    x_bytes = p.N * p.H * p.W * p.groups * 2
    y_bytes = p.N * p.Ho * p.Wo * p.groups * 2
    if max(x_bytes, y_bytes) > DW_DGRAD_WIN_MAX_TENSOR_BYTES:
        return (
            f"X/Y of {max(x_bytes, y_bytes)} bytes exceed the "
            f"{DW_DGRAD_WIN_MAX_TENSOR_BYTES}-byte sentinel addressing range"
        )
    return None


def is_valid_dw_fwd_wmma_spec(
    spec: DwFwdWmmaSpec, arch: str = "gfx1250"
) -> Tuple[bool, str]:
    if arch not in DW_FWD_WMMA_ARCHES:
        return False, f"needs one of {DW_FWD_WMMA_ARCHES} (got {arch})"
    why = dw_fwd_wmma_reason(spec)
    return (False, why) if why else (True, "ok")


def dw_fwd_wmma_win_spec(spec: DwFwdWmmaSpec) -> DirectDepthwiseDgradWindowedSpec:
    """The windowed Toeplitz spec whose wave32 ``fwd`` form is ``spec``'s
    kernel: the transposed problem (its dX the forward output, its dY the
    forward input, its PAD ``KH - 1 - PAD``) with the same knobs. The
    Toeplitz builder pads both window axes by the forward ``PAD``, so a
    rectangular filter keeps its stride-1 geometry."""
    p = spec.problem
    return DirectDepthwiseDgradWindowedSpec(
        problem=replace(p, H=p.Ho, W=p.Wo, PAD=p.KH - 1 - p.PAD),
        name=spec.name,
        block_waves=spec.block_waves,
        block_h=spec.block_h,
        mfma=True,
        w_fold=spec.w_fold,
        prefetch_rows=spec.prefetch_rows,
        wave_size=DW_FWD_WMMA_WAVE,
        fwd=True,
    )


def build_dw_fwd_wmma(spec: DwFwdWmmaSpec, arch: str = "gfx1250") -> KernelDef:
    """Build the Toeplitz WMMA depthwise forward kernel.

    Tensor roles (direct forward ABI):
      A param -- X: input,   [N, H, W, groups], NHWC
      B param -- W: weights, [groups, KH, KW, 1], KRSC
      D param -- Y: output,  [N, Ho, Wo, groups], NHWK

    Y[n, ho, wo, c] = sum_{r, s} W[c, r, s] * X[n, ho - PAD + r, wo - PAD + s, c].

    The kernel is the windowed Toeplitz builder's wave32 ``fwd`` form
    (:func:`dw_fwd_wmma_win_spec`) under this family's name; this module
    owns the forward's limits (:func:`dw_fwd_wmma_reason`).
    """
    ok, why = is_valid_dw_fwd_wmma_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid DwFwdWmmaSpec for {arch}: {why}")
    return _build_dw_dgrad_win_toeplitz(
        dw_fwd_wmma_win_spec(spec), arch, kernel_name=spec.kernel_name()
    )
