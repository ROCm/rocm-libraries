# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Implicit-GEMM backward-data convolution (dgrad) kernel instance.

Computes the input gradient of a 2-D convolution:

    dX[n, hi, wi, c] = sum_{y, x, k} dY[n, ho, wo, k] * W[k, y, x, c]
    where  ho = (hi + pH - y*dH) / sH   (must be integer and in [0, Ho))
           wo = (wi + pW - x*dW) / sW   (must be integer and in [0, Wo))

This is an implicit-GEMM of shape:

    M     = N*Hi*Wi       (input spatial positions)
    N_dg  = C             (input channels)
    K_dg  = Y*X*K         (filter spatial × output channels — reduction)

So:
    A operand: dY (output gradient), layout NHWK  →  GEMM A: (M, K_dg)
    B operand: W  (weights),         layout KYXC  →  GEMM B: (K_dg, N_dg)
    D operand: dX (input gradient),  layout NHWC  →  GEMM D: (M, N_dg)

All convolutions are handled by a single tiled kernel using a tilde
decomposition into ``y_tilde × x_tilde`` sub-GEMMs.  For stride=1 this
degenerates to one sub-GEMM and the epilogue uses a direct buffer_store.
For stride > 1 or split_k > 1 the epilogue uses global_atomic_fadd.

GEMM dimension mapping
----------------------
Forward::

    GEMM-M   = N*Ho*Wo      (output spatial positions)
    GEMM-N   = K            (output channels)
    GEMM-K   = Y*X*C        (filter × input channel)

Dgrad::

    GEMM-M   = N*Hi*Wi      (input spatial positions)
    GEMM-N   = C            (input channels)
    GEMM-K   = Y*X*K        (filter spatial × output channels, reduction)

Split-K
-------
When ``split_k > 1`` the K_dg reduction is partitioned across ``split_k``
Z-grid CTAs.  The caller must zero-initialise ``dX`` before launch; all
kernels (including stride=1 with split_k=1) accept a ``sub_gemm_buf``
parameter carrying the tilde-decomposition record(s).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from math import ceil as _ceil, gcd as _gcd
from typing import List, Optional, Sequence, Tuple

from ...core.ir import (
    BF16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from ...helpers.atoms import MfmaAtom, mfma_atom
from ...helpers.epilogues import CShuffleEpilogue, DirectEpilogue
from ...helpers.geometry import WarpGrid
from ...helpers.layouts import LdsLayout
from ...helpers.loads import AsyncTileLoader, CoalescedTileLoader
from ...helpers.mfma_gemm_inner import decode_mfma_lanes
from ...helpers.pipeline import SoftwarePipeline
from ...helpers.schedule import SchedulePolicy
from ...helpers.spec import kernel_name_join
from ...helpers.tensor_view import make_buffer_resource
from ...helpers.transforms import TensorDescriptor, embed, pad, unmerge_magic
from ._conv_implicit_gemm_common import (
    ConvAccumulatorEpilogue,
    ConvDataSpec,
    ConvProblem,
    _apply_accumulator_epilogue,
    _choose_load_vec_for,
    _emit_frag_smem_load,
    _emit_mfma,
    _emit_smem_load,
    _ir_dtype,
    build_wavelet_loaders,
    compute_wavelet_epi_barriers,
    emit_wavelet_kloop,
)


# ---------------------------------------------------------------------
# Dgrad-specific GEMM dimension helpers
# ---------------------------------------------------------------------


def _dg_M(p: ConvProblem) -> int:
    """Dgrad GEMM-M: input spatial positions."""
    return p.N * p.Hi * p.Wi


def _dg_N(p: ConvProblem) -> int:
    """Dgrad GEMM-N: input channels (per group)."""
    return p.cpg


def _dg_K(p: ConvProblem) -> int:
    """Dgrad GEMM-K (reduction): filter spatial × output channels (per group)."""
    return p.Y * p.X * p.kpg


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _floor_div(a: int, b: int) -> int:
    return a // b


# ---------------------------------------------------------------------
# Tilde decomposition (for stride > 1)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TildeDecomposition:
    """Precomputed tilde parameters for the backward-data convolution.

    When stride > 1 the backward convolution decomposes into
    ``y_tilde * x_tilde`` independent sub-GEMMs, each covering a subset
    of filter positions and input spatial positions.  When stride == 1
    and dilation == 1 this degenerates to a single sub-GEMM.
    """

    gcd_h: int
    gcd_w: int
    y_tilde: int
    x_tilde: int
    y_dot: int
    x_dot: int
    h_tilde: int
    w_tilde: int


def compute_tilde(p: ConvProblem) -> TildeDecomposition:
    """Compute the tilde decomposition from a ConvProblem."""
    gcd_h = _gcd(p.sH, p.dH)
    gcd_w = _gcd(p.sW, p.dW)
    y_tilde = p.sH // gcd_h
    x_tilde = p.sW // gcd_w
    y_dot = _ceil_div(p.Y, y_tilde)
    x_dot = _ceil_div(p.X, x_tilde)
    h_tilde = p.Ho + _ceil_div(p.dH * (p.Y - 1), p.sH) if p.Y > 1 else p.Ho
    w_tilde = p.Wo + _ceil_div(p.dW * (p.X - 1), p.sW) if p.X > 1 else p.Wo
    return TildeDecomposition(
        gcd_h=gcd_h,
        gcd_w=gcd_w,
        y_tilde=y_tilde,
        x_tilde=x_tilde,
        y_dot=y_dot,
        x_dot=x_dot,
        h_tilde=h_tilde,
        w_tilde=w_tilde,
    )


@dataclass(frozen=True)
class SubGemmParams:
    """Per-sub-GEMM parameters for the tilde decomposition."""

    i_ytilde: int
    i_xtilde: int
    y_dot_slice: int
    x_dot_slice: int
    h_tilde_slice_begin: int
    h_tilde_slice: int
    w_tilde_slice_begin: int
    w_tilde_slice: int
    gemm_m: int
    gemm_n: int
    gemm_k: int
    block_start: int
    block_end: int

    # Pre-computed coefficients for the kernel's offset arithmetic.
    a_embed_h_coeff: int
    a_embed_w_coeff: int
    b_y_stride: int
    b_y_offset: int
    b_x_stride: int
    b_x_offset: int
    d_h_stride: int
    d_h_offset: int
    d_w_stride: int
    d_w_offset: int
    gemm_k_padded: int

    RECORD_NUM_FIELDS: int = 22

    def pack_i32(self) -> List[int]:
        """Pack this record as a list of 22 signed i32 values."""
        return [
            self.block_start,
            0,  # placeholder: num_m_tiles, overwritten by pack_sub_gemm_buffer
            0,  # placeholder: num_n_tiles, overwritten by pack_sub_gemm_buffer
            self.gemm_m,
            self.gemm_k,
            self.h_tilde_slice,
            self.w_tilde_slice,
            self.h_tilde_slice_begin,
            self.w_tilde_slice_begin,
            self.y_dot_slice,
            self.x_dot_slice,
            self.a_embed_h_coeff,
            self.a_embed_w_coeff,
            self.b_y_stride,
            self.b_y_offset,
            self.b_x_stride,
            self.b_x_offset,
            self.d_h_stride,
            self.d_h_offset,
            self.d_w_stride,
            self.d_w_offset,
            self.gemm_k_padded,
        ]


def enumerate_sub_gemms(
    p: ConvProblem,
    tilde: TildeDecomposition,
    tile_m: int,
    tile_n: int,
    tile_k: int = 32,
    split_k: int = 1,
) -> List[SubGemmParams]:
    """Enumerate all non-empty sub-GEMMs for the tilde decomposition.

    Returns a list of :class:`SubGemmParams` sorted by ``(i_ytilde, i_xtilde)``,
    with cumulative ``block_start`` / ``block_end`` ranges suitable for a
    binary search dispatch inside the kernel.
    """
    sub_gemms: List[SubGemmParams] = []
    cumulative_tiles = 0

    for i_yt in range(tilde.y_tilde):
        for i_xt in range(tilde.x_tilde):
            y_dot_slice = _ceil_div(p.Y - i_yt, tilde.y_tilde)
            x_dot_slice = _ceil_div(p.X - i_xt, tilde.x_tilde)
            if y_dot_slice <= 0 or x_dot_slice <= 0:
                continue

            h_tilde_slice_begin = _floor_div(
                max(0, p.pH - p.dH * (tilde.y_tilde - 1)), p.sH
            )
            h_tilde_slice_end = min(
                tilde.h_tilde,
                _ceil_div(p.pH + p.Hi - 1, p.sH) + 1,
            )
            w_tilde_slice_begin = _floor_div(
                max(0, p.pW - p.dW * (tilde.x_tilde - 1)), p.sW
            )
            w_tilde_slice_end = min(
                tilde.w_tilde,
                _ceil_div(p.pW + p.Wi - 1, p.sW) + 1,
            )

            h_tilde_slice = h_tilde_slice_end - h_tilde_slice_begin
            w_tilde_slice = w_tilde_slice_end - w_tilde_slice_begin
            if h_tilde_slice <= 0 or w_tilde_slice <= 0:
                continue

            gemm_m = p.N * h_tilde_slice * w_tilde_slice
            gemm_n = p.cpg
            gemm_k = y_dot_slice * x_dot_slice * p.kpg

            num_m_tiles = _ceil_div(gemm_m, tile_m)
            num_n_tiles = _ceil_div(gemm_n, tile_n)
            num_tiles = num_m_tiles * num_n_tiles

            block_start = cumulative_tiles
            block_end = cumulative_tiles + num_tiles
            cumulative_tiles = block_end

            sg = SubGemmParams(
                i_ytilde=i_yt,
                i_xtilde=i_xt,
                y_dot_slice=y_dot_slice,
                x_dot_slice=x_dot_slice,
                h_tilde_slice_begin=h_tilde_slice_begin,
                h_tilde_slice=h_tilde_slice,
                w_tilde_slice_begin=w_tilde_slice_begin,
                w_tilde_slice=w_tilde_slice,
                gemm_m=gemm_m,
                gemm_n=gemm_n,
                gemm_k=gemm_k,
                block_start=block_start,
                block_end=block_end,
                a_embed_h_coeff=-(p.dH // tilde.gcd_h),
                a_embed_w_coeff=-(p.dW // tilde.gcd_w),
                b_y_stride=tilde.y_tilde,
                b_y_offset=i_yt,
                b_x_stride=tilde.x_tilde,
                b_x_offset=i_xt,
                d_h_stride=p.sH,
                d_h_offset=p.dH * i_yt + p.sH * h_tilde_slice_begin - p.pH,
                d_w_stride=p.sW,
                d_w_offset=p.dW * i_xt + p.sW * w_tilde_slice_begin - p.pW,
                gemm_k_padded=_ceil_div(gemm_k, tile_k * split_k) * (tile_k * split_k),
            )
            sub_gemms.append(sg)

    return sub_gemms


def pack_sub_gemm_buffer(
    sub_gemms: List[SubGemmParams], tile_m: int, tile_n: int
) -> List[int]:
    """Pack all sub-GEMM records into a flat i32 buffer for the kernel."""
    buf: List[int] = []
    for sg in sub_gemms:
        record = sg.pack_i32()
        record[1] = _ceil_div(sg.gemm_m, tile_m)
        record[2] = _ceil_div(sg.gemm_n, tile_n)
        buf.extend(record)
    return buf


# ---------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------


def make_dgrad_dy_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (m, k_dg) -> NHWK offset descriptor for dY (output gradient).

    dY is stored in NHWK layout.  In the dgrad GEMM:
      - the M dimension indexes input spatial positions ``(n, hi, wi)``
      - the K_dg reduction dimension indexes filter+output-channel ``(y, x, k_out)``

    The embed transforms compute the *output* spatial coordinates from the
    *input* coordinates and filter position:
      ``ho = hi*1 + y*(-1) + pH``  (inverse of the forward ``hi = ho*sH - pH + y*dH``)
      ``wo = wi*1 + x*(-1) + pW``

    The embed's ``lo=0, hi=Ho`` (and ``lo=0, hi=Wo``) bounds check handles
    out-of-bounds access: when ``ho`` falls outside ``[0, Ho)`` the descriptor
    returns an invalid (OOB-clamped) offset and the buffer load silently
    returns zero.

    2-D (stride=1, dilation=1)::

        naive(NHWK):         (n, ho, wo, k_out)
        unmerge('m' → n, hi, wi)
        embed([hi, y] → ho, strides=[1, -1], offset=pH, lo=0, hi=Ho)
        embed([wi, x] → wo, strides=[1, -1], offset=pW, lo=0, hi=Wo)
        unmerge('k_dg' → k_out, y, x)
        pad('y'), pad('x')   →  partial-tile boundary guard

    The unmerge order ``[k_out, y, x]`` places output channels (k_out) as the
    slowest-varying component of ``k_dg``.  This is critical for split-K
    correctness: a contiguous K-slice partitions k_out (channels) while still
    visiting all (y, x) filter positions within each slice.  Reversing the
    order (y/x outermost) would cross-contaminate spatial positions across
    K-slices and produce wrong results.
    """
    return TensorDescriptor.naive(
        "dY_nhwk",
        lengths=[p.N, p.Ho, p.Wo, p.K],
        dtype=_ir_dtype(dtype),
        coord_names=["n", "ho", "wo", "k_out"],
    ).transform(
        unmerge_magic("m", into=["n", "hi", "wi"], dims=[p.N, p.Hi, p.Wi]),
        embed(
            upper=["hi", "y"],
            into="ho",
            strides=[1, -1],
            offset=p.pH,
            lo=0,
            hi=p.Ho,
        ),
        embed(
            upper=["wi", "x"],
            into="wo",
            strides=[1, -1],
            offset=p.pW,
            lo=0,
            hi=p.Wo,
        ),
        unmerge_magic("k_dg", into=["k_out", "y", "x"], dims=[p.K, p.Y, p.X]),
        pad("k_out", lo=0, hi=p.K),
        pad("y", lo=0, hi=p.Y),
        pad("x", lo=0, hi=p.X),
    )


def make_dgrad_w_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (c, k_dg) -> KYXC offset descriptor for W (weights).

    W is stored in KYXC layout.  In the dgrad GEMM:
      - the N_dg dimension is ``c`` (input channels, the non-reduction column)
      - the K_dg reduction dimension is ``(y, x, k_out)``

    2-D::

        naive(KYXC):          (k_out, y, x, c)
        unmerge('k_dg' → k_out, y, x)   →  user-facing: (c, k_dg)
        pad('y'), pad('x')               →  partial-tile boundary guard

    The unmerge order must match the A descriptor (k_out outermost) so the
    same k_dg value maps to the same (k_out, y, x) triple in both operands.
    """
    return TensorDescriptor.naive(
        "W_kyxc",
        lengths=[p.K, p.Y, p.X, p.C],
        dtype=_ir_dtype(dtype),
        coord_names=["k_out", "y", "x", "c"],
    ).transform(
        unmerge_magic("k_dg", into=["k_out", "y", "x"], dims=[p.K, p.Y, p.X]),
        pad("k_out", lo=0, hi=p.K),
        pad("y", lo=0, hi=p.Y),
        pad("x", lo=0, hi=p.X),
    )


def make_dgrad_dx_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (m, c) -> NHWC offset descriptor for dX (input gradient).

    dX is stored in NHWC layout.  In the dgrad GEMM:
      - the M dimension indexes input spatial positions ``(n, hi, wi)``
      - the N_dg dimension is ``c`` (input channels)

    2-D::

        naive(NHWC):          (n, hi, wi, c)
        unmerge('m' → n, hi, wi)   →  user-facing: (m, c)
    """
    return TensorDescriptor.naive(
        "dX_nhwc",
        lengths=[p.N, p.Hi, p.Wi, p.C],
        dtype=_ir_dtype(dtype),
        coord_names=["n", "hi", "wi", "c"],
    ).transform(
        unmerge_magic("m", into=["n", "hi", "wi"], dims=[p.N, p.Hi, p.Wi]),
    )


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DgradConvSpec:
    """One concrete implicit-GEMM backward-data convolution configuration.

    GEMM orientation:
      M     = N*Hi*Wi       (input spatial positions)
      N_dg  = C             (input channels)
      K_dg  = Y*X*K         (filter spatial × output channels — reduction)

    Pipeline, epilogue, and async-DMA options are the same as
    :class:`~.conv_implicit_gemm.ImplicitGemmConvSpec`.

    Grouped convolution (``groups > 1``, channels-per-group > 1) and 2-D are
    supported; group-merging and 3-D are not.  Grouped runs grid-per-group with
    the conv group on ``blockIdx.y`` (see :func:`_build_tilde_dgrad`).
    """

    problem: ConvProblem
    name: str = "conv_igemm_dgrad"
    data: ConvDataSpec = field(default_factory=ConvDataSpec)

    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 64

    warp_m: int = 2
    warp_n: int = 2

    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16

    wave_size: int = 64

    pipeline: str = "mem"
    epilogue: str = "default"
    async_dma: bool = False
    unroll_k: bool = False
    lds_k_pad: Optional[int] = None

    vector_size_a: Optional[int] = None
    vector_size_b: Optional[int] = None
    vector_size_c: Optional[int] = None

    lds_layout: Optional[LdsLayout] = None

    chiplet_swizzle: bool = False
    chiplet_wgm: int = 8
    chiplet_num_xcds: int = 8
    chiplet_chunk_size: int = 64

    waves_per_eu: Optional[int] = None
    acc_epilogue: ConvAccumulatorEpilogue = field(
        default_factory=ConvAccumulatorEpilogue
    )
    split_k: int = 1
    # Number of extra load waves for pipeline="wavelet" (gfx1250/WMMA only).
    # Ignored for all other pipelines.
    num_load_waves: int = 4
    # cshuffle LDS aliasing — same semantics as ImplicitGemmConvSpec.cshuffle_no_alias.
    # Wavelet forces additive (no_alias=True) because A/B stay live across both branches.
    cshuffle_no_alias: bool = False

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * self.wave_size

    @property
    def launch_block_size(self) -> int:
        """Total threads launched per workgroup.

        For ``pipeline="wavelet"`` this is ``block_size + num_load_waves * wave_size``;
        for all other pipelines it equals ``block_size``.
        """
        if self.pipeline == "wavelet":
            return self.block_size + self.num_load_waves * self.wave_size
        return self.block_size

    @property
    def k_atoms_per_tile_k(self) -> int:
        return self.tile_k // self.warp_tile_k

    @property
    def mfmas_per_warp_m(self) -> int:
        return self.tile_m // (self.warp_m * self.warp_tile_m)

    @property
    def mfmas_per_warp_n(self) -> int:
        return self.tile_n // (self.warp_n * self.warp_tile_n)

    @property
    def atom(self) -> MfmaAtom:
        return mfma_atom(
            self.data.dtype_a, self.warp_tile_m, self.warp_tile_n, self.warp_tile_k
        )

    # ---- vector-size helpers ----

    @staticmethod
    def default_vector_sizes(C: int, K: int, dtype: str) -> "Tuple[int, int, int]":
        """Return ``(vec_a, vec_b, vec_c)`` for a dgrad problem.

        Dgrad memory layout:
          A (dY):  NHWK → last dim K → vec_a
          B (W):   KYXC → last dim C → vec_b
          D (dX):  NHWC → last dim C → vec_c
        """
        sizes = [8, 4, 2, 1] if dtype != "fp32" else [4, 2, 1]

        def _vec(n: int) -> int:
            return next(v for v in sizes if n % v == 0)

        return _vec(K), _vec(C), _vec(C)

    # ---- dgrad GEMM dimensions ----

    @property
    def dg_M(self) -> int:
        return _dg_M(self.problem)

    @property
    def dg_N(self) -> int:
        return _dg_N(self.problem)

    @property
    def dg_K(self) -> int:
        return _dg_K(self.problem)

    def dg_K_padded(self) -> int:
        """K_dg rounded up to the nearest multiple of ``tile_k * split_k``.

        Used by the builder as the loop upper bound so each split-K slice
        spans exactly ``dg_K_padded // split_k`` K-elements, which is itself
        a multiple of ``tile_k``.  Loads past the real ``dg_K`` are safe:
        the buffer descriptor's OOB-clamp flag silently returns zero for
        out-of-range byte offsets, so the MFMA contribution is zero.
        """
        stride = self.tile_k * self.split_k
        k = _dg_K(self.problem)
        return ((k + stride - 1) // stride) * stride

    def compute_sub_gemms(self) -> List[SubGemmParams]:
        """Return the list of sub-GEMMs for the tilde decomposition."""
        tilde = compute_tilde(self.problem)
        return enumerate_sub_gemms(
            self.problem,
            tilde,
            self.tile_m,
            self.tile_n,
            tile_k=self.tile_k,
            split_k=max(1, self.split_k),
        )

    @property
    def needs_atomic(self) -> bool:
        """True when the epilogue must use atomic-add.

        split_k>1 always requires atomics (multiple CTAs accumulate into the same
        dX elements).  Multiple tilde sub-GEMMs with split_k=1 do NOT require
        atomics: the tilde decomposition guarantees disjoint writes, so
        _emit_dgrad_tilde_direct/cshuffle_epilogue can issue plain buffer_store.
        """
        return self.split_k > 1

    @property
    def is_strided(self) -> bool:
        """True when stride > 1 or dilation > 1 (tilde decomposition needed)."""
        p = self.problem
        return p.sH != 1 or p.sW != 1 or p.dH != 1 or p.dW != 1

    def kernel_name(self) -> str:
        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"w{self.warp_m}x{self.warp_n}",
            f"a{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}",
            f"{self.pipeline}_{self.epilogue}",
            self.acc_epilogue.tag(),
            flags={
                "async": self.async_dma,
                f"spk{self.split_k}": self.split_k > 1,
                "spkauto": self.split_k == -1,
            },
        )

    def validate(self) -> None:
        if self.tile_m % (self.warp_m * self.warp_tile_m) != 0:
            raise ValueError(
                f"tile_m {self.tile_m} not divisible by warp_m * warp_tile_m "
                f"({self.warp_m} * {self.warp_tile_m})"
            )
        if self.tile_n % (self.warp_n * self.warp_tile_n) != 0:
            raise ValueError(
                f"tile_n {self.tile_n} not divisible by warp_n * warp_tile_n "
                f"({self.warp_n} * {self.warp_tile_n})"
            )
        if self.tile_k % self.warp_tile_k != 0:
            raise ValueError(
                f"tile_k {self.tile_k} not divisible by warp_tile_k {self.warp_tile_k}"
            )
        if self.block_size > 1024:
            raise ValueError(f"block_size {self.block_size} > 1024")
        if self.split_k < -1 or self.split_k == 0:
            raise ValueError(
                f"split_k must be -1 (auto), 1 (disabled), or >1 (fixed); "
                f"got {self.split_k}"
            )
        if self.split_k > 1:
            if self.data.dtype_d not in ("fp32", "bf16", "fp16"):
                raise ValueError(
                    f"split_k > 1 requires dtype_d in fp32/bf16/fp16 "
                    f"(got {self.data.dtype_d!r})"
                )
            if self.data.dtype_d in ("bf16", "fp16") and self.problem.cpg % 2 != 0:
                raise ValueError(
                    f"split_k > 1 with dtype_d={self.data.dtype_d!r} requires even "
                    f"channels-per-group (packed <2 x dtype> atomic pairs on the "
                    f"innermost NHWC dimension must stay within one group's slab); "
                    f"got cpg={self.problem.cpg}"
                )
        p = self.problem
        if p.is_3d:
            raise ValueError(
                "DgradConvSpec: 3-D convolution is not yet supported for the dgrad "
                "direction (Phase 1 is 2-D only)"
            )
        layout = self.effective_lds_layout()
        if self.async_dma:
            layout.validate_for_async()
        if self.async_dma and self.lds_k_pad not in (None, 0):
            raise ValueError(
                "async_dma requires lds_k_pad to be 0/None because "
                "raw_ptr_buffer_load_lds writes a packed lane-contiguous tile"
            )
        if (
            self.acc_epilogue.clamp_min is not None
            and self.acc_epilogue.clamp_max is not None
            and self.acc_epilogue.clamp_min > self.acc_epilogue.clamp_max
        ):
            raise ValueError(
                "acc_epilogue clamp_min must be <= clamp_max "
                f"(got {self.acc_epilogue.clamp_min} > {self.acc_epilogue.clamp_max})"
            )

    def effective_lds_layout(self) -> LdsLayout:
        if self.lds_layout is not None:
            layout = self.lds_layout
        elif self.lds_k_pad is not None:
            layout = LdsLayout.padded_k(self.tile_k, self.lds_k_pad)
        elif self.async_dma:
            layout = LdsLayout.packed_async(self.tile_k)
        else:
            layout = LdsLayout.padded_k(self.tile_k, 8 if self.tile_k >= 16 else 0)
        layout.validate()
        return layout


# ---------------------------------------------------------------------
# Arch-aware spec validation
# ---------------------------------------------------------------------


def is_valid_dgrad_spec(spec: DgradConvSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``."""
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    p = spec.problem
    if p.is_3d:
        return False, "dgrad only supports 2-D convolution currently"
    if p.groups > 1 and p.cpg == 1:
        return False, (
            "depthwise dgrad (channels-per-group == 1) is not supported by the "
            "implicit-GEMM grouped path"
        )

    if spec.tile_m % (spec.warp_m * spec.warp_tile_m):
        return False, "tile_m not divisible by warp_m * warp_tile_m"
    if spec.tile_n % (spec.warp_n * spec.warp_tile_n):
        return False, "tile_n not divisible by warp_n * warp_tile_n"
    if spec.tile_k % spec.warp_tile_k:
        return False, "tile_k not divisible by warp_tile_k"
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > {target.max_threads_per_block} "
            f"(hardware cap) on {arch}"
        )
    if (
        spec.vector_size_c is not None
        and spec.vector_size_c > 1
        and spec.epilogue == "default"
        and spec.split_k <= 1  # atomic (split_k>1) ignores vector_size_c
        and not spec.is_strided  # tilde non-atomic also uses scalar direct — vec_c ignored
    ):
        return False, (
            f"default epilogue is not supported with vector size c: {spec.vector_size_c}"
        )

    family = "wmma" if target.wave_size == 32 else "mma"
    if spec.wave_size != target.wave_size:
        return False, (
            f"spec wave_size {spec.wave_size} != {arch} wave_size {target.wave_size}"
        )

    sk = spec.split_k
    if sk < -1 or sk == 0:
        return False, f"split_k must be -1 (auto), 1, or >1 (got {sk})"
    if sk > 1 and family != "mma":
        return False, f"split_k > 1 is CDNA-only (got family {family!r} on {arch})"
    if sk > 1 and spec.data.dtype_d not in ("fp32", "bf16", "fp16"):
        return False, (
            f"split_k > 1 requires dtype_d in fp32/bf16/fp16 for atomic accumulation "
            f"(got {spec.data.dtype_d!r})"
        )
    if sk > 1 and spec.data.dtype_d in ("bf16", "fp16") and p.cpg % 2 != 0:
        return False, (
            f"split_k > 1 with dtype_d={spec.data.dtype_d!r} requires even "
            f"channels-per-group (packed <2 x dtype> atomic pairs on the innermost "
            f"NHWC dimension must stay within one group's slab); got cpg={p.cpg}"
        )

    atom = (spec.warp_tile_m, spec.warp_tile_n, spec.warp_tile_k)
    if not target.mma.has_shape(
        family=family,
        a_dtype=spec.data.dtype_a,
        b_dtype=spec.data.dtype_b,
        c_dtype="fp32",
        m=spec.warp_tile_m,
        n=spec.warp_tile_n,
        k=spec.warp_tile_k,
    ):
        return False, f"unsupported {spec.data.dtype_a} warp_tile {atom} on {arch}"

    _ab_dtype_bytes = 4 if spec.data.dtype_a in ("fp32",) else 2
    _lds_layout = spec.effective_lds_layout()
    _a_shape = _lds_layout.storage_shape(spec.tile_m)
    _b_shape = _lds_layout.storage_shape(spec.tile_n)
    _ab_bytes = (
        _a_shape[0] * _a_shape[1] + _b_shape[0] * _b_shape[1]
    ) * _ab_dtype_bytes
    _double = spec.pipeline == "compv4" or spec.async_dma or spec.unroll_k
    _ab_lds = _ab_bytes * (2 if _double else 1)
    _c_dtype_bytes = 4 if spec.data.dtype_d == "fp32" else 2
    _c_lds = (
        spec.tile_m * spec.tile_n * _c_dtype_bytes if spec.epilogue == "cshuffle" else 0
    )
    # wavelet: A/B stay live across both scf_if_else branches, so the LDS pool
    # cannot alias C onto the A/B region even when cshuffle_no_alias=False.
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    _total_lds = (_ab_lds + _c_lds) if _no_alias else max(_ab_lds, _c_lds)
    if not target.fits_lds(_total_lds):
        return False, (
            f"LDS budget {_total_lds} bytes "
            f"(A/B={'x2 ' if _double else ''}{_ab_bytes}, C={_c_lds}) "
            f"> {target.lds_capacity_bytes} cap on {arch}"
        )

    if spec.pipeline == "wavelet":
        if spec.num_load_waves < 1:
            return False, "pipeline='wavelet' requires num_load_waves >= 1"
        if family != "wmma":
            return False, (
                "pipeline='wavelet' is WMMA/gfx1250 only: on MFMA targets "
                "the single-buffer LDS is overwritten each K iteration and load/math "
                "waves execute sequentially rather than truly concurrently."
            )
        if spec.async_dma:
            return False, (
                "pipeline='wavelet' is incompatible with async_dma=True: "
                "the wavelet loaders are only constructed in the non-async branch "
                "and a_wavelet_loader/b_wavelet_loader would be None at fetch time."
            )
        _mfmas_m = spec.tile_m // (spec.warp_m * spec.warp_tile_m)
        _mfmas_n = spec.tile_n // (spec.warp_n * spec.warp_tile_n)
        _k_iters = (spec.dg_K + spec.tile_k - 1) // spec.tile_k
        _wmma_cost = _k_iters * _mfmas_m * _mfmas_n
        _WMMA_COST_LIMIT = 4096
        if _wmma_cost > _WMMA_COST_LIMIT:
            return False, (
                f"pipeline='wavelet' unrolled WMMA count {_wmma_cost} "
                f"(K_iters={_k_iters} × mfmas={_mfmas_m}×{_mfmas_n}) "
                f"exceeds compile-time limit {_WMMA_COST_LIMIT}; "
                f"reduce tile_k, tile_m, or tile_n"
            )
        _launch_block = spec.launch_block_size
        if _launch_block > target.max_threads_per_block:
            return False, (
                f"launch_block_size {_launch_block} > {target.max_threads_per_block} "
                f"(hardware cap) on {arch}"
            )

    if family == "wmma":
        # gfx1250 (wavelet-capable) supports 16x16x32; other RDNA (mem-only) supports 16x16x16.
        # Both atoms are accepted for the mem pipeline on gfx1250; wavelet supports both too.
        if atom not in ((16, 16, 16), (16, 16, 32)):
            return False, (
                f"WMMA dgrad supports 16x16x16 or 16x16x32 (got {atom}) on {arch}"
            )
        if spec.pipeline not in ("mem", "wavelet"):
            return False, (
                f"WMMA dgrad supports only 'mem' or 'wavelet' pipeline "
                f"(got {spec.pipeline!r}) on {arch}"
            )
        if spec.epilogue not in ("default", "cshuffle"):
            return False, (
                f"WMMA dgrad supports 'default' and 'cshuffle' epilogues "
                f"(got {spec.epilogue!r}) on {arch}"
            )
        for flag, label in (
            (spec.async_dma, "async_dma"),
            (spec.unroll_k, "unroll_k"),
            (spec.chiplet_swizzle, "chiplet_swizzle"),
            (
                spec.split_k > 1 and spec.pipeline != "wavelet",
                "split_k > 1 (non-wavelet WMMA)",
            ),
        ):
            if flag:
                return False, f"WMMA dgrad does not support {label} on {arch}"

    return True, "ok"


def _dgrad_mma_family(arch: str) -> str:
    from ...core.arch import ArchTarget

    return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma"


def _resolve_dgrad_op(spec: DgradConvSpec, arch: str):
    from ...core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    op = target.mma.op_for_shape(
        family=_dgrad_mma_family(arch),
        a_dtype=spec.data.dtype_a,
        b_dtype=spec.data.dtype_b,
        c_dtype="fp32",
        m=spec.warp_tile_m,
        n=spec.warp_tile_n,
        k=spec.warp_tile_k,
    )
    if op is None:
        raise ValueError(
            f"no MMA atom for dgrad warp_tile "
            f"({spec.warp_tile_m},{spec.warp_tile_n},{spec.warp_tile_k}) on {arch}"
        )
    return op


# ---------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------


def build_implicit_gemm_conv_dgrad(
    spec: DgradConvSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Build the IR for one implicit-GEMM backward-data conv kernel.

    GEMM shape (per sub-GEMM):
        M     = N*HTildeSlice*WTildeSlice   (input spatial positions)
        N_dg  = C                           (input channels)
        K_dg  = YDotSlice*XDotSlice*K       (filter × output channels, reduction)

    Operands:
        A (dY): output-gradient tensor, NHWK layout.
        B (W):  weight tensor, KYXC layout.
        D (dX): input-gradient output, NHWC layout.

    All convolutions are handled by a single tiled kernel path with a
    runtime-parameterised sub-GEMM dispatch (a parameter buffer stores
    per-sub-GEMM constants; each CTA binary-searches it).  For stride=1
    the tilde decomposition has exactly one sub-GEMM, so the binary search
    degenerates and the epilogue emits a direct buffer_store (no atomics).
    For stride > 1 or split_k > 1, multiple CTAs write to overlapping dX
    positions and the epilogue uses global_atomic_fadd.
    """
    if spec.split_k == -1:
        from ...helpers.split_k import select_split_k_wgrad
        from dataclasses import replace as _dc_replace

        # Re-use the wgrad split-K selector: the formula is purely
        # geometry-agnostic (grid tile count arithmetic), so passing
        # dgrad GEMM dimensions under the wg_* parameter names is safe.
        decision = select_split_k_wgrad(
            wg_M=_dg_M(spec.problem),
            wg_N=_dg_N(spec.problem),
            wg_K=_dg_K(spec.problem),
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            arch=arch,
        )
        spec = _dc_replace(spec, split_k=decision.split_k)

    spec.validate()
    ok, why = is_valid_dgrad_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid dgrad spec for {arch}: {why}")

    return _build_tilde_dgrad(spec, arch)


_RECORD_FIELDS = 22  # number of i32 fields per SubGemmRecord


def _emit_binary_search(
    b: IRBuilder,
    flat_block_id: Value,
    sub_gemm_buf: Value,
    num_sub_gemms: int,
) -> Value:
    """Emit a compile-time-unrolled binary search over block_starts.

    Returns the sub-GEMM index (i32) for the CTA at ``flat_block_id``.
    The ``block_start`` field is at offset 0 in each record.
    """
    import math

    lo = b.const_i32(0)
    hi = b.const_i32(num_sub_gemms)
    c_record_stride = b.const_i32(_RECORD_FIELDS)

    max_iters = int(math.ceil(math.log2(max(num_sub_gemms, 2)))) + 1
    for _ in range(max_iters):
        mid = b.div(b.add(lo, hi), b.const_i32(2))
        mid_block_start_offset = b.mul(mid, c_record_stride)
        mid_block_start = b.global_load_i32(sub_gemm_buf, mid_block_start_offset)
        take_lo = b.cmp_le(mid_block_start, flat_block_id)
        lo = b.select(take_lo, mid, lo)
        hi = b.select(take_lo, hi, mid)

    return lo


def _emit_load_record_field(
    b: IRBuilder, sub_gemm_buf: Value, sg_idx: Value, field_idx: int
) -> Value:
    """Load one i32 field from the sub-GEMM parameter buffer."""
    offset = b.add(
        b.mul(sg_idx, b.const_i32(_RECORD_FIELDS)),
        b.const_i32(field_idx),
    )
    return b.global_load_i32(sub_gemm_buf, offset)


def _build_tilde_dgrad(
    spec: DgradConvSpec,
    arch: str,
) -> KernelDef:
    """Build the tiled dgrad kernel for any stride.

    A parameter buffer holds per-sub-GEMM constants. Each CTA binary-searches
    it to find its sub-GEMM, loads the record fields, and uses runtime
    arithmetic for tensor offsets.

    For stride=1 (1 sub-GEMM, split_k=1): ``needs_atomic`` is False and the
    epilogue emits a direct buffer_store — equivalent to the old fast path.
    For stride>1 or split_k>1: the epilogue uses global_atomic_fadd to
    accumulate partial results from overlapping CTAs.
    """
    p = spec.problem
    sub_gemms = spec.compute_sub_gemms()
    num_sub_gemms = len(sub_gemms)
    tilde = compute_tilde(p)

    ir_dtype_a = _ir_dtype(spec.data.dtype_a)
    ir_dtype_b = _ir_dtype(spec.data.dtype_b)
    ir_dtype_d = _ir_dtype(spec.data.dtype_d)

    b = IRBuilder(spec.kernel_name())
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    dY = b.param(
        "dY", PtrType(ir_dtype_a, "global"), noalias=True, readonly=True, align=16
    )
    W = b.param(
        "W", PtrType(ir_dtype_b, "global"), noalias=True, readonly=True, align=16
    )
    # split_k>1 uses atomic_add; tilde split_k=1 uses direct store — both allow writeonly.
    _dx_writeonly = spec.split_k <= 1
    dX = b.param(
        "dX",
        PtrType(ir_dtype_d, "global"),
        noalias=True,
        writeonly=_dx_writeonly,
        align=16,
    )
    dY_bytes = b.param("dY_bytes", I32)
    W_bytes = b.param("W_bytes", I32)
    dX_bytes = b.param("dX_bytes", I32)
    sub_gemm_buf = b.param(
        "sub_gemm_buf", PtrType(I32, "global"), noalias=True, readonly=True, align=4
    )
    num_sub_gemms_param = b.param("num_sub_gemms", I32)

    op = _resolve_dgrad_op(spec, arch)
    atom = spec.atom if op.family == "mma" else None
    a_per_lane = op.a_frag_len
    b_per_lane = op.b_frag_len
    _smem_dtype: Optional[Type] = (
        BF16 if op.a_dtype == "bf16" else F32 if op.a_dtype == "fp32" else None
    )
    c_per_lane = op.c_frag_len

    block_m, block_n, block_k = spec.tile_m, spec.tile_n, spec.tile_k

    # Use a 1D grid: block_id_x covers all sub-GEMMs' tiles.
    flat_block_id = b.block_id_x()

    # Binary search to find which sub-GEMM this CTA belongs to.
    sg_idx = _emit_binary_search(b, flat_block_id, sub_gemm_buf, num_sub_gemms)

    # Load all record fields for this sub-GEMM.
    def _ld(field_idx: int) -> Value:
        return _emit_load_record_field(b, sub_gemm_buf, sg_idx, field_idx)

    rec_block_start = _ld(0)
    rec_num_m_tiles = _ld(1)
    rec_num_n_tiles = _ld(2)
    rec_gemm_m = _ld(3)
    rec_gemm_k = _ld(4)
    rec_h_tilde_slice = _ld(5)
    rec_w_tilde_slice = _ld(6)
    rec_h_tilde_slice_begin = _ld(7)
    rec_w_tilde_slice_begin = _ld(8)
    rec_y_dot_slice = _ld(9)
    rec_x_dot_slice = _ld(10)
    rec_a_embed_h_coeff = _ld(11)
    rec_a_embed_w_coeff = _ld(12)
    rec_b_y_stride = _ld(13)
    rec_b_y_offset = _ld(14)
    rec_b_x_stride = _ld(15)
    rec_b_x_offset = _ld(16)
    rec_d_h_stride = _ld(17)
    rec_d_h_offset = _ld(18)
    rec_d_w_stride = _ld(19)
    rec_d_w_offset = _ld(20)

    # Compute local tile indices within this sub-GEMM.
    local_flat = b.sub(flat_block_id, rec_block_start)
    local_m_tile = b.div(local_flat, rec_num_n_tiles)
    local_n_tile = b.mod(local_flat, rec_num_n_tiles)

    # Warp grid: bind() emits tid/lane/warp decomposition. We override
    # block_m/n_off with our own local tile offsets (not from block_id_x/y).
    grid = WarpGrid.from_atom(
        op,
        tile_m=block_m,
        tile_n=block_n,
        tile_k=block_k,
        warp_m=spec.warp_m,
        warp_n=spec.warp_n,
        wave_size=spec.wave_size,
    ).bind(b, block_m_axis="y", block_n_axis="x")
    tid = grid.tid
    lane = grid.lane
    warp_id = grid.warp_id
    warp_m_idx = grid.warp_m_idx
    warp_n_idx = grid.warp_n_idx

    block_m_off_v = b.mul(local_m_tile, b.const_i32(block_m))
    block_n_off_v = b.mul(local_n_tile, b.const_i32(block_n))
    grid = dc_replace(grid, block_m_off=block_m_off_v, block_n_off=block_n_off_v)

    c0 = b.const_i32(0)
    c_block_k = b.const_i32(block_k)

    _is_split_k = spec.split_k > 1
    if _is_split_k:
        rec_gemm_k_padded = _ld(21)
        c_split_k = b.const_i32(spec.split_k)
        k_slice = b.div(rec_gemm_k_padded, c_split_k)
        k_lo = b.mul(b.block_id_z(), k_slice)
        k_hi = b.add(k_lo, k_slice)
    else:
        k_lo = c0
        k_hi = rec_gemm_k

    # Compile-time problem constants for offset computation.
    c_Ho = b.const_i32(p.Ho)
    c_Wo = b.const_i32(p.Wo)
    c_K = b.const_i32(p.K)
    c_Hi = b.const_i32(p.Hi)
    c_Wi = b.const_i32(p.Wi)
    c_C = b.const_i32(p.C)
    c_Y = b.const_i32(p.Y)
    c_X = b.const_i32(p.X)
    c_dg_N = b.const_i32(p.cpg)

    # Grouped conv (groups > 1): the conv group rides blockIdx.y.  blockIdx.z is
    # split_k and blockIdx.x is the flat tilde-tile index, so y is free (it is
    # launched as 1 for ungrouped and its WarpGrid block_m offset is overridden
    # below).  Each CTA handles exactly one group: its reduction is confined to
    # that group's kpg output channels and it writes that group's cpg
    # input-channel slab of dX.  The group offsets the absolute output-channel
    # base (k_out = g*kpg + local) on dY/W and the absolute input-channel base
    # (c = g*cpg + local) on W/dX, and the k_sub decode divides by kpg (not the
    # total K).  For groups == 1 nothing is emitted, keeping the IR byte-identical.
    grouped = p.groups > 1
    if grouped:
        group_idx = b.block_id_y()
        c_kpg = b.const_i32(p.kpg)
        k_out_group_base = b.mul(group_idx, c_kpg)
        c_group_base = b.mul(group_idx, b.const_i32(p.cpg))
    else:
        c_kpg = None
        k_out_group_base = None
        c_group_base = None

    # Precompute tiling divisors (runtime, from record).
    # k_sub decomposition order: [ydot, xdot, k_out] (k_out innermost, CK-compatible).
    # k_sub = ydot * XDotSlice * K + xdot * K + k_out
    # Consecutive k_sub → consecutive k_out → contiguous in dY (NHWK, stride-1 in K).
    hw_tilde = b.mul(rec_h_tilde_slice, rec_w_tilde_slice)

    # LDS allocation.
    lds_layout = spec.effective_lds_layout()
    A_smem = b.smem_alloc(
        ir_dtype_a, lds_layout.storage_shape(block_m), name_hint="A_smem"
    )
    B_smem = b.smem_alloc(
        ir_dtype_b, lds_layout.storage_shape(block_n), name_hint="B_smem"
    )

    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n
    k_atoms = spec.k_atoms_per_tile_k

    acc_init = b.zero_vec_f32(c_per_lane)
    accs = [
        (f"acc_m{mi}_n{ni}", acc_init) for mi in range(mfmas_m) for ni in range(mfmas_n)
    ]

    threads = spec.block_size
    # Per-group vector sizes: dY (NHWK) vectorises along the per-group output-channel
    # run kpg (== K when ungrouped); B (W, KYXC) vectorises along the per-group input
    # channel run cpg (== C when ungrouped).  Using (cpg, kpg) keeps groups==1
    # byte-identical while making grouped loads respect the per-group extents.
    _def_vec_a, _def_vec_b, _ = DgradConvSpec.default_vector_sizes(
        p.cpg, p.kpg, spec.data.dtype_a
    )
    # load_vec_a: k_out is innermost in k_dg → consecutive k_sub → consecutive k_out
    # → contiguous in dY (NHWK, last dim K).  Condition: kpg % load_vec_a == 0.
    # For split_k > 1 the slice boundary may not align to K_conv so use 1 there.
    if spec.split_k <= 1:
        _safe_vec_a = _choose_load_vec_for(
            block_m, block_n, block_k, threads, spec.data.dtype_a
        )
        _safe_vec_a = min(_def_vec_a, _safe_vec_a)
        load_vec_a = (
            spec.vector_size_a if spec.vector_size_a is not None else _safe_vec_a
        )
    else:
        load_vec_a = 1
    # load_vec_b: B (W, KYXC) — the GEMM row axis is N_dg = c (input channels), which
    # is the stride-1 axis of KYXC.  Vectorise along the free (row) axis and transpose
    # into the row-major (N, K) LDS tile on store (CoalescedTileLoader vector_axis="row"),
    # exactly as wgrad does for its B (X, NHWC) operand.  Condition: cpg % load_vec_b
    # == 0 (cpg == C when ungrouped); the group base g*cpg is a multiple of cpg.
    _vb = CoalescedTileLoader.choose_vec(
        tile_rows=block_n,
        tile_cols=block_k,
        block_size=threads,
        max_vec=_def_vec_b,
        vector_axis="row",
    )
    if spec.vector_size_b is not None:
        load_vec_b = spec.vector_size_b
        axis_b = "row" if load_vec_b > 1 else "col"
    elif _vb > 1:
        load_vec_b = _vb
        axis_b = "row"
    else:
        load_vec_b = 1
        axis_b = "col"

    # Buffer resources for A (dY), B (W), D (dX).
    dy_buf_rsrc = make_buffer_resource(b, dY, num_bytes=dY_bytes)
    w_buf_rsrc = make_buffer_resource(b, W, num_bytes=W_bytes)
    dx_buf_rsrc = make_buffer_resource(b, dX, num_bytes=dX_bytes)
    dy_rsrc = dy_buf_rsrc.rsrc
    w_rsrc = w_buf_rsrc.rsrc
    dx_rsrc = dx_buf_rsrc.rsrc

    k_off_capture: List[Optional[Value]] = [None]

    # ---- Runtime offset descriptors ----
    # These closures compute (byte_offset, valid) using runtime arithmetic
    # instead of the TensorDescriptor transform system.

    def dy_descriptor(b_: IRBuilder, row: Value, col: Value):
        """A (dY, NHWK) offset: (m_sub_local, k_sub_local) → element offset."""
        m_sub = b_.add(block_m_off_v, row)
        k_sub = b_.add(k_off_capture[0], col)

        # Decompose k_sub → (ydot, xdot, k_out)  [k_out innermost, CK-compatible]
        # k_sub = ydot * xdot_slice * kpg + xdot * kpg + k_out.  The reduction is
        # per-group, so the decode divisor is kpg (== K when ungrouped) and k_out
        # is group-local in [0, kpg); the absolute NHWK channel is g*kpg + k_out.
        # Consecutive k_sub → consecutive k_out → contiguous in dY (NHWK, last dim
        # K) within one group. Condition for vector loads: kpg % load_vec_a == 0.
        _kdiv = c_kpg if grouped else c_K
        k_out = b_.mod(k_sub, _kdiv)
        yx_rem = b_.div(k_sub, _kdiv)
        ydot = b_.div(yx_rem, rec_x_dot_slice)
        xdot = b_.mod(yx_rem, rec_x_dot_slice)
        k_out_abs = b_.add(k_out, k_out_group_base) if grouped else k_out

        # Decompose m_sub → (n, htilde_local, wtilde_local)
        n = b_.div(m_sub, hw_tilde)
        m_rem = b_.mod(m_sub, hw_tilde)
        htl = b_.div(m_rem, rec_w_tilde_slice)
        wtl = b_.mod(m_rem, rec_w_tilde_slice)

        # ho = htl + h_tilde_slice_begin + ydot * a_embed_h_coeff
        ho = b_.add(
            b_.add(htl, rec_h_tilde_slice_begin), b_.mul(ydot, rec_a_embed_h_coeff)
        )
        wo = b_.add(
            b_.add(wtl, rec_w_tilde_slice_begin), b_.mul(xdot, rec_a_embed_w_coeff)
        )

        # Bounds check: 0 <= ho < Ho, 0 <= wo < Wo, k_out < K
        ho_ok = b_.land(b_.cmp_ge(ho, c0), b_.cmp_lt(ho, c_Ho))
        wo_ok = b_.land(b_.cmp_ge(wo, c0), b_.cmp_lt(wo, c_Wo))
        k_ok = b_.cmp_lt(k_out, _kdiv)
        valid = b_.land(b_.land(ho_ok, wo_ok), k_ok)

        # NHWK linear offset: (n * Ho + ho) * Wo * K + wo * K + k_out_abs
        # (the K stride stays total-K; only the channel index carries g*kpg).
        offset = b_.add(
            b_.mul(b_.add(b_.mul(n, c_Ho), ho), b_.mul(c_Wo, c_K)),
            b_.add(b_.mul(wo, c_K), k_out_abs),
        )
        safe_offset = b_.select(valid, offset, b_.const_i32(0))
        return safe_offset, valid

    def w_descriptor(b_: IRBuilder, row: Value, col: Value):
        """B (W, KYXC) offset: (c_local, k_sub_local) → element offset."""
        c_val = b_.add(block_n_off_v, row)
        k_sub = b_.add(k_off_capture[0], col)

        # Same k_out-innermost decomposition as dy_descriptor (must match).
        # c (row axis) is stride-1 in KYXC; vectorised loads along c use
        # vector_axis="row".  The reduction is per-group, so the decode divisor is
        # kpg (== K when ungrouped) and k_out is group-local in [0, kpg).
        _kdiv = c_kpg if grouped else c_K
        k_out = b_.mod(k_sub, _kdiv)
        yx_rem = b_.div(k_sub, _kdiv)
        ydot = b_.div(yx_rem, rec_x_dot_slice)
        xdot = b_.mod(yx_rem, rec_x_dot_slice)
        k_out_abs = b_.add(k_out, k_out_group_base) if grouped else k_out

        # y = ydot * b_y_stride + b_y_offset
        y = b_.add(b_.mul(ydot, rec_b_y_stride), rec_b_y_offset)
        x = b_.add(b_.mul(xdot, rec_b_x_stride), rec_b_x_offset)

        # Bounds check: y < Y, x < X, k_out < kpg (group-local)
        valid = b_.land(
            b_.land(b_.cmp_lt(y, c_Y), b_.cmp_lt(x, c_X)), b_.cmp_lt(k_out, _kdiv)
        )

        # Weight is stored PER-GROUP packed: KYXC == [K, Y, X, cpg] (mirrors the
        # forward make_b_descriptor and the wgrad packed dW output).  The group is
        # carried entirely by the absolute output channel k_out_abs = g*kpg+k_out
        # (different k rows hold different groups' slabs); the input channel is
        # group-local c_val in [0, cpg) and the last-dim stride is cpg.  Ungrouped:
        # cpg == C and k_out_abs == k_out, so this is byte-identical.
        #   offset = ((k_out_abs * Y + y) * X + x) * cpg + c_local
        _cstride = c_dg_N if grouped else c_C
        offset = b_.add(
            b_.mul(b_.add(b_.mul(b_.add(b_.mul(k_out_abs, c_Y), y), c_X), x), _cstride),
            c_val,
        )
        safe_offset = b_.select(valid, offset, b_.const_i32(0))
        return safe_offset, valid

    # Loaders (sync only for now — no async_dma in tilde kernel).
    a_sync_loader = CoalescedTileLoader(
        tile_rows=block_m,
        tile_cols=block_k,
        block_size=threads,
        load_vec=load_vec_a,
        elem_dtype=ir_dtype_a,
    )
    b_sync_loader = CoalescedTileLoader(
        tile_rows=block_n,
        tile_cols=block_k,
        block_size=threads,
        load_vec=load_vec_b,
        elem_dtype=ir_dtype_b,
        vector_axis=axis_b,
    )
    if spec.pipeline == "wavelet":
        a_wavelet_loader, b_wavelet_loader = build_wavelet_loaders(
            num_load_waves=spec.num_load_waves,
            wave_size=spec.wave_size,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            load_vec_a=load_vec_a,
            load_vec_b=load_vec_b,
            ir_dtype_a=ir_dtype_a,
            ir_dtype_b=ir_dtype_b,
            vector_axis_b=axis_b,
        )
    else:
        a_wavelet_loader = None
        b_wavelet_loader = None

    schedule = SchedulePolicy.for_pipeline(spec.pipeline)
    schedule.emit_prologue(b)

    def emit_load_phase(k_off: Value, A_dst: Value, B_dst: Value) -> None:
        k_off_capture[0] = k_off
        a_sync_loader.load(
            b, tid=tid, smem_dst=A_dst, descriptor=dy_descriptor, rsrc=dy_rsrc
        )
        b_sync_loader.load(
            b, tid=tid, smem_dst=B_dst, descriptor=w_descriptor, rsrc=w_rsrc
        )

    def emit_mfma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value]
    ) -> List[Value]:
        if op.family == "wmma":
            a_map = op.a_layout()
            b_map = op.b_layout()
            a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
            b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0)
            warp_m_off = grid.warp_m_off(b)
            warp_n_off = grid.warp_n_off(b)
            new_accs: List[Value] = list(iter_vars)
            for kk in range(k_atoms):
                k_tile_base = b.const_i32(kk * spec.warp_tile_k)
                a_rows = []
                for mi in range(mfmas_m):
                    atom_row = b.add(warp_m_off, b.const_i32(mi * spec.warp_tile_m))
                    a_rows.append(
                        _emit_frag_smem_load(
                            b,
                            A_src,
                            a_row_in_atom,
                            a_k_in_atom,
                            atom_row,
                            k_tile_base,
                            a_per_lane,
                            smem_dtype=_smem_dtype,
                        )
                    )
                b_cols = []
                for ni in range(mfmas_n):
                    atom_row = b.add(warp_n_off, b.const_i32(ni * spec.warp_tile_n))
                    b_cols.append(
                        _emit_frag_smem_load(
                            b,
                            B_src,
                            b_col_in_atom,
                            b_k_in_atom,
                            atom_row,
                            k_tile_base,
                            b_per_lane,
                            smem_dtype=_smem_dtype,
                        )
                    )
                flat = 0
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        new_accs[flat] = b.mma(
                            op, a_rows[mi], b_cols[ni], new_accs[flat]
                        )
                        flat += 1
            return new_accs

        decoded = decode_mfma_lanes(b, atom, lane)
        m_in_atom = decoded.m_in_atom
        n_in_atom = decoded.n_in_atom
        k_blk = decoded.k_blk
        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)
        new_accs: List[Value] = list(iter_vars)
        for kk in range(k_atoms):
            col_base = b.add(
                b.mul(k_blk, b.const_i32(a_per_lane)),
                b.const_i32(kk * spec.warp_tile_k),
            )
            a_rows = []
            for mi in range(mfmas_m):
                a_row = b.add(
                    warp_m_off,
                    b.add(b.const_i32(mi * spec.warp_tile_m), m_in_atom),
                )
                a_rows.append(
                    _emit_smem_load(
                        b, A_src, a_row, col_base, a_per_lane, smem_dtype=_smem_dtype
                    )
                )
            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off,
                    b.add(b.const_i32(ni * spec.warp_tile_n), n_in_atom),
                )
                b_cols.append(
                    _emit_smem_load(
                        b, B_src, b_row, col_base, b_per_lane, smem_dtype=_smem_dtype
                    )
                )
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    acc = _emit_mfma(b, atom, a_rows[mi], b_cols[ni], new_accs[flat])
                    new_accs[flat] = acc
                    flat += 1
            schedule.emit_after_mfma_step(
                b,
                ds_read_count=mfmas_m + mfmas_n,
                mfma_count=mfmas_m * mfmas_n,
            )
        return new_accs

    # ---- epilogue dispatch helper (shared by wavelet and standard paths) ----
    def _dispatch_dgrad_epilogue(final_accs):
        _tilde_kwargs = dict(
            bounds_m=rec_gemm_m,
            bounds_n=c_dg_N,
            hw_tilde=hw_tilde,
            w_tilde_slice=rec_w_tilde_slice,
            d_h_stride=rec_d_h_stride,
            d_h_offset=rec_d_h_offset,
            d_w_stride=rec_d_w_stride,
            d_w_offset=rec_d_w_offset,
            c_Hi=c_Hi,
            c_Wi=c_Wi,
            c_C=c_C,
            c_group_base=c_group_base,
        )
        use_cshuffle = spec.epilogue == "cshuffle"
        is_wmma = op.family == "wmma"

        if spec.split_k > 1:
            _emit_dgrad_tilde_atomic_epilogue(
                b,
                spec,
                atom,
                final_accs,
                warp_m_idx,
                warp_n_idx,
                lane,
                block_m_off_v,
                block_n_off_v,
                dX,
                c_per_lane,
                rec_gemm_m,
                c_dg_N,
                rec_h_tilde_slice,
                rec_w_tilde_slice,
                rec_d_h_stride,
                rec_d_h_offset,
                rec_d_w_stride,
                rec_d_w_offset,
                c_Hi,
                c_Wi,
                c_C,
                c_group_base=c_group_base,
            )
        elif not spec.is_strided:
            if is_wmma:
                if use_cshuffle:
                    _emit_dgrad_cshuffle_epilogue_wmma(
                        b,
                        spec,
                        op,
                        final_accs,
                        grid,
                        dx_rsrc,
                        c_group_base=c_group_base,
                    )
                else:
                    _emit_dgrad_direct_epilogue_wmma(
                        b,
                        spec,
                        op,
                        final_accs,
                        warp_m_idx,
                        warp_n_idx,
                        lane,
                        block_m_off_v,
                        block_n_off_v,
                        dx_rsrc,
                        c0,
                        c_group_base=c_group_base,
                    )
            elif use_cshuffle:
                _emit_dgrad_cshuffle_epilogue(
                    b, spec, final_accs, grid, dx_rsrc, c_group_base=c_group_base
                )
            else:
                _emit_dgrad_direct_epilogue(
                    b, spec, final_accs, grid, dx_rsrc, c_group_base=c_group_base
                )
        elif is_wmma:
            if use_cshuffle:
                _emit_dgrad_tilde_cshuffle_epilogue(
                    b, spec, atom, grid, final_accs, dx_rsrc, op=op, **_tilde_kwargs
                )
            else:
                _emit_dgrad_tilde_direct_epilogue_wmma(
                    b,
                    spec,
                    op,
                    final_accs,
                    warp_m_idx,
                    warp_n_idx,
                    lane,
                    block_m_off_v,
                    block_n_off_v,
                    dx_rsrc,
                    c0,
                    **_tilde_kwargs,
                )
        else:
            assert atom is not None
            if use_cshuffle:
                _emit_dgrad_tilde_cshuffle_epilogue(
                    b, spec, atom, grid, final_accs, dx_rsrc, **_tilde_kwargs
                )
            else:
                _emit_dgrad_tilde_direct_epilogue(
                    b, spec, atom, grid, final_accs, dx_rsrc, **_tilde_kwargs
                )

    # ---- K loop ----
    if spec.pipeline == "wavelet":
        # Wavelet load/math wave specialization (gfx1250/WMMA only).
        # K_iters is a compile-time constant. dgrad uses dg_K_padded() to
        # keep it uniform across sub-GEMMs (tilde decomposition may vary k_hi
        # per sub-GEMM at runtime; wavelet unrolls at Python time so it uses the
        # worst-case padded K — OOB loads are clamped to 0 by the buffer resource).
        slice_k = (
            spec.dg_K_padded()
            if spec.split_k <= 1
            else (spec.dg_K_padded() // spec.split_k)
        )
        K_iters = (slice_k + block_k - 1) // block_k

        n_math_warps = spec.warp_m * spec.warp_n
        b.kernel.attrs["max_workgroup_size"] = spec.launch_block_size
        _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
        _epi_barriers = compute_wavelet_epi_barriers(spec.epilogue, _no_alias)

        def _dgrad_wavelet_epilogue(final_accs_in):
            fa = _apply_accumulator_epilogue(b, spec.acc_epilogue, final_accs_in)
            _dispatch_dgrad_epilogue(fa)

        emit_wavelet_kloop(
            b=b,
            warp_id=warp_id,
            tid=tid,
            n_math_warps=n_math_warps,
            math_block_size=spec.block_size,
            K_iters=K_iters,
            block_k=block_k,
            k_lo=k_lo,
            A_smem=A_smem,
            B_smem=B_smem,
            a_wavelet_loader=a_wavelet_loader,
            b_wavelet_loader=b_wavelet_loader,
            a_descriptor=dy_descriptor,
            b_descriptor=w_descriptor,
            a_rsrc=dy_rsrc,
            b_rsrc=w_rsrc,
            k_off_capture=k_off_capture,
            accs=accs,
            emit_mfma_phase=emit_mfma_phase,
            emit_epilogue_fn=_dgrad_wavelet_epilogue,
            epi_barriers=_epi_barriers,
        )
        return b.kernel

    for_op = b.scf_for_iter(k_lo, k_hi, c_block_k, accs, iv_name="k0")
    with for_op as (k0, iter_vars):
        emit_load_phase(k0, A_smem, B_smem)
        b.sync()
        new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)
        b.sync()
        b.scf_yield(*new_accs)
    final_accs = for_op.results

    # ---- epilogue ----
    # Epilogue dispatch.  Two independent axes:
    #   split_k:    >1 → atomic,  =1 → direct (tilde guarantees disjoint writes)
    #   is_strided: False → stride-1 descriptor,  True → runtime tilde addr_fn
    # Within each leaf: wmma vs mfma selects the accumulator-layout helper,
    # and epilogue="cshuffle" vs "default" selects LDS-staged vs scalar stores.
    final_accs = _apply_accumulator_epilogue(b, spec.acc_epilogue, final_accs)
    _dispatch_dgrad_epilogue(final_accs)

    return b.kernel


def _emit_dgrad_tilde_atomic_epilogue(
    b: IRBuilder,
    spec: DgradConvSpec,
    atom: MfmaAtom,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    dx_ptr: Value,
    c_per_lane: int,
    gemm_m: Value,
    gemm_n: Value,
    h_tilde_slice: Value,
    w_tilde_slice: Value,
    d_h_stride: Value,
    d_h_offset: Value,
    d_w_stride: Value,
    d_w_offset: Value,
    c_Hi: Value,
    c_Wi: Value,
    c_C: Value,
    c_group_base: Optional[Value] = None,
) -> None:
    """Atomic-add epilogue for the tilde kernel.

    Computes the dX NHWC offset at runtime using the sub-GEMM record's
    d_h/d_w stride/offset fields, then issues atomic-add.
    """
    from ...helpers.atoms import c_warp_params, make_c_warp_dstr_encoding
    from ...helpers.distribution import make_static_tile_distribution

    dtype_d = spec.data.dtype_d
    p = spec.problem
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))
    block_warp_m_off = b.add(block_m_off, warp_m_off)
    block_warp_n_off = b.add(block_n_off, warp_n_off)

    _, __, kc_m1, kc_nlane = c_warp_params(atom)
    c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom))

    c_nlane = b.const_i32(kc_nlane)
    n_in_atom = b.mod(lane, c_nlane)
    m_blk = b.div(lane, c_nlane)
    p_lane = [m_blk, n_in_atom]

    rows: List[Value] = []
    cols: List[Value] = []
    for i in range(c_per_lane):
        ys = [b.const_i32(i // kc_m1), b.const_i32(i % kc_m1)]
        x_row, x_col = c_dist.calculate_x(b, ys=ys, ps=[p_lane])
        rows.append(x_row)
        cols.append(x_col)

    c0 = b.const_i32(0)
    hw_tilde = b.mul(h_tilde_slice, w_tilde_slice)

    flat = 0
    for mi in range(mfmas_m):
        atom_m_base = b.add(block_warp_m_off, b.const_i32(mi * spec.warp_tile_m))
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            atom_n_base = b.add(block_warp_n_off, b.const_i32(ni * spec.warp_tile_n))

            for i in range(c_per_lane):
                c_m = b.add(atom_m_base, rows[i])
                c_n = b.add(atom_n_base, cols[i])
                # Absolute dX channel = g*cpg + c_n (group base is even, so the
                # packed <2 x dtype> pair parity below is unchanged); bounds use
                # the group-local c_n (< gemm_n = cpg).
                c_n_off = b.add(c_n, c_group_base) if c_group_base is not None else c_n

                # Decompose c_m (the M index) into (n, htl, wtl) at runtime
                n_val = b.div(c_m, hw_tilde)
                m_rem = b.mod(c_m, hw_tilde)
                htl = b.div(m_rem, w_tilde_slice)
                wtl = b.mod(m_rem, w_tilde_slice)

                # Compute hi = htl * d_h_stride + d_h_offset
                hi = b.add(b.mul(htl, d_h_stride), d_h_offset)
                wi = b.add(b.mul(wtl, d_w_stride), d_w_offset)

                # Bounds: 0 <= hi < Hi, 0 <= wi < Wi, c_m < gemm_m, c_n < gemm_n
                m_ok = b.cmp_lt(c_m, gemm_m)
                n_ok = b.cmp_lt(c_n, gemm_n)
                hi_ok = b.land(b.cmp_ge(hi, c0), b.cmp_lt(hi, c_Hi))
                wi_ok = b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, c_Wi))
                ok = b.land(b.land(m_ok, n_ok), b.land(hi_ok, wi_ok))

                # NHWC offset: ((n * Hi + hi) * Wi + wi) * C + c
                dx_offset = b.add(
                    b.mul(b.add(b.mul(b.add(b.mul(n_val, c_Hi), hi), c_Wi), wi), c_C),
                    c_n_off,
                )

                val_f32 = b.vec_extract(acc, i)
                with b.scf_if(ok):
                    if dtype_d == "fp32":
                        b.global_atomic_add(dx_ptr, dx_offset, val_f32)
                    elif dtype_d == "bf16":
                        val_cvt = b.trunc_f32_to_bf16(val_f32)
                        zero = b.trunc_f32_to_bf16(b.const_f32(0.0))
                        c_n_is_odd = b.mod(c_n_off, b.const_i32(2))
                        is_odd = b.cmp_ne(c_n_is_odd, b.const_i32(0))
                        c_n_even = b.sub(c_n_off, c_n_is_odd)
                        off_even = b.add(
                            b.mul(
                                b.add(b.mul(b.add(b.mul(n_val, c_Hi), hi), c_Wi), wi),
                                c_C,
                            ),
                            c_n_even,
                        )
                        v_even = b.select(is_odd, zero, val_cvt)
                        v_odd = b.select(is_odd, val_cvt, zero)
                        vec = b.vec_pack([v_even, v_odd], val_cvt.type)
                        b.global_atomic_add_pk_bf16(dx_ptr, off_even, vec)
                    else:
                        val_cvt = b.trunc_f32_to_f16(val_f32)
                        zero = b.trunc_f32_to_f16(b.const_f32(0.0))
                        c_n_is_odd = b.mod(c_n_off, b.const_i32(2))
                        is_odd = b.cmp_ne(c_n_is_odd, b.const_i32(0))
                        c_n_even = b.sub(c_n_off, c_n_is_odd)
                        off_even = b.add(
                            b.mul(
                                b.add(b.mul(b.add(b.mul(n_val, c_Hi), hi), c_Wi), wi),
                                c_C,
                            ),
                            c_n_even,
                        )
                        v_even = b.select(is_odd, zero, val_cvt)
                        v_odd = b.select(is_odd, val_cvt, zero)
                        vec = b.vec_pack([v_even, v_odd], val_cvt.type)
                        b.global_atomic_add_pk_f16(dx_ptr, off_even, vec)


def _emit_dgrad_direct_epilogue(
    b: IRBuilder,
    spec: DgradConvSpec,
    accs: Sequence[Value],
    grid: WarpGrid,
    dx_rsrc: Value,
    c_group_base: Optional[Value] = None,
) -> None:
    """Per-lane scalar store to dX via the input-gradient descriptor."""
    p = spec.problem
    dX_desc = make_dgrad_dx_descriptor(p, dtype=spec.data.dtype_d)

    def dx_addr(b_: IRBuilder, m_val: Value, n_val: Value):
        # n_val is the group-local input channel (< cpg); the absolute NHWC
        # channel is g*cpg + n_val.  Ungrouped: c_group_base is None → c = n_val.
        c = b_.add(n_val, c_group_base) if c_group_base is not None else n_val
        return dX_desc.offset(b_, m=m_val, c=c)

    DirectEpilogue(atom=spec.atom, grid=grid, out_dtype=spec.data.dtype_d).store(
        b,
        accs=accs,
        addr_fn=dx_addr,
        d_rsrc=dx_rsrc,
        bounds=(b.const_i32(_dg_M(p)), b.const_i32(_dg_N(p))),
    )


def _emit_dgrad_direct_epilogue_wmma(
    b: IRBuilder,
    spec: DgradConvSpec,
    op,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    dx_rsrc: Value,
    c0: Value,
    c_group_base: Optional[Value] = None,
) -> None:
    """Per-lane store for the WMMA (gfx1151) accumulator layout into dX."""
    p = spec.problem
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))

    c_M = b.const_i32(_dg_M(p))
    c_N = b.const_i32(_dg_N(p))
    dX_desc = make_dgrad_dx_descriptor(p, dtype=spec.data.dtype_d)
    c_map = op.c_layout()
    _fp32_out = spec.data.dtype_d == "fp32"
    _bf16_out = spec.data.dtype_d == "bf16"
    _elem_bytes = 4 if _fp32_out else 2

    flat = 0
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            atom_m_off = b.add(
                b.add(block_m_off, warp_m_off),
                b.const_i32(mi * spec.warp_tile_m),
            )
            atom_n_off = b.add(
                b.add(block_n_off, warp_n_off),
                b.const_i32(ni * spec.warp_tile_n),
            )
            for i in range(op.c_frag_len):
                row_off, col_off = c_map.coord(b, lane, i)
                m_val = b.add(atom_m_off, row_off)
                n_val = b.add(atom_n_off, col_off)
                m_ok = b.cmp_lt(m_val, c_M)
                n_ok = b.cmp_lt(n_val, c_N)
                ok = b.land(m_ok, n_ok)

                v_f32 = b.vec_extract(acc, i)
                _c = b.add(n_val, c_group_base) if c_group_base is not None else n_val
                dx_off_elems, _ = dX_desc.offset(b, m=m_val, c=_c)
                dx_off_bytes = b.mul(dx_off_elems, b.const_i32(_elem_bytes))
                safe_off = b.select(ok, dx_off_bytes, b.const_i32((1 << 31) - 1))
                if _fp32_out:
                    b.buffer_store_f32(dx_rsrc, safe_off, c0, v_f32)
                elif _bf16_out:
                    b.buffer_store_bf16(
                        dx_rsrc, safe_off, c0, b.trunc_f32_to_bf16(v_f32)
                    )
                else:
                    b.buffer_store_f16(dx_rsrc, safe_off, c0, b.trunc_f32_to_f16(v_f32))


def _emit_dgrad_tilde_direct_epilogue_wmma(
    b: IRBuilder,
    spec: DgradConvSpec,
    op,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    dx_rsrc: Value,
    c0: Value,
    *,
    bounds_m: Value,
    bounds_n: Value,
    hw_tilde: Value,
    w_tilde_slice: Value,
    d_h_stride: Value,
    d_h_offset: Value,
    d_w_stride: Value,
    d_w_offset: Value,
    c_Hi: Value,
    c_Wi: Value,
    c_C: Value,
    c_group_base: Optional[Value] = None,
) -> None:
    """Per-lane tilde store for WMMA (gfx1151) accumulator layout into dX.

    Mirrors _emit_dgrad_direct_epilogue_wmma but uses the tilde address
    mapping instead of the stride=1 descriptor.  Safe for split_k=1 because
    the tilde decomposition guarantees disjoint writes.
    """
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))

    c_map = op.c_layout()
    _fp32_out = spec.data.dtype_d == "fp32"
    _bf16_out = spec.data.dtype_d == "bf16"
    _elem_bytes = 4 if _fp32_out else 2

    flat = 0
    for mi in range(mfmas_m):
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            atom_m_off = b.add(
                b.add(block_m_off, warp_m_off), b.const_i32(mi * spec.warp_tile_m)
            )
            atom_n_off = b.add(
                b.add(block_n_off, warp_n_off), b.const_i32(ni * spec.warp_tile_n)
            )
            for i in range(op.c_frag_len):
                row_off, col_off = c_map.coord(b, lane, i)
                m_val = b.add(atom_m_off, row_off)
                n_val = b.add(atom_n_off, col_off)

                # Tilde M decomposition: m_val → (n_batch, htl, wtl)
                n_batch = b.div(m_val, hw_tilde)
                m_rem = b.mod(m_val, hw_tilde)
                htl = b.div(m_rem, w_tilde_slice)
                wtl = b.mod(m_rem, w_tilde_slice)
                hi = b.add(b.mul(htl, d_h_stride), d_h_offset)
                wi = b.add(b.mul(wtl, d_w_stride), d_w_offset)

                m_ok = b.cmp_lt(m_val, bounds_m)
                n_ok = b.cmp_lt(n_val, bounds_n)
                hi_ok = b.land(b.cmp_ge(hi, c0), b.cmp_lt(hi, c_Hi))
                wi_ok = b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, c_Wi))
                ok = b.land(b.land(m_ok, n_ok), b.land(hi_ok, wi_ok))

                # NHWC offset: ((n_batch*Hi + hi)*Wi + wi)*C + c_abs
                _c = b.add(n_val, c_group_base) if c_group_base is not None else n_val
                _o = b.add(
                    b.mul(b.add(b.mul(b.add(b.mul(n_batch, c_Hi), hi), c_Wi), wi), c_C),
                    _c,
                )
                dx_off_bytes = b.mul(_o, b.const_i32(_elem_bytes))
                safe_off = b.select(ok, dx_off_bytes, b.const_i32((1 << 31) - 1))

                v_f32 = b.vec_extract(acc, i)
                if _fp32_out:
                    b.buffer_store_f32(dx_rsrc, safe_off, c0, v_f32)
                elif _bf16_out:
                    b.buffer_store_bf16(
                        dx_rsrc, safe_off, c0, b.trunc_f32_to_bf16(v_f32)
                    )
                else:
                    b.buffer_store_f16(dx_rsrc, safe_off, c0, b.trunc_f32_to_f16(v_f32))


def _dgrad_store_vec(spec: DgradConvSpec) -> int:
    """Deduce max_store_vec for dX (last dim = C) mirroring default_vector_sizes."""
    if spec.vector_size_c is not None:
        return spec.vector_size_c
    # dX (NHWC) channel run within one conv group is cpg (== C when ungrouped).
    _, __, vec_c = DgradConvSpec.default_vector_sizes(
        spec.problem.cpg, spec.problem.kpg, spec.data.dtype_d
    )
    return vec_c


def _dgrad_stride1_dx_addr(
    b: IRBuilder,
    spec: DgradConvSpec,
    c_group_base: Optional[Value] = None,
) -> tuple:
    """Return (addr_fn, bounds) for the stride-1 dX descriptor."""
    p = spec.problem
    dX_desc = make_dgrad_dx_descriptor(p, dtype=spec.data.dtype_d)

    def dx_addr(b_: IRBuilder, m_val: Value, n_val: Value):
        c = b_.add(n_val, c_group_base) if c_group_base is not None else n_val
        return dX_desc.offset(b_, m=m_val, c=c)

    bounds = (b.const_i32(_dg_M(p)), b.const_i32(_dg_N(p)))
    return dx_addr, bounds


def _emit_dgrad_cshuffle_epilogue_wmma(
    b: IRBuilder,
    spec: DgradConvSpec,
    op,
    accs: Sequence[Value],
    grid: WarpGrid,
    dx_rsrc: Value,
    c_group_base: Optional[Value] = None,
) -> None:
    """WMMA cshuffle epilogue for stride=1 dX (uses from_grid_op)."""
    _war_barriers = 2 if spec.pipeline == "wavelet" else 1
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    dx_addr, bounds = _dgrad_stride1_dx_addr(b, spec, c_group_base)
    _epi = CShuffleEpilogue.from_grid_op(
        op=op,
        grid=grid,
        max_store_vec=_dgrad_store_vec(spec),
        out_dtype=spec.data.dtype_d,
        no_alias=_no_alias,
    )
    _epi = dc_replace(_epi, war_barriers=_war_barriers)
    _epi.store(b, accs=accs, addr_fn=dx_addr, d_rsrc=dx_rsrc, bounds=bounds)


def _emit_dgrad_cshuffle_epilogue(
    b: IRBuilder,
    spec: DgradConvSpec,
    accs: Sequence[Value],
    grid: WarpGrid,
    dx_rsrc: Value,
    c_group_base: Optional[Value] = None,
) -> None:
    """MFMA cshuffle epilogue for stride=1 dX."""
    _war_barriers = 2 if spec.pipeline == "wavelet" else 1
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    dx_addr, bounds = _dgrad_stride1_dx_addr(b, spec, c_group_base)
    _epi = CShuffleEpilogue.from_grid(
        atom=spec.atom,
        grid=grid,
        max_store_vec=_dgrad_store_vec(spec),
        out_dtype=spec.data.dtype_d,
        no_alias=_no_alias,
    )
    _epi = dc_replace(_epi, war_barriers=_war_barriers)
    _epi.store(
        b,
        accs=accs,
        addr_fn=dx_addr,
        d_rsrc=dx_rsrc,
        bounds=bounds,
    )


def _tilde_dx_addr_fn(
    b: IRBuilder,
    m_global: Value,
    n_global: Value,
    hw_tilde: Value,
    w_tilde_slice: Value,
    d_h_stride: Value,
    d_h_offset: Value,
    d_w_stride: Value,
    d_w_offset: Value,
    c_Hi: Value,
    c_Wi: Value,
    c_C: Value,
    c_group_base: Optional[Value] = None,
):
    """Compute NHWC element offset + hi/wi validity for the tilde epilogues.

    Maps (m_global, n_global) -> (element_offset, hw_valid) where:
      m_global decomposes as (n_val, htl, wtl) via hw_tilde / w_tilde_slice
      hi = htl * d_h_stride + d_h_offset
      wi = wtl * d_w_stride + d_w_offset
      offset = ((n_val * Hi + hi) * Wi + wi) * C + n_global

    For fixed m_global the N dimension (channel) is contiguous in dX memory
    (stride 1 in C), enabling vector stores along N via the cshuffle epilogue.

    Returns (offset, hw_valid) — bounds_m / bounds_n (m < gemm_m, n < C) are
    handled by the epilogue caller via its `bounds` argument.
    """
    c0 = b.const_i32(0)

    n_val = b.div(m_global, hw_tilde)
    m_rem = b.mod(m_global, hw_tilde)
    htl = b.div(m_rem, w_tilde_slice)
    wtl = b.mod(m_rem, w_tilde_slice)

    hi = b.add(b.mul(htl, d_h_stride), d_h_offset)
    wi = b.add(b.mul(wtl, d_w_stride), d_w_offset)

    hi_ok = b.land(b.cmp_ge(hi, c0), b.cmp_lt(hi, c_Hi))
    wi_ok = b.land(b.cmp_ge(wi, c0), b.cmp_lt(wi, c_Wi))
    hw_ok = b.land(hi_ok, wi_ok)

    _o0 = b.mul(n_val, c_Hi)
    _o1 = b.add(_o0, hi)
    _o2 = b.mul(_o1, c_Wi)
    _o3 = b.add(_o2, wi)
    _o4 = b.mul(_o3, c_C)
    # Absolute NHWC channel = g*cpg + n_global (ungrouped: c_group_base is None).
    _n = b.add(n_global, c_group_base) if c_group_base is not None else n_global
    offset = b.add(_o4, _n)

    # Return raw element offset (no sentinel select here).
    # DirectEpilogue/CShuffleEpilogue apply the sentinel via hw_ok (the valid flag),
    # matching stride-1 (dX_desc.offset) which also returns raw offset + valid.
    return offset, hw_ok


def _emit_dgrad_tilde_direct_epilogue(
    b: IRBuilder,
    spec: DgradConvSpec,
    atom: MfmaAtom,
    grid: WarpGrid,
    accs: Sequence[Value],
    dx_rsrc: Value,
    *,
    bounds_m: Value,
    bounds_n: Value,
    hw_tilde: Value,
    w_tilde_slice: Value,
    d_h_stride: Value,
    d_h_offset: Value,
    d_w_stride: Value,
    d_w_offset: Value,
    c_Hi: Value,
    c_Wi: Value,
    c_C: Value,
    c_group_base: Optional[Value] = None,
) -> None:
    """Scalar (per-element) direct store for the tilde non-atomic path.

    Mirrors _emit_dgrad_direct_epilogue for the strided (tilde) case.
    """

    def dx_addr(b_: IRBuilder, m_val: Value, n_val: Value):
        offset, hw_ok = _tilde_dx_addr_fn(
            b_,
            m_val,
            n_val,
            hw_tilde,
            w_tilde_slice,
            d_h_stride,
            d_h_offset,
            d_w_stride,
            d_w_offset,
            c_Hi,
            c_Wi,
            c_C,
            c_group_base,
        )
        return offset, hw_ok

    DirectEpilogue(atom=atom, grid=grid, out_dtype=spec.data.dtype_d).store(
        b,
        accs=accs,
        addr_fn=dx_addr,
        d_rsrc=dx_rsrc,
        bounds=(bounds_m, bounds_n),
    )


def _emit_dgrad_tilde_cshuffle_epilogue(
    b: IRBuilder,
    spec: DgradConvSpec,
    atom: Optional[MfmaAtom],
    grid: WarpGrid,
    accs: Sequence[Value],
    dx_rsrc: Value,
    *,
    op=None,
    bounds_m: Value,
    bounds_n: Value,
    hw_tilde: Value,
    w_tilde_slice: Value,
    d_h_stride: Value,
    d_h_offset: Value,
    d_w_stride: Value,
    d_w_offset: Value,
    c_Hi: Value,
    c_Wi: Value,
    c_C: Value,
    c_group_base: Optional[Value] = None,
) -> None:
    """LDS-staged wide store for the tilde non-atomic path.

    Mirrors _emit_dgrad_cshuffle_epilogue for the strided (tilde) case.
    The N dimension (channel index) is contiguous in dX for fixed m, so
    the cshuffle epilogue can issue vector_size_c-wide buffer stores along N.
    Supports both MFMA (via atom) and WMMA (via op with from_grid_op).
    """

    def dx_addr(b_: IRBuilder, m_val: Value, n_val: Value):
        offset, hw_ok = _tilde_dx_addr_fn(
            b_,
            m_val,
            n_val,
            hw_tilde,
            w_tilde_slice,
            d_h_stride,
            d_h_offset,
            d_w_stride,
            d_w_offset,
            c_Hi,
            c_Wi,
            c_C,
            c_group_base,
        )
        return offset, hw_ok

    _war_barriers = 2 if spec.pipeline == "wavelet" else 1
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    _cshuffle_kwargs = {
        "max_store_vec": _dgrad_store_vec(spec),
        "out_dtype": spec.data.dtype_d,
        "no_alias": _no_alias,
    }
    is_wmma = op is not None and getattr(op, "family", None) == "wmma"
    if is_wmma:
        epi = CShuffleEpilogue.from_grid_op(op=op, grid=grid, **_cshuffle_kwargs)
    else:
        epi = CShuffleEpilogue.from_grid(atom=atom, grid=grid, **_cshuffle_kwargs)
    epi = dc_replace(epi, war_barriers=_war_barriers)
    epi.store(
        b,
        accs=accs,
        addr_fn=dx_addr,
        d_rsrc=dx_rsrc,
        bounds=(bounds_m, bounds_n),
    )
