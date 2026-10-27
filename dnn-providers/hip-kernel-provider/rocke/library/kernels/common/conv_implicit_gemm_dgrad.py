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
from math import gcd as _gcd
from typing import Dict, List, Optional, Sequence, Tuple

from rocke.core.ir import (
    BF16,
    F16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from rocke.helpers.atoms import MfmaAtom, mfma_atom
from rocke.helpers.epilogues import CShuffleEpilogue, DirectEpilogue
from rocke.helpers.geometry import WarpGrid
from rocke.helpers.grid import chiplet_transform_chunked_runtime
from rocke.helpers.layouts import ConvKOuterFragmentReader, LdsLayout
from rocke.helpers.loads import CoalescedTileLoader
from rocke.helpers.magic_division import (
    calculate_magic_numbers,
    do_magic_division_dynamic,
)
from rocke.helpers.mfma_gemm_inner import decode_mfma_lanes
from rocke.helpers.schedule import SchedulePolicy
from rocke.helpers.spec import ceil_div, kernel_name_join
from rocke.helpers.tensor_view import make_buffer_resource
from rocke.helpers.transforms import (
    DynamicTensorDescriptor,
    TensorDescriptor,
    embed,
    pad,
    unmerge_magic,
    unmerge_magic_dynamic,
)
from kernels.common.conv_abi import conv_arg_names
from kernels.common._conv_implicit_gemm_common import (
    ConvAccumulatorEpilogue,
    ConvDataSpec,
    ConvProblem,
    _apply_accumulator_epilogue,
    _choose_load_vec_for,
    coalesced_load_reason,
    vector_width_reason,
    _emit_frag_smem_load,
    _emit_mfma,
    _emit_smem_load,
    _ir_dtype,
    _build_wavelet_loaders,
    compute_wavelet_epi_barriers,
    emit_param_block,
    emit_wavelet_kloop_dynamic,
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


def dgrad_split_k_available(
    p: ConvProblem, dtype_d: str, *, wave_size: int = 64
) -> tuple[bool, str]:
    """Can a dgrad kernel for ``p`` be built with ``split_k > 1``?

    The same rule :meth:`DgradConvSpec.validate` and :func:`is_valid_dgrad_spec`
    enforce on a concrete spec, asked before a degree is chosen so an auto
    degree never lands on a spec the builder then rejects. The split partials
    are atomic-added into dX: that needs the MFMA (wave64) path, an
    fp32/bf16/fp16 dX, and -- for the packed ``<2 x dtype>`` 16-bit atomic --
    an even channels-per-group so a pair never straddles two groups' slabs.
    """
    if wave_size != 64:
        return False, "split_k > 1 is CDNA-only (wave64 MFMA path)"
    if dtype_d not in ("fp32", "bf16", "fp16"):
        return False, f"split_k > 1 requires dtype_d in fp32/bf16/fp16, got {dtype_d!r}"
    if dtype_d in ("bf16", "fp16") and p.cpg % 2 != 0:
        return False, (
            f"split_k > 1 with dtype_d={dtype_d!r} requires even channels-per-group; "
            f"got cpg={p.cpg}"
        )
    return True, "ok"


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
    y_dot = ceil_div(p.Y, y_tilde)
    x_dot = ceil_div(p.X, x_tilde)
    h_tilde = p.Ho + ceil_div(p.dH * (p.Y - 1), p.sH) if p.Y > 1 else p.Ho
    w_tilde = p.Wo + ceil_div(p.dW * (p.X - 1), p.sW) if p.X > 1 else p.Wo
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


def width_fold(p: ConvProblem) -> int:
    """The largest ``f`` :func:`fold_width` can fold ``p``'s width by (1: none).

    ``f`` divides the filter width, the W stride, the input width and the W
    padding; the W dilation is 1 and the problem 2-D and ungrouped.
    """
    if p.is_3d or p.groups != 1 or p.dW != 1:
        return 1
    return _gcd(_gcd(p.X, p.sW), _gcd(p.Wi, p.pW))


def fold_width(p: ConvProblem, f: Optional[int] = None) -> ConvProblem:
    """``p`` with ``f`` adjacent W positions folded into the channels.

    Write ``w = w' * f + r`` and ``x = x' * f + r'``. With ``f`` dividing
    ``sW`` and ``pW``, an input column pairs with a filter tap only when
    ``r == r'``, so the convolution is the one over ``f * C`` channels
    ``(r, c)`` with ``X / f`` taps, stride ``sW / f``, padding ``pW / f`` and
    width ``Wi / f``. With ``f`` dividing ``X`` and ``Wi`` as well, ``dX``
    (NHWC) and ``W`` (KYXC) hold exactly that convolution's tensors, and
    ``dY`` is unchanged (``Wo`` is the same), so a kernel run on the folded
    problem computes ``p``'s ``dX`` in place.

    It is a dgrad of fewer, wider tilde sub-GEMMs: a patch embedding
    (stride = filter, no padding) goes from ``Y * X`` sub-GEMMs of ``C``
    channels to ``Y`` of ``X * C``, reading ``dY`` ``X`` times less and with
    vector-wide channel runs. ``f`` defaults to :func:`width_fold`.
    """
    if f is None:
        f = width_fold(p)
    if f == 1:
        return p
    if p.Wi % f or p.X % f or p.sW % f or p.pW % f or p.dW != 1:
        raise ValueError(f"cannot fold the width of {p} by {f}")
    if p.is_3d or p.groups != 1:
        raise ValueError(f"cannot fold the width of a 3-D or grouped problem {p}")
    return dc_replace(
        p, Wi=p.Wi // f, C=p.C * f, X=p.X // f, sW=p.sW // f, pW=p.pW // f
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
            y_dot_slice = ceil_div(p.Y - i_yt, tilde.y_tilde)
            x_dot_slice = ceil_div(p.X - i_xt, tilde.x_tilde)
            if y_dot_slice <= 0 or x_dot_slice <= 0:
                continue

            h_tilde_slice_begin = _floor_div(
                max(0, p.pH - p.dH * (tilde.y_tilde - 1)), p.sH
            )
            h_tilde_slice_end = min(
                tilde.h_tilde,
                ceil_div(p.pH + p.Hi - 1, p.sH) + 1,
            )
            w_tilde_slice_begin = _floor_div(
                max(0, p.pW - p.dW * (tilde.x_tilde - 1)), p.sW
            )
            w_tilde_slice_end = min(
                tilde.w_tilde,
                ceil_div(p.pW + p.Wi - 1, p.sW) + 1,
            )

            h_tilde_slice = h_tilde_slice_end - h_tilde_slice_begin
            w_tilde_slice = w_tilde_slice_end - w_tilde_slice_begin
            if h_tilde_slice <= 0 or w_tilde_slice <= 0:
                continue

            gemm_m = p.N * h_tilde_slice * w_tilde_slice
            gemm_n = p.cpg
            gemm_k = y_dot_slice * x_dot_slice * p.kpg

            num_m_tiles = ceil_div(gemm_m, tile_m)
            num_n_tiles = ceil_div(gemm_n, tile_n)
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
                gemm_k_padded=ceil_div(gemm_k, tile_k * split_k) * (tile_k * split_k),
            )
            sub_gemms.append(sg)

    return sub_gemms


def is_patch_problem(p: ConvProblem) -> bool:
    """Does every dX pixel of ``p`` take at most one filter tap?

    True for a patch embedding: a 2-D ungrouped convolution with no padding
    or dilation whose stride is at least its filter on both axes. Then the
    dgrad is one GEMM (see ``DgradConvSpec.patch``).
    """
    return (
        not p.is_3d
        and p.groups == 1
        and p.pH == 0
        and p.pW == 0
        and p.dH == 1
        and p.dW == 1
        and p.sH >= p.Y
        and p.sW >= p.X
    )


def patch_sub_gemms(
    p: ConvProblem, tile_m: int, tile_n: int, tile_k: int = 32
) -> List[SubGemmParams]:
    """The one sub-GEMM record a ``patch`` kernel runs ``p`` with.

    The record of the pointwise problem over ``(N, Ho, Wo)`` with ``Y*X*C``
    channels -- the GEMM ``dY[M, K] @ W[K, Y*X*C]`` -- with the dX strides
    of ``p``: the record maps GEMM row ``(n, ho, wo)`` to dX pixel
    ``(ho*sH, wo*sW)``, where the patch's first element is. Its tap strides
    are the filter's, so the K tail past ``K`` (``ydot`` = 1 when ``K`` is
    not a multiple of the K tile) lands on ``y = Y``, outside the filter,
    and reads no W row -- with a tap stride of 1 it would read row ``y = 1``.
    """
    if not is_patch_problem(p):
        raise ValueError(f"{p} is not a patch embedding")
    v = ConvProblem(N=p.N, Hi=p.Ho, Wi=p.Wo, C=p.Y * p.X * p.C, K=p.K, Y=1, X=1)
    (sg,) = enumerate_sub_gemms(v, compute_tilde(v), tile_m, tile_n, tile_k=tile_k)
    return [
        dc_replace(sg, b_y_stride=p.Y, b_x_stride=p.X, d_h_stride=p.sH, d_w_stride=p.sW)
    ]


def _magic_i32_pair(divisor: int) -> List[int]:
    """``(mult, shift)`` of :func:`calculate_magic_numbers`, mult as signed i32.

    ``calculate_magic_numbers`` rejects a non-positive divisor; a record whose
    fields produce one is corrupt and must fail here, not be silently
    renormalized to 1. C++ twin: ``rocke_pack_sub_gemm_buffer``.
    """
    mult, shift = calculate_magic_numbers(int(divisor))
    return [mult - (1 << 32) if mult >= (1 << 31) else mult, shift]


def sub_gemm_magic_pairs(sg: SubGemmParams) -> List[int]:
    """The magic-division table entry of one sub-GEMM record.

    The kernel decodes ``k -> (ydot, xdot, k_out)`` and
    ``m -> (n, htl, wtl)`` by the record's own extents; a runtime ``udiv``
    there costs a long VALU sequence on every K step. One ``(mult, shift)``
    pair per divisor, in :data:`_MAGIC_FIELDS` order: kpg, x_dot_slice,
    h_tilde_slice*w_tilde_slice, w_tilde_slice.
    """
    kpg = sg.gemm_k // (sg.y_dot_slice * sg.x_dot_slice)
    out: List[int] = []
    for d in (
        kpg,
        sg.x_dot_slice,
        sg.h_tilde_slice * sg.w_tilde_slice,
        sg.w_tilde_slice,
    ):
        out += _magic_i32_pair(d)
    return out


def pack_sub_gemm_buffer(
    sub_gemms: List[SubGemmParams], tile_m: int, tile_n: int
) -> List[int]:
    """Pack all sub-GEMM records into a flat i32 buffer for the kernel.

    The buffer is the ``_RECORD_FIELDS``-wide records, then one
    ``_MAGIC_FIELDS``-wide magic-division entry per record
    (:func:`sub_gemm_magic_pairs`). The table follows the records rather than
    widening them so the record stride -- and every binary built against
    it -- is unchanged; the kernel finds it at ``num_sub_gemms`` records in.
    """
    buf: List[int] = []
    for sg in sub_gemms:
        record = sg.pack_i32()
        record[1] = ceil_div(sg.gemm_m, tile_m)
        record[2] = ceil_div(sg.gemm_n, tile_n)
        buf.extend(record)
    for sg in sub_gemms:
        buf.extend(sub_gemm_magic_pairs(sg))
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


def make_dgrad_dx_descriptor_dynamic(b: IRBuilder, params: Dict[str, Value]):
    """AOT counterpart of :func:`make_dgrad_dx_descriptor`: ``(m, c) -> NHWC``.

    ``m -> (n, hi, wi)`` peels one extent at a time, so it divides by ``Wi``
    and then by ``Hi``; the host supplies a magic pair per extent.
    """
    return DynamicTensorDescriptor.create(
        "dX_nhwc",
        coord_names=["n", "hi", "wi", "c"],
        strides=[
            params["p_dX_stride_n"],
            params["p_dX_stride_hi"],
            params["p_dX_stride_wi"],
            b.const_i32(1),
        ],
    ).transform(
        unmerge_magic_dynamic(
            "m",
            into=["n", "hi", "wi"],
            magic_triples=[
                (
                    params["p_magic_m_Hi_mult"],
                    params["p_magic_m_Hi_shift"],
                    params["p_Hi"],
                ),
                (
                    params["p_magic_m_Wi_mult"],
                    params["p_magic_m_Wi_shift"],
                    params["p_Wi"],
                ),
            ],
        ),
    )


# ---------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------


# The K-outer B tile is fed by an LDS transpose read (ds_read_tr16_b64 on
# gfx950 wave64, ds_load_tr16_b128 on gfx1250 wave32).
# Architectures whose LDS transpose read can feed a K-outer tile, and the wave
# size each requires. gfx950 wave64 MFMA uses ds_read_b64_tr_b16 (4 per lane);
# gfx1250 wave32 WMMA uses ds_load_tr16_b128 (8 per lane). The IR op is shared;
# core/isa/backend.py picks the opcode.
_LDS_K_OUTER_ARCH_WAVE = {"gfx950": 64, "gfx1250": 32}
_LDS_K_OUTER_ARCH = "gfx950"  # the wave64 regime

# Row stride pad, in elements, for the K-outer B tile. The row stride must not
# be a multiple of the 32-dword LDS bank period or the transpose read
# degenerates: the only lane term that walks rows is ((l%16)//4), which
# contributes zero bank spread whenever (stride_elems * 2 / 4) % 32 == 0. A pad
# of 8 makes a 64-wide tile 36 dwords (36 % 32 == 4), spreading the four
# row-groups across banks, and keeps each row 16-byte aligned for the b128
# store side. Shared by the builder and is_valid_dgrad_spec so the charged and
# the allocated shape cannot drift. Mirrors ROCKE_DGRAD_KOUTER_PAD.
_KOUTER_PAD = 8


def _default_lds_k_pad(tile_k: int) -> int:
    """Row pad of the M-outer LDS tiles when ``lds_k_pad`` is None."""
    return 8 if tile_k >= 16 else 0


# Upper bound on fp32 accumulators per lane accepted by is_valid_dgrad_spec.
# Mirrors ROCKE_DGRAD_MAX_ACC_REGS_PER_LANE.
_MAX_ACC_REGS_PER_LANE = 256


def _acc_regs_per_lane(spec: DgradConvSpec) -> int:
    """fp32 accumulator registers per lane held by one warp's output tile."""
    return (
        spec.mfmas_per_warp_m
        * spec.mfmas_per_warp_n
        * (spec.warp_tile_m * spec.warp_tile_n // spec.wave_size)
    )


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

    # Store the B (W, KYXC) LDS tile K-outer as T[k][n] and transpose during the
    # MFMA operand fetch with ds_read_tr16_b64.
    #
    # Unlike wgrad this is a B-ONLY layout. dgrad's A (dY, NHWK) already has a
    # stride-1 reduction axis -- k_out is innermost in k_dg -- so its loader is
    # already vector_axis="col" with a single wide smem_store_vN, and its
    # M-outer fragment read is already conflict-free via lds_k_pad. Flipping A
    # would put the global vector along m=(n,hi,wi), stride K in NHWK, which
    # destroys coalescing for zero write-side gain.
    #
    # B is the sole scatter: c (KYXC's stride-1 axis) is the GEMM free axis, so
    # axis_b="row" makes _store_tile emit one ds_write_b16 per element, and
    # adjacent lanes step whole padded rows -- a bank-degenerate write the
    # lds_k_pad tuning cannot fix, because that pad is derived for a row step of
    # 1 on the read path.
    #
    # Default False so every existing dgrad golden stays byte-identical.
    lds_k_outer: bool = False

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
    # Largest tilde sub-GEMM count this compiled kernel can dispatch over.
    # The record count is runtime (it follows stride/dilation), and so is the
    # CTA's binary search over the record buffer: it runs ceil(log2(count))
    # steps, which the kernel works out against this bound. The tilde
    # decomposition yields y_tilde*x_tilde records, bounded by sH*sW, so 1024
    # covers strides up to 32x32 (the patch embeddings of vision
    # transformers). A larger bound only lengthens a few scalar compares. A
    # problem needing more is rejected by is_valid_dgrad_spec rather than
    # silently mis-dispatched.
    max_sub_gemms: int = 1024

    # Decode the reduction index once per K tile, in uniform registers.
    #
    # The kernel is shape-generic: kpg, the filter-tap counts and every stride
    # are kernargs or sub-GEMM record fields, so the plain K loop splits each
    # load chunk's k index into (ydot, xdot, k_out) with two runtime signed
    # divisions and then forms its offset from runtime-stride products -- per
    # chunk, per iteration, on the path to the first buffer_load.
    #
    # When kpg is a multiple of tile_k (checked at run time), a K tile never
    # straddles a filter tap: k_out of every element is the tile's base k_out
    # plus its column. This flag adds a K loop that carries the tile's
    # (k_out, xdot, ydot) as uniform iteration arguments, advances it with a
    # compare-and-wrap per tile, and splits every offset into a loop-invariant
    # per-element part and a uniform per-tile part. Problems whose kpg is not a
    # multiple of tile_k run the original loop instead, which the kernel also
    # carries, so the binary serves the same problems either way. Both orders
    # visit the same k values in the same sequence, so the results match.
    #
    # The wavelet pipeline has its own K loop and rejects the flag.
    #
    # Default False: strictly additive, so every existing config emits
    # byte-identical IR and keeps its kernel name. See
    # ``default_uniform_k_decode`` for the selection policy.
    uniform_k_decode: bool = False

    # Run a patch embedding (stride >= filter on both axes, no padding, no
    # dilation, ungrouped: every dX pixel takes at most one filter tap) as ONE
    # GEMM: M = N*Ho*Wo, N = Y*X*C (the filter's (y, x, c), contiguous in
    # KYXC), K = K. The tilde decomposition would split it into Y*X sub-GEMMs
    # of C columns each -- three for an RGB stem -- each re-reading all of dY.
    #
    # The sub-GEMM record is the pointwise problem's (patch_sub_gemms), so the
    # K loop is untouched: the record's ydot = xdot = 0 makes the dY offset
    # (n*Ho + ho)*Wo*K + wo*K + k and the W offset k*Y*X*C + n on the real
    # kernargs. Only the dX store differs: column n = (y, x*C + c) lands
    # y rows and x*C + c elements past the patch's first element
    # (patch_sub_gemms sets d_h/w_stride to the strides). The caller zeroes
    # the pixels no patch covers, as for any stride > filter.
    #
    # Default False: strictly additive.
    patch: bool = False
    # Stride-1 tap-outer K loop: the reduction runs as two nested loops,
    # filter tap (y, x) outer and output-channel chunk inner, instead of one
    # loop over k_dg = (y, x, k_out). Stride 1 and dilation 1 make tap (y, x)
    # read dY pixel (hi + pH - y, wi + pW - x), so the tap decode is uniform
    # outer-loop arithmetic on the kernargs (one scalar divide per tap), the
    # pixel decode (n, hi, wi) and a tap's dY bounds are invariant across the
    # chunk loop, and each chunk adds one uniform offset per operand. Every K
    # tile has to lie inside one tap, so kpg must be a multiple of tile_k:
    # unlike uniform_k_decode there is no fallback loop in the binary, so the
    # validator rejects a problem that breaks this, and the AOT cache offers
    # the binary only to problems that keep it. Also needs split_k == 1 and
    # the wave64 MFMA loop, and replaces the loop uniform_k_decode would build
    # (setting both is rejected). Default False; tags the name ``tapk``.
    tap_outer_k: bool = False

    # XCD-contiguous tile order for grouped stride-1 kernels on the targets in
    # _XCD_TILE_ORDER_ARCHES (see xcd_tile_order_xcds). Default on: the
    # kernel decides from its kernargs whether to remap, and the remap is a
    # bijection of the launch grid, so the knob only exists to turn the order
    # off for an A/B comparison. Never swept. False tags the name ``noxcd``.
    xcd_tile_order: bool = True

    # dY halo reuse (default 0 = off), a mode of the tap-outer K loop. With
    # stride 1, dilation 1 and an output the size of the input (Ho == Hi, Wo
    # == Wi), filter tap (y, x) reads dY pixel m + (pH - y)*Wo + (pW - x) for
    # dX pixel m in the linear (n, h, w) order, so every tap of a tile of
    # tile_m consecutive dX pixels reads one contiguous dY range: the tile
    # plus (Y-1)*Wo + (X-1) halo pixels. The K loop then runs output-channel
    # chunk outer and filter tap inner (unrolled), loads that range into LDS
    # once per chunk and serves every tap from it at a row shift, instead of
    # re-gathering dY per (tap, chunk). A lane whose shifted pixel leaves the
    # image reads an all-zero LDS row instead (one select per fragment row).
    #   1: the staged halo; B (W) is still loaded per (chunk, tap), with two
    #      barriers per tap.
    #   2: as 1, plus a double-buffered B tile: tap t+1's W tile is read into
    #      registers before tap t's MFMAs and stored after them, one barrier
    #      per tap (the prefetch is pinned with sched_barrier on both sides).
    # Needs tap_outer_k (whose rules it inherits), Ho == Hi, Wo == Wi, a
    # filter of more than one tap and halo_max_wo >= Wo; the validator rejects
    # anything else. The filter is baked (the taps are unrolled; an AOT binary
    # records it as filter_h/w), and so is the pad, which Ho == Hi at stride 1
    # fixes to ((Y-1)/2, (X-1)/2). Wo, the tap shifts and the row masks are
    # runtime. Tags ``halo<n>``.
    dy_halo: int = 0
    # With dy_halo: raise the wave priority (s_setprio level, 1..3) around
    # each tap's MFMA block. Tags ``hprio<level>``.
    dy_halo_setprio: int = 0
    # With dy_halo and lds_k_outer: row pad of the K-outer B tile in elements
    # instead of _KOUTER_PAD (0 keeps _KOUTER_PAD). A multiple of 8 keeps every
    # row 16-byte aligned for the wide LDS store and the transpose read. Tags
    # ``hkp<pad>``.
    dy_halo_kouter_pad: int = 0
    # With dy_halo: the widest output row (Wo) the kernel serves. The staged
    # halo is sized for it -- tile_m + (Y-1)*halo_max_wo + (X-1) LDS rows --
    # so the LDS shape is a property of the spec, not of the problem, and a
    # problem with Wo <= halo_max_wo runs on the same binary. Required (> 0)
    # with dy_halo, 0 otherwise. Tags ``hwo<n>``.
    halo_max_wo: int = 0

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * self.wave_size

    @property
    def dy_halo_lds_rows(self) -> int:
        """LDS rows of the staged dY halo tile (one row = tile_k channels).

        tile_m + (Y-1)*halo_max_wo + (X-1) pixels, padded to a whole number of
        loader passes (block_size threads x 8 elements, so the loader splits
        the tile evenly at every load width), plus a trailing all-zero row.
        Shared by the builder and the LDS charge so the charged and the
        allocated shape cannot drift. Y and X are the baked filter.
        """
        p = self.problem
        rows = self.tile_m + (p.Y - 1) * self.halo_max_wo + (p.X - 1)
        pass_elems = self.block_size * 8
        quantum = pass_elems // _gcd(pass_elems, self.tile_k)
        return -(-rows // quantum) * quantum + 1

    @property
    def kouter_b_pad(self) -> int:
        """Row pad of the K-outer B tile (see dy_halo_kouter_pad)."""
        if self.dy_halo > 0 and self.dy_halo_kouter_pad > 0:
            return self.dy_halo_kouter_pad
        return _KOUTER_PAD

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

    @staticmethod
    def default_lds_k_outer(
        *,
        arch: str,
        dtype_b: str,
        warp_tile_n: int,
        cpg: int,
        wave_size: int = 64,
        pipeline: str = "mem",
    ) -> bool:
        """Whether the K-outer B tile is the better layout for this spec.

        Note two deliberate divergences from the wgrad predicate.

        The gate is ASYMMETRIC: it keys on ``dtype_b``/``warp_tile_n`` only,
        never their A-side counterparts, because only the B tile flips. Copying
        wgrad's symmetric predicate would over-reject dgrad specs whose A side
        differs.

        It also carries ``cpg``. The saving is proportional to the B load width,
        and that width collapses to 1 when the stride-1 channel run is odd -- at
        which point ``axis_b`` is already "col" and there is no scatter to
        remove, so K-outer would be a small pure regression rather than a win.

        The ``lds_k_outer`` field itself still defaults to False; this is the
        selection policy, not the default.
        """
        if arch not in _LDS_K_OUTER_ARCH_WAVE:
            return False
        if wave_size != _LDS_K_OUTER_ARCH_WAVE[arch]:
            return False
        if pipeline == "wavelet":
            # The wavelet loader does not implement the K-outer tile (see the
            # validate() gate). Selecting M-outer here keeps the combination
            # off the sweep entirely -- it costs the transpose-on-store this
            # optimisation removes, but it is correct, whereas the pair is not.
            return False
        if dtype_b not in ("bf16", "fp16"):
            return False
        if wave_size == 32:
            # gfx1250 WMMA 16x16x32 is the only wave32 atom with a derived
            # lane mapping.
            if warp_tile_n != 16:
                return False
        elif warp_tile_n not in (16, 32):
            return False
        # 16-bit free-axis width > 1 requires an even channel run.
        return cpg % 2 == 0

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
        if self.patch:
            return patch_sub_gemms(
                self.problem, self.tile_m, self.tile_n, tile_k=self.tile_k
            )
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

    @property
    def folds_sub_gemm_record(self) -> bool:
        """True when the kernel builds its sub-GEMM record from kernargs.

        At stride 1 the tilde decomposition is the identity in both spatial
        axes (y_tilde = x_tilde = 1, whatever the dilation, pad, filter or
        grouping), so there is exactly one sub-GEMM and every field of its
        record is an existing kernarg or a constant:

        ``[0, num_pid_m, num_pid_n, dg_M, dg_K, Hi, Wi, pH, pW, Y, X, -dH,
        -dW, 1, 0, 1, 0, 1, 0, 1, 0, roundup(dg_K, tile_k*split_k)]``.

        The kernel then skips the binary search over the record buffer and
        the 21 record loads behind it, and the unit strides and zero offsets
        fold into the address arithmetic. This keys on the stride alone: an
        AOT binary carries it as its stride capability (the probe problem and
        every problem the binary serves agree on it), never on a sub-GEMM
        count evaluated on one problem. The ABI is unchanged; the record
        buffer is still passed (its magic-division table is read).
        """
        p = self.problem
        return p.sH == 1 and p.sW == 1

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
                # Mandatory, not cosmetic: the compile cache keys on
                # kernel.name, so an M-outer and a K-outer spec differing only
                # in LDS layout would collide on one symbol and an A/B sweep
                # would silently measure one kernel twice.
                "kouter": self.lds_k_outer,
                f"spk{self.split_k}": self.split_k > 1,
                "spkauto": self.split_k == -1,
                # A different K-loop body; only tagged when set.
                "ukd": self.uniform_k_decode,
                "patch": self.patch,
                # The A row pad changes the LDS layout, so a non-default pad
                # must reach the name or it collides with the default build.
                f"kp{self.lds_k_pad}": self.lds_k_pad is not None
                and self.lds_k_pad != _default_lds_k_pad(self.tile_k),
                # A different K-loop body; only tagged when set.
                "tapk": self.tap_outer_k,
                # The launch-order tile walk (A/B knob, see xcd_tile_order).
                "noxcd": not self.xcd_tile_order,
                # The dY halo K loop and its knobs (see dy_halo).
                f"halo{self.dy_halo}": self.dy_halo > 0,
                f"hprio{self.dy_halo_setprio}": self.dy_halo_setprio > 0,
                f"hkp{self.dy_halo_kouter_pad}": self.dy_halo_kouter_pad > 0,
                f"hwo{self.halo_max_wo}": self.halo_max_wo > 0,
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
        if self.lds_k_outer:
            # ds_read_tr16_b64 is a 16-bit transpose read, so only the B side
            # constrains the dtype -- A keeps the ordinary _emit_smem_load.
            if self.data.dtype_b not in ("bf16", "fp16"):
                raise ValueError(
                    "lds_k_outer requires a 16-bit B dtype (ds_read_tr16_b64 is "
                    f"a 16-bit transpose read); got dtype_b={self.data.dtype_b!r}"
                )
            if self.warp_tile_n not in (16, 32):
                raise ValueError(
                    "lds_k_outer requires warp_tile_n in (16, 32) -- the "
                    "transpose-read lane mapping is derived per 16-lane group "
                    f"over the atom edge; got warp_tile_n={self.warp_tile_n}"
                )
            if self.wave_size not in (64, 32):
                raise ValueError(
                    "lds_k_outer requires wave_size 64 (ds_read_tr16_b64) or 32 "
                    f"(ds_load_tr16_b128); got {self.wave_size}"
                )
            if self.wave_size == 32 and (
                self.warp_tile_n != 16 or self.warp_tile_k != 32
            ):
                # One atom in the wave32 regime: gfx1250 WMMA 16x16x32.
                raise ValueError(
                    "lds_k_outer on wave32 supports only the 16x16x32 atom (got "
                    f"{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k})"
                )
            if self.lds_layout is not None:
                # Reject rather than ignore: under K-outer the B shape is
                # computed straight from (block_k, block_n + _KOUTER_PAD) and
                # never consults the layout object, so honouring an explicit
                # one would be a silent lie.
                raise ValueError(
                    "lds_k_outer does not honour an explicit lds_layout: the "
                    "K-outer B row stride is fixed by the transpose-read bank "
                    "argument (_KOUTER_PAD). Leave lds_layout=None."
                )
            if self.async_dma:
                raise ValueError(
                    "lds_k_outer is not supported with async_dma on dgrad: the "
                    "tilde builder has no direct global->LDS path"
                )
            if self.pipeline == "wavelet":
                # Same shape of problem as async_dma above: the alternate load
                # path does not implement K-outer. build_wavelet_loaders pins
                # the B tile to (block_n, block_k) and takes the unswapped
                # descriptor, so it writes the tile M-outer while the compute
                # phase reads it through _tr_frag. The allocation is K-outer
                # -- (block_k, block_n + _KOUTER_PAD) -- so this is not merely
                # a transposed operand: the row stride is wrong for every
                # element and, whenever block_n > block_k (the usual case), the
                # store runs off the end of B_smem into a neighbouring LDS
                # allocation. Verified numerically wrong on gfx1250.
                raise ValueError(
                    "lds_k_outer is not supported with pipeline='wavelet' on "
                    "dgrad: the wavelet loader writes the B tile M-outer into a "
                    "K-outer allocation (wrong row stride, and out of bounds "
                    "when tile_n > tile_k)"
                )
        _ukd_why = _uniform_k_decode_conflict(self)
        if _ukd_why:
            raise ValueError(_ukd_why)
        _patch_why = _patch_conflict(self)
        if _patch_why:
            raise ValueError(_patch_why)
        _tapk_why = _tap_outer_k_conflict(self)
        if _tapk_why:
            raise ValueError(_tapk_why)
        _halo_why = _dy_halo_conflict(self)
        if _halo_why:
            raise ValueError(_halo_why)
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
            layout = LdsLayout.padded_k(self.tile_k, _default_lds_k_pad(self.tile_k))
        layout.validate()
        return layout

    @staticmethod
    def default_uniform_k_decode(
        *,
        arch: str,
        wave_size: int = 64,
        pipeline: str = "mem",
    ) -> bool:
        """Whether a dgrad spec should decode its reduction index per K tile.

        Selection policy only, in the same role as ``default_lds_k_outer``:
        the ``uniform_k_decode`` field itself defaults to ``False`` and callers
        that build specs for a sweep or a cache ask this what to pass.

        The flag replaces the per-chunk division chain of the K loop with
        uniform per-tile arithmetic whenever the problem allows it and keeps
        the original loop for the problems that do not, so it never needs a
        problem extent to decide and holds for a shape-generic AOT kernel.
        Unlike wgrad's ``prefetch_addr`` it adds no live state across the
        loop, so the policy has no tile-dependent split: it is on for every
        tile of the regime it was evaluated in -- gfx950 wave64, not the
        wavelet loop -- and off elsewhere until evaluated there.
        """
        if arch != "gfx950" or wave_size != 64:
            return False
        return pipeline != "wavelet"


def _uniform_k_decode_conflict(spec: "DgradConvSpec") -> str:
    """Why ``spec.uniform_k_decode`` cannot be honoured, or ``""`` if it can.

    The flag rewrites the synchronous K loop. The wavelet pipeline is a
    separate load/math-wave body that never reaches it, so accepting the flag
    there would only rename a kernel whose code it does not change. Shared by
    ``validate`` and ``is_valid_dgrad_spec`` so the hard and soft gates cannot
    drift apart.
    """
    if spec.uniform_k_decode and spec.pipeline == "wavelet":
        return (
            "uniform_k_decode is not supported with pipeline='wavelet': the "
            "wavelet load/math split has its own K loop"
        )
    return ""


def _patch_conflict(spec: "DgradConvSpec") -> str:
    """Why ``spec.patch`` cannot be honoured, or ``""`` if it can.

    The patch store is the tilde epilogue's, with one plain store per
    element: split-K's atomic epilogue and the wavelet pipeline's own
    epilogue do not carry it, and its column split has no group term.
    """
    if not spec.patch:
        return ""
    if spec.split_k != 1:
        return "patch requires split_k == 1 (the atomic epilogue has no patch store)"
    if spec.pipeline == "wavelet":
        return "patch is not supported with pipeline='wavelet'"
    if spec.problem.groups != 1:
        return "patch serves ungrouped problems only"
    if not spec.is_strided:
        return "patch runs on the tilde (strided) path"
    if not is_patch_problem(spec.problem):
        return (
            "patch serves patch embeddings only (no padding or dilation, "
            "stride >= filter)"
        )
    return ""


def _tap_outer_k_conflict(spec: DgradConvSpec) -> str:
    """Why ``spec.tap_outer_k`` cannot be honoured, or ``""`` if it can.

    The tap-outer loop is correct only where every K tile lies inside one
    filter tap and the tap is the plain filter (y, x). It has no fallback
    loop, so a problem outside that is rejected rather than built: a binary
    must never silently compute a problem it cannot. Shared by ``validate``
    and ``is_valid_dgrad_spec``; the reasons are mirrored verbatim by the C++
    validator.
    """
    if not spec.tap_outer_k:
        return ""
    p = spec.problem
    if p.sH != 1 or p.sW != 1 or p.dH != 1 or p.dW != 1:
        return (
            "tap_outer_k needs stride 1 and dilation 1 (the filter tap is the "
            "plain (y, x) only there)"
        )
    if p.kpg % spec.tile_k != 0:
        return (
            f"tap_outer_k needs kpg % tile_k == 0 (a K tile must not straddle "
            f"two filter taps); got kpg={p.kpg}, tile_k={spec.tile_k}"
        )
    if spec.split_k != 1:
        return "tap_outer_k needs split_k == 1 (a K slice could start mid-tap)"
    if spec.wave_size != 64 or spec.pipeline == "wavelet":
        return (
            "tap_outer_k is a wave64 MFMA K loop (the wavelet pipeline has its "
            "own K loop)"
        )
    if spec.uniform_k_decode:
        return (
            "tap_outer_k replaces the K loop uniform_k_decode builds; set one "
            "of them"
        )
    return ""


def _dy_halo_conflict(spec: DgradConvSpec) -> str:
    """Why the dY halo knobs cannot be honoured, or ``""`` if they can.

    Every knob that would otherwise be silently ignored is rejected, so a
    kernel name never claims a knob that did not take effect, and the halo
    loop is never built for a problem it cannot compute. Shared by
    ``validate`` and ``is_valid_dgrad_spec``; the reasons are mirrored
    verbatim by the C++ validator.
    """
    if spec.dy_halo not in (0, 1, 2):
        return (
            "dy_halo must be 0 (off), 1 (staged halo) or 2 (staged halo with "
            f"double-buffered B); got {spec.dy_halo}"
        )
    if spec.dy_halo_setprio < 0 or spec.dy_halo_setprio > 3:
        return f"dy_halo_setprio must be in 0..3; got {spec.dy_halo_setprio}"
    if spec.dy_halo_kouter_pad < 0 or spec.dy_halo_kouter_pad % 8 != 0:
        return (
            "dy_halo_kouter_pad must be a non-negative multiple of 8 (16-byte "
            "rows for the wide LDS store and the transpose read); got "
            f"{spec.dy_halo_kouter_pad}"
        )
    if spec.halo_max_wo < 0:
        return f"halo_max_wo must be non-negative; got {spec.halo_max_wo}"
    if spec.dy_halo == 0:
        if spec.dy_halo_setprio > 0:
            return "dy_halo_setprio needs dy_halo > 0"
        if spec.dy_halo_kouter_pad > 0:
            return "dy_halo_kouter_pad needs dy_halo > 0 and lds_k_outer"
        if spec.halo_max_wo > 0:
            return "halo_max_wo needs dy_halo > 0"
        return ""
    p = spec.problem
    if not spec.tap_outer_k:
        return (
            "dy_halo needs the tap-outer K loop (tap_outer_k: stride 1, "
            "dilation 1, kpg % tile_k == 0, no split-K, wave64)"
        )
    if p.Ho != p.Hi or p.Wo != p.Wi:
        return "dy_halo needs an output the size of the input (Ho == Hi and Wo == Wi)"
    if p.Y * p.X == 1:
        return "dy_halo needs a filter with more than one tap"
    if spec.halo_max_wo <= 0:
        return (
            "dy_halo needs halo_max_wo > 0 (the widest Wo served; the staged "
            "halo is sized for it)"
        )
    if p.Wo > spec.halo_max_wo:
        return (
            f"dy_halo serves Wo <= halo_max_wo; got Wo={p.Wo}, "
            f"halo_max_wo={spec.halo_max_wo}"
        )
    if spec.dy_halo_kouter_pad > 0 and not spec.lds_k_outer:
        return "dy_halo_kouter_pad needs dy_halo > 0 and lds_k_outer"
    return ""


# Targets whose grouped stride-1 kernels lay their tiles out XCD-contiguous
# (see xcd_tile_order_xcds), and the XCD count of each. Evaluated on gfx950
# only. Mirrors ROCKE_DGRAD_XCD_TILE_ORDER_ARCH.
_XCD_TILE_ORDER_ARCHES = {"gfx950": 8}


def xcd_tile_order_xcds(spec: DgradConvSpec, arch: str) -> int:
    """XCD count the kernel lays its tile order out for, or 0 (launch order).

    The hardware hands workgroups to the XCDs round-robin in launch order
    (``blockIdx.x`` fastest, the group on ``blockIdx.y``), and each XCD has
    its own L2. Neighbouring tiles of one group -- the M tiles that read the
    group's weight slice and the N tiles that read the same dY rows -- land
    on different XCDs, and every XCD refetches each group's operands. The
    XCD-contiguous order gives each XCD one contiguous range of the ``(group,
    tile)`` space instead (:func:`chiplet_transform_chunked_runtime`, one
    chunk of ``num_wgs / num_xcds`` workgroups per XCD; the remainder below a
    multiple of the XCD count keeps launch order), so a group's operands are
    fetched into one L2.

    Build time: the target is in :data:`_XCD_TILE_ORDER_ARCHES`, the knob is
    on, the kernel is grouped (an ungrouped problem shares its operands
    across every tile, so launch order already spreads them), the record is
    folded (stride 1: one sub-GEMM, whose tile count is ``num_pid_m *
    num_pid_n``) and there is no split-K (its slices ride ``blockIdx.z``).
    Run time, from the kernargs, with one uniform select: ``groups > 1``,
    ``Y*X > 1 or num_pid_n > 1`` (with one tap and one N tile no dY row is
    shared between tiles, and the order only moves the streamed rows apart)
    and ``num_wgs >= num_xcds``. A grouped binary serves groups == 1 too, so
    the run-time test is what keeps an ungrouped launch in launch order.
    """
    num_xcds = _XCD_TILE_ORDER_ARCHES.get(arch, 0)
    if not num_xcds or not spec.xcd_tile_order:
        return 0
    if spec.problem.groups <= 1 or spec.split_k != 1:
        return 0
    if not spec.folds_sub_gemm_record:
        return 0
    return num_xcds


def emit_xcd_tile_order(
    b: IRBuilder, block_x: Value, params: dict[str, Value], *, num_xcds: int
) -> tuple[Value, Value]:
    """The ``(flat tile id, group)`` of this workgroup under the XCD order.

    ``block_x`` is the launch's ``block_id_x`` (the tile, ``num_pid_m *
    num_pid_n`` of them) and ``block_id_y`` the group. Their linear id is
    remapped (:func:`chiplet_transform_chunked_runtime`, the default chunk)
    when ``groups > 1``, ``Y*X > 1 or num_pid_n > 1`` and ``num_wgs >=
    num_xcds`` hold on the kernargs -- a uniform select -- and split back
    into tile and group; otherwise the split returns ``(block_id_x,
    block_id_y)``. See :func:`python_xcd_tile_order` for the model.
    """
    c1 = b.const_i32(1)
    flat_tiles = b.mul(params["p_num_pid_m"], params["p_num_pid_n"])
    num_wgs = b.mul(flat_tiles, params["p_groups"])
    linear_id = b.add(block_x, b.mul(b.block_id_y(), flat_tiles))
    remapped = chiplet_transform_chunked_runtime(
        b, linear_id, num_wgs=num_wgs, num_xcds=num_xcds
    )
    shares_rows = b.lor(
        b.cmp_gt(b.mul(params["p_Y"], params["p_X"]), c1),
        b.cmp_gt(params["p_num_pid_n"], c1),
    )
    use_xcd = b.land(
        b.land(b.cmp_gt(params["p_groups"], c1), shares_rows),
        b.cmp_ge(num_wgs, b.const_i32(num_xcds)),
    )
    logical_id = b.select(use_xcd, remapped, linear_id)
    return b.mod(logical_id, flat_tiles), b.div(logical_id, flat_tiles)


def python_xcd_tile_order(
    bx: int,
    by: int,
    *,
    num_pid_m: int,
    num_pid_n: int,
    groups: int,
    taps: int,
    num_xcds: int = _XCD_TILE_ORDER_ARCHES["gfx950"],
) -> tuple[int, int]:
    """Pure-Python model of :func:`emit_xcd_tile_order`: ``(tile, group)``."""
    from rocke.helpers.grid import python_chiplet_transform_chunked_runtime

    flat_tiles = num_pid_m * num_pid_n
    num_wgs = flat_tiles * groups
    linear_id = bx + by * flat_tiles
    use = groups > 1 and (taps > 1 or num_pid_n > 1) and num_wgs >= num_xcds
    if use:
        linear_id = python_chiplet_transform_chunked_runtime(
            linear_id, num_wgs=num_wgs, num_xcds=num_xcds
        )
    return linear_id % flat_tiles, linear_id // flat_tiles


# ---------------------------------------------------------------------
# Arch-aware spec validation
# ---------------------------------------------------------------------


def _dgrad_lds_charge(spec: DgradConvSpec) -> tuple[int, int, int]:
    """``(a_b_bytes, c_bytes, total_bytes)`` of LDS the builder allocates.

    The builder allocates one A and one B tile whatever the pipeline: compv4
    is a schedule policy on this kernel, not a second LDS buffer (and
    async_dma / unroll_k, which would double-buffer, are rejected), so
    nothing is charged twice.
    """
    _ab_dtype_bytes = 4 if spec.data.dtype_a in ("fp32",) else 2
    _lds_layout = spec.effective_lds_layout()
    # A is genuinely still M-outer; only B flips, so only B's charge branches.
    # Both sides read the B pad (kouter_b_pad) and the halo rows
    # (dy_halo_lds_rows) from the spec so the charged shape and the allocated
    # shape cannot drift.
    _a_shape = _lds_layout.storage_shape(
        spec.dy_halo_lds_rows if spec.dy_halo > 0 else spec.tile_m
    )
    _b_shape = (
        (spec.tile_k, spec.tile_n + spec.kouter_b_pad)
        if spec.lds_k_outer
        else _lds_layout.storage_shape(spec.tile_n)
    )
    # dy_halo=2 double-buffers the B tile.
    _b_copies = 2 if spec.dy_halo == 2 else 1
    _ab_bytes = (
        _a_shape[0] * _a_shape[1] + _b_copies * _b_shape[0] * _b_shape[1]
    ) * _ab_dtype_bytes
    _c_dtype_bytes = 4 if spec.data.dtype_d == "fp32" else 2
    _c_lds = (
        spec.tile_m * spec.tile_n * _c_dtype_bytes if spec.epilogue == "cshuffle" else 0
    )
    # wavelet: A/B stay live across both scf_if_else branches, so the LDS pool
    # cannot alias C onto the A/B region even when cshuffle_no_alias=False.
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    _total_lds = (_ab_bytes + _c_lds) if _no_alias else max(_ab_bytes, _c_lds)
    return _ab_bytes, _c_lds, _total_lds


def dgrad_lds_bytes(spec: DgradConvSpec) -> int:
    """Bytes of LDS one workgroup of ``spec`` allocates.

    The charge :func:`is_valid_dgrad_spec` checks against the LDS capacity.
    """
    return _dgrad_lds_charge(spec)[2]


def _kouter_frag_reason(spec: DgradConvSpec, arch: str) -> str | None:
    """Why the K-outer B fragment cannot be read, or None.

    The transpose read returns a fixed number of elements per lane
    (ds_read_tr16_b64: 4 on wave64; ds_load_tr16_b128: 8 on wave32), so the
    atom's B fragment length has to be a multiple of it. The builder raises on
    a violation; this mirrors that rule into the validator so a sweep filters
    the configuration out instead of losing it to a build error.
    """
    if not spec.lds_k_outer:
        return None
    try:
        b_per_lane = _resolve_dgrad_op(spec, arch).b_frag_len
    except ValueError as e:
        return str(e)
    tr_lanes = 8 if spec.wave_size == 32 else 4
    if b_per_lane % tr_lanes:
        tr_insn = "ds_load_tr16_b128" if spec.wave_size == 32 else "ds_read_tr16_b64"
        return (
            f"lds_k_outer needs a B fragment length that is a multiple of "
            f"{tr_lanes} ({tr_insn} returns {tr_lanes} elements per lane on "
            f"wave{spec.wave_size}); got b_per_lane={b_per_lane} for atom "
            f"{spec.warp_tile_m}x{spec.warp_tile_n}x{spec.warp_tile_k}"
        )
    return None


def is_valid_dgrad_spec(spec: DgradConvSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``."""
    from rocke.core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    p = spec.problem
    if p.is_3d:
        return False, "dgrad only supports 2-D convolution currently"
    if spec.max_sub_gemms < 1:
        return False, "max_sub_gemms must be at least 1"
    # Before the record count: a patch record exists for patch problems only.
    _patch_why = _patch_conflict(spec)
    if _patch_why:
        return False, _patch_why
    _n_sub = len(spec.compute_sub_gemms())
    if _n_sub > spec.max_sub_gemms:
        return False, (
            f"this problem decomposes into {_n_sub} tilde sub-GEMMs but the "
            f"kernel's record table supports at most {spec.max_sub_gemms} "
            f"(max_sub_gemms); raise max_sub_gemms to build a kernel that "
            f"can serve it"
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
    # The default epilogue stores scalar. Where a path happens to ignore
    # vector_size_c (split-K atomic, the strided tilde store) the width is
    # still rejected rather than silently dropped, so a spec never names a
    # store width its kernel does not use -- and the AOT cache never holds two
    # binaries that differ only in an ignored width.
    if (
        spec.vector_size_c is not None
        and spec.vector_size_c > 1
        and spec.epilogue == "default"
    ):
        return False, (
            f"default epilogue is not supported with vector size c: {spec.vector_size_c}"
        )
    _why = vector_width_reason(
        (
            ("a", spec.vector_size_a, spec.data.dtype_a),
            ("b", spec.vector_size_b, spec.data.dtype_b),
            ("c", spec.vector_size_c, spec.data.dtype_d),
        )
    )
    if _why is not None:
        return False, _why

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
    if spec.patch and family != "mma":
        return False, f"patch is CDNA-only (got family {family!r} on {arch})"
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

    # The dgrad builder has no direct global->LDS path, no K-unrolled loop and
    # no chiplet-swizzled tile order: these knobs used to be accepted and then
    # silently ignored (async_dma / unroll_k still doubled the charged LDS,
    # lowering occupancy for nothing). Reject them so a sweep cannot report a
    # kernel that differs from what its label says. (The WMMA block below gave
    # the same answer for its family; this covers both.)
    for flag, label in (
        (spec.async_dma, "async_dma"),
        (spec.unroll_k, "unroll_k"),
        (spec.chiplet_swizzle, "chiplet_swizzle"),
    ):
        if flag:
            return False, (
                f"dgrad does not implement {label} (the builder has no such path; "
                f"the flag would be silently ignored)"
            )

    # Accumulator footprint per lane. At 512 fp32 accumulators the tile alone
    # fills a wave64's whole register file (256 VGPRs + 256 AGPRs): the
    # compiler spills to scratch, and such a spilled kernel has been observed
    # to return wrong dX. A correctness guard, not a no-spill guarantee: some
    # configs at 256 accumulators still spill (they compute correct dX). The
    # AOT grid already stops at the same bound (CACHE_MAX_ACC_REGS); this
    # covers the JIT and dispatch paths.
    _acc_regs = _acc_regs_per_lane(spec)
    if _acc_regs > _MAX_ACC_REGS_PER_LANE:
        return False, (
            f"accumulator tile needs {_acc_regs} fp32 registers per lane "
            f"(> {_MAX_ACC_REGS_PER_LANE}); it fills the whole register file"
        )

    if spec.lds_k_outer:
        # This gate has NO counterpart in validate() and that split is the whole
        # point: without it an older target builds cleanly and emits
        # ds_read_tr16_b64, surfacing as an assembler failure far from the cause.
        if arch not in _LDS_K_OUTER_ARCH_WAVE:
            return False, (
                f"lds_k_outer requires one of {sorted(_LDS_K_OUTER_ARCH_WAVE)} "
                f"(the LDS transpose read); got {arch}"
            )
        if spec.wave_size != _LDS_K_OUTER_ARCH_WAVE[arch]:
            # The lane mapping is derived per wave size, so the arch and the
            # wave must agree or the formula addresses a layout the hardware
            # does not implement.
            return False, (
                f"lds_k_outer on {arch} requires wave_size="
                f"{_LDS_K_OUTER_ARCH_WAVE[arch]}; got {spec.wave_size}"
            )
        # Soft mirrors of the validate() gates, so sweep drivers get a reason
        # string instead of swallowing a builder ValueError into a silent skip.
        if spec.lds_layout is not None:
            return False, "lds_k_outer does not honour an explicit lds_layout"
        if spec.async_dma:
            return False, "lds_k_outer is not supported with async_dma on dgrad"
        if spec.pipeline == "wavelet":
            return False, (
                "lds_k_outer is not supported with pipeline='wavelet' on dgrad "
                "(the wavelet loader writes the B tile M-outer into a K-outer "
                "allocation)"
            )
        _kouter_why = _kouter_frag_reason(spec, arch)
        if _kouter_why is not None:
            return False, _kouter_why

    _ab_bytes, _c_lds, _total_lds = _dgrad_lds_charge(spec)
    if not target.fits_lds(_total_lds):
        return False, (
            f"LDS budget {_total_lds} bytes (A/B={_ab_bytes}, C={_c_lds}) "
            f"> {target.lds_capacity_bytes} cap on {arch}"
        )

    _ukd_why = _uniform_k_decode_conflict(spec)
    if _ukd_why:
        return False, _ukd_why
    _tapk_why = _tap_outer_k_conflict(spec)
    if _tapk_why:
        return False, _tapk_why
    _halo_why = _dy_halo_conflict(spec)
    if _halo_why:
        return False, _halo_why
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

    # The dY tile loader. An explicit vector_size_a is used verbatim (split_k
    # <= 1; the split path loads scalar) and has to split the tile evenly over
    # the block's threads, or the builder only finds out halfway through a
    # build. B's width is chosen by the loader itself, and the wavelet loaders
    # pick their own, so neither can fail this way.
    if (
        spec.pipeline != "wavelet"
        and spec.split_k <= 1
        and spec.vector_size_a is not None
    ):
        why = coalesced_load_reason(
            "A", spec.tile_m, spec.tile_k, spec.block_size, spec.vector_size_a
        )
        if why is not None:
            return False, why

    return True, "ok"


def _dgrad_mma_family(arch: str) -> str:
    from rocke.core.arch import ArchTarget

    return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma"


def _resolve_dgrad_op(spec: DgradConvSpec, arch: str):
    from rocke.core.arch import ArchTarget

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
    *,
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
        from dataclasses import replace as _dc_replace

        from kernels.common.conv_split_k import resolve_conv_split_k

        resolved = resolve_conv_split_k(
            "dgrad",
            spec.problem,
            arch=arch,
            dtype=spec.data.dtype_d,
            tile_m=spec.tile_m,
            tile_n=spec.tile_n,
            tile_k=spec.tile_k,
            block_size=spec.block_size,
        )
        spec = _dc_replace(spec, split_k=resolved.split_k)

    spec.validate()
    ok, why = is_valid_dgrad_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid dgrad spec for {arch}: {why}")

    return _build_tilde_dgrad(spec, arch)


_RECORD_FIELDS = 22  # number of i32 fields per SubGemmRecord
_MAGIC_FIELDS = 8  # i32 fields per magic-division table entry


def _emit_binary_search(
    b: IRBuilder,
    flat_block_id: Value,
    sub_gemm_buf: Value,
    num_sub_gemms_val: Value,
    max_sub_gemms: int,
) -> Value:
    """Emit a binary search over the record buffer's block_starts.

    Returns the sub-GEMM index (i32) for the CTA at ``flat_block_id``. The
    ``block_start`` field is at offset 0 in each record.

    The *bound* is the runtime ``num_sub_gemms_val`` kernarg -- the tilde
    decomposition produces a different record count per stride/dilation, so
    an AOT kernel cannot fold it in. So is the *trip count*: each halving
    leaves at most ``ceil(size / 2)`` records, so ``ceil(log2(n))`` of them
    bracket ``n`` records, and a uniform loop runs exactly that many. A
    stride-1 problem (one record) skips the search outright instead of
    waiting out a chain of dependent scalar loads sized for the largest
    decomposition. ``max_sub_gemms`` only bounds the trip-count computation.
    """
    import math

    c0 = b.const_i32(0)
    c1 = b.const_i32(1)
    c_record_stride = b.const_i32(_RECORD_FIELDS)

    # ceil(log2(n)) = the number of powers of two 2**t (t >= 0) below n.
    n_iters = c0
    for t in range(int(math.ceil(math.log2(max(max_sub_gemms, 2))))):
        below = b.cmp_lt(b.const_i32(1 << t), num_sub_gemms_val)
        n_iters = b.add(n_iters, b.select(below, c1, c0))

    search = b.scf_for_iter(
        c0,
        n_iters,
        c1,
        [("bs_lo", c0), ("bs_hi", num_sub_gemms_val)],
        iv_name="bs_it",
    )
    with search as (_it, (lo, hi)):
        mid = b.div(b.add(lo, hi), b.const_i32(2))
        mid_block_start_offset = b.mul(mid, c_record_stride)
        mid_block_start = b.global_load_i32(sub_gemm_buf, mid_block_start_offset)
        take_lo = b.cmp_le(mid_block_start, flat_block_id)
        b.scf_yield(b.select(take_lo, mid, lo), b.select(take_lo, hi, mid))

    return search.results[0]


@dataclass(frozen=True)
class _MagicDiv:
    """A runtime divisor with its host-computed magic-division pair."""

    value: Value
    mult: Value
    shift: Value


def _magic_divmod(b: IRBuilder, n: Value, d: _MagicDiv) -> Tuple[Value, Value]:
    """``(n // d, n % d)`` through ``d``'s magic pair.

    ``n`` must be non-negative and below 2**31 (the magic-division range).
    """
    q = do_magic_division_dynamic(b, n, d.mult, d.shift)
    return q, b.sub(n, b.mul(q, d.value))


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
    arithmetic for tensor offsets. At stride 1 the single record is rebuilt
    from the kernargs instead (``DgradConvSpec.folds_sub_gemm_record``).

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

    # split_k>1 uses atomic_add; tilde split_k=1 uses direct store — both allow writeonly.
    _dx_writeonly = spec.split_k <= 1

    # ---- Runtime problem-dimension parameters (AOT) -------------------------
    # Emitted straight from the ordered ABI list so the kernel's parameter
    # order and conv_args_signature(direction="dgrad") cannot drift apart. Kernargs
    # are packed positionally (see rocke.runtime.packing), so a divergence
    # would silently shift every later argument.
    # library/tests/test_conv_abi.py pins the two together.
    params = emit_param_block(
        b,
        conv_arg_names(direction="dgrad"),
        declare_ptr=lambda name, kind: (
            b.param(
                name,
                PtrType(ir_dtype_a, "global"),
                noalias=True,
                readonly=True,
                align=16,
            )
            if kind == "a"
            else (
                b.param(
                    name,
                    PtrType(ir_dtype_b, "global"),
                    noalias=True,
                    readonly=True,
                    align=16,
                )
                if kind == "b"
                else (
                    b.param(
                        name,
                        PtrType(ir_dtype_d, "global"),
                        noalias=True,
                        writeonly=_dx_writeonly,
                        align=16,
                    )
                    if kind == "d"
                    else b.param(
                        name,
                        PtrType(I32, "global"),
                        noalias=True,
                        readonly=True,
                        align=4,
                    )
                )
            )
        ),
    )
    dY = params["dY"]
    W = params["W"]
    dX = params["dX"]
    dY_bytes = params["dY_bytes"]
    W_bytes = params["W_bytes"]
    dX_bytes = params["dX_bytes"]
    p_N = params["p_N"]
    p_Hi = params["p_Hi"]
    p_Wi = params["p_Wi"]
    p_C = params["p_C"]
    p_K = params["p_K"]
    p_Y = params["p_Y"]
    p_X = params["p_X"]
    p_groups = params["p_groups"]
    p_Ho = params["p_Ho"]
    p_Wo = params["p_Wo"]
    p_cpg = params["p_cpg"]
    p_kpg = params["p_kpg"]
    p_dg_M = params["p_dg_M"]  # N*Hi*Wi (output spatial)
    p_dg_N = params["p_dg_N"]  # cpg (output channels per group)
    p_dg_K = params["p_dg_K"]  # Y*X*kpg (reduction)
    p_num_pid_m = params["p_num_pid_m"]
    p_num_pid_n = params["p_num_pid_n"]
    sub_gemm_buf = params["sub_gemm_buf"]
    num_sub_gemms_param = params["num_sub_gemms"]
    # -------------------------------------------------------------------------

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

    # XCD-contiguous tile order (see xcd_tile_order_xcds): remap the
    # launch-order linear id over (group, tile) so each XCD receives one
    # contiguous range, when the kernargs ask for it, and split it back into
    # the tile and the group. Off, the select keeps the linear id, whose split
    # is (block_id_x, block_id_y) again.
    xcd_group: Value | None = None
    num_xcds = xcd_tile_order_xcds(spec, arch)
    if num_xcds:
        flat_block_id, xcd_group = emit_xcd_tile_order(
            b, flat_block_id, params, num_xcds=num_xcds
        )

    fold_record = spec.folds_sub_gemm_record
    if fold_record:
        # Stride 1 (DgradConvSpec.folds_sub_gemm_record): the only sub-GEMM's
        # record is rebuilt from the kernargs, so there is nothing to search
        # and nothing to load.
        sg_idx = None
        c0_rec = b.const_i32(0)
        c1_rec = b.const_i32(1)
        _rec_kernargs = {
            0: c0_rec,  # block_start
            1: p_num_pid_m,  # num_m_tiles
            2: p_num_pid_n,  # num_n_tiles
            3: p_dg_M,  # gemm_m = N*Hi*Wi
            4: p_dg_K,  # gemm_k = Y*X*kpg
            5: p_Hi,  # h_tilde_slice
            6: p_Wi,  # w_tilde_slice
            7: params["p_pH"],  # h_tilde_slice_begin
            8: params["p_pW"],  # w_tilde_slice_begin
            9: p_Y,  # y_dot_slice
            10: p_X,  # x_dot_slice
            13: c1_rec,  # b_y_stride
            14: c0_rec,  # b_y_offset
            15: c1_rec,  # b_x_stride
            16: c0_rec,  # b_x_offset
            17: c1_rec,  # d_h_stride
            18: c0_rec,  # d_h_offset
            19: c1_rec,  # d_w_stride
            20: c0_rec,  # d_w_offset
        }

        def _ld(field_idx: int) -> Value:
            if field_idx == 11:  # a_embed_h_coeff = -dH
                return b.sub(c0_rec, params["p_dH"])
            if field_idx == 12:  # a_embed_w_coeff = -dW
                return b.sub(c0_rec, params["p_dW"])
            if field_idx == 21:  # gemm_k_padded: dg_K rounded to tile_k*split_k
                c_kq = b.const_i32(spec.tile_k * spec.split_k)
                c_kq1 = b.const_i32(spec.tile_k * spec.split_k - 1)
                return b.mul(b.div(b.add(p_dg_K, c_kq1), c_kq), c_kq)
            return _rec_kernargs[field_idx]

    else:
        # Binary search to find which sub-GEMM this CTA belongs to. The bound
        # is the runtime kernarg; only the unroll depth is fixed at build time.
        sg_idx = _emit_binary_search(
            b, flat_block_id, sub_gemm_buf, num_sub_gemms_param, spec.max_sub_gemms
        )

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

    # Magic-division pairs of this record's divisors (see
    # sub_gemm_magic_pairs); the table starts after the last record. With
    # the folded record the table holds one entry, right after the one
    # record, so its loads take constant offsets and depend on nothing.
    if fold_record:

        def _ld_magic(j: int) -> Value:
            return b.global_load_i32(sub_gemm_buf, b.const_i32(_RECORD_FIELDS + j))

    else:
        _magic_base = b.add(
            b.mul(num_sub_gemms_param, b.const_i32(_RECORD_FIELDS)),
            b.mul(sg_idx, b.const_i32(_MAGIC_FIELDS)),
        )

        def _ld_magic(j: int) -> Value:
            return b.global_load_i32(sub_gemm_buf, b.add(_magic_base, b.const_i32(j)))

    rec_kpg_magic = (_ld_magic(0), _ld_magic(1))
    rec_xdot_magic = (_ld_magic(2), _ld_magic(3))
    rec_hw_magic = (_ld_magic(4), _ld_magic(5))
    rec_w_magic = (_ld_magic(6), _ld_magic(7))

    # Compute local tile indices within this sub-GEMM. The folded record's
    # block_start is 0, so the subtraction is skipped.
    local_flat = flat_block_id if fold_record else b.sub(flat_block_id, rec_block_start)
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

    # Runtime problem values — replace former compile-time constants.
    c_Ho = p_Ho
    c_Wo = p_Wo
    c_K = p_K
    c_Hi = p_Hi
    c_Wi = p_Wi
    c_C = p_C
    c_Y = p_Y
    c_X = p_X
    c_dg_N = p_cpg
    # patch: the GEMM's N is the filter's (y, x, c) -- Y*X*C columns, a run of
    # X*C contiguous in dX per filter row y, the rows Wi*C apart. Only the
    # store reads these (see DgradConvSpec.patch).
    patch_cols: Optional[Tuple[Value, Value]] = None
    if spec.patch:
        if op.family != "mma":
            raise ValueError("patch dgrad is implemented for MFMA only")
        c_patch_xc = b.mul(p_X, p_C)
        c_dg_N = b.mul(p_Y, c_patch_xc)
        patch_cols = (c_patch_xc, b.mul(p_Wi, p_C))
    # dY row count N*Ho*Wo, used by the pointwise fast path's bounds check.
    c_dY_rows = b.mul(b.mul(p_N, p_Ho), p_Wo)

    grouped = p.groups > 1
    if grouped:
        # Under the XCD-contiguous order the group comes from the remapped id.
        group_idx = b.block_id_y() if xcd_group is None else xcd_group
        c_kpg = p_kpg
        k_out_group_base = b.mul(group_idx, c_kpg)
        c_group_base = b.mul(group_idx, p_cpg)
    else:
        c_kpg = None
        k_out_group_base = None
        c_group_base = None

    # Precompute tiling divisors (runtime, from record).
    # k_sub decomposition order: [ydot, xdot, k_out] (k_out innermost, CK-compatible).
    # k_sub = ydot * XDotSlice * K + xdot * K + k_out
    # Consecutive k_sub → consecutive k_out → contiguous in dY (NHWK, stride-1 in K).
    hw_tilde = b.mul(rec_h_tilde_slice, rec_w_tilde_slice)
    # The record's divisors, each with its magic pair.
    kpg_div = _MagicDiv(c_kpg if grouped else c_K, *rec_kpg_magic)
    xdot_div = _MagicDiv(rec_x_dot_slice, *rec_xdot_magic)
    hw_div = _MagicDiv(hw_tilde, *rec_hw_magic)
    w_div = _MagicDiv(rec_w_tilde_slice, *rec_w_magic)

    # LDS allocation.
    lds_layout = spec.effective_lds_layout()
    # The staged dY halo tile (DgradConvSpec.dy_halo) replaces the A tile; see
    # dy_halo_lds_rows for its shape.
    A_smem = b.smem_alloc(
        ir_dtype_a,
        lds_layout.storage_shape(spec.dy_halo_lds_rows if spec.dy_halo else block_m),
        name_hint="A_smem",
    )
    if spec.lds_k_outer:
        # K-outer: rows are K, columns are the free axis N. See _KOUTER_PAD for
        # why the stride must not be a multiple of the 32-dword bank period.
        b_shape = (block_k, block_n + spec.kouter_b_pad)
    else:
        b_shape = lds_layout.storage_shape(block_n)
    B_smem = b.smem_alloc(ir_dtype_b, b_shape, name_hint="B_smem")
    # dy_halo=2: the second B buffer of the per-tap double buffer.
    B_smem2 = (
        b.smem_alloc(ir_dtype_b, b_shape, name_hint="B_smem2")
        if spec.dy_halo == 2
        else None
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
    if spec.lds_k_outer:
        # Recompute in col mode over the swapped tile. This is provably
        # value-preserving: choose_vec tests tile_rows in "row" mode and
        # tile_cols in "col" mode, so both calls test the same extent (block_n)
        # against the same product (block_n * block_k). load_vec_b is therefore
        # invariant under the swap -- the emitted global buffer_loads are
        # unchanged and ONLY the LDS store instruction changes, from
        # load_vec_b scalar ds_write_b16 to one wide smem_store_vN.
        _max_vb = (
            _def_vec_b
            if spec.vector_size_b is None
            else min(_def_vec_b, spec.vector_size_b)
        )
        load_vec_b = CoalescedTileLoader.choose_vec(
            tile_rows=block_k,
            tile_cols=block_n,
            block_size=threads,
            max_vec=_max_vb,
            vector_axis="col",
        )
        axis_b = "col"
    elif spec.vector_size_b is not None:
        # Clamp, exactly as the K-outer branch above does. vector_size_* is a
        # CAP, not a demand -- wgrad documents it that way and passes
        # vector_size_c through as ``max_store_vec`` -- so an explicit width
        # wider than the tile geometry supports must be narrowed, not obeyed.
        # Taking it verbatim let a spec pass is_valid_dgrad_spec and then raise
        # from CoalescedTileLoader.vecs_per_thread deep in the builder.
        #
        # Emission-neutral for every spec that already built: choose_vec's
        # accepted set is a strict subset of vecs_per_thread's, and the
        # tile_n % (warp_n * warp_tile_n) rule the validator already enforces
        # makes the axis-divisibility condition free, so this returns exactly
        # spec.vector_size_b wherever the verbatim path worked, and a narrower
        # width only where it used to raise.
        load_vec_b = CoalescedTileLoader.choose_vec(
            tile_rows=block_n,
            tile_cols=block_k,
            block_size=threads,
            max_vec=min(_def_vec_b, spec.vector_size_b),
            vector_axis="row",
        )
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

        # Pointwise (Y=X=1, stride 1, pad 0, ungrouped) fast path. The tilde
        # decomposition is the identity here -- h_tilde_slice == Ho,
        # w_tilde_slice == Wo, y_dot_slice == x_dot_slice == 1, gemm_k == kpg --
        # so (n*Ho+ho)*Wo*K + wo*K + k_out reduces exactly to m_sub*K + k_sub.
        # Forward and wgrad both already special-case this; dgrad did not, and
        # the generic form costs a runtime divide plus a tautological bounds
        # predicate INSIDE the K-loop.
        if p.is_pointwise and not grouped:
            off = b_.add(b_.mul(m_sub, c_K), k_sub)
            ok = b_.land(
                b_.cmp_lt(m_sub, c_dY_rows),
                b_.cmp_lt(k_sub, c_K),
            )
            return off, ok

        # Decompose k_sub → (ydot, xdot, k_out)  [k_out innermost, CK-compatible]
        # k_sub = ydot * xdot_slice * kpg + xdot * kpg + k_out.  The reduction is
        # per-group, so the decode divisor is kpg (== K when ungrouped) and k_out
        # is group-local in [0, kpg); the absolute NHWK channel is g*kpg + k_out.
        # Consecutive k_sub → consecutive k_out → contiguous in dY (NHWK, last dim
        # K) within one group. Condition for vector loads: kpg % load_vec_a == 0.
        _kdiv = kpg_div.value
        yx_rem, k_out = _magic_divmod(b_, k_sub, kpg_div)
        ydot, xdot = _magic_divmod(b_, yx_rem, xdot_div)
        k_out_abs = b_.add(k_out, k_out_group_base) if grouped else k_out

        # Decompose m_sub → (n, htilde_local, wtilde_local)
        n, m_rem = _magic_divmod(b_, m_sub, hw_div)
        htl, wtl = _magic_divmod(b_, m_rem, w_div)

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

        # Pointwise fast path: Y == X == 1 means y == x == 0, so KYXC is just
        # [K, cpg] and the offset is k_sub*C + c_val. Must stay in lockstep with
        # the dy_descriptor fast path above.
        if p.is_pointwise and not grouped:
            off = b_.add(b_.mul(k_sub, c_C), c_val)
            ok = b_.land(b_.cmp_lt(k_sub, c_K), b_.cmp_lt(c_val, c_C))
            return off, ok

        # Same k_out-innermost decomposition as dy_descriptor (must match).
        # c (row axis) is stride-1 in KYXC; vectorised loads along c use
        # vector_axis="row".  The reduction is per-group, so the decode divisor is
        # kpg (== K when ungrouped) and k_out is group-local in [0, kpg).
        _kdiv = kpg_div.value
        yx_rem, k_out = _magic_divmod(b_, k_sub, kpg_div)
        ydot, xdot = _magic_divmod(b_, yx_rem, xdot_div)
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
    # Flipping the tile makes the free axis the column axis, so _store_tile
    # takes its existing wide-store branch. No change to loads.py is required.
    _b_tile_rows, _b_tile_cols = (
        (block_k, block_n) if spec.lds_k_outer else (block_n, block_k)
    )
    b_sync_loader = CoalescedTileLoader(
        tile_rows=_b_tile_rows,
        tile_cols=_b_tile_cols,
        block_size=threads,
        load_vec=load_vec_b,
        elem_dtype=ir_dtype_b,
        vector_axis=axis_b,
    )
    if spec.pipeline == "wavelet":
        a_wavelet_loader, b_wavelet_loader = _build_wavelet_loaders(
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

    if spec.lds_k_outer:

        def _b_desc_fn(b_: IRBuilder, row: Value, col: Value):
            return w_descriptor(b_, col, row)

    else:
        _b_desc_fn = w_descriptor

    # The fragment length is per-atom, not a constant: on wave64 it is 8 for
    # 32x32x16 and the MFMA 16x16x32, 4 for 16x16x16 and 32x32x8; on wave32 the
    # WMMA 16x16x32 carries 16. Hardcoding a stride would read past the end of
    # the K-outer tile and return garbage.
    #
    # The required multiple is the width the transpose read returns per lane,
    # which differs by regime: ds_read_tr16_b64 returns 4, ds_load_tr16_b128
    # returns 8. Checking 4 on wave32 would admit a fragment length of 4 or 12,
    # where the ``n // 8`` read loop is empty (or truncates) and the fragment is
    # built from an empty ``parts`` list -- an IndexError at build time rather
    # than a wrong answer, but the guard should state the real invariant.
    _tr_lanes = 8 if spec.wave_size == 32 else 4
    _tr_insn = "ds_load_tr16_b128" if spec.wave_size == 32 else "ds_read_tr16_b64"
    if spec.lds_k_outer and b_per_lane % _tr_lanes != 0:
        raise ValueError(
            f"lds_k_outer needs a B fragment length that is a multiple of "
            f"{_tr_lanes} ({_tr_insn} returns {_tr_lanes} elements per lane on "
            f"wave{spec.wave_size}); got b_per_lane={b_per_lane} for atom "
            f"{spec.warp_tile_m}x{spec.warp_tile_n}x{spec.warp_tile_k}"
        )

    # Materialised only on the K-outer path: emitting these unconditionally
    # would add IR ops to every existing config and move every dgrad golden.
    if spec.lds_k_outer:
        _tr_reader = ConvKOuterFragmentReader(wave_size=spec.wave_size).bind(b, lane)

    def _tr_frag(smem: Value, mn_base: Value, k_base: Value, mn_atom: int, n: int):
        """One MMA operand fragment from a K-outer tile via transpose reads.

        Thin binding of :class:`ConvKOuterFragmentReader`, which owns the lane
        mapping for both wave regimes and is shared with the other backward
        instance. The mapping is the part that is easy to get subtly wrong --
        two engines agreeing on the same wrong formula still reads a transposed
        operand -- so it lives in one place, mirroring ``rocke_conv_tr_frag`` in
        the C++ engine.
        """
        return _tr_reader.fragment(
            b,
            smem,
            mn_base,
            k_base,
            mn_atom=mn_atom,
            n=n,
            dtype=_smem_dtype if _smem_dtype is not None else F16,
        )

    def emit_load_phase(k_off: Value, A_dst: Value, B_dst: Value) -> None:
        k_off_capture[0] = k_off
        a_sync_loader.load(
            b, tid=tid, smem_dst=A_dst, descriptor=dy_descriptor, rsrc=dy_rsrc
        )
        # The K-outer tile is indexed (k, free) while w_descriptor takes
        # (free, k). w_descriptor itself is untouched, so the global addressing
        # and its OOB select stay byte-identical -- this is a swap, not a
        # redesign.
        b_sync_loader.load(
            b, tid=tid, smem_dst=B_dst, descriptor=_b_desc_fn, rsrc=w_rsrc
        )

    def emit_mfma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value], a_row_map=None
    ) -> List[Value]:
        """One K tile of MMAs. ``a_row_map`` (MFMA only, the dY halo loop)
        rewrites each A fragment row before its LDS read."""
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
                    if spec.lds_k_outer:
                        # B only: dgrad's A tile is genuinely still M-outer.
                        b_cols.append(
                            _tr_frag(
                                B_src,
                                atom_row,
                                k_tile_base,
                                spec.warp_tile_n,
                                b_per_lane,
                            )
                        )
                        continue
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
                if a_row_map is not None:
                    a_row = a_row_map(a_row)
                a_rows.append(
                    _emit_smem_load(
                        b, A_src, a_row, col_base, a_per_lane, smem_dtype=_smem_dtype
                    )
                )
            b_cols = []
            for ni in range(mfmas_n):
                if spec.lds_k_outer:
                    b_cols.append(
                        _tr_frag(
                            B_src,
                            b.add(warp_n_off, b.const_i32(ni * spec.warp_tile_n)),
                            b.const_i32(kk * spec.warp_tile_k),
                            spec.warp_tile_n,
                            b_per_lane,
                        )
                    )
                    continue
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
            hw_tilde=hw_div,
            w_tilde_slice=w_div,
            d_h_stride=rec_d_h_stride,
            d_h_offset=rec_d_h_offset,
            d_w_stride=rec_d_w_stride,
            d_w_offset=rec_d_w_offset,
            c_Hi=c_Hi,
            c_Wi=c_Wi,
            c_C=c_C,
            c_group_base=c_group_base,
        )
        if patch_cols is not None:
            _tilde_kwargs["patch_cols"] = patch_cols
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
                hw_div,
                w_div,
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
                        params,
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
                        params,
                        c_group_base=c_group_base,
                    )
            elif use_cshuffle:
                _emit_dgrad_cshuffle_epilogue(
                    b,
                    spec,
                    final_accs,
                    grid,
                    dx_rsrc,
                    params,
                    c_group_base=c_group_base,
                )
            else:
                _emit_dgrad_direct_epilogue(
                    b,
                    spec,
                    final_accs,
                    grid,
                    dx_rsrc,
                    params,
                    c_group_base=c_group_base,
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

    def _emit_uniform_k_decode_loops() -> List[Value]:
        """The ``uniform_k_decode`` K loops. Returns the final accumulators.

        Loop 1 runs when kpg is a multiple of tile_k. It carries the K tile's
        decoded reduction coordinate (k_out, xdot, ydot) -- uniform across the
        workgroup -- and advances it by one tile per iteration. Every element
        of the tile then has k_out = tile k_out + column, so each load offset is
        a loop-invariant per-element part plus a uniform per-tile part, and the
        A bounds compare the invariant (ho, wo) base against shifted uniform
        limits. The k_out < kpg term of the plain descriptors is always true
        here and is dropped; B keeps its (y, x) test, which is what zeroes the
        split-K tiles past gemm_k.

        Loop 2 is the plain K loop and runs otherwise. Exactly one of the two
        has a non-empty range, chosen per problem from the kernargs.
        """
        c_kd = c_kpg if grouped else c_K
        c_xs = rec_x_dot_slice
        c1 = b.const_i32(1)
        uk_ok = b.cmp_eq(b.mod(c_kd, c_block_k), c0)
        q_lo = b.div(k_lo, c_kd)
        uk_k0 = b.mod(k_lo, c_kd)
        uk_x0 = b.mod(q_lo, c_xs)
        uk_y0 = b.div(q_lo, c_xs)
        # dY (NHWK): ho moves by ydot*a_embed_h_coeff rows of Wo*K, wo by
        # xdot*a_embed_w_coeff rows of K.
        wo_k = b.mul(c_Wo, c_K)
        a_step_y = b.mul(rec_a_embed_h_coeff, wo_k)
        a_step_x = b.mul(rec_a_embed_w_coeff, c_K)
        # W (KYXC, per-group packed): offset = k*Y*X*Cs + y*X*Cs + x*Cs + c.
        c_cs = c_dg_N if grouped else c_C
        b_step_y = b.mul(c_X, c_cs)
        b_step_k = b.mul(c_Y, b_step_y)
        uk_hi = b.select(uk_ok, k_hi, k_lo)

        uk_cap: List[Optional[Tuple[Value, ...]]] = [None]

        def tile_uniforms(ko: Value, xd: Value, yd: Value) -> Tuple[Value, ...]:
            a_h = b.mul(yd, rec_a_embed_h_coeff)
            a_w = b.mul(xd, rec_a_embed_w_coeff)
            ho_lo = b.sub(c0, a_h)
            ho_hi = b.sub(c_Ho, a_h)
            wo_lo = b.sub(c0, a_w)
            wo_hi = b.sub(c_Wo, a_w)
            a_off = b.add(b.add(ko, b.mul(yd, a_step_y)), b.mul(xd, a_step_x))
            if grouped:
                a_off = b.add(a_off, k_out_group_base)
            y = b.add(b.mul(yd, rec_b_y_stride), rec_b_y_offset)
            x = b.add(b.mul(xd, rec_b_x_stride), rec_b_x_offset)
            k_abs = b.add(ko, k_out_group_base) if grouped else ko
            b_off = b.add(
                b.mul(k_abs, b_step_k), b.add(b.mul(y, b_step_y), b.mul(x, c_cs))
            )
            b_ok = b.land(b.cmp_lt(y, c_Y), b.cmp_lt(x, c_X))
            return (ho_lo, ho_hi, wo_lo, wo_hi, a_off, b_off, b_ok)

        def dy_uk(b_: IRBuilder, row: Value, col: Value):
            ho_lo, ho_hi, wo_lo, wo_hi, a_off = uk_cap[0][:5]
            m_sub = b_.add(block_m_off_v, row)
            n, m_rem = _magic_divmod(b_, m_sub, hw_div)
            htl, wtl = _magic_divmod(b_, m_rem, w_div)
            ho_b = b_.add(htl, rec_h_tilde_slice_begin)
            wo_b = b_.add(wtl, rec_w_tilde_slice_begin)
            valid = b_.land(
                b_.land(b_.cmp_ge(ho_b, ho_lo), b_.cmp_lt(ho_b, ho_hi)),
                b_.land(b_.cmp_ge(wo_b, wo_lo), b_.cmp_lt(wo_b, wo_hi)),
            )
            row_off = b_.add(
                b_.mul(b_.add(b_.mul(n, c_Ho), ho_b), wo_k), b_.mul(wo_b, c_K)
            )
            offset = b_.add(b_.add(row_off, col), a_off)
            return b_.select(valid, offset, c0), valid

        def w_uk(b_: IRBuilder, row: Value, col: Value):
            b_off, b_ok = uk_cap[0][5:]
            c_val = b_.add(block_n_off_v, row)
            offset = b_.add(b_.add(b_.mul(col, b_step_k), c_val), b_off)
            return b_.select(b_ok, offset, c0), b_ok

        if spec.lds_k_outer:

            def w_uk_fn(b_: IRBuilder, row: Value, col: Value):
                return w_uk(b_, col, row)

        else:
            w_uk_fn = w_uk

        n_acc = len(accs)
        uk_iter = list(accs) + [("uk_k", uk_k0), ("uk_x", uk_x0), ("uk_y", uk_y0)]
        for_uk = b.scf_for_iter(k_lo, uk_hi, c_block_k, uk_iter, iv_name="k0")
        with for_uk as (_k0, iter_vars):
            ko, xd, yd = iter_vars[n_acc:]
            uk_cap[0] = tile_uniforms(ko, xd, yd)
            # Every global read of the tile before the first LDS write. With
            # the address math this short there is nothing left to interleave
            # with, and a read-then-write per chunk lets the register allocator
            # recycle one staging register across all chunks, which serialises
            # the reads on their full latency.
            a_staged = a_sync_loader.load_global(
                b, tid=tid, descriptor=dy_uk, rsrc=dy_rsrc
            )
            b_staged = b_sync_loader.load_global(
                b, tid=tid, descriptor=w_uk_fn, rsrc=w_rsrc
            )
            a_sync_loader.store_lds(b, smem_dst=A_smem, staged=a_staged)
            b_sync_loader.store_lds(b, smem_dst=B_smem, staged=b_staged)
            b.sync()
            new_accs = emit_mfma_phase(A_smem, B_smem, list(iter_vars[:n_acc]))
            b.sync()
            # Next tile: k_out wraps into the next filter tap.
            nk = b.add(ko, c_block_k)
            wrap_k = b.cmp_ge(nk, c_kd)
            nk = b.select(wrap_k, c0, nk)
            nx = b.add(xd, b.select(wrap_k, c1, c0))
            wrap_x = b.cmp_ge(nx, c_xs)
            nx = b.select(wrap_x, c0, nx)
            ny = b.add(yd, b.select(wrap_x, c1, c0))
            b.scf_yield(*new_accs, nk, nx, ny)
        uk_accs = list(for_uk.results)[:n_acc]

        fb_lo = b.select(uk_ok, k_hi, k_lo)
        fb_iter = [(f"fb_{name}", v) for (name, _), v in zip(accs, uk_accs)]
        for_fb = b.scf_for_iter(fb_lo, k_hi, c_block_k, fb_iter, iv_name="k1")
        with for_fb as (k1, iter_vars):
            emit_load_phase(k1, A_smem, B_smem)
            b.sync()
            new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)
            b.sync()
            b.scf_yield(*new_accs)
        return list(for_fb.results)

    def _emit_tap_outer_loops() -> list[Value]:
        """The ``tap_outer_k`` K loops. Returns the final accumulators.

        Filter tap (y, x) outer, output-channel chunk inner. At stride 1 and
        dilation 1 tap (y, x) reads dY pixel (hi + dh, wi + dw), dh = pH - y,
        dw = pW - x: the tap decode is uniform arithmetic on the kernargs,
        once per tap; the pixel decode (n, hi, wi), and so a tap's dY bounds,
        are invariant across the chunk loop; and each chunk adds one uniform
        offset per operand (kb on dY, w_k on W). kpg is a multiple of tile_k
        (validator and cache), so a chunk never straddles a tap, needs no k
        bound, and the W offset needs no predicate. Same reduction order as
        the flat K loop.
        """
        c1 = b.const_i32(1)
        c_kd = c_kpg if grouped else c_K
        c_cs = c_dg_N if grouped else c_C
        c_taps = b.mul(c_Y, c_X)
        c_yxc = b.mul(c_taps, c_cs)
        tap_cap: dict[str, Value] = {}

        def dy_tap(b_: IRBuilder, row: Value, col: Value):
            # NHWK: ((n*Ho + hi)*Wo + wi)*K + col, plus the tap's pixel shift
            # (dh*Wo + dw)*K and the chunk base. An out-of-image lane is sent
            # to the loader's OOB sentinel by ``valid``.
            m_sub = b_.add(block_m_off_v, row)
            n, m_rem = _magic_divmod(b_, m_sub, hw_div)
            hi, wi = _magic_divmod(b_, m_rem, w_div)
            ho = b_.add(hi, tap_cap["dh"])
            wo = b_.add(wi, tap_cap["dw"])
            valid = b_.land(
                b_.land(b_.cmp_ge(ho, c0), b_.cmp_lt(ho, c_Ho)),
                b_.land(b_.cmp_ge(wo, c0), b_.cmp_lt(wo, c_Wo)),
            )
            pix = b_.add(
                b_.mul(
                    b_.add(b_.mul(b_.add(b_.mul(n, c_Ho), hi), c_Wo), wi),
                    c_K,
                ),
                col,
            )
            offset = b_.add(b_.add(pix, tap_cap["dy_tap"]), tap_cap["kb"])
            return offset, valid

        def w_tap(b_: IRBuilder, row: Value, col: Value):
            # KYXC (per-group packed): (k_abs*Y*X + tap)*Cs + c with
            # k_abs = kb + col; w_k = kb*Y*X*Cs + tap*Cs.
            c_val = b_.add(block_n_off_v, row)
            offset = b_.add(b_.add(b_.mul(col, c_yxc), c_val), tap_cap["w_k"])
            return offset, None

        if spec.lds_k_outer:

            def w_tap_fn(b_: IRBuilder, row: Value, col: Value):
                return w_tap(b_, col, row)

        else:
            w_tap_fn = w_tap

        tap_for = b.scf_for_iter(c0, c_taps, c1, accs, iv_name="tap")
        with tap_for as (tap, tap_vars):
            ydot = b.div(tap, c_X)
            xdot = b.sub(tap, b.mul(ydot, c_X))
            tap_cap["dh"] = b.sub(params["p_pH"], ydot)
            tap_cap["dw"] = b.sub(params["p_pW"], xdot)
            tap_cap["dy_tap"] = b.mul(
                b.add(b.mul(tap_cap["dh"], c_Wo), tap_cap["dw"]), c_K
            )
            tap_cs = b.mul(tap, c_cs)
            chunk_accs = [(f"{name}_t", v) for (name, _), v in zip(accs, tap_vars)]
            chunk_for = b.scf_for_iter(c0, c_kd, c_block_k, chunk_accs, iv_name="kb")
            with chunk_for as (kb0, iter_vars):
                kb = b.add(kb0, k_out_group_base) if grouped else kb0
                tap_cap["kb"] = kb
                tap_cap["w_k"] = b.add(b.mul(kb, c_yxc), tap_cs)
                # Every global read of the tile before the first LDS write.
                a_staged = a_sync_loader.load_global(
                    b, tid=tid, descriptor=dy_tap, rsrc=dy_rsrc
                )
                b_staged = b_sync_loader.load_global(
                    b, tid=tid, descriptor=w_tap_fn, rsrc=w_rsrc
                )
                a_sync_loader.store_lds(b, smem_dst=A_smem, staged=a_staged)
                b_sync_loader.store_lds(b, smem_dst=B_smem, staged=b_staged)
                b.sync()
                new_accs = emit_mfma_phase(A_smem, B_smem, list(iter_vars))
                b.sync()
                b.scf_yield(*new_accs)
            b.scf_yield(*chunk_for.results)
        return list(tap_for.results)

    def _emit_dy_halo_loop() -> list[Value]:
        """The ``dy_halo`` K loop. Returns the final accumulators.

        Output-channel chunk outer, filter tap inner (unrolled over the baked
        Y x X). Per chunk the dY range of the tile -- dX pixels [m0, m0 +
        tile_m) shifted by every tap -- is loaded into ``A_smem`` once: LDS
        row r holds linear dY pixel m0 + min_shift + r, min_shift = (pH - Y +
        1)*Wo + (pW - X + 1) the most negative tap shift, and tap (y, x) reads
        fragment row a_row + (Y-1-y)*Wo + (X-1-x). A lane whose pixel (hi +
        dh, wi + dw) leaves the image (dh = pH - y, dw = pW - x) reads the
        trailing all-zero row instead. The pad is baked with the filter: at
        stride 1, Ho == Hi and Wo == Wi make it ((Y-1)/2, (X-1)/2), so the
        sign of every dh / dw -- which bounds a tap needs -- is fixed. Wo,
        the shifts, the bounds and the loader's row count are kernargs.
        """
        Y, X = p.Y, p.X
        pH, pW = p.pH, p.pW
        lds_rows = spec.dy_halo_lds_rows
        # The last row is the zero row; the loader fills the rest.
        load_rows = lds_rows - 1
        c_kd = c_kpg if grouped else c_K
        c_cs = c_dg_N if grouped else c_C
        c_yxc = b.mul(b.const_i32(Y * X), c_cs)
        halo_loader = CoalescedTileLoader(
            tile_rows=load_rows,
            tile_cols=block_k,
            block_size=threads,
            load_vec=load_vec_a,
            elem_dtype=ir_dtype_a,
        )
        cap: dict[str, Value] = {}
        # min_shift and the rows a tile really needs, both in pixels.
        min_shift = b.add(b.mul(b.const_i32(pH - Y + 1), c_Wo), b.const_i32(pW - X + 1))
        real_rows = b.add(b.mul(b.const_i32(Y - 1), c_Wo), b.const_i32(block_m + X - 1))
        halo_base = b.add(block_m_off_v, min_shift)
        c_zero_row = b.const_i32(lds_rows - 1)

        def halo_dy(b_: IRBuilder, row: Value, col: Value):
            # Linear dY pixel q (N*Ho*Wo == N*Hi*Wi == dg_M pixels); rows past
            # the tile's real extent (Wo < halo_max_wo) load nothing.
            q = b_.add(halo_base, row)
            valid = b_.land(
                b_.land(b_.cmp_ge(q, c0), b_.cmp_lt(q, p_dg_M)),
                b_.cmp_lt(row, real_rows),
            )
            offset = b_.add(b_.add(b_.mul(q, c_K), col), cap["kb"])
            return offset, valid

        def w_halo(b_: IRBuilder, row: Value, col: Value):
            # KYXC (per-group packed): (k_abs*Y*X + tap)*Cs + c with
            # k_abs = kb + col; w_k = kb*Y*X*Cs + tap*Cs.
            c_val = b_.add(block_n_off_v, row)
            offset = b_.add(b_.add(b_.mul(col, c_yxc), c_val), cap["w_k"])
            return offset, None

        if spec.lds_k_outer:

            def w_halo_fn(b_: IRBuilder, row: Value, col: Value):
                return w_halo(b_, col, row)

        else:
            w_halo_fn = w_halo

        # Zero row: every thread writes the same zeros (benign); the first
        # chunk's barrier orders it before any read, and the loader never
        # writes it.
        zvec = min(8, block_k)
        zcol = b.mul(b.mod(tid, b.const_i32(block_k // zvec)), b.const_i32(zvec))
        b.smem_store_vN(A_smem, [c_zero_row, zcol], b.zero_vec(ir_dtype_a, zvec), zvec)

        def row_map(y: int, x: int):
            dh, dw = pH - y, pW - x
            # Scalars of the tap, once: its row shift and the bounds it needs.
            shift = b.add(b.mul(b.const_i32(Y - 1 - y), c_Wo), b.const_i32(X - 1 - x))
            h_hi = b.sub(c_Ho, b.const_i32(dh)) if dh > 0 else None
            w_hi = b.sub(c_Wo, b.const_i32(dw)) if dw > 0 else None

            def _map(a_row: Value) -> Value:
                _, rem = _magic_divmod(b, b.add(block_m_off_v, a_row), hw_div)
                hi, wi = _magic_divmod(b, rem, w_div)
                conds = []
                if dh < 0:
                    conds.append(b.cmp_ge(hi, b.const_i32(-dh)))
                if h_hi is not None:
                    conds.append(b.cmp_lt(hi, h_hi))
                if dw < 0:
                    conds.append(b.cmp_ge(wi, b.const_i32(-dw)))
                if w_hi is not None:
                    conds.append(b.cmp_lt(wi, w_hi))
                row = b.add(a_row, shift)
                if not conds:
                    return row
                ok = conds[0]
                for c_ in conds[1:]:
                    ok = b.land(ok, c_)
                return b.select(ok, row, c_zero_row)

            return _map

        n_taps = Y * X

        def mfma_tap(t: int, b_src: Value, cur: list[Value]) -> list[Value]:
            y, x = divmod(t, X)
            if spec.dy_halo_setprio > 0:
                b.s_setprio(spec.dy_halo_setprio)
            out = emit_mfma_phase(A_smem, b_src, cur, a_row_map=row_map(y, x))
            if spec.dy_halo_setprio > 0:
                b.s_setprio(0)
            return out

        b_bufs = [B_smem] if B_smem2 is None else [B_smem, B_smem2]
        chunk_for = b.scf_for_iter(c0, c_kd, c_block_k, accs, iv_name="kb")
        with chunk_for as (kb0, iter_vars):
            kb = b.add(kb0, k_out_group_base) if grouped else kb0
            cap["kb"] = kb
            w_kb = b.mul(kb, c_yxc)

            def set_tap(t: int) -> None:
                cap["w_k"] = b.add(w_kb, b.mul(b.const_i32(t), c_cs))

            cur = list(iter_vars)
            if spec.dy_halo == 2:
                # Chunk prologue: the halo and tap 0's B tile.
                set_tap(0)
                a_st = halo_loader.load_global(
                    b, tid=tid, descriptor=halo_dy, rsrc=dy_rsrc
                )
                b_st = b_sync_loader.load_global(
                    b, tid=tid, descriptor=w_halo_fn, rsrc=w_rsrc
                )
                halo_loader.store_lds(b, smem_dst=A_smem, staged=a_st)
                b_sync_loader.store_lds(b, smem_dst=b_bufs[0], staged=b_st)
                b.sync()
                for t in range(n_taps):
                    nxt = None
                    if t + 1 < n_taps:
                        # Tap t+1's B tile into registers; the fence keeps
                        # the loads ahead of the MFMAs (the scheduler
                        # otherwise sinks them next to their LDS store).
                        set_tap(t + 1)
                        nxt = b_sync_loader.load_global(
                            b, tid=tid, descriptor=w_halo_fn, rsrc=w_rsrc
                        )
                        b.sched_barrier(0)
                    cur = mfma_tap(t, b_bufs[t % 2], cur)
                    if nxt is not None:
                        b.sched_barrier(0)
                        b_sync_loader.store_lds(
                            b, smem_dst=b_bufs[(t + 1) % 2], staged=nxt
                        )
                    # Orders the store above before tap t+1 reads it, and tap
                    # t's reads of b_bufs[t % 2] before tap t+2 overwrites
                    # it. After the last tap it protects A_smem from the next
                    # chunk's halo.
                    b.sync()
            else:
                for t in range(n_taps):
                    set_tap(t)
                    a_st = None
                    if t == 0:
                        a_st = halo_loader.load_global(
                            b, tid=tid, descriptor=halo_dy, rsrc=dy_rsrc
                        )
                    b_st = b_sync_loader.load_global(
                        b, tid=tid, descriptor=w_halo_fn, rsrc=w_rsrc
                    )
                    if a_st is not None:
                        halo_loader.store_lds(b, smem_dst=A_smem, staged=a_st)
                    b_sync_loader.store_lds(b, smem_dst=b_bufs[0], staged=b_st)
                    b.sync()
                    cur = mfma_tap(t, b_bufs[0], cur)
                    b.sync()
            b.scf_yield(*cur)
        return list(chunk_for.results)

    # ---- K loop ----
    if spec.pipeline == "wavelet":
        # Wavelet load/math wave specialization (gfx1250/WMMA only).
        # The loop runs over this CTA's runtime [k_lo, k_hi) slice, which the
        # tilde decomposition may vary per sub-GEMM; tiles past k_hi are
        # clamped to 0 by the buffer resource.
        n_math_warps = spec.warp_m * spec.warp_n
        b.kernel.attrs["max_workgroup_size"] = spec.launch_block_size
        _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
        _epi_barriers = compute_wavelet_epi_barriers(spec.epilogue, _no_alias)

        def _dgrad_wavelet_epilogue(final_accs_in):
            fa = _apply_accumulator_epilogue(b, spec.acc_epilogue, final_accs_in)
            _dispatch_dgrad_epilogue(fa)

        emit_wavelet_kloop_dynamic(
            b=b,
            warp_id=warp_id,
            tid=tid,
            n_math_warps=n_math_warps,
            math_block_size=spec.block_size,
            k_lo=k_lo,
            k_hi=k_hi,
            block_k=block_k,
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

    if spec.dy_halo:
        final_accs = _emit_dy_halo_loop()
    elif spec.tap_outer_k:
        final_accs = _emit_tap_outer_loops()
    elif spec.uniform_k_decode:
        final_accs = _emit_uniform_k_decode_loops()
    else:
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
    hw_tilde: _MagicDiv,
    w_tilde_slice: _MagicDiv,
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
    from rocke.helpers.atoms import c_warp_params, make_c_warp_dstr_encoding
    from rocke.helpers.distribution import make_static_tile_distribution

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
                n_val, m_rem = _magic_divmod(b, c_m, hw_tilde)
                htl, wtl = _magic_divmod(b, m_rem, w_tilde_slice)

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
    params: Dict[str, Value],
    c_group_base: Optional[Value] = None,
) -> None:
    """Per-lane scalar store to dX via the input-gradient descriptor."""
    dx_addr, bounds = _dgrad_stride1_dx_addr(b, spec, params, c_group_base)
    DirectEpilogue(atom=spec.atom, grid=grid, out_dtype=spec.data.dtype_d).store(
        b,
        accs=accs,
        addr_fn=dx_addr,
        d_rsrc=dx_rsrc,
        bounds=bounds,
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
    params: Dict[str, Value],
    c_group_base: Optional[Value] = None,
) -> None:
    """Per-lane store for the WMMA (gfx1151) accumulator layout into dX."""
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))

    c_M = params["p_dg_M"]
    c_N = params["p_dg_N"]
    dX_desc = make_dgrad_dx_descriptor_dynamic(b, params)
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
    hw_tilde: _MagicDiv,
    w_tilde_slice: _MagicDiv,
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
                n_batch, m_rem = _magic_divmod(b, m_val, hw_tilde)
                htl, wtl = _magic_divmod(b, m_rem, w_tilde_slice)
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
    params: Dict[str, Value],
    c_group_base: Optional[Value] = None,
) -> tuple:
    """Return ``(addr_fn, bounds)`` for the stride-1 dX descriptor.

    Both the descriptor and the bounds come from the AOT kernarg block, so
    one compiled kernel serves any shape.
    """
    dX_desc = make_dgrad_dx_descriptor_dynamic(b, params)

    def dx_addr(b_: IRBuilder, m_val: Value, n_val: Value):
        # n_val is the group-local input channel (< cpg); the absolute NHWC
        # channel is g*cpg + n_val. Ungrouped: c_group_base is None -> c = n_val.
        c = b_.add(n_val, c_group_base) if c_group_base is not None else n_val
        return dX_desc.offset(b_, m=m_val, c=c)

    return dx_addr, (params["p_dg_M"], params["p_dg_N"])


def _emit_dgrad_cshuffle_epilogue_wmma(
    b: IRBuilder,
    spec: DgradConvSpec,
    op,
    accs: Sequence[Value],
    grid: WarpGrid,
    dx_rsrc: Value,
    params: Dict[str, Value],
    c_group_base: Optional[Value] = None,
) -> None:
    """WMMA cshuffle epilogue for stride=1 dX (uses from_grid_op)."""
    _war_barriers = 2 if spec.pipeline == "wavelet" else 1
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    dx_addr, bounds = _dgrad_stride1_dx_addr(b, spec, params, c_group_base)
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
    params: Dict[str, Value],
    c_group_base: Optional[Value] = None,
) -> None:
    """MFMA cshuffle epilogue for stride=1 dX."""
    _war_barriers = 2 if spec.pipeline == "wavelet" else 1
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    dx_addr, bounds = _dgrad_stride1_dx_addr(b, spec, params, c_group_base)
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
    hw_tilde: _MagicDiv,
    w_tilde_slice: _MagicDiv,
    d_h_stride: Value,
    d_h_offset: Value,
    d_w_stride: Value,
    d_w_offset: Value,
    c_Hi: Value,
    c_Wi: Value,
    c_C: Value,
    c_group_base: Optional[Value] = None,
    patch_cols: Optional[Tuple[Value, Value]] = None,
):
    """Compute NHWC element offset + hi/wi validity for the tilde epilogues.

    Maps (m_global, n_global) -> (element_offset, hw_valid) where:
      m_global decomposes as (n_val, htl, wtl) via hw_tilde / w_tilde_slice
      hi = htl * d_h_stride + d_h_offset
      wi = wtl * d_w_stride + d_w_offset
      offset = ((n_val * Hi + hi) * Wi + wi) * C + n_global

    For fixed m_global the N dimension (channel) is contiguous in dX memory
    (stride 1 in C), enabling vector stores along N via the cshuffle epilogue.

    With ``patch_cols = (X*C, Wi*C)`` (``DgradConvSpec.patch``) n_global is
    a filter element ``(y, x*C + c)`` of the patch at (hi, wi), and lands
    ``y*Wi*C + x*C + c`` past it: still contiguous over a run of X*C.

    Returns (offset, hw_valid) — bounds_m / bounds_n (m < gemm_m, n < C) are
    handled by the epilogue caller via its `bounds` argument.
    """
    c0 = b.const_i32(0)

    n_val, m_rem = _magic_divmod(b, m_global, hw_tilde)
    htl, wtl = _magic_divmod(b, m_rem, w_tilde_slice)

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
    if patch_cols is not None:
        patch_xc, patch_row = patch_cols
        patch_y = b.div(n_global, patch_xc)
        patch_rem = b.sub(n_global, b.mul(patch_y, patch_xc))
        _n = b.add(b.mul(patch_y, patch_row), patch_rem)
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
    hw_tilde: _MagicDiv,
    w_tilde_slice: _MagicDiv,
    d_h_stride: Value,
    d_h_offset: Value,
    d_w_stride: Value,
    d_w_offset: Value,
    c_Hi: Value,
    c_Wi: Value,
    c_C: Value,
    c_group_base: Optional[Value] = None,
    patch_cols: Optional[Tuple[Value, Value]] = None,
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
            patch_cols,
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
    hw_tilde: _MagicDiv,
    w_tilde_slice: _MagicDiv,
    d_h_stride: Value,
    d_h_offset: Value,
    d_w_stride: Value,
    d_w_offset: Value,
    c_Hi: Value,
    c_Wi: Value,
    c_C: Value,
    c_group_base: Optional[Value] = None,
    patch_cols: Optional[Tuple[Value, Value]] = None,
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
            patch_cols,
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
