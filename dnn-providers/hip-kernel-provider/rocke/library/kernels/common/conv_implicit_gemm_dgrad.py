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

import os

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
from rocke.helpers.schedule import DS_READ as _SG_DS_READ
from rocke.helpers.schedule import DS_WRITE as _SG_DS_WRITE
from rocke.helpers.schedule import MFMA as _SG_MFMA
from rocke.helpers.schedule import VMEM_READ as _SG_VMEM_READ
from rocke.helpers.schedule import VALU as _SG_VALU
from rocke.helpers.schedule import SALU as _SG_SALU
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
# K-packed transpose-on-store loader for the B (W) tile
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class KPackedRowLoader:
    """Global -> LDS loader of a ``(rows=n, cols=k)`` tile whose global run
    lies along ``n`` (``vector_axis="row"``), storing ``kpack`` k values per
    LDS write instead of one.

    :class:`CoalescedTileLoader` in row mode gives each thread vectors at
    k values a whole thread-block apart, so the transpose to the row-major
    ``(n, k)`` LDS tile is one 16-bit store per element. Here a thread's
    unit is ``kpack`` vectors at consecutive k of one n chunk, and lane
    ``t`` takes k group ``t % (tile_cols / kpack)``: element ``i`` of the
    ``kpack`` vectors is one ``kpack``-wide store at ``[n0 + i, k0]``, and
    the lanes of a store walk consecutive k groups of one row (no bank
    conflict beyond the store's own width). Same ``load_global`` /
    ``store_lds`` / ``load`` API as the coalesced loader; the global reads
    are the same bytes, in another lane order.
    """

    tile_rows: int
    tile_cols: int
    block_size: int
    load_vec: int
    kpack: int
    elem_dtype: Type
    # Lane order of the units: False = k groups fastest (consecutive lanes
    # store consecutive k groups of one row); True = n chunks fastest
    # (consecutive lanes read consecutive load_vec runs of one k row: whole
    # cache lines per load; the stores then need a row swizzle).
    n_fast: bool = False

    @staticmethod
    def fits(tile_rows, tile_cols, block_size, load_vec, kpack) -> bool:
        if load_vec < 2 or tile_rows % load_vec or tile_cols % kpack:
            return False
        units = (tile_rows // load_vec) * (tile_cols // kpack)
        return units >= block_size and units % block_size == 0

    @classmethod
    def choose_vec(cls, tile_rows, tile_cols, block_size, max_vec, kpack) -> int:
        """The widest load width <= ``max_vec`` that fits, or 0 (none)."""
        v = max_vec
        while v >= 2:
            if cls.fits(tile_rows, tile_cols, block_size, v, kpack):
                return v
            v //= 2
        return 0

    def _units(self, b: IRBuilder, tid: Value):
        k_groups = self.tile_cols // self.kpack
        units = (self.tile_rows // self.load_vec) * k_groups
        c_kg = b.const_i32(k_groups)
        out = []
        n_chunks = self.tile_rows // self.load_vec
        for e in range(units // self.block_size):
            u = b.add(b.const_i32(e * self.block_size), tid)
            if self.n_fast:
                kg = b.div(u, b.const_i32(n_chunks))
                nc = b.sub(u, b.mul(kg, b.const_i32(n_chunks)))
            else:
                nc = b.div(u, c_kg)
                kg = b.sub(u, b.mul(nc, c_kg))
            out.append(
                (b.mul(nc, b.const_i32(self.load_vec)), b.mul(kg, b.const_i32(self.kpack)))
            )
        return out

    def load_global(self, b: IRBuilder, *, tid: Value, descriptor, rsrc: Value):
        from rocke.helpers.loads import _ELEM_BYTES

        c_eb = b.const_i32(_ELEM_BYTES[self.elem_dtype.name])
        c0 = b.const_i32(0)
        c_oob = b.const_i32((1 << 31) - 1)
        staged = []
        for row, col in self._units(b, tid):
            vs = []
            for j in range(self.kpack):
                off, valid = descriptor(b, row, b.add(col, b.const_i32(j)) if j else col)
                off_b = b.mul(off, c_eb)
                if valid is not None:
                    off_b = b.select(valid, off_b, c_oob)
                vs.append(b.buffer_load_vN(rsrc, off_b, c0, self.elem_dtype, self.load_vec))
            staged.append((row, col, vs))
        return staged

    def store_lds(self, b: IRBuilder, *, smem_dst: Value, staged: list) -> None:
        for row, col, vs in staged:
            for i in range(self.load_vec):
                r = b.add(row, b.const_i32(i)) if i else row
                packed = b.vec_pack(
                    [b.vec_extract(v, i) for v in vs], self.elem_dtype
                )
                b.smem_store_vN(smem_dst, [r, col], packed, self.kpack)

    def load(self, b: IRBuilder, *, tid: Value, smem_dst: Value, descriptor, rsrc):
        self.store_lds(
            b,
            smem_dst=smem_dst,
            staged=self.load_global(b, tid=tid, descriptor=descriptor, rsrc=rsrc),
        )


@dataclass(frozen=True)
class XTapKPackLoader:
    """:class:`KPackedRowLoader` over the ``n_taps`` B (W) tiles of the x-halo
    loop as one unit space, so the loads stay ``load_vec`` wide when one
    tile has fewer kpack units than threads: unit ``u`` is tap
    ``u // units_per_tap``; the last pass leaves whole waves idle (their reads
    are out of range and their stores go to junk rows past the
    ``n_taps * tile_rows`` tile rows). Staged rows are rows of that stacked
    ``(n_taps * tile_rows + load_vec, tile_cols)`` LDS tile.
    ``descriptor(b, x, row, col)`` gives the offset of tap ``x``.
    """

    n_taps: int
    tile_rows: int
    tile_cols: int
    block_size: int
    load_vec: int
    kpack: int
    elem_dtype: Type

    @property
    def units(self) -> int:
        return self.n_taps * (self.tile_rows // self.load_vec) * (self.tile_cols // self.kpack)

    @staticmethod
    def fits(n_taps, tile_rows, tile_cols, load_vec, kpack, wave: int = 64) -> bool:
        if load_vec < 2 or kpack < 2 or tile_rows % load_vec or tile_cols % kpack:
            return False
        units = n_taps * (tile_rows // load_vec) * (tile_cols // kpack)
        return units % wave == 0

    def load_global(self, b: IRBuilder, *, tid: Value, descriptor, rsrc: Value):
        from rocke.helpers.loads import _ELEM_BYTES

        c_eb = b.const_i32(_ELEM_BYTES[self.elem_dtype.name])
        c0 = b.const_i32(0)
        c_oob = b.const_i32((1 << 31) - 1)
        k_groups = self.tile_cols // self.kpack
        upt = (self.tile_rows // self.load_vec) * k_groups
        c_upt = b.const_i32(upt)
        c_kg = b.const_i32(k_groups)
        units = self.units
        staged = []
        for e in range(-(-units // self.block_size)):
            u = b.add(b.const_i32(e * self.block_size), tid)
            partial = (e + 1) * self.block_size > units
            active = b.cmp_lt(u, b.const_i32(units)) if partial else None
            if active is not None:
                u = b.select(active, u, c0)
            x = b.div(u, c_upt)
            lu = b.sub(u, b.mul(x, c_upt))
            nc = b.div(lu, c_kg)
            kg = b.sub(lu, b.mul(nc, c_kg))
            row = b.mul(nc, b.const_i32(self.load_vec))
            col = b.mul(kg, b.const_i32(self.kpack))
            vs = []
            for j in range(self.kpack):
                off, valid = descriptor(b, x, row, b.add(col, b.const_i32(j)) if j else col)
                off_b = b.mul(off, c_eb)
                if active is not None:
                    valid = active if valid is None else b.land(valid, active)
                if valid is not None:
                    off_b = b.select(valid, off_b, c_oob)
                vs.append(b.buffer_load_vN(rsrc, off_b, c0, self.elem_dtype, self.load_vec))
            srow = b.add(b.mul(x, b.const_i32(self.tile_rows)), row)
            if active is not None:
                srow = b.select(active, srow, b.const_i32(self.n_taps * self.tile_rows))
            staged.append((srow, col, vs))
        return staged

    def store_lds(self, b: IRBuilder, *, smem_dst: Value, staged: list) -> None:
        for row, col, vs in staged:
            for i in range(self.load_vec):
                r = b.add(row, b.const_i32(i)) if i else row
                packed = b.vec_pack([b.vec_extract(v, i) for v in vs], self.elem_dtype)
                b.smem_store_vN(smem_dst, [r, col], packed, self.kpack)


# ---------------------------------------------------------------------
# Dgrad-specific GEMM dimension helpers
# ---------------------------------------------------------------------


def _dg_M(p: ConvProblem) -> int:
    """Dgrad GEMM-M: input spatial positions."""
    return p.N * (p.Di if p.is_3d else 1) * p.Hi * p.Wi


def _dg_N(p: ConvProblem) -> int:
    """Dgrad GEMM-N: input channels (per group)."""
    return p.cpg


def _dg_K(p: ConvProblem) -> int:
    """Dgrad GEMM-K (reduction): filter spatial × output channels (per group)."""
    return (p.Z if p.is_3d else 1) * p.Y * p.X * p.kpg


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

            # 3-D (stride 1 along every axis, see _3d_conflict): every input
            # depth row is one more slice of the record's rows, so the grid
            # covers N*Di*Hi*Wi. The reduction stays the record's 2-D one
            # (its kpg magic pair divides gemm_k by y_dot*x_dot): a 3-D
            # kernel folds the record and reads its K extent, Z*Y*X*kpg,
            # from p_dg_K.
            gemm_m = p.N * (p.Di if p.is_3d else 1) * h_tilde_slice * w_tilde_slice
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


def make_dgrad_dx_descriptor_dynamic(
    b: IRBuilder, params: Dict[str, Value], *, is_3d: bool = False
):
    """AOT counterpart of :func:`make_dgrad_dx_descriptor`: ``(m, c) -> N[D]HWC``.

    ``m -> (n, [di,] hi, wi)`` peels one extent at a time, so it divides by
    ``Wi``, then by ``Hi`` (then by ``Di``); the host supplies a magic pair per
    extent.
    """
    if is_3d:
        return DynamicTensorDescriptor.create(
            "dX_ndhwc",
            coord_names=["n", "di", "hi", "wi", "c"],
            strides=[
                params["p_dX_stride_n"],
                params["p_dX_stride_di"],
                params["p_dX_stride_hi"],
                params["p_dX_stride_wi"],
                b.const_i32(1),
            ],
        ).transform(
            unmerge_magic_dynamic(
                "m",
                into=["n", "di", "hi", "wi"],
                magic_triples=[
                    (
                        params[f"p_magic_m_{d}_mult"],
                        params[f"p_magic_m_{d}_shift"],
                        params[f"p_{d}"],
                    )
                    for d in ("Di", "Hi", "Wi")
                ],
            ),
        )
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


# tapk_pingpong 4/5: largest filter extent per axis of the straight-line tap
# mask (see _pp_fast_a).
_PP_MAX_FILTER = 3

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
    # Software pipelining of the tap-outer K loop (needs tap_outer_k). The
    # (tap, chunk) nest runs as one flat loop over its tiles, carrying the next
    # tile's (k_out chunk, x, y[, z]) as uniform iteration arguments, and each
    # tile's global reads are issued before the previous tile's MFMAs, so the
    # load latency hides behind the math instead of stalling every tile.
    #   1: one LDS buffer; per tile: read tile t+1 into registers, MFMAs of
    #      tile t, barrier, store tile t+1, barrier.
    #   2: two LDS buffers (A and B, twice the LDS); the loop body is unrolled
    #      twice so each phase binds a build-time buffer: read t+1, MFMAs of t
    #      from one buffer, store t+1 into the other, one barrier per tile. An
    #      odd tile count zero-fills the trailing phase (its dY reads are
    #      predicated off).
    #   3: one LDS buffer, two tiles ahead: tile t+1 is held in registers
    #      across the loop back edge and tile t+2's reads are issued before
    #      tile t's MFMAs (twice the staging registers of 1).
    # Same reduction order as the plain tap-outer loop. Default 0 (off); tags
    # ``tpf<n>``.
    tapk_prefetch: int = 0
    # dY reuse across the filter's x taps in the tap-outer K loop (needs
    # tap_outer_k and pW == 0). At stride 1 and dilation 1, tap (.., x) of dX
    # pixel m reads the dY pixel tap (.., 0) of pixel m - x reads, whenever
    # both lie in one dX row (wi >= x); with pW == 0 a pixel with wi < x has
    # no dY pixel for tap x at all (wo = wi - x < 0). So the loop runs the
    # (z, y) taps outer and the chunks inner, stages per iteration the tile's
    # tap-(z, y, 0) dY rows plus the X-1 rows before it in LDS once, and the X
    # x taps (unrolled; X is baked) read it at a row shift of -x, a lane with
    # wi < x reading a trailing all-zero LDS row instead. The X W tiles of the
    # iteration are staged side by side. X times fewer dY reads and barriers.
    # Default False; tags ``xh``.
    tapk_xhalo: bool = False
    # Store the B (W) tile ``b_kpack`` k values per LDS write (2, 4 or 8; 0 =
    # off) with KPackedRowLoader, where it would otherwise be transposed on
    # the store one element at a time (vector_axis "row", M-outer B: every
    # target without an LDS transpose read). Each thread then loads b_kpack
    # vectors at consecutive k of one c chunk. The tile must split into
    # whole units per thread (KPackedRowLoader.fits). Default 0; tags
    # ``bkp<n>``.
    b_kpack: int = 0
    # With tapk_xhalo and tapk_prefetch 1/3: interleave the next tile's
    # global loads among the MFMAs, one load every ``tapk_ilv`` MFMAs
    # (sched_group_barrier VMEM_READ 1 / MFMA n), instead of issuing them all
    # as one block ahead of the MFMAs. Pays on 1-wave/SIMD tiles, where the
    # in-order wave otherwise stalls on TA back-pressure at every load issue
    # with no co-resident wave to cover it. Default 0 (block issue); tags
    # ``ilv<n>``.
    tapk_ilv: int = 0
    # With tapk_xhalo: stage exactly tile_m + X - 1 dY rows (the tile's own
    # rows by whole loader passes plus the X - 1 rows before it on the first
    # lanes) instead of padding them to whole loader passes. Frees LDS (e.g.
    # 41 KB -> 32.5 KB at t256 tile_k 32 with kpad 4: 2 WG/CU). Default False;
    # tags ``xhx``.
    xhalo_exact: bool = False
    # XCD count of an XCD-contiguous tile order that also applies to
    # ungrouped launches (see xcd_tile_order_xcds); 0 = the per-target
    # default of _XCD_TILE_ORDER_ARCHES (grouped launches only). Tags
    # ``xcd<n>``.
    xcd_remap: int = 0
    # cshuffle epilogue: stage C through LDS in ``cshuffle_m_chunks`` row
    # slabs (tile_m / m_chunks rows each), so the C staging buffer no longer
    # sets the kernel's LDS (and occupancy). Default 1; tags ``mch<n>``.
    cshuffle_m_chunks: int = 1
    # With tapk_xhalo and tapk_prefetch 1/3: read the LDS fragments of the
    # MFMA step ``tapk_lds_ahead`` steps ahead (a step = one x tap's k atom),
    # a software pipeline over the whole tile, so the ds_read latency is
    # hidden behind MFMAs on 1-wave/SIMD tiles. Default 0 (reads right before
    # their MFMAs); tags ``lda<n>``.
    tapk_lds_ahead: int = 0
    # With tapk_xhalo and tapk_prefetch 1/3: the K loop's two barriers drain
    # LDS only (s_waitcnt lgkmcnt(0) + s_barrier), not VMEM, so the global
    # loads of tiles ahead stay in flight across them. Default False; tags
    # ``lsb``.
    tapk_lds_barrier: bool = False
    # With tapk_xhalo and tapk_prefetch 1/3: s_setprio 1 over the loop's load
    # issue + MFMA block, 0 before its barrier (1); 2 also clusters the phase:
    # next-tile loads, then all LDS fragment reads, then all MFMAs (Triton's
    # block-ping-pong loop form; needs tapk_ilv == tapk_lds_ahead == 0).
    # Default 0; tags ``sp<n>``.
    tapk_setprio: int = 0
    # With tapk_xhalo: XOR-swizzle the 16-byte chunks of the unpadded A (dY
    # halo) and B (W) LDS rows, keyed on the row's 256-byte line (the fixed
    # fwd lds_xor key), instead of padding the rows (lds_k_pad must be 0).
    # Default False; tags ``lxor``.
    lds_xor: bool = False
    # With tapk_xhalo, tapk_prefetch 1/3 and b_kpack: load the X B (W) tiles
    # of an x-halo tile as one stacked tile (XTapKPackLoader) at the full
    # load width, instead of one tile at a time at the width a single tile
    # admits. Default False; tags ``bw``.
    xh_b_wide: bool = False
    # With tapk_xhalo and lds_xor: pad the B (W) LDS rows by this many
    # elements instead of swizzling them (A stays swizzled): the kpack B
    # stores of lanes 16 rows apart then fall on different banks. Default 0;
    # tags ``bpad<n>``.
    xh_b_kpad: int = 0
    # With tap_outer_k and tapk_prefetch 1 (the plain loop, no x-halo) and 8
    # waves: the K loop as a block ping-pong. Per K tile four phases, each
    # closed by an LDS-only barrier: [mem A] the next tile's global loads and
    # every LDS fragment read of this tile; [compute A] the first half of the
    # MFMAs under s_setprio 1; [mem B] the next tile's LDS stores; [compute B]
    # the other half. The upper four waves run one barrier behind the lower
    # four (one extra barrier before the loop, the matching one after it), so
    # the two waves of a SIMD alternate memory and MFMA phases. 1 = that loop;
    # 2 = the same with kpack-2 fragment reads (one 16-byte read per lane
    # feeds two k atoms); 3 = 2 with the b_kpack B loads n-chunk-fastest
    # across lanes (whole cache lines per load; KPackedRowLoader.n_fast) and
    # the lds_xor key also XOR the row's 8-row group; 4 = 3 with the dY loads
    # addressed through a per-row tap mask built before the loop (no
    # per-tile bounds checks; <= 31 filter taps; needs tapk_pp_sched); 5 = 4
    # with the lower half's [mem A] barrier not draining its LDS reads. With lds_xor the A/B rows are unpadded and their
    # 16-byte chunks XOR the row index. Default 0; tags ``bpp<n>``.
    tapk_pingpong: int = 0
    # With tapk_pingpong: where the next tiles' global loads go. 0: tile
    # t+1's loads in [mem A]; 1: tile t+2's loads in [mem B] right after tile
    # t+1's LDS stores, carried in registers across the back edge ([mem A]
    # is then the fragment reads alone); 2: as 1, but the loads (and their
    # address math) interleaved with the MFMAs of [compute B]; 3: as 2, in
    # [compute A] (two staging register sets live there); 4: the A loads in
    # [compute A], the B loads in [compute B]; 5: the A loads in [mem B] (as
    # 1), the B loads in [compute B] (as 2); 6: the B loads in [mem B], the
    # A loads in [compute B]. Default 0; tags ``pps<n>``.
    tapk_pp_sched: int = 0
    # With tapk_pingpong and the cshuffle epilogue (split_k 1, 16x16 bf16):
    # stage C through LDS with the 16-byte chunks of a row XOR its 4-row
    # group, so the four 4-row groups of one ds_write_b16 land on different
    # banks (the default staging has them 4 x 512 B apart: one bank). 1 =
    # that; 2 = 1 with the MFMA operands swapped (the accumulators hold C
    # transposed: a lane's 4 values are 4 consecutive channels of one
    # pixel), so a staging write is one ds_write_b64 per atom instead of
    # four ds_write_b16, its 16-byte chunks XOR the row's low 3 bits; and
    # the f32 -> bf16 rounding without the NaN fix-up (_pp_f32_to_bf16).
    # Default 0; tags ``ppe<n>``.
    tapk_pp_epi: int = 0
    # With tapk_xhalo and lds_xor: store the A (dY halo) LDS tile as K0 x Mp x
    # 4 (slabs of 4-element k groups, Mp rows each) instead of XOR-swizzled
    # rows. An MFMA operand fetch (16 lanes, 16 consecutive rows, 8 bytes
    # each) then reads 128 contiguous bytes, conflict-free at every x-tap row
    # shift (the 32-byte rows of tile_k 16 leave the lxor key 2 phases: a
    # 2-way conflict on every A fragment read); each 16-byte store becomes
    # two 8-byte stores, kept conflict-free by the slab pitch Mp (see
    # xh_a_k4_rows). Default False; tags ``ak4``.
    xh_a_k4: bool = False
    # With tapk_xhalo and tapk_prefetch 1/3, bit mask: 1 = issue the next
    # tile's B (W) loads before its A (dY) loads; 2 = s_waitcnt lgkmcnt(0) at
    # the loop top (nothing is pending there; it clears a kernarg s_load the
    # preheader leaves in flight, which otherwise makes LLVM wait lgkmcnt(0)
    # before the first MFMA of every iteration instead of a partial count).
    # 4 = (tapk_prefetch 3) two tiles per loop iteration, so the tile loaded
    # two ahead stays in its registers (a one-tile body copies them at the
    # latch, after a vmcnt(0) that drains the loads just issued).
    # Default 0; tags ``xo<n>``.
    xh_order: int = 0
    # With tapk_xhalo and tapk_prefetch 1/3: the MFMAs of the last
    # ``xh_defer`` MFMA steps (a step = one x tap's k atom) of a tile run after
    # the loop's first barrier, interleaved with the next tile's LDS stores
    # (their fragments are read before it) -- fwd's defer_mfma. Default 0;
    # tags ``xdf<n>``.
    xh_defer: int = 0

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
    def xhalo_lds_rows(self) -> int:
        """LDS rows of the staged dY tile of ``tapk_xhalo``.

        xhalo_exact: the tile's tile_m pixels, the X - 1 pixels before it, a
        junk row (written by the lanes the back-row load leaves idle) and a
        trailing all-zero row. Otherwise tile_m + X - 1 pixels padded to whole
        loader passes (as dy_halo_lds_rows), plus a trailing all-zero row.
        """
        if self.xhalo_exact:
            return self.tile_m + self.problem.X + 1
        rows = self.tile_m + self.problem.X - 1
        pass_elems = self.block_size * 8
        quantum = pass_elems // _gcd(pass_elems, self.tile_k)
        return -(-rows // quantum) * quantum + 1

    @property
    def xh_a_k4_rows(self) -> int:
        """Rows Mp of one k-group slab of the ``xh_a_k4`` A tile: at least
        xhalo_lds_rows, with Mp = R/2 (mod 8) for the R rows one 16-lane pass
        of the 8-byte A stores covers (R = 16 * 8 / tile_k at load width 8),
        so the two slabs a pass writes start 8 * R bytes apart in the 128-byte
        bank window."""
        rows = self.xhalo_lds_rows
        r = max(1, 16 * 8 // self.tile_k)
        want = (r // 2) % 8 if r >= 2 else 1
        return rows + (want - rows) % 8

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
                f"tpf{self.tapk_prefetch}": self.tapk_prefetch > 0,
                "xh": self.tapk_xhalo,
                f"bkp{self.b_kpack}": self.b_kpack > 0,
                # x-halo loop knobs (see tapk_ilv / xhalo_exact / xcd_remap /
                # cshuffle_m_chunks): different code, so they reach the name.
                f"ilv{self.tapk_ilv}": self.tapk_ilv > 0,
                "xhx": self.xhalo_exact,
                f"xcd{self.xcd_remap}": self.xcd_remap > 0,
                f"mch{self.cshuffle_m_chunks}": self.cshuffle_m_chunks > 1,
                f"lda{self.tapk_lds_ahead}": self.tapk_lds_ahead > 0,
                "lsb": self.tapk_lds_barrier,
                f"sp{self.tapk_setprio}": self.tapk_setprio > 0,
                "lxor": self.lds_xor,
                "bw": self.xh_b_wide,
                f"bpad{self.xh_b_kpad}": self.xh_b_kpad > 0,
                f"bpp{self.tapk_pingpong}": self.tapk_pingpong > 0,
                f"pps{self.tapk_pp_sched}": self.tapk_pp_sched > 0,
                f"ppe{self.tapk_pp_epi}": self.tapk_pp_epi > 0,
                "ak4": self.xh_a_k4,
                f"xo{self.xh_order}": self.xh_order > 0,
                f"xdf{self.xh_defer}": self.xh_defer > 0,
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
        _3d_why = _3d_conflict(self)
        if _3d_why:
            raise ValueError(_3d_why)
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
        _tapk_why = _tap_outer_k_conflict(self) or _tapk_prefetch_conflict(self)
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


def _3d_conflict(spec: "DgradConvSpec") -> str:
    """Why a 3-D (NDHWC) ``spec`` cannot be built, or ``""`` if it can.

    3-D dgrad runs on the folded stride-1 record only
    (``folds_sub_gemm_record``): stride 1 along all three axes. Dilation and
    padding are free. The depth axis then never splits the GEMM -- every input
    depth row ``di`` takes filter tap ``z`` from output row
    ``do = di + pD - z*dD`` -- so the kernel decodes ``m -> (n, di, hi, wi)``
    and ``k -> (z, y, x, k_out)`` and the tilde records stay 2-D. The K loops
    that implement it are the plain loop and the tap-outer loop; the
    ``uniform_k_decode`` loop, the dY halo, ``patch`` and the wavelet
    pipeline are 2-D only and rejected. Shared by ``validate`` and
    ``is_valid_dgrad_spec``.
    """
    p = spec.problem
    if not p.is_3d:
        return ""
    if p.sD != 1 or p.sH != 1 or p.sW != 1:
        return (
            "3-D dgrad needs stride 1 along depth, height and width (the "
            "folded stride-1 record); got stride "
            f"({p.sD}, {p.sH}, {p.sW})"
        )
    if spec.pipeline == "wavelet":
        return "3-D dgrad is not supported with pipeline='wavelet'"
    if spec.uniform_k_decode:
        return "3-D dgrad does not implement the uniform_k_decode K loop"
    if spec.dy_halo:
        return "3-D dgrad does not implement the dY halo K loop"
    if spec.patch:
        return "3-D dgrad does not implement patch"
    return ""


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
    if (
        p.sH != 1
        or p.sW != 1
        or p.dH != 1
        or p.dW != 1
        or (p.is_3d and (p.sD != 1 or p.dD != 1))
    ):
        return (
            "tap_outer_k needs stride 1 and dilation 1 (the filter tap is the "
            "plain (y, x) only there)"
        )
    if p.kpg % spec.tile_k != 0:
        return (
            f"tap_outer_k needs kpg % tile_k == 0 (a K tile must not straddle "
            f"two filter taps); got kpg={p.kpg}, tile_k={spec.tile_k}"
        )
    if spec.split_k != 1 and not (
        (spec.tapk_xhalo or spec.tapk_pingpong)
        and spec.tapk_prefetch in (1, 3)
        and spec.split_k > 1
    ):  # the x-halo prefetch / ping-pong loops slice whole tiles
        return "tap_outer_k needs split_k == 1 (a K slice could start mid-tap)"
    if spec.split_k > 1 and spec.epilogue != "cshuffle":
        # Its packed-atomic store treats dX as a dense (M, C) matrix, which
        # it is for groups == 1 (identity_supports_problem enforces that).
        return "x-halo split-K needs the cshuffle epilogue"
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


def _tapk_pingpong_conflict(spec: DgradConvSpec) -> str:
    """Why ``spec.tapk_pingpong`` cannot be honoured, or ``""``."""
    if spec.tapk_pp_epi not in (0, 1, 2) or (
        spec.tapk_pp_epi
        and not (
            spec.tapk_pingpong
            and spec.epilogue == "cshuffle"
            and spec.split_k == 1
            and spec.warp_tile_m == 16
            and spec.warp_tile_n == 16
            and spec.data.dtype_d == "bf16"
        )
    ):
        return "tapk_pp_epi 1/2 needs tapk_pingpong, the cshuffle epilogue, split_k 1, 16x16 atoms, bf16 dX"
    if spec.tapk_pp_sched not in (0, 1, 2, 3, 4, 5, 6) or (spec.tapk_pp_sched and not spec.tapk_pingpong):
        return "tapk_pp_sched is 0..6 and needs tapk_pingpong"
    if not spec.tapk_pingpong:
        return ""
    if spec.tapk_pingpong not in (1, 2, 3, 4, 5):
        return f"tapk_pingpong must be 0..5; got {spec.tapk_pingpong}"
    if spec.tapk_pingpong >= 3 and not spec.b_kpack:
        return "tapk_pingpong 3/4 (n-fast kpack B loads) needs b_kpack"
    if spec.tapk_pingpong >= 4:
        p = spec.problem
        if max(p.Y, p.X, p.Z if p.is_3d else 1) > _PP_MAX_FILTER:
            return (
                f"tapk_pingpong 4/5 builds its tap mask for filters of up to "
                f"{_PP_MAX_FILTER} per axis"
            )
        if not spec.tapk_pp_sched:
            return "tapk_pingpong 4/5 needs tapk_pp_sched (the two-tiles-ahead loop)"
    if spec.tapk_pingpong >= 2 and spec.k_atoms_per_tile_k % 2:
        return "tapk_pingpong 2+ (kpack-2 reads) needs an even number of k atoms per tile"
    if not (
        spec.tap_outer_k
        and spec.tapk_prefetch == 1
        and not spec.tapk_xhalo
        and not spec.dy_halo
        and not spec.lds_k_outer
        and spec.wave_size == 64
        and spec.pipeline != "wavelet"
        and spec.warp_m * spec.warp_n == 8
        and spec.k_atoms_per_tile_k % 2 == 0
    ):
        return (
            "tapk_pingpong needs the plain tap-outer loop with tapk_prefetch 1 "
            "(no x-halo / dY halo), 8 waves, the M-outer B tile, "
            "the wave64 MFMA path and an even number of k atoms per tile"
        )
    return ""


def _tapk_prefetch_conflict(spec: DgradConvSpec) -> str:
    """Why ``spec.tapk_prefetch`` / ``tapk_xhalo`` / ``b_kpack`` cannot be honoured."""
    _pp_why = _tapk_pingpong_conflict(spec)
    if _pp_why:
        return _pp_why
    if spec.b_kpack:
        if spec.b_kpack not in (2, 4, 8):
            return f"b_kpack must be 0, 2, 4 or 8; got {spec.b_kpack}"
        if spec.lds_k_outer:
            return "b_kpack applies to the M-outer B tile only (lds_k_outer is set)"
        if spec.pipeline == "wavelet":
            return "b_kpack is not supported with pipeline='wavelet'"
        vb = _dgrad_b_load_vec(spec)
        if not KPackedRowLoader.choose_vec(
            spec.tile_n, spec.tile_k, spec.block_size, vb, spec.b_kpack
        ):
            return (
                f"b_kpack={spec.b_kpack} does not split the {spec.tile_n}x"
                f"{spec.tile_k} B tile into whole units per thread at any load "
                f"width in [2, {vb}]"
            )
    if spec.tapk_xhalo:
        if not spec.tap_outer_k:
            return "tapk_xhalo is a mode of the tap-outer K loop; it needs tap_outer_k"
        if spec.dy_halo:
            return "tapk_xhalo and dy_halo are exclusive"
        if spec.tapk_prefetch == 2:
            return "tapk_xhalo does not implement tapk_prefetch=2 (no LDS ping-pong)"
        if spec.problem.pW != 0:
            return (
                "tapk_xhalo needs pW == 0 (a pixel with wi < x then has no dY "
                f"pixel for tap x); got pW={spec.problem.pW}"
            )
        if spec.problem.X < 2:
            return "tapk_xhalo needs a filter with X > 1"
    if spec.tapk_prefetch not in (0, 1, 2, 3):
        return f"tapk_prefetch must be 0, 1, 2 or 3; got {spec.tapk_prefetch}"
    if spec.tapk_prefetch == 0:
        return ""
    if not spec.tap_outer_k:
        return "tapk_prefetch pipelines the tap-outer K loop; it needs tap_outer_k"
    if spec.dy_halo:
        return "tapk_prefetch does not apply to the dY halo K loop"
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
    num_xcds = spec.xcd_remap or _XCD_TILE_ORDER_ARCHES.get(arch, 0)
    if not num_xcds or not spec.xcd_tile_order:
        return 0
    if (spec.problem.groups <= 1 and not spec.xcd_remap) or spec.split_k != 1:
        return 0
    if not spec.folds_sub_gemm_record:
        return 0
    return num_xcds


def emit_xcd_tile_order(
    b: IRBuilder,
    block_x: Value,
    params: dict[str, Value],
    *,
    num_xcds: int,
    ungrouped: bool = False,
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
    # ungrouped (DgradConvSpec.xcd_remap): the order also serves groups == 1.
    _grp_ok = (
        b.cmp_ge(params["p_groups"], c1)
        if ungrouped
        else b.cmp_gt(params["p_groups"], c1)
    )
    use_xcd = b.land(
        b.land(_grp_ok, shares_rows),
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


def _dgrad_b_load_vec(spec: DgradConvSpec) -> int:
    """The B (W) load width of the M-outer row-mode loader (the builder's)."""
    _, def_vec_b, _ = DgradConvSpec.default_vector_sizes(
        spec.problem.cpg, spec.problem.kpg, spec.data.dtype_a
    )
    max_vb = def_vec_b if spec.vector_size_b is None else min(def_vec_b, spec.vector_size_b)
    try:
        return CoalescedTileLoader.choose_vec(
            tile_rows=spec.tile_n,
            tile_cols=spec.tile_k,
            block_size=spec.block_size,
            max_vec=max_vb,
            vector_axis="row",
        )
    except ValueError:
        return 1


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
        spec.dy_halo_lds_rows
        if spec.dy_halo > 0
        else spec.xhalo_lds_rows if spec.tapk_xhalo else spec.tile_m
    )
    if spec.tapk_xhalo and spec.xh_a_k4:
        # K0 slabs of xh_a_k4_rows rows x 4 elements: tile_k * Mp elements.
        _a_shape = (spec.xh_a_k4_rows, spec.tile_k)
    _b_shape = (
        (spec.tile_k, spec.tile_n + spec.kouter_b_pad)
        if spec.lds_k_outer
        else _lds_layout.storage_shape(spec.tile_n)
    )
    # dy_halo=2 double-buffers the B tile; tapk_prefetch=2 both tiles.
    _b_copies = 2 if (spec.dy_halo == 2 or spec.tapk_prefetch == 2) else 1
    _a_copies = 2 if spec.tapk_prefetch == 2 else 1
    if spec.tapk_xhalo:
        _b_copies = spec.problem.X
    if spec.tapk_xhalo and spec.xh_b_kpad and not spec.lds_k_outer:
        _b_shape = (_b_shape[0], spec.tile_k + spec.xh_b_kpad)
    _b_rows = _b_copies * _b_shape[0]
    if spec.tapk_xhalo and spec.xh_b_wide:
        # One stacked tile of the X taps plus load_vec junk rows.
        _b_rows = spec.problem.X * spec.tile_n + _dgrad_b_load_vec(spec)
    _ab_bytes = (
        _a_copies * _a_shape[0] * _a_shape[1] + _b_rows * _b_shape[1]
    ) * _ab_dtype_bytes
    _c_dtype_bytes = 4 if spec.data.dtype_d == "fp32" else 2
    _c_lds = (
        spec.tile_m // max(1, spec.cshuffle_m_chunks) * spec.tile_n * _c_dtype_bytes
        if spec.epilogue == "cshuffle"
        else 0
    )
    # wavelet: A/B stay live across both scf_if_else branches, so the LDS pool
    # cannot alias C onto the A/B region even when cshuffle_no_alias=False.
    _no_alias = spec.cshuffle_no_alias or spec.pipeline == "wavelet"
    _total_lds = (_ab_bytes + _c_lds) if _no_alias else max(_ab_bytes, _c_lds)
    return _ab_bytes, _c_lds, _total_lds


def _xhalo_knob_conflict(spec: DgradConvSpec) -> str:
    """Why the x-halo loop knobs (tapk_ilv, xhalo_exact, xcd_remap,
    cshuffle_m_chunks) do not apply to ``spec``, or ``""``."""
    if spec.tapk_ilv < 0 or (
        spec.tapk_ilv and not (spec.tapk_xhalo and spec.tapk_prefetch in (1, 3))
    ):
        return "tapk_ilv needs tapk_xhalo with tapk_prefetch 1 or 3"
    if spec.tapk_lds_ahead < 0 or (
        spec.tapk_lds_ahead
        and not (
            spec.tapk_xhalo
            and spec.tapk_prefetch in (1, 3)
            and not spec.lds_k_outer
            and spec.wave_size == 64
        )
    ):
        return "tapk_lds_ahead needs tapk_xhalo with tapk_prefetch 1 or 3 (MFMA, M-outer B)"
    if spec.lds_xor and not (
        (spec.tapk_xhalo or spec.tapk_pingpong)
        and spec.lds_k_pad == 0
        and spec.lds_layout is None
        and not spec.lds_k_outer
        and spec.tile_k in (16, 32, 64, 128)
        and spec.wave_size == 64
        and spec.pipeline != "wavelet"
    ):
        return (
            "lds_xor needs tapk_xhalo, lds_k_pad 0, the M-outer B tile, "
            "tile_k in 16..128 and the wave64 MFMA path"
        )
    if spec.xh_b_kpad and not (
        spec.tapk_xhalo and spec.lds_xor and spec.xh_b_kpad in (2, 4, 6, 8)
    ):
        return "xh_b_kpad (2/4/6/8) needs tapk_xhalo with lds_xor"
    if spec.xh_b_wide and not (
        spec.tapk_xhalo
        and spec.tapk_prefetch in (1, 3)
        and spec.b_kpack
        and not spec.lds_k_outer
        and XTapKPackLoader.fits(
            spec.problem.X, spec.tile_n, spec.tile_k, _dgrad_b_load_vec(spec), spec.b_kpack
        )
    ):
        return (
            "xh_b_wide needs tapk_xhalo, tapk_prefetch 1 or 3, b_kpack, the "
            "M-outer B tile and whole waves of stacked B units"
        )
    if spec.xh_a_k4 and not (
        spec.tapk_xhalo
        and spec.lds_xor
        and spec.tile_k in (16, 32, 64)
        and spec.warp_tile_m * spec.warp_tile_k // spec.wave_size == 4
        and spec.warp_tile_n * spec.warp_tile_k // spec.wave_size == 4
    ):
        return (
            "xh_a_k4 needs tapk_xhalo with lds_xor, tile_k 16..64 and an MFMA "
            "atom of 4 k values per lane"
        )
    if spec.xh_order not in range(8) or (
        spec.xh_order and not (spec.tapk_xhalo and spec.tapk_prefetch in (1, 3))
    ) or (spec.xh_order & 4 and spec.tapk_prefetch != 3):
        return (
            "xh_order is 0..7 and needs tapk_xhalo with tapk_prefetch 1 or 3 "
            "(bit 4: tapk_prefetch 3)"
        )
    if spec.xh_defer:
        n_steps = spec.problem.X * spec.k_atoms_per_tile_k
        if not (
            spec.tapk_xhalo
            and spec.tapk_prefetch in (1, 3)
            and spec.tapk_setprio != 2
            and not spec.lds_k_outer
            and spec.wave_size == 64
            and 0 < spec.xh_defer < n_steps
        ):
            return (
                "xh_defer needs tapk_xhalo with tapk_prefetch 1 or 3 (MFMA, "
                "M-outer B, not tapk_setprio 2) and leaves >= 1 step before the barrier"
            )
    if spec.tapk_setprio not in (0, 1, 2) or (
        spec.tapk_setprio
        and not (spec.tapk_xhalo and spec.tapk_prefetch in (1, 3))
    ):
        return "tapk_setprio is 0, 1 or 2 and needs tapk_xhalo with tapk_prefetch 1 or 3"
    if spec.tapk_setprio == 2 and (spec.tapk_ilv or spec.tapk_lds_ahead):
        return "tapk_setprio 2 clusters the phase itself: tapk_ilv / tapk_lds_ahead must be 0"
    if spec.tapk_lds_barrier and not (
        spec.tapk_xhalo and spec.tapk_prefetch in (1, 3)
    ):
        return "tapk_lds_barrier needs tapk_xhalo with tapk_prefetch 1 or 3"
    if spec.xhalo_exact and not spec.tapk_xhalo:
        return "xhalo_exact needs tapk_xhalo"
    if spec.xcd_remap < 0:
        return "xcd_remap must be >= 0"
    mc = spec.cshuffle_m_chunks
    if mc != 1:
        if mc < 1 or spec.epilogue != "cshuffle" or spec.split_k != 1:
            return "cshuffle_m_chunks > 1 needs the cshuffle epilogue and split_k 1"
        if spec.data.dtype_d != "bf16":
            return "cshuffle_m_chunks > 1 is bf16 output only"
        if spec.mfmas_per_warp_m % mc:
            return (
                f"cshuffle_m_chunks {mc} must divide mfmas_per_warp_m "
                f"{spec.mfmas_per_warp_m}"
            )
    return ""


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
    _3d_why = _3d_conflict(spec)
    if _3d_why:
        return False, _3d_why
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
    _tapk_why = _tap_outer_k_conflict(spec) or _tapk_prefetch_conflict(spec)
    if _tapk_why:
        return False, _tapk_why
    _halo_why = _dy_halo_conflict(spec)
    if _halo_why:
        return False, _halo_why
    _xh_why = _xhalo_knob_conflict(spec)
    if _xh_why:
        return False, _xh_why
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
        conv_arg_names(direction="dgrad", is_3d=p.is_3d),
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
    # 3-D (NDHWC, stride 1 along every axis; see _3d_conflict): the GEMM row
    # m decodes to (n', hi, wi) by the 2-D record, n' = n*Di + di, and n' then
    # to (n, di) by the kernarg magic pair; the reduction row (y', x, k_out)
    # to y' = z*Y + y. dY row do = di + pD - z*dD.
    is_3d = p.is_3d
    if is_3d:
        p_Z = params["p_Z"]
        p_Do = params["p_Do"]
        p_pD = params["p_pD"]
        p_dD = params["p_dD"]
        di_div = _MagicDiv(
            params["p_Di"],
            params["p_magic_m_Di_mult"],
            params["p_magic_m_Di_shift"],
        )
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
            b, flat_block_id, params, num_xcds=num_xcds,
            ungrouped=bool(spec.xcd_remap),
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
    # dY row count N*[Do*]Ho*Wo, used by the pointwise fast path's bounds check.
    c_dY_rows = b.mul(b.mul(p_N, p_Ho), p_Wo)
    if is_3d:
        c_dY_rows = b.mul(c_dY_rows, p_Do)

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
    _xh_ak4 = spec.tapk_xhalo and spec.xh_a_k4
    # xh_a_k4: K0 slabs of Mp rows x 4 elements (row k0 * Mp + r).
    _ak4_mp = spec.xh_a_k4_rows if _xh_ak4 else 0
    A_smem = b.smem_alloc(
        ir_dtype_a,
        (block_k // 4 * _ak4_mp, 4)
        if _xh_ak4
        else lds_layout.storage_shape(
            spec.dy_halo_lds_rows
            if spec.dy_halo
            else spec.xhalo_lds_rows if spec.tapk_xhalo else block_m
        ),
        name_hint="A_smem",
    )
    if spec.lds_k_outer:
        # K-outer: rows are K, columns are the free axis N. See _KOUTER_PAD for
        # why the stride must not be a multiple of the 32-dword bank period.
        b_shape = (block_k, block_n + spec.kouter_b_pad)
    else:
        b_shape = lds_layout.storage_shape(block_n)
        if spec.tapk_xhalo and spec.xh_b_kpad:
            # xh_b_kpad: the B tiles padded (not swizzled) under lds_xor.
            b_shape = (block_n, block_k + spec.xh_b_kpad)
    if spec.tapk_xhalo and spec.xh_b_wide:
        # xh_b_wide: the X taps' B tiles stacked (+ load_vec junk rows).
        B_smem = b.smem_alloc(
            ir_dtype_b,
            (p.X * block_n + _dgrad_b_load_vec(spec), b_shape[1]),
            name_hint="B_smem",
        )
    else:
        B_smem = b.smem_alloc(ir_dtype_b, b_shape, name_hint="B_smem")
    # dy_halo=2: the second B buffer of the per-tap double buffer.
    B_smem2 = (
        b.smem_alloc(ir_dtype_b, b_shape, name_hint="B_smem2")
        if spec.dy_halo == 2 or spec.tapk_prefetch == 2
        else None
    )
    # tapk_xhalo: one B tile per x tap.
    B_smem_x = (
        [B_smem] * p.X
        if spec.tapk_xhalo and spec.xh_b_wide
        else [B_smem]
        + [
            b.smem_alloc(ir_dtype_b, b_shape, name_hint=f"B_smem_x{i}")
            for i in range(1, p.X)
        ]
        if spec.tapk_xhalo
        else None
    )
    # Row of tap x's B tile in its LDS buffer (stacked with xh_b_wide).
    xh_b_off = [x * block_n if spec.xh_b_wide else 0 for x in range(p.X)]
    # tapk_prefetch=2: the second A buffer of the ping-pong.
    A_smem2 = (
        b.smem_alloc(
            ir_dtype_a, lds_layout.storage_shape(block_m), name_hint="A_smem2"
        )
        if spec.tapk_prefetch == 2
        else None
    )

    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n
    k_atoms = spec.k_atoms_per_tile_k

    # lds_xor: the column of tile element (row, col) -- its 16-byte chunk XOR
    # the row's 256-byte line index (rows of tile_k < 128 share a line; a key
    # of row % chunks would alias rows r and r + 256/row_bytes).
    _xor_sh = {16: 3, 32: 2, 64: 1, 128: 0}.get(block_k, 0)

    def xh_swz(b_: IRBuilder, row: Value, col: Value) -> Value:
        if not spec.lds_xor:
            return col
        r = b_.lshr(row, b_.const_i32(_xor_sh)) if _xor_sh else row
        key = b_.land(r, b_.const_i32(block_k // 8 - 1))
        return b_.xor(col, b_.shl(key, b_.const_i32(3)))

    def xh_swz_b(b_: IRBuilder, row: Value, col: Value) -> Value:
        """xh_swz for the B tiles (unswizzled when they are padded)."""
        return col if spec.xh_b_kpad else xh_swz(b_, row, col)

    def xh_a_read(b_: IRBuilder, smem: Value, row: Value, col: Value, n: int):
        """One A fragment read (``n`` elements of row ``row`` from column
        ``col``) in the x-halo A layout (lds_xor swizzle or xh_a_k4 slabs)."""
        if _xh_ak4:
            if n != 4:
                raise ValueError("xh_a_k4: A fragments of 4 elements only")
            k0 = b_.lshr(col, b_.const_i32(2))
            r = b_.add(b_.mul(k0, b_.const_i32(_ak4_mp)), row)
            return _emit_smem_load(b_, smem, r, b_.const_i32(0), n, smem_dtype=_smem_dtype)
        return _emit_smem_load(b_, smem, row, xh_swz(b_, row, col), n, smem_dtype=_smem_dtype)

    def xh_store(loader, dst: Value, staged: list, is_b: bool = False) -> None:
        """``loader.store_lds`` with the lds_xor column swizzle."""
        if _xh_ak4 and not is_b:
            # xh_a_k4: each vector as 4-element pieces, one per k-group slab.
            vec = loader.load_vec
            if vec % 4:
                raise ValueError("xh_a_k4: A load width must be a multiple of 4")
            c_mp = b.const_i32(_ak4_mp)
            for row, col, v in staged:
                k0 = b.lshr(col, b.const_i32(2))
                for j in range(vec // 4):
                    kj = b.add(k0, b.const_i32(j)) if j else k0
                    piece = b.vec_pack(
                        [b.vec_extract(v, 4 * j + i) for i in range(4)], ir_dtype_a
                    )
                    b.smem_store_vN(
                        dst, [b.add(b.mul(kj, c_mp), row), b.const_i32(0)], piece, 4
                    )
            return
        if not spec.lds_xor or (is_b and spec.xh_b_kpad):
            loader.store_lds(b, smem_dst=dst, staged=staged)
            return
        if isinstance(loader, (KPackedRowLoader, XTapKPackLoader)):
            for row, col, vs in staged:
                for i in range(loader.load_vec):
                    r = b.add(row, b.const_i32(i)) if i else row
                    packed = b.vec_pack(
                        [b.vec_extract(v, i) for v in vs], loader.elem_dtype
                    )
                    b.smem_store_vN(dst, [r, xh_swz(b, r, col)], packed, loader.kpack)
            return
        if getattr(loader, "vector_axis", "col") == "row" and loader.load_vec > 1:
            if getattr(loader, "row_kpack", 1) != 1:
                raise ValueError("lds_xor: row_kpack B loads are not swizzled")
            for row, col, v in staged:
                for i in range(loader.load_vec):
                    r = b.add(row, b.const_i32(i)) if i else row
                    b.smem_store_vN(dst, [r, xh_swz(b, r, col)], b.vec_extract(v, i), 1)
            return
        vec = loader.load_vec
        for row, col, v in staged:
            b.smem_store_vN(dst, [row, xh_swz(b, row, col)], v, vec)

    # tapk_pp_epi: the ping-pong loop hands its accumulators to the
    # epilogue already converted to bf16 (see _tapk_pingpong_loop).
    _pp_state: dict = {}
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
    if spec.split_k <= 1 or (
        (spec.tapk_xhalo or spec.tapk_pingpong) and spec.tap_outer_k
    ):
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
        do_ok = None
        if is_3d:
            # n is n*Di + di; ydot is z*Y + y (the record's y_dot_slice is Y).
            n, di = _magic_divmod(b_, n, di_div)
            zdot = b_.div(ydot, c_Y)
            ydot = b_.sub(ydot, b_.mul(zdot, c_Y))
            do = b_.sub(b_.add(di, p_pD), b_.mul(zdot, p_dD))
            do_ok = b_.land(b_.cmp_ge(do, c0), b_.cmp_lt(do, p_Do))
            n = b_.add(b_.mul(n, p_Do), do)

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
        if do_ok is not None:
            valid = b_.land(valid, do_ok)

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

        zdot = None
        if is_3d:
            # ydot is z*Y + y; the K tail past dg_K lands on z >= Z.
            zdot = b_.div(ydot, c_Y)
            ydot = b_.sub(ydot, b_.mul(zdot, c_Y))

        # y = ydot * b_y_stride + b_y_offset
        y = b_.add(b_.mul(ydot, rec_b_y_stride), rec_b_y_offset)
        x = b_.add(b_.mul(xdot, rec_b_x_stride), rec_b_x_offset)

        # Bounds check: y < Y, x < X, k_out < kpg (group-local)
        valid = b_.land(
            b_.land(b_.cmp_lt(y, c_Y), b_.cmp_lt(x, c_X)), b_.cmp_lt(k_out, _kdiv)
        )
        if zdot is not None:
            valid = b_.land(valid, b_.cmp_lt(zdot, p_Z))
            # KZYXC: the depth tap rides the filter row index.
            y = b_.add(b_.mul(zdot, c_Y), y)
            k_out_abs = b_.mul(k_out_abs, p_Z)

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
    if spec.b_kpack:
        if axis_b != "row" or load_vec_b < 2:
            raise ValueError(
                f"b_kpack needs a row-mode B load of width >= 2 (got axis "
                f"{axis_b!r}, width {load_vec_b})"
            )
        # The widest width that gives every thread whole kpack units; a
        # narrower global read is the price of the packed LDS store.
        b_sync_loader = KPackedRowLoader(
            tile_rows=block_n,
            tile_cols=block_k,
            block_size=threads,
            load_vec=KPackedRowLoader.choose_vec(
                block_n, block_k, threads, load_vec_b, spec.b_kpack
            ),
            kpack=spec.b_kpack,
            elem_dtype=ir_dtype_b,
            n_fast=spec.tapk_pingpong >= 3,
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
        A_src: Value,
        B_src: Value,
        iter_vars: Sequence[Value],
        a_row_map=None,
        b_row_off: int = 0,
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
                a_rows.append(xh_a_read(b, A_src, a_row, col_base, a_per_lane))
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
                    b.add(b.const_i32(ni * spec.warp_tile_n + b_row_off), n_in_atom),
                )
                b_cols.append(
                    _emit_smem_load(
                        b,
                        B_src,
                        b_row,
                        xh_swz_b(b, b_row, col_base),
                        b_per_lane,
                        smem_dtype=_smem_dtype,
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

    def emit_mfma_phases_ahead(
        phases: Sequence[tuple],
        iter_vars: Sequence[Value],
        depth: int = 1,
        defer: int = 0,
        deferred: Optional[list] = None,
    ) -> List[Value]:
        """``emit_mfma_phase`` over several (A_src, B_src, a_row_map) phases
        as one software pipeline (MFMA, M-outer LDS only): the fragments of
        step ``i + depth`` (a step = one phase's kk) are read from LDS before
        the MFMAs of step ``i``, so the ds_read latency hides behind MFMAs
        (DgradConvSpec.tapk_lds_ahead). With ``defer`` the MFMAs of the last
        ``defer`` steps are not emitted: one closure per step (accs -> accs)
        is appended to ``deferred`` instead (DgradConvSpec.xh_defer)."""
        decoded = decode_mfma_lanes(b, atom, lane)
        m_in_atom = decoded.m_in_atom
        n_in_atom = decoded.n_in_atom
        k_blk = decoded.k_blk
        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)
        steps = [(ph, kk) for ph in phases for kk in range(k_atoms)]
        a_row_cache: dict = {}

        def reads(step):
            ph, kk = step
            A_src, B_src, a_row_map = ph[:3]
            b_off = ph[3] if len(ph) > 3 else 0
            col_base = b.add(
                b.mul(k_blk, b.const_i32(a_per_lane)),
                b.const_i32(kk * spec.warp_tile_k),
            )
            a_rows = []
            for mi in range(mfmas_m):
                key = (id(a_row_map), mi)
                if key not in a_row_cache:
                    a_row = b.add(
                        warp_m_off,
                        b.add(b.const_i32(mi * spec.warp_tile_m), m_in_atom),
                    )
                    if a_row_map is not None:
                        a_row = a_row_map(a_row)
                    a_row_cache[key] = a_row
                a_rows.append(
                    xh_a_read(b, A_src, a_row_cache[key], col_base, a_per_lane)
                )
            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off,
                    b.add(b.const_i32(ni * spec.warp_tile_n + b_off), n_in_atom),
                )
                b_cols.append(
                    _emit_smem_load(
                        b,
                        B_src,
                        b_row,
                        xh_swz_b(b, b_row, col_base),
                        b_per_lane,
                        smem_dtype=_smem_dtype,
                    )
                )
            return a_rows, b_cols

        new_accs: List[Value] = list(iter_vars)
        frags = [reads(s) for s in steps[:depth]]
        for i in range(len(steps)):
            if i + depth < len(steps):
                frags.append(reads(steps[i + depth]))
            a_rows, b_cols = frags[i]
            frags[i] = None
            if defer and i >= len(steps) - defer:

                def _deferred(accs_in, a_rows=a_rows, b_cols=b_cols):
                    out = list(accs_in)
                    f_ = 0
                    for mi_ in range(mfmas_m):
                        for ni_ in range(mfmas_n):
                            out[f_] = _emit_mfma(b, atom, a_rows[mi_], b_cols[ni_], out[f_])
                            f_ += 1
                    return out

                deferred.append(_deferred)
                continue
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    new_accs[flat] = _emit_mfma(
                        b, atom, a_rows[mi], b_cols[ni], new_accs[flat]
                    )
                    flat += 1
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

        if (
            spec.split_k > 1
            and (spec.tapk_xhalo or spec.tapk_pingpong)
            and use_cshuffle
            and not is_wmma
        ):
            # x-halo split-K: stride-1 dX of an ungrouped problem is a dense
            # (M, C) row-major matrix, so the cshuffle packed-atomic epilogue
            # (wgrad's) applies with row stride C (identity_supports_problem
            # serves groups == 1 only).
            _epi = CShuffleEpilogue.from_grid(
                atom=spec.atom,
                grid=grid,
                max_store_vec=_dgrad_store_vec(spec),
                out_dtype=spec.data.dtype_d,
            )
            _epi.atomic_store(
                b, accs=final_accs, dw_ptr=dX, wg_N=c_C, bounds=(rec_gemm_m, c_dg_N)
            )
        elif spec.split_k > 1:
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
                    **({"pre_bf16": _pp_state["bf16"]} if "bf16" in _pp_state else {}),
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
        c_yx = b.mul(c_Y, c_X)
        # 3-D: taps (z, y, x), KZYXC rows of Z*Y*X*Cs.
        c_taps = b.mul(p_Z, c_yx) if is_3d else c_yx
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
            if is_3d:
                # NDHWK: n -> (n, di); the dY row is n*Do + di (+ dd, which
                # rides dy_tap), in range when 0 <= di + dd < Do.
                n, di = _magic_divmod(b_, n, di_div)
                do = b_.add(di, tap_cap["dd"])
                valid = b_.land(
                    valid, b_.land(b_.cmp_ge(do, c0), b_.cmp_lt(do, p_Do))
                )
                n = b_.add(b_.mul(n, p_Do), di)
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
            if is_3d:
                zdot = b.div(tap, c_yx)
                yx = b.sub(tap, b.mul(zdot, c_yx))
            else:
                yx = tap
            ydot = b.div(yx, c_X)
            xdot = b.sub(yx, b.mul(ydot, c_X))
            tap_cap["dh"] = b.sub(params["p_pH"], ydot)
            tap_cap["dw"] = b.sub(params["p_pW"], xdot)
            _row_shift = tap_cap["dh"]
            if is_3d:
                tap_cap["dd"] = b.sub(p_pD, zdot)
                _row_shift = b.add(b.mul(tap_cap["dd"], c_Ho), _row_shift)
            tap_cap["dy_tap"] = b.mul(
                b.add(b.mul(_row_shift, c_Wo), tap_cap["dw"]), c_K
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

    def _pp_swz(row: Value, col: Value) -> Value:
        """tapk_pingpong + lds_xor: the 16-byte chunk XOR the row index (one
        phase per row, Triton's rotating swizzle; tile_k 64 = 128-byte rows,
        so 8 consecutive rows of one fragment read hit 8 distinct chunks)."""
        if not spec.lds_xor:
            return col
        # 64-byte rows (tile_k 32) pair up in a 128-byte bank window: the
        # key is row // 2 there, so the 8 rows of one fragment read cover
        # the window's 8 chunks.
        r = b.lshr(row, b.const_i32(1)) if block_k == 32 else row
        if spec.tapk_pingpong >= 3:
            # Also the row's 8-row group: lanes storing rows 8 apart (the
            # n-fast B loader) land on different chunks too.
            r = b.xor(r, b.lshr(row, b.const_i32(3)))
        key = b.land(r, b.const_i32(block_k // 8 - 1))
        return b.xor(col, b.shl(key, b.const_i32(3)))

    def _pp_store(loader, dst: Value, staged: list) -> None:
        """loader.store_lds with the _pp_swz column swizzle."""
        if not spec.lds_xor:
            loader.store_lds(b, smem_dst=dst, staged=staged)
            return
        if isinstance(loader, KPackedRowLoader):
            for row, col, vs in staged:
                for i in range(loader.load_vec):
                    r = b.add(row, b.const_i32(i)) if i else row
                    packed = b.vec_pack(
                        [b.vec_extract(v, i) for v in vs], loader.elem_dtype
                    )
                    b.smem_store_vN(dst, [r, _pp_swz(r, col)], packed, loader.kpack)
            return
        if getattr(loader, "vector_axis", "col") != "col":
            raise ValueError("tapk_pingpong lds_xor needs a col-mode or kpack loader")
        for row, col, v in staged:
            b.smem_store_vN(dst, [row, _pp_swz(row, col)], v, loader.load_vec)

    def _tapk_pingpong_loop(
        set_tile, advance, load_tile, store_tile, nxt0, st_names, n_tiles,
        fast_a=None,
    ) -> list[Value]:
        """``tapk_pingpong``: the plain tapk_prefetch=1 loop as a block
        ping-pong over 8 waves (see DgradConvSpec.tapk_pingpong). Every
        fragment of the tile is read in [mem A], so the partner half's
        [mem B] stores of the next tile cannot overwrite data still to be
        read; each phase ends in lgkmcnt(0) + s_barrier."""
        c1 = b.const_i32(1)
        n_acc = len(accs)
        n_st = len(st_names)
        n_w = spec.warp_m * spec.warp_n
        decoded = decode_mfma_lanes(b, atom, lane)
        warp_m_off = grid.warp_m_off(b)
        warp_n_off = grid.warp_n_off(b)
        a_idx = [
            b.add(warp_m_off, b.add(b.const_i32(mi * spec.warp_tile_m), decoded.m_in_atom))
            for mi in range(mfmas_m)
        ]
        b_idx = [
            b.add(warp_n_off, b.add(b.const_i32(ni * spec.warp_tile_n), decoded.n_in_atom))
            for ni in range(mfmas_n)
        ]
        # 2: kpack-2 fragment reads -- one 2*a_per_lane-wide read feeds the
        # k atoms 2p and 2p+1 (lane k block j reads k 32p + 8j .. +8, the
        # first half for atom 2p, the second for 2p+1). A and B use the same
        # map, so every k meets its partner once: a permutation of the sum.
        kp = 2 if spec.tapk_pingpong >= 2 else 1
        k_lane = b.mul(decoded.k_blk, b.const_i32(kp * a_per_lane))

        def _part(v, h, n):
            if kp == 1:
                return v
            return b.vec_pack([b.vec_extract(v, h * n + i) for i in range(n)], _smem_dtype)

        def frag_reads():
            out = []
            for pk in range(k_atoms // kp):
                col = b.add(k_lane, b.const_i32(pk * kp * spec.warp_tile_k))
                a_w = [
                    _emit_smem_load(
                        b, A_smem, r, _pp_swz(r, col), kp * a_per_lane,
                        smem_dtype=_smem_dtype,
                    )
                    for r in a_idx
                ]
                b_w = [
                    _emit_smem_load(
                        b, B_smem, r, _pp_swz(r, col), kp * b_per_lane,
                        smem_dtype=_smem_dtype,
                    )
                    for r in b_idx
                ]
                for h in range(kp):
                    out.append(
                        (
                            [_part(v, h, a_per_lane) for v in a_w],
                            [_part(v, h, b_per_lane) for v in b_w],
                        )
                    )
            return out

        def mfmas(frags, acc_in):
            new_accs = list(acc_in)
            for a_rows, b_cols in frags:
                flat = 0
                for mi in range(mfmas_m):
                    for ni in range(mfmas_n):
                        if spec.tapk_pp_epi == 2:
                            # C transposed (see DgradConvSpec.tapk_pp_epi).
                            new_accs[flat] = _emit_mfma(
                                b, atom, b_cols[ni], a_rows[mi], new_accs[flat]
                            )
                        else:
                            new_accs[flat] = _emit_mfma(
                                b, atom, a_rows[mi], b_cols[ni], new_accs[flat]
                            )
                        flat += 1
            return new_accs

        wave = b.readfirstlane(b.div(tid, b.const_i32(spec.wave_size)))
        upper = b.cmp_ge(wave, b.const_i32(n_w // 2))
        with b.scf_if(upper):
            b.sync_lds_only()
        def _pp_two_ahead(frag_reads, mfmas, wave, half):
            """tapk_pp_sched 1: tile t+2's global loads are issued in [mem B]
            right after tile t+1's LDS stores and ride in registers across
            the back edge, so [mem A] is the fragment reads alone and every
            load has a whole iteration to land."""

            def flat(parts):
                out = []
                for part in parts:
                    for _row, _col, v in part:
                        out.extend(v if isinstance(v, list) else [v])
                return out

            def rebuild(template, vals):
                vals = list(vals)
                out = []
                for part in template:
                    p_out = []
                    for row, col, v in part:
                        if isinstance(v, list):
                            p_out.append((row, col, vals[: len(v)]))
                            del vals[: len(v)]
                        else:
                            p_out.append((row, col, vals.pop(0)))
                    out.append(p_out)
                return out

            set_tile(*nxt0)
            staged1 = load_tile()
            nxt1 = advance(*nxt0)
            regs0 = flat(staged1)
            n_regs = len(regs0)
            it = (
                list(accs)
                + [(f"pp_r{i}", v) for i, v in enumerate(regs0)]
                + list(zip(st_names, nxt1[:n_st]))
            )
            loop = b.scf_for_iter(c0, n_tiles, c1, it, iv_name="t")
            with loop as (_t, iv):
                regs = iv[n_acc : n_acc + n_regs]
                st = list(iv[n_acc + n_regs :])
                if not is_3d:
                    st.append(None)
                pps = spec.tapk_pp_sched
                staged2 = None

                def ld(part=None):
                    if fast_a is None:
                        return load_tile(part)
                    return (
                        fast_a() if part != "b" else None,
                        load_tile("b")[1] if part != "a" else None,
                    )

                def issue_loads():
                    return ld()

                def interleave(n_mf, n_vm):
                    # The loads' address math co-issues with the MFMAs; the
                    # loads go in the phase's second half, one per 2 MFMAs.
                    first_vm = n_mf // 2
                    left = n_vm
                    for j in range(n_mf):
                        b.sched_group_barrier(_SG_MFMA, 1, 0)
                        b.sched_group_barrier(_SG_VALU, 4, 0)
                        b.sched_group_barrier(_SG_SALU, 2, 0)
                        if j >= first_vm and left > 0 and (j - first_vm) % 2 == 1:
                            b.sched_group_barrier(_SG_VMEM_READ, 1, 0)
                            left -= 1
                    if left:
                        b.sched_group_barrier(_SG_VMEM_READ, left, 0)

                n_mf_half = half * mfmas_m * mfmas_n
                # [mem A]
                frags = frag_reads()
                b.sched_barrier(0)
                if spec.tapk_pingpong == 5:
                    # Only the upper half's reads race a store (the lower
                    # half's [mem B] follows it); the lower half's next phase
                    # is compute A, so its reads may still be in flight
                    # across this barrier (its MFMAs wait for them).
                    with b.scf_if(upper):
                        b.s_waitcnt(lgkmcnt=0)
                    b.s_barrier_bare()
                else:
                    b.sync_lds_only()
                # [compute A] (pps 3: + tile t+2's loads, pps 4: its A loads,
                # interleaved). Tile t+2's uniform offsets and the coordinate
                # advance are scalar work: co-issued with the MFMAs here.
                b.s_setprio(1)
                if spec.tapk_pp_epi:
                    # Tiles t+2 >= n_tiles do not exist: their loads read
                    # out of range.
                    set_tile(*st, live=b.cmp_lt(b.add(_t, b.const_i32(2)), n_tiles))
                else:
                    set_tile(*st)
                nxt = advance(*st)
                if pps == 3:
                    staged2 = issue_loads()
                if pps == 4:
                    a2 = ld("a")[0]
                acc_a = mfmas(frags[:half], list(iv[:n_acc]))
                if pps == 3:
                    interleave(n_mf_half, len(flat(staged2)))
                elif pps == 4:
                    interleave(n_mf_half, len(flat([a2])))
                else:
                    for _j in range(n_mf_half):
                        b.sched_group_barrier(_SG_MFMA, 1, 0)
                        b.sched_group_barrier(_SG_SALU, 3, 0)
                b.sched_barrier(0)
                b.s_setprio(0)
                b.sync_lds_only()
                # [mem B]: store tile t+1 (pps 1: and issue tile t+2's loads),
                # advance the tile coordinate.
                store_tile(rebuild(staged1, regs), A_smem, B_smem)
                if pps == 1:
                    staged2 = issue_loads()
                if pps == 5:
                    a2 = ld("a")[0]
                if pps == 6:
                    b2 = ld("b")[1]
                b.sched_barrier(0)
                b.sync_lds_only()
                # [compute B] (pps 2: + tile t+2's loads, interleaved)
                b.s_setprio(1)
                if pps == 2:
                    staged2 = issue_loads()
                if pps in (4, 5):
                    staged2 = (a2, ld("b")[1])
                if pps == 6:
                    staged2 = (ld("a")[0], b2)
                acc_b = mfmas(frags[half:], acc_a)
                if pps == 2:
                    interleave(n_mf_half, len(flat(staged2)))
                if pps in (4, 5):
                    interleave(n_mf_half, len(flat([staged2[1]])))
                if pps == 6:
                    interleave(n_mf_half, len(flat([staged2[0]])))
                b.sched_barrier(0)
                b.s_setprio(0)
                b.sync_lds_only()
                b.scf_yield(*acc_b, *flat(staged2), *nxt[:n_st])
            return _pp_finish(list(loop.results)[:n_acc], wave)

        half = k_atoms // 2
        if spec.tapk_pp_sched:
            return _pp_two_ahead(frag_reads, mfmas, wave, half)
        it = list(accs) + list(zip(st_names, nxt0[:n_st]))
        loop = b.scf_for_iter(c0, n_tiles, c1, it, iv_name="t")
        with loop as (_t, iv):
            st = list(iv[n_acc:])
            if not is_3d:
                st.append(None)
            # [mem A]
            set_tile(*st)
            staged = load_tile()
            frags = frag_reads()
            b.sched_barrier(0)
            b.sync_lds_only()
            # [compute A]
            b.s_setprio(1)
            acc_a = mfmas(frags[:half], list(iv[:n_acc]))
            b.sched_barrier(0)
            b.s_setprio(0)
            b.sync_lds_only()
            # [mem B]
            store_tile(staged, A_smem, B_smem)
            b.sched_barrier(0)
            b.sync_lds_only()
            # [compute B]
            b.s_setprio(1)
            acc_b = mfmas(frags[half:], acc_a)
            b.sched_barrier(0)
            b.s_setprio(0)
            b.sync_lds_only()
            nxt = advance(*st)
            b.scf_yield(*acc_b, *nxt[:n_st])
        return _pp_finish(list(loop.results)[:n_acc], wave)

    def _pp_finish(results: list, wave: Value) -> list:
        """The lower half's matching barrier after the ping-pong loop. With
        tapk_pp_epi the accumulators are converted to bf16 before it, so the
        lower half's conversion overlaps the upper half's last MFMA phase."""
        if spec.tapk_pp_epi and spec.acc_epilogue.is_identity():
            _pp_state["bf16"] = [_pp_f32_to_bf16(b, spec, a) for a in results]
        with b.scf_if(b.cmp_lt(wave, b.const_i32(spec.warp_m * spec.warp_n // 2))):
            b.sync_lds_only()
        return results

    def _emit_tap_outer_prefetch_loop() -> list[Value]:
        """The pipelined tap-outer K loop (``tapk_prefetch``).

        The (tap, chunk) nest of :func:`_emit_tap_outer_loops` flattened into
        one loop over its ``taps * kpg / tile_k`` tiles, in the same order. The
        loop carries the next tile's uniform coordinate (chunk base kb, x, y
        and, in 3-D, z) and advances it with a compare-and-wrap. Each body
        issues the next tile's global reads, then the current tile's MFMAs,
        then stores the reads to LDS; ``sched_barrier`` pins the reads ahead
        of the MFMAs (left alone, the scheduler sinks them next to their
        stores and the latency is exposed again).
        """
        c1 = b.const_i32(1)
        c_kd = c_kpg if grouped else c_K
        c_cs = c_dg_N if grouped else c_C
        c_yx = b.mul(c_Y, c_X)
        c_taps = b.mul(p_Z, c_yx) if is_3d else c_yx
        c_yxc = b.mul(c_taps, c_cs)
        c_wo_k = b.mul(c_Wo, c_K)
        c_ho_wo_k = b.mul(c_Ho, c_wo_k)
        n_tiles = b.mul(c_taps, b.div(c_kd, c_block_k))
        cap: dict[str, Value] = {}

        def set_tile(kb0: Value, xd: Value, yd: Value, zd: Value | None, live=None):
            dh = b.sub(params["p_pH"], yd)
            dw = b.sub(params["p_pW"], xd)
            cap["dh"], cap["dw"] = dh, dw
            tap = b.add(b.mul(yd, c_X), xd)
            dy_tap = b.add(b.mul(dh, c_wo_k), b.mul(dw, c_K))
            if is_3d:
                dd = b.sub(p_pD, zd)
                cap["dd"] = dd
                tap = b.add(b.mul(zd, c_yx), tap)
                dy_tap = b.add(b.mul(dd, c_ho_wo_k), dy_tap)
            kb = b.add(kb0, k_out_group_base) if grouped else kb0
            cap["a_off"] = b.add(dy_tap, kb)
            cap["w_k"] = b.add(b.mul(kb, c_yxc), b.mul(tap, c_cs))
            cap["live"] = live
            cap["tap"] = tap

        def advance(kb0, xd, yd, zd):
            nk = b.add(kb0, c_block_k)
            wrap_k = b.cmp_ge(nk, c_kd)
            nk = b.select(wrap_k, c0, nk)
            nx = b.add(xd, b.select(wrap_k, c1, c0))
            wrap_x = b.cmp_ge(nx, c_X)
            nx = b.select(wrap_x, c0, nx)
            ny = b.add(yd, b.select(wrap_x, c1, c0))
            if not is_3d:
                return nk, nx, ny, None
            wrap_y = b.cmp_ge(ny, c_Y)
            ny = b.select(wrap_y, c0, ny)
            nz = b.add(zd, b.select(wrap_y, c1, c0))
            return nk, nx, ny, nz

        def dy_pf(b_: IRBuilder, row: Value, col: Value):
            m_sub = b_.add(block_m_off_v, row)
            n, m_rem = _magic_divmod(b_, m_sub, hw_div)
            hi, wi = _magic_divmod(b_, m_rem, w_div)
            ho = b_.add(hi, cap["dh"])
            wo = b_.add(wi, cap["dw"])
            if spec.tapk_pingpong:
                # One unsigned compare per axis (a negative coordinate wraps
                # past the bound): half the mask ops of the signed pair.
                valid = b_.land(b_.cmp_ult(ho, c_Ho), b_.cmp_ult(wo, c_Wo))
                if is_3d:
                    n, di = _magic_divmod(b_, n, di_div)
                    valid = b_.land(valid, b_.cmp_ult(b_.add(di, cap["dd"]), p_Do))
                    n = b_.add(b_.mul(n, p_Do), di)
                pix = b_.add(
                    b_.mul(b_.add(b_.mul(b_.add(b_.mul(n, c_Ho), hi), c_Wo), wi), c_K),
                    col,
                )
                return b_.add(pix, cap["a_off"]), valid
            valid = b_.land(
                b_.land(b_.cmp_ge(ho, c0), b_.cmp_lt(ho, c_Ho)),
                b_.land(b_.cmp_ge(wo, c0), b_.cmp_lt(wo, c_Wo)),
            )
            if is_3d:
                n, di = _magic_divmod(b_, n, di_div)
                do = b_.add(di, cap["dd"])
                valid = b_.land(
                    valid, b_.land(b_.cmp_ge(do, c0), b_.cmp_lt(do, p_Do))
                )
                n = b_.add(b_.mul(n, p_Do), di)
            if cap["live"] is not None:
                valid = b_.land(valid, cap["live"])
            pix = b_.add(
                b_.mul(b_.add(b_.mul(b_.add(b_.mul(n, c_Ho), hi), c_Wo), wi), c_K),
                col,
            )
            return b_.add(pix, cap["a_off"]), valid

        def w_pf(b_: IRBuilder, row: Value, col: Value):
            c_val = b_.add(block_n_off_v, row)
            off = b_.add(b_.add(b_.mul(col, c_yxc), c_val), cap["w_k"])
            if spec.tapk_pp_epi and cap["live"] is not None:
                # tapk_pp_epi: the two-ahead loads past the last tile are
                # sent out of range (no memory traffic, nothing to wait on).
                return off, cap["live"]
            return off, None

        if spec.lds_k_outer:

            def w_pf_fn(b_: IRBuilder, row: Value, col: Value):
                return w_pf(b_, col, row)

        else:
            w_pf_fn = w_pf

        def load_tile(part=None):
            # part "a" / "b": that operand's loads only (tapk_pp_sched 4).
            a_st = (
                a_sync_loader.load_global(b, tid=tid, descriptor=dy_pf, rsrc=dy_rsrc)
                if part != "b"
                else None
            )
            b_st = (
                b_sync_loader.load_global(b, tid=tid, descriptor=w_pf_fn, rsrc=w_rsrc)
                if part != "a"
                else None
            )
            return a_st, b_st

        def store_tile(st, a_dst, b_dst):
            if spec.tapk_pingpong:
                _pp_store(a_sync_loader, a_dst, st[0])
                _pp_store(b_sync_loader, b_dst, st[1])
                return
            a_sync_loader.store_lds(b, smem_dst=a_dst, staged=st[0])
            b_sync_loader.store_lds(b, smem_dst=b_dst, staged=st[1])

        def _pp_fast_a():
            """tapk_pingpong 4: the A (dY) tile loads without per-tile masks.
            Each thread's rows get their pixel offset and a bit mask of the
            filter taps that leave the image (bit = tap index) once, before
            the loop; a tile's load is then pixel + the tile's uniform
            offset, with bit 31 set from the tap's mask bit (an offset past
            2 GiB, which the buffer resource reads as zero)."""
            from rocke.helpers.loads import _ELEM_BYTES

            lv = a_sync_loader.load_vec
            cpv = block_k // lv
            c_cpv = b.const_i32(cpv)
            eb_sh = {1: 0, 2: 1, 4: 2}[_ELEM_BYTES[ir_dtype_a.name]]
            rows, cols, pix, crd = [], [], [], []
            for e in range(a_sync_loader.vecs_per_thread):
                idx = b.add(b.const_i32(e * threads), tid)
                row = b.div(idx, c_cpv)
                col = b.mul(b.sub(idx, b.mul(row, c_cpv)), b.const_i32(lv))
                m_sub = b.add(block_m_off_v, row)
                n, m_rem = _magic_divmod(b, m_sub, hw_div)
                hi, wi = _magic_divmod(b, m_rem, w_div)
                di = None
                if is_3d:
                    n, di = _magic_divmod(b, n, di_div)
                    n = b.add(b.mul(n, p_Do), di)
                pix.append(
                    b.add(
                        b.mul(b.add(b.mul(b.add(b.mul(n, c_Ho), hi), c_Wo), wi), c_K),
                        col,
                    )
                )
                rows.append(row)
                cols.append(col)
                crd.append((hi, wi, di))
            c1_ = b.const_i32(1)
            F = _PP_MAX_FILTER
            # Per axis, the valid filter offsets f of a row form one run:
            # 0 <= coord + pad - f < out  <=>  f in [coord+pad-out+1,
            # coord+pad], clamped to [0, F): a run of bits. The tap mask is
            # their product over (z, y, x), straight-line for filters of up
            # to F per axis (the validator enforces it).

            def axis_bits(coord, pad, out, ext):
                # ext: the axis' filter extent (<= F), so no bit spills
                # into the next tap group of the packed mask.
                hi_ = b.add(coord, pad)
                lo_ = b.add(b.sub(hi_, out), c1_)
                lo_c = b.smin(b.smax(lo_, c0), ext)
                hi_c = b.smin(hi_, b.sub(ext, c1_))
                width = b.smax(b.add(b.sub(hi_c, lo_c), c1_), c0)
                return b.shl(b.sub(b.shl(c1_, width), c1_), lo_c)

            masks = []
            zs = range(F) if is_3d else range(1)
            for hi, wi, di in crd:
                vx = axis_bits(wi, params["p_pW"], c_Wo, c_X)
                vy = axis_bits(hi, params["p_pH"], c_Ho, c_Y)
                vz = axis_bits(di, p_pD, p_Do, p_Z) if is_3d else c1_
                valid = c0
                for z in zs:
                    for y in range(F):
                        ok_u = b.cmp_lt(b.const_i32(y), c_Y)
                        if is_3d:
                            ok_u = b.land(ok_u, b.cmp_lt(b.const_i32(z), p_Z))
                        sh = b.mul(b.add(b.mul(b.const_i32(z), c_Y), b.const_i32(y)), c_X)
                        sh = b.select(ok_u, sh, c0)
                        bit = b.land(b.lshr(vy, b.const_i32(y)), c1_)
                        if is_3d:
                            bit = b.land(bit, b.lshr(vz, b.const_i32(z)))
                        part = b.select(ok_u, b.mul(bit, vx), c0)
                        valid = b.lor(valid, b.shl(part, sh))
                masks.append(b.xor(valid, b.const_i32(-1)))
            c_sh = b.const_i32(eb_sh)
            c31 = b.const_i32(31)

            def load_a():
                staged = []
                for e in range(len(rows)):
                    off = b.add(pix[e], cap["a_off"])
                    off = b.shl(off, c_sh) if eb_sh else off
                    bad = b.land(b.lshr(masks[e], cap["tap"]), c1_)
                    if cap["live"] is not None:
                        bad = b.lor(bad, b.select(cap["live"], c0, c1_))
                    off = b.lor(off, b.shl(bad, c31))
                    v = b.buffer_load_vN(dy_rsrc, off, c0, ir_dtype_a, lv)
                    staged.append((rows[e], cols[e], v))
                return staged

            return load_a

        # Prologue: tile 0 into the first buffer.
        z0 = c0 if is_3d else None
        s0 = (c0, c0, c0, z0)
        if spec.tapk_pingpong and spec.split_k > 1:
            # Split-K (ping-pong): slice z takes tiles [t0, t0 + per) of the
            # flat (z, y, x, chunk) order -- whole tiles, so no slice starts
            # mid-tap; its first tile's coordinate is decoded once.
            c_sk = b.const_i32(spec.split_k)
            per = b.div(b.add(n_tiles, b.sub(c_sk, c1)), c_sk)
            t0 = b.mul(b.block_id_z(), per)
            t_end = b.select(b.cmp_lt(b.add(t0, per), n_tiles), b.add(t0, per), n_tiles)
            cnt = b.sub(t_end, t0)
            n_tiles = b.select(b.cmp_gt(cnt, c0), cnt, c0)
            nkc = b.div(c_kd, c_block_k)
            tap0 = b.div(t0, nkc)
            kb_0 = b.mul(b.sub(t0, b.mul(tap0, nkc)), c_block_k)
            yz0 = b.div(tap0, c_X)
            x_0 = b.sub(tap0, b.mul(yz0, c_X))
            if is_3d:
                z_0 = b.div(yz0, c_Y)
                s0 = (kb_0, x_0, b.sub(yz0, b.mul(z_0, c_Y)), z_0)
            else:
                s0 = (kb_0, x_0, yz0, None)
        set_tile(*s0)
        pp_fast_a = None
        if spec.tapk_pingpong >= 4:
            # The tap masks are built while tile 0's loads are in flight.
            st0 = load_tile()
            pp_fast_a = _pp_fast_a()
            store_tile(st0, A_smem, B_smem)
        else:
            store_tile(load_tile(), A_smem, B_smem)
        b.sync()
        nxt0 = advance(*s0)
        n_st = 4 if is_3d else 3
        st_names = ["pf_kb", "pf_x", "pf_y", "pf_z"][:n_st]
        n_acc = len(accs)

        if spec.tapk_prefetch == 3:
            # Two tiles ahead: tile t+1 rides in registers across the back
            # edge (loop-carried), tile t+2's reads are issued before tile
            # t's MFMAs, so a read has two tiles of math to land.
            def flat(st):
                out = []
                for part in st:
                    for _row, _col, v in part:
                        out.extend(v if isinstance(v, list) else [v])
                return out

            def rebuild(template, vals):
                vals = list(vals)
                out = []
                for part in template:
                    p_out = []
                    for row, col, v in part:
                        if isinstance(v, list):
                            n = len(v)
                            p_out.append((row, col, vals[:n]))
                            del vals[:n]
                        else:
                            p_out.append((row, col, vals.pop(0)))
                    out.append(p_out)
                return out

            set_tile(*nxt0)
            st1 = load_tile()
            nxt1 = advance(*nxt0)
            regs0 = flat(st1)
            n_regs = len(regs0)
            it = (
                list(accs)
                + [(f"pf_r{i}", v) for i, v in enumerate(regs0)]
                + list(zip(st_names, nxt1[:n_st]))
            )
            loop = b.scf_for_iter(c0, n_tiles, c1, it, iv_name="t")
            with loop as (_t, iv):
                regs = iv[n_acc : n_acc + n_regs]
                st = iv[n_acc + n_regs :]
                set_tile(*st, *([None] if not is_3d else []))
                staged2 = load_tile()
                b.sched_barrier(0)
                new_accs = emit_mfma_phase(A_smem, B_smem, list(iv[:n_acc]))
                b.sync()
                store_tile(rebuild(staged2, regs), A_smem, B_smem)
                b.sync()
                nxt = advance(*st, *([None] if not is_3d else []))
                b.scf_yield(*new_accs, *flat(staged2), *nxt[:n_st])
            return list(loop.results)[:n_acc]

        if spec.tapk_pingpong:
            return _tapk_pingpong_loop(
                set_tile, advance, load_tile, store_tile, nxt0, st_names, n_tiles,
                fast_a=pp_fast_a,
            )

        if spec.tapk_prefetch == 1:
            it = list(accs) + list(zip(st_names, nxt0[:n_st]))
            loop = b.scf_for_iter(c0, n_tiles, c1, it, iv_name="t")
            with loop as (_t, iv):
                st = iv[n_acc:]
                set_tile(*st, *([None] if not is_3d else []))
                staged = load_tile()
                b.sched_barrier(0)
                new_accs = emit_mfma_phase(A_smem, B_smem, list(iv[:n_acc]))
                b.sync()
                store_tile(staged, A_smem, B_smem)
                b.sync()
                nxt = advance(*st, *([None] if not is_3d else []))
                b.scf_yield(*new_accs, *nxt[:n_st])
            return list(loop.results)[:n_acc]

        # tapk_prefetch == 2: ping-pong over two buffers, two tiles per body.
        c2 = b.const_i32(2)
        it = list(accs) + list(zip(st_names, nxt0[:n_st]))
        loop = b.scf_for_iter(c0, n_tiles, c2, it, iv_name="t2")
        with loop as (t2, iv):
            st = list(iv[n_acc:])
            if not is_3d:
                st.append(None)
            # Phase A: read tile t2+1 (predicated: it may not exist), MFMAs of
            # tile t2 from buffer 0, store into buffer 1.
            set_tile(*st, live=b.cmp_lt(b.add(t2, c1), n_tiles))
            staged = load_tile()
            b.sched_barrier(0)
            accs_a = emit_mfma_phase(A_smem, B_smem, list(iv[:n_acc]))
            b.sched_barrier(0)
            store_tile(staged, A_smem2, B_smem2)
            b.sync()
            # Phase B: read tile t2+2 into buffer 0, MFMAs of t2+1.
            st2 = list(advance(*st))
            set_tile(*st2)
            staged = load_tile()
            b.sched_barrier(0)
            accs_b = emit_mfma_phase(A_smem2, B_smem2, accs_a)
            b.sched_barrier(0)
            store_tile(staged, A_smem, B_smem)
            b.sync()
            nxt = advance(*st2)
            b.scf_yield(*accs_b, *nxt[:n_st])
        return list(loop.results)[:n_acc]

    def _xhalo_prefetch(xh_dy, w_xh_fn, halo_loader, row_map, cap) -> list[Value]:
        """``tapk_xhalo`` with ``tapk_prefetch`` 1 or 3: the (z, y, chunk)
        nest as one flat loop carrying the next tile's (kb, y, z), each
        tile's reads (the staged dY rows and the X W tiles) issued one
        (``1``) or two (``3``, the next tile in loop-carried registers)
        tiles ahead of its MFMAs. One LDS buffer."""
        XB = p.X
        c1 = b.const_i32(1)
        c_kd = c_kpg if grouped else c_K
        c_cs = c_dg_N if grouped else c_C
        c_yx = b.mul(c_Y, c_X)
        c_taps = b.mul(p_Z, c_yx) if is_3d else c_yx
        c_zy = b.mul(p_Z, c_Y) if is_3d else c_Y
        c_yxc = b.mul(c_taps, c_cs)
        n_tiles = b.mul(c_zy, b.div(c_kd, c_block_k))
        if spec.xh_b_wide:
            # xh_b_wide: the X taps' B tiles as one stacked unit space.
            xh_wide_loader = XTapKPackLoader(
                n_taps=XB,
                tile_rows=block_n,
                tile_cols=block_k,
                block_size=threads,
                load_vec=_dgrad_b_load_vec(spec),
                kpack=spec.b_kpack,
                elem_dtype=ir_dtype_b,
            )

            def w_wide_fn(b_: IRBuilder, x: Value, row: Value, col: Value):
                c_val = b_.add(block_n_off_v, row)
                return (
                    b_.add(
                        b_.add(b_.mul(col, c_yxc), c_val),
                        b_.add(cap["w_k0"], b_.mul(x, c_cs)),
                    ),
                    None,
                )

        else:
            xh_wide_loader = None
            w_wide_fn = None

        def set_tile(kb0, yd, zd):
            cap["dh"] = b.sub(params["p_pH"], yd)
            zy = yd
            if is_3d:
                cap["dd"] = b.sub(p_pD, zd)
                zy = b.add(b.mul(zd, c_Y), yd)
            kb = b.add(kb0, k_out_group_base) if grouped else kb0
            cap["kb"] = kb
            cap["w_k0"] = b.add(b.mul(kb, c_yxc), b.mul(b.mul(zy, c_X), c_cs))

        def advance(kb0, yd, zd):
            nk = b.add(kb0, c_block_k)
            wrap_k = b.cmp_ge(nk, c_kd)
            nk = b.select(wrap_k, c0, nk)
            ny = b.add(yd, b.select(wrap_k, c1, c0))
            if not is_3d:
                return nk, ny, None
            wrap_y = b.cmp_ge(ny, c_Y)
            ny = b.select(wrap_y, c0, ny)
            nz = b.add(zd, b.select(wrap_y, c1, c0))
            return nk, ny, nz

        def load_tile():
            # xh_order bit 1: the (small, L2-resident) B loads first.
            b_first = bool(spec.xh_order & 1)
            if not b_first:
                a_st = halo_loader.load_global(b, tid=tid, descriptor=xh_dy, rsrc=dy_rsrc)
            if xh_wide_loader is not None:
                w_st = xh_wide_loader.load_global(
                    b, tid=tid, descriptor=w_wide_fn, rsrc=w_rsrc
                )
                if b_first:
                    a_st = halo_loader.load_global(
                        b, tid=tid, descriptor=xh_dy, rsrc=dy_rsrc
                    )
                return [a_st, w_st]
            b_sts = []
            for x in range(XB):
                cap["w_k"] = b.add(cap["w_k0"], b.mul(b.const_i32(x), c_cs))
                b_sts.append(
                    b_sync_loader.load_global(
                        b, tid=tid, descriptor=w_xh_fn, rsrc=w_rsrc
                    )
                )
            if b_first:
                a_st = halo_loader.load_global(b, tid=tid, descriptor=xh_dy, rsrc=dy_rsrc)
            return [a_st] + b_sts

        def store_tile(st):
            xh_store(halo_loader, A_smem, st[0])
            if xh_wide_loader is not None:
                xh_store(xh_wide_loader, B_smem, st[1], is_b=True)
                return
            for x in range(XB):
                xh_store(b_sync_loader, B_smem_x[x], st[1 + x], is_b=True)

        def mfma(cur, deferred=None):
            if spec.tapk_lds_ahead or spec.xh_defer:
                return emit_mfma_phases_ahead(
                    [(A_smem, B_smem_x[x], row_map(x), xh_b_off[x]) for x in range(XB)],
                    cur,
                    depth=spec.tapk_lds_ahead,
                    defer=spec.xh_defer,
                    deferred=deferred,
                )
            for x in range(XB):
                cur = emit_mfma_phase(
                    A_smem, B_smem_x[x], cur, a_row_map=row_map(x), b_row_off=xh_b_off[x]
                )
            return cur

        def flat(st):
            out = []
            for part in st:
                for _row, _col, v in part:
                    out.extend(v if isinstance(v, list) else [v])
            return out

        def rebuild(template, vals):
            vals = list(vals)
            out = []
            for part in template:
                p_out = []
                for row, col, v in part:
                    if isinstance(v, list):
                        n = len(v)
                        p_out.append((row, col, vals[:n]))
                        del vals[:n]
                    else:
                        p_out.append((row, col, vals.pop(0)))
                out.append(p_out)
            return out

        n_st = 3 if is_3d else 2
        st_names = ["xp_kb", "xp_y", "xp_z"][:n_st]
        n_acc = len(accs)
        s0 = (c0, c0, c0 if is_3d else None)
        if spec.split_k > 1:
            # Split-K: slice z takes tiles [t0, t0 + per) of the flat
            # (z, y, chunk) order; whole tiles, so a slice never starts mid-tap.
            c_sk = b.const_i32(spec.split_k)
            per = b.div(b.add(n_tiles, b.sub(c_sk, c1)), c_sk)
            t0 = b.mul(b.block_id_z(), per)
            t_end = b.select(b.cmp_lt(b.add(t0, per), n_tiles), b.add(t0, per), n_tiles)
            cnt = b.sub(t_end, t0)
            n_tiles = b.select(b.cmp_gt(cnt, c0), cnt, c0)
            nkc = b.div(c_kd, c_block_k)
            zy0 = b.div(t0, nkc)
            kb_0 = b.mul(b.sub(t0, b.mul(zy0, nkc)), c_block_k)
            if is_3d:
                z_0 = b.div(zy0, c_Y)
                s0 = (kb_0, b.sub(zy0, b.mul(z_0, c_Y)), z_0)
            else:
                s0 = (kb_0, zy0, None)
        set_tile(*s0)
        store_tile(load_tile())
        b.sync()
        s1 = advance(*s0)
        if spec.tapk_prefetch == 3:
            set_tile(*s1)
            regs0 = flat(load_tile())
            s_next = advance(*s1)
        else:
            regs0 = []
            s_next = s1
        n_regs = len(regs0)
        it = (
            list(accs)
            + [(f"xp_r{i}", v) for i, v in enumerate(regs0)]
            + list(zip(st_names, s_next[:n_st]))
        )

        def _xh_body(acc_in, regs, st):
            """One tile of the loop: next loads, this tile's MFMAs, store."""
            set_tile(*st)
            if spec.xh_order & 2:
                # Nothing is pending at the loop top but the preheader's
                # kernarg s_load: drain it so the first MFMA waits with a
                # partial lgkmcnt.
                b.s_waitcnt(lgkmcnt=0)
            sprio = spec.tapk_setprio
            if sprio:
                # tapk_setprio: the wave's load issue + MFMA block at raised
                # priority (Triton's phase form), so with 2+ waves per SIMD
                # the arbiter favours the wave in its MFMA phase.
                b.sched_barrier(0)
                b.s_setprio(1)
            staged = load_tile()
            ilv = spec.tapk_ilv
            lda = spec.tapk_lds_ahead
            if not ilv and not lda and sprio != 2:
                # Every load issued as one block ahead of the MFMAs.
                b.sched_barrier(0)
            dfr_fns: list = []
            new_accs = mfma(list(acc_in), dfr_fns)
            if sprio == 2:
                # Phase-clustered: the next tile's global loads, then every
                # LDS fragment read of the tile, then all its MFMAs.
                n_mfma = (
                    XB
                    * spec.k_atoms_per_tile_k
                    * spec.mfmas_per_warp_m
                    * spec.mfmas_per_warp_n
                )
                n_rd = (
                    XB
                    * spec.k_atoms_per_tile_k
                    * (spec.mfmas_per_warp_m + spec.mfmas_per_warp_n)
                )
                b.sched_group_barrier(_SG_VMEM_READ, len(flat(staged)), 0)
                b.sched_group_barrier(_SG_DS_READ, n_rd, 0)
                b.sched_group_barrier(_SG_MFMA, n_mfma, 0)
                b.sched_barrier(0)
            if lda:
                # tapk_lds_ahead: one hint sequence for the whole tile. The
                # first step's fragment reads go first; then after each MFMA
                # ``lda`` reads of a later step (the next step's reads are
                # issued under the current step's first MFMAs, not right
                # before their own MFMAs) and, with tapk_ilv, one next-tile
                # global load every ``ilv`` MFMAs.
                n_vmem = len(flat(staged))
                per_step = spec.mfmas_per_warp_m * spec.mfmas_per_warp_n
                n_steps = XB * spec.k_atoms_per_tile_k
                reads = spec.mfmas_per_warp_m + spec.mfmas_per_warp_n
                if not ilv:
                    b.sched_group_barrier(_SG_VMEM_READ, n_vmem, 0)
                    n_vmem = 0
                b.sched_group_barrier(_SG_DS_READ, reads, 0)
                left_vmem = n_vmem
                n_pre = n_steps - spec.xh_defer
                for s_i in range(n_pre):
                    left_ds = reads if s_i + 1 < n_steps else 0
                    if spec.xh_defer and s_i == n_pre - 1:
                        # xh_defer: the deferred steps' reads too.
                        left_ds = reads * (n_steps - 1 - s_i)
                    for j in range(per_step):
                        b.sched_group_barrier(_SG_MFMA, 1, 0)
                        if left_ds > 0:
                            k = min(lda, left_ds)
                            b.sched_group_barrier(_SG_DS_READ, k, 0)
                            left_ds -= k
                        g = s_i * per_step + j
                        if ilv and left_vmem > 0 and g % ilv == ilv - 1:
                            b.sched_group_barrier(_SG_VMEM_READ, 1, 0)
                            left_vmem -= 1
                if left_vmem > 0:
                    b.sched_group_barrier(_SG_VMEM_READ, left_vmem, 0)
                if spec.xh_defer and left_ds > 0:
                    b.sched_group_barrier(_SG_DS_READ, left_ds, 0)
                b.sched_barrier(0)
            elif ilv:
                # tapk_ilv: one next-tile load every ``ilv`` MFMAs (CK hot-loop
                # style), the rest of the MFMAs after the last load.
                n_vmem = len(flat(staged))
                n_mfma = (
                    (XB * spec.k_atoms_per_tile_k - spec.xh_defer)
                    * spec.mfmas_per_warp_m
                    * spec.mfmas_per_warp_n
                )
                used = 0
                for _ in range(n_vmem):
                    b.sched_group_barrier(_SG_VMEM_READ, 1, 0)
                    k = min(ilv, n_mfma - used)
                    if k > 0:
                        b.sched_group_barrier(_SG_MFMA, k, 0)
                        used += k
                b.sched_barrier(0)
            # tapk_lds_barrier: LDS-only barriers, so tpf3's next-tile loads
            # stay in flight across them (sync drains vmcnt too).
            _bar = b.sync_lds_only if spec.tapk_lds_barrier else b.sync
            if sprio:
                b.sched_barrier(0)
                b.s_setprio(0)
            _bar()
            if dfr_fns:
                # xh_defer: the deferred MFMAs under the next tile's stores,
                # 1 ds_write : 2 MFMAs (fwd defer_mfma's measured best).
                store_tile(rebuild(staged, regs) if n_regs else staged)
                for fn_ in dfr_fns:
                    new_accs = fn_(new_accs)
                n_m = len(dfr_fns) * spec.mfmas_per_warp_m * spec.mfmas_per_warp_n
                n_w = len(staged[0]) * (
                    max(1, halo_loader.load_vec // 4) if _xh_ak4 else 1
                ) + sum(len(part) for part in staged[1:]) * _dgrad_b_load_vec(spec)
                per_w = min(2, max(1, n_m // max(1, n_w)))
                for _ in range(min(n_w, n_m)):
                    b.sched_group_barrier(_SG_DS_WRITE, 1, 0)
                    b.sched_group_barrier(_SG_MFMA, per_w, 0)
                b.sched_barrier(0)
            else:
                store_tile(rebuild(staged, regs) if n_regs else staged)
            _bar()
            nxt = advance(*st)
            carry = flat(staged) if n_regs else []
            return new_accs, carry, list(nxt)

        def _xh_iv(iv):
            st = list(iv[n_acc + n_regs :])
            if not is_3d:
                st.append(None)
            return list(iv[:n_acc]), iv[n_acc : n_acc + n_regs], st

        # xh_order bit 4 (tapk_prefetch 3): two tiles per iteration, so the
        # registers of the tile loaded two ahead are carried without the
        # loop-latch copies (and their vmcnt(0)) a one-tile body needs.
        unroll2 = bool(spec.xh_order & 4)
        n_iter = b.div(n_tiles, b.const_i32(2)) if unroll2 else n_tiles
        loop = b.scf_for_iter(c0, n_iter, c1, it, iv_name="t")
        with loop as (_t, iv):
            a_, r_, s_ = _xh_body(*_xh_iv(iv))
            if unroll2:
                a_, r_, s_ = _xh_body(a_, r_, s_)
            b.scf_yield(*a_, *r_, *s_[:n_st])
        if not unroll2:
            return list(loop.results)[:n_acc]
        # The odd last tile.
        tail_it = [(f"{name}_t1", v) for (name, _), v in zip(it, loop.results)]
        tail = b.scf_for_iter(
            c0, b.sub(n_tiles, b.mul(n_iter, b.const_i32(2))), c1, tail_it, iv_name="t1"
        )
        with tail as (_t1, iv):
            a_, r_, s_ = _xh_body(*_xh_iv(iv))
            b.scf_yield(*a_, *r_, *s_[:n_st])
        return list(tail.results)[:n_acc]

    def _emit_tap_outer_xhalo_loop() -> list[Value]:
        """The ``tapk_xhalo`` K loop. Returns the final accumulators.

        (z, y) taps outer, output-channel chunks inner, x taps unrolled. LDS
        row r of ``A_smem`` holds, for the iteration's (z, y) and chunk, the
        dY row tap (z, y, 0) reads for dX pixel m0 - (X-1) + r (pW == 0, so
        that is dY pixel (do, ho, wi)); tap x reads it for fragment row a at
        r = a + X-1-x, or the trailing zero row where wi(a) < x.
        """
        XB = p.X
        c1 = b.const_i32(1)
        c_kd = c_kpg if grouped else c_K
        c_cs = c_dg_N if grouped else c_C
        c_yx = b.mul(c_Y, c_X)
        c_taps = b.mul(p_Z, c_yx) if is_3d else c_yx
        c_zy = b.mul(p_Z, c_Y) if is_3d else c_Y
        c_yxc = b.mul(c_taps, c_cs)
        def _xhalo_padded_parts(lds_rows, c_zero_row, cap):
            """Loader / descriptor / row map of the padded halo: LDS row r
            holds pixel m0 - (X-1) + r; the rows are padded to whole loader
            passes (the rows past tile_m + X - 1 load nothing)."""
            c_real_rows = b.const_i32(block_m + XB - 1)
            c_back = b.const_i32(XB - 1)
            loader = CoalescedTileLoader(
                tile_rows=lds_rows - 1,
                tile_cols=block_k,
                block_size=threads,
                load_vec=load_vec_a,
                elem_dtype=ir_dtype_a,
            )

            def dy(b_: IRBuilder, row: Value, col: Value):
                m = b_.sub(b_.add(block_m_off_v, row), c_back)
                m_ok = b_.land(
                    b_.land(b_.cmp_ge(m, c0), b_.cmp_lt(m, p_dg_M)),
                    b_.cmp_lt(row, c_real_rows),
                )
                return _xh_dy_tail(b_, m, m_ok, col, cap)

            def row_map(x: int):
                shift = b.const_i32(XB - 1 - x)
                c_x = b.const_i32(x)

                def _map(a_row: Value) -> Value:
                    row = b.add(a_row, shift)
                    if x == 0:
                        return row
                    _, rem = _magic_divmod(b, b.add(block_m_off_v, a_row), hw_div)
                    _, wi = _magic_divmod(b, rem, w_div)
                    return b.select(b.cmp_ge(wi, c_x), row, c_zero_row)

                return _map

            return loader, dy, row_map

        def _xhalo_exact_parts(lds_rows, c_zero_row, cap):
            """xhalo_exact: LDS rows [0, tile_m) hold the tile's own pixels,
            [tile_m, tile_m + X-1) the back pixels m0 - (X-1).., then a junk
            row (idle lanes of the back load write it) and the zero row."""
            back = XB - 1
            c_junk_row = b.const_i32(lds_rows - 2)
            main_loader = CoalescedTileLoader(
                tile_rows=block_m,
                tile_cols=block_k,
                block_size=threads,
                load_vec=load_vec_a,
                elem_dtype=ir_dtype_a,
            )
            back_span = block_k // load_vec_a
            n_back_vec = back * back_span
            if n_back_vec > threads:
                raise ValueError(
                    f"xhalo_exact: {n_back_vec} back-row vectors exceed "
                    f"{threads} threads"
                )

            class _HaloLoader:
                """The main tile (exact loader passes) plus the back rows,
                one vector per lane for the first back*tile_k/vec lanes."""

                load_vec = load_vec_a

                @staticmethod
                def load_global(b_, *, tid, descriptor, rsrc):
                    staged = main_loader.load_global(
                        b_, tid=tid, descriptor=descriptor, rsrc=rsrc
                    )
                    c_span = b_.const_i32(back_span)
                    j = b_.div(tid, c_span)
                    col = b_.mul(b_.mod(tid, c_span), b_.const_i32(load_vec_a))
                    lane_ok = b_.cmp_lt(tid, b_.const_i32(n_back_vec))
                    off, valid = descriptor(b_, b_.sub(j, b_.const_i32(back)), col)
                    valid = b_.land(valid, lane_ok)
                    off_b = b_.select(
                        valid,
                        b_.mul(off, b_.const_i32(2 if ir_dtype_a != F32 else 4)),
                        b_.const_i32((1 << 31) - 1),
                    )
                    v = b_.buffer_load_vN(rsrc, off_b, c0, ir_dtype_a, load_vec_a)
                    row = b_.select(
                        lane_ok, b_.add(j, b_.const_i32(block_m)), c_junk_row
                    )
                    return list(staged) + [(row, col, v)]

                @staticmethod
                def store_lds(b_, *, smem_dst, staged):
                    for row, col, v in staged:
                        b_.smem_store_vN(smem_dst, [row, col], v, load_vec_a)

            def dy(b_: IRBuilder, row: Value, col: Value):
                # row: the pixel's offset from the tile's first pixel m0 (the
                # back rows pass -back..-1).
                m = b_.add(block_m_off_v, row)
                m_ok = b_.land(b_.cmp_ge(m, c0), b_.cmp_lt(m, p_dg_M))
                return _xh_dy_tail(b_, m, m_ok, col, cap)

            def row_map(x: int):
                c_x = b.const_i32(x)
                # Pixel m0 + a - x: main row a - x, or back row a - x + back.
                c_wrap = b.const_i32(block_m + back - x)

                def _map(a_row: Value) -> Value:
                    if x == 0:
                        return a_row
                    row = b.select(
                        b.cmp_ge(a_row, c_x), b.sub(a_row, c_x), b.add(a_row, c_wrap)
                    )
                    _, rem = _magic_divmod(b, b.add(block_m_off_v, a_row), hw_div)
                    _, wi = _magic_divmod(b, rem, w_div)
                    return b.select(b.cmp_ge(wi, c_x), row, c_zero_row)

                return _map

            return _HaloLoader(), dy, row_map

        def _xh_dy_tail(b_: IRBuilder, m: Value, m_ok: Value, col: Value, cap):
            """dY offset / validity of dX pixel ``m`` at the iteration's tap."""
            m = b_.select(m_ok, m, c0)
            n, m_rem = _magic_divmod(b_, m, hw_div)
            hi, wi = _magic_divmod(b_, m_rem, w_div)
            ho = b_.add(hi, cap["dh"])
            valid = b_.land(
                m_ok,
                b_.land(
                    b_.land(b_.cmp_ge(ho, c0), b_.cmp_lt(ho, c_Ho)),
                    b_.cmp_lt(wi, c_Wo),
                ),
            )
            if is_3d:
                n, di = _magic_divmod(b_, n, di_div)
                do = b_.add(di, cap["dd"])
                valid = b_.land(
                    valid, b_.land(b_.cmp_ge(do, c0), b_.cmp_lt(do, p_Do))
                )
                n = b_.add(b_.mul(n, p_Do), do)
            pix = b_.add(b_.mul(b_.add(b_.mul(n, c_Ho), ho), c_Wo), wi)
            offset = b_.add(b_.add(b_.mul(pix, c_K), col), cap["kb"])
            return offset, valid

        back = XB - 1
        lds_rows = spec.xhalo_lds_rows
        c_zero_row = b.const_i32(lds_rows - 1)
        cap: dict[str, Value] = {}
        if spec.xhalo_exact:
            halo_loader, xh_dy, row_map = _xhalo_exact_parts(lds_rows, c_zero_row, cap)
        else:
            halo_loader, xh_dy, row_map = _xhalo_padded_parts(lds_rows, c_zero_row, cap)

        def w_xh(b_: IRBuilder, row: Value, col: Value):
            c_val = b_.add(block_n_off_v, row)
            return b_.add(b_.add(b_.mul(col, c_yxc), c_val), cap["w_k"]), None

        if spec.lds_k_outer:

            def w_xh_fn(b_: IRBuilder, row: Value, col: Value):
                return w_xh(b_, col, row)

        else:
            w_xh_fn = w_xh

        # Zero row: every thread writes the same zeros (benign); the first
        # barrier orders it before any read, and the loader never writes it.
        if _xh_ak4:
            # One 4-element zero piece per k-group slab.
            zk0 = b.mod(tid, b.const_i32(block_k // 4))
            b.smem_store_vN(
                A_smem,
                [b.add(b.mul(zk0, b.const_i32(_ak4_mp)), c_zero_row), c0],
                b.zero_vec(ir_dtype_a, 4),
                4,
            )
        else:
            zvec = min(8, block_k)
            zcol = b.mul(b.mod(tid, b.const_i32(block_k // zvec)), b.const_i32(zvec))
            b.smem_store_vN(A_smem, [c_zero_row, zcol], b.zero_vec(ir_dtype_a, zvec), zvec)

        if spec.tapk_prefetch:
            return _xhalo_prefetch(xh_dy, w_xh_fn, halo_loader, row_map, cap)
        if spec.xh_b_wide:
            raise ValueError("xh_b_wide is built on the x-halo prefetch loop only")

        zy_for = b.scf_for_iter(c0, c_zy, c1, accs, iv_name="zy")
        with zy_for as (zy, zy_vars):
            if is_3d:
                zdot = b.div(zy, c_Y)
                ydot = b.sub(zy, b.mul(zdot, c_Y))
                cap["dd"] = b.sub(p_pD, zdot)
            else:
                ydot = zy
            cap["dh"] = b.sub(params["p_pH"], ydot)
            # Tap (z, y, 0) and the W offset of its x = 0 tile.
            tap0_cs = b.mul(b.mul(zy, c_X), c_cs)
            chunk_accs = [(f"{name}_xh", v) for (name, _), v in zip(accs, zy_vars)]
            chunk_for = b.scf_for_iter(c0, c_kd, c_block_k, chunk_accs, iv_name="kb")
            with chunk_for as (kb0, iter_vars):
                kb = b.add(kb0, k_out_group_base) if grouped else kb0
                cap["kb"] = kb
                w_k0 = b.add(b.mul(kb, c_yxc), tap0_cs)
                a_st = halo_loader.load_global(
                    b, tid=tid, descriptor=xh_dy, rsrc=dy_rsrc
                )
                b_sts = []
                for x in range(XB):
                    cap["w_k"] = b.add(w_k0, b.mul(b.const_i32(x), c_cs))
                    b_sts.append(
                        b_sync_loader.load_global(
                            b, tid=tid, descriptor=w_xh_fn, rsrc=w_rsrc
                        )
                    )
                xh_store(halo_loader, A_smem, a_st)
                for x in range(XB):
                    xh_store(b_sync_loader, B_smem_x[x], b_sts[x], is_b=True)
                b.sync()
                cur = list(iter_vars)
                for x in range(XB):
                    cur = emit_mfma_phase(A_smem, B_smem_x[x], cur, a_row_map=row_map(x))
                b.sync()
                b.scf_yield(*cur)
            b.scf_yield(*chunk_for.results)
        return list(zy_for.results)

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
    elif spec.tap_outer_k and spec.tapk_xhalo:
        final_accs = _emit_tap_outer_xhalo_loop()
    elif spec.tap_outer_k and spec.tapk_prefetch:
        final_accs = _emit_tap_outer_prefetch_loop()
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
    dX_desc = make_dgrad_dx_descriptor_dynamic(
        b, params, is_3d=spec.problem.is_3d
    )
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
    dX_desc = make_dgrad_dx_descriptor_dynamic(
        b, params, is_3d=spec.problem.is_3d
    )

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
    pre_bf16: Optional[Sequence[Value]] = None,
) -> None:
    """MFMA cshuffle epilogue for stride=1 dX."""
    if spec.tapk_pp_epi:
        return _emit_pp_cshuffle_epilogue(
            b, spec, accs, grid, dx_rsrc, params, c_group_base, pre_bf16=pre_bf16
        )
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
    _epi = dc_replace(
        _epi,
        war_barriers=_war_barriers,
        m_chunks=spec.cshuffle_m_chunks,
    )
    _epi.store(
        b,
        accs=accs,
        addr_fn=dx_addr,
        d_rsrc=dx_rsrc,
        bounds=bounds,
    )


def _pp_f32_to_bf16(b: IRBuilder, spec: "DgradConvSpec", acc: Value) -> Value:
    """f32 -> bf16 (round to nearest even) of a ping-pong accumulator vector.

    tapk_pp_epi 2 drops the NaN fix-up of the generic conversion: an MFMA
    accumulator of bf16 operands is NaN only as a propagated bf16 NaN (low
    16 bits zero) or the canonical quiet NaN, which the rounding add keeps a
    NaN; infinities and overflow round as RNE does. 2 VALU per element plus
    one pack per pair instead of ~6."""
    if spec.tapk_pp_epi != 2:
        return b.vec_trunc_f32_to_bf16(acc)
    from rocke.core.ir import I32 as _I32, VectorType as _VT

    n = acc.type.count
    bits = b.vec_bitcast(acc, _VT(_I32, n))
    c1 = b.const_i32(1)
    c16 = b.const_i32(16)
    bias = b.const_i32(0x7FFF)
    his = []
    for i in range(n):
        x = b.vec_extract(bits, i)
        lsb = b.land(b.lshr(x, c16), c1)
        his.append(b.lshr(b.add(b.add(x, bias), lsb), c16))
    words = [b.lor(his[2 * j], b.shl(his[2 * j + 1], c16)) for j in range(n // 2)]
    return b.vec_bitcast(b.vec_pack(words, _I32), _VT(BF16, n))


def _emit_pp_cshuffle_epilogue(
    b: IRBuilder,
    spec: DgradConvSpec,
    accs: Sequence[Value],
    grid: WarpGrid,
    dx_rsrc: Value,
    params: Dict[str, Value],
    c_group_base: Optional[Value] = None,
    pre_bf16: Optional[Sequence[Value]] = None,
) -> None:
    """``tapk_pp_epi``: the cshuffle store of the ping-pong kernels (16x16
    atoms, bf16 dX) in ``cshuffle_m_chunks`` row slabs, with the staging
    rows' 16-byte chunks XOR-swizzled by the row's 4-row group: one
    ds_write_b16 of a wave writes rows 4*m_blk + i of 16 columns for
    m_blk = 0..3, which the unswizzled 512-byte rows put on one bank."""
    from rocke.helpers.tensor_view import make_lds_view

    dx_addr, bounds = _dgrad_stride1_dx_addr(b, spec, params, c_group_base)
    atom = spec.atom
    mc = spec.cshuffle_m_chunks
    mfmas_m, mfmas_n = spec.mfmas_per_warp_m, spec.mfmas_per_warp_n
    cm = mfmas_m // mc
    am, an = atom.m, atom.n
    rows_c = spec.tile_m // mc
    tile_n = spec.tile_n
    threads = spec.block_size
    sv = 8
    c_view = make_lds_view(b, dtype=BF16, shape=(rows_c, tile_n), name_hint="C_smem")
    c_smem = c_view.base
    c1 = b.const_i32(1)
    c3 = b.const_i32(3)
    c4 = b.const_i32(4)

    def swz(row: Value, col: Value) -> Value:
        if spec.tapk_pp_epi == 2:
            return b.xor(col, b.shl(b.land(row, b.const_i32(7)), c3))
        return b.xor(col, b.shl(b.land(b.lshr(row, b.const_i32(2)), c3), c4))

    warp_m_off_c = b.mul(grid.warp_m_idx, b.const_i32(cm * am))
    warp_n_off = grid.warp_n_off(b)
    c_threads = b.const_i32(threads)
    c_tnv = b.const_i32(tile_n // sv)
    c_slab = b.const_i32(cm * am)
    c_wrows = b.const_i32(mfmas_m * am)
    c_eb = b.const_i32(2)
    oob = b.const_i32((1 << 31) - 1)
    vecs_per_thread = (rows_c * tile_n // sv) // threads
    if vecs_per_thread * threads * sv != rows_c * tile_n:
        raise ValueError("tapk_pp_epi: the C slab does not split into 8-wide vectors per thread")
    for c in range(mc):
        b.sync()
        for mi_l in range(cm):
            mi = c * cm + mi_l
            for ni in range(mfmas_n):
                v = (
                    pre_bf16[mi * mfmas_n + ni]
                    if pre_bf16 is not None
                    else _pp_f32_to_bf16(b, spec, accs[mi * mfmas_n + ni])
                )
                if spec.tapk_pp_epi == 2:
                    # Transposed C: lane l holds row l % 16, columns
                    # 4 * (l / 16) .. +3 of the atom -- one 8-byte write.
                    r = b.land(grid.lane, b.const_i32(am - 1))
                    cc = b.shl(b.lshr(grid.lane, c4), b.const_i32(2))
                    row = b.add(warp_m_off_c, b.add(b.const_i32(mi_l * am), r))
                    col = b.add(warp_n_off, b.add(b.const_i32(ni * an), cc))
                    b.smem_store_vN(c_smem, [row, swz(row, col)], v, atom.c_per_lane)
                    continue
                for i in range(atom.c_per_lane):
                    r, cc = atom.lane_to_output(b, grid.lane, i)
                    row = b.add(warp_m_off_c, b.add(b.const_i32(mi_l * am), r))
                    col = b.add(warp_n_off, b.add(b.const_i32(ni * an), cc))
                    b.smem_store_vN(c_smem, [row, swz(row, col)], b.vec_extract(v, i), 1)
        b.sync()
        for e in range(vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), grid.tid)
            row = b.div(vec_idx, c_tnv)
            col = b.mul(b.sub(vec_idx, b.mul(row, c_tnv)), b.const_i32(sv))
            w = b.div(row, c_slab)
            r_in = b.sub(row, b.mul(w, c_slab))
            g_row = b.add(b.add(b.mul(w, c_wrows), b.const_i32(c * cm * am)), r_in)
            m_val = b.add(grid.block_m_off, g_row)
            n_val = b.add(grid.block_n_off, col)
            ok = DirectEpilogue._bounds_check(b, m_val, n_val, bounds, vec_n=sv)
            off_elems, valid = dx_addr(b, m_val, n_val)
            if valid is not None:
                ok = valid if ok is None else b.land(ok, valid)
            off_b = b.mul(off_elems, c_eb)
            safe = b.select(ok, off_b, oob) if ok is not None else off_b
            vbf = b.smem_load_vN(c_smem, row, swz(row, col), dtype=BF16, n=sv)
            b.buffer_store_vN_bf16(dx_rsrc, safe, b.const_i32(0), vbf, sv // 2)


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
