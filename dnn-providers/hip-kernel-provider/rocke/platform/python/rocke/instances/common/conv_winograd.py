# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Winograd convolution kernel instances (NHWC × KYXC -> NHWK, stride=1, 3x3 filter).

Multi-pass Winograd pipeline — three separate kernels compiled at runtime via
the rocke IR:

  1. **Data transform**   — B^T * input_patch * B   per (n, tile_h, tile_w, c)
  2. **Filter transform** — G * filter_patch * G^T  per (k, c)   [done once]
  3. **GEMM**             — batched element-wise multiply across xform domain,
                           one batched GEMM per (xform_h, xform_w) position;
                           use the existing ``gemm_universal`` infrastructure.
  4. **Output transform** — A^T * acc_tile * A      per (n, tile_h, tile_w, k)

Only the data-transform kernel, the filter-transform kernel, and the
output-transform kernel are authored here.  The GEMM step uses
``build_implicit_gemm_conv`` (or any batched GEMM provider) externally — the
spec exposes the required workspace shapes so the caller can wire them up.

Authoring style::

    spec = WinogradConvSpec(
        problem=WinogradProblem(N=8, Hi=56, Wi=56, C=64, K=64, pH=1, pW=1),
        out_tile=4,           # F(4,3): 4-output × 3-filter, 6×6 transform domain
        block_c=64,           # channels per block in data/filter transforms
        block_k=64,           # output channels per block in filter/output transforms
        block_nhw=4,          # (n, tile_h, tile_w) triples per block
    )
    data_kdef   = build_winograd_data_transform(spec)
    filter_kdef = build_winograd_filter_transform(spec)
    out_kdef    = build_winograd_output_transform(spec)

Shared internal helpers live in
:mod:`._conv_winograd_common` (underscore-prefixed, internal).

**C++ engine mirror required** — per the byte-identity rule,
a C++ counterpart in ``platform/cpp/`` must be added in the same PR as this
file before the family can be declared done.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from ...core.ir import (
    F16,
    F32,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
)
from ...helpers.spec import kernel_name_join
from ...helpers.tensor_view import make_buffer_resource
from ._conv_winograd_common import (  # noqa: F401 — re-exported for callers
    WINOGRAD_TILES,
    WinogradProblem,
    WinogradTile,
    emit_data_transform,
    emit_filter_transform,
    emit_output_transform,
)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WinogradConvSpec:
    """Configuration for the three Winograd kernel instances.

    ``out_tile`` selects the tile variant:
      - 2 → F(2,3): 2-output × 3-filter, 4×4 transform domain (lower overhead)
      - 4 → F(4,3): 4-output × 3-filter, 6×6 transform domain (better FLOP reduction)

    Threading geometry:
      Data-transform kernel:
        grid:  (ceil(N*tH*tW / block_nhw), ceil(C / block_c), 1)
        block: (block_nhw * block_c, 1, 1)

      Filter-transform kernel:
        grid:  (ceil(K / block_k), ceil(C / block_c), 1)
        block: (block_k * block_c, 1, 1)

      Output-transform kernel:
        grid:  (ceil(N*tH*tW / block_nhw), ceil(K / block_k), 1)
        block: (block_nhw * block_k, 1, 1)
    """

    problem: WinogradProblem
    name: str = "conv_winograd"

    out_tile: int = 4

    block_c: int = 32
    block_k: int = 32
    block_nhw: int = 4

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.out_tile not in (2, 4):
            raise ValueError(f"out_tile must be 2 or 4, got {self.out_tile}")
        if self.block_c < 1 or self.block_k < 1 or self.block_nhw < 1:
            raise ValueError("block_c, block_k, block_nhw must be >= 1")
        if self.block_nhw * self.block_c > 1024:
            raise ValueError(
                f"data/output transform block size {self.block_nhw * self.block_c} > 1024"
            )
        if self.block_k * self.block_c > 1024:
            raise ValueError(
                f"filter transform block size {self.block_k * self.block_c} > 1024"
            )

    @property
    def tile(self) -> WinogradTile:
        return WINOGRAD_TILES[(self.out_tile, 3)]

    @property
    def xform_size(self) -> int:
        return self.tile.xform_size

    @property
    def tiles_h(self) -> int:
        return math.ceil(self.problem.Ho / self.out_tile)

    @property
    def tiles_w(self) -> int:
        return math.ceil(self.problem.Wo / self.out_tile)

    @property
    def num_tiles(self) -> int:
        return self.tiles_h * self.tiles_w

    def kernel_name(self, suffix: str) -> str:
        p = self.problem
        return kernel_name_join(
            self.name,
            suffix,
            f"N{p.N}H{p.Hi}W{p.Wi}C{p.C}K{p.K}",
            f"f{self.out_tile}x3",
            f"bc{self.block_c}bk{self.block_k}bnhw{self.block_nhw}",
        )

    def data_workspace_shape(self) -> Tuple[int, int, int, int]:
        """(xform_h, xform_w, N*tiles_h*tiles_w, C) — data transform output."""
        xs = self.xform_size
        return (xs, xs, self.problem.N * self.num_tiles, self.problem.C)

    def filter_workspace_shape(self) -> Tuple[int, int, int, int]:
        """(xform_h, xform_w, K, C) — filter transform output."""
        xs = self.xform_size
        return (xs, xs, self.problem.K, self.problem.C)

    def gemm_result_shape(self) -> Tuple[int, int, int, int]:
        """(xform_h, xform_w, N*tiles_h*tiles_w, K) — per-position GEMM output."""
        xs = self.xform_size
        return (xs, xs, self.problem.N * self.num_tiles, self.problem.K)


# ---------------------------------------------------------------------------
# Arch validation
# ---------------------------------------------------------------------------

_SUPPORTED_ARCHES = frozenset(
    [
        "gfx942",
        "gfx950",
        "gfx1100",
        "gfx1101",
        "gfx1151",
        "gfx1200",
        "gfx1201",
        "gfx1250",
    ]
)


def is_valid_spec(spec: WinogradConvSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``."""
    if arch not in _SUPPORTED_ARCHES:
        return False, (
            f"Winograd conv not tested on {arch}; "
            f"supported: {sorted(_SUPPORTED_ARCHES)}"
        )
    p = spec.problem
    if p.C % spec.block_c != 0:
        return False, f"C={p.C} not divisible by block_c={spec.block_c}"
    if p.K % spec.block_k != 0:
        return False, f"K={p.K} not divisible by block_k={spec.block_k}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------


def winograd_data_transform_grid(spec: WinogradConvSpec) -> Tuple[int, int, int]:
    """Launch grid: (grid_nhw, grid_c, 1)."""
    p = spec.problem
    total_nhw = p.N * spec.num_tiles
    return (math.ceil(total_nhw / spec.block_nhw), math.ceil(p.C / spec.block_c), 1)


def winograd_filter_transform_grid(spec: WinogradConvSpec) -> Tuple[int, int, int]:
    """Launch grid: (grid_k, grid_c, 1)."""
    p = spec.problem
    return (math.ceil(p.K / spec.block_k), math.ceil(p.C / spec.block_c), 1)


def winograd_output_transform_grid(spec: WinogradConvSpec) -> Tuple[int, int, int]:
    """Launch grid: (grid_nhw, grid_k, 1)."""
    p = spec.problem
    total_nhw = p.N * spec.num_tiles
    return (math.ceil(total_nhw / spec.block_nhw), math.ceil(p.K / spec.block_k), 1)


# ---------------------------------------------------------------------------
# Internal: OOB-safe offset helper
# ---------------------------------------------------------------------------

_OOB_SENTINEL = (1 << 31) - 1  # clamps to 0 in AMD buffer descriptor


def _safe_off(b: IRBuilder, valid: "Value", real_off: "Value") -> "Value":
    """Return real_off when valid, or the OOB sentinel (AMD buffer clamp) otherwise."""
    return b.select(valid, real_off, b.const_i32(_OOB_SENTINEL))


# ---------------------------------------------------------------------------
# Kernel 1: Data transform  —  B^T * input_patch * B
# ---------------------------------------------------------------------------


def build_winograd_data_transform(
    spec: WinogradConvSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Build the data-transform kernel.

    Each thread covers one (n, tile_h, tile_w, c) combination.  It loads the
    corresponding (xform_size x xform_size) patch from the padded NHWC input,
    emits the B^T × patch × B arithmetic as explicit fp32 SSA, and writes the
    (xform_size × xform_size) result to the flat data workspace tensor.

    Kernel parameters::

        A            : ptr<f16> global  — NHWC input  [N, Hi, Wi, C]
        A_bytes      : i32
        DataWs       : ptr<f32> global  — data workspace [xs*xs * N*tiles * C]
        DataWs_bytes : i32
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid WinogradConvSpec for {arch}: {why}")

    tile = spec.tile
    xs = tile.xform_size
    ot = tile.out_tile
    p = spec.problem
    block_nhw = spec.block_nhw
    block_c = spec.block_c

    b = IRBuilder(spec.kernel_name("data_xform"))
    b.kernel.attrs["max_workgroup_size"] = block_nhw * block_c

    A_ptr = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    DataWs = b.param(
        "DataWs", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
    )
    DataWs_bytes = b.param("DataWs_bytes", I32)

    c0 = b.const_i32(0)
    c2 = b.const_i32(2)
    c4 = b.const_i32(4)
    c_Hi = b.const_i32(p.Hi)
    c_Wi = b.const_i32(p.Wi)
    c_C = b.const_i32(p.C)
    c_pH = b.const_i32(p.pH)
    c_pW = b.const_i32(p.pW)
    c_tiles_w = b.const_i32(spec.tiles_w)
    c_ot = b.const_i32(ot)
    c_num_tiles = b.const_i32(spec.num_tiles)
    c_ntiles_total = b.const_i32(p.N * spec.num_tiles)

    tid = b.thread_id_x()
    bid_nhw = b.block_id_x()
    bid_c = b.block_id_y()

    c_block_nhw = b.const_i32(block_nhw)
    c_block_c = b.const_i32(block_c)
    local_nhw = b.mod(tid, c_block_nhw)
    local_c = b.div(tid, c_block_nhw)

    nhw_idx = b.add(b.mul(bid_nhw, c_block_nhw), local_nhw)
    c_idx = b.add(b.mul(bid_c, c_block_c), local_c)

    # Decompose nhw_idx -> (n, tile_h, tile_w)
    tile_idx = b.mod(nhw_idx, c_num_tiles)
    n_idx = b.div(nhw_idx, c_num_tiles)
    tile_h_idx = b.div(tile_idx, c_tiles_w)
    tile_w_idx = b.mod(tile_idx, c_tiles_w)

    # Top-left corner in padded coords
    hi_base = b.sub(b.mul(tile_h_idx, c_ot), c_pH)
    wi_base = b.sub(b.mul(tile_w_idx, c_ot), c_pW)

    a_rsrc = make_buffer_resource(b, A_ptr, num_bytes=A_bytes).rsrc
    dws_rsrc = make_buffer_resource(b, DataWs, num_bytes=DataWs_bytes).rsrc

    # Guard: threads beyond problem bounds do nothing
    nhw_valid = b.cmp_lt(nhw_idx, c_ntiles_total)
    c_valid = b.cmp_lt(c_idx, c_C)
    both_valid = b.land(nhw_valid, c_valid)

    with b.scf_if(both_valid):
        # Load xform_size x xform_size input patch
        patch: list = [[None] * xs for _ in range(xs)]
        for rr in range(xs):
            for cc in range(xs):
                hi = b.add(hi_base, b.const_i32(rr))
                wi = b.add(wi_base, b.const_i32(cc))
                # Bounds check: 0 <= hi < Hi and 0 <= wi < Wi
                hi_ok = b.land(
                    b.cmp_ge(hi, c0),
                    b.cmp_lt(hi, c_Hi),
                )
                wi_ok = b.land(
                    b.cmp_ge(wi, c0),
                    b.cmp_lt(wi, c_Wi),
                )
                in_bounds = b.land(hi_ok, wi_ok)

                # NHWC offset: ((n*Hi + hi)*Wi + wi)*C + c, then * 2 for f16 bytes
                row_off = b.add(b.mul(n_idx, c_Hi), hi)
                col_off = b.add(b.mul(row_off, c_Wi), wi)
                elem_off = b.add(b.mul(col_off, c_C), c_idx)
                byte_off = b.mul(elem_off, c2)
                # OOB-safe: AMD buffer descriptor returns 0 for sentinel offset
                safe = _safe_off(b, in_bounds, byte_off)
                loaded_f16 = b.buffer_load_f16(a_rsrc, safe, c0)
                patch[rr][cc] = b.cast_to_f32(loaded_f16)

        # Apply B^T * patch * B
        xformed = emit_data_transform(b, tile, patch)

        # Store (xs x xs) results to data workspace
        # Layout [xs*xs * N*num_tiles * C]: offset(xh, xw, nhw, c)
        #   = ((xh*xs + xw) * ntotal + nhw) * C + c
        for xh in range(xs):
            for xw in range(xs):
                xpos = b.const_i32(xh * xs + xw)
                nhw_layer = b.add(b.mul(xpos, c_ntiles_total), nhw_idx)
                ws_off = b.add(b.mul(nhw_layer, c_C), c_idx)
                ws_byte = b.mul(ws_off, c4)
                b.buffer_store_f32(dws_rsrc, ws_byte, c0, xformed[xh][xw])

    return b.kernel


# ---------------------------------------------------------------------------
# Kernel 2: Filter transform  —  G * filter * G^T
# ---------------------------------------------------------------------------


def build_winograd_filter_transform(
    spec: WinogradConvSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Build the filter-transform kernel.

    Each thread covers one (k, c) pair.  It reads the 3×3 filter patch from
    the KYXC weight tensor, emits G × g × G^T in fp32 SSA, and writes the
    (xform_size × xform_size) result to the filter workspace tensor.

    Kernel parameters::

        W              : ptr<f16> global  — KYXC filter  [K, 3, 3, C]
        W_bytes        : i32
        FilterWs       : ptr<f32> global  — filter workspace [xs*xs * K * C]
        FilterWs_bytes : i32
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid WinogradConvSpec for {arch}: {why}")

    tile = spec.tile
    xs = tile.xform_size
    fs = tile.filter_size
    p = spec.problem
    block_k = spec.block_k
    block_c = spec.block_c

    b = IRBuilder(spec.kernel_name("filter_xform"))
    b.kernel.attrs["max_workgroup_size"] = block_k * block_c

    W_ptr = b.param("W", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    W_bytes = b.param("W_bytes", I32)
    FilterWs = b.param(
        "FilterWs", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
    )
    FilterWs_bytes = b.param("FilterWs_bytes", I32)

    c0 = b.const_i32(0)
    c2 = b.const_i32(2)
    c4 = b.const_i32(4)
    c_K = b.const_i32(p.K)
    c_C = b.const_i32(p.C)
    c_fs = b.const_i32(fs)

    tid = b.thread_id_x()
    bid_k = b.block_id_x()
    bid_c = b.block_id_y()

    c_block_k = b.const_i32(block_k)
    c_block_c = b.const_i32(block_c)
    local_k = b.mod(tid, c_block_k)
    local_c = b.div(tid, c_block_k)

    k_idx = b.add(b.mul(bid_k, c_block_k), local_k)
    c_idx = b.add(b.mul(bid_c, c_block_c), local_c)

    w_rsrc = make_buffer_resource(b, W_ptr, num_bytes=W_bytes).rsrc
    fws_rsrc = make_buffer_resource(b, FilterWs, num_bytes=FilterWs_bytes).rsrc

    k_valid = b.cmp_lt(k_idx, c_K)
    c_valid_v = b.cmp_lt(c_idx, c_C)
    both_valid = b.land(k_valid, c_valid_v)

    with b.scf_if(both_valid):
        # KYXC layout: element (k, y, x, c): offset = ((k*fs + y)*fs + x)*C + c
        filter_patch: list = [[None] * fs for _ in range(fs)]
        for fy in range(fs):
            for fx in range(fs):
                yx_off = b.add(b.mul(k_idx, c_fs), b.const_i32(fy))
                yx_off2 = b.add(b.mul(yx_off, c_fs), b.const_i32(fx))
                elem_off = b.add(b.mul(yx_off2, c_C), c_idx)
                byte_off = b.mul(elem_off, c2)
                loaded_f16 = b.buffer_load_f16(w_rsrc, byte_off, c0)
                filter_patch[fy][fx] = b.cast_to_f32(loaded_f16)

        # Apply G * patch * G^T
        xformed = emit_filter_transform(b, tile, filter_patch)

        # Store to filter workspace: layout [xs*xs * K * C]
        # offset(xh, xw, k, c) = ((xh*xs + xw)*K + k)*C + c
        for xh in range(xs):
            for xw in range(xs):
                xpos = b.const_i32(xh * xs + xw)
                kc_layer = b.add(b.mul(xpos, c_K), k_idx)
                ws_off = b.add(b.mul(kc_layer, c_C), c_idx)
                ws_byte = b.mul(ws_off, c4)
                b.buffer_store_f32(fws_rsrc, ws_byte, c0, xformed[xh][xw])

    return b.kernel


# ---------------------------------------------------------------------------
# Kernel 3: Output transform  —  A^T * gemm_result * A
# ---------------------------------------------------------------------------


def build_winograd_output_transform(
    spec: WinogradConvSpec,
    arch: str = "gfx950",
) -> KernelDef:
    """Build the output-transform kernel.

    Each thread covers one (n, tile_h, tile_w, k) combination.  It reads the
    (xform_size × xform_size) accumulated tensor from the GEMM result workspace,
    emits A^T × acc × A in fp32 SSA, converts to fp16, and scatters the
    (out_tile × out_tile) spatial results into the NHWK output tensor.

    Kernel parameters::

        GemmWs       : ptr<f32> global  — GEMM result workspace [xs*xs * N*tiles * K]
        GemmWs_bytes : i32
        D            : ptr<f16> global  — NHWK output  [N, Ho, Wo, K]
        D_bytes      : i32
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid WinogradConvSpec for {arch}: {why}")

    tile = spec.tile
    xs = tile.xform_size
    ot = tile.out_tile
    p = spec.problem
    block_nhw = spec.block_nhw
    block_k = spec.block_k

    b = IRBuilder(spec.kernel_name("output_xform"))
    b.kernel.attrs["max_workgroup_size"] = block_nhw * block_k

    GemmWs = b.param(
        "GemmWs", PtrType(F32, "global"), noalias=True, readonly=True, align=16
    )
    GemmWs_bytes = b.param("GemmWs_bytes", I32)
    D_ptr = b.param("D", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    D_bytes = b.param("D_bytes", I32)

    c0 = b.const_i32(0)
    c2 = b.const_i32(2)
    c4 = b.const_i32(4)
    c_Ho = b.const_i32(p.Ho)
    c_Wo = b.const_i32(p.Wo)
    c_K = b.const_i32(p.K)
    c_tiles_w = b.const_i32(spec.tiles_w)
    c_ot = b.const_i32(ot)
    c_num_tiles = b.const_i32(spec.num_tiles)
    c_ntiles_total = b.const_i32(p.N * spec.num_tiles)

    tid = b.thread_id_x()
    bid_nhw = b.block_id_x()
    bid_k = b.block_id_y()

    c_block_nhw = b.const_i32(block_nhw)
    c_block_k = b.const_i32(block_k)
    local_nhw = b.mod(tid, c_block_nhw)
    local_k = b.div(tid, c_block_nhw)

    nhw_idx = b.add(b.mul(bid_nhw, c_block_nhw), local_nhw)
    k_idx = b.add(b.mul(bid_k, c_block_k), local_k)

    # Decompose nhw_idx -> (n, tile_h, tile_w)
    tile_idx = b.mod(nhw_idx, c_num_tiles)
    n_idx = b.div(nhw_idx, c_num_tiles)
    tile_h_idx = b.div(tile_idx, c_tiles_w)
    tile_w_idx = b.mod(tile_idx, c_tiles_w)

    gws_rsrc = make_buffer_resource(b, GemmWs, num_bytes=GemmWs_bytes).rsrc
    d_rsrc = make_buffer_resource(b, D_ptr, num_bytes=D_bytes).rsrc

    nhw_valid = b.cmp_lt(nhw_idx, c_ntiles_total)
    k_valid = b.cmp_lt(k_idx, c_K)
    both_valid = b.land(nhw_valid, k_valid)

    with b.scf_if(both_valid):
        # Load (xs x xs) tile from GEMM workspace
        # Layout [xs*xs * N*num_tiles * K]: offset(xh, xw, nhw, k)
        #   = ((xh*xs + xw)*ntotal + nhw)*K + k
        acc_tile: list = [[None] * xs for _ in range(xs)]
        for xh in range(xs):
            for xw in range(xs):
                xpos = b.const_i32(xh * xs + xw)
                nhw_layer = b.add(b.mul(xpos, c_ntiles_total), nhw_idx)
                ws_off = b.add(b.mul(nhw_layer, c_K), k_idx)
                ws_byte = b.mul(ws_off, c4)
                acc_tile[xh][xw] = b.buffer_load(gws_rsrc, ws_byte, c0, F32)

        # Apply A^T * acc * A  -> (out_tile x out_tile) result
        out_vals = emit_output_transform(b, tile, acc_tile)

        # Scatter into NHWK output
        # ho = tile_h * out_tile + oy,  wo = tile_w * out_tile + ox
        # Offset: ((n*Ho + ho)*Wo + wo)*K + k, * 2 for f16 bytes
        for oy in range(ot):
            for ox in range(ot):
                ho = b.add(b.mul(tile_h_idx, c_ot), b.const_i32(oy))
                wo = b.add(b.mul(tile_w_idx, c_ot), b.const_i32(ox))
                ho_ok = b.cmp_lt(ho, c_Ho)
                wo_ok = b.cmp_lt(wo, c_Wo)
                in_bounds = b.land(ho_ok, wo_ok)

                row_off = b.add(b.mul(n_idx, c_Ho), ho)
                col_off = b.add(b.mul(row_off, c_Wo), wo)
                elem_off = b.add(b.mul(col_off, c_K), k_idx)
                byte_off = b.mul(elem_off, c2)
                safe = _safe_off(b, in_bounds, byte_off)

                out_f16 = b.trunc_f32_to_f16(out_vals[oy][ox])
                b.buffer_store_f16(d_rsrc, safe, c0, out_f16)

    return b.kernel
