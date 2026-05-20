# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-LDS 2D transpose: in-register sub-tile transpose per wave.

Direct port of CK Tile's ``BatchedTransposePipeline``
(``include/ck_tile/ops/batched_transpose``) "pipeline=0" path that
the reference C++ kernel measures at ~5.5 TB/s on MI355X for a
4096×4096 fp16 transpose. The win comes from a single observation:

    A square sub-tile owned by one lane can be transposed entirely
    in that lane's register file. No LDS staging, no inter-lane
    shuffle, no barrier.

Pipeline (one wave64 per CTA, one ``[TILE_M, TILE_N]`` tile per CTA;
default ``TILE_M = TILE_N = 64`` so each lane owns an 8×8 sub-tile):

    1. Lane ``t = (lr, lc)`` (``lr = t / 8``, ``lc = t % 8``) issues
       8 wide ``global_load_v8_f16`` from
       ``X[block_m + lr*8 + i, block_n + lc*8 .. lc*8 + 7]`` for
       ``i in 0..7``. After the loads each lane holds an 8×8 fp16
       sub-tile spread across 8 ``<8 x half>`` registers.
    2. In-register transpose: build 8 output registers where
       ``out[i] = (rows[0][i], rows[1][i], ..., rows[7][i])``. With
       full unrolling and SSA register naming, the LLVM AMDGPU
       backend lowers this to ``v_perm_b32`` byte-permute chains
       (verified in the disassembly: 32 ``v_perm_b32`` instructions,
       same count as CK Tile's hand-rolled
       ``transpose_vectors<..., bytesize2_2x2_tag>``). No LDS, no
       barrier.
    3. Lane ``t`` issues 8 wide ``global_store_v8_f16`` to
       ``Y[block_n + lc*8 + i, block_m + lr*8 .. lr*8 + 7]`` for
       ``i in 0..7`` -- the transposed sub-tile.

For an ``M × K`` input the grid is ``(M / 64, K / 64, 1)`` -- one
wave64 per output 64×64 tile, plenty for any practical shape.

Measured perf vs CK Tile's ``tile_example_batched_transpose
-pipeline=0`` on MI355X (gfx950, fp16, square shapes, 2000-iter tight
loop, ``no_fence`` launches so per-call overhead doesn't dominate):

==================== ============== ============== ===========
shape                this kernel    CK Tile p=0    ratio
==================== ============== ============== ===========
4096 × 4096          ~5470 GB/s     ~5470 GB/s     1.00×
8192 × 8192          ~5810 GB/s     ~6150 GB/s     0.95×
16384 × 16384        ~4060 GB/s     ~4200 GB/s     0.97×
==================== ============== ============== ===========

Below 4096² a *single* Python-launched kernel is enqueue/replay bound:
the GPU finishes the kernel faster than Python can submit the next
launch, so event timing over a Python loop includes empty stream gaps.
Capturing a graph that contains many transpose launches exposes the
actual GPU kernel time and closes the gap:

==================== ============== ============== ===========
shape                graph-batched  CK Tile p=0    ratio
==================== ============== ============== ===========
1024 × 1024          ~2.0 us        ~2.9 us        1.45×
2048 × 2048          ~3.3 us        ~4.0 us        1.20×
4096 × 4096          ~11.8 us       ~12.1 us       1.03×
==================== ============== ============== ===========

This is the same launch-overhead fix used by
:mod:`ck_dsl.instances.fused_moe_e2e`: capture a steady-state graph and
replay it when tensor pointers are stable.

Why this beats LDS staging
--------------------------

* The CTA is a single wave (64 threads), so there is **no
  workgroup barrier**. Two-phase LDS-staged transposes need at
  least one ``s_barrier`` between the LDS write and the column-
  strided read; this kernel needs none.
* Each thread issues exactly **8 vmem reads + 8 vmem writes**,
  all 16 B vector ops, all dword-aligned. The hardware coalesces
  the 64 threads' 16 B ops into 64 contiguous 16 B beats per
  cache line.
* The in-register transpose is **purely SSA**. No memory traffic,
  no LDS bank-conflict hazard, and the AMDGPU backend has a
  byte-permute pattern matcher (``v_perm_b32``) that fuses the
  ``insertelement`` chain produced by ``vec_pack``.
* The grid scales linearly with problem size, so HBM bandwidth is
  the only knob that matters at scale.

Validation contract:

* ``f16`` / ``bf16``;
* ``M`` and ``K`` must be multiples of ``tile_m`` / ``tile_n``;
* default tile is 64 × 64 (one lane = one 8 × 8 sub-tile, vec = 8).
  Other ``(tile_m, tile_n, vec)`` tuples are accepted as long as
  ``tile_m % vec == 0``, ``tile_n % vec == 0``, and the resulting
  per-lane sub-tile is a square (``tile_m / vec == tile_n / vec``)
  -- the in-register transpose only works for square per-lane
  sub-tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ..core.ir import I32, IRBuilder, KernelDef, PtrType
from ..helpers.io import io_ir_type
from ..helpers.spec import (
    SignatureBuilder,
    kernel_name_join,
)


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class TransposeBcSpec:
    """Configuration for the no-LDS in-register transpose kernel.

    Default tile ``64 × 64`` with ``vec = 8`` puts one wave64 per
    CTA, with each of the 64 lanes owning an 8 × 8 sub-tile and
    transposing it entirely in registers (8 vmem reads + 8 vmem
    writes per lane).
    """

    tile_m: int = 64
    """Output rows processed per CTA."""

    tile_n: int = 64
    """Output columns processed per CTA."""

    vec: int = 8
    """Halves per global vector op (= sub-tile side length).

    Each lane's sub-tile is ``vec × vec`` halves: ``vec`` rows of
    ``vec`` contiguous columns, loaded as ``vec`` separate
    ``global_load_v{vec}_f16`` ops. The in-register transpose
    builds ``vec`` output registers by gathering element ``i`` of
    each input register; the LLVM AMDGPU backend lowers the
    resulting ``insertelement`` chain to ``v_perm_b32``-style
    byte permutes (the same instruction CK Tile's
    ``transpose_vectors`` emits explicitly).

    For ``vec = 8`` (default) each lane handles 64 halves. ``vec
    = 4`` (16 halves per lane) and ``vec = 2`` (4 halves per lane)
    are also supported but waste lane bandwidth on small problems.
    """

    dtype: DType = "f16"
    use_buffer_io: bool = False
    """Use AMDGPU raw buffer-resource loads/stores for fp16.

    This matches CK Tile's ``buffer_load_dwordx4`` / ``buffer_store_dwordx4``
    instruction form and removes flat 64-bit address arithmetic from
    the hot path. It helps rectangular shapes where address arithmetic
    dominates, but currently regresses square bandwidth-saturated
    shapes, so the default stays on flat global ops and callers can
    opt in per shape.
    """
    name: str = "ck_dsl_transpose_bc"

    @property
    def lanes_per_row(self) -> int:
        return self.tile_n // self.vec

    @property
    def lanes_per_col(self) -> int:
        return self.tile_m // self.vec

    @property
    def block_size(self) -> int:
        # Wave64 per CTA: lanes form a (lanes_per_col, lanes_per_row)
        # grid; each lane owns one (vec, vec) sub-tile. The total
        # lane count equals tile_m * tile_n / vec / vec.
        return self.lanes_per_col * self.lanes_per_row

    def kernel_name(self) -> str:
        return kernel_name_join(
            self.name,
            self.dtype,
            f"{self.tile_m}x{self.tile_n}",
            f"v{self.vec}",
            "noLDS",
            "buf" if self.use_buffer_io else "",
        )


def is_valid_spec(spec: TransposeBcSpec) -> Tuple[bool, str]:
    if spec.dtype not in ("f16", "bf16"):
        return False, f"unsupported dtype {spec.dtype!r}"
    if spec.vec not in (2, 4, 8):
        return False, f"vec must be in {{2, 4, 8}} (got {spec.vec})"
    if spec.tile_m % spec.vec or spec.tile_n % spec.vec:
        return False, "tile_m and tile_n must be multiples of vec"
    if spec.tile_m // spec.vec != spec.tile_n // spec.vec:
        return False, (
            "per-lane sub-tile must be square: "
            f"tile_m/vec ({spec.tile_m // spec.vec}) "
            f"!= tile_n/vec ({spec.tile_n // spec.vec})"
        )
    bs = spec.block_size
    if bs > 1024:
        return False, f"block_size {bs} > 1024 hardware cap"
    if bs < 64:
        return False, (
            f"block_size {bs} < 64; use a larger tile (the kernel "
            "is designed for one-wave-per-CTA scheduling)"
        )
    if bs % 64:
        return False, (
            f"block_size {bs} must be a multiple of wave64 "
            f"(got tile_m={spec.tile_m} * tile_n={spec.tile_n} "
            f"/ vec^2 = {bs})"
        )
    return True, "ok"


# ---------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------


def build_transpose_bc(spec: TransposeBcSpec) -> KernelDef:
    """Build the no-LDS in-register transpose kernel.

    Kernel signature: ``(X: ptr, Y: ptr, M: i32, K: i32)``.

    Layout (default ``tile = 64×64``, ``vec = 8``):

    * Grid: ``(K / 64, M / 64, 1)``. One CTA per output 64×64 tile.
    * Block: ``(64, 1, 1)`` -- a single wave64.
    * Each lane owns an 8×8 sub-tile of the input tile and writes
      its transposed sub-tile to the output. No LDS, no barrier.
    """
    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid transpose_bc spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    TM, TN, VEC, BS = spec.tile_m, spec.tile_n, spec.vec, spec.block_size
    # Per-lane sub-tile: (VEC × VEC) halves. ``LANES_N`` is the
    # number of lanes laid out along the N axis (a square wave64
    # requires ``LANES_M == LANES_N`` which the validator enforces).
    LANES_N = spec.lanes_per_row  # TN // VEC

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)
    K = b.param("K", I32)

    use_buffer_io = spec.use_buffer_io and spec.dtype == "f16"
    if use_buffer_io:
        # Match CK Tile's no-LDS pipeline: raw_ptr_buffer_load/store
        # with 32-bit byte offsets and one SGPR buffer resource per
        # operand. This avoids per-lane 64-bit flat-address arithmetic
        # in the hot path and is especially important for rectangular
        # shapes where address math can otherwise dominate the
        # 16-byte vmem payload.
        num_bytes = b.mul(b.mul(M, K), b.const_i32(2))
        x_rsrc = b.buffer_rsrc(X, num_bytes)
        y_rsrc = b.buffer_rsrc(Y, num_bytes)
        zero_soffset = b.const_i32(0)
    else:
        x_rsrc = None
        y_rsrc = None
        zero_soffset = None

    tid = b.thread_id_x()
    c_lanes_n = b.const_i32(LANES_N)
    c_vec = b.const_i32(VEC)

    # Lane (lr, lc) inside the wave: lr = tid / LANES_N, lc = tid % LANES_N.
    # Lane (lr, lc) owns input X[block_m + lr*VEC + i, block_n + lc*VEC + j]
    # for i, j in 0..VEC-1, and writes the transposed tile
    # Y[block_n + lc*VEC + i, block_m + lr*VEC + j].
    lr = b.div(tid, c_lanes_n)
    lc = b.mod(tid, c_lanes_n)

    # Grid axis order: ``block_id_x`` -> N (columns), ``block_id_y`` -> M
    # (rows). This matches CK Tile's ``BatchedTransposeKernel::GridSize``
    # (``grid_size_x = ceil_div(height = K, dim_block_h)``,
    # ``grid_size_y = ceil_div(width = M, dim_block_w)``) and gives
    # consecutive CTAs along ``block_id_x`` consecutive K-tile columns
    # of the same M-row group -- which lets L2 reuse the source rows
    # of ``X`` between adjacent CTAs. Transposing the axes (X -> M,
    # Y -> N) costs ~5x throughput on tall-thin shapes like
    # ``(M=1024, K=16384)`` because consecutive CTAs would land on
    # rows 32 KiB apart, defeating L2 reuse on the source.
    block_n = b.mul(b.block_id_x(), b.const_i32(TN))
    block_m = b.mul(b.block_id_y(), b.const_i32(TM))

    my_m = b.add(block_m, b.mul(lr, c_vec))  # base row of this lane's sub-tile
    my_n = b.add(block_n, b.mul(lc, c_vec))  # base col of this lane's sub-tile

    # ----- Phase 1: gather VEC rows of VEC contiguous halves each.
    # ``rows[i]`` is a ``<VEC x half>`` SSA register holding
    # X[my_m + i, my_n .. my_n + VEC - 1]. Issued as VEC independent
    # ``global_load_v{VEC}_f16`` ops; the GCN scheduler issues them
    # out-of-order so the per-lane vmem latency overlaps freely.
    rows = []
    for i in range(VEC):
        row_idx = b.add(my_m, b.const_i32(i)) if i > 0 else my_m
        # X is row-major with stride K halves per row.
        off = b.add(b.mul(row_idx, K), my_n)
        if use_buffer_io:
            byte_off = b.mul(off, b.const_i32(2))
            rows.append(
                b.buffer_load_vN_f16(x_rsrc, byte_off, zero_soffset, dwords=VEC // 2)
            )
        else:
            rows.append(b.global_load_vN(X, off, io_ty, n=VEC))

    # ----- Phase 2: in-register transpose.
    # ``out_rows[i]`` should be ``<VEC x half>`` holding
    # Y[my_n + i, my_m .. my_m + VEC - 1] = (X[my_m + j, my_n + i])_j.
    # In our ``rows`` notation that is
    # ``out_rows[i] = (rows[0][i], rows[1][i], ..., rows[VEC-1][i])``.
    # The ``insertelement`` chain that ``vec_pack`` emits is the
    # canonical pattern the AMDGPU backend matches into
    # ``v_perm_b32`` byte permutes (the explicit
    # ``__builtin_amdgcn_perm`` calls in CK Tile's
    # ``transpose_vectors_apply_impl(..., bytesize2_2x2_tag)``).
    # No LDS, no cross-lane shuffles -- everything is one lane's
    # SSA register file.
    out_rows = []
    for i in range(VEC):
        elems = [b.vec_extract(rows[j], i) for j in range(VEC)]
        out_rows.append(b.vec_pack(elems, io_ty))

    # ----- Phase 3: scatter VEC rows of VEC contiguous halves each.
    # Y is ``[K, M]`` row-major with stride M halves per row.
    for i in range(VEC):
        out_row_idx = b.add(my_n, b.const_i32(i)) if i > 0 else my_n
        off = b.add(b.mul(out_row_idx, M), my_m)
        if use_buffer_io:
            byte_off = b.mul(off, b.const_i32(2))
            b.buffer_store_vN_f16(
                y_rsrc, byte_off, zero_soffset, out_rows[i], dwords=VEC // 2
            )
        else:
            b.global_store_vN(Y, off, out_rows[i], VEC)

    return b.kernel


# ---------------------------------------------------------------------
# Launch helpers
# ---------------------------------------------------------------------


def transpose_bc_grid(m: int, k: int, spec: TransposeBcSpec) -> Tuple[int, int, int]:
    """Launch grid for an ``[m, k]`` input.

    One CTA (= one wave64) per ``[tile_m, tile_n]`` output tile.
    Grid X is the K-tile axis and Y is the M-tile axis so consecutive
    CTAs scan along K within a fixed M slab -- matches CK Tile's
    ``BatchedTransposeKernel::GridSize`` and keeps the L2 hot on the
    source rows of ``X``.
    """
    return (
        (k + spec.tile_n - 1) // spec.tile_n,
        (m + spec.tile_m - 1) // spec.tile_m,
        1,
    )


def transpose_bc_signature(spec: TransposeBcSpec):
    return (
        SignatureBuilder()
        .ptr("X", spec.dtype)
        .ptr("Y", spec.dtype)
        .scalar("M", "i32")
        .scalar("K", "i32")
        .build()
    )


__all__ = [
    "TransposeBcSpec",
    "build_transpose_bc",
    "is_valid_spec",
    "transpose_bc_grid",
    "transpose_bc_signature",
]
