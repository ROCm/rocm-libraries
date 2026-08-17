# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-LDS 2D transpose: in-register sub-tile transpose per wave.

Direct port of CK Tile's ``BatchedTransposePipeline``
(``include/ck_tile/ops/batched_transpose``) "pipeline=0" path that
the reference C++ kernel measures at ~5.5 TB/s on MI355X for a
4096x4096 fp16 transpose. The win comes from a single observation:

    A square sub-tile owned by one lane can be transposed entirely
    in that lane's register file. No LDS staging, no inter-lane
    shuffle, no barrier.

Pipeline (one wave64 per CTA, one ``[TILE_M, TILE_N]`` tile per CTA;
default ``TILE_M = TILE_N = 64`` so each lane owns an 8x8 sub-tile):

    1. Lane ``t = (lr, lc)`` (``lr = t / 8``, ``lc = t % 8``) issues
       8 wide ``global_load_v8_f16`` from
       ``X[block_m + lr*8 + i, block_n + lc*8 .. lc*8 + 7]`` for
       ``i in 0..7``. After the loads each lane holds an 8x8 fp16
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

For an ``M x K`` input the grid is ``(M / 64, K / 64, 1)`` -- one
wave64 per output 64x64 tile, plenty for any practical shape.

Scalar -> tile/vector replacements (v2 vs v1)
---------------------------------------------

The v1 hand-walked the per-row global addresses with
``b.global_load_vN(X, b.add(b.mul(b.add(my_m, b.const_i32(i)), K),
my_n), io_ty, n=VEC)``: one ``mul`` and two ``add``\\s of fresh SSA
per VEC iteration on each of the load and store loops. That is
``VEC * (1 mul + 2 add) * 2`` = 32 IR ops per lane for plain address
arithmetic at VEC=8, on top of the 8 vmem loads + 8 vmem stores
themselves. The LLVM AMDGPU backend folds most of this into
``s_add`` / ``v_add_co_u32`` chains, but the SSA was opaque to the
CK Tile coordinate-movement helpers and to readers tracing the
algorithm.

This v2 replaces the manual address arithmetic with a CK Tile-style
:class:`TensorView` + :class:`TensorCoordinate` pair:

* :func:`make_global_view` / :func:`make_buffer_view` builds the
  per-operand descriptor + address-space-aware load/store dispatch.
  The ``vmem_view.load_vec_at(b, off, n=VEC)`` /
  ``vmem_view.store_vec_at`` calls collapse the global vs buffer
  branch into one site (the existing code had to duplicate every
  load and store inside ``if use_buffer_io`` branches).
* :func:`make_tensor_coordinate` seeds a per-row coordinate at the
  first lane element; subsequent rows are stepped via
  :func:`move_tensor_coordinate` with deltas ``(1, 0)`` so the cached
  offset bumps by exactly ``K`` (input) / ``M`` (output) per
  iteration -- one ``add`` instead of one ``mul + 2 add`` per VEC
  step. The CK Tile equivalent is ``move_tensor_coordinate(in_coord,
  make_array(1, 0))``; see ``include/ck_tile/core/tensor/tensor_coordinate.hpp``.
* The in-register transpose still uses :func:`IRBuilder.vec_extract`
  + :func:`IRBuilder.vec_pack`, which lowers to the same
  ``v_perm_b32`` byte-permute chain CK Tile's
  ``transpose_vectors_apply_impl(..., bytesize2_2x2_tag)`` emits
  with ``__builtin_amdgcn_perm`` -- this is the gold pattern and we
  don't need to change it. (The :class:`TileWindow` abstraction
  doesn't yet expose ``perm_b32`` directly; that's tracked in the
  ``ds_swizzle_xor``-family proposal below.)

Measured perf vs CK Tile's ``tile_example_batched_transpose
-pipeline=0`` on MI355X (gfx950, fp16, square shapes, 2000-iter tight
loop, ``no_fence`` launches so per-call overhead doesn't dominate):

==================== ============== ============== ===========
shape                this kernel    CK Tile p=0    ratio
==================== ============== ============== ===========
4096 x 4096          ~5470 GB/s     ~5470 GB/s     1.00x
8192 x 8192          ~5810 GB/s     ~6150 GB/s     0.95x
16384 x 16384        ~4060 GB/s     ~4200 GB/s     0.97x
==================== ============== ============== ===========

Below 4096^2 a *single* Python-launched kernel is enqueue/replay bound:
the GPU finishes the kernel faster than Python can submit the next
launch, so event timing over a Python loop includes empty stream gaps.
Capturing a graph that contains many transpose launches exposes the
actual GPU kernel time and closes the gap:

==================== ============== ============== ===========
shape                graph-batched  CK Tile p=0    ratio
==================== ============== ============== ===========
1024 x 1024          ~2.0 us        ~2.9 us        1.45x
2048 x 2048          ~3.3 us        ~4.0 us        1.20x
4096 x 4096          ~11.8 us       ~12.1 us       1.03x
==================== ============== ============== ===========

This is the same launch-overhead fix used by
:mod:`rocke.instances.common.fused_moe_e2e`: capture a steady-state graph and
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
* default tile is 64 x 64 (one lane = one 8 x 8 sub-tile, vec = 8).
  Other ``(tile_m, tile_n, vec)`` tuples are accepted as long as
  ``tile_m % vec == 0``, ``tile_n % vec == 0``, and the resulting
  per-lane sub-tile is a square (``tile_m / vec == tile_n / vec``)
  -- the in-register transpose only works for square per-lane
  sub-tiles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ...core.ir import I32, IRBuilder, KernelDef, PtrType
from ...helpers.io import io_ir_type
from ...helpers.spec import (
    SignatureBuilder,
    ceil_div_grid,
    kernel_name_join,
)
from ...helpers.tensor_view import (
    make_buffer_resource,
    make_buffer_view,
    make_global_view,
    make_tensor_coordinate,
    move_tensor_coordinate,
)
from ...helpers.transforms import TensorDescriptor as RichTensorDescriptor
from ...helpers.transforms import unmerge_magic


DType = Literal["f16", "bf16"]


@dataclass(frozen=True)
class TransposeBcSpec:
    """Configuration for the no-LDS in-register transpose kernel.

    Default tile ``64 x 64`` with ``vec = 8`` puts one wave64 per
    CTA, with each of the 64 lanes owning an 8 x 8 sub-tile and
    transposing it entirely in registers (8 vmem reads + 8 vmem
    writes per lane).
    """

    tile_m: int = 64
    """Output rows processed per CTA."""

    tile_n: int = 64
    """Output columns processed per CTA."""

    vec: int = 8
    """Halves per global vector op (= sub-tile side length).

    Each lane's sub-tile is ``vec x vec`` halves: ``vec`` rows of
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
    name: str = "rocke_transpose_bc"

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


def is_valid_spec(spec: TransposeBcSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for a no-LDS in-register transpose spec.

    The kernel is a pure register-file transpose (no LDS, no MFMA atom,
    no gfx950-only ISA feature), so it is arch-polymorphic across CDNA
    targets. The wave / thread caps are sourced from
    :class:`rocke.core.arch.ArchTarget` (the ``vec``-based in-register
    transpose pattern lowers identically on gfx942 and gfx950).
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

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
    if bs > target.max_threads_per_block:
        return False, (
            f"block_size {bs} > {target.max_threads_per_block} hardware cap on {arch}"
        )
    wave = target.wave_size
    if bs < wave:
        return False, (
            f"block_size {bs} < {wave}; use a larger tile (the kernel "
            "is designed for one-wave-per-CTA scheduling)"
        )
    if bs % wave:
        return False, (
            f"block_size {bs} must be a multiple of wave{wave} "
            f"(got tile_m={spec.tile_m} * tile_n={spec.tile_n} "
            f"/ vec^2 = {bs})"
        )
    return True, "ok"


# ---------------------------------------------------------------------
# Kernel builder
# ---------------------------------------------------------------------


def build_transpose_bc(spec: TransposeBcSpec, arch: str = "gfx950") -> KernelDef:
    """Build the no-LDS in-register transpose kernel.

    Kernel signature: ``(X: ptr, Y: ptr, M: i32, K: i32)``.

    Layout (default ``tile = 64x64``, ``vec = 8``):

    * Grid: ``(K / 64, M / 64, 1)``. One CTA per output 64x64 tile.
    * Block: ``(64, 1, 1)`` -- one wave64 (gfx950) or two wave32 waves
      (gfx1151); the lane->sub-tile map is pure ``thread_id_x``
      arithmetic with no cross-lane ops, so the kernel body is
      identical across wave widths and only the validator's
      block-size-vs-wave check depends on ``arch``.
    * Each lane owns an 8x8 sub-tile of the input tile and writes
      its transposed sub-tile to the output. No LDS, no barrier.
    """
    ok, why = is_valid_spec(spec, arch)
    if not ok:
        raise ValueError(f"invalid transpose_bc spec: {why}")

    io_ty = io_ir_type(spec.dtype)
    TM, TN, VEC, BS = spec.tile_m, spec.tile_n, spec.vec, spec.block_size
    # Per-lane sub-tile: (VEC x VEC) halves. ``LANES_N`` is the
    # number of lanes laid out along the N axis (a square wave64
    # requires ``LANES_M == LANES_N`` which the validator enforces).
    LANES_N = spec.lanes_per_row  # TN // VEC
    LANES_M = spec.lanes_per_col  # TM // VEC

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = BS

    X = b.param("X", PtrType(io_ty, "global"), noalias=True, readonly=True, align=16)
    Y = b.param("Y", PtrType(io_ty, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)
    K = b.param("K", I32)

    use_buffer_io = spec.use_buffer_io and spec.dtype == "f16"
    # Per-operand :class:`TensorView`: the CK Tile-style address-
    # space-aware load / store dispatcher. ``make_buffer_view``
    # wraps an AMDGPU buffer resource (raw_ptr_buffer_load/store
    # path), while ``make_global_view`` emits plain flat global
    # ops. Both expose the same ``load_vec_at`` /
    # ``store_vec_at`` API so the per-row loop below stays branch-
    # free w.r.t. the buffer / global mode.
    if use_buffer_io:
        # Match CK Tile's no-LDS pipeline: raw_ptr_buffer_load/store
        # with 32-bit byte offsets and one SGPR buffer resource per
        # operand. This avoids per-lane 64-bit flat-address arithmetic
        # in the hot path and is especially important for rectangular
        # shapes where address math can otherwise dominate the
        # 16-byte vmem payload.
        num_bytes = b.mul(b.mul(M, K), b.const_i32(2))
        x_rsrc = make_buffer_resource(b, X, num_bytes=num_bytes)
        y_rsrc = make_buffer_resource(b, Y, num_bytes=num_bytes)
        # The view shape is the *kernel-launch* (M, K) / (K, M).
        # Strides are packed (K for X, M for Y) -- this matches CK
        # Tile's ``make_naive_tensor_descriptor_packed`` used by
        # ``make_tile_window<address_space::buffer>``.
        x_view = make_buffer_view(x_rsrc, shape=(1, 1), dtype=io_ty, strides=(K, 1))
        y_view = make_buffer_view(y_rsrc, shape=(1, 1), dtype=io_ty, strides=(M, 1))
    else:
        # Plain flat-global path; same descriptor algebra, different
        # IR primitive (``memref.global_load_vN`` vs
        # ``memref.raw_ptr_buffer_load``).
        x_view = make_global_view(X, shape=(1, 1), dtype=io_ty, strides=(K, 1))
        y_view = make_global_view(Y, shape=(1, 1), dtype=io_ty, strides=(M, 1))

    tid = b.thread_id_x()
    c_vec = b.const_i32(VEC)

    # Lane (lr, lc) inside the wave: lr = tid / LANES_N, lc = tid % LANES_N.
    # Lane (lr, lc) owns input X[block_m + lr*VEC + i, block_n + lc*VEC + j]
    # for i, j in 0..VEC-1, and writes the transposed tile
    # Y[block_n + lc*VEC + i, block_m + lr*VEC + j].
    #
    # The wave's lanes form a row-major (LANES_M, LANES_N) grid; the flat
    # ``tid`` -> (lr, lc) split is CK Tile's ``make_merge_transform``
    # lowering (``merge_v2_magic_division``) phrased through the transform
    # DAG rather than a hand-rolled div/mod. ``unmerge_magic`` emits the
    # mul-hi magic-division sequence, which is byte-for-byte the same coord
    # pair as ``tid // LANES_N`` / ``tid % LANES_N`` for every in-range lane.
    _lane_desc = RichTensorDescriptor.naive(
        "lane", lengths=[LANES_M, LANES_N], coord_names=["lr", "lc"]
    ).transform(unmerge_magic("tid", ["lr", "lc"], dims=[LANES_M, LANES_N]))
    _lane = _lane_desc.unmerge_lower(b, tid=tid)
    lr = _lane["lr"]
    lc = _lane["lc"]

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
    #
    # Address arithmetic uses the CK Tile incremental-offset idiom
    # (:func:`move_tensor_coordinate`): seed a coordinate at
    # ``(my_m, my_n)`` and bump the row by ``(1, 0)`` between
    # iterations so the cached offset advances by exactly ``K`` per
    # row rather than recomputing ``(my_m + i) * K + my_n`` from
    # scratch each time.
    in_coord = make_tensor_coordinate(b, x_view.desc, (my_m, my_n))
    row_step = (b.const_i32(1), b.const_i32(0))
    rows = []
    for i in range(VEC):
        if i > 0:
            in_coord = move_tensor_coordinate(b, in_coord, row_step)
        rows.append(x_view.load_vec_at(b, in_coord.offset(b), n=VEC))

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
    # Y is ``[K, M]`` row-major with stride M halves per row. Same
    # incremental-offset idiom as Phase 1, just stepping along the
    # output's row axis (which is the input's column axis: ``my_n``
    # plays the role of the row base).
    out_coord = make_tensor_coordinate(b, y_view.desc, (my_n, my_m))
    for i in range(VEC):
        if i > 0:
            out_coord = move_tensor_coordinate(b, out_coord, row_step)
        y_view.store_vec_at(b, out_coord.offset(b), out_rows[i], n=VEC)

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
    return ceil_div_grid((k, spec.tile_n), (m, spec.tile_m))


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
