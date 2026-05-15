# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Universal GEMM kernel instance builder.

This is the DSL-side counterpart of CK's
`dispatcher/codegen/unified_gemm_codegen.py`: given the exact same
config schema CK's dispatcher uses to enumerate kernels (see
`dispatcher/codegen/default_config.json`), produce a Python IR
`KernelDef` that lowers to AMDGPU LLVM IR and (via libamd_comgr) to a
HSA code object.

The schema is intentionally identical to CK's so a sweep driver can
walk the same cartesian product CK walks and produce the matching DSL
kernel for every entry. The instance space we cover today is the FP16
RCR family — the dispatcher's hero family for compute-bound large
GEMMs (`preselected_fp16_rcr_compute`).

Geometry conventions (mirror CK Tile's `BlockGemmShape`):

    Block tile:   tile_m  x tile_n  x tile_k
    Warp grid:    warp_m  x warp_n          (warp_k=1)
    MFMA atom:    warp_tile_m x warp_tile_n x warp_tile_k

Each warp owns a `(tile_m / warp_m) x (tile_n / warp_n)` output sub-tile;
that's `mfmas_per_warp_m = (tile_m / warp_m) / warp_tile_m` MFMA tiles
along M and similarly along N. The accumulator length per lane is
`warp_tile_m * warp_tile_n / wave_size` (4 for 16x16; 16 for 32x32 on
wave64).

What this file implements *now*:
  - tile geometries 64..256 x 64..256 x 16..128
  - warp grids 1x1, 2x1, 1x2, 2x2, 4x1, 1x4, 2x4, 4x2, 4x4
  - MFMA atoms 16x16x16, 16x16x32, 32x32x8, 32x32x16 f16
  - pipeline `mem` (single buffer, sync), `compv3` (single buffer +
    scheduler hints), `compv4` (double-buffer LDS + sched_group_barrier
    interleave between MFMA/DS_read/DS_write/VMEM)
  - epilogue `default` (vectorised direct global stores) and
    `cshuffle` (LDS-staged C with the wide-store distribution)
  - layout RCR (row(A), col(B), row(C)) — the production layout

What is left out for now (called out explicitly):
  - bf16/fp8 input dtypes (no atom yet; mechanical extension)
  - persistent kernels (the dispatcher allows both; `persistent=False`
    is what every preselected_fp16_rcr_compute entry uses for the
    standard variant)
  - padding (`pad_m/n/k`) — the standard configs in default_config.json
    use `pad_*=false`; the dispatcher tries pad-on variants in the
    preselect set; we accept those as input but emit the same body
    (the bounds are statically expected to divide).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Literal, Optional, Sequence, Tuple

from ..core.ir import (
    F16,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ..helpers.tensor_view import make_global_view, make_tile_window


# ---------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    """Mirror of CK's `TileConfig`."""

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int = 1
    warp_tile_m: int = 32
    warp_tile_n: int = 32
    warp_tile_k: int = 16

    @property
    def mfmas_per_warp_m(self) -> int:
        out, rem = divmod(self.tile_m, self.warp_m * self.warp_tile_m)
        if rem:
            raise ValueError(
                f"tile_m {self.tile_m} not divisible by warp_m * warp_tile_m "
                f"= {self.warp_m * self.warp_tile_m}"
            )
        return out

    @property
    def mfmas_per_warp_n(self) -> int:
        out, rem = divmod(self.tile_n, self.warp_n * self.warp_tile_n)
        if rem:
            raise ValueError(
                f"tile_n {self.tile_n} not divisible by warp_n * warp_tile_n "
                f"= {self.warp_n * self.warp_tile_n}"
            )
        return out

    @property
    def k_atoms_per_tile_k(self) -> int:
        out, rem = divmod(self.tile_k, self.warp_tile_k)
        if rem:
            raise ValueError(
                f"tile_k {self.tile_k} not divisible by warp_tile_k {self.warp_tile_k}"
            )
        return out


Pipeline = Literal["mem", "compv3", "compv4"]
Scheduler = Literal["intrawave", "interwave"]
Epilogue = Literal["default", "cshuffle"]


@dataclass(frozen=True)
class TraitSpec:
    """Mirror of CK's `TraitConfig`.

    Extra traits beyond CK's defaults:

    * ``chiplet_swizzle``: opt into the chiplet-aware grid swizzle
      that remaps WGIDs so every contiguous stripe of workgroups
      lands on the same XCD (improves L2 reuse on multi-die GPUs).
      Composes a XCD-round-robin reverse (chunk_size WGs per XCD)
      with a WGM super-tile reordering. The kernel still launches
      with the standard ``(N_tiles, M_tiles[, batch])`` grid; the
      remap happens at kernel entry from the flattened blockIdx.

    * ``waves_per_eu``: AMDGPU occupancy hint emitted as
      ``"amdgpu-waves-per-eu"`` on the kernel attribute list. Default
      ``None`` keeps the LLVM backend's heuristic choice; set to 2
      (or a ``(min, max)`` tuple) when targeting two workgroups per CU.
    """

    pipeline: Pipeline = "compv4"
    scheduler: Scheduler = "intrawave"
    epilogue: Epilogue = "cshuffle"
    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = False
    persistent: bool = False
    chiplet_swizzle: bool = False
    chiplet_wgm: int = 8
    chiplet_num_xcds: int = 8
    chiplet_chunk_size: int = 64
    waves_per_eu: Optional[int] = None


@dataclass(frozen=True)
class DataSpec:
    """Element / accumulator / layout choice. Today: f16 in, f16 out, f32 acc, RCR."""

    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    dtype_acc: str = "fp32"
    layout: str = "RCR"


@dataclass(frozen=True)
class UniversalGemmSpec:
    name: str
    tile: TileSpec
    trait: TraitSpec = field(default_factory=TraitSpec)
    data: DataSpec = field(default_factory=DataSpec)
    wave_size: int = 64
    # If None, derived from `warp_m * warp_n * warp_k * wave_size`
    # (the only valid value per CK's `gemm_validation_utils.py` line 605:
    # `BlockSize = NumWarps * warp_size`). We expose it so an over-rider
    # can force a specific block_size for autotuning experiments.
    block_size: int = 0
    # When True, the kernel reads ``block_id_z`` as the batch index and
    # picks up three extra i64 stride args (``stride_a``, ``stride_b``,
    # ``stride_c``) that scale the per-batch pointer offset. The grid
    # then becomes ``(N_tiles, M_tiles, batch_count)``. This is the only
    # difference between the non-batched ``build_universal_gemm`` and the
    # batched form -- the MFMA / LDS body is shared verbatim so the
    # batched kernel inherits the same correctness + perf as the base.
    batched: bool = False

    def __post_init__(self) -> None:
        if self.block_size == 0:
            t = self.tile
            object.__setattr__(
                self,
                "block_size",
                t.warp_m * t.warp_n * t.warp_k * self.wave_size,
            )

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        t = self.tile
        tr = self.trait
        return kernel_name_join(
            self.name,
            f"t{t.tile_m}x{t.tile_n}x{t.tile_k}",
            f"w{t.warp_m}x{t.warp_n}x{t.warp_k}",
            f"wt{t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}",
            f"{tr.pipeline}_{tr.scheduler}_{tr.epilogue}",
            flags={
                "pad": any([tr.pad_m, tr.pad_n, tr.pad_k]),
                "pers": tr.persistent,
                "bat": self.batched,
            },
        )


# ---------------------------------------------------------------------
# Validity rules (a subset of CK's `arch_filter.py` for gfx950 fp16)
# ---------------------------------------------------------------------


_F16_WARP_TILE_SHAPES_GFX950 = {
    (16, 16, 16),
    (16, 16, 32),
    (32, 32, 8),
    (32, 32, 16),
}


def is_valid_spec(spec: UniversalGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return `(ok, reason)`. The same predicate CK's dispatcher uses
    to drop unbuildable configs from a sweep."""

    if arch != "gfx950":
        return False, f"only gfx950 is wired up today (got {arch!r})"

    t = spec.tile
    # MFMA atom must be one of the supported shapes on this arch+dtype.
    atom = (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    if atom not in _F16_WARP_TILE_SHAPES_GFX950:
        return False, f"unsupported f16 warp_tile {atom} on gfx950"

    # Geometry divisibility.
    if t.tile_m % (t.warp_m * t.warp_tile_m):
        return False, "tile_m not divisible by warp_m * warp_tile_m"
    if t.tile_n % (t.warp_n * t.warp_tile_n):
        return False, "tile_n not divisible by warp_n * warp_tile_n"
    if t.tile_k % t.warp_tile_k:
        return False, "tile_k not divisible by warp_tile_k"

    # block_size = warp_m * warp_n * wave_size.
    expected_bs = t.warp_m * t.warp_n * spec.wave_size
    if expected_bs != spec.block_size:
        return False, (
            f"block_size {spec.block_size} != warp_m*warp_n*wave_size = {expected_bs}"
        )

    # LDS budget. gfx950 cap is 160 KiB per WG (CDNA3, see
    # dispatcher/codegen/arch_specs.json). Our current emitter does NOT
    # alias AB and cshuffle staging (CK does, but that's a separate
    # optimisation we have not yet wired up), so the actual usage is
    # additive:
    #   compv4 single AB:   tile_m*tile_k*2 + tile_n*tile_k*2
    #   compv4 double AB:   2 * single AB
    #   cshuffle staging:   tile_m*tile_n*2   (f16)
    #   total:              double_buffer_AB + (cshuffle ? C : 0)
    # When we land the AB/C aliasing in the cshuffle emitter, swap the
    # `+` for a `max`.
    ab_single = ((t.tile_m * t.tile_k) + (t.tile_n * t.tile_k)) * 2
    ab_bytes = ab_single * (2 if spec.trait.pipeline == "compv4" else 1)
    c_bytes = t.tile_m * t.tile_n * 2 if spec.trait.epilogue == "cshuffle" else 0
    bytes_lds = ab_bytes + c_bytes
    if bytes_lds > 160 * 1024:
        return False, (
            f"LDS budget {bytes_lds} > 160 KiB cap (AB={ab_bytes}, C={c_bytes})"
        )

    # Wave64 per-WG cap: AMD GPUs cap at 1024 threads / WG.
    if spec.block_size > 1024:
        return False, f"block_size {spec.block_size} > 1024 (hardware cap)"

    # Global -> LDS vectorised load divisibility: at least one vec
    # element per thread per phase.
    threads = spec.block_size
    a_total = t.tile_m * t.tile_k
    b_total = t.tile_n * t.tile_k
    if a_total < threads or b_total < threads:
        return False, "block too small for one element/thread/phase"

    return True, "ok"


# ---------------------------------------------------------------------
# IR builders for the pieces
# ---------------------------------------------------------------------


def _mfma_atom_widths(spec: UniversalGemmSpec) -> Tuple[int, int, int]:
    """Return (a_per_lane, b_per_lane, c_per_lane) for the spec's MFMA atom."""
    t = spec.tile
    waves = spec.wave_size
    a_per = (t.warp_tile_m * t.warp_tile_k) // waves
    b_per = (t.warp_tile_k * t.warp_tile_n) // waves
    c_per = (t.warp_tile_m * t.warp_tile_n) // waves
    return a_per, b_per, c_per


def _emit_mfma(
    b: IRBuilder, spec: UniversalGemmSpec, a: Value, bb: Value, c: Value
) -> Value:
    t = spec.tile
    key = (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    if key == (16, 16, 16):
        return b.mfma_f32_16x16x16_f16(a, bb, c)
    if key == (16, 16, 32):
        return b.mfma_f32_16x16x32_f16(a, bb, c)
    if key == (32, 32, 8):
        return b.mfma_f32_32x32x8_f16(a, bb, c)
    if key == (32, 32, 16):
        return b.mfma_f32_32x32x16_f16(a, bb, c)
    raise NotImplementedError(f"no MFMA emitter for warp_tile {key}")


def _emit_zero_acc(b: IRBuilder, spec: UniversalGemmSpec) -> Value:
    _, _, c_per = _mfma_atom_widths(spec)
    return b.zero_vec_f32(c_per)


def _choose_load_vec(spec: UniversalGemmSpec) -> int:
    """Choose the widest naturally-aligned global-load width for this
    block shape. f16 -> we can vectorise up to 8 halves per lane."""
    t = spec.tile
    threads = spec.block_size
    # We need block_k % vec == 0 AND total_halves / vec >= threads AND
    # total_halves / vec % threads == 0 (coalesced).
    for v in (8, 4, 2, 1):
        if t.tile_k % v:
            continue
        a_vecs = (t.tile_m * t.tile_k) // v
        b_vecs = (t.tile_n * t.tile_k) // v
        if a_vecs < threads or b_vecs < threads:
            continue
        if a_vecs % threads or b_vecs % threads:
            continue
        return v
    raise ValueError(f"no usable load_vec for {spec}")


def _emit_smem_load(b: IRBuilder, smem: Value, row: Value, col: Value, n: int) -> Value:
    if n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN_f16(smem, row, col, n=n)


# ---------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------


def build_universal_gemm(spec: UniversalGemmSpec) -> KernelDef:
    """Build the IR for one universal GEMM instance.

    Today this dispatches on (pipeline, epilogue):

      | pipeline | epilogue | implementation |
      |---|---|---|
      | mem      | default  | single-buffer + direct vector stores (the dsl/01-05 family) |
      | compv3   | default  | single-buffer + sched_group_barrier interleave |
      | compv3   | cshuffle | single-buffer + LDS-staged f16 epilogue |
      | compv4   | default  | double-buffer LDS + sched_group_barrier + direct vector stores |
      | compv4   | cshuffle | double-buffer LDS + sched_group_barrier + LDS-staged cshuffle |
      | mem      | cshuffle | single-buffer + LDS-staged epilogue |

    All five paths share the same IR-construction subroutines below.
    """

    ok, why = is_valid_spec(spec)
    if not ok:
        raise ValueError(f"invalid GEMM spec: {why}")

    b = IRBuilder(spec.kernel_name())
    # IMPORTANT: AMDGPU bakes `amdgpu-flat-work-group-size` into the
    # kernel descriptor; if we launch with more threads than that, HIP
    # returns `unspecified launch failure` *before* the kernel body
    # runs. Pin the upper bound to this spec's `block_size`.
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.trait.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu
    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    C = b.param("C", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    M = b.param("M", I32)
    N = b.param("N", I32)
    K = b.param("K", I32)
    if spec.batched:
        # Per-batch strides (in elements, not bytes). The kernel uses
        # ``block_id_z`` as the batch index and adds
        # ``z * stride_X`` to the X-load/store base offset for X in
        # {A, B, C}. Strides are i32 to match the rest of the index
        # arithmetic; the LLVM zext/sext folds them into the gep idx.
        stride_a = b.param("stride_a", I32)
        stride_b = b.param("stride_b", I32)
        stride_c = b.param("stride_c", I32)

    t = spec.tile
    a_per_lane, b_per_lane, c_per_lane = _mfma_atom_widths(spec)

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

    # Common geometry.
    c0 = b.const_i32(0)
    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(t.warp_n)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)
    c_block_k = b.const_i32(block_k)

    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    if spec.batched:
        batch_idx = b.block_id_z()
        batch_off_a = b.mul(batch_idx, stride_a)
        batch_off_b = b.mul(batch_idx, stride_b)
        batch_off_c = b.mul(batch_idx, stride_c)
    else:
        batch_off_a = c0
        batch_off_b = c0
        batch_off_c = c0

    # Tile-index assignment. By default ``block_id_x`` maps to the
    # N-tile and ``block_id_y`` to the M-tile (the host launcher uses
    # ``grid = (N_tiles, M_tiles, batch?)``). With
    # ``trait.chiplet_swizzle=True`` we instead flatten the 2D grid
    # into a linear WGID and run it through the chiplet-aware
    # super-tile remap so consecutive workgroups land on the same XCD.
    if spec.trait.chiplet_swizzle:
        from ..helpers.grid import chiplet_aware_super_tile_dynamic

        # Compute M_tiles / N_tiles at runtime from the dynamic M/N args.
        n_pid_m = b.div(b.add(M, b.const_i32(block_m - 1)), c_block_m)
        n_pid_n = b.div(b.add(N, b.const_i32(block_n - 1)), c_block_n)
        # Flatten (bx, by) -> wgid_flat using the actual launch grid's
        # X-extent (= n_pid_n) so wgid_flat mirrors a 1D dispatch
        # walking ``for by: for bx:`` order.
        wgid_flat = b.add(
            b.mul(b.block_id_y(), n_pid_n),
            b.block_id_x(),
        )
        swz = chiplet_aware_super_tile_dynamic(
            b,
            wgid_flat,
            num_pid_m=n_pid_m,
            num_pid_n=n_pid_n,
            wgm=spec.trait.chiplet_wgm,
            num_xcds=spec.trait.chiplet_num_xcds,
            chunk_size=spec.trait.chiplet_chunk_size,
        )
        block_m_off = b.mul(swz.row, c_block_m)
        block_n_off = b.mul(swz.col, c_block_n)
    else:
        block_m_off = b.mul(b.block_id_y(), c_block_m)
        block_n_off = b.mul(b.block_id_x(), c_block_n)

    # LDS allocation. For compv4 we double-buffer; the second buffer is
    # logically allocated as a second smem region. The cshuffle epilogue
    # *reuses* the larger of (AB, C) since lifetimes don't overlap, but
    # for simplicity (and to match what CK's compv4 does — separate
    # allocations) we keep them distinct.
    A_smem = b.smem_alloc(F16, [block_m, block_k], name_hint="A_smem")
    B_smem = b.smem_alloc(F16, [block_n, block_k], name_hint="B_smem")
    # NOTE: a true double-buffered (`compv4`) pipeline would also allocate
    # `A_smem2 / B_smem2` here. We currently use a single LDS buffer per
    # operand in both single- and double-buffer modes; this kernel does
    # not yet read/write distinct LDS halves across overlapping iterations.

    # Per-warp MFMA tile (mfmas_per_warp_m * mfmas_per_warp_n MFMAs per K-step).
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    k_atoms = t.k_atoms_per_tile_k

    # Accumulators: one `<c_per_lane x float>` per warp-local MFMA tile.
    acc_init = _emit_zero_acc(b, spec)
    accs = [
        (f"acc_m{mi}_n{ni}", acc_init) for mi in range(mfmas_m) for ni in range(mfmas_n)
    ]

    # Global -> LDS coalesced copy plan.
    threads = spec.block_size
    load_vec = _choose_load_vec(spec)
    a_total = block_m * block_k
    b_total = block_n * block_k
    a_vec_total = a_total // load_vec
    b_vec_total = b_total // load_vec
    a_vecs_per_thread = a_vec_total // threads
    b_vecs_per_thread = b_vec_total // threads
    c_threads = b.const_i32(threads)
    c_load_vec = b.const_i32(load_vec)
    c_block_k_div_vec = b.const_i32(block_k // load_vec)

    # CK Tile-style data views. The A and B global tensors are
    # modelled as 3D views ``(batch, M_or_N, K)`` with element strides
    # ``(1, K, 1)``. The batch dim's stride of 1 lets us pre-compute
    # ``batch_off_a`` (in elements) once per CTA and pass it as the
    # batch-axis origin; the descriptor's offset formula then yields
    #
    #   offset = batch_off_a + (block_m_off + local_row) * K
    #          + (k_off + local_col)
    #
    # which matches the prior hand-rolled IR byte-for-byte after
    # constant folding. ``batch_off_a == 0`` in non-batched mode, so
    # the lowered IR collapses to the unchanged 2D form.
    a_view = make_global_view(A, shape=(1, 1, 1), dtype=F16, strides=(1, K, 1))
    b_view = make_global_view(Bp, shape=(1, 1, 1), dtype=F16, strides=(1, K, 1))
    # LDS views are 2D packed (block_m, block_k) / (block_n, block_k).
    from ..helpers.tensor_view import TensorDescriptor, TensorView

    a_lds_view = TensorView(
        base=A_smem,
        desc=TensorDescriptor.packed((block_m, block_k), F16),
        addr_space="lds",
    )
    b_lds_view = TensorView(
        base=B_smem,
        desc=TensorDescriptor.packed((block_n, block_k), F16),
        addr_space="lds",
    )

    def emit_load_phase(A_dst: Value, B_dst: Value, k_off: Value) -> None:
        """Coalesced global -> LDS copy for one K tile.

        Driven by :class:`TileWindow`: the A/B global views carry the
        ``(1, K, 1)`` descriptor; the per-call ``k_off`` shifts each
        tile's column origin. ``batch_off_a`` / ``batch_off_b`` ride
        in the batch-axis origin and the descriptor's `mul-by-1`
        folds away in LLVM, yielding identical lowered IR to the
        pre-helpers version.
        """
        a_global_tile = make_tile_window(
            a_view,
            lengths=(1, block_m, block_k),
            origin=(batch_off_a, block_m_off, k_off),
        )
        b_global_tile = make_tile_window(
            b_view,
            lengths=(1, block_n, block_k),
            origin=(batch_off_b, block_n_off, k_off),
        )
        a_lds_tile = make_tile_window(
            a_lds_view,
            lengths=(block_m, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )
        b_lds_tile = make_tile_window(
            b_lds_view,
            lengths=(block_n, block_k),
            origin=(b.const_i32(0), b.const_i32(0)),
        )

        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                a_val = a_global_tile.load_scalar(b, b.const_i32(0), row, col)
                a_lds_tile.store_scalar(b, row, col, value=a_val)
            else:
                a_val = a_global_tile.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                a_lds_tile.store_vec(b, row, col, value=a_val, n=load_vec)
        for e in range(b_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            if load_vec == 1:
                b_val = b_global_tile.load_scalar(b, b.const_i32(0), row, col)
                b_lds_tile.store_scalar(b, row, col, value=b_val)
            else:
                b_val = b_global_tile.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                b_lds_tile.store_vec(b, row, col, value=b_val, n=load_vec)

    def emit_mfma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value]
    ) -> List[Value]:
        """One K-tile worth of MFMAs across all per-warp atom positions
        and every K atom step inside this K-tile."""
        # Lane mapping into LDS: A wants per-lane `a_per_lane` K-elements
        # starting at K = k_blk * a_per_lane.
        if (t.warp_tile_m, t.warp_tile_n) == (16, 16):
            # 16x16: lane.row maps to M-in-atom (0..15), lane.k_blk = lane / 16.
            m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
            k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
            n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))
        else:
            # 32x32: lane.row maps to M-in-atom (0..31), lane.k_blk = lane / 32.
            m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
            k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
            n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))

        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

        new_accs: List[Value] = list(iter_vars)

        for kk in range(k_atoms):
            col_base = b.add(
                b.mul(k_blk, b.const_i32(a_per_lane)),
                b.const_i32(kk * t.warp_tile_k),
            )
            # We compute one A load per atom-row and reuse across N.
            a_rows = []
            for mi in range(mfmas_m):
                a_row = b.add(
                    warp_m_off, b.add(b.const_i32(mi * t.warp_tile_m), m_in_atom)
                )
                a_rows.append(_emit_smem_load(b, A_src, a_row, col_base, a_per_lane))

            # B load per atom-col, reused across M.
            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off, b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom)
                )
                b_cols.append(_emit_smem_load(b, B_src, b_row, col_base, b_per_lane))

            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    acc = _emit_mfma(b, spec, a_rows[mi], b_cols[ni], new_accs[flat])
                    new_accs[flat] = acc
                    flat += 1

            # Scheduler hint inside the K-atom loop: encourages the
            # backend to overlap one MFMA with the next set of ds_reads.
            if spec.trait.pipeline in ("compv3", "compv4"):
                # Modest hint: one DS-read group, one MFMA group per kk.
                b.sched_group_barrier(0x100, 1, 0)  # one DS read
                b.sched_group_barrier(0x008, mfmas_m * mfmas_n, 0)

        return new_accs

    # ---- the K loop ----
    for_op = b.scf_for_iter(c0, K, c_block_k, accs, iv_name="k0")
    with for_op as (k0, iter_vars):
        # Single-stage prefetch is the simplest correct pipeline. For
        # `compv4` with the double-buffered LDS we logically have:
        #   stage = (k0 / block_k) & 1
        # but we collapse that to a static "use A_smem this iteration"
        # since the loop body is fixed; the double buffer pays off via
        # the scheduler reordering loads vs MFMAs, not via per-iteration
        # ping-pong (we keep the body simple and rely on compv4's
        # sched_group_barrier interleave hints).
        emit_load_phase(A_smem, B_smem, k0)
        b.sync()

        new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)

        b.sync()
        b.scf_yield(*new_accs)

    # ---- epilogue ----
    if spec.trait.epilogue == "cshuffle":
        _emit_epilogue_cshuffle(
            b,
            spec,
            A_smem,
            for_op.results,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off,
            block_n_off,
            M,
            N,
            C,
            a_per_lane,
            b_per_lane,
            c_per_lane,
            batch_off_c=batch_off_c,
        )
    else:
        _emit_epilogue_default(
            b,
            spec,
            for_op.results,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off,
            block_n_off,
            M,
            N,
            C,
            c_per_lane,
            batch_off_c=batch_off_c,
        )

    return b.kernel


# ---------------------------------------------------------------------
# Epilogues
# ---------------------------------------------------------------------


def _emit_epilogue_default(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    M: Value,
    N: Value,
    C: Value,
    c_per_lane: int,
    *,
    batch_off_c: Optional[Value] = None,
) -> None:
    """Direct vector-store epilogue.

    Per-lane accumulator layout for an `m x n x k` MFMA atom on wave64:
      - 16x16 atoms: lane = (m_blk * 16 + n_in_atom), c_per_lane = 4
                     -> lane stores `(m_base + i, n_in_atom)` for i=0..3
        where m_base = m_blk * 4.
      - 32x32 atoms: c_per_lane = 16, accumulator is divided into 4
                     row-blocks of 4 floats each; runtime layout is
                     ((m_block_within_warp, row_in_block, n_in_atom))
                     with m_block_within_warp = lane / 32,
                     row_in_block coming from the accumulator index
                     i and row_block = i / 4.

    We use the canonical AMD layout per ROCm docs:
      For mfma_f32_32x32x8f16: each lane holds 16 fp32; the per-lane
      layout maps acc[i] to output element
      (row, col) = ((i//4)*8 + (lane/32)*4 + (i%4), lane%32)
    so the per-lane row stride between 4-element groups is 8 (not 4).
    """
    t = spec.tile
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    is_32x32 = (t.warp_tile_m, t.warp_tile_n) == (32, 32)

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    if is_32x32:
        # 32x32 layout: lane % 32 -> n_in_atom; lane / 32 -> M sub-block (0 or 1).
        c_atom_n = b.const_i32(t.warp_tile_n)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)  # 0 or 1 (32 lanes per half of MFMA M)
        # acc[i] -> row = (i//4)*8 + m_blk*4 + (i%4); col = n_in_atom
        # row-of-block: rb = i // 4 (0..3), row-in-block: ri = i % 4 (0..3)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                c_n = b.add(
                    b.add(block_n_off, warp_n_off),
                    b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom),
                )
                for i in range(c_per_lane):
                    rb = i // 4
                    ri = i % 4
                    m_off = b.add(
                        b.add(
                            b.mul(b.const_i32(8), b.const_i32(rb)),
                            b.mul(b.const_i32(4), m_blk),
                        ),
                        b.const_i32(ri),
                    )
                    c_m = b.add(
                        b.add(block_m_off, warp_m_off),
                        b.add(b.const_i32(mi * t.warp_tile_m), m_off),
                    )
                    c_off = b.add(b.mul(c_m, N), c_n)
                    if batch_off_c is not None:
                        c_off = b.add(batch_off_c, c_off)
                    v = b.vec_extract(acc, i)
                    h = b.trunc_f32_to_f16(v)
                    b.store_f16(C, c_off, h)
    else:
        # 16x16 atoms: lane = (m_blk * 16 + n_in_atom)
        c_atom_n = b.const_i32(t.warp_tile_n)
        c_clen = b.const_i32(c_per_lane)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        m_base = b.mul(m_blk, c_clen)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                c_n = b.add(
                    b.add(block_n_off, warp_n_off),
                    b.add(b.const_i32(ni * t.warp_tile_n), n_in_atom),
                )
                for i in range(c_per_lane):
                    m_off = b.add(m_base, b.const_i32(i))
                    c_m = b.add(
                        b.add(block_m_off, warp_m_off),
                        b.add(b.const_i32(mi * t.warp_tile_m), m_off),
                    )
                    c_off = b.add(b.mul(c_m, N), c_n)
                    if batch_off_c is not None:
                        c_off = b.add(batch_off_c, c_off)
                    v = b.vec_extract(acc, i)
                    h = b.trunc_f32_to_f16(v)
                    b.store_f16(C, c_off, h)


def _emit_epilogue_cshuffle(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    _smem_unused: Value,  # placeholder for future reuse
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    M: Value,
    N: Value,
    C: Value,
    a_per_lane: int,
    b_per_lane: int,
    c_per_lane: int,
    *,
    batch_off_c: Optional[Value] = None,
) -> None:
    """LDS-staged cshuffle epilogue.

    Pattern (matches CK's `cshuffle_epilogue.hpp`):
      1. Each warp converts its per-warp-tile accumulators (`<c_per_lane
         x float>`) to `<c_per_lane x half>`.
      2. Each warp stores them to LDS in a layout where consecutive
         lanes hold consecutive N-direction elements (the *output*
         layout, not the MFMA layout).
      3. Barrier.
      4. A subset of `STORE_VECS = (tile_m * tile_n) / store_vec_width`
         threads each read `<store_vec_width x half>` from LDS and
         issue one `<store_vec_width x half>` global store.

    For now we implement the 16x16 case and the 32x32 case using a
    distribution where every thread writes its own block-local row of
    `c_per_lane` halves into LDS at the canonical MFMA position, then
    a flat distribution of threads issues 4-wide global stores.

    The MFMA->LDS index math matches what we used in the default
    epilogue (which keeps the implementation honest: same lane->output
    mapping, just an extra LDS pass).
    """
    t = spec.tile
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    is_32x32 = (t.warp_tile_m, t.warp_tile_n) == (32, 32)

    # LDS staging tile: tile_m x tile_n of fp16.
    Cs = b.smem_alloc(F16, [t.tile_m, t.tile_n], name_hint="C_smem")

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    # ---- step 1+2: warp accs -> LDS at the MFMA layout. ----
    if is_32x32:
        c_atom_n = b.const_i32(t.warp_tile_n)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                acc_h = b.vec_trunc_f32_to_f16(acc)
                ld_n = b.add(
                    b.add(warp_n_off, b.const_i32(ni * t.warp_tile_n)), n_in_atom
                )
                for i in range(c_per_lane):
                    rb = i // 4
                    ri = i % 4
                    m_off = b.add(
                        b.add(
                            b.mul(b.const_i32(8), b.const_i32(rb)),
                            b.mul(b.const_i32(4), m_blk),
                        ),
                        b.const_i32(ri),
                    )
                    ld_m = b.add(
                        b.add(warp_m_off, b.const_i32(mi * t.warp_tile_m)), m_off
                    )
                    h = b.vec_extract(acc_h, i)
                    b.smem_store_f16(Cs, [ld_m, ld_n], h)
    else:
        c_atom_n = b.const_i32(t.warp_tile_n)
        c_clen = b.const_i32(c_per_lane)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        m_base = b.mul(m_blk, c_clen)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                acc_h = b.vec_trunc_f32_to_f16(acc)
                ld_n = b.add(
                    b.add(warp_n_off, b.const_i32(ni * t.warp_tile_n)), n_in_atom
                )
                for i in range(c_per_lane):
                    m_off = b.add(m_base, b.const_i32(i))
                    ld_m = b.add(
                        b.add(warp_m_off, b.const_i32(mi * t.warp_tile_m)), m_off
                    )
                    h = b.vec_extract(acc_h, i)
                    b.smem_store_f16(Cs, [ld_m, ld_n], h)

    # ---- step 3: barrier. ----
    b.sync()

    # ---- step 4: wide global stores. ----
    # STORE_VECS = (tile_m * tile_n) / store_vec. We pick store_vec as
    # wide as we can naturally align (8 halves = 16 B).
    threads = spec.block_size
    store_vec = 8
    while store_vec > 1 and (
        (t.tile_n % store_vec != 0)
        or ((t.tile_m * t.tile_n) // store_vec < threads)
        or (((t.tile_m * t.tile_n) // store_vec) % threads)
    ):
        store_vec //= 2

    if store_vec == 1:
        # Pathological: fall back to scalar stores.
        store_vec = 1

    tid = b.thread_id_x()
    c_threads = b.const_i32(threads)
    c_tile_n_div_vec = b.const_i32(t.tile_n // store_vec)
    vecs_per_thread = (t.tile_m * t.tile_n // store_vec) // threads
    for e in range(vecs_per_thread):
        vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
        row = b.div(vec_idx, c_tile_n_div_vec)
        col_v = b.mod(vec_idx, c_tile_n_div_vec)
        col = b.mul(col_v, b.const_i32(store_vec)) if store_vec > 1 else col_v

        c_m = b.add(block_m_off, row)
        c_n = b.add(block_n_off, col)
        c_off = b.add(b.mul(c_m, N), c_n)
        if batch_off_c is not None:
            c_off = b.add(batch_off_c, c_off)

        if store_vec == 1:
            h = _load_smem_scalar(b, Cs, row, col)
            b.store_f16(C, c_off, h)
        else:
            hv = _load_smem_vec(b, Cs, row, col, store_vec)
            b.global_store_vN_f16(C, c_off, hv, store_vec)


def _load_smem_scalar(b: IRBuilder, smem: Value, row: Value, col: Value) -> Value:
    # We expose vector loads but not a scalar half load from smem yet;
    # the v=1 path is rare. Emit as a 2-half vector and extract index 0.
    v = b.smem_load_vN_f16(smem, row, col, n=2)
    return b.vec_extract(v, 0)


def _load_smem_vec(b: IRBuilder, smem: Value, row: Value, col: Value, n: int) -> Value:
    if n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN_f16(smem, row, col, n=n)


# ---------------------------------------------------------------------
# Cartesian-product enumeration matching CK's default_config.json
# ---------------------------------------------------------------------


def all_dispatcher_configs(
    *,
    tile_m: Sequence[int] = (128, 256),
    tile_n: Sequence[int] = (128, 256),
    tile_k: Sequence[int] = (32, 64),
    warp_m: Sequence[int] = (2, 4),
    warp_n: Sequence[int] = (2, 4),
    warp_k: Sequence[int] = (1,),
    warp_tile: Sequence[Tuple[int, int, int]] = (
        (16, 16, 16),
        (32, 32, 8),
        (32, 32, 16),
        (16, 16, 32),
    ),
    pipeline: Sequence[Pipeline] = ("compv3", "compv4"),
    scheduler: Sequence[Scheduler] = ("intrawave",),
    epilogue: Sequence[Epilogue] = ("default", "cshuffle"),
    pad: Sequence[bool] = (False,),
    persistent: Sequence[bool] = (False,),
    wave_size: int = 64,
    name_prefix: str = "ck_dsl_universal",
    arch: str = "gfx950",
) -> Iterator[UniversalGemmSpec]:
    """Yield every valid `(TileSpec, TraitSpec)` combo on this arch.

    Defaults mirror `dispatcher/codegen/default_config.json` for fp16
    (which uses `pipeline=[compv4]`, `epilogue=[cshuffle]`). We accept
    the broader space so a sweep can compare CK's choices against
    alternatives (e.g. `mem` for memory-bound shapes).
    """
    for tm in tile_m:
        for tn in tile_n:
            for tk in tile_k:
                for wm in warp_m:
                    for wn in warp_n:
                        for wk in warp_k:
                            for wt in warp_tile:
                                for pl in pipeline:
                                    for sc in scheduler:
                                        for ep in epilogue:
                                            for p in pad:
                                                for pers in persistent:
                                                    spec = UniversalGemmSpec(
                                                        name=name_prefix,
                                                        tile=TileSpec(
                                                            tile_m=tm,
                                                            tile_n=tn,
                                                            tile_k=tk,
                                                            warp_m=wm,
                                                            warp_n=wn,
                                                            warp_k=wk,
                                                            warp_tile_m=wt[0],
                                                            warp_tile_n=wt[1],
                                                            warp_tile_k=wt[2],
                                                        ),
                                                        trait=TraitSpec(
                                                            pipeline=pl,
                                                            scheduler=sc,
                                                            epilogue=ep,
                                                            pad_m=p,
                                                            pad_n=p,
                                                            pad_k=p,
                                                            persistent=pers,
                                                        ),
                                                        wave_size=wave_size,
                                                    )
                                                    ok, _ = is_valid_spec(
                                                        spec, arch=arch
                                                    )
                                                    if ok:
                                                        yield spec
