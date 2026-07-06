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

from ...core.ir import (
    BF16,
    F16,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from ...helpers.io import io_ir_type
from ...helpers.spec import WarpTileBlockSizeMixin, choose_load_vec
from ...helpers.tensor_view import make_global_view, make_tile_window


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


Pipeline = Literal["mem", "compv3", "compv4", "wsp3", "wmma_v1"]
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
    # P40: when True, the kernel expects the B operand pre-shuffled by
    # the host into the layout :func:`rocke.helpers.preshuffle.
    # host_preshuffle_layout` produces, and the per-lane B-load uses
    # :func:`emit_preshuffleb_offset` to compute the byte offset
    # (one ``buffer_load_dwordx4`` per K-tile vs the per-K-element
    # strided scalar loads of the column-major path). The flag-free
    # default keeps the canonical strided-scalar load.
    preshuffle_b: bool = False
    # P41: DirectToLDS (DTLA/DTLB). When True, the global -> LDS load
    # phase emits ``raw.ptr.buffer.load.lds`` (one instruction per chunk;
    # hardware writes the dword payload straight into LDS) instead of
    # the round-trip ``buffer_load -> VGPR -> ds_write_b128`` pair the
    # default path uses. Matches Tensile's ``DTLA1_DTLB1`` token in
    # tuned kernels like ``MT16x16x512`` on gfx950. Constraints inherited
    # from :class:`AsyncTileLoader`: per-lane payload must be 4 / 12 / 16
    # bytes (dwords in {1,3,4}; dwords=2 is rejected by the intrinsic).
    direct_to_lds: bool = False
    # Per-operand cache hints when direct_to_lds is True. Default mirrors
    # rocBLAS skinny-M behaviour: A is small (M*K halves) and reused
    # across CTAs through L2 -> CACHE_ALL; B is the 32 MiB weight matrix,
    # one-shot streamed -> CACHE_STREAM.
    dtl_cache_a: int = 0  # CACHE_ALL
    dtl_cache_b: int = 2  # CACHE_STREAM
    # Prefetch ping-pong on top of direct_to_lds. Allocates two LDS buffers
    # per operand and issues next-iter DTLA loads while current-iter MFMAs
    # run, then waits with ``s_waitcnt vmcnt(loads_in_flight)`` so only the
    # current tile's loads have to complete before MFMAs start. Requires
    # ``direct_to_lds=True``. Doubles LDS usage (knocks 2 WGs/CU down to 1
    # at typical tile sizes) but hides the global-load latency that
    # otherwise serialises every K-tile.
    dtl_prefetch: bool = False
    # MoE active-tile early-exit. When True (only honored in
    # ``batched=True`` mode), the kernel takes two extra args
    # (``SortedTokenIds: ptr<i32>``, ``slot_size: i32``) and at CTA
    # entry computes a wave-uniform ``do_work`` predicate from
    # ``SortedTokenIds[block_id_z * slot_size + block_id_y * tile_m]``.
    # The K-loop + epilogue are wrapped in ``scf.if(do_work)`` so an
    # inactive expert tile (``first_row_token == -1``) skips all
    # MFMAs, LDS reads, and HBM stores. Enables CK Tile-style
    # "expert-by-expert" MoE dispatch over a static grid that's
    # padded to the worst-case ``experts`` size.
    active_tile_skip: bool = False
    # LDS K-padding: extra columns added to each AB LDS row so the per-row
    # byte stride is not a power-of-two multiple of the bank count, breaking
    # the stride-based bank-conflict pattern on the MFMA ds_reads (gfx950 has
    # 64 banks; an unpadded block_k=32 f16 row = 64 B stride aliases banks
    # every 4 rows). Default 0 = no change (golden-gate-safe). Use a multiple
    # of the load vector width (e.g. 16 halves) so the wide vector ds_read /
    # ds_write stay naturally aligned -- pad=4 breaks 16-byte alignment and
    # regresses hard. Measured NEUTRAL-to-NEGATIVE on square fp16 (the extra
    # LDS costs occupancy and outweighs the conflict reduction) -- prefer
    # ``lds_swizzle`` (zero LDS cost). Kept as an opt-in knob for shapes/dtypes
    # where the trade may pay. Non-DTL only (the direct-to-LDS path computes
    # flat byte offsets that assume a contiguous block_k stride).
    lds_k_pad: int = 0
    # LDS XOR swizzle (st_16x32-style): toggle the 32-byte column group on
    # rows with bit 3 set so consecutive-row MFMA ds_reads hit different banks.
    # ZERO LDS overhead (unlike lds_k_pad). Applied identically to the LDS
    # store and ds_read columns -> bit-exact. Measured ~+3% on square fp16/bf16.
    # Default off (golden-gate-safe). Non-DTL only; 2-byte dtypes.
    lds_swizzle: bool = False
    # Split-K over the production universal body. When > 1, the kernel:
    #   * takes a third grid dim ``block_id_z`` in [0, split_k) selecting
    #     a K-slice ``[z*ks, (z+1)*ks)`` (``ks = K // split_k``) for the
    #     CTA's K-loop, instead of the full ``[0, K)``;
    #   * replaces the ``C`` output param with an f32 workspace
    #     ``Cf32[M, N]`` and the epilogue atomic-adds each warp's f32
    #     accumulator into it (instead of the direct/cshuffle bf16 store).
    # The grid becomes ``(N_tiles, M_tiles, split_k)``. The caller must
    # zero-init the f32 workspace before launch and cast it to the target
    # dtype afterwards (Python-side finalisation, matching the v1 StreamK
    # atomic-strategy contract). This re-uses the fast vectorized +
    # LDS-double-buffered ``compv4`` load/MFMA inner verbatim; only the
    # K-loop bound and the epilogue change. ``split_k == 1`` (default)
    # keeps the canonical single-K-pass body byte-identical.
    split_k: int = 1
    # compv4/compv3 schedule directives: the per-cluster s_setprio(1/0) +
    # sched_barrier(0) fences (_mma_cluster) AND the two-stage
    # sched_group_barrier HotLoop interleave (_emit_hotloop_schedule). On gfx950
    # these OVER-CONSTRAIN comgr's backend scheduler -- removing them lets the
    # hardware scheduler pack MFMAs tighter. Measured +1.9-2.5% (MfmaUtil
    # 63->68%) on square fp16/bf16 GEMM, 200/200 GPU-event-timed cycles, relerr=0
    # (see optimization/utilities/skills/empirical-case-studies.md, Case Study 7).
    # None (default): arch-resolved -> hints OFF on gfx950 (take the uplift), ON
    # elsewhere (preserve the historical emission). True/False forces the choice.
    emit_sched_hints: Optional[bool] = None


@dataclass(frozen=True)
class DataSpec:
    """Element / accumulator / layout choice. Today: f16 in, f16 out, f32 acc, RCR."""

    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_c: str = "fp16"
    dtype_acc: str = "fp32"
    layout: str = "RCR"


def mono_data_spec(dtype: str) -> DataSpec:
    """Canonical single-element-type :class:`DataSpec` (A == B == C == ``dtype``).

    The ``f16`` / ``fp16`` aliases collapse to ``fp16``; the accumulator
    (``fp32``) and layout (``RCR``) keep the :class:`DataSpec` defaults. The
    GEMM-family wrappers (batched / grouped / fused-MoE) all share this exact
    mapping, so routing them through one helper keeps the produced
    :class:`DataSpec` -- and therefore the kernel name and emitted IR --
    byte-identical to the hand-rolled bodies it replaces.
    """
    dt = "fp16" if dtype in ("f16", "fp16") else dtype
    return DataSpec(dtype_a=dt, dtype_b=dt, dtype_c=dt)


@dataclass(frozen=True)
class UniversalGemmSpec(WarpTileBlockSizeMixin):
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
        self._init_block_size()

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        t = self.tile
        tr = self.trait
        return kernel_name_join(
            self.name,
            self.data.dtype_a,
            f"t{t.tile_m}x{t.tile_n}x{t.tile_k}",
            f"w{t.warp_m}x{t.warp_n}x{t.warp_k}",
            f"wt{t.warp_tile_m}x{t.warp_tile_n}x{t.warp_tile_k}",
            f"{tr.pipeline}_{tr.scheduler}_{tr.epilogue}",
            flags={
                "pad": any([tr.pad_m, tr.pad_n, tr.pad_k]),
                "pers": tr.persistent,
                "bat": self.batched,
                "preb": tr.preshuffle_b,
                "dtl": tr.direct_to_lds,
                "pref": tr.dtl_prefetch,
                "actt": tr.active_tile_skip,
                f"spk{tr.split_k}": tr.split_k > 1,
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

_BF16_WARP_TILE_SHAPES_GFX950 = {
    (16, 16, 16),
    (16, 16, 32),
}


def _dtype_ir(name: str) -> Type:
    """Resolve GEMM storage dtype strings to IR types.

    Universal GEMM currently supports homogeneous A/B/C storage dtypes
    for f16 and bf16 with f32 accumulation.
    """
    return io_ir_type(name)


def _mma_family(arch: str) -> str:
    """The MMA atom family this target uses: ``"wmma"`` for the RDNA
    wave32 targets (gfx11xx), ``"mma"`` (MFMA) for CDNA.

    The single body in :func:`build_universal_gemm` selects the atom from
    the target's catalog using this family, so the same spec resolves to an
    MFMA op_id on gfx942/gfx950 and a WMMA op_id on gfx1151.
    """
    from ...core.arch import ArchTarget

    return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma"


def _resolve_mma_op(spec: "UniversalGemmSpec", arch: str):
    """Resolve the :class:`~rocke.core.arch.MmaOp` for ``spec`` on ``arch``.

    The op carries the backend ``op_id`` (consumed by the target-neutral
    :meth:`IRBuilder.mma`), the per-lane fragment vector lengths
    (``a_frag_len`` / ``b_frag_len`` / ``c_frag_len``) and the
    lane/slot -> tile-coordinate layout maps that drive the fragment loads and
    the accumulator scatter. Returns ``None`` if the target has no atom for the
    spec's warp-tile shape + dtype (callers turn that into a structured reject).
    """
    from ...core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    t = spec.tile
    a_name = spec.data.dtype_a  # homogeneous A/B/C (validated in _storage_dtype)
    return target.mma.op_for_shape(
        family=_mma_family(arch),
        a_dtype=a_name,
        b_dtype=a_name,
        c_dtype="fp32",
        m=t.warp_tile_m,
        n=t.warp_tile_n,
        k=t.warp_tile_k,
    )


def _storage_dtype(spec: UniversalGemmSpec) -> Type:
    d = spec.data
    if d.dtype_a != d.dtype_b or d.dtype_a != d.dtype_c:
        raise ValueError(
            "UniversalGemmSpec currently requires homogeneous A/B/C dtypes; "
            f"got A={d.dtype_a}, B={d.dtype_b}, C={d.dtype_c}"
        )
    if d.dtype_acc not in ("fp32", "f32"):
        raise ValueError(
            f"UniversalGemmSpec only supports fp32 accumulation, got {d.dtype_acc!r}"
        )
    if d.layout != "RCR":
        raise ValueError(
            f"UniversalGemmSpec only supports RCR layout, got {d.layout!r}"
        )
    return _dtype_ir(d.dtype_a)


def _ab_lds_plan(spec: UniversalGemmSpec, arch: str) -> Tuple[int, bool, bool]:
    """AB LDS double-buffer plan, shared by the validity gate and emitter.

    Returns ``(ab_single, db, two_buf)`` as pure Python ints/bools:

    * ``ab_single`` — bytes for one (single-buffered) AB LDS region,
      ``(tile_m*tile_k + tile_n*tile_k) * 2``.
    * ``db`` — whether the compv4 software-pipelined double buffer is
      enabled (compv4, direct epilogue, not DTL, and the doubled buffer
      still fits 2 WG/CU).
    * ``two_buf`` — whether the AB LDS is double-buffered at all
      (``dtl_prefetch`` ping-pong OR ``db``).

    Keeping ``is_valid_spec`` and ``build_universal_gemm`` in lock-step
    through this single source avoids the class of bug where the gate
    reserves 2x AB but the emitter single-buffers (or vice-versa).
    """
    from ...core.arch import ArchTarget

    t = spec.tile
    ab_single = ((t.tile_m * t.tile_k) + (t.tile_n * t.tile_k)) * 2
    lds_cap = ArchTarget.from_gfx(arch).lds_capacity_bytes
    db_fits_2wg = (2 * ab_single) * 2 <= lds_cap
    db = (
        spec.trait.pipeline == "compv4"
        and spec.trait.epilogue != "cshuffle"
        and not spec.trait.direct_to_lds
        and db_fits_2wg
    )
    two_buf = bool(spec.trait.dtl_prefetch) or db
    return ab_single, db, two_buf


def is_valid_spec(spec: UniversalGemmSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return `(ok, reason)`. The same predicate CK's dispatcher uses
    to drop unbuildable configs from a sweep.

    Architecture facts (legal MFMA atoms, LDS capacity, max threads/block)
    are sourced from :class:`rocke.core.arch.ArchTarget`, so this predicate
    is arch-aware: a warp-tile atom that exists on gfx950 but not gfx942
    (e.g. the wide ``16x16x32`` f16 atom) is rejected for gfx942 with a
    structured reason instead of crashing comgr at lower time.
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    t = spec.tile
    try:
        _storage_dtype(spec)  # validates homogeneous dtypes / fp32 acc / RCR
    except ValueError as e:
        return False, str(e)

    # MMA atom must be in the target's catalog for this dtype combo. The
    # family is selected from the target's wave size: CDNA (wave64) uses MFMA,
    # the RDNA wave32 targets (gfx11xx) use WMMA. The same warp-tile shape thus
    # resolves to an MFMA op_id on gfx942/gfx950 and a WMMA op_id on gfx1151.
    family = _mma_family(arch)
    a_name = spec.data.dtype_a  # homogeneous A/B/C (checked above)
    atom = (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    if not target.supports_dtype_combo(a_name, a_name, "fp32", family=family):
        return False, f"unsupported GEMM dtype {a_name!r} on {arch}"
    if not target.mma.has_shape(
        family=family,
        a_dtype=a_name,
        b_dtype=a_name,
        c_dtype="fp32",
        m=t.warp_tile_m,
        n=t.warp_tile_n,
        k=t.warp_tile_k,
    ):
        return False, f"unsupported {a_name} warp_tile {atom} on {arch}"

    # The spec's wave size must match the target's. ``spec.wave_size`` is baked
    # into ``block_size`` at construction, so a wave64 spec cannot launch on a
    # wave32 target (and vice versa); reject early with a structured reason
    # rather than emitting a kernel whose lane geometry the hardware rejects.
    if spec.wave_size != target.wave_size:
        return False, (
            f"spec wave_size {spec.wave_size} != {arch} wave_size {target.wave_size}"
        )

    # WMMA coverage is intentionally narrower than the full CDNA MFMA matrix:
    # gfx11/gfx12 RDNA supports the 16x16x16 atom and gfx1250 supports the
    # gfx1250-class 16x16x32 atom, both through the simple ``mem`` pipeline +
    # ``default`` epilogue. The richer pipelines (compv3 / compv4 scheduler
    # interleave, cshuffle LDS-staged C, DTLA, preshuffle) encode MFMA-shaped
    # assumptions and are gated off until ported. CDNA MFMA keeps the full
    # matrix.
    if family == "wmma":
        supported_atoms = {(16, 16, 16)}
        if arch == "gfx1250":
            supported_atoms = {(16, 16, 32)}
        if atom not in supported_atoms:
            supported = ", ".join(
                f"{m}x{n}x{k}" for (m, n, k) in sorted(supported_atoms)
            )
            return False, f"WMMA path supports only {supported} (got {atom}) on {arch}"
        if spec.trait.pipeline not in ("mem", "wmma_v1"):
            return False, (
                f"WMMA path supports only the 'mem' or 'wmma_v1' pipeline "
                f"(got {spec.trait.pipeline!r}) on {arch}"
            )
        if spec.trait.epilogue != "default":
            return False, (
                f"WMMA path supports only the 'default' epilogue "
                f"(got {spec.trait.epilogue!r}) on {arch}"
            )
        for flag, label in (
            (spec.trait.preshuffle_b, "preshuffle_b"),
            (spec.trait.direct_to_lds, "direct_to_lds"),
            (spec.trait.dtl_prefetch, "dtl_prefetch"),
            (spec.trait.active_tile_skip, "active_tile_skip"),
            (spec.trait.chiplet_swizzle, "chiplet_swizzle"),
        ):
            if flag:
                return False, f"WMMA path does not support {label} on {arch}"

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

    # LDS budget. The cap is the target's per-WG LDS capacity (160 KiB on
    # gfx950 / CDNA4, 64 KiB on gfx942 / CDNA3). Our current emitter does
    # NOT alias AB and cshuffle staging (CK does; a separate optimisation
    # we have not yet wired up), so the actual usage is additive:
    #   compv4 single AB:   tile_m*tile_k*2 + tile_n*tile_k*2
    #   compv4 double AB:   2 * single AB
    #   cshuffle staging:   tile_m*tile_n*2   (f16)
    #   total:              double_buffer_AB + (cshuffle ? C : 0)
    # When we land the AB/C aliasing in the cshuffle emitter, swap the
    # `+` for a `max`.
    # AB is double-buffered (2x LDS) only when the emitter actually
    # ping-pongs two halves: ``dtl_prefetch`` (DTLA ping-pong) or the
    # compv4 software-pipelined double buffer, which the emitter enables
    # for the direct epilogue but NOT for cshuffle (whose separate C
    # staging would overflow the budget — see ``_db`` in
    # :func:`build_universal_gemm`). Keeping this gate in lock-step with
    # the emitter avoids both spurious rejections (over-reserving 2x AB
    # for a single-buffered compv4+cshuffle) and under-reserving. Both
    # sites share :func:`_ab_lds_plan` to stay in lock-step.
    ab_single, _, _ab_dbl = _ab_lds_plan(spec, arch)
    ab_bytes = ab_single * (2 if _ab_dbl else 1)
    c_bytes = t.tile_m * t.tile_n * 2 if spec.trait.epilogue == "cshuffle" else 0
    bytes_lds = ab_bytes + c_bytes
    if not target.fits_lds(bytes_lds):
        return False, (
            f"LDS budget {bytes_lds} > {target.lds_capacity_bytes} cap "
            f"(AB={ab_bytes}, C={c_bytes}) on {arch}"
        )

    # Per-WG thread cap (hardware limit from the target).
    if spec.block_size > target.max_threads_per_block:
        return False, (
            f"block_size {spec.block_size} > {target.max_threads_per_block} "
            f"(hardware cap) on {arch}"
        )

    # Global -> LDS vectorised load divisibility: at least one vec
    # element per thread per phase.
    threads = spec.block_size
    a_total = t.tile_m * t.tile_k
    b_total = t.tile_n * t.tile_k
    if a_total < threads or b_total < threads:
        return False, "block too small for one element/thread/phase"

    # Split-K (over the production body): the K-slice each CTA processes
    # is ``ks = K // split_k`` and must itself be a whole number of
    # K-tiles. We can only validate the K-slice divisibility at build
    # time when K is a compile-time fact, which it is not in the
    # universal body (K is a runtime arg). So we only check the
    # static invariants here: split_k >= 1, and the atomic-add epilogue
    # is only wired for the MFMA (CDNA) family. The K % split_k and
    # ks % tile_k divisibility are the caller's responsibility (mirrors
    # the v1 StreamK contract, where the partitioner enforces it).
    sk = spec.trait.split_k
    if sk < 1:
        return False, f"split_k must be >= 1 (got {sk})"
    if sk > 1 and family != "mma":
        return False, f"split_k > 1 is CDNA-only (got family {family!r} on {arch})"

    return True, "ok"


# ---------------------------------------------------------------------
# IR builders for the pieces
# ---------------------------------------------------------------------


def _mfma_atom_widths(spec: UniversalGemmSpec) -> Tuple[int, int, int]:
    """Return (a_per_lane, b_per_lane, c_per_lane) for the spec's MFMA atom.

    Kept for the MFMA-only callers (``moe_gemm_fused``) that compute the
    per-lane widths straight from the wave64 geometry. The contract-driven
    GEMM body uses :func:`_atom_frag_lengths` (op-sourced) instead, which is
    equal to this for MFMA but correct for WMMA too.
    """
    t = spec.tile
    waves = spec.wave_size
    a_per = (t.warp_tile_m * t.warp_tile_k) // waves
    b_per = (t.warp_tile_k * t.warp_tile_n) // waves
    c_per = (t.warp_tile_m * t.warp_tile_n) // waves
    return a_per, b_per, c_per


def _emit_mfma(
    b: IRBuilder, spec: UniversalGemmSpec, a: Value, bb: Value, c: Value
) -> Value:
    """Emit the MFMA for ``spec``'s warp-tile atom (MFMA-only callers).

    Routes through the same op_id the contract resolver picks, so the emission
    is byte-identical to the prior ISA-named helpers. The unified GEMM body
    uses :func:`_emit_mma` (op-driven, works for WMMA too).
    """
    t = spec.tile
    key = (t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    dtype = _storage_dtype(spec)
    if dtype == F16:
        if key == (16, 16, 16):
            return b.mfma_f32_16x16x16_f16(a, bb, c)
        if key == (16, 16, 32):
            return b.mfma_f32_16x16x32_f16(a, bb, c)
        if key == (32, 32, 8):
            return b.mfma_f32_32x32x8_f16(a, bb, c)
        if key == (32, 32, 16):
            return b.mfma_f32_32x32x16_f16(a, bb, c)
    if dtype == BF16:
        if key == (16, 16, 16):
            return b.mfma_f32_16x16x16_bf16(a, bb, c)
        if key == (16, 16, 32):
            return b.mfma_f32_16x16x32_bf16(a, bb, c)
    raise NotImplementedError(f"no MFMA emitter for {dtype.name} warp_tile {key}")


def _emit_zero_acc(b: IRBuilder, spec: UniversalGemmSpec) -> Value:
    """Zero accumulator vector sized from the spec geometry (MFMA-only callers)."""
    _, _, c_per = _mfma_atom_widths(spec)
    return b.zero_vec_f32(c_per)


def _atom_frag_lengths(op) -> Tuple[int, int, int]:
    """Return ``(a_frag_len, b_frag_len, c_frag_len)`` — the per-lane fragment
    vector lengths of the resolved :class:`~rocke.core.arch.MmaOp`.

    These come from the MMA contract, **not** from the ``shape / wave`` formula:
    the two are equal for MFMA (wave64) but diverge for WMMA, where the
    hardware replicates each operand fragment across the two half-waves
    (``a_frag_len == 16`` for the 16x16x16 atom, where ``16*16/32 == 8`` would
    be wrong). Driving everything off ``op.*_frag_len`` is what lets one body
    emit both ISAs.
    """
    return op.a_frag_len, op.b_frag_len, op.c_frag_len


def _emit_mma(b: IRBuilder, op, a: Value, bb: Value, c: Value) -> Value:
    """Target-neutral matmul: ``D = a * bb + c`` via :meth:`IRBuilder.mma`.

    The backend dispatches ``op.op_id`` to the matching MFMA call on CDNA or
    the WMMA call on RDNA. This reproduces the prior ISA-named helper emission
    byte-for-byte on CDNA (the helpers were already thin wrappers over
    :meth:`mma`).
    """
    return b.mma(op, a, bb, c)


def _emit_zero_acc_op(b: IRBuilder, op) -> Value:
    """Zero accumulator vector sized from the resolved op's ``c_frag_len``."""
    return b.zero_vec_f32(op.c_frag_len)


def _choose_load_vec(spec: UniversalGemmSpec) -> int:
    """Choose the widest naturally-aligned global-load width for this
    block shape. f16 -> we can vectorise up to 8 halves per lane.

    Thin adapter over :func:`rocke.helpers.spec.choose_load_vec`, the shared
    single-source picker; kept so callers (incl. ``moe_gemm_fused``) that
    pass a whole spec stay unchanged."""
    t = spec.tile
    return choose_load_vec(t.tile_m, t.tile_n, t.tile_k, spec.block_size)


def _emit_smem_load(
    b: IRBuilder, smem: Value, row: Value, col: Value, n: int, dtype: Type
) -> Value:
    if dtype == F16 and n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN(smem, row, col, dtype=dtype, n=n)


def _hotloop_inst_list(spec: UniversalGemmSpec, load_vec: int):
    """Build the per-K-tile :class:`HotLoopInstList` for ``spec``.

    Pure arithmetic from the block tile geometry + the spec's MFMA atom
    timing (``MfmaAtom.mfma_cycle`` / ``k_per_xdlops``), mirroring CK's
    ``BlockwiseGemmXdlops_pipeline_hotloop_inst``. The global buffer-load
    width is the kernel's coalesced ``load_vec`` (elements/lane/load); the
    LDS read/write widths default to the atom K-pack inside
    :meth:`HotLoopInstList.from_geometry` (the comp_v4 ``*_LDS_*_Width =
    KPerXDL`` convention).
    """
    from ...helpers.atoms import mfma_atom
    from ...helpers.schedule import HotLoopInstList

    t = spec.tile
    atom = mfma_atom(spec.data.dtype_a, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    return HotLoopInstList.from_geometry(
        atom=atom,
        block_size=spec.block_size,
        m_per_block=t.tile_m,
        n_per_block=t.tile_n,
        k_per_block=t.tile_k,
        m_repeat=t.mfmas_per_warp_m,
        n_repeat=t.mfmas_per_warp_n,
        a_buffer_load_width=load_vec,
        b_buffer_load_width=load_vec,
    )


def _hotloop_well_formed(il, pipeline: str) -> bool:
    """Whether the v3/v4 schedule for ``il`` yields only non-negative counts.

    The CK HotLoopScheduler is only instantiated for tiles large enough that
    every emitted ``sched_group_barrier`` count is >= 0. For very small / K-
    light tiles the integer-divided LDS counts collapse to 0 and the v4
    trailing ``C_MFMA / num_issue - 3`` group can go negative, which is not a
    legal hint. In those degenerate cases the caller keeps the flat hint
    (still a pure scheduling choice -> numerically identical either way).
    """
    num_buffer_load = il.a_buffer_load_inst_num + il.b_buffer_load_inst_num
    if num_buffer_load <= 0:
        return False
    if pipeline == "compv3":
        num_mfma_stage1 = il.c_mfma_inst_num - (
            il.num_dsread_a_mfma + il.num_dsread_b_mfma
        )
        if num_mfma_stage1 < 0:
            return False
        num_mfma_per_issue = num_mfma_stage1 // num_buffer_load
        # Stage-1 groups emit (num_mfma_per_issue - num_dswrite_per_issue_*)
        # MFMAs; that count must stay non-negative.
        if il.a_buffer_load_inst_num <= 0 or il.b_buffer_load_inst_num <= 0:
            return False
        dswr_a = il.a_lds_write_inst_num // il.a_buffer_load_inst_num
        dswr_b = il.b_lds_write_inst_num // il.b_buffer_load_inst_num
        return (num_mfma_per_issue - dswr_a) >= 0 and (num_mfma_per_issue - dswr_b) >= 0
    # compv4: the trailing MFMA group is C_MFMA / num_issue - 3.
    return (il.c_mfma_inst_num // num_buffer_load - 3) >= 0


def _emit_hotloop_schedule(b: IRBuilder, spec: UniversalGemmSpec, load_vec: int):
    """Emit the comp_v3 / comp_v4 two-stage HotLoop schedule once per K-tile.

    Replaces the old flat per-kk ``sched_group_barrier`` hint. Falls back to
    the flat hint for degenerate small tiles whose derived counts would
    produce an illegal (negative) group (see :func:`_hotloop_well_formed`).
    WMMA never reaches here (gated to the ``mem`` pipeline upstream).
    """
    from ...helpers.schedule import SchedulePolicy

    t = spec.tile
    pipeline = spec.trait.pipeline
    il = _hotloop_inst_list(spec, load_vec)
    policy = SchedulePolicy.for_pipeline(pipeline)
    if _hotloop_well_formed(il, pipeline):
        if pipeline == "compv3":
            policy.emit_compv3_hotloop(b, il)
        else:
            policy.emit_compv4_hotloop(b, il)
        return
    # Degenerate tile: keep the prior flat hint (numerically identical;
    # both are pure scheduling hints).
    b.sched_group_barrier(0x100, 1, 0)  # one DS read
    b.sched_group_barrier(0x008, t.mfmas_per_warp_m * t.mfmas_per_warp_n, 0)


# ---------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------


def build_universal_gemm(spec: UniversalGemmSpec, arch: str = "gfx950") -> KernelDef:
    """Build the IR for one universal GEMM instance on ``arch``.

    The body is **MMA-contract driven**: it resolves the warp-tile atom from
    the target's catalog (an MFMA op_id on gfx942/gfx950, a WMMA op_id on
    gfx1151), then drives the per-lane fragment loads, the matmul, and the
    accumulator scatter through that op's layout maps / fragment lengths and the
    target-neutral :meth:`IRBuilder.mma`. The same source thus emits MFMA on
    CDNA (byte-identical to the prior ISA-named emission) and WMMA on RDNA.

    On CDNA this dispatches on (pipeline, epilogue):

      | pipeline | epilogue | implementation |
      |---|---|---|
      | mem      | default  | single-buffer + direct vector stores (the dsl/01-05 family) |
      | compv3   | default  | single-buffer + sched_group_barrier interleave |
      | compv3   | cshuffle | single-buffer + LDS-staged f16 epilogue |
      | compv4   | default  | double-buffer LDS + sched_group_barrier + direct vector stores |
      | compv4   | cshuffle | double-buffer LDS + sched_group_barrier + LDS-staged cshuffle |
      | mem      | cshuffle | single-buffer + LDS-staged epilogue |

    All five CDNA paths share the same IR-construction subroutines below. The
    RDNA/WMMA target (gfx1151) is gated to the ``mem`` pipeline + ``default``
    epilogue + 16x16x16 atom subset (see :func:`is_valid_spec`); the richer
    pipelines/epilogues remain CDNA-only until ported.
    """

    # Warp-specialized producer/consumer 3-stage pipeline.
    # Lives in a SEPARATE emitter (gemm_wsp3.py) and produces new kernel
    # names, so the byte-identical golden gate over the existing
    # mem/compv3/compv4 paths is untouched (no existing emission edited).
    if spec.trait.pipeline == "wsp3":
        from .gemm_wsp3 import build_wsp3_gemm

        return build_wsp3_gemm(spec, arch=arch)

    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid GEMM spec: {why}")

    op = _resolve_mma_op(spec, arch)
    if op is None:  # pragma: no cover - is_valid_spec already rejects this
        raise ValueError(f"no MMA atom for spec on {arch}")

    b = IRBuilder(spec.kernel_name())
    # IMPORTANT: AMDGPU bakes `amdgpu-flat-work-group-size` into the
    # kernel descriptor; if we launch with more threads than that, HIP
    # returns `unspecified launch failure` *before* the kernel body
    # runs. Pin the upper bound to this spec's `block_size`.
    b.kernel.attrs["max_workgroup_size"] = spec.block_size
    if spec.trait.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.trait.waves_per_eu
    storage_dtype = _storage_dtype(spec)
    _split_k = spec.trait.split_k
    _is_split_k = _split_k > 1
    A = b.param(
        "A", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    Bp = b.param(
        "B", PtrType(storage_dtype, "global"), noalias=True, readonly=True, align=16
    )
    # In split-K mode the output is the f32 accumulation workspace the
    # epilogue atomic-adds into (the caller casts it to the target dtype
    # afterwards). It is read+write (atomicrmw), so drop the writeonly
    # attribute. Outside split-K the C param is byte-identical to before.
    if _is_split_k:
        from ...core.ir import F32 as _F32

        C = b.param("Cf32", PtrType(_F32, "global"), noalias=True, align=4)
    else:
        C = b.param(
            "C",
            PtrType(storage_dtype, "global"),
            noalias=True,
            writeonly=True,
            align=16,
        )
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
    if spec.batched and spec.trait.active_tile_skip:
        # MoE active-tile gate. ``SortedTokenIds`` carries the
        # bucket -> token-id map; ``-1`` marks an inactive padded
        # row. ``slot_size`` is the per-expert padded row count.
        sorted_token_ids = b.param(
            "SortedTokenIds",
            PtrType(I32, "global"),
            noalias=True,
            readonly=True,
            align=4,
        )
        slot_size_p = b.param("slot_size", I32)

    t = spec.tile
    a_per_lane, b_per_lane, c_per_lane = _atom_frag_lengths(op)

    block_m = t.tile_m
    block_n = t.tile_n
    block_k = t.tile_k

    # Common geometry.
    c0 = b.const_i32(0)
    # Split-K K-slice bounds. Each CTA along ``block_id_z`` owns the slice
    # ``[z * ks, (z+1) * ks)`` where ``ks = K // split_k``; the K-loop runs
    # over that slice instead of ``[0, K)``. ``block_id_z`` is CTA-uniform so
    # the slice base is SGPR-pinned. For ``split_k == 1`` these collapse to
    # ``k_lo = 0`` / ``k_hi = K`` (the bound expressions below substitute them
    # verbatim, so the non-split body stays byte-identical).
    if _is_split_k:
        ks = b.div(K, b.const_i32(_split_k))
        k_lo = b.to_sgpr_u32(b.mul(b.block_id_z(), ks))
        k_hi = b.to_sgpr_u32(b.add(k_lo, ks))
    else:
        k_lo = c0
        k_hi = None  # use K directly so the bound SSA is unchanged
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

    # LDS XOR swizzle (st_16x32-style, zero LDS overhead). For 2-byte
    # elements the 32-byte (16-half) column group is toggled on rows whose
    # bit 3 is set: ``col ^= ((row>>3)&1)<<4``. Applied identically to the
    # LDS store column and the MFMA ds_read column, so the physical address
    # for any logical (row,col) agrees on both sides (correctness preserved)
    # while consecutive rows hit different banks. Opt-in via CK_SWIZZLE; the
    # 16-half toggle preserves the 8-half load-vector alignment.
    _SWZ = spec.trait.lds_swizzle
    # compv4 schedule-hint policy: explicit trait override wins; the default
    # (None) takes the measured gfx950 uplift (hints OFF) and preserves the
    # historical emission on every other arch. Gates both the per-cluster
    # s_setprio/sched_barrier fences and the sched_group_barrier HotLoop.
    _sched_hints = (
        spec.trait.emit_sched_hints
        if spec.trait.emit_sched_hints is not None
        else (arch != "gfx950")
    )
    # LDS XOR swizzle: ``col ^= ((row >> R) % 2^W) << L``. XOR of any
    # deterministic function of ``row`` into ``col`` is correctness-preserving
    # as long as it is applied identically to the LDS store-source + ds_read
    # columns (it is) and stays in [0, block_k).
    #
    # The optimal (measured 0.0% LDSBankConflict) parameters are derived from
    # the geometry, not guessed:
    #   * L = log2(elem)  -- swizzle whole ``elem``-half (b128) vector slots so
    #                        every target is naturally 16B-aligned.
    #   * W = log2(block_k/elem) -- use ALL element slots in the row.
    #   * R = 0           -- key the swizzle on the LOW row bits, i.e. the
    #                        ``m_in_atom`` (0..warp_tile_m-1) that actually
    #                        aliases (every M-row of an atom reads the same col,
    #                        and the row*stride term vanishes mod 32 banks, so
    #                        only a low-bit element permutation decorrelates them).
    # The earlier default (R=3,W=1,L=4) keyed on HIGH row bits / 2 slots and so
    # floored at 25% (4-way). Env vars override for sweeps; geometry fallback to
    # the 2-slot toggle if elem/slots aren't a clean power-of-2 split.
    import os as _os

    def _ilog2(x):
        return x.bit_length() - 1 if x and (x & (x - 1)) == 0 else None

    _swz_elem = a_per_lane if a_per_lane == b_per_lane else 0
    _swz_slots = (
        (block_k // _swz_elem) if (_swz_elem and block_k % _swz_elem == 0) else 0
    )
    _auto_l = _ilog2(_swz_elem)
    _auto_w = _ilog2(_swz_slots)
    if _auto_l is not None and _auto_w is not None and _auto_w >= 1:
        _def_r, _def_w, _def_l = 0, _auto_w, _auto_l
    else:
        _def_r, _def_w, _def_l = 3, 1, 4
    _swz_r = int(_os.environ.get("CK_SWZ_R", str(_def_r)))
    _swz_w = int(_os.environ.get("CK_SWZ_W", str(_def_w)))
    _swz_l = int(_os.environ.get("CK_SWZ_L", str(_def_l)))
    _c_swr, _c_swmod, _c_swl = (
        b.const_i32(_swz_r),
        b.const_i32(1 << _swz_w),
        b.const_i32(_swz_l),
    )

    def _swz_col(col, row):
        if not _SWZ:
            return col
        return b.xor(col, b.shl(b.mod(b.lshr(row, _c_swr), _c_swmod), _c_swl))

    if spec.batched:
        # P57: every wave-uniform i32 derived from the batch axis is
        # pinned into an SGPR via ``to_sgpr_u32`` so the per-tile
        # address arithmetic stays in scalar registers rather than
        # being re-materialised in VGPRs at every consumer (one
        # ``v_readfirstlane_b32`` saved per use across the K-loop +
        # epilogue). Mirrors CK Tile ``batched_gemm_kernel.hpp:215-230``
        # ``amd_wave_read_first_lane`` pattern.
        batch_idx = b.to_sgpr_u32(b.block_id_z())
        batch_off_a = b.to_sgpr_u32(b.mul(batch_idx, stride_a))
        batch_off_b = b.to_sgpr_u32(b.mul(batch_idx, stride_b))
        batch_off_c = b.to_sgpr_u32(b.mul(batch_idx, stride_c))
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
        from ...helpers.grid import chiplet_aware_super_tile_dynamic

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
        # P57: pin chiplet-swizzled offsets in SGPR. ``swz.row`` / ``swz.col``
        # are wave-uniform (function of CTA-uniform inputs), so
        # ``readfirstlane`` is safe.
        block_m_off = b.to_sgpr_u32(b.mul(swz.row, c_block_m))
        block_n_off = b.to_sgpr_u32(b.mul(swz.col, c_block_n))
    else:
        # P57: pin the non-swizzled batched-GEMM offsets in SGPR too.
        block_m_off = b.to_sgpr_u32(b.mul(b.block_id_y(), c_block_m))
        block_n_off = b.to_sgpr_u32(b.mul(b.block_id_x(), c_block_n))

    # LDS allocation. compv4 (non-DTL, direct epilogue) double-buffers AB
    # so the next K-tile's global->LDS copy overlaps the current tile's
    # MFMA/ds_read work (real software pipelining; see ``_emit_kloop_db``).
    # Both halves live in one ``[2*block_m, block_k]`` / ``[2*block_n,
    # block_k]`` smem region and the K-loop alternates the parity row
    # origin. The cshuffle epilogue allocates a separate tile_m*tile_n C
    # staging buffer, so it stays single-buffered (doubling AB on top of C
    # overflows the budget).
    #
    # When ``dtl_prefetch`` is on, we likewise double the M/N dimension of
    # each LDS buffer; the ``parity`` (0 or 1) selects which half is the
    # current write target / read source, and the K-loop alternates so
    # next-iter DTLA writes go to the buffer the MFMAs aren't reading.
    _prefetch = bool(spec.trait.dtl_prefetch)
    if _prefetch and not spec.trait.direct_to_lds:
        raise ValueError("dtl_prefetch requires direct_to_lds=True")
    # compv4 (non-DTL) double-buffers the AB LDS so the next K-tile's
    # global->LDS copy overlaps the current K-tile's MFMA/ds_read work
    # (a real software-pipelined prefetch). The validity gate already
    # reserves 2x AB LDS for compv4, so this consumes budget that was
    # previously reserved-but-unused. DTL keeps its own ping-pong path.
    # Double-buffer only when the epilogue does NOT also stage C through
    # LDS: the cshuffle epilogue allocates a separate tile_m*tile_n C
    # buffer, so doubling AB on top of it overflows the LDS budget and
    # crushes occupancy (a net regression). The direct ("default")
    # epilogue stores straight to global, leaving the whole LDS budget
    # for the AB ping-pong.
    # Occupancy guard: doubling AB only pays off if the doubled buffer
    # still leaves room for the same number of workgroups per CU. On
    # gfx950 the LDS-limited occupancy step is at half the per-CU LDS
    # (2 WG/CU needs <= cap/2 bytes each). If the doubled AB would push a
    # config from 2 WG/CU down to 1, the lost occupancy outweighs the
    # prefetch overlap (measured: 256x128x64 regresses -3% when it drops
    # to 1 WG/CU). In that case we keep the single-buffer pipeline.
    # AB LDS double-buffer plan, shared with the validity gate via
    # :func:`_ab_lds_plan` so the reserved/used budget stays in lock-step.
    _, _db, _two_buf = _ab_lds_plan(spec, arch)
    _A_LDS_M = 2 * block_m if _two_buf else block_m
    _B_LDS_N = 2 * block_n if _two_buf else block_n
    # LDS K-padding (non-DTL only): widen each row's stride to break the
    # bank-conflict alias. The logical column range stays [0, block_k); only
    # the row stride grows, so the read GEP (alloc shape[1]) and the
    # store_vec TensorView (with_strides below) both pick up the padded stride.
    _lds_pad = spec.trait.lds_k_pad if not spec.trait.direct_to_lds else 0
    _lds_k = block_k + _lds_pad
    A_smem = b.smem_alloc(storage_dtype, [_A_LDS_M, _lds_k], name_hint="A_smem")
    B_smem = b.smem_alloc(storage_dtype, [_B_LDS_N, _lds_k], name_hint="B_smem")
    # When ``_two_buf`` (compv4 DB or dtl_prefetch) the two halves share
    # this one ``2*block_m``/``2*block_n``-tall region; the K-loop's
    # ``lds_parity`` shifts the load-store and MFMA-read row origin by
    # ``parity * block_m`` / ``parity * block_n`` between them. Single-
    # buffer pipelines (``mem``, ``compv3``, cshuffle-compv4) use the
    # bottom half only.

    # Per-warp MFMA tile (mfmas_per_warp_m * mfmas_per_warp_n MFMAs per K-step).
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    k_atoms = t.k_atoms_per_tile_k

    # Accumulators: one `<c_per_lane x float>` per warp-local MFMA tile.
    acc_init = _emit_zero_acc_op(b, op)
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
    a_view = make_global_view(
        A, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    b_view = make_global_view(
        Bp, shape=(1, 1, 1), dtype=storage_dtype, strides=(1, K, 1)
    )
    # LDS views are 2D packed (block_m, block_k) / (block_n, block_k).
    from ...helpers.tensor_view import TensorDescriptor, TensorView

    # When LDS K-padding is on, the logical shape stays (block_*, block_k) but
    # the row stride is the padded ``_lds_k`` so the store_vec addressing
    # agrees with the read GEP (which uses the padded alloc shape).
    if _lds_pad:
        _a_lds_desc = TensorDescriptor.with_strides(
            (block_m, block_k), (_lds_k, 1), storage_dtype
        )
        _b_lds_desc = TensorDescriptor.with_strides(
            (block_n, block_k), (_lds_k, 1), storage_dtype
        )
    else:
        _a_lds_desc = TensorDescriptor.packed((block_m, block_k), storage_dtype)
        _b_lds_desc = TensorDescriptor.packed((block_n, block_k), storage_dtype)
    a_lds_view = TensorView(base=A_smem, desc=_a_lds_desc, addr_space="lds")
    b_lds_view = TensorView(base=B_smem, desc=_b_lds_desc, addr_space="lds")

    # DirectToLDS (DTLA/DTLB) plumbing. We issue
    # ``async_buffer_load_lds_addr`` directly because
    # :class:`AsyncTileLoader` assumes a tile shape where the row dimension
    # wraps at ``halves_per_chunk`` — that's an attention-specific layout
    # and does not match a [block_m, block_k] GEMM tile with block_k >>
    # halves_per_chunk.
    #
    # The intrinsic writes ``dwords * 4`` bytes per lane lane-contiguous
    # starting at the wave-uniform ``lds_dst``. For our tile we lay
    # ``block_size`` lanes across the tile such that each lane covers one
    # chunk of ``dwords * 2`` halves (bf16 elements). Passes cover the
    # remaining chunks_total / block_size iterations.
    if spec.trait.direct_to_lds:
        from ...core.ir import I64 as _I64

        _DTL_DWORDS = 4  # 16 bytes/lane
        _DTL_HALVES = _DTL_DWORDS * 2  # 8 elements (bf16 halves) per lane chunk
        _DTL_BYTES_PER_LANE = _DTL_DWORDS * 4
        if (block_k % _DTL_HALVES) != 0:
            raise ValueError(
                f"direct_to_lds requires block_k % {_DTL_HALVES} == 0 (got {block_k})"
            )
        _dtl_a_chunks = (block_m * block_k) // _DTL_HALVES
        _dtl_b_chunks = (block_n * block_k) // _DTL_HALVES
        _dtl_a_passes = (_dtl_a_chunks + spec.block_size - 1) // spec.block_size
        _dtl_b_passes = (_dtl_b_chunks + spec.block_size - 1) // spec.block_size
        # block_size lanes write block_size * BYTES_PER_LANE bytes per pass
        # to LDS. Single workgroup writes share the LDS base.
        _dtl_pass_bytes = spec.block_size * _DTL_BYTES_PER_LANE

        _dtl_big_bytes = b.const_i32(0x7FFF0000)
        _dtl_a_rsrc = b.buffer_rsrc(A, _dtl_big_bytes)
        _dtl_b_rsrc = b.buffer_rsrc(Bp, _dtl_big_bytes)
        _dtl_a_lds_base = b.smem_addr_of(A_smem)
        _dtl_b_lds_base = b.smem_addr_of(B_smem)
        _dtl_zero_soff = b.const_i32(0)
        # chunks-per-row in halves: block_k halves wide / chunk width
        _dtl_chunks_per_row = block_k // _DTL_HALVES
        _dtl_c_chunks_per_row = b.const_i32(_dtl_chunks_per_row)
        _dtl_c_halves_per_chunk = b.const_i32(_DTL_HALVES)
        _dtl_c_block_size = b.const_i32(spec.block_size)

    def emit_load_phase(A_dst: Value, B_dst: Value, k_off: Value, lds_parity=0) -> None:
        """Coalesced global -> LDS copy for one K tile.

        Driven by :class:`TileWindow`: the A/B global views carry the
        ``(1, K, 1)`` descriptor; the per-call ``k_off`` shifts each
        tile's column origin. ``batch_off_a`` / ``batch_off_b`` ride
        in the batch-axis origin and the descriptor's `mul-by-1`
        folds away in LLVM, yielding identical lowered IR to the
        pre-helpers version.

        ``lds_parity`` (DTLA + prefetch only): 0 writes the first LDS
        half-buffer; 1 writes the second. May be a Python int (0/1) or
        an i32 :class:`Value` for runtime parity. Ignored when prefetch
        is off.
        """
        if spec.trait.direct_to_lds:
            # For each pass, every lane copies one chunk of HALVES halves
            # from global -> LDS via the hardware DTLA/DTLB intrinsic.
            #   chunk_idx_global = tid + pass * block_size
            #   tile_row         = chunk_idx_global / chunks_per_row
            #   tile_col_halves  = (chunk_idx_global % chunks_per_row) * HALVES
            # The LDS destination address advances by pass_bytes per pass.
            #
            # Parity offset (prefetch ping-pong): adds the size of one
            # half-buffer in bytes to the LDS base when writing the second
            # buffer; size = block_m*block_k*2 for A, block_n*block_k*2 for B.
            # ``lds_parity`` may be a Python int (compile-time) or an i32
            # Value (runtime, carried as a scf.for iter-arg).
            _parity_is_value = isinstance(lds_parity, Value)
            a_half_bytes = block_m * block_k * 2
            b_half_bytes = block_n * block_k * 2
            if _prefetch and _parity_is_value:
                # Runtime parity: precompute a single i64 byte offset and
                # add it once to the LDS base per pass.
                _a_par_v = b.zext(b.mul(lds_parity, b.const_i32(a_half_bytes)), _I64)
                _b_par_v = b.zext(b.mul(lds_parity, b.const_i32(b_half_bytes)), _I64)
                a_lds_par_base = b.smem_ptr_add(_dtl_a_lds_base, _a_par_v)
                b_lds_par_base = b.smem_ptr_add(_dtl_b_lds_base, _b_par_v)
            else:
                a_lds_par_base = _dtl_a_lds_base
                b_lds_par_base = _dtl_b_lds_base
            a_parity_bytes_static = (
                lds_parity * a_half_bytes if _prefetch and not _parity_is_value else 0
            )
            b_parity_bytes_static = (
                lds_parity * b_half_bytes if _prefetch and not _parity_is_value else 0
            )
            c2 = b.const_i32(2)
            # Per-wave LDS offset. ``async_buffer_load_lds_addr`` is a
            # wave-level intrinsic: every wave writes
            # ``wave_size * BYTES_PER_LANE`` contiguous bytes starting
            # at the wave-uniform ``lds_dst``. With multiple waves per
            # WG, each wave must target a different LDS slice or they
            # stomp on each other. The block-wide ``chunk_idx`` already
            # uses ``tid``, so wave w covers chunks [w*wave .. (w+1)*wave),
            # which in the LDS layout (block_m x block_k row-major)
            # corresponds to a ``wave_size * BYTES_PER_LANE`` slice
            # starting at ``w * wave_size * BYTES_PER_LANE`` within the
            # per-pass region. We compute the per-wave LDS base once.
            _wave_bytes = spec.wave_size * _DTL_BYTES_PER_LANE
            if t.warp_m * t.warp_n * t.warp_k > 1:
                _warp_id = b.div(tid, c_wave)
                _wave_par_off = b.zext(b.mul(_warp_id, b.const_i32(_wave_bytes)), _I64)
                a_lds_wave_base = b.smem_ptr_add(a_lds_par_base, _wave_par_off)
                b_lds_wave_base = b.smem_ptr_add(b_lds_par_base, _wave_par_off)
            else:
                a_lds_wave_base = a_lds_par_base
                b_lds_wave_base = b_lds_par_base
            for p in range(_dtl_a_passes):
                pass_off_bytes = p * _dtl_pass_bytes + a_parity_bytes_static
                pass_lds_a = (
                    b.smem_ptr_add(
                        a_lds_wave_base, b.zext(b.const_i32(pass_off_bytes), _I64)
                    )
                    if pass_off_bytes > 0
                    else a_lds_wave_base
                )
                chunk_idx = b.add(tid, b.const_i32(p * spec.block_size))
                row = b.div(chunk_idx, _dtl_c_chunks_per_row)
                col_v = b.mod(chunk_idx, _dtl_c_chunks_per_row)
                col = b.mul(col_v, _dtl_c_halves_per_chunk)
                # element offset for A row in halves. With lds_swizzle, load the
                # global element at col^f(row) into the lane-linear LDS slot so
                # the swizzled ds_read (col_base^f(row)) fetches it back (XOR is
                # its own inverse) -- the LDS dest stays wave-contiguous.
                off_elems = b.add(
                    batch_off_a,
                    b.add(
                        b.mul(b.add(block_m_off, row), K),
                        b.add(k_off, _swz_col(col, row)),
                    ),
                )
                off_bytes = b.mul(off_elems, c2)
                b.async_buffer_load_lds_addr(
                    _dtl_a_rsrc,
                    pass_lds_a,
                    off_bytes,
                    _dtl_zero_soff,
                    _DTL_DWORDS,
                    coherency=spec.trait.dtl_cache_a,
                )
            for p in range(_dtl_b_passes):
                pass_off_bytes = p * _dtl_pass_bytes + b_parity_bytes_static
                pass_lds_b = (
                    b.smem_ptr_add(
                        b_lds_wave_base, b.zext(b.const_i32(pass_off_bytes), _I64)
                    )
                    if pass_off_bytes > 0
                    else b_lds_wave_base
                )
                chunk_idx = b.add(tid, b.const_i32(p * spec.block_size))
                row = b.div(chunk_idx, _dtl_c_chunks_per_row)
                col_v = b.mod(chunk_idx, _dtl_c_chunks_per_row)
                col = b.mul(col_v, _dtl_c_halves_per_chunk)
                off_elems = b.add(
                    batch_off_b,
                    b.add(
                        b.mul(b.add(block_n_off, row), K),
                        b.add(k_off, _swz_col(col, row)),
                    ),
                )
                off_bytes = b.mul(off_elems, c2)
                b.async_buffer_load_lds_addr(
                    _dtl_b_rsrc,
                    pass_lds_b,
                    off_bytes,
                    _dtl_zero_soff,
                    _DTL_DWORDS,
                    coherency=spec.trait.dtl_cache_b,
                )
            # The following ``b.sync()`` (in the caller) lowers to
            # ``s_waitcnt vmcnt(0) lgkmcnt(0) ; s_barrier``, which drains
            # the in-flight DTLA writes before any wave reads LDS.
            return

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

        # Double-buffer parity row offset (compv4 non-DTL): the LDS
        # ``store_vec``/``store_scalar`` paths index the smem base
        # directly (origin is not applied for the lds addr space), so we
        # fold the parity offset into the destination ``row`` here. Half 1
        # lands at row block_m / block_n in the [2*block_m, block_k] /
        # [2*block_n, block_k] alloc. ``lds_parity`` is a Python int
        # (compile-time prologue) or an i32 Value (runtime loop iter-arg).
        def _ld_a_row(r: Value) -> Value:
            if not _db:
                return r
            if isinstance(lds_parity, Value):
                return b.add(r, b.mul(lds_parity, b.const_i32(block_m)))
            return b.add(r, b.const_i32(lds_parity * block_m)) if lds_parity else r

        def _ld_b_row(r: Value) -> Value:
            if not _db:
                return r
            if isinstance(lds_parity, Value):
                return b.add(r, b.mul(lds_parity, b.const_i32(block_n)))
            return b.add(r, b.const_i32(lds_parity * block_n)) if lds_parity else r

        # Coalesced (row, col_v, col) decode of a per-thread vec index,
        # shared by the A / B-canonical / B-preshuffle copy loops. Takes
        # the already-built ``vec_idx`` so the emitted SSA matches the
        # inline order verbatim (callers that also need ``vec_idx`` for a
        # global offset compute it before calling).
        def _vec_rc(vec_idx: Value) -> Tuple[Value, Value, Value]:
            row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            return row, col_v, col

        def _pad_k_valid(elem_col: Value) -> Value:
            return b.cmp_lt(b.add(k_off, elem_col), K)

        def _zero_storage_scalar() -> Value:
            return b.cast_f32_to(b.const_f32(0.0), storage_dtype)

        def _mask_storage_scalar(value: Value, valid: Value) -> Value:
            return b.select(valid, value, _zero_storage_scalar())

        def _load_a_pad_k(row: Value, col: Value, n: int) -> Value:
            vals: List[Value] = []
            for i in range(n):
                elem_col = b.add(col, b.const_i32(i)) if i else col
                valid = _pad_k_valid(elem_col)
                safe_col = b.select(valid, elem_col, b.const_i32(0))
                raw = a_global_tile.load_scalar(b, b.const_i32(0), row, safe_col)
                vals.append(_mask_storage_scalar(raw, valid))
            return vals[0] if n == 1 else b.vec_pack(vals, storage_dtype)

        def _load_b_pad_k(row: Value, col: Value, n: int) -> Value:
            vals: List[Value] = []
            for i in range(n):
                elem_col = b.add(col, b.const_i32(i)) if i else col
                valid = _pad_k_valid(elem_col)
                safe_col = b.select(valid, elem_col, b.const_i32(0))
                raw = b_global_tile.load_scalar(b, b.const_i32(0), row, safe_col)
                vals.append(_mask_storage_scalar(raw, valid))
            return vals[0] if n == 1 else b.vec_pack(vals, storage_dtype)

        def _load_preshuffled_b_pad_k(
            base_off: Value, vec_idx: Value, col: Value, n: int
        ) -> Value:
            vals: List[Value] = []
            for i in range(n):
                elem_col = b.add(col, b.const_i32(i)) if i else col
                valid = _pad_k_valid(elem_col)
                raw_off = b.add(
                    base_off, b.add(b.mul(vec_idx, c_load_vec), b.const_i32(i))
                )
                safe_off = b.select(valid, raw_off, base_off)
                raw = b.global_load(Bp, safe_off, storage_dtype)
                vals.append(_mask_storage_scalar(raw, valid))
            return vals[0] if n == 1 else b.vec_pack(vals, storage_dtype)

        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            row, col_v, col = _vec_rc(vec_idx)
            if load_vec == 1:
                a_val = (
                    _load_a_pad_k(row, col, 1)
                    if spec.trait.pad_k
                    else a_global_tile.load_scalar(b, b.const_i32(0), row, col)
                )
                a_lds_tile.store_scalar(
                    b, _ld_a_row(row), _swz_col(col, row), value=a_val
                )
            else:
                a_val = (
                    _load_a_pad_k(row, col, load_vec)
                    if spec.trait.pad_k
                    else a_global_tile.load_vec(b, b.const_i32(0), row, col, n=load_vec)
                )
                a_lds_tile.store_vec(
                    b, _ld_a_row(row), _swz_col(col, row), value=a_val, n=load_vec
                )
        # B-load branch.
        #
        # Canonical (row-major B): each lane reads ``load_vec`` halves at
        # `(n_global_row, k_global_col)`; address arithmetic is strided
        # (one row of B is K halves apart). The wave's lanes hit
        # different rows -> multiple discontiguous VMEM transactions per
        # K-tile.
        #
        # ``preshuffle_b`` (B is pre-shuffled to ``(k_tiles, n_tiles,
        # block_n, block_k)`` per batch by the host): the (block_n x
        # block_k) tile for the current `(n_tile, k_tile)` lives as
        # ``block_n * block_k`` consecutive elements in B. Loading it
        # contiguously into ``B_smem`` (which is row-major
        # ``(block_n, block_k)``) yields **the same** in-LDS layout as
        # the canonical path, so the MFMA inner loop is unchanged. The
        # win is that the per-K global loads collapse to one wide
        # ``buffer_load_dwordxN`` burst per warp.
        if spec.trait.preshuffle_b:
            n_tile_idx = b.div(block_n_off, c_block_n)
            k_tile_idx = b.div(k_off, c_block_k)
            # ceil(N / block_n); pre-condition: caller pads N to a
            # multiple of block_n so the ceiling is N / block_n exactly.
            n_tile_count = b.div(N, c_block_n)
            tile_offset_elements = b.mul(
                b.add(b.mul(k_tile_idx, n_tile_count), n_tile_idx),
                b.const_i32(block_n * block_k),
            )
            base_off = b.add(batch_off_b, tile_offset_elements)
            for e in range(b_vecs_per_thread):
                vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
                glob_off = b.add(base_off, b.mul(vec_idx, c_load_vec))
                row, col_v, col = _vec_rc(vec_idx)
                if load_vec == 1:
                    b_val = (
                        _load_preshuffled_b_pad_k(base_off, vec_idx, col, 1)
                        if spec.trait.pad_k
                        else b.global_load(Bp, glob_off, storage_dtype)
                    )
                    b_lds_tile.store_scalar(
                        b, _ld_b_row(row), _swz_col(col, row), value=b_val
                    )
                else:
                    b_val = (
                        _load_preshuffled_b_pad_k(base_off, vec_idx, col, load_vec)
                        if spec.trait.pad_k
                        else b.global_load_vN(Bp, glob_off, storage_dtype, load_vec)
                    )
                    b_lds_tile.store_vec(
                        b, _ld_b_row(row), _swz_col(col, row), value=b_val, n=load_vec
                    )
        else:
            for e in range(b_vecs_per_thread):
                vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
                row, col_v, col = _vec_rc(vec_idx)
                if load_vec == 1:
                    b_val = (
                        _load_b_pad_k(row, col, 1)
                        if spec.trait.pad_k
                        else b_global_tile.load_scalar(b, b.const_i32(0), row, col)
                    )
                    b_lds_tile.store_scalar(
                        b, _ld_b_row(row), _swz_col(col, row), value=b_val
                    )
                else:
                    b_val = (
                        _load_b_pad_k(row, col, load_vec)
                        if spec.trait.pad_k
                        else b_global_tile.load_vec(
                            b, b.const_i32(0), row, col, n=load_vec
                        )
                    )
                    b_lds_tile.store_vec(
                        b, _ld_b_row(row), _swz_col(col, row), value=b_val, n=load_vec
                    )

    def _emit_frag_smem_load(
        src: Value,
        mn_in_atom: Value,
        k_in_atom: Value,
        atom_mn_base: Value,
        k_tile_base: Value,
        frag_len: int,
    ) -> Value:
        """Load one ``frag_len``-wide operand fragment from a row-major LDS tile.

        Both the A LDS tile ``(block_m, block_k)`` and the B LDS tile
        ``(block_n, block_k)`` are row-major with the M/N index as the row and K
        as the column. The fragment for one lane occupies a single tile row
        (``atom_mn_base + mn_in_atom``) and ``frag_len`` contiguous K columns
        starting at ``k_tile_base + k_in_atom`` — true for both the MFMA and
        WMMA layout maps, whose A/B fragment slots are K-contiguous. fp16/bf16
        smem loads cap at 8 lanes, so a wider fragment (WMMA ``<16 x half>``) is
        assembled from 8-wide chunks.
        """
        lds_row = b.add(atom_mn_base, mn_in_atom)
        lds_col = b.add(k_tile_base, k_in_atom)
        max_vec = 8 if storage_dtype in (F16, BF16) else frag_len
        if frag_len <= max_vec:
            return _emit_smem_load(b, src, lds_row, lds_col, frag_len, storage_dtype)
        frag = None
        for off in range(0, frag_len, max_vec):
            chunk = _emit_smem_load(
                b,
                src,
                lds_row,
                b.add(lds_col, b.const_i32(off)),
                max_vec,
                storage_dtype,
            )
            frag = chunk if frag is None else b.vec_concat(frag, chunk)
        return frag

    def _emit_wmma_phase(
        A_src: Value,
        B_src: Value,
        iter_vars: Sequence[Value],
    ) -> List[Value]:
        """One K-tile of WMMA atoms, fully MMA-contract driven.

        Operand fragments come straight from the op's A/B layout maps and the
        matmul is emitted target-neutrally (the backend picks the WMMA
        intrinsic). The A map yields ``(row, k)`` and the B map ``(k, col)``;
        in both LDS tiles the M/N index is the row and K the column, so we read
        the M-coord from the A map and the N-coord (= ``col``) from the B map.
        """
        a_map = op.a_layout()  # (row, k):   row = lane % 16, k = slot
        b_map = op.b_layout()  # (k, col):   col = lane % 16, k = slot
        a_row_in_atom, a_k_in_atom = a_map.coord(b, lane, 0)
        b_k_in_atom, b_col_in_atom = b_map.coord(b, lane, 0)
        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
        new_accs: List[Value] = list(iter_vars)
        for kk in range(k_atoms):
            k_tile_base = b.const_i32(kk * t.warp_tile_k)
            a_rows = []
            for mi in range(mfmas_m):
                atom_row = b.add(warp_m_off, b.const_i32(mi * t.warp_tile_m))
                a_rows.append(
                    _emit_frag_smem_load(
                        A_src,
                        a_row_in_atom,
                        a_k_in_atom,
                        atom_row,
                        k_tile_base,
                        a_per_lane,
                    )
                )
            b_cols = []
            for ni in range(mfmas_n):
                atom_row = b.add(warp_n_off, b.const_i32(ni * t.warp_tile_n))
                b_cols.append(
                    _emit_frag_smem_load(
                        B_src,
                        b_col_in_atom,
                        b_k_in_atom,
                        atom_row,
                        k_tile_base,
                        b_per_lane,
                    )
                )
            flat = 0
            for mi in range(mfmas_m):
                for ni in range(mfmas_n):
                    new_accs[flat] = _emit_mma(
                        b, op, a_rows[mi], b_cols[ni], new_accs[flat]
                    )
                    flat += 1
        # WMMA intrawave compute schedule (the simple/mem k-loop keeps the LDS
        # store + global load in a separate barrier region, so this region holds
        # only the operand ds_reads + WMMAs). Opt-in via the 'wmma_v1' pipeline.
        if spec.trait.pipeline == "wmma_v1":
            from ...helpers.schedule import SchedulePolicy, WmmaHotLoopInstList

            il = WmmaHotLoopInstList.from_geometry(
                block_size=spec.block_size,
                m_per_block=t.tile_m,
                n_per_block=t.tile_n,
                k_per_block=t.tile_k,
                m_repeat=mfmas_m,
                n_repeat=mfmas_n,
                m_per_wmma=t.warp_tile_m,
                n_per_wmma=t.warp_tile_n,
                k_per_wmma=t.warp_tile_k,
                a_frag_len=a_per_lane,
                b_frag_len=b_per_lane,
                a_dtype_bytes=2,
                b_dtype_bytes=2,
            )
            SchedulePolicy.for_pipeline("wmma_v1").emit_wmma_compute_schedule(b, il)
        return new_accs

    def emit_mfma_phase(
        A_src: Value,
        B_src: Value,
        iter_vars: Sequence[Value],
        lds_parity=0,
    ) -> List[Value]:
        """One K-tile worth of MMAs across all per-warp atom positions
        and every K atom step inside this K-tile.

        On RDNA (WMMA) this delegates to the fully contract-driven
        :func:`_emit_wmma_phase`. On CDNA (MFMA) it keeps the byte-identical
        arch-formula fragment load below; the matmul + accumulator length are
        contract-driven (``_emit_mma`` / ``op.c_frag_len``).

        ``lds_parity`` (DTLA + prefetch only): shifts every A row by
        ``parity * block_m`` and every B row by ``parity * block_n`` so
        the reads target the half-buffer that holds the freshly-loaded
        K-tile. Ignored when prefetch is off. May be a Python int or
        runtime i32 Value.
        """
        if op.family == "wmma":
            return _emit_wmma_phase(A_src, B_src, iter_vars)
        _mp_is_val = isinstance(lds_parity, Value)
        if _prefetch or _db:
            if _mp_is_val:
                a_par_row_v = b.mul(lds_parity, b.const_i32(block_m))
                b_par_row_v = b.mul(lds_parity, b.const_i32(block_n))
                a_row_parity_off = 0  # carried as a Value addition below
                b_row_parity_off = 0
            else:
                a_par_row_v = None
                b_par_row_v = None
                a_row_parity_off = lds_parity * block_m
                b_row_parity_off = lds_parity * block_n
        else:
            a_par_row_v = None
            b_par_row_v = None
            a_row_parity_off = 0
            b_row_parity_off = 0
        # Lane mapping into LDS: A wants per-lane `a_per_lane` K-elements
        # starting at K = k_blk * a_per_lane. The decode is identical for the
        # 16x16 and 32x32 atoms — both put the M-in-atom in ``lane %
        # warp_tile_m``, the N-in-atom in ``lane % warp_tile_n``, and the
        # K-sub-block in ``lane / warp_tile_m`` — so there is no per-shape
        # branch. (The op's A/B layout maps encode the same arithmetic; we
        # keep the inline form here because ``emit_mfma_phase`` also needs the
        # raw ``k_blk`` quotient, which the slot-indexed layout map folds into
        # its ``k`` coordinate.)
        m_in_atom = b.mod(lane, b.const_i32(t.warp_tile_m))
        k_blk = b.div(lane, b.const_i32(t.warp_tile_m))
        n_in_atom = b.mod(lane, b.const_i32(t.warp_tile_n))

        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

        # k_blk * a_per_lane is the lane-local K offset inside the
        # current K-tile; reused across every kk to form ``col_base``.
        # Hoisting saves a mul + add per kk and matches CK Tile's
        # block_gemm where the lane-K base is computed once outside
        # the warp_gemm_dispatcher inner loop.
        k_blk_kbase = b.mul(k_blk, b.const_i32(a_per_lane))

        new_accs: List[Value] = list(iter_vars)

        # ---- per-fragment LDS read helpers (shared by the legacy and PLR
        # emission paths). Each returns one ds_read'd operand fragment for the
        # given K-atom step ``kk`` and atom-row/col index. Pure address arith +
        # one ``_emit_smem_load``; identical math to the inlined legacy form. ----
        def _read_a(kk: int, mi: int) -> Value:
            col_base = b.add(k_blk_kbase, b.const_i32(kk * t.warp_tile_k))
            a_row = b.add(
                warp_m_off,
                b.add(b.const_i32(mi * t.warp_tile_m + a_row_parity_off), m_in_atom),
            )
            if a_par_row_v is not None:
                a_row = b.add(a_row, a_par_row_v)
            return _emit_smem_load(
                b, A_src, a_row, _swz_col(col_base, a_row), a_per_lane, storage_dtype
            )

        def _read_b(kk: int, ni: int) -> Value:
            col_base = b.add(k_blk_kbase, b.const_i32(kk * t.warp_tile_k))
            b_row = b.add(
                warp_n_off,
                b.add(b.const_i32(ni * t.warp_tile_n + b_row_parity_off), n_in_atom),
            )
            if b_par_row_v is not None:
                b_row = b.add(b_row, b_par_row_v)
            return _emit_smem_load(
                b, B_src, b_row, _swz_col(col_base, b_row), b_per_lane, storage_dtype
            )

        def _mma_cluster(a_rows, b_cols) -> None:
            # Reference-faithful MFMA clusters: bracket each m-row's MFMA
            # cluster with s_setprio(1)/(0) + a sched_barrier(0) fence so the
            # post-RA scheduler keeps the matrix pipe at high priority through
            # the cluster and cannot float the surrounding ds_read/buffer_load
            # across it. Enabled with lds_swizzle (the reference-faithful mode).
            for mi in range(mfmas_m):
                if _SWZ and _sched_hints:
                    b.s_setprio(1)
                for ni in range(mfmas_n):
                    flat = mi * mfmas_n + ni
                    new_accs[flat] = _emit_mma(
                        b, op, a_rows[mi], b_cols[ni], new_accs[flat]
                    )
                if _SWZ and _sched_hints:
                    b.s_setprio(0)
                    b.sched_barrier(0)

        for kk in range(k_atoms):
            # We compute one A load per atom-row and reuse across N.
            a_rows = [_read_a(kk, mi) for mi in range(mfmas_m)]
            # B load per atom-col, reused across M.
            b_cols = [_read_b(kk, ni) for ni in range(mfmas_n)]
            _mma_cluster(a_rows, b_cols)

        # Quantitative two-stage HotLoop schedule (E3, E6 L1).
        #
        # The flat per-kk hint (one DS-read group + one all-MFMA group) left
        # ds_write / buffer-load placement entirely to the backend. The
        # comp_v3 / comp_v4 HotLoopScheduler instead pins the whole K-tile's
        # MFMA / ds_read / ds_write / VMEM interleave to the measured MFMA
        # shadow, derived purely from the tile geometry + dtype + the spec
        # atom's timing (``MfmaAtom.mfma_cycle`` / ``k_per_xdlops``). It is
        # issued ONCE per K-tile (matching the C++, which calls
        # ``HotLoopScheduler()`` once per hot iteration) rather than per kk.
        #
        # ``sched_group_barrier`` is a pure scheduling hint -- changing only
        # its operands cannot change the kernel's numeric result, so this is
        # numerically identical to the flat-hint emission.
        if _sched_hints and spec.trait.pipeline in ("compv3", "compv4"):
            _emit_hotloop_schedule(b, spec, load_vec)

        return new_accs

    # ---- active-tile gate ----
    # When ``trait.active_tile_skip`` is on, the K-loop + epilogue
    # are wrapped in ``scf.if(do_work)``. The condition is wave-
    # uniform (CTA-uniform inputs + a single CTA-uniform global
    # load) so the AMDGPU backend collapses the whole if-then to
    # a scalar branch around the body; inactive tiles do one
    # ``buffer_load_dword`` of the bucket head and exit.
    do_work_cond: Optional[Value] = None
    if spec.batched and spec.trait.active_tile_skip:
        # Bucket head index for THIS CTA's (expert, m-tile) slot:
        # ``block_id_z * slot_size + block_m_off``. Using
        # ``block_m_off`` (already chiplet-swizzle-aware and
        # tile_m-aligned) keeps the gate consistent with the actual
        # rows the address arithmetic below will read.
        bucket_head = b.add(b.mul(b.block_id_z(), slot_size_p), block_m_off)
        first_token = b.global_load_i32(sorted_token_ids, bucket_head)
        do_work_cond = b.cmp_ge(first_token, c0)

    def emit_compute_and_epilogue() -> None:
        if _prefetch:
            _emit_kloop_prefetch()
        elif _db:
            _emit_kloop_db()
        else:
            _emit_kloop_simple()
        _emit_epilogue()

    def _emit_kloop_db() -> None:
        """compv4 double-buffered K-loop (non-DTL, VGPR-staged).

        Two LDS half-buffers per operand. Each iteration issues the
        *next* K-tile's global->VGPR->LDS copy into the half the MFMAs
        are not reading, so the global-load + ds_write traffic overlaps
        the current tile's ds_reads + MFMAs. ``parity`` is carried as an
        i32 iter-arg and flips each step.

        Barrier discipline (single barrier per iter, placed at the
        *start*, half ``p`` = current read, ``p^1`` = next write):
          * sync() : drains the previous iter's ds_writes (so half p,
                     loaded last iter, is RAW-safe to read) AND the
                     previous iter's ds_reads (so half p^1 is WAR-safe to
                     overwrite), then barriers all waves.
          * issue load -> half(p^1)   (next tile, overlaps this MFMA)
          * MFMA <- half(p)
        The prologue issues tile 0's load into half 0 (no barrier — the
        loop's first start-sync covers it); the epilogue runs the final
        tile's MFMA from the last-written half after one drain barrier.
        """
        c1_i32 = b.const_i32(1)
        K_minus_one_tile = b.sub(_k_upper, c_block_k)

        # Prologue: issue tile 0's load into half 0. The first iteration's
        # start-sync drains these ds_writes before the MFMA reads half 0.
        # In split-K mode tile 0 is the slice base ``k_lo``.
        emit_load_phase(A_smem, B_smem, k_lo, lds_parity=0)

        loop_args = [("par", c0)] + list(accs)
        for_op = b.scf_for_iter(
            k_lo, K_minus_one_tile, c_block_k, loop_args, iv_name="k0"
        )
        with for_op as (k0, iter_vars):
            parity = iter_vars[0]
            acc_iter = iter_vars[1:]
            next_parity = b.sub(c1_i32, parity)
            k_next = b.add(k0, c_block_k)
            # Start-of-iter barrier: makes half(parity) RAW-safe to read
            # (its load was issued last iter / prologue) and half(next)
            # WAR-safe to overwrite (its reads finished last iter).
            b.sync()
            # Issue next-tile global->VGPR->LDS copy into the other half;
            # this VMEM + ds_write stream overlaps the MFMAs below.
            emit_load_phase(A_smem, B_smem, k_next, lds_parity=next_parity)
            new_accs = emit_mfma_phase(A_smem, B_smem, acc_iter, lds_parity=parity)
            b.scf_yield(next_parity, *new_accs)

        # Epilogue: drain the last in-loop load (it wrote the K-1 tile
        # into half ``final_parity``), barrier, then MFMA that tile.
        final_parity = for_op.results[0]
        b.sync()
        epi_accs = emit_mfma_phase(
            A_smem, B_smem, for_op.results[1:], lds_parity=final_parity
        )
        nonlocal _for_results
        _for_results = epi_accs

    # K-loop upper bound: full ``K`` (byte-identical) or the split-K slice
    # end ``k_hi``. Computed lazily so the non-split path's SSA is unchanged.
    _k_upper = K if k_hi is None else k_hi

    def _emit_kloop_simple() -> None:
        for_op = b.scf_for_iter(k_lo, _k_upper, c_block_k, accs, iv_name="k0")
        with for_op as (k0, iter_vars):
            # Single-buffer, load-then-compute pipeline (``mem``,
            # ``compv3``, and the cshuffle-epilogue ``compv4`` that opts
            # out of AB double-buffering). compv4 with the direct epilogue
            # instead routes through ``_emit_kloop_db``; the scheduler
            # hints in ``emit_mfma_phase`` carry the in-tile interleave.
            emit_load_phase(A_smem, B_smem, k0)
            b.sync()

            new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)

            b.sync()
            b.scf_yield(*new_accs)
        nonlocal _for_results
        _for_results = for_op.results

    def _emit_kloop_prefetch() -> None:
        """Software-pipelined K-loop with DTLA ping-pong.

        Structure:
          prologue:        DTLA load tile 0 -> half 0
          for k in [0, K-block_k) step block_k:
              parity = (k / block_k) & 1   ; next_parity = parity ^ 1
              DTLA load tile k+block_k -> half (parity ^ 1)
              s_waitcnt vmcnt(loads_in_flight_for_next_tile) ; barrier
              MFMA from half parity
              ; no end-barrier — half (parity ^ 1) has its own LDS region
          epilogue:        ; last tile already loaded into the final parity
              s_waitcnt vmcnt(0) ; barrier
              MFMA from final parity

        The post-issue ``s_waitcnt vmcnt(N)`` keeps N loads (= next
        tile's count) in flight while the current tile drains, so the
        loop's MFMA work overlaps the next tile's HBM transfers.
        """
        loads_per_tile = _dtl_a_passes + _dtl_b_passes
        # vmcnt is 6 bits on gfx950 (max 63). If next-tile loads exceed
        # that, we'd saturate and the prefetch buys nothing extra.
        if loads_per_tile > 63:
            # Fall back to the non-prefetched path; the constant would
            # have to be encoded as 63 either way.
            _emit_kloop_simple()
            return

        # Prologue: load tile 0 into half 0 (the slice base in split-K mode).
        emit_load_phase(A_smem, B_smem, k_lo, lds_parity=0)

        # Loop bounds: iterate k in [k_lo, k_hi - block_k), so the last tile
        # is handled by the epilogue (which doesn't need to issue a
        # next-tile load).
        K_minus_one_tile = b.sub(_k_upper, c_block_k)
        # iter_args: (parity_i32, acc...). parity flips each iter.
        c1_i32 = b.const_i32(1)
        loop_args = [("par", c0)] + list(accs)
        for_op = b.scf_for_iter(
            k_lo, K_minus_one_tile, c_block_k, loop_args, iv_name="k0"
        )
        with for_op as (k0, iter_vars):
            parity = iter_vars[0]
            acc_iter = iter_vars[1:]
            next_parity = b.sub(c1_i32, parity)  # 1 - parity (0/1 only)
            k_next = b.add(k0, c_block_k)
            # Single-barrier software pipeline: ONE s_waitcnt + ONE WG barrier
            # per K-tile (vs the prior WAR+RAW two-barrier form, which halved
            # the available MFMA-shadow time at 1 WG/CU). The single barrier
            # serves both hazards because we issue the next-tile write AFTER it:
            #   * vmcnt(0)  -> the current tile's DTL loads (issued last iter
            #                  into half(parity)) have landed: RAW-safe to read.
            #   * lgkmcnt(0)-> the previous iter's ds_reads of half(next_parity)
            #                  have drained: WAR-safe to overwrite that half.
            #   * s_barrier -> WG rendezvous so the freshly-loaded current half
            #                  is visible to every wave before any MFMA reads it.
            # The async next-tile load is issued AFTER the barrier (cannot race
            # the just-drained reads -> no second barrier needed) but BEFORE the
            # MFMAs (its HBM transfer overlaps the matrix work). The prior
            # structure issued that write BEFORE draining, which both raced and
            # forced the extra barrier this collapses away.
            b.s_waitcnt(vmcnt=0, lgkmcnt=0)
            b.s_barrier_bare()
            emit_load_phase(A_smem, B_smem, k_next, lds_parity=next_parity)
            new_accs = emit_mfma_phase(A_smem, B_smem, acc_iter, lds_parity=parity)
            b.scf_yield(next_parity, *new_accs)

        # Epilogue: drain the final tile's loads, rendezvous, MFMA last tile.
        final_parity = for_op.results[0]
        b.s_waitcnt(vmcnt=0, lgkmcnt=0)
        b.s_barrier_bare()
        epi_accs = emit_mfma_phase(
            A_smem, B_smem, for_op.results[1:], lds_parity=final_parity
        )
        nonlocal _for_results
        _for_results = epi_accs

    _for_results: Sequence[Value] = ()

    def _emit_epilogue() -> None:
        # ---- epilogue ----
        fused_ep = getattr(spec, "_fused_epilogue", None)
        if _is_split_k:
            # Split-K: atomic-add each warp's f32 accumulator into the
            # Cf32[M, N] workspace. Reuses the same MFMA acc -> (row, col)
            # scatter as the direct epilogue, but the per-cell write is an
            # f32 atomicrmw fadd instead of a bf16 global store. The caller
            # zero-inits the workspace and casts it to bf16 afterwards.
            _emit_epilogue_split_k(
                b,
                spec,
                _for_results,
                warp_m_idx,
                warp_n_idx,
                lane,
                block_m_off,
                block_n_off,
                M,
                N,
                C,
                c_per_lane,
            )
            return
        if spec.trait.epilogue == "cshuffle":
            _emit_epilogue_cshuffle(
                b,
                spec,
                A_smem,
                _for_results,
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
                fused_epilogue=fused_ep,
            )
        else:
            _emit_epilogue_default(
                b,
                spec,
                op,
                _for_results,
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
                fused_epilogue=fused_ep,
            )

    if do_work_cond is None:
        emit_compute_and_epilogue()
    else:
        with b.scf_if(do_work_cond):
            emit_compute_and_epilogue()

    return b.kernel


# ---------------------------------------------------------------------
# Epilogues
# ---------------------------------------------------------------------


def _emit_mfma_acc_scatter(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    lane: Value,
    accs: Sequence[Value],
    m_base_off: Value,
    n_base_off: Value,
    c_per_lane: int,
    storage_dtype: Type,
    per_cell,
    *,
    n_base_first: bool = False,
) -> None:
    """Walk an MFMA warp tile's per-lane accumulator slots in output order.

    This is the single source of the MFMA accumulator -> (row, col) scatter
    that the *default* (direct global store) and the *cshuffle* (LDS staging)
    epilogues both need. It reproduces the canonical AMD MFMA output layout
    (``MfmaAtom.lane_to_output`` / the ``c_layout`` map in
    :mod:`rocke.core.arch`) with the same hoisting the two epilogues used to
    open-code in four near-identical blocks (default-16/32, cshuffle-16/32):

    * 16x16 atom: ``row = m_blk*c_per_lane + i``, ``col = n_in_atom``.
    * 32x32 atom: ``row = (i//4)*8 + m_blk*4 + (i%4)``, ``col = n_in_atom``,
      where the per-slot ``(i//4)*8 + (i%4)`` ramp is folded to one host
      constant exactly as before.

    ``m_base_off`` / ``n_base_off`` are the warp-tile base offsets the caller
    already holds: the default epilogue passes ``block + warp`` offsets (the
    store targets global), the cshuffle epilogue passes ``warp`` offsets (the
    block offset is added later at the wide global store). The per-cell
    callback ``per_cell(c_m, c_n, acc_h, i)`` owns the actual write.

    ``n_base_first`` selects the column-offset association so each call site's
    prior IR is reproduced: the default epilogue grouped it as
    ``n_base + (ni*wtn + n_in_atom)`` (``False``); the cshuffle epilogue
    grouped it as ``(n_base + ni*wtn) + n_in_atom`` (``True``). Both are
    arithmetically equal; the flag only chooses the column grouping.

    The MFMA C-fragment ``(row_in_atom, col_in_atom)`` decode is now sourced
    from the atom's CK-Tile ``CWarpDstrEncoding`` (a
    :class:`~rocke.helpers.distribution.TileDistribution`) instead of the
    open-coded ``m_blk``/``lane_m``/``row_ramp`` lane arithmetic. The single
    lane is split into the distribution's P sub-sequence ``[m_blk, n_in_atom]
    = [lane // kCNLane, lane % kCNLane]`` and the per-lane accumulator slot
    ``i`` into the two Y dims ``(i // kCM1PerLane, i % kCM1PerLane)``;
    ``calculate_x`` then yields ``row_in_atom = lane_m + row_ramp`` and
    ``col_in_atom = n_in_atom``, identical to the prior formulas for every
    supported atom (16x16 and 32x32). This is Tier-2: the same div/mod and
    add/mul values are produced, but the SSA emission order differs from the
    hand-hoisted form.
    """
    from ...helpers.atoms import c_warp_params, make_c_warp_dstr_encoding, mfma_atom
    from ...helpers.distribution import make_static_tile_distribution

    t = spec.tile
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    # The CWarpDstrEncoding is keyed on the warp-tile (m, n) only (the lane/
    # slot -> (row, col) layout is dtype-independent), so resolve the atom by
    # the spec's input dtype + warp-tile shape.
    atom = mfma_atom(
        spec.data.dtype_a,
        t.warp_tile_m,
        t.warp_tile_n,
        t.warp_tile_k,
    )
    _, _kc_mlane, kc_m1, kc_nlane = c_warp_params(atom)
    c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom))

    c_nlane = b.const_i32(kc_nlane)
    # Single P (the lane) -> [m_blk, n_in_atom] = [lane // kCNLane, lane %
    # kCNLane]. n_in_atom is also the column coordinate (col_in_atom).
    n_in_atom = b.mod(lane, c_nlane)
    m_blk = b.div(lane, c_nlane)
    p_lane = [m_blk, n_in_atom]

    # Per accumulator slot ``i`` the two Y dims decompose row-major over
    # (kCM0PerLane, kCM1PerLane): y0 = i // kCM1PerLane, y1 = i % kCM1PerLane.
    # calculate_x reconstructs the full row_in_atom (lane + slot) and the
    # col_in_atom (= n_in_atom) from these.
    row_in_atom: List[Value] = []
    col_in_atom: Optional[Value] = None
    for i in range(c_per_lane):
        ys = [b.const_i32(i // kc_m1), b.const_i32(i % kc_m1)]
        x_row, x_col = c_dist.calculate_x(b, ys=ys, ps=[p_lane])
        row_in_atom.append(x_row)
        col_in_atom = x_col  # constant across i (depends only on n_in_atom)

    flat = 0
    for mi in range(mfmas_m):
        base_m = b.add(m_base_off, b.const_i32(mi * t.warp_tile_m))
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            acc_h = b.vec_cast_f32_to(acc, storage_dtype)
            if n_base_first:
                c_n = b.add(
                    b.add(n_base_off, b.const_i32(ni * t.warp_tile_n)),
                    col_in_atom,
                )
            else:
                c_n = b.add(
                    n_base_off,
                    b.add(b.const_i32(ni * t.warp_tile_n), col_in_atom),
                )
            for i in range(c_per_lane):
                c_m = b.add(base_m, row_in_atom[i])
                per_cell(c_m, c_n, acc_h, i)


def _emit_epilogue_default(
    b: IRBuilder,
    spec: UniversalGemmSpec,
    op,
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
    fused_epilogue=None,  # Optional[FusedEpilogue] from helpers.fuse
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
    storage_dtype = _storage_dtype(spec)
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n
    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    # A fused epilogue (BiasAdd, ReLU, residual, ...) declares its extra
    # kernel params once before the scatter, then transforms each per-lane
    # scalar at its (c_m, c_n) below. Guarded by ``None`` so the matmul-only
    # default path (and the byte-identical CDNA MFMA store) is unchanged.
    if fused_epilogue is not None:
        fused_epilogue.declare_params(b)
        record_runtime = getattr(fused_epilogue, "record_runtime", None)
        if record_runtime is not None:
            record_runtime(b, N=N)

    def _store_masked(c_m: Value, c_n: Value, c_off: Value, h: Value) -> None:
        """Per-element global store with optional ``pad_m`` / ``pad_n`` guard.

        Lane-level scatter does not benefit from the vec-aligned check
        used in the cshuffle path; per-element ``c_n < N`` is the
        natural mask here and matches what the partially-tiled output
        column requires.
        """
        if not (pad_m or pad_n):
            b.global_store(C, c_off, h, align=2)
            return
        checks: List[Value] = []
        if pad_m:
            checks.append(b.cmp_lt(c_m, M))
        if pad_n:
            checks.append(b.cmp_lt(c_n, N))
        in_bounds = checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])
        with b.scf_if(in_bounds):
            b.global_store(C, c_off, h, align=2)

    # ---- WMMA (RDNA) accumulator scatter: fully MMA-contract driven. ----
    # The WMMA accumulator distributes the M x N tile across wave32 lanes
    # differently from MFMA, so we ask the op's accumulator layout map for the
    # (row, col) of every per-lane slot rather than hard-coding the lane math.
    # One slot -> one f16 store. (The supported WMMA subset is the single
    # 16x16x16 atom -> mfmas_m == mfmas_n == 1.)
    if op.family == "wmma":
        c_map = op.c_layout()
        block_warp_m_off = b.add(block_m_off, warp_m_off)
        block_warp_n_off = b.add(block_n_off, warp_n_off)
        flat = 0
        for mi in range(mfmas_m):
            atom_m = b.add(block_warp_m_off, b.const_i32(mi * t.warp_tile_m))
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                atom_n = b.add(block_warp_n_off, b.const_i32(ni * t.warp_tile_n))
                acc_h = b.vec_cast_f32_to(acc, storage_dtype)
                for i in range(c_per_lane):
                    row_in_atom, col_in_atom = c_map.coord(b, lane, i)
                    c_m = b.add(atom_m, row_in_atom)
                    c_n = b.add(atom_n, col_in_atom)
                    c_off = b.add(b.mul(c_m, N), c_n)
                    if batch_off_c is not None:
                        c_off = b.add(batch_off_c, c_off)
                    h = b.vec_extract(acc_h, i)
                    if fused_epilogue is not None:
                        h = fused_epilogue.apply_scalar(b, h, c_m, c_n)
                    _store_masked(c_m, c_n, c_off, h)
        return

    # Compile-time invariants that do not depend on (mi, ni, i).
    # Hoisting them avoids re-emitting the same add chain per
    # accumulator slot; the IR's constant folder collapses them
    # downstream, but building them once keeps the IR small and the
    # lowered LLVM readable. The accumulator -> (row, col) scatter (16x16
    # and 32x32 layouts, with the per-mi base hoist + the single-constant
    # per-slot ramp) lives in :func:`_emit_mfma_acc_scatter` so the default
    # and cshuffle epilogues share one copy of the MFMA output-layout math.
    block_warp_m_off = b.add(block_m_off, warp_m_off)
    block_warp_n_off = b.add(block_n_off, warp_n_off)

    def _store_cell(c_m: Value, c_n: Value, acc_h: Value, i: int) -> None:
        c_off = b.add(b.mul(c_m, N), c_n)
        if batch_off_c is not None:
            c_off = b.add(batch_off_c, c_off)
        h = b.vec_extract(acc_h, i)
        if fused_epilogue is not None:
            h = fused_epilogue.apply_scalar(b, h, c_m, c_n)
        _store_masked(c_m, c_n, c_off, h)

    _emit_mfma_acc_scatter(
        b,
        spec,
        lane,
        accs,
        block_warp_m_off,
        block_warp_n_off,
        c_per_lane,
        storage_dtype,
        _store_cell,
    )


def _emit_epilogue_split_k(
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
    Cf32: Value,
    c_per_lane: int,
) -> None:
    """Split-K atomic-add epilogue.

    Each warp owns a per-lane ``<c_per_lane x f32>`` accumulator that is the
    partial product over this CTA's K-slice. We scatter every slot to its
    output ``(c_m, c_n)`` (the canonical MFMA layout, identical to the direct
    epilogue) and atomic-add the raw f32 value into ``Cf32[c_m, c_n]``. The
    split-K reduction across the ``split_k`` CTAs that share a ``(m_tile,
    n_tile)`` converges to the full f32 GEMM; the caller casts the workspace
    to the target dtype.

    The MFMA C-fragment ``(row_in_atom, col_in_atom)`` decode reuses the
    atom's ``CWarpDstrEncoding`` distribution exactly as
    :func:`_emit_mfma_acc_scatter` does, so the scattered output coords match
    the direct epilogue cell-for-cell -- only the per-cell write differs
    (f32 atomicrmw fadd vs bf16 global store).
    """
    from ...helpers.atoms import c_warp_params, make_c_warp_dstr_encoding, mfma_atom
    from ...helpers.distribution import make_static_tile_distribution

    t = spec.tile
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))
    block_warp_m_off = b.add(block_m_off, warp_m_off)
    block_warp_n_off = b.add(block_n_off, warp_n_off)

    atom = mfma_atom(spec.data.dtype_a, t.warp_tile_m, t.warp_tile_n, t.warp_tile_k)
    _, _kc_mlane, kc_m1, kc_nlane = c_warp_params(atom)
    c_dist = make_static_tile_distribution(make_c_warp_dstr_encoding(atom))

    c_nlane = b.const_i32(kc_nlane)
    n_in_atom = b.mod(lane, c_nlane)
    m_blk = b.div(lane, c_nlane)
    p_lane = [m_blk, n_in_atom]

    row_in_atom: List[Value] = []
    col_in_atom: Optional[Value] = None
    for i in range(c_per_lane):
        ys = [b.const_i32(i // kc_m1), b.const_i32(i % kc_m1)]
        x_row, x_col = c_dist.calculate_x(b, ys=ys, ps=[p_lane])
        row_in_atom.append(x_row)
        col_in_atom = x_col

    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    flat = 0
    for mi in range(mfmas_m):
        base_m = b.add(block_warp_m_off, b.const_i32(mi * t.warp_tile_m))
        for ni in range(mfmas_n):
            acc = accs[flat]
            flat += 1
            c_n = b.add(
                block_warp_n_off,
                b.add(b.const_i32(ni * t.warp_tile_n), col_in_atom),
            )
            for i in range(c_per_lane):
                c_m = b.add(base_m, row_in_atom[i])
                c_off = b.add(b.mul(c_m, N), c_n)
                val = b.vec_extract(acc, i)
                checks: List[Value] = []
                if pad_m:
                    checks.append(b.cmp_lt(c_m, M))
                if pad_n:
                    checks.append(b.cmp_lt(c_n, N))
                if checks:
                    in_bounds = (
                        checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])
                    )
                    with b.scf_if(in_bounds):
                        b.global_atomic_add_f32(Cf32, c_off, val)
                else:
                    b.global_atomic_add_f32(Cf32, c_off, val)


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
    fused_epilogue=None,  # Optional[FusedEpilogue] from helpers.fuse
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
    storage_dtype = _storage_dtype(spec)
    mfmas_m = t.mfmas_per_warp_m
    mfmas_n = t.mfmas_per_warp_n

    # If a fused epilogue is attached (e.g. BiasAdd, ReLU), give it a
    # chance to declare any extra kernel params (bias pointer, …). It
    # caches the SSA values internally so apply_vec / apply_scalar
    # can reach them at the per-element transform site below. The
    # ``record_runtime`` call lets residual-style ops use the
    # contiguous-(M, N) layout's row stride (= N) without an extra
    # i32 kernel param.
    if fused_epilogue is not None:
        fused_epilogue.declare_params(b)
        record_runtime = getattr(fused_epilogue, "record_runtime", None)
        if record_runtime is not None:
            record_runtime(b, N=N)

    # LDS staging tile: tile_m x tile_n of output storage dtype.
    Cs = b.smem_alloc(storage_dtype, [t.tile_m, t.tile_n], name_hint="C_smem")

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * t.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * t.warp_tile_n))

    # ---- step 1+2: warp accs -> LDS at the MFMA layout. ----
    #
    # For both 16x16 and 32x32 we hoist the per-tile (warp + mi *
    # warp_tile_m, warp + ni * warp_tile_n + n_in_atom) bases outside
    # the per-element loop. The MFMA output layout pins consecutive
    # lanes to consecutive N columns and scatters the per-lane
    # c_per_lane slots across M rows, so the smem store itself stays
    # scalar (n=1). What we can hoist is the address arithmetic:
    # every (mi, ni, i) re-derived the warp + mi offset in the
    # original form.
    def _smem_cell(ld_m: Value, ld_n: Value, acc_h: Value, i: int) -> None:
        h = b.vec_extract(acc_h, i)
        b.smem_store_vN(Cs, [ld_m, ld_n], h, n=1)

    # Same MFMA accumulator -> (row, col) layout as the default epilogue, but
    # the base offsets are warp-relative (the block offset is applied at the
    # wide global store in step 4) and each cell writes to the LDS staging
    # tile instead of global. The shared scatter keeps the 16x16 / 32x32 row
    # math + hoisting in one place.
    _emit_mfma_acc_scatter(
        b,
        spec,
        lane,
        accs,
        warp_m_off,
        warp_n_off,
        c_per_lane,
        storage_dtype,
        _smem_cell,
        n_base_first=True,
    )

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
    pad_m = bool(spec.trait.pad_m)
    pad_n = bool(spec.trait.pad_n)

    # Coalesced (row, col_v, col) decode of a per-thread vec index for the
    # wide C store. Takes the already-built ``vec_idx`` so the emitted SSA
    # matches the inline order verbatim.
    def _vec_rc(vec_idx: Value) -> Tuple[Value, Value, Value]:
        row = b.div(vec_idx, c_tile_n_div_vec)
        col_v = b.mod(vec_idx, c_tile_n_div_vec)
        col = b.mul(col_v, b.const_i32(store_vec)) if store_vec > 1 else col_v
        return row, col_v, col

    for e in range(vecs_per_thread):
        vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
        row, col_v, col = _vec_rc(vec_idx)

        c_m = b.add(block_m_off, row)
        c_n = b.add(block_n_off, col)
        c_off = b.add(b.mul(c_m, N), c_n)
        if batch_off_c is not None:
            c_off = b.add(batch_off_c, c_off)

        # Out-of-bounds guard: when ``pad_m`` / ``pad_n`` is set the
        # caller may launch with grid dims that round up past the
        # logical ``M`` / ``N``. The cshuffle staging is sized to the
        # full tile (always in-bounds for LDS), but the global store
        # must skip rows / columns past the logical extent. For
        # non-vector-aligned N tails we fall back to element-granular
        # guarded stores so valid columns at the start of the final
        # vector are not dropped.
        in_bounds: Optional[Value] = None
        if pad_m or pad_n:
            checks: List[Value] = []
            if pad_m:
                checks.append(b.cmp_lt(c_m, M))
            if pad_n:
                if store_vec == 1:
                    checks.append(b.cmp_lt(c_n, N))
                else:
                    c_n_last = b.add(c_n, b.const_i32(store_vec - 1))
                    checks.append(b.cmp_lt(c_n_last, N))
            in_bounds = checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])

        if store_vec == 1:
            h = _load_smem_scalar(b, Cs, row, col, storage_dtype)
            if fused_epilogue is not None:
                # Treat the scalar as a length-1 vector for op uniformity.
                # The fused epilogue may transform the value (bias-add,
                # activation, scale, ...); we then unpack back to scalar
                # before the global store.
                h = fused_epilogue.apply_scalar(b, h, c_m, c_n)
            if in_bounds is not None:
                with b.scf_if(in_bounds):
                    b.global_store(C, c_off, h, align=2)
            else:
                b.global_store(C, c_off, h, align=2)
        else:
            hv = _load_smem_vec(b, Cs, row, col, store_vec, storage_dtype)
            if pad_n:
                for i in range(store_vec):
                    c_n_i = b.add(c_n, b.const_i32(i)) if i else c_n
                    c_off_i = b.add(c_off, b.const_i32(i)) if i else c_off
                    h = b.vec_extract(hv, i)
                    if fused_epilogue is not None:
                        h = fused_epilogue.apply_scalar(b, h, c_m, c_n_i)
                    checks: List[Value] = []
                    if pad_m:
                        checks.append(b.cmp_lt(c_m, M))
                    checks.append(b.cmp_lt(c_n_i, N))
                    elem_in_bounds = (
                        checks[0] if len(checks) == 1 else b.land(checks[0], checks[1])
                    )
                    with b.scf_if(elem_in_bounds):
                        b.global_store(C, c_off_i, h, align=2)
            else:
                if fused_epilogue is not None:
                    hv = fused_epilogue.apply_vec(b, hv, c_m, c_n, n_elems=store_vec)
                if in_bounds is not None:
                    with b.scf_if(in_bounds):
                        b.global_store_vN(C, c_off, hv, store_vec)
                else:
                    b.global_store_vN(C, c_off, hv, store_vec)


def _load_smem_scalar(
    b: IRBuilder, smem: Value, row: Value, col: Value, dtype: Type
) -> Value:
    # We expose vector loads but not a scalar half load from smem yet;
    # the v=1 path is rare. Emit as a 2-half vector and extract index 0.
    v = b.smem_load_vN(smem, row, col, dtype=dtype, n=2)
    return b.vec_extract(v, 0)


def _load_smem_vec(
    b: IRBuilder, smem: Value, row: Value, col: Value, n: int, dtype: Type
) -> Value:
    if dtype == F16 and n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN(smem, row, col, dtype=dtype, n=n)


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
    name_prefix: str = "rocke_universal",
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
