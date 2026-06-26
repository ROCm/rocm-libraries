# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Implicit-GEMM convolution kernel instance (NHWC × KYXC -> NHWK).

This is the standard implicit-GEMM convolution problem authored entirely
through the rocke IR + coordinate-transform DAG. It mirrors how CK
Tile expresses convolution (`tensor_descriptor` with
`unmerge_transform` for `(m -> n, ho, wo)`, `embed_transform` for
`(ho, y -> hi)`, and `pad_transform` for boundary checks), and the
GEMM body is the same compv4-style pipeline we use for square GEMM
in `gemm_universal.py`. The only kernel-authoring change vs GEMM is
how the A tile's address is computed.

Authoring style (this is what the kernel writer types):

    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(N=8, Hi=56, Wi=56, C=64,
                            K=64, Y=3, X=3,
                            sH=1, sW=1, pH=1, pW=1, dH=1, dW=1),
        tile_m=64, tile_n=64, tile_k=64,
        warp_m=2, warp_n=2,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    )
    kernel = build_implicit_gemm_conv(spec)

Internally, A's per-element address is computed via a transform DAG:

    A_desc = TensorDescriptor.naive('A', [N, Hi, Wi, C], coord_names=
        ['n', 'hi', 'wi', 'c'])
        .transform(unmerge('m', into=['n','ho','wo'], dims=[N,Ho,Wo]))
        .transform(embed(['ho','y'], 'hi', strides=(sH,dH), offset=-pH,
                          lo=0, hi=Hi))
        .transform(embed(['wo','x'], 'wi', strides=(sW,dW), offset=-pW,
                          lo=0, hi=Wi))
        .transform(unmerge('k', into=['y','x','c'], dims=[Y,X,C]))

At every per-thread A-tile load we call `A_desc.offset(b, m=m_val,
k=k_val)` which emits the bounds-checked NHWC offset for that
implicit-GEMM (m, k) point — the same SSA dataflow a hand-written
implicit-GEMM kernel would compute.

B (KYXC weight) and D (NHWK output) use naive descriptors with extra
`unmerge` transforms for the output store (so the epilogue can write
NHWK from MFMA's (m_in_tile, n_in_tile) layout).

Target: CK Tile's `cktile_fixed_lean` reference for this shape
(`N=8, Hi=Wi=56, C=K=64, Y=X=3`) reaches ~250 TFLOPS in CUDA-graph
mode. We aim to beat that on the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from typing import Callable, List, Optional, Sequence, Tuple

from ...core.ir import (
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
from ...helpers.atoms import MfmaAtom, mfma_atom
from ...helpers.epilogues import CShuffleEpilogue, DirectEpilogue
from ...helpers.geometry import WarpGrid
from ...helpers.layouts import LdsLayout
from ...helpers.loads import AsyncTileLoader, CoalescedTileLoader
from ...helpers.mfma_gemm_inner import decode_mfma_lanes
from ...helpers.pipeline import SoftwarePipeline
from ...helpers.schedule import SchedulePolicy
from ...helpers.spec import choose_load_vec
from ...helpers.tensor_view import (
    make_buffer_resource,
)
from ...helpers.transforms import TensorDescriptor, embed, pad, unmerge_magic


_DTYPE_TO_IR: dict = {"f16": F16, "fp16": F16, "bf16": BF16, "fp32": F32}


def _ir_dtype(dtype: str) -> Type:
    """Map a dtype string (``"fp16"``, ``"bf16"``, ``"fp32"``) to an IR ``Type``."""
    t = _DTYPE_TO_IR.get(dtype)
    if t is None:
        raise ValueError(
            f"unsupported conv dtype {dtype!r}; choose fp16, bf16, or fp32"
        )
    return t


# ---------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ConvDataSpec:
    """Element / accumulator dtype choice for the conv kernel.

    Layouts:
      A: NHWC, dtype_a (input activations)
      B: KYXC, dtype_b (weights)
      D: NHWK, dtype_d (output)
      accumulator: dtype_acc (always fp32)
    """

    dtype_a: str = "fp16"
    dtype_b: str = "fp16"
    dtype_d: str = "fp16"
    dtype_acc: str = "fp32"


@dataclass(frozen=True)
class ConvProblem:
    """2-D or 3-D convolution shape parameters.

    2-D (default — ``Di`` / ``Z`` are ``None``):
      A: NHWC,  shape ``[N, Hi, Wi, C]``
      B: KYXC, shape ``[K, Y, X, C]``
      D: NHWK,  shape ``[N, Ho, Wo, K]``
      M = N*Ho*Wo,  N_gemm = K,  K_gemm = Y*X*C

     3-D (set ``Di`` and ``Z``):
       A: NDHWC, shape ``[N, Di, Hi, Wi, C]``
       B: KZYXC, shape ``[K, Z, Y, X, C]``
       D: NDHWK, shape ``[N, Do, Ho, Wo, K]``
       M = N*Do*Ho*Wo,  N_gemm = K,  K_gemm = Z*Y*X*C
    """

    N: int
    Hi: int
    Wi: int
    C: int
    K: int
    Y: int
    X: int
    sH: int = 1
    sW: int = 1
    pH: int = 0
    pW: int = 0
    dH: int = 1
    dW: int = 1
    # 3-D-only fields; leave as None for 2-D convolutions.
    Di: Optional[int] = None
    Z: Optional[int] = None
    sD: Optional[int] = None
    pD: Optional[int] = None
    dD: Optional[int] = None

    def __post_init__(self) -> None:
        depth = (self.Di, self.Z, self.sD, self.pD, self.dD)
        any_set = any(v is not None for v in depth)
        all_set = all(v is not None for v in depth)
        if any_set and not all_set:
            raise ValueError(
                "3-D ConvProblem requires Di, Z, sD, pD, dD (set all or leave all as None)"
            )

    @property
    def is_3d(self) -> bool:
        return self.Di is not None

    # ---- depth spatial output (3-D only) ----
    @property
    def Do(self) -> Optional[int]:
        if not self.is_3d:
            return None
        return (self.Di + 2 * self.pD - self.dD * (self.Z - 1) - 1) // self.sD + 1

    @property
    def Ho(self) -> int:
        return (self.Hi + 2 * self.pH - self.dH * (self.Y - 1) - 1) // self.sH + 1

    @property
    def Wo(self) -> int:
        return (self.Wi + 2 * self.pW - self.dW * (self.X - 1) - 1) // self.sW + 1

    @property
    def M(self) -> int:
        base = self.N * self.Ho * self.Wo
        return base * self.Do if self.is_3d else base

    @property
    def N_gemm(self) -> int:
        return self.K

    @property
    def K_gemm(self) -> int:
        z = self.Z if self.is_3d else 1
        return z * self.Y * self.X * self.C

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N_gemm * self.K_gemm

    def short(self) -> str:
        if self.is_3d:
            return (
                f"N{self.N}D{self.Di}H{self.Hi}W{self.Wi}C{self.C}"
                f"_K{self.K}Z{self.Z}Y{self.Y}X{self.X}"
            )
        return f"N{self.N}H{self.Hi}W{self.Wi}C{self.C}_K{self.K}Y{self.Y}X{self.X}"


@dataclass(frozen=True)
class ConvAccumulatorEpilogue:
    """Static fp32 accumulator transform applied before the conv store.

    This is intentionally narrower than ``helpers.fuse.FusedEpilogue``:
    it runs directly on MFMA accumulator fragments inside the hand-authored
    conv instance. The default is identity, preserving the historical conv IR.
    """

    bias: float = 0.0
    scale: float = 1.0
    relu: bool = False
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None

    def is_identity(self) -> bool:
        return (
            self.bias == 0.0
            and self.scale == 1.0
            and not self.relu
            and self.clamp_min is None
            and self.clamp_max is None
        )

    def tag(self) -> str:
        if self.is_identity():
            return ""
        pieces: List[str] = []
        if self.bias != 0.0:
            pieces.append(f"bias{self.bias:g}")
        if self.scale != 1.0:
            pieces.append(f"scale{self.scale:g}")
        if self.relu:
            pieces.append("relu")
        if self.clamp_min is not None or self.clamp_max is not None:
            lo = "-inf" if self.clamp_min is None else f"{self.clamp_min:g}"
            hi = "inf" if self.clamp_max is None else f"{self.clamp_max:g}"
            pieces.append(f"clamp{lo}to{hi}")
        return "epi_" + "_".join(pieces)


@dataclass(frozen=True)
class ImplicitGemmConvSpec:
    """One concrete implicit-GEMM convolution kernel configuration.

    The geometry conventions match `gemm_universal.UniversalGemmSpec`:
      - `tile_m x tile_n x tile_k` is the per-block tile.
      - `warp_m x warp_n` is the warp grid.
      - `warp_tile_m x warp_tile_n x warp_tile_k` is the MFMA atom.

    Pipeline options (mirror CK's compv4 family):
      - `pipeline="mem"`      : single-buffer LDS, no scheduler hints
      - `pipeline="compv3"`   : single-buffer LDS + sched_group_barrier
                                interleave hints (overlap MFMA/DS_read)
      - `pipeline="compv4"`   : double-buffer LDS (ping-pong A_smem/B_smem)
                                + sched hints + s_setprio to push the K-loop
                                into compute steady state

    Epilogue options:
      - `epilogue="default"`  : per-lane scalar fp16 stores via the D
                                descriptor; correctness-first.
      - `epilogue="cshuffle"` : LDS-stage the accumulators in MFMA layout,
                                then re-read in coalesced (row-major)
                                layout for wide-vector global stores
                                (runbook §9.3 — the single largest perf
                                lever once the K loop is bandwidth-bound).

    Memory-pipeline options:
      - `async_dma=False`     : (default) classic
                                `buffer_load_vN_f16 -> register ->
                                smem_store_vN_f16` path. Adds K-padded
                                LDS (`K_pad = block_k + 8`) to avoid
                                ds_read bank conflicts.
      - `async_dma=True`      : direct DRAM->LDS via
                                `raw_ptr_buffer_load_lds` (runbook §6.3).
                                The intrinsic writes lane-contiguous LDS,
                                so the LDS layout must be plain
                                `[block_m, block_k]` (no K-pad). Consumers
                                still emit standard 2D ds_reads; LDS
                                bank-conflict avoidance moves into the
                                consumer's address arithmetic (XOR
                                swizzle) if it becomes the next
                                bottleneck. Place `s_waitcnt(vmcnt=0)`
                                before the MFMA phase.
      - `lds_k_pad=None`      : default LDS K-stride policy. Sync path
                                uses `+8` when `block_k >= 16`; async
                                path uses `+0` because
                                `raw_ptr_buffer_load_lds` writes a
                                lane-contiguous packed tile. Set this
                                explicitly to tune or isolate LDS bank
                                conflict effects in sweeps.
    """

    problem: ConvProblem
    name: str = "conv_igemm"
    data: ConvDataSpec = field(default_factory=ConvDataSpec)

    tile_m: int = 64
    tile_n: int = 64
    # tile_k=64 + the 32x32x16 atom is the measured-best default on gfx950:
    # for the bake-off shape (N8 C64 K64) it ties the old tk128/16x16x32
    # default, and for compute-bound shapes (e.g. N16 C256 K256) it is ~1.4x
    # faster (619 vs 445 TFLOPS, gfx950). All shipped callers (bake_off,
    # tests, probes, hip_lowering_parity, verify_dsl_docs) already override to
    # this config; the stale tk128/16x16x32 default only penalised callers who
    # relied on the dataclass defaults.
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
    unroll_k: bool = False  # NEW: Clean Python-level K-loop unrolling
    lds_k_pad: Optional[int] = None
    # Per-operand vector widths (elements). ``None`` means auto-select via the
    # shared ``choose_load_vec`` / ``CShuffleEpilogue.from_grid`` heuristics.
    vector_size_a: Optional[int] = None
    vector_size_b: Optional[int] = None
    vector_size_c: Optional[int] = None
    lds_layout: Optional[LdsLayout] = None
    # Chiplet-aware grid swizzle (multi-XCD L2 locality). When True,
    # the kernel flattens its 2D blockIdx into a linear WGID, runs it
    # through ``chiplet_aware_super_tile`` (compile-time variant — conv
    # tile counts are derived from the problem shape so they are known
    # statically), and uses the remapped (pid_m, pid_n) for tile offsets.
    chiplet_swizzle: bool = False
    chiplet_wgm: int = 8
    chiplet_num_xcds: int = 8
    chiplet_chunk_size: int = 64
    # AMDGPU occupancy hint: emits ``amdgpu-waves-per-eu`` on the
    # kernel attribute list. ``None`` keeps the backend's default.
    waves_per_eu: Optional[int] = None
    # P87: K0/K1 split for implicit-GEMM conv. Set ``k0_k1_split=True``
    # to drive :class:`rocke.helpers.loads.CoalescedTileLoader`'s
    # ``inner_dim`` parameter (P33) so the loader processes whole C
    # rows contiguously and the MFMA loop iterates ``kk`` over K1
    # only. ``None`` (default) keeps the legacy flat-K behaviour.
    k0_k1_split: bool = False
    # P86: grouped convolution. ``groups > 1`` uses the descriptor
    # DAG's ``unmerge('group', into=...)`` to recover the per-group
    # `(C/groups, K/groups)` slabs so each group's GEMM stays
    # within its own slab. The implementation mirrors CK Tile's
    # ``GroupedConvolutionForward`` shape; for ``groups == 1`` (the
    # default) the descriptor reduces to the flat-conv form and the
    # kernel emits the same code as before.
    groups: int = 1
    # Static accumulator epilogue used by the gfx950 deep-fusion prototype.
    # It composes simple fp32 VALU transforms directly on MFMA accumulator
    # fragments before the existing direct/cshuffle store path.
    acc_epilogue: ConvAccumulatorEpilogue = ConvAccumulatorEpilogue()

    @property
    def block_size(self) -> int:
        return self.warp_m * self.warp_n * self.wave_size

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

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            p.short(),
            f"t{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"w{self.warp_m}x{self.warp_n}",
            f"a{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}",
            f"{self.pipeline}_{self.epilogue}",
            self.acc_epilogue.tag(),
            flags={"async": self.async_dma},
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


def is_valid_spec(spec: ImplicitGemmConvSpec, arch: str = "gfx950") -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for ``spec`` on ``arch``.

    The MFMA warp-tile atom (``warp_tile_m x warp_tile_n x
    warp_tile_k``) and the per-WG LDS capacity are sourced from
    :class:`rocke.core.arch.ArchTarget`, so this predicate is
    arch-aware: a warp-tile atom that exists on gfx950 but not gfx942
    (e.g. the default wide ``16x16x32`` f16 atom) is rejected for gfx942
    with a structured reason instead of crashing comgr at lower time
    (``LLVM ERROR: Cannot select intrinsic ...``).
    """
    from ...core.arch import ArchTarget

    try:
        target = ArchTarget.from_gfx(arch)
    except KeyError as e:
        return False, str(e)

    # Geometry divisibility (mirrors ``spec.validate``).
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

    # The MMA *family* is selected from the target's wave size: CDNA (wave64)
    # uses MFMA, the RDNA wave32 targets (gfx11xx) use WMMA. The same warp-tile
    # shape thus resolves to an MFMA op_id on gfx942/gfx950 and a WMMA op_id on
    # gfx1151. ``spec.wave_size`` is baked into ``block_size``, so it must match
    # the target's wave size or the lane geometry is wrong on hardware.
    family = "wmma" if target.wave_size == 32 else "mma"
    if spec.wave_size != target.wave_size:
        return False, (
            f"spec wave_size {spec.wave_size} != {arch} wave_size {target.wave_size}"
        )

    # MMA atom must be in the target's catalog for the requested dtype.
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

    # WMMA (RDNA wave32) coverage mirrors the unified GEMM's narrow subset: the
    # 16x16x16 atom with the simple ``mem`` pipeline + ``default`` epilogue and
    # synchronous descriptor-driven loads. The richer MFMA-shaped paths
    # (compv3/compv4 scheduler interleave, cshuffle LDS-staged C, async DMA,
    # K-unroll, chiplet swizzle, grouped conv) are gated off until ported.
    if family == "wmma":
        if atom != (16, 16, 16):
            return False, f"WMMA conv supports only 16x16x16 (got {atom}) on {arch}"
        if spec.pipeline != "mem":
            return False, (
                f"WMMA conv supports only the 'mem' pipeline "
                f"(got {spec.pipeline!r}) on {arch}"
            )
        if spec.epilogue != "default":
            return False, (
                f"WMMA conv supports only the 'default' epilogue "
                f"(got {spec.epilogue!r}) on {arch}"
            )
        for flag, label in (
            (spec.async_dma, "async_dma"),
            (spec.unroll_k, "unroll_k"),
            (spec.chiplet_swizzle, "chiplet_swizzle"),
        ):
            if flag:
                return False, f"WMMA conv does not support {label} on {arch}"
        if spec.groups != 1:
            return False, f"WMMA conv supports only groups=1 (got {spec.groups})"

    return True, "ok"


def _conv_mma_family(arch: str) -> str:
    from ...core.arch import ArchTarget

    return "wmma" if ArchTarget.from_gfx(arch).wave_size == 32 else "mma"


def _resolve_conv_op(spec: ImplicitGemmConvSpec, arch: str):
    """Resolve the :class:`~rocke.core.arch.MmaOp` for ``spec`` on ``arch``.

    Returns the MFMA op on CDNA and the WMMA op on gfx1151. The op carries the
    backend ``op_id`` (consumed by :meth:`IRBuilder.mma`), the per-lane fragment
    lengths, and the lane/slot -> tile-coordinate layout maps that drive the
    fragment loads and the accumulator scatter — the cross-arch contract that
    lets one conv body emit both ISAs.
    """
    from ...core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    op = target.mma.op_for_shape(
        family=_conv_mma_family(arch),
        a_dtype=spec.data.dtype_a,
        b_dtype=spec.data.dtype_b,
        c_dtype="fp32",
        m=spec.warp_tile_m,
        n=spec.warp_tile_n,
        k=spec.warp_tile_k,
    )
    if op is None:
        raise ValueError(
            f"no MMA atom for conv warp_tile "
            f"({spec.warp_tile_m},{spec.warp_tile_n},{spec.warp_tile_k}) on {arch}"
        )
    return op


# ---------------------------------------------------------------------
# Descriptor builders (the user-visible "transform-DAG" surface)
# ---------------------------------------------------------------------


def make_a_descriptor(
    p: ConvProblem, decompose_m: bool = True, dtype: str = "fp16"
) -> TensorDescriptor:
    """Build the (m, k) -> N[D]HWC linear-offset descriptor for the input.

    2-D DAG  (``p.is_3d`` is False):
      naive(NHWC):                       (n, hi, wi, c)
      + unmerge('m' -> n, ho, wo):       (hi, wi, c, m, ho, wo) intermediate
      + embed((ho, y) -> hi):            (wi, c, m, y, wo)      intermediate
      + embed((wo, x) -> wi):            (c, m, y, x)           intermediate
      + unmerge('k' -> y, x, c):         (m, k)                 user-facing
      + pad('y' lo=0 hi=Y):              boundary check
      + pad('x' lo=0 hi=X):              boundary check

    3-D DAG  (``p.is_3d`` is True):
      naive(NDHWC):                      (n, di, hi, wi, c)
      + unmerge('m' -> n, do, ho, wo)
      + embed((do, z) -> di)
      + embed((ho, y) -> hi)
      + embed((wo, x) -> wi)
      + unmerge('k' -> z, y, x, c)
      + pad('z'), pad('y'), pad('x')

    When ``decompose_m`` is ``False`` the leading ``unmerge('m' -> ...)``
    is dropped and the user-facing upper coords become ``(n, ho, wo, k)``
    directly. This is a strict win for callers that already hold ``(ho, wo)``
    cheaply (e.g. computed via shift/mask from the tile row): the default
    chain would re-decompose ``m = ho*Wo + wo`` back into ``(n, ho, wo)`` via
    two magic divisions (~10 VALU per A coord) — a pure round-trip. Feeding
    ``(n, ho, wo)`` straight in produces a bit-identical offset while skipping
    both the caller-side flatten and the descriptor-side magic unmerge.

    The ``embed`` transforms encode the convolution affine maps
    ``hi = ho*sH - pH + y*dH`` and ``wi = wo*sW - pW + x*dW``, with the
    convolution boundary check baked into the descriptor's validity predicate.
    The ``pad`` transforms add per-coord bound checks on ``y`` and ``x``: when
    ``K_gemm`` is not divisible by the block ``tile_k``, the K-loop loads past
    ``K_gemm-1`` and the unmerge produces ``y >= Y`` or ``x >= X``. Without
    these ``pad`` transforms the kernel would read valid-looking offsets that
    *cross* into adjacent weight rows and blend wrong weights into the
    accumulator.
    """
    transforms = []
    if p.is_3d:
        if decompose_m:
            transforms.append(
                unmerge_magic(
                    "m", into=["n", "do", "ho", "wo"], dims=[p.N, p.Do, p.Ho, p.Wo]
                )
            )
        transforms += [
            embed(
                upper=["do", "z"],
                into="di",
                strides=[p.sD, p.dD],
                offset=-p.pD,
                lo=0,
                hi=p.Di,
            ),
            embed(
                upper=["ho", "y"],
                into="hi",
                strides=[p.sH, p.dH],
                offset=-p.pH,
                lo=0,
                hi=p.Hi,
            ),
            embed(
                upper=["wo", "x"],
                into="wi",
                strides=[p.sW, p.dW],
                offset=-p.pW,
                lo=0,
                hi=p.Wi,
            ),
            unmerge_magic("k", into=["z", "y", "x", "c"], dims=[p.Z, p.Y, p.X, p.C]),
            pad("z", lo=0, hi=p.Z),
            pad("y", lo=0, hi=p.Y),
            pad("x", lo=0, hi=p.X),
        ]
        return TensorDescriptor.naive(
            "A_ndhwc",
            lengths=[p.N, p.Di, p.Hi, p.Wi, p.C],
            dtype=_ir_dtype(dtype),
            coord_names=["n", "di", "hi", "wi", "c"],
        ).transform(*transforms)
    else:
        if decompose_m:
            transforms.append(
                unmerge_magic(upper="m", into=["n", "ho", "wo"], dims=[p.N, p.Ho, p.Wo])
            )
        transforms += [
            embed(
                upper=["ho", "y"],
                into="hi",
                strides=[p.sH, p.dH],
                offset=-p.pH,
                lo=0,
                hi=p.Hi,
            ),
            embed(
                upper=["wo", "x"],
                into="wi",
                strides=[p.sW, p.dW],
                offset=-p.pW,
                lo=0,
                hi=p.Wi,
            ),
            unmerge_magic(upper="k", into=["y", "x", "c"], dims=[p.Y, p.X, p.C]),
            # pad('y'/'x'): guard against partial K-tile overruns into adjacent weight rows.
            pad("y", lo=0, hi=p.Y),
            pad("x", lo=0, hi=p.X),
        ]
        return TensorDescriptor.naive(
            "A_nhwc",
            lengths=[p.N, p.Hi, p.Wi, p.C],
            dtype=_ir_dtype(dtype),
            coord_names=["n", "hi", "wi", "c"],
        ).transform(*transforms)


def make_b_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (n_gemm, k_gemm) -> K[Z]YXC linear-offset descriptor for the weight.

    KYXC (2-D) / KZYXC (3-D) is a flat row-major layout, and the
    implicit-GEMM treats ``n_gemm = k_out`` and
    ``k_gemm = y*X*C + x*C + c`` (2-D) or ``k_gemm = z*Y*X*C + y*X*C + x*C + c`` (3-D).
    The descriptor is a renaming + unmerge:

      2-D:
        naive(KYXC)  (k_out, y, x, c)
        + unmerge('k_gemm' -> y, x, c)
        + pad('y' lo=0 hi=Y)   boundary check for partial K-tile
        + pad('x' lo=0 hi=X)   boundary check for partial K-tile

      3-D:
        naive(KZYXC)  (k_out, z, y, x, c)
        + unmerge('k_gemm' -> z, y, x, c)
        + pad('z'), pad('y'), pad('x')

    The ``pad`` transforms catch the ``k_gemm >= K_gemm`` case (when the
    K-loop's last tile is partial): without them, the naive offset computation
    produces a value that's still in-buffer-bounds but indexes into the *next*
    ``k_out``'s weights, contaminating the accumulator.

    A also has ``pad('y')`` / ``pad('x')``, so when A's mask is 0 for the
    partial K-tile the MFMA contribution is 0 regardless of B. Padding B here
    is defense-in-depth so a future A-load change doesn't silently regress.
    """
    if p.is_3d:
        return TensorDescriptor.naive(
            "B_kzyxc",
            lengths=[p.K, p.Z, p.Y, p.X, p.C],
            dtype=_ir_dtype(dtype),
            coord_names=["k_out", "z", "y", "x", "c"],
        ).transform(
            unmerge_magic(
                "k_gemm", into=["z", "y", "x", "c"], dims=[p.Z, p.Y, p.X, p.C]
            ),
            pad("z", lo=0, hi=p.Z),
            pad("y", lo=0, hi=p.Y),
            pad("x", lo=0, hi=p.X),
        )
    return TensorDescriptor.naive(
        "B_kyxc",
        lengths=[p.K, p.Y, p.X, p.C],
        dtype=_ir_dtype(dtype),
        coord_names=["k_out", "y", "x", "c"],
    ).transform(
        unmerge_magic(upper="k_gemm", into=["y", "x", "c"], dims=[p.Y, p.X, p.C]),
        pad("y", lo=0, hi=p.Y),
        pad("x", lo=0, hi=p.X),
    )


def make_d_descriptor(p: ConvProblem, dtype: str = "fp16") -> TensorDescriptor:
    """Build the (m, k_out) -> N[D]HWK linear-offset descriptor for the output.

    2-D:  naive(NHWK):  (n, ho, wo, k_out)
          + unmerge('m' -> n, ho, wo):  user-facing = (m, k_out)

    3-D:  naive(NDHWK): (n, do, ho, wo, k_out)
          + unmerge('m' -> n, do, ho, wo): user-facing = (m, k_out)
    """
    if p.is_3d:
        return TensorDescriptor.naive(
            "D_ndhwk",
            lengths=[p.N, p.Do, p.Ho, p.Wo, p.K],
            dtype=_ir_dtype(dtype),
            coord_names=["n", "do", "ho", "wo", "k_out"],
        ).transform(
            unmerge_magic(
                "m", into=["n", "do", "ho", "wo"], dims=[p.N, p.Do, p.Ho, p.Wo]
            ),
        )
    return TensorDescriptor.naive(
        "D_nhwk",
        lengths=[p.N, p.Ho, p.Wo, p.K],
        dtype=_ir_dtype(dtype),
        coord_names=["n", "ho", "wo", "k_out"],
    ).transform(
        unmerge_magic(upper="m", into=["n", "ho", "wo"], dims=[p.N, p.Ho, p.Wo]),
    )


# ---------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------


def _emit_mfma(b: IRBuilder, atom: MfmaAtom, a: Value, bv: Value, c: Value) -> Value:
    return atom.emit(b, a, bv, c)


def _emit_smem_load(
    b: IRBuilder,
    smem: Value,
    row: Value,
    col: Value,
    n: int,
    *,
    smem_dtype: Optional[Type] = None,
) -> Value:
    if smem_dtype is not None and smem_dtype is not F16:
        vec = b.smem_load_vN(smem, row, col, dtype=smem_dtype, n=n)
        # MFMA fp32 intrinsics take a scalar float, not a vector.
        return b.vec_extract(vec, 0) if (smem_dtype is F32 and n == 1) else vec
    if n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN_f16(smem, row, col, n=n)


def _apply_accumulator_epilogue(
    b: IRBuilder,
    epilogue: ConvAccumulatorEpilogue,
    accs: Sequence[Value],
) -> List[Value]:
    """Apply a static fp32 epilogue to each accumulator fragment.

    The transform is scalar per accumulator lane, then packed back into the
    original vector width so the existing direct/cshuffle epilogues can consume
    the result unchanged.
    """

    if epilogue.is_identity():
        return list(accs)

    out: List[Value] = []
    c_zero = b.const_f32(0.0)
    c_bias = b.const_f32(epilogue.bias) if epilogue.bias != 0.0 else None
    c_scale = b.const_f32(epilogue.scale) if epilogue.scale != 1.0 else None
    c_clamp_min = (
        b.const_f32(epilogue.clamp_min) if epilogue.clamp_min is not None else None
    )
    c_clamp_max = (
        b.const_f32(epilogue.clamp_max) if epilogue.clamp_max is not None else None
    )

    for acc in accs:
        elems: List[Value] = []
        for i in range(acc.type.count):
            v = b.vec_extract(acc, i)
            if c_bias is not None:
                v = b.fadd(v, c_bias)
            if c_scale is not None:
                v = b.fmul(v, c_scale)
            if epilogue.relu:
                v = b.fmax(v, c_zero)
            if c_clamp_min is not None:
                v = b.fmax(v, c_clamp_min)
            if c_clamp_max is not None:
                v = b.fmin(v, c_clamp_max)
            elems.append(v)
        out.append(b.vec_pack(elems, elems[0].type))
    return out


def _emit_frag_smem_load(
    b: IRBuilder,
    src: Value,
    mn_in_atom: Value,
    k_in_atom: Value,
    atom_mn_base: Value,
    k_tile_base: Value,
    frag_len: int,
    *,
    smem_dtype: Optional[Type] = None,
) -> Value:
    """Load one ``frag_len``-wide operand fragment from a row-major LDS tile.

    Both the A LDS tile ``(block_m, block_k)`` and the B LDS tile
    ``(block_n, block_k)`` are row-major with the M/N index as the row and K as
    the column. One lane's fragment occupies a single tile row
    (``atom_mn_base + mn_in_atom``) and ``frag_len`` contiguous K columns from
    ``k_tile_base + k_in_atom`` — true for both the MFMA and WMMA layout maps,
    whose A/B fragment slots are K-contiguous. fp16 smem loads cap at 8 lanes,
    so a wider fragment (WMMA ``<16 x half>``) is assembled from 8-wide chunks.
    """
    lds_row = b.add(atom_mn_base, mn_in_atom)
    lds_col = b.add(k_tile_base, k_in_atom)
    if frag_len <= 8:
        return _emit_smem_load(
            b, src, lds_row, lds_col, frag_len, smem_dtype=smem_dtype
        )
    frag = None
    for off in range(0, frag_len, 8):
        chunk = _emit_smem_load(
            b, src, lds_row, b.add(lds_col, b.const_i32(off)), 8, smem_dtype=smem_dtype
        )
        frag = chunk if frag is None else b.vec_concat(frag, chunk)
    return frag


def _choose_load_vec(spec: ImplicitGemmConvSpec) -> int:
    """Pick the widest load vector width that divides the K tile and
    distributes evenly over `block_size` threads, respecting the
    hardware limit of 4 dwords (16 bytes) per buffer_load.

    Thin adapter over the shared :func:`rocke.helpers.spec.choose_load_vec`."""
    _eb = {"fp16": 2, "bf16": 2, "fp32": 4}.get(spec.data.dtype_a, 2)
    return choose_load_vec(
        spec.tile_m, spec.tile_n, spec.tile_k, spec.block_size, elem_bytes=_eb
    )


def build_implicit_gemm_conv(
    spec: ImplicitGemmConvSpec,
    arch: str = "gfx950",
    extra_params: Optional[Callable[[IRBuilder], object]] = None,
    m_index_fn: Optional[Callable[[IRBuilder, Value, WarpGrid], Value]] = None,
    a_mhw_index_fn: Optional[
        Callable[[IRBuilder, Value, WarpGrid], Tuple[Value, Value, Value]]
    ] = None,
    input_cache_setup: Optional[
        Callable[[IRBuilder, ImplicitGemmConvSpec, WarpGrid, Value], object]
    ] = None,
    a_load_override: Optional[
        Callable[
            [IRBuilder, ImplicitGemmConvSpec, Value, Value, WarpGrid, object], None
        ]
    ] = None,
    a_operand_override: Optional[
        Callable[
            [
                IRBuilder,
                ImplicitGemmConvSpec,
                Value,
                Value,
                Value,
                int,
                WarpGrid,
                object,
            ],
            Value,
        ]
    ] = None,
    epilogue_override: Optional[
        Callable[
            [IRBuilder, ImplicitGemmConvSpec, Sequence[Value], WarpGrid, Value, object],
            None,
        ]
    ] = None,
) -> KernelDef:
    """Build the IR for one implicit-GEMM conv instance.

    Shape:
        M = N * Ho * Wo,
        N_gemm = K,
        K_gemm = Y * X * C.
    Block tile: tile_m x tile_n x tile_k MFMA atoms at warp_tile_m x
        warp_tile_n x warp_tile_k.
    Pipeline: single-buffer LDS, sync barriers, direct vector global
    stores. (compv4 + cshuffle is a follow-on.)

    ``arch`` (``"gfx942"`` / ``"gfx950"``) selects the target GPU. The
    warp-tile MFMA atom is validated against the
    :class:`rocke.core.arch.ArchTarget` catalog (via
    :func:`is_valid_spec`) before lowering, so requesting an atom that
    only exists on gfx950 (e.g. the default wide ``16x16x32`` f16 atom)
    for ``gfx942`` fails with a clean structured error instead of an
    ``LLVM ERROR`` from comgr.
    """
    spec.validate()
    ok, why = is_valid_spec(spec, arch=arch)
    if not ok:
        raise ValueError(f"invalid conv_igemm spec for {arch}: {why}")
    p = spec.problem
    ir_dtype_a = _ir_dtype(spec.data.dtype_a)
    ir_dtype_b = _ir_dtype(spec.data.dtype_b)
    ir_dtype_d = _ir_dtype(spec.data.dtype_d)

    b = IRBuilder(spec.kernel_name())
    if spec.waves_per_eu is not None:
        b.kernel.attrs["waves_per_eu"] = spec.waves_per_eu

    A = b.param(
        "A", PtrType(ir_dtype_a, "global"), noalias=True, readonly=True, align=16
    )
    Bp = b.param(
        "B", PtrType(ir_dtype_b, "global"), noalias=True, readonly=True, align=16
    )
    D = b.param(
        "D", PtrType(ir_dtype_d, "global"), noalias=True, writeonly=True, align=16
    )
    extra_context = extra_params(b) if extra_params is not None else None
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    # Resolve the MMA atom from the target catalog: an MFMA op on CDNA, a WMMA
    # op on gfx1151. ``op`` carries the per-lane fragment lengths and the
    # lane/slot -> coordinate layout maps that drive both the fragment loads and
    # the accumulator scatter, so the single conv body emits both ISAs. The
    # legacy ``MfmaAtom`` (``spec.atom``) is only used by the MFMA-specific
    # phases; it is ``None`` on the WMMA path.
    op = _resolve_conv_op(spec, arch)
    atom = spec.atom if op.family == "mma" else None
    a_per_lane = op.a_frag_len
    b_per_lane = op.b_frag_len
    # LDS operand element type must match the MFMA operand type.
    # bf16 → BF16, fp32 → F32; fp16 and fp8 go through the default f16 path.
    _smem_dtype: Optional[Type] = (
        BF16 if op.a_dtype == "bf16" else F32 if op.a_dtype == "fp32" else None
    )
    c_per_lane = op.c_frag_len

    block_m, block_n, block_k = spec.tile_m, spec.tile_n, spec.tile_k

    # ---- Tile-level block/warp/lane decomposition (CK Tile ``BlockGemmShape``) ----
    # ``WarpGrid.from_atom(...).bind(b)`` emits ``thread_id_x`` /
    # ``lane`` / ``warp_id`` / ``warp_m_idx`` / ``warp_n_idx`` /
    # ``block_m_off`` / ``block_n_off`` as one named decomposition,
    # along with ``max_workgroup_size`` on the kernel attribute list.
    # Replaces ~8 lines of hand-rolled ``b.div``/``b.mod`` lane arithmetic
    # plus the manual ``block_m_off = block_id_y * block_m`` math.
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
    # The mfma cshuffle/direct epilogues consume warp_m_idx/warp_n_idx via the
    # bound ``grid``; the WMMA direct epilogue still takes them explicitly.
    warp_m_idx = grid.warp_m_idx
    warp_n_idx = grid.warp_n_idx

    c0 = b.const_i32(0)
    c_block_k = b.const_i32(block_k)
    c_K_gemm = b.const_i32(p.K_gemm)

    # Grid: (block_n_idx, block_m_idx, 1). We follow gemm_universal:
    # block.x indexes N tile, block.y indexes M tile.
    #
    # With ``spec.chiplet_swizzle=True`` the (bx, by) is first flattened
    # into a linear WGID and run through the chiplet-aware super-tile
    # remap so consecutive WGs share an XCD (and an L2 slice). Conv
    # tile counts are derived from the problem shape and known at
    # build time, so we use the compile-time variant. We override the
    # ``WarpGrid``'s default block offsets via ``dataclasses.replace``
    # so the downstream helpers (loaders, epilogues) automatically
    # pick up the remapped origins.
    if spec.chiplet_swizzle:
        from ...helpers.grid import chiplet_aware_super_tile

        num_pid_m = (p.M + block_m - 1) // block_m
        num_pid_n = (p.N_gemm + block_n - 1) // block_n
        c_num_pid_n = b.const_i32(num_pid_n)
        wgid_flat = b.add(b.mul(b.block_id_y(), c_num_pid_n), b.block_id_x())
        swz = chiplet_aware_super_tile(
            b,
            wgid_flat,
            num_pid_m=num_pid_m,
            num_pid_n=num_pid_n,
            wgm=spec.chiplet_wgm,
            num_xcds=spec.chiplet_num_xcds,
            chunk_size=spec.chiplet_chunk_size,
        )
        block_m_off_v = b.mul(swz.row, b.const_i32(block_m))
        block_n_off_v = b.mul(swz.col, b.const_i32(block_n))
        grid = dc_replace(grid, block_m_off=block_m_off_v, block_n_off=block_n_off_v)
    else:
        block_m_off_v = grid.block_m_off
        block_n_off_v = grid.block_n_off

    # LDS bank-conflict avoidance for the sync path: pad each K-row
    # by 8 halves so the stride is `block_k + 8` not `block_k`.
    # Adjacent lanes reading `ds_read_b128` (16 bytes = 4 banks each)
    # at the same K offset but different M rows would otherwise hit the
    # same banks; the +8 half pad (=16 bytes) shifts each row by 1
    # bank cycle. The pad-by-8 trick is a standard remedy: e.g. a
    # `K_PAD = 136 = 128 + 8` row stride for a `block_k = 128` tile.
    #
    # Async path (runbook §6.3) writes lane-contiguous LDS via
    # `raw_ptr_buffer_load_lds`, so K-pad would break the layout the
    # intrinsic produces. Use plain `[block_m, block_k]` instead;
    # bank conflicts (if they bind) move into the consumer's
    # ds_read distribution.
    lds_layout = spec.effective_lds_layout()
    if spec.async_dma:
        lds_layout.validate_for_async()
    A_smem = b.smem_alloc(
        ir_dtype_a, lds_layout.storage_shape(block_m), name_hint="A_smem"
    )
    B_smem = b.smem_alloc(
        ir_dtype_b, lds_layout.storage_shape(block_n), name_hint="B_smem"
    )
    # Async DMA only buys overlap when there is a second buffer to
    # write into while the MFMA phase reads from the first. Force
    # double-buffering whenever the pipeline opts into async DMA,
    # regardless of the chosen `compv*` flag.
    double_buffer = spec.pipeline == "compv4" or spec.async_dma or spec.unroll_k
    if double_buffer:
        A_smem2 = b.smem_alloc(
            ir_dtype_a, lds_layout.storage_shape(block_m), name_hint="A_smem2"
        )
        B_smem2 = b.smem_alloc(
            ir_dtype_b, lds_layout.storage_shape(block_n), name_hint="B_smem2"
        )
    else:
        A_smem2 = A_smem
        B_smem2 = B_smem

    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n
    k_atoms = spec.k_atoms_per_tile_k

    acc_init = b.zero_vec_f32(c_per_lane)
    accs = [
        (f"acc_m{mi}_n{ni}", acc_init) for mi in range(mfmas_m) for ni in range(mfmas_n)
    ]

    threads = spec.block_size
    _auto_load_vec = _choose_load_vec(spec)
    load_vec_a = (
        spec.vector_size_a if spec.vector_size_a is not None else _auto_load_vec
    )
    load_vec_b = (
        spec.vector_size_b if spec.vector_size_b is not None else _auto_load_vec
    )
    # ``CoalescedTileLoader`` derives ``vecs_per_thread`` /
    # ``cols_per_vec`` internally from ``(tile_rows, tile_cols,
    # block_size, load_vec)`` and re-emits the per-iter constants
    # (``c_threads``, ``c_load_vec``, ``c_cols_per_vec``) once per
    # ``load()`` invocation, which the AMDGPU backend constant-folds.

    # The two descriptors used for global loads. The A descriptor is
    # the conv-coord-transform DAG; B is a simple naive (KYXC) +
    # unmerge for K_gemm.
    A_desc = make_a_descriptor(
        p, decompose_m=(a_mhw_index_fn is None), dtype=spec.data.dtype_a
    )
    B_desc = make_b_descriptor(p, dtype=spec.data.dtype_b)

    # CK Tile-style buffer views over A / B / D. ``make_buffer_resource``
    # wraps ``b.buffer_rsrc(ptr, num_bytes)`` and pre-binds a zero
    # ``soffset``; the resulting :class:`BufferResource` carries
    # everything ``raw_ptr_buffer_load`` needs. The buffer's DW3 flags
    # silently clamp OOB byte offsets to zero on loads / drop them on
    # stores -- the canonical AMDGPU tail-safe load idiom used for
    # both the conv padding-zone reads and the tail-of-grid epilogue
    # stores.
    a_buf_rsrc = make_buffer_resource(b, A, num_bytes=A_bytes)
    b_buf_rsrc = make_buffer_resource(b, Bp, num_bytes=B_bytes)
    d_buf_rsrc = make_buffer_resource(b, D, num_bytes=D_bytes)
    a_rsrc = a_buf_rsrc.rsrc
    b_rsrc = b_buf_rsrc.rsrc
    d_rsrc = d_buf_rsrc.rsrc
    input_cache_context = (
        input_cache_setup(b, spec, grid, a_rsrc)
        if input_cache_setup is not None
        else None
    )

    # Descriptor callbacks shared by both sync and async paths.
    # `(row, col)` are in the (tile_local M, tile_local K halves)
    # coordinate system. The descriptor returns
    # `(element_offset, valid_predicate)`.
    def a_descriptor(b_: IRBuilder, row: Value, col: Value):
        k_val = b_.add(k_off_capture[0], col)
        if a_mhw_index_fn is not None:
            # Decomposed A descriptor: feed (n, ho, wo) straight in, skipping
            # the m-flatten -> magic-unmerge round-trip (see make_a_descriptor).
            n_v, ho_v, wo_v = a_mhw_index_fn(b_, row, grid)
            return A_desc.offset(b_, n=n_v, ho=ho_v, wo=wo_v, k=k_val)
        m_val = (
            m_index_fn(b_, row, grid)
            if m_index_fn is not None
            else b_.add(block_m_off_v, row)
        )
        return A_desc.offset(b_, m=m_val, k=k_val)

    def b_descriptor(b_: IRBuilder, row: Value, col: Value):
        k_out = b_.add(block_n_off_v, row)
        kg = b_.add(k_off_capture[0], col)
        return B_desc.offset(b_, k_out=k_out, k_gemm=kg)

    # `k_off_capture` lets the closures pick up the current k0 from
    # the K-loop body without recompiling the loaders per iteration.
    # The list is mutated by `emit_load_phase`.
    k_off_capture: List[Optional[Value]] = [None]

    if spec.async_dma:
        # Async DRAM -> LDS via `raw_ptr_buffer_load_lds`. Each wave
        # writes lane-contiguous LDS at the wave-uniform base computed
        # by AsyncTileLoader. Consumers (the MFMA phase) must place an
        # `s_waitcnt(vmcnt=0)` before the first ds_read.
        a_loader = AsyncTileLoader.from_tile(
            tile_rows=block_m,
            tile_cols=block_k,
            block_size=threads,
            wave_size=spec.wave_size,
            elem_dtype=ir_dtype_a,
        )
        b_loader = AsyncTileLoader.from_tile(
            tile_rows=block_n,
            tile_cols=block_k,
            block_size=threads,
            wave_size=spec.wave_size,
            elem_dtype=ir_dtype_b,
        )
        a_sync_loader = None
        b_sync_loader = None
    else:
        a_loader = None
        b_loader = None
        # Sync path: ``CoalescedTileLoader`` encapsulates the
        # "pick load_vec, per-thread div/mod into (row, col),
        # buffer_load_vN_f16 -> smem_store_vN_f16" pattern. Same per-iter
        # IR shape as the prior hand-rolled loop, but the per-thread
        # chunk math (``cols_per_vec``, ``vecs_per_thread``, OOB sentinel
        # routing via the descriptor's ``valid`` predicate) is centralised
        # so a future tile-shape sweep picks it up uniformly across
        # GEMM, conv, and attention loads.
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
        )

    schedule = SchedulePolicy.for_pipeline(
        "async_dma" if spec.async_dma else spec.pipeline
    )
    schedule.emit_prologue(b)

    def emit_load_phase(k_off: Value, A_dst: Value, B_dst: Value) -> None:
        """Global -> LDS copy for one K tile via the descriptor DAG.

        For A: each (a_row, a_col) inside the block maps to
            m = block_m_off + a_row
            k = k_off + a_col
        and the A descriptor turns (m, k) -> NHWC linear offset with
        the convolution bounds check (pad on y/x, embed for hi/wi).
        The buffer rsrc is created with flag 0x00027000 which encodes
        proper bounds checking; OOB byte offsets silently return 0
        (the runbook §6.1 lever for tail-safe loads).

        For B: same (b_row, b_col) -> (k_out, k_gemm) -> KYXC linear
        offset, also bounds-checked via the pad on y/x (catches the
        partial-last-tile case).

        `spec.async_dma=True` switches the load path to
        ``AsyncTileLoader`` + ``raw_ptr_buffer_load_lds`` (runbook §6.3).
        Otherwise we use ``CoalescedTileLoader`` which emits the
        register-staged ``buffer_load_vN -> smem_store_vN`` pipeline.
        """
        k_off_capture[0] = k_off

        if spec.async_dma:
            # ``CACHE_STREAM`` (SLC=1) is correct here: each K-tile is
            # consumed exactly once by the MFMA phase of the same iter
            # and then overwritten by the next iter's prefetch. Marking
            # the loads as streaming keeps them from evicting useful
            # cache lines (e.g. the B matrix's columns that *will* be
            # re-read across multiple M-tiles within the same XCD).
            from ...core.ir import CACHE_STREAM

            a_slot = a_loader.bind(b, smem_dst=A_dst, wave_id=warp_id)
            a_slot.issue(
                b,
                tid=tid,
                rsrc=a_rsrc,
                descriptor=a_descriptor,
                coherency=CACHE_STREAM,
            )
            b_slot = b_loader.bind(b, smem_dst=B_dst, wave_id=warp_id)
            b_slot.issue(
                b,
                tid=tid,
                rsrc=b_rsrc,
                descriptor=b_descriptor,
                coherency=CACHE_STREAM,
            )
            return

        # Sync path: ``CoalescedTileLoader.load`` emits the per-thread
        # ``buffer_load_vN_f16 -> smem_store_vN_f16`` chunks. Each
        # callback gets ``(row, col)`` inside the tile-local frame and
        # returns the global element offset + validity predicate, so
        # the conv-coord-transform DAG drives the address arithmetic
        # while the loader owns the thread distribution.
        if a_load_override is not None:
            a_load_override(b, spec, k_off, A_dst, grid, input_cache_context)
        else:
            a_sync_loader.load(
                b,
                tid=tid,
                smem_dst=A_dst,
                descriptor=a_descriptor,
                rsrc=a_rsrc,
            )
        b_sync_loader.load(
            b,
            tid=tid,
            smem_dst=B_dst,
            descriptor=b_descriptor,
            rsrc=b_rsrc,
        )

    def emit_wmma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value]
    ) -> List[Value]:
        """One K-tile of WMMA atoms, fully MMA-contract driven (gfx1151).

        Mirrors ``gemm_universal._emit_wmma_phase``: operand fragments come from
        the op's A/B layout maps and the matmul is emitted target-neutrally via
        :meth:`IRBuilder.mma` (the backend selects the WMMA intrinsic). The A
        map yields ``(row, k)`` and the B map ``(k, col)``; both LDS tiles store
        the M/N index as the row and K as the column, so we read the M-coord
        from the A map and the N-coord (= ``col``) from the B map.
        """
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
                    new_accs[flat] = b.mma(op, a_rows[mi], b_cols[ni], new_accs[flat])
                    flat += 1
        return new_accs

    def emit_mfma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value]
    ) -> List[Value]:
        """One K-tile worth of MFMAs across all per-warp atom positions.

        For the `compv4`/`compv3` pipelines we interleave sched_group_barrier
        hints inside the K-atom loop so the AMDGPU backend overlaps DS_read
        + MFMA + VMEM traffic. The hints don't reorder our SSA — they tell
        the post-RA scheduler what groups to keep together.
        """
        if op.family == "wmma":
            return emit_wmma_phase(A_src, B_src, iter_vars)
        # Tile-level lane decode (CK Tile ``BlockGemmAdaptor`` analogue):
        #   16x16:  m_in_atom = lane % 16,  k_blk = lane / 16,  n_in_atom = lane % 16
        #   32x32:  m_in_atom = lane % 32,  k_blk = lane / 32,  n_in_atom = lane % 32
        # ``decode_mfma_lanes`` returns a frozen :class:`LaneDecode`
        # carrying these as named SSA fields, so the MFMA loop never
        # mis-derives them (a perennial copy-paste hazard).
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
                    warp_m_off, b.add(b.const_i32(mi * spec.warp_tile_m), m_in_atom)
                )
                if a_operand_override is not None:
                    a_rows.append(
                        a_operand_override(
                            b,
                            spec,
                            a_row,
                            k_off_capture[0],
                            col_base,
                            a_per_lane,
                            grid,
                            input_cache_context,
                        )
                    )
                else:
                    a_rows.append(
                        _emit_smem_load(
                            b,
                            A_src,
                            a_row,
                            col_base,
                            a_per_lane,
                            smem_dtype=_smem_dtype,
                        )
                    )

            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off, b.add(b.const_i32(ni * spec.warp_tile_n), n_in_atom)
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

            # sched_group_barrier hints for `compv3`/`compv4`. The group
            # masks follow the runbook §7.3 convention:
            #   0x100 = DS_READ, 0x008 = MFMA, 0x020 = DS_WRITE, 0x040 = VMEM
            #
            # We tell the scheduler: one group of (mfmas_m + mfmas_n)
            # DS_READs (one per row of A and one per col of B inside
            # this kk step), then one group of mfmas_m * mfmas_n
            # MFMAs. Forcing DS_READs ahead of MFMAs gives the
            # backend latitude to issue the LDS reads early so the
            # MFMA pipeline doesn't stall waiting for operands.
            #
            # Tried alternative patterns:
            # - 1+1 paired (DS+MFMA+DS+MFMA...): 175 TFLOPS (worse).
            # - none:                            tested below.
            # - 1 DS group + 1 MFMA group:       186 TFLOPS (best).
            schedule.emit_after_mfma_step(
                b,
                ds_read_count=mfmas_m + mfmas_n,
                mfma_count=mfmas_m * mfmas_n,
            )

        return new_accs

    # ---- the K loop ----
    # Two code paths:
    #
    # 1) Sync path (`async_dma=False`): emit a single `scf.for_iter`
    #    body that runs the load + barrier + MFMA + barrier sequence.
    #    No software pipelining; each iter waits for its own load.
    #
    # 2) Async path (`async_dma=True`): Python-unroll the K loop and
    #    ping-pong between `A_smem`/`A_smem2` (and `B_smem`/`B_smem2`)
    #    so that the load for iter `t+1` is issued while the MFMA for
    #    iter `t` runs. This is the runbook §8.1 software-pipeline
    #    pattern. The `s_waitcnt(vmcnt=0)` drains only the *previous*
    #    iter's DMA before consumers read its LDS buffer; the next
    #    iter's DMA is already in flight against the other buffer.
    #
    #    K_gemm / block_k is the number of unrolled iters; for the
    #    bake-off shape this is 9 (576 / 64), generating ~9x more IR
    #    but staying well under the 160 KiB LDS budget and the
    #    per-kernel ISA size limits.
    if spec.unroll_k:
        # Double-buffered Python-unrolled K-loop software pipeline.
        #
        # Stage tile it+1 into the alternate LDS buffer while the MFMA for
        # tile it reads the current buffer. The buffers are disjoint, so the
        # next tile's global->LDS writes overlap the current tile's ds_read +
        # MFMA work instead of being serialized behind a barrier (the bug in
        # the old single-buffer form, which also omitted the trailing barrier
        # and thus raced the next load against the current MFMA's LDS reads).
        #
        # One barrier per iteration does double duty: it publishes the tile
        # just prefetched into `nxt` before that tile's MFMA next iteration,
        # and it orders the current tile's ds_reads ahead of the it+2 prefetch
        # that reuses the same buffer two iterations later.
        K_iters = (p.K_gemm + block_k - 1) // block_k
        current_accs = [v for _, v in accs]
        bufs = [(A_smem, B_smem), (A_smem2, B_smem2)]

        # Prologue: stage tile 0 into buffer 0 and publish it.
        emit_load_phase(b.const_i32(0), bufs[0][0], bufs[0][1])
        b.sync()

        for it in range(K_iters):
            cur = bufs[it % 2]
            if it + 1 < K_iters:
                nxt = bufs[(it + 1) % 2]
                emit_load_phase(b.const_i32((it + 1) * block_k), nxt[0], nxt[1])
            # The prefetch above clobbered k_off_capture with tile it+1's
            # offset. Restore tile it's offset so an `a_operand_override`
            # (if any) addresses the tile actually consumed by this MFMA.
            k_off_capture[0] = b.const_i32(it * block_k)
            current_accs = emit_mfma_phase(cur[0], cur[1], current_accs)
            b.sync()

        final_accs = current_accs
    elif not spec.async_dma:
        for_op = b.scf_for_iter(c0, c_K_gemm, c_block_k, accs, iv_name="k0")
        with for_op as (k0, iter_vars):
            emit_load_phase(k0, A_smem, B_smem)
            b.sync()
            new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)
            b.sync()
            b.scf_yield(*new_accs)
        final_accs = for_op.results
    else:
        # async_dma path (now fixed as of d6119ef2b8a)
        K_iters = (p.K_gemm + block_k - 1) // block_k
        bufs = [(A_smem, B_smem), (A_smem2, B_smem2)]

        pipeline = SoftwarePipeline(
            num_iters=K_iters,
            double_buffer=double_buffer,
            wait_vmcnt=True,
            sync_after_wait=True,
            sync_before_issue=True,
            overlap_vmcnt=True,
        )

        def issue_load(it: int, buf_pair):
            emit_load_phase(b.const_i32(it * block_k), buf_pair[0], buf_pair[1])

        def compute(_it: int, buf_pair, state):
            return emit_mfma_phase(buf_pair[0], buf_pair[1], state)

        final_accs = pipeline.run_ping_pong(
            b,
            buffers=bufs,
            initial_state=[v for _, v in accs],
            issue_load=issue_load,
            compute=compute,
            schedule=schedule,
        )

    # ---- epilogue ----
    final_accs = _apply_accumulator_epilogue(b, spec.acc_epilogue, final_accs)
    # Both ``DirectEpilogue`` and ``CShuffleEpilogue`` consume the bound
    # :class:`WarpGrid`, which carries the per-warp / per-block / per-lane
    # SSA values plus the tile origins. The conv-specific bit is the
    # D-descriptor address callback.
    if epilogue_override is not None:
        epilogue_override(b, spec, final_accs, grid, d_rsrc, extra_context)
    elif spec.epilogue == "cshuffle":
        _emit_cshuffle_epilogue(b, spec, final_accs, grid, d_rsrc)
    elif op.family == "wmma":
        # WMMA (RDNA) direct epilogue still uses the explicit per-warp/lane
        # decomposition (it predates the helper-based path); pass the bound
        # grid's components.
        _emit_direct_epilogue_wmma(
            b,
            spec,
            op,
            final_accs,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off_v,
            block_n_off_v,
            d_rsrc,
            c0,
        )
    else:
        _emit_direct_epilogue(b, spec, final_accs, grid, d_rsrc)
    return b.kernel


# ---------------------------------------------------------------------
# Epilogue: direct per-lane vector global stores via the D descriptor
# ---------------------------------------------------------------------


def _emit_direct_epilogue(
    b: IRBuilder,
    spec: ImplicitGemmConvSpec,
    accs: Sequence[Value],
    grid: WarpGrid,
    d_rsrc: Value,
) -> None:
    """Per-lane scalar store driven by the D descriptor DAG.

    Delegates to :class:`rocke.helpers.epilogues.DirectEpilogue`,
    which owns the per-(mi, ni)-atom + per-``c_per_lane``-slot lane
    loop and the OOB-sentinel address routing. The conv-specific
    bit is the ``addr_fn``: the D descriptor maps
    ``(m, k_out) -> NHWK linear element offset`` via the
    coordinate-transform DAG.
    """
    p = spec.problem
    D_desc = make_d_descriptor(p, dtype=spec.data.dtype_d)

    def d_addr(b_: IRBuilder, m_val: Value, n_val: Value):
        return D_desc.offset(b_, m=m_val, k_out=n_val)

    DirectEpilogue(atom=spec.atom, grid=grid).store(
        b,
        accs=accs,
        addr_fn=d_addr,
        d_rsrc=d_rsrc,
        bounds=(b.const_i32(p.M), b.const_i32(p.N_gemm)),
    )


def _emit_direct_epilogue_wmma(
    b: IRBuilder,
    spec: ImplicitGemmConvSpec,
    op,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    d_rsrc: Value,
    c0: Value,
) -> None:
    """Per-lane fp16 store for the WMMA (gfx1151) accumulator layout.

    The WMMA wave32 accumulator scatters the M x N tile across lanes
    differently from MFMA, so the (row, col) of every per-lane slot comes from
    the op's accumulator layout map (``op.c_layout()``) rather than the
    MFMA-specific ``MfmaAtom.lane_to_output``. Each slot is one f16 store routed
    through the same D descriptor + OOB-safe buffer-store idiom as the MFMA
    direct epilogue.
    """
    p = spec.problem
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))

    c_M = b.const_i32(p.M)
    c_N = b.const_i32(p.N_gemm)
    D_desc = make_d_descriptor(p, dtype=spec.data.dtype_d)
    c_map = op.c_layout()

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
                v_f16 = b.trunc_f32_to_f16(v_f32)

                d_off_elems, _ = D_desc.offset(b, m=m_val, k_out=n_val)
                d_off_bytes = b.mul(d_off_elems, b.const_i32(2))
                safe_off = b.select(ok, d_off_bytes, b.const_i32((1 << 31) - 1))
                b.buffer_store_f16(d_rsrc, safe_off, c0, v_f16)


def _emit_cshuffle_epilogue(
    b: IRBuilder,
    spec: ImplicitGemmConvSpec,
    accs: Sequence[Value],
    grid: WarpGrid,
    d_rsrc: Value,
) -> None:
    """LDS-staged cshuffle epilogue — the runbook §9.3 lever.

    Delegates to :class:`rocke.helpers.epilogues.CShuffleEpilogue`,
    which implements the canonical three-stage pattern (mirrors CK
    Tile's ``cshuffle_epilogue.hpp``):

      1. Each lane converts its `<c_per_lane x f32>` accumulator to
         `<c_per_lane x f16>` and stores them into an
         `[tile_m x tile_n]` LDS region at the MFMA *output* layout.
      2. ``block_sync_lds`` (s_barrier).
      3. A flat distribution of `block_size` threads reads
         `<store_vec x f16>` from LDS at consecutive row-major
         positions and issues one
         `<store_vec x f16>` buffer_store_short_or_b{32,64,128}.

    For the bake-off shape (block_m=64, block_n=64, block_size=256,
    store_vec=8) this swaps 4096 scalar fp16 stores per block for
    512 wide-aligned 16-byte stores — same bytes, fully coalesced.

    The conv-specific bit is the ``addr_fn``: the D descriptor maps
    ``(m, k_out) -> NHWK linear element offset`` via the
    coordinate-transform DAG.
    """
    p = spec.problem
    D_desc = make_d_descriptor(p, dtype=spec.data.dtype_d)

    def d_addr(b_: IRBuilder, m_val: Value, n_val: Value):
        return D_desc.offset(b_, m=m_val, k_out=n_val)

    _cshuffle_kwargs: dict = {}
    if spec.vector_size_c is not None:
        _cshuffle_kwargs["max_store_vec"] = spec.vector_size_c
    CShuffleEpilogue.from_grid(atom=spec.atom, grid=grid, **_cshuffle_kwargs).store(
        b,
        accs=accs,
        addr_fn=d_addr,
        d_rsrc=d_rsrc,
        bounds=(b.const_i32(p.M), b.const_i32(p.N_gemm)),
    )
