# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Implicit-GEMM convolution kernel instance with DSL AUTOMATIC optimizations.

This version showcases Phase 3 + Phase 4 compiler optimizations:
- Phase 3: Automatic loop unrolling (unroll=True)
- Phase 4a: Automatic barrier elision (elide_trailing_barrier=True)

DIFFERENCE from conv_implicit_gemm.py (manual version):
- Manual version: Python-level unrolling with hand-optimized barrier placement
- Auto version: Single unified loop body, DSL handles optimizations

Performance: Both achieve ~660+ TFLOPS on ResNet50 conv3_1
Code simplicity: Auto version is ~80 lines (vs ~160 for manual duplication)

Authoring style (this is what the kernel writer types):

    spec = ImplicitGemmConvSpec(
        problem=ConvProblem(N=8, Hi=56, Wi=56, C=64,
                            K=64, R=3, S=3,
                            sH=1, sW=1, pH=1, pW=1, dH=1, dW=1),
        tile_m=64, tile_n=64, tile_k=128,
        warp_m=2, warp_n=2,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        unroll_k=True  # DSL handles unrolling AND barrier optimization
    )
    kernel = build_implicit_gemm_conv_auto(spec)

Benefits of automatic approach:
1. Single loop body (no duplication)
2. Safer (DSL ensures correct barriers in final iteration)
3. Easier to maintain (one code path)
4. Same performance (compiler matches manual optimization)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..core.ir import (
    F16,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Value,
)
from ..helpers.atoms import MfmaAtom, mfma_atom
from ..helpers.layouts import LdsLayout
from ..helpers.loads import AsyncTileLoader
from ..helpers.pipeline import SoftwarePipeline
from ..helpers.schedule import SchedulePolicy
from ..transforms import TensorDescriptor, embed, pad, unmerge


# ---------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ConvProblem:
    """The convolution shape parameters.

    Layouts (matching `/workspace/dsl_bake_off/src/conv_problem.hpp`):
      A: NHWC fp16, shape `[N, Hi, Wi, C]`
      B: KRSC fp16, shape `[K, R, S, C]`
      D: NHWK fp16, shape `[N, Ho, Wo, K]`

    Implicit-GEMM packing:
      M = N * Ho * Wo
      N_gemm = K
      K_gemm = R * S * C
    """

    N: int
    Hi: int
    Wi: int
    C: int
    K: int
    R: int
    S: int
    sH: int = 1
    sW: int = 1
    pH: int = 0
    pW: int = 0
    dH: int = 1
    dW: int = 1

    @property
    def Ho(self) -> int:
        return (self.Hi + 2 * self.pH - self.dH * (self.R - 1) - 1) // self.sH + 1

    @property
    def Wo(self) -> int:
        return (self.Wi + 2 * self.pW - self.dW * (self.S - 1) - 1) // self.sW + 1

    @property
    def M(self) -> int:
        return self.N * self.Ho * self.Wo

    @property
    def N_gemm(self) -> int:
        return self.K

    @property
    def K_gemm(self) -> int:
        return self.R * self.S * self.C

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N_gemm * self.K_gemm

    def short(self) -> str:
        return f"N{self.N}H{self.Hi}W{self.Wi}C{self.C}_K{self.K}R{self.R}S{self.S}"


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

    tile_m: int = 64
    tile_n: int = 64
    tile_k: int = 128

    warp_m: int = 2
    warp_n: int = 2

    warp_tile_m: int = 16
    warp_tile_n: int = 16
    warp_tile_k: int = 32

    wave_size: int = 64

    pipeline: str = "mem"
    epilogue: str = "default"
    async_dma: bool = False
    unroll_k: bool = False  # NEW: Clean Python-level K-loop unrolling
    lds_k_pad: Optional[int] = None
    lds_layout: Optional[LdsLayout] = None

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
        return mfma_atom("f16", self.warp_tile_m, self.warp_tile_n, self.warp_tile_k)

    def kernel_name(self) -> str:
        p = self.problem
        return (
            f"{self.name}_{p.short()}_t{self.tile_m}x{self.tile_n}x{self.tile_k}"
            f"_w{self.warp_m}x{self.warp_n}"
            f"_a{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}"
            f"_{self.pipeline}_{self.epilogue}"
            f"{'_async' if self.async_dma else ''}"
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
# Descriptor builders (the user-visible "transform-DAG" surface)
# ---------------------------------------------------------------------


def make_a_descriptor(p: ConvProblem) -> TensorDescriptor:
    """Build the (m, k) -> NHWC linear-offset descriptor for the input.

    DAG:
      naive(NHWC):                       (n, hi, wi, c)
      + unmerge('m' -> n, ho, wo):       (hi, wi, c, m, ho, wo) intermediate
      + embed((ho, r) -> hi):            (wi, c, m, r, wo)      intermediate
      + embed((wo, s) -> wi):            (c, m, r, s)           intermediate
      + unmerge('k' -> r, s, c):         (m, k)                 user-facing
      + pad('r' lo=0 hi=R):              boundary check
      + pad('s' lo=0 hi=S):              boundary check

    The two `embed` transforms encode the convolution affine map
    `hi = ho*sH - pH + r*dH` and `wi = wo*sW - pW + s*dW`, with the
    convolution boundary check baked into the descriptor's validity
    predicate. The two `pad` transforms add per-coord bound checks on
    `r` and `s`: when `K_gemm` is not divisible by the block tile_k,
    the K-loop loads past `K_gemm-1` and the unmerge produces `r >= R`
    or `s >= S`. Without these `pad` transforms the kernel would read
    valid-looking offsets that *cross* into adjacent KRSC rows and
    blend wrong weights into the accumulator.
    """
    transforms = [
        unmerge(upper="m", into=["n", "ho", "wo"], dims=[p.N, p.Ho, p.Wo]),
        embed(
            upper=["ho", "r"],
            into="hi",
            strides=[p.sH, p.dH],
            offset=-p.pH,
            lo=0,
            hi=p.Hi,
        ),
        embed(
            upper=["wo", "s"],
            into="wi",
            strides=[p.sW, p.dW],
            offset=-p.pW,
            lo=0,
            hi=p.Wi,
        ),
        unmerge(upper="k", into=["r", "s", "c"], dims=[p.R, p.S, p.C]),
    ]
    # Only add r/s pads if K_gemm doesn't cleanly divide expected tile_k
    # multiples; today we always include them for safety. They are cheap
    # and they protect against the partial-K-tile failure mode.
    transforms += [
        pad("r", lo=0, hi=p.R),
        pad("s", lo=0, hi=p.S),
    ]
    return TensorDescriptor.naive(
        "A_nhwc",
        lengths=[p.N, p.Hi, p.Wi, p.C],
        dtype=F16,
        coord_names=["n", "hi", "wi", "c"],
    ).transform(*transforms)


def make_b_descriptor(p: ConvProblem) -> TensorDescriptor:
    """Build the (n_gemm, k_gemm) -> KRSC linear-offset descriptor for the weight.

    KRSC is a flat row-major `[K, R, S, C]` layout, and the implicit-GEMM
    treats `n_gemm = k_out` and `k_gemm = r*S*C + s*C + c`. So the
    descriptor is just a renaming:
      naive(KRSC):              (k_out, r, s, c)
      + unmerge('k_gemm'
         -> (r, s, c))
      + pad('r' lo=0 hi=R):     boundary check for partial K-tile
      + pad('s' lo=0 hi=S):     boundary check for partial K-tile

    The user-facing coords are then (k_out, k_gemm). The two `pad`
    transforms catch the `k_gemm >= K_gemm` case (when the K-loop's
    last tile is partial): without them, the naive `k_out*RSC + r*SC
    + s*C + c` computation produces an offset that's still
    in-buffer-bounds (because B's total size is K*RSC) but indexes
    into the NEXT k_out's weights, contaminating the load.

    A consumes its B via MFMA at K=32 atoms, and A's load also has
    `pad('r')` / `pad('s')`, so when A's mask is 0 for the partial
    K-tile the MFMA contribution is 0 regardless of B. Padding B
    here is defense-in-depth so a future A-load change doesn't
    silently regress.
    """
    return TensorDescriptor.naive(
        "B_krsc",
        lengths=[p.K, p.R, p.S, p.C],
        dtype=F16,
        coord_names=["k_out", "r", "s", "c"],
    ).transform(
        unmerge(upper="k_gemm", into=["r", "s", "c"], dims=[p.R, p.S, p.C]),
        pad("r", lo=0, hi=p.R),
        pad("s", lo=0, hi=p.S),
    )


def make_d_descriptor(p: ConvProblem) -> TensorDescriptor:
    """Build the (m, k_out) -> NHWK linear-offset descriptor for the output.

    naive(NHWK): (n, ho, wo, k_out)
    + unmerge('m' -> (n, ho, wo)): user-facing = (m, k_out)
    """
    return TensorDescriptor.naive(
        "D_nhwk",
        lengths=[p.N, p.Ho, p.Wo, p.K],
        dtype=F16,
        coord_names=["n", "ho", "wo", "k_out"],
    ).transform(
        unmerge(upper="m", into=["n", "ho", "wo"], dims=[p.N, p.Ho, p.Wo]),
    )


# ---------------------------------------------------------------------
# Kernel body
# ---------------------------------------------------------------------


def _emit_mfma(b: IRBuilder, atom: MfmaAtom, a: Value, bv: Value, c: Value) -> Value:
    return atom.emit(b, a, bv, c)


def _emit_smem_load(b: IRBuilder, smem: Value, row: Value, col: Value, n: int) -> Value:
    if n == 4:
        return b.smem_load_v4_f16(smem, row, col)
    return b.smem_load_vN_f16(smem, row, col, n=n)


def _choose_load_vec(spec: ImplicitGemmConvSpec) -> int:
    """Pick the largest fp16 load vector width that divides the K tile and
    distributes evenly over `block_size` threads. Same rule as GEMM."""
    threads = spec.block_size
    for v in (8, 4, 2, 1):
        if spec.tile_k % v:
            continue
        a_vecs = (spec.tile_m * spec.tile_k) // v
        b_vecs = (spec.tile_n * spec.tile_k) // v
        if a_vecs < threads or b_vecs < threads:
            continue
        if a_vecs % threads or b_vecs % threads:
            continue
        return v
    raise ValueError(f"no usable load_vec for {spec}")


def build_implicit_gemm_conv_auto(spec: ImplicitGemmConvSpec) -> KernelDef:
    """Build the IR for one implicit-GEMM conv instance.

    Shape:
        M = N * Ho * Wo,
        N_gemm = K,
        K_gemm = R * S * C.
    Block tile: tile_m x tile_n x tile_k MFMA atoms at warp_tile_m x
        warp_tile_n x warp_tile_k.
    Pipeline: single-buffer LDS, sync barriers, direct vector global
    stores. (compv4 + cshuffle is a follow-on.)
    """
    spec.validate()
    p = spec.problem

    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block_size

    A = b.param("A", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    Bp = b.param("B", PtrType(F16, "global"), noalias=True, readonly=True, align=16)
    D = b.param("D", PtrType(F16, "global"), noalias=True, writeonly=True, align=16)
    A_bytes = b.param("A_bytes", I32)
    B_bytes = b.param("B_bytes", I32)
    D_bytes = b.param("D_bytes", I32)

    atom = spec.atom
    a_per_lane = atom.a_per_lane
    b_per_lane = atom.b_per_lane
    c_per_lane = atom.c_per_lane

    block_m, block_n, block_k = spec.tile_m, spec.tile_n, spec.tile_k

    c0 = b.const_i32(0)
    c_wave = b.const_i32(spec.wave_size)
    c_warps_n = b.const_i32(spec.warp_n)
    c_block_m = b.const_i32(block_m)
    c_block_n = b.const_i32(block_n)
    c_block_k = b.const_i32(block_k)
    c_K_gemm = b.const_i32(p.K_gemm)

    tid = b.thread_id_x()
    warp_id = b.div(tid, c_wave)
    warp_m_idx = b.div(warp_id, c_warps_n)
    warp_n_idx = b.mod(warp_id, c_warps_n)
    lane = b.mod(tid, c_wave)

    # Grid: (block_n_idx, block_m_idx, 1). We follow gemm_universal:
    # block.x indexes N tile, block.y indexes M tile.
    block_n_off_v = b.mul(b.block_id_x(), c_block_n)
    block_m_off_v = b.mul(b.block_id_y(), c_block_m)

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
    A_smem = b.smem_alloc(F16, lds_layout.storage_shape(block_m), name_hint="A_smem")
    B_smem = b.smem_alloc(F16, lds_layout.storage_shape(block_n), name_hint="B_smem")
    # Async DMA only buys overlap when there is a second buffer to
    # write into while the MFMA phase reads from the first. Force
    # double-buffering whenever the pipeline opts into async DMA,
    # regardless of the chosen `compv*` flag.
    double_buffer = spec.pipeline == "compv4" or spec.async_dma
    if double_buffer:
        A_smem2 = b.smem_alloc(
            F16, lds_layout.storage_shape(block_m), name_hint="A_smem2"
        )
        B_smem2 = b.smem_alloc(
            F16, lds_layout.storage_shape(block_n), name_hint="B_smem2"
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
    load_vec = _choose_load_vec(spec)
    a_vec_total = (block_m * block_k) // load_vec
    b_vec_total = (block_n * block_k) // load_vec
    a_vecs_per_thread = a_vec_total // threads
    b_vecs_per_thread = b_vec_total // threads
    c_threads = b.const_i32(threads)
    c_load_vec = b.const_i32(load_vec)
    c_block_k_div_vec = b.const_i32(block_k // load_vec)

    # The two descriptors used for global loads. The A descriptor is
    # the conv-coord-transform DAG; B is a simple naive (KRSC) +
    # unmerge for K_gemm.
    A_desc = make_a_descriptor(p)
    B_desc = make_b_descriptor(p)

    # Buffer resources so we get free OOB clamping for the bounds we
    # don't catch in the predicate (the A_bytes / B_bytes sizes act as
    # an outer fence; we still rely on the descriptor's `valid`
    # predicate to zero pad-region reads since OOB clamping returns 0
    # but isn't bit-accurate against the CPU/torch reference if we
    # ever shift the pointer base).
    a_rsrc = b.buffer_rsrc(A, A_bytes)
    b_rsrc = b.buffer_rsrc(Bp, B_bytes)
    d_rsrc = b.buffer_rsrc(D, D_bytes)

    # Descriptor callbacks shared by both sync and async paths.
    # `(row, col)` are in the (tile_local M, tile_local K halves)
    # coordinate system. The descriptor returns
    # `(element_offset, valid_predicate)`.
    def a_descriptor(b_: IRBuilder, row: Value, col: Value):
        m_val = b_.add(block_m_off_v, row)
        k_val = b_.add(k_off_capture[0], col)
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
        )
        b_loader = AsyncTileLoader.from_tile(
            tile_rows=block_n,
            tile_cols=block_k,
            block_size=threads,
            wave_size=spec.wave_size,
        )
    else:
        a_loader = None
        b_loader = None

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
        the convolution bounds check (pad on r/s, embed for hi/wi).
        The buffer rsrc is created with flag 0x00027000 which encodes
        proper bounds checking; OOB byte offsets silently return 0
        (the runbook §6.1 lever for tail-safe loads).

        For B: same (b_row, b_col) -> (k_out, k_gemm) -> KRSC linear
        offset, also bounds-checked via the pad on r/s (catches the
        partial-last-tile case).

        `spec.async_dma=True` switches the load path to
        `raw_ptr_buffer_load_lds` (runbook §6.3). Otherwise we use
        register-staged `buffer_load_vN -> smem_store_vN`.
        """
        k_off_capture[0] = k_off

        if spec.async_dma:
            a_slot = a_loader.bind(b, smem_dst=A_dst, wave_id=warp_id)
            a_slot.issue(b, tid=tid, rsrc=a_rsrc, descriptor=a_descriptor)
            b_slot = b_loader.bind(b, smem_dst=B_dst, wave_id=warp_id)
            b_slot.issue(b, tid=tid, rsrc=b_rsrc, descriptor=b_descriptor)
            return

        # Sync path: register-staged DRAM -> LDS.
        oob_sentinel = b.const_i32((1 << 31) - 1)
        c_half_bytes = b.const_i32(2)

        for e in range(a_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            a_row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            a_col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            a_off_elems, a_valid = a_descriptor(b, a_row, a_col)
            a_off_bytes = b.mul(a_off_elems, c_half_bytes)
            safe_off = (
                b.select(a_valid, a_off_bytes, oob_sentinel)
                if a_valid is not None
                else a_off_bytes
            )
            if load_vec == 1:
                a_val = b.buffer_load_f16(a_rsrc, safe_off, c0)
                b.smem_store_f16(A_dst, [a_row, a_col], a_val)
            else:
                dwords = load_vec // 2
                a_vec = b.buffer_load_vN_f16(a_rsrc, safe_off, c0, dwords)
                b.smem_store_vN_f16(A_dst, [a_row, a_col], a_vec, load_vec)

        for e in range(b_vecs_per_thread):
            vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
            b_row = b.div(vec_idx, c_block_k_div_vec)
            col_v = b.mod(vec_idx, c_block_k_div_vec)
            b_col = b.mul(col_v, c_load_vec) if load_vec > 1 else col_v
            b_off_elems, b_valid = b_descriptor(b, b_row, b_col)
            b_off_bytes = b.mul(b_off_elems, c_half_bytes)
            safe_b_off = (
                b.select(b_valid, b_off_bytes, oob_sentinel)
                if b_valid is not None
                else b_off_bytes
            )
            if load_vec == 1:
                b_val = b.buffer_load_f16(b_rsrc, safe_b_off, c0)
                b.smem_store_f16(B_dst, [b_row, b_col], b_val)
            else:
                dwords = load_vec // 2
                b_vec = b.buffer_load_vN_f16(b_rsrc, safe_b_off, c0, dwords)
                b.smem_store_vN_f16(B_dst, [b_row, b_col], b_vec, load_vec)

    def emit_mfma_phase(
        A_src: Value, B_src: Value, iter_vars: Sequence[Value]
    ) -> List[Value]:
        """One K-tile worth of MFMAs across all per-warp atom positions.

        For the `compv4`/`compv3` pipelines we interleave sched_group_barrier
        hints inside the K-atom loop so the AMDGPU backend overlaps DS_read
        + MFMA + VMEM traffic. The hints don't reorder our SSA — they tell
        the post-RA scheduler what groups to keep together.
        """
        # Lane mapping (consistent with the MfmaAtom contract):
        # 16x16:  m_in_atom = lane % 16,  k_blk = lane / 16,  n_in_atom = lane % 16
        # 32x32:  m_in_atom = lane % 32,  k_blk = lane / 32,  n_in_atom = lane % 32
        m_in_atom = b.mod(lane, b.const_i32(spec.warp_tile_m))
        k_blk = b.div(lane, b.const_i32(spec.warp_tile_m))
        n_in_atom = b.mod(lane, b.const_i32(spec.warp_tile_n))

        warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
        warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))

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
                a_rows.append(_emit_smem_load(b, A_src, a_row, col_base, a_per_lane))

            b_cols = []
            for ni in range(mfmas_n):
                b_row = b.add(
                    warp_n_off, b.add(b.const_i32(ni * spec.warp_tile_n), n_in_atom)
                )
                b_cols.append(_emit_smem_load(b, B_src, b_row, col_base, b_per_lane))

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
    # ---- K-loop: DSL AUTOMATIC OPTIMIZATION VERSION ----
    # This version uses a SINGLE unified loop body.
    # The DSL automatically:
    #   - Unrolls the loop if unroll=True (Phase 3)
    #   - Elides trailing barriers in non-final iterations (Phase 4a)
    #
    # No manual duplication needed!

    if spec.unroll_k:
        # DSL automatic unrolling + barrier optimization
        for_op = b.scf_for_iter(
            c0, c_K_gemm, c_block_k, accs,
            iv_name="k0",
            unroll=True,  # Phase 3: emit straight-line code
            elide_trailing_barrier=True  # Phase 4a: elide inter-iteration barriers
        )
        with for_op as (k0, iter_vars):
            # Load phase: global load → LDS write
            emit_load_phase(k0, A_smem, B_smem)
            b.sync()  # Barrier after LDS writes (required in all iterations)

            # MFMA phase: LDS read → MFMA execute
            new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)

            # Trailing barrier for correctness
            b.sync()  # DSL automatically elides this in iterations 0..N-2!
                      # Kept only in final iteration for safety

            b.scf_yield(*new_accs)
        final_accs = for_op.results
    elif not spec.async_dma:
        # Normal loop (not unrolled)
        for_op = b.scf_for_iter(c0, c_K_gemm, c_block_k, accs, iv_name="k0")
        with for_op as (k0, iter_vars):
            emit_load_phase(k0, A_smem, B_smem)
            b.sync()
            new_accs = emit_mfma_phase(A_smem, B_smem, iter_vars)
            b.sync()
            b.scf_yield(*new_accs)
        final_accs = for_op.results
    else:
        # DEPRECATED: async_dma path is broken (36% perf loss)
        raise ValueError(
            "async_dma=True is broken and causes 36% performance loss. "
            "Use unroll_k=True instead for DSL automatic K-loop optimization."
        )

    # ---- epilogue ----
    if spec.epilogue == "cshuffle":
        _emit_cshuffle_epilogue(
            b,
            spec,
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
        _emit_direct_epilogue(
            b,
            spec,
            final_accs,
            warp_m_idx,
            warp_n_idx,
            lane,
            block_m_off_v,
            block_n_off_v,
            d_rsrc,
            c0,
        )
    return b.kernel


# ---------------------------------------------------------------------
# Epilogue: direct per-lane vector global stores via the D descriptor
# ---------------------------------------------------------------------


def _emit_direct_epilogue(
    b: IRBuilder,
    spec: ImplicitGemmConvSpec,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    d_rsrc: Value,
    c0: Value,
) -> None:
    """Per-lane scalar-fp16 store driven by the D descriptor DAG.

    The accumulator layout for a 16x16 atom (the bake-off baseline)
    is per-lane: lane = m_blk * 16 + n_in_atom with m_blk = lane / 16
    and the 4 floats in acc map to rows (m_blk * 4 + i) for i=0..3,
    column n_in_atom. We then ask the D descriptor for the final
    NHWK linear offset and store there with a per-lane bounds check
    on m < M and (n + n_in_atom) < N.
    """
    p = spec.problem
    atom = spec.atom
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))

    c_M = b.const_i32(p.M)
    c_N = b.const_i32(p.N_gemm)
    D_desc = make_d_descriptor(p)

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
            for i in range(atom.c_per_lane):
                row_off, col_off = atom.lane_to_output(b, lane, i)
                m_val = b.add(atom_m_off, row_off)
                n_val = b.add(atom_n_off, col_off)
                m_ok = b.cmp_lt(m_val, c_M)
                n_ok = b.cmp_lt(n_val, c_N)
                ok = b.land(m_ok, n_ok)

                v_f32 = b.vec_extract(acc, i)
                v_f16 = b.trunc_f32_to_f16(v_f32)

                d_off_elems, _ = D_desc.offset(b, m=m_val, k_out=n_val)
                d_off_bytes = b.mul(d_off_elems, b.const_i32(2))
                # CK Tile direct-epilogue trick: OOB byte offsets are
                # silently dropped by the buffer rsrc (see
                # `cktile_fixed_lean_kernel.hpp`:
                # `b_offs[bi] = valid ? real_off_b : 0x80000000`).
                safe_off = b.select(ok, d_off_bytes, b.const_i32((1 << 31) - 1))
                b.buffer_store_f16(d_rsrc, safe_off, c0, v_f16)


def _emit_cshuffle_epilogue(
    b: IRBuilder,
    spec: ImplicitGemmConvSpec,
    accs: Sequence[Value],
    warp_m_idx: Value,
    warp_n_idx: Value,
    lane: Value,
    block_m_off: Value,
    block_n_off: Value,
    d_rsrc: Value,
    c0: Value,
) -> None:
    """LDS-staged cshuffle epilogue — the runbook §9.3 lever.

    Three-stage pattern (mirrors CK Tile's `cshuffle_epilogue.hpp`):
      1. Each lane converts its `<c_per_lane x f32>` accumulator to
         `<c_per_lane x f16>` and stores them into an `[tile_m x
         tile_n]` LDS region at the MFMA *output* layout
         (`row = warp_m_off + atom_m_off + lane_to_output_row`,
         `col = warp_n_off + atom_n_off + n_in_atom`). Each scalar
         store is a single `ds_write_b16`.
      2. `block_sync_lds` (s_barrier).
      3. A flat distribution of `block_size` threads reads
         `<store_vec x f16>` from LDS at consecutive row-major
         positions, computes the NHWK output address via the same
         D descriptor (so an output that lives at (m=M_tile_off+r,
         k_out=N_tile_off+c*store_vec)) and issues one
         `<store_vec x f16>` buffer_store_short_or_b{32,64,128}.

    The win over the direct epilogue:
      - Direct: every lane writes 4 scalar fp16's at MFMA-layout
        positions. Adjacent lanes are 16 columns apart in N
        (within the same M row), so the writes are NOT coalesced.
      - Cshuffle: 256 threads each write `store_vec` (typically 8)
        contiguous halves of the OUTPUT layout. One wide store per
        thread, perfectly coalesced.

    Concretely for the bake-off shape (block_m=64, block_n=64,
    block_size=256, store_vec=8): the direct epilogue issues
    `4 atoms * 4 slots = 16` scalar `buffer_store_short` per thread
    = 4096 stores/block. Cshuffle issues `64*64/8/256 = 2` wide
    stores per thread = 512 wide stores/block. Each store moves 16
    bytes contiguously, so we go from 4096*2 = 8 KB written via
    scalar stores to 512*16 = 8 KB written via wide stores —
    *same bytes but wide-aligned, fully coalesced*.
    """
    p = spec.problem
    t = spec
    atom = spec.atom
    mfmas_m = spec.mfmas_per_warp_m
    mfmas_n = spec.mfmas_per_warp_n

    warp_m_off = b.mul(warp_m_idx, b.const_i32(mfmas_m * spec.warp_tile_m))
    warp_n_off = b.mul(warp_n_idx, b.const_i32(mfmas_n * spec.warp_tile_n))

    # Step 1: stage accumulators in LDS at the MFMA output layout.
    # tile_m * tile_n halves of LDS staging.
    C_smem = b.smem_alloc(F16, [t.tile_m, t.tile_n], name_hint="C_smem")

    is_32x32 = (spec.warp_tile_m, spec.warp_tile_n) == (32, 32)

    if is_32x32:
        c_atom_n = b.const_i32(spec.warp_tile_n)
        n_in_atom = b.mod(lane, c_atom_n)
        m_blk = b.div(lane, c_atom_n)
        flat = 0
        for mi in range(mfmas_m):
            for ni in range(mfmas_n):
                acc = accs[flat]
                flat += 1
                acc_h = b.vec_trunc_f32_to_f16(acc)
                ld_n = b.add(
                    b.add(warp_n_off, b.const_i32(ni * spec.warp_tile_n)),
                    n_in_atom,
                )
                for i in range(atom.c_per_lane):
                    rb = i // 4
                    ri = i % 4
                    m_off = b.add(
                        b.add(b.const_i32(rb * 8), b.mul(m_blk, b.const_i32(4))),
                        b.const_i32(ri),
                    )
                    ld_m = b.add(
                        b.add(warp_m_off, b.const_i32(mi * spec.warp_tile_m)),
                        m_off,
                    )
                    h = b.vec_extract(acc_h, i)
                    b.smem_store_f16(C_smem, [ld_m, ld_n], h)
    else:
        c_atom_n = b.const_i32(spec.warp_tile_n)
        c_clen = b.const_i32(atom.c_per_lane)
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
                    b.add(warp_n_off, b.const_i32(ni * spec.warp_tile_n)),
                    n_in_atom,
                )
                for i in range(atom.c_per_lane):
                    m_off = b.add(m_base, b.const_i32(i))
                    ld_m = b.add(
                        b.add(warp_m_off, b.const_i32(mi * spec.warp_tile_m)),
                        m_off,
                    )
                    h = b.vec_extract(acc_h, i)
                    b.smem_store_f16(C_smem, [ld_m, ld_n], h)

    # Step 2: barrier.
    b.sync()

    # Step 3: wide global stores. We pick the widest `store_vec` that
    # the tile_n divides and the block_size can evenly distribute.
    threads = spec.block_size
    store_vec = 8
    while store_vec > 1:
        if (
            spec.tile_n % store_vec == 0
            and (spec.tile_m * spec.tile_n) // store_vec >= threads
            and ((spec.tile_m * spec.tile_n) // store_vec) % threads == 0
        ):
            break
        store_vec //= 2

    tid = b.thread_id_x()
    c_threads = b.const_i32(threads)
    c_tile_n_div_vec = b.const_i32(spec.tile_n // store_vec)
    vecs_per_thread = (spec.tile_m * spec.tile_n // store_vec) // threads

    D_desc = make_d_descriptor(p)
    c_M = b.const_i32(p.M)
    c_N = b.const_i32(p.N_gemm)
    c_half_bytes = b.const_i32(2)

    for e in range(vecs_per_thread):
        vec_idx = b.add(b.mul(b.const_i32(e), c_threads), tid)
        row = b.div(vec_idx, c_tile_n_div_vec)
        col_v = b.mod(vec_idx, c_tile_n_div_vec)
        col = b.mul(col_v, b.const_i32(store_vec)) if store_vec > 1 else col_v

        m_val = b.add(block_m_off, row)
        n_val = b.add(block_n_off, col)
        m_ok = b.cmp_lt(m_val, c_M)
        # The store is wide, so we need the whole `store_vec`-wide range
        # to be in-bounds. `n + store_vec - 1 < N` ↔ `n + store_vec <= N`.
        n_end = b.add(n_val, b.const_i32(store_vec))
        n_ok = b.cmp_le(n_end, c_N)
        ok = b.land(m_ok, n_ok)

        d_off_elems, _ = D_desc.offset(b, m=m_val, k_out=n_val)
        d_off_bytes = b.mul(d_off_elems, c_half_bytes)
        safe_off = b.select(ok, d_off_bytes, b.const_i32((1 << 31) - 1))

        if store_vec == 1:
            # Pathological fallback — single half load + scalar store.
            v = b.smem_load_vN_f16(C_smem, row, col, n=2)
            h = b.vec_extract(v, 0)
            b.buffer_store_f16(d_rsrc, safe_off, c0, h)
        else:
            v = (
                b.smem_load_v4_f16(C_smem, row, col)
                if store_vec == 4
                else b.smem_load_vN_f16(C_smem, row, col, n=store_vec)
            )
            # buffer_store_vN_f16 dwords = store_vec / 2
            dwords = store_vec // 2
            b.buffer_store_vN_f16(d_rsrc, safe_off, c0, v, dwords)
