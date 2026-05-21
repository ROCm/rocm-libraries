# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native CK DSL entry points for AITER unified attention.

This module intentionally separates *feature selection* from *kernel emission*.
The selector mirrors AITER's Triton wrapper exactly; kernel emission is gated
until every required primitive and correctness/perf path is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ..core.ir import BF16, F16, F32, I32, IRBuilder, KernelDef, PtrType, Type, Value
from ..helpers.compile import compile_kernel
from ..runtime.launcher import (
    KernelLauncher,
    LaunchConfig,
    LaunchSummary,
    PipelineLauncher,
    StreamConfig,
    WorkspaceSpec,
    WorkspacePool,
    _resolved_fence,
    launch_kernel,
    make_kernel,
    wait_stream_and_release,
)

from ..helpers.attention import (
    Attention2DConfig,
    Attention3DConfig,
    PagedKvDescriptor,
    select_2d_config,
    select_3d_config,
    use_2d_kernel,
)
from ..transforms import TensorDescriptor
from .attention_tiled_2d import (
    UnifiedAttention2DTiledSpec,
    build_unified_attention_2d_tiled,
    supports_tiled_2d,
)
from .attention_tiled_3d import (
    UnifiedAttention3DTiledSpec,
    UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled,
    build_unified_attention_reduce_tiled,
    supports_tiled_3d,
)


@dataclass(frozen=True)
class UnifiedAttentionProblem:
    total_q: int
    num_seqs: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    max_seqlen_q: int
    max_seqlen_k: int
    dtype: str
    q_dtype: Optional[str] = None
    sliding_window: int = 0
    softcap: float = 0.0
    use_sinks: bool = False
    use_alibi: bool = False
    use_qq_bias: bool = False
    use_fp8: bool = False
    num_sms: int = 120
    # AMDGPU occupancy hint ("amdgpu-waves-per-eu"). The 2D-tiled and
    # 3D-tiled specs both honour this knob; the scalar paths ignore it
    # because they already fit at 1 wave per workgroup. ``None`` keeps
    # the LLVM backend's heuristic choice.
    waves_per_eu: Optional[int] = None
    # Compile backend for the tiled 2D path:
    #   - ``None`` (default): auto-pick. Uses the LLVM-direct
    #     pipeline (``compile_kernel``) except for large prefill
    #     (``max_seqlen_q > 512 or num_seqs * max_seqlen_q > 1024``),
    #     where ``hipcc --genco`` is measurably faster (≈5% on
    #     ``b4_q1000_kv1000``) thanks to clang's heavier scheduling.
    #   - ``"llvm"``: always use the LLVM-direct path (~90ms compile).
    #   - ``"hipcc"``: always lower to HIP C++ and compile via hipcc
    #     (~450ms compile but ~5% faster on long-running kernels).
    # See ``probe_hip_lowering.py`` for the per-shape comparison.
    compile_backend: Optional[str] = None

    @property
    def num_queries_per_kv(self) -> int:
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
        return self.num_query_heads // self.num_kv_heads

    @property
    def all_decode(self) -> bool:
        return self.max_seqlen_q == 1

    @property
    def total_num_q_blocks_upper_bound(self) -> int:
        block_m = (
            16
            if self.num_queries_per_kv <= 16
            else _next_power_of_2(self.num_queries_per_kv)
        )
        block_q = block_m // self.num_queries_per_kv
        return self.total_q // block_q + self.num_seqs

    def select_path(self) -> str:
        target = self.num_sms * 4
        num_2d = self.total_num_q_blocks_upper_bound * self.num_kv_heads
        return (
            "2d"
            if use_2d_kernel(
                head_size=self.head_size,
                sliding_window=self.sliding_window,
                all_decode=self.all_decode,
                max_seqlen_q=self.max_seqlen_q,
                max_seqlen_k=self.max_seqlen_k,
                target_num_prgms=target,
                num_2d_prgms=num_2d,
            )
            else "3d"
        )

    def select_2d(self) -> Attention2DConfig:
        num_2d = self.total_num_q_blocks_upper_bound * self.num_kv_heads
        return select_2d_config(
            block_size=self.block_size,
            head_size=self.head_size,
            sliding_window=self.sliding_window,
            all_decode=self.all_decode,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            num_queries_per_kv=self.num_queries_per_kv,
            num_2d_prgms=num_2d,
        )

    def select_3d(self) -> Tuple[Attention3DConfig, Attention3DConfig]:
        target = self.num_sms * 4
        num_2d = self.total_num_q_blocks_upper_bound * self.num_kv_heads
        return select_3d_config(
            head_size=self.head_size,
            block_size=self.block_size,
            element_size=2 if self.dtype in ("fp16", "bf16") else 1,
            max_seqlen_k=self.max_seqlen_k,
            target_num_prgms=target,
            num_2d_prgms=num_2d,
        )


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()


def supports_native_unified_attention(
    problem: UnifiedAttentionProblem,
) -> Tuple[bool, str]:
    """Return whether CK DSL can run this problem without fallback today.

    This is deliberately strict. It prevents a partially implemented backend
    from being selected in `auto` mode and gives test code a single place to
    check coverage.

    Scalar 2D backend coverage (this returns True for these):
    - head_size in {64, 128, 256} (the scalar kernel loops over head_size with
      ``b.unroll(p.head_size)``, so any HD that divides cleanly through the
      online-softmax accumulator works)
    - block_size in {16, 32, 64} (used only as the modulus in PagedKvDescriptor
      address arithmetic)
    - dtype in {fp16, bf16}
    - has_sinks: yes
    - sliding_window: yes
    - softcap: yes
    """
    if problem.head_size not in (64, 128, 256):
        return False, f"unsupported head_size {problem.head_size}"
    if problem.block_size not in (16, 32, 64):
        return False, f"unsupported block_size {problem.block_size}"
    if problem.dtype not in ("fp16", "bf16"):
        return False, f"unsupported dtype {problem.dtype}"
    # FP8 K/V cache: scalar 2D backend does not implement the FP8 dequant
    # path yet; only the tiled 2D and tiled 3D backends do (see
    # ``supports_native_unified_attention_tiled`` and ``_3d_tiled``).
    if problem.use_fp8 or problem.q_dtype is not None:
        return False, "FP8 unified attention is not enabled in the scalar 2D path yet"
    if problem.use_alibi:
        return False, "ALiBi slopes are not enabled in CK DSL attention yet"
    if problem.use_qq_bias:
        return False, "QQ bias is not enabled in CK DSL attention yet"
    return True, "supported by scalar CK DSL 2D attention backend"


def supports_native_unified_attention_tiled(
    problem: UnifiedAttentionProblem,
) -> Tuple[bool, str]:
    """Return whether the optimized tiled MFMA path can run this problem."""
    return supports_tiled_2d(
        head_size=problem.head_size,
        block_size=problem.block_size,
        dtype=problem.dtype,
        num_queries_per_kv=problem.num_queries_per_kv,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        use_fp8=problem.use_fp8,
        q_dtype=problem.q_dtype,
        num_warps=_select_2d_num_warps(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size=_select_2d_tile_size(problem),
    )


def supports_native_unified_attention_3d_tiled(
    problem: UnifiedAttentionProblem,
) -> Tuple[bool, str]:
    """Return whether the optimized tiled MFMA 3D split-KV path can run this."""
    return supports_tiled_3d(
        head_size=problem.head_size,
        block_size=problem.block_size,
        dtype=problem.dtype,
        num_queries_per_kv=problem.num_queries_per_kv,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        use_fp8=problem.use_fp8,
        q_dtype=problem.q_dtype,
        kv_storage_dtype=_kv_storage_dtype(problem),
    )


_ATTN_CACHE: Dict[Tuple, bytes] = {}
_ATTN_TILED_CACHE: Dict[Tuple, bytes] = {}
_ATTN_3D_TILED_CACHE: Dict[Tuple, Tuple[bytes, str, bytes, str]] = {}


def _cache_key(problem: UnifiedAttentionProblem) -> Tuple:
    return (
        "scalar",
        problem.total_q,
        problem.num_seqs,
        problem.num_query_heads,
        problem.num_kv_heads,
        problem.head_size,
        problem.block_size,
        problem.max_seqlen_q,
        problem.max_seqlen_k,
        problem.dtype,
        problem.sliding_window,
        bool(problem.use_sinks),
        bool(problem.softcap > 0),
    )


def _select_2d_tile_size(problem: UnifiedAttentionProblem) -> int:
    """Choose ``tile_size`` (T) for the tiled 2D kernel.

    ``T`` is the number of KV tokens consumed per outer-loop iter (per
    kernel iteration). Larger ``T`` amortises the outer-loop overhead
    (block-table lookup, async-DMA issue, softmax/PV scheduling) but
    only pays off if there is enough KV per CTA to keep the multi-block
    descriptor's wave-uniform path filled.

    **Universal ``T = 2 * BS``** post the Q-in-registers + single-buffer-V
    refactor: the LDS savings (8 KiB Q + 8 KiB V) make the multi-block
    path fit comfortably for every workload class on MI355X. For decode,
    the higher per-iter amortization beats the smaller-tile choice by
    ~24% (measured ``/workspace/probe_prefill_sweep.py``: decode_b1
    drops 34 µs → 26 µs).

    The kernel's own gate (``supports_tiled_2d``) re-validates the
    choice against the per-wave-tokens / LDS-budget constraints.

    **FP8 sliding-window long-prefill exception** (round-2 cluster-B
    sweep, ``/workspace/rounds/_summaries/prod_nw_sweep.log``): when the
    sliding window prunes the kv-loop to a handful of iters per CTA,
    bigger T over-allocates LDS without amortising any per-iter cost.
    For ``use_fp8 + sliding_window > 0 + max_seqlen_q > 256``, drop to
    ``T = block_size`` (single block per iter, 32 tokens) — measured
    1.15-1.30× win on every FP8 SW long-prefill shape.
    """
    # Sliding-window long-prefill FP8 exception. The latest broad sweep
    # confirmed this should stay FP8-only: for bf16 SW long-prefill the
    # correctness-clean winner was T=64, not T=32
    # (``/workspace/trace_bench/sweep_attention2d_configs.json``:
    # bf16_sw_n16_q1000_k1050 best T=64/mw16/hipcc at ~257us; T=32
    # variants were >=258us and often incorrect under hipcc).
    if problem.use_fp8 and problem.sliding_window > 0 and problem.max_seqlen_q > 256:
        return problem.block_size
    # T = 4 * block_size = 128 was tested on bf16 long-prefill (the
    # ``n=402 q=1000 k=1050`` regression bucket) and got WORSE, not
    # better: 333us → 953us. The reason is the async DMA loader uses
    # ``kv_calls_per_tile = (T * HD) // KV_HALVES_PER_CALL`` issuing
    # one ``raw.ptr.buffer.load.lds`` per call. Doubling T doubles the
    # call count which saturates the VMEM issue queue (16→32 calls per
    # tile per warp). The pipeline back-pressure cost outweighs the
    # halved iter count. T=2*bs remains the sweet spot for non-SW.
    return 2 * problem.block_size


def _select_2d_num_warps(problem: UnifiedAttentionProblem) -> int:
    """Choose ``num_warps`` for the tiled 2D kernel.

    The kernel supports ``num_warps in {1, 2, 4, 8}`` (each warp owns 16
    rows of ``BLOCK_M``; no cross-warp reduction). The kernel also has a
    hard ceiling ``T * HD >= num_warps * 64 * 8`` — at ``THREADS *
    halves_per_call`` per call we need that many halves available in the
    per-tile KV slab, otherwise the async DMA underfills.

    Tuning thresholds (calibrated against the trace shapes documented in
    ``/workspace/trace_bench_report.md`` on MI355X / gfx950; the
    ``warps``-sweep harness is at ``/workspace/probe_prefill_sweep.py``
    and the prefill-time harness is at
    ``/workspace/probe_prefill_time.py``):

    **Post Q-in-registers + single-buffer-V refactor** (measured at
    ``/workspace/probe_prefill_sweep.py`` on MI355X):

      ``q <= 64``    (decode + tiny prefill) -> 1   (small grid; tiny
                                                     per-CTA work)
      ``q <= 128``                           -> 2
      ``q in (128, 256]``                    -> 4   (medium prefill;
                                                     BLOCK_M=64 wins
                                                     against q=256's
                                                     CTA count)
      ``q > 256``                            -> 2   (NW=2 dominates large
                                                     prefill after the
                                                     LDS savings — was
                                                     NW=4 in the pre-
                                                     refactor heuristic)

    The result is further clamped against:

      - the architectural ceiling above (``T * HD >= num_warps * 512``);
      - the per-wave-tokens constraint (``WAVE*8/HD <= BS``);
      - an LDS-budget check (``<= 96 KiB`` so we keep >= 1 CTA/CU on
        MI355X comfortably).
    """
    if problem.max_seqlen_q <= 64:
        target = 1
    elif problem.max_seqlen_q <= 128:
        target = 2
    elif problem.max_seqlen_q <= 256:
        target = 4
    elif problem.num_seqs <= 1:
        # Long prefill with num_seqs <= 1: the production-shape sweep
        # (``/workspace/rounds/_summaries/prefill_n_sweep.log``) shows
        # nw=2 beats nw=4 by 3-5% at n=1 (fewer CTAs hurt under-saturated
        # GPU at single-seq). The crossover happens at n=2 where nw=4
        # takes over (4-16% faster than nw=2 from n=2 up to bench-cap n=16).
        # Single-seq prefill is the only production regime where nw=2 wins.
        target = 2
    else:
        # Long prefill (q > 256) with num_seqs >= 2: nw=4 wins.
        # Round-1 cluster-A sweep
        # (``/workspace/rounds/round_01_bf16_q1000_v1/cluster_a_sweep.md``)
        # showed ``nw=4 mw=16 T=2*BS`` beats ``nw=2`` on 14 of 17 long-prefill
        # bf16 targets by 1.04-1.87× (no regressions). Resource analysis
        # (``resources.json``): nw=4 keeps WGs/CU at 4 (vs default 5), VGPR
        # at 121-126 (no spills), LDS at 36 KB (fits 4 WG/CU comfortably);
        # BLOCK_M=64 (BLOCK_Q=8) cuts the q-block CTA count in half vs
        # BLOCK_M=32 (BLOCK_Q=4), reducing per-CTA dispatch overhead.
        target = 4

    HD = problem.head_size
    BS = problem.block_size
    T = _select_2d_tile_size(problem)
    WORK_BYTES = 2
    # Step down until all constraints are satisfied.
    while target > 1:
        THREADS = 64 * target
        BLOCK_M = 16 * target
        # Architectural ceiling: T * HD halves must satisfy at least one
        # async-DMA call's worth of lane-contiguous payload.
        if (T * HD) < THREADS * 8:
            target //= 2
            continue
        # Per-wave tokens must fit within one block (wave-uniform
        # block_table lookup constraint, enforced by supports_tiled_2d).
        per_wave_tokens = (64 * 8) // HD
        if per_wave_tokens > BS:
            target //= 2
            continue
        lds_bytes = (
            BLOCK_M * HD * WORK_BYTES
            + 2 * T * HD * WORK_BYTES
            + 2 * T * HD * WORK_BYTES
            + BLOCK_M * T * WORK_BYTES
            + BLOCK_M * HD * 4
        )
        if lds_bytes <= 96 * 1024:
            break
        target //= 2
    return max(1, target)


def _kv_storage_dtype(problem: UnifiedAttentionProblem) -> Optional[str]:
    """Return ``"fp8e4m3"`` for the FP8 K/V cache path, else ``None``.

    The upstream API uses ``use_fp8=True`` to flip into the FP8 K/V cache
    code path (with bf16/fp16 query, bf16/fp16 output, per-tensor
    ``k_scale``/``v_scale``). The kernel takes ``kv_storage_dtype`` so the
    same plumbing can later add bf8e5m2 or other low-precision K/V.
    """
    return "fp8e4m3" if problem.use_fp8 else None


def _tiled_cache_key(problem: UnifiedAttentionProblem) -> Tuple:
    """Compute the tiled-2D cache key WITHOUT building the spec dataclass.

    On the hot path (every launch) we just need a hashable tuple to
    look up the cached launcher. Building a full ``UnifiedAttention2DTiledSpec``
    dataclass (17 fields, frozen=True validation) adds ~3µs per launch
    which is material on decode kernels (~20µs). The spec is only built
    on cache miss (first launch per shape) inside ``_get_2d_launcher``.

    Note: this MUST match the spec-derived key — every selector knob
    that affects the kernel build is included. If a selector changes
    behaviour, update both this function and ``_tiled_spec_from_problem``.
    """
    return (
        "tiled",
        problem.num_seqs,
        problem.num_query_heads,
        problem.num_kv_heads,
        problem.head_size,
        problem.block_size,
        problem.dtype,
        problem.sliding_window,
        bool(problem.use_sinks),
        bool(problem.softcap > 0),
        bool(problem.use_alibi),
        bool(problem.use_qq_bias),
        _select_2d_num_warps(problem),
        _kv_storage_dtype(problem),
        _select_2d_tile_size(problem),
        _select_2d_waves_per_eu(problem),
        _select_2d_block_m_per_warp(problem),
        _enable_mfma_32x32(problem),
        _enable_transposed_qk_32x32(problem),
        _enable_register_pv(problem),
        _select_2d_compile_backend(problem),
    )


def _select_2d_waves_per_eu(problem: UnifiedAttentionProblem) -> Optional[int]:
    """Choose ``waves_per_eu`` for the tiled 2D kernel.

    Triton's ``select_2d_config`` uses ``waves_per_eu=2`` for every config
    (verified at ``/workspace/aiter/aiter/ops/triton/attention/unified_attention.py``).
    We match that: it gives the LLVM backend more VGPR headroom per wave
    (less risk of spill to scratch / LDS) while still meeting the
    occupancy targets (the double-buffered K/V kernel runs at 2-3 WGs/CU
    on MI355X depending on shape; pushing for wpe=3 was a marginal
    +5% on isolated workloads but no consistent gain over the full
    workload spectrum once we A/B'd it against Triton's choice).

    If the problem itself pinned ``waves_per_eu`` (via the public
    ``UnifiedAttentionProblem.waves_per_eu`` field), respect that.
    """
    if problem.waves_per_eu is not None:
        return problem.waves_per_eu
    # FP8 long-prefill specialisation. The sync FP8 dequant path runs
    # extra VALU per iter (cvt_pk_f32_fp8 + scale + cvt_to_bf16 per K/V
    # tile element) which makes the inner loop more VALU-bound than the
    # bf16 path. ``waves_per_eu=3`` reduces the VGPR-per-wave budget so
    # the compiler can schedule more concurrent waves on the same SIMD
    # to hide the extra VALU. Sweep on ``n=16 q=1024 k=4096 fp8 no-sw``
    # (the dominant trace bucket): wpe=2 → 4383us, wpe=3 → 3531us = 1.24×
    # win, measured via /tmp/sweep_long_prefill.py. For SHORT prefill /
    # decode the VALU pressure is already low so wpe=2 stays better.
    if problem.use_fp8 and problem.max_seqlen_q > 256 and problem.num_seqs >= 2:
        return 3
    return 2


def _fp8_qk_loader_fits(problem: UnifiedAttentionProblem) -> bool:
    """True iff the async-fp8 K loader can tile the per-iter K bytes.

    ``raw.ptr.buffer.load.lds`` accepts dwords ∈ {1, 3, 4} = {4, 12, 16}
    bytes per lane (a hardware quirk -- dwords=2 is rejected). We need
    one of those per-call payloads to evenly tile ``T*HD`` bytes given
    ``num_warps * 64`` lanes per CTA.
    """
    tile_bytes = _select_2d_tile_size(problem) * problem.head_size
    threads = _select_2d_num_warps(problem) * 64
    for bpl in (16, 12, 4):
        payload = threads * bpl
        if tile_bytes >= payload and tile_bytes % payload == 0:
            return True
    return False


def _enable_fp8_mfma_qk(problem: UnifiedAttentionProblem) -> bool:
    """Heuristic: enable the ULP-correct fp8-K-LDS path when it helps.

    The path is bit-identical to the sync-dequant default. The win
    pattern from the production trace:
      - decode / short prefill: wins 10-55% (loader LDS writes are the
        bottleneck; we save them by storing K as fp8 in LDS).
      - long prefill no-SW (many KV iters * many MFMAs): loses ~10%
        (per-MFMA in-register dequant accumulates faster than the
        loader LDS writes we replaced).
    Gate on (sliding-window OR ``max_seqlen_k <= 16 * T``) plus the
    loader fitness check.
    """
    if not problem.use_fp8:
        return False
    if not _fp8_qk_loader_fits(problem):
        return False
    T_eff = _select_2d_tile_size(problem)
    return problem.sliding_window > 0 or problem.max_seqlen_k <= 16 * T_eff


def _enable_mfma_32x32(problem: UnifiedAttentionProblem) -> bool:
    """Enable the in-kernel 32x32x16 migration on shapes where it wins.

    The transposed 32x32 path (``use_mfma_32x32=True`` +
    ``use_transposed_qk_32x32=True``) has been parity-validated against the
    default 16x16x32 path across 17 representative shapes (bf16 long /
    short prefill, multi-batch, sliding-window, GQA, hd64/128/256,
    decode). Measured speedup vs default (MI355X, bf16):

      * multi-batch prefill (num_seqs >= 2, max_seqlen_q >= 256):
        1.21-1.48x on hd64, 1.28-1.39x on hd128
      * single-batch long prefill (num_seqs == 1): 0.74-0.85x (slower)
      * decode / very short prefill: ~tie

    The win pattern is driven by the softmax: the transposed layout
    reduces K-reduce work from 5 stages of intra-32-lane butterfly to a
    single cross-half xor, and amortises that win across the multi-CTA
    parallelism that multi-batch shapes provide. Single-batch prefill
    has fewer CTAs to absorb the (still-present) PV scalar-V-load and
    PT cross-half xor overhead.

    The non-transposed 32x32 path is currently slower than default on
    every shape we tested (its PV uses P_lds with the standard
    ds_read_tr16 V reader but pays the higher register pressure), so we
    only enable mfma_32x32 in conjunction with the transposed flag.
    """
    return _enable_transposed_qk_32x32(problem)


def _enable_transposed_qk_32x32(problem: UnifiedAttentionProblem) -> bool:
    """Heuristic for the transposed-K layout.

    The transposed path requires ``block_m_per_warp == 32`` (the M32N32K16
    MFMA shape) so the conditions here MUST be a strict subset of the
    ``_select_2d_block_m_per_warp`` conditions that pick ``mw=32``. The
    latter requires ``max_seqlen_q > 256 AND num_seqs >= 2 AND not (sw
    > 0 AND not use_fp8)``. Adding extra gates beyond those silently
    breaks the post-init validation in the spec dataclass.

    Beyond the mw=32 prereq we gate on:

      * dtype == bf16 (fp16/fp8 paths still use the default kernel)
      * no FP8 K/V (transposed path doesn't dequant K/V from fp8 yet)
      * no ALiBi or QQ bias (transposed mask block doesn't fold them yet)
      * head_size in {64, 128} (hd=256 not benchmarked yet)
      * no softcap / sinks (not wired into transposed softmax yet)
    """
    if problem.dtype != "bf16":
        return False
    if problem.use_fp8:
        return False
    if problem.use_alibi or problem.use_qq_bias:
        return False
    if problem.softcap > 0 or problem.use_sinks:
        return False
    if problem.head_size not in (64, 128):
        return False
    # Must match _select_2d_block_m_per_warp's mw=32 conditions exactly.
    if not (problem.max_seqlen_q > 256 and problem.num_seqs >= 2):
        return False
    if problem.sliding_window > 0 and not problem.use_fp8:
        return False
    return True


def _enable_register_pv(problem: UnifiedAttentionProblem) -> bool:
    """Enable register-resident P for the existing 16x16x32 2D path.

    Hard-disabled by default while the lane transform is being validated.
    Tests/experiments can monkey-patch this selector; once parity and trace
    benches are clean it can be enabled for bf16 no-window long-prefill.
    """
    return False


def _tiled_spec_from_problem(
    problem: UnifiedAttentionProblem,
) -> UnifiedAttention2DTiledSpec:
    return UnifiedAttention2DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
        num_warps=_select_2d_num_warps(problem),
        waves_per_eu=_select_2d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size=_select_2d_tile_size(problem),
        block_m_per_warp=_select_2d_block_m_per_warp(problem),
        use_mfma_32x32=_enable_mfma_32x32(problem),
        use_transposed_qk_32x32=_enable_transposed_qk_32x32(problem),
        use_register_pv=_enable_register_pv(problem),
        use_fp8_mfma_qk=_enable_fp8_mfma_qk(problem),
    )


def _select_2d_block_m_per_warp(problem: UnifiedAttentionProblem) -> int:
    """Choose ``block_m_per_warp`` for the tiled 2D kernel.

    ``block_m_per_warp=32`` stacks two MFMA-M=16 atoms per warp so each
    warp processes 32 query rows instead of 16. This doubles per-CTA
    work and halves the CTA count.

    Empirical sweep on the bench-harness slowest shapes (see
    ``/tmp/sweep_long_prefill.py`` results):

      - **fp8 long-prefill**, ``n=16 q=1024 k=4096 no-sw``: ``mw=32 T=64``
        beats ``mw=16 T=64`` by 6% (4669us → 4383us). The sync FP8
        dequant cost per CTA stays roughly constant when M doubles
        because the dequant cost is per-K-byte not per-M-row, so
        halving the CTA count cleanly halves the total dequant cost.
        Production sweep on n=4: 1.08-1.32× wins.

      - **bf16 long-prefill no-sw**, ``n=16 q=1000 k=1050``: ``mw=32 T=64``
        beats ``mw=16 T=64`` by 6% (712us → 668us). The per-CTA
        prelude amortisation (binary search for seq_idx, Q gather,
        Acc zero, sinks load) cleanly halves with the CTA count, and
        the per-iter MFMA + LDS-staging overhead doubles but stays
        below the per-CTA prelude savings.

      - **bf16 long-prefill SW**, ``n=16 q=1000 k=1050 sw=128``:
        the broad sweep revised the earlier local result: correctness-
        clean best is ``mw=16 T=64 hipcc`` at ~257us. ``mw=32`` and
        T=32 do not beat it consistently, and several hipcc T=32
        combinations are numerically wrong. Keep bf16 SW on mw=16.

    Cost: VGPR pressure rises (each warp tracks 32 rows × QK_N_TILES +
    32 rows × PV_N_TILES of f32 accumulators), pushing occupancy from
    4 → 2 CTAs/CU. For long-prefill the per-CTA throughput gain wins
    the trade. For SHORT prefill / decode shapes the per-CTA prelude
    is already negligible (short kv-loop hides it inside the very few
    iters), and the 4→2 CTAs/CU occupancy loss costs more than the
    per-CTA throughput gain — so keep mw=16 there.

    Gate: ``max_seqlen_q > 256 and num_seqs >= 2``. Below this, mw=16
    is consistently within noise of mw=32 in the per-shape sweep.
    """
    if (
        problem.max_seqlen_q > 256
        and problem.num_seqs >= 2
        and not (problem.sliding_window > 0 and not problem.use_fp8)
    ):
        return 32
    return 16


def _num_segments(problem: UnifiedAttentionProblem) -> int:
    """Mirror AITER ``select_3d_config`` num_segments derivation exactly."""
    attn_cfg, _ = problem.select_3d()
    return attn_cfg.NUM_SEGMENTS_PER_SEQ


def _tiled_3d_spec_from_problem(
    problem: UnifiedAttentionProblem,
) -> UnifiedAttention3DTiledSpec:
    return UnifiedAttention3DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0,
        num_segments=_num_segments(problem),
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        num_seqs=problem.num_seqs,
        waves_per_eu=problem.waves_per_eu,
        kv_storage_dtype=_kv_storage_dtype(problem),
    )


def _tiled_3d_cache_key(problem: UnifiedAttentionProblem) -> Tuple:
    return (
        "tiled3d",
        problem.num_seqs,
        problem.num_query_heads,
        problem.num_kv_heads,
        problem.head_size,
        problem.block_size,
        problem.dtype,
        problem.sliding_window,
        bool(problem.use_sinks),
        bool(problem.softcap > 0),
        bool(problem.use_alibi),
        bool(problem.use_qq_bias),
        _num_segments(problem),
        _kv_storage_dtype(problem),
    )


def _3d_signature(dtype: str, *, kv_dtype: Optional[str] = None):
    from ..helpers.spec import SignatureBuilder

    io_dtype = "f16" if dtype == "fp16" else "bf16"
    kv_io = kv_dtype if kv_dtype else io_dtype
    return (
        SignatureBuilder()
        .ptr("segm_output_ptr", "f32")
        .ptr("segm_max_ptr", "f32")
        .ptr("segm_expsum_ptr", "f32")
        .ptr("query_ptr", io_dtype)
        .ptr("key_cache_ptr", kv_io)
        .ptr("value_cache_ptr", kv_io)
        .ptr("sink_ptr", io_dtype)
        .ptr("block_tables_ptr", "i32")
        .ptr("seq_lens_ptr", "i32")
        .ptr("alibi_slopes_ptr", "f32")
        .ptr("qq_bias_ptr", "f32")
        .ptr("query_start_len_ptr", "i32")
        .scalar("scale", "f32")
        .scalar("k_scale", "f32")
        .scalar("v_scale", "f32")
        .scalar("softcap", "f32")
        .scalar("num_seqs", "i32")
        .scalar("block_table_stride", "i32")
        .scalar("qq_bias_stride_0", "i32")
        .build()
    )


def _reduce_signature(dtype: str):
    from ..helpers.spec import SignatureBuilder

    io_dtype = "f16" if dtype == "fp16" else "bf16"
    return (
        SignatureBuilder()
        .ptr("output_ptr", io_dtype)
        .ptr("segm_output_ptr", "f32")
        .ptr("segm_max_ptr", "f32")
        .ptr("segm_expsum_ptr", "f32")
        .ptr("seq_lens_ptr", "i32")
        .build()
    )


def _attn_signature(
    dtype: str,
    *,
    include_bt_stride: bool,
    include_qq_bias_stride: bool = False,
    kv_dtype: Optional[str] = None,
):
    from ..helpers.spec import SignatureBuilder

    io_dtype = "f16" if dtype == "fp16" else "bf16"
    # K/V cache dtype defaults to the working dtype (bf16/fp16). The FP8
    # K/V path passes ``kv_dtype="fp8e4m3"`` so the signature uses 1-byte
    # pointers for K/V cache instead of 2-byte.
    kv_io = kv_dtype if kv_dtype else io_dtype
    sb = (
        SignatureBuilder()
        .ptr("output_ptr", io_dtype)
        .ptr("query_ptr", io_dtype)
        .ptr("key_cache_ptr", kv_io)
        .ptr("value_cache_ptr", kv_io)
        .ptr("sink_ptr", io_dtype)
        .ptr("block_tables_ptr", "i32")
        .ptr("seq_lens_ptr", "i32")
        .ptr("alibi_slopes_ptr", "f32")
        .ptr("qq_bias_ptr", "f32")
        .ptr("query_start_len_ptr", "i32")
        .scalar("scale", "f32")
        .scalar("k_scale", "f32")
        .scalar("v_scale", "f32")
        .scalar("out_scale", "f32")
        .scalar("softcap", "f32")
        .scalar("num_seqs", "i32")
    )
    if include_bt_stride:
        sb.scalar("block_table_stride", "i32")
    if include_qq_bias_stride:
        sb.scalar("qq_bias_stride_0", "i32")
    return sb.build()


def _attn_values(
    *,
    problem: UnifiedAttentionProblem,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    softmax_scale: float,
    block_table,
    softcap: float,
    sinks,
    bt_stride: int,
    include_bt_stride: bool,
    alibi_slopes=None,
    qq_bias=None,
    qq_bias_stride_0: int = 0,
    include_qq_bias_stride: bool = False,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out_scale: float = 1.0,
):
    vals = {
        "output_ptr": out,
        "query_ptr": q,
        "key_cache_ptr": k,
        "value_cache_ptr": v,
        "sink_ptr": sinks,
        "block_tables_ptr": block_table,
        "seq_lens_ptr": seqused_k,
        "alibi_slopes_ptr": alibi_slopes if alibi_slopes is not None else 0,
        "qq_bias_ptr": qq_bias if qq_bias is not None else 0,
        "query_start_len_ptr": cu_seqlens_q,
        "scale": float(softmax_scale),
        "k_scale": float(k_scale),
        "v_scale": float(v_scale),
        "out_scale": float(out_scale),
        "softcap": float(softcap),
        "num_seqs": int(problem.num_seqs),
    }
    if include_bt_stride:
        vals["block_table_stride"] = int(bt_stride)
    if include_qq_bias_stride:
        vals["qq_bias_stride_0"] = int(qq_bias_stride_0)
    return vals


def _run_3d_tiled(
    *,
    problem: UnifiedAttentionProblem,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    softmax_scale: float,
    block_table,
    softcap: float,
    sinks,
    bt_stride: int,
    warmup: int,
    attempts: int,
    alibi_slopes=None,
    qq_bias=None,
    qq_bias_stride_0: int = 0,
    stream: int = 0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    """Launch the tiled 3D segment + reduce kernels.

    Mirrors AITER's 3D path:
      1. Compile (and cache) both kernels for this problem shape.
      2. Allocate the per-segment workspace tensors `segm_output`,
         `segm_max`, `segm_expsum`.
      3. Launch the 3D segment kernel with grid
         `(total_num_q_blocks, num_kv_heads, num_segments)`.
      4. Launch the reduce kernel with grid `(total_q, num_query_heads, 1)`.
    """
    num_segments = _num_segments(problem)
    cache_key = _tiled_3d_cache_key(problem)

    # Lazily build (and cache) the PipelineLauncher + WorkspacePool for
    # this problem shape. This single object owns: the compiled HSACO
    # blobs, the loaded HIP module handles, the kernel function
    # handles, and the segm_* workspace tensors. All five
    # categories of lifetime / race / overhead bugs documented in
    # ``ck_dsl/runtime/launcher.py`` are removed by construction; the
    # only remaining per-call cost is packing args and issuing two
    # ``hipModuleLaunchKernel`` calls on the caller's stream.
    pipeline, pool = _get_3d_pipeline(problem, cache_key, num_segments)
    workspace = pool.prepare(
        _attention_3d_workspace_specs(problem, num_segments, q.device)
    )
    segm_output = workspace["segm_output"]
    segm_max = workspace["segm_max"]
    segm_expsum = workspace["segm_expsum"]

    block_q = (
        16 // problem.num_queries_per_kv if problem.num_queries_per_kv <= 16 else 1
    )
    total_num_q_blocks = problem.total_q // block_q + problem.num_seqs

    seg_vals = {
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "query_ptr": q,
        "key_cache_ptr": k,
        "value_cache_ptr": v,
        "sink_ptr": sinks,
        "block_tables_ptr": block_table,
        "seq_lens_ptr": seqused_k,
        "alibi_slopes_ptr": alibi_slopes if alibi_slopes is not None else 0,
        "qq_bias_ptr": qq_bias if qq_bias is not None else 0,
        "query_start_len_ptr": cu_seqlens_q,
        "scale": float(softmax_scale),
        "k_scale": float(k_scale),
        "v_scale": float(v_scale),
        "softcap": float(softcap),
        "num_seqs": int(problem.num_seqs),
        "block_table_stride": int(bt_stride),
        "qq_bias_stride_0": int(qq_bias_stride_0),
    }
    red_vals = {
        "output_ptr": out,
        "segm_output_ptr": segm_output,
        "segm_max_ptr": segm_max,
        "segm_expsum_ptr": segm_expsum,
        "seq_lens_ptr": seqused_k,
    }
    seg_grid = (
        int(total_num_q_blocks),
        int(problem.num_kv_heads),
        int(num_segments),
    )
    seg_block = (64, 1, 1)
    red_grid = (int(problem.total_q), int(problem.num_query_heads), 1)
    red_block = (64, 1, 1)

    # M1: drive the segment + reduce chain through the CK-Tile-style
    # primitive instead of ``PipelineLauncher.__call__``. ``pipeline``
    # is still the cache anchor (one ``KernelLauncher`` per stage,
    # built once and held over the problem's lifetime); we just
    # extract its stages here and bake one closure per stage with
    # :func:`make_kernel`. ``launch_kernel`` then submits both
    # closures on ``stream`` in declaration order under
    # :func:`no_fence`. Same-stream FIFO ordering still guarantees
    # the reduce kernel observes the segment kernel's writes.
    seg_launcher, red_launcher = pipeline.stages
    launch_kernel(
        StreamConfig(stream_id=int(stream)),
        make_kernel(seg_launcher, seg_vals, seg_grid, seg_block),
        make_kernel(red_launcher, red_vals, red_grid, red_block),
    )
    # Preserve :class:`PipelineLauncher`'s implicit last-stage fence
    # semantic when not under an outer :func:`no_fence` context. The
    # closures produced by :func:`make_kernel` are always
    # ``fence=False``, and :func:`launch_kernel` itself does not
    # implicitly sync on the non-timing path -- so without this
    # explicit drain, callers that read the output tensor on the
    # host immediately after :func:`run_unified_attention_torch`
    # returns would race the reduce kernel. ``_resolved_fence(True)``
    # returns False inside :func:`time_launches`'s :func:`no_fence`
    # body (so the timing loop's outer event-sync remains the only
    # sync point) and True everywhere else.
    if _resolved_fence(True):
        wait_stream_and_release(int(stream))
    return LaunchSummary(launches=2)


# Per-cache-key (pipeline, workspace_pool) pairs. Built lazily at first
# dispatch for a given problem shape; reused across every subsequent
# dispatch and every timing-loop iteration. This is the same shape as
# CK Tile's `fmha_bwd_launcher` (one object per problem instance, owns
# kernels + workspace, survives every launch).
_3D_PIPELINES: Dict[Tuple, Tuple[PipelineLauncher, WorkspacePool]] = {}
_2D_LAUNCHERS: Dict[Tuple, KernelLauncher] = {}
_SCALAR_LAUNCHERS: Dict[Tuple, KernelLauncher] = {}


def _attention_3d_workspace_specs(
    problem: UnifiedAttentionProblem,
    num_segments: int,
    device,
) -> Tuple[WorkspaceSpec, WorkspaceSpec, WorkspaceSpec]:
    """CK Tile-style workspace declaration for the split-KV 3D pipeline.

    This is the Python equivalent of FMHA forward split-KV's
    `lse_acc_ptr` + `o_acc_ptr` sizing in
    `example/ck_tile/01_fmha/fmha_fwd_runner.hpp`: all scratch shapes
    are derived from the problem up front, owned by a long-lived pool,
    and passed to the segment and reduce kernels by pointer.
    """
    try:
        import torch

        f32 = torch.float32
    except Exception:
        # CPU-only/static tests can still ask for byte accounting without
        # importing torch. WorkspacePool.required_nbytes understands this
        # string fallback via `_dtype_element_size`.
        f32 = "float32"

    return (
        WorkspaceSpec(
            "segm_output",
            (problem.total_q, problem.num_query_heads, num_segments, problem.head_size),
            f32,
            device,
        ),
        WorkspaceSpec(
            "segm_max",
            (problem.total_q, problem.num_query_heads, num_segments),
            f32,
            device,
        ),
        WorkspaceSpec(
            "segm_expsum",
            (problem.total_q, problem.num_query_heads, num_segments),
            f32,
            device,
        ),
    )


def attention_3d_workspace_nbytes(
    problem: UnifiedAttentionProblem,
    *,
    device=None,
) -> int:
    """Return required split-KV 3D workspace bytes for `problem`.

    Public helper for tests/bench harnesses that want to report scratch
    usage before dispatch. The `device` value only matters for the
    eventual allocation, not byte accounting.
    """
    return WorkspacePool.required_nbytes(
        _attention_3d_workspace_specs(problem, _num_segments(problem), device)
    )


def _get_3d_pipeline(
    problem: UnifiedAttentionProblem,
    cache_key: Tuple,
    num_segments: int,
) -> Tuple[PipelineLauncher, WorkspacePool]:
    if cache_key in _3D_PIPELINES:
        return _3D_PIPELINES[cache_key]
    if cache_key not in _ATTN_3D_TILED_CACHE:
        seg_spec = _tiled_3d_spec_from_problem(problem)
        reduce_spec = UnifiedAttentionReduceTiledSpec(
            head_size=problem.head_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            num_segments=num_segments,
        )
        seg_art = compile_kernel(
            build_unified_attention_3d_tiled(seg_spec), capture_ir_text=False
        )
        red_art = compile_kernel(
            build_unified_attention_reduce_tiled(reduce_spec), capture_ir_text=False
        )
        _ATTN_3D_TILED_CACHE[cache_key] = (
            seg_art.hsaco,
            seg_art.kernel_name,
            red_art.hsaco,
            red_art.kernel_name,
        )
    seg_hsaco, seg_kname, red_hsaco, red_kname = _ATTN_3D_TILED_CACHE[cache_key]
    seg_launcher = KernelLauncher(
        hsaco=seg_hsaco,
        kernel_name=seg_kname,
        signature=_3d_signature(problem.dtype, kv_dtype=_kv_storage_dtype(problem)),
        cache_key=("3d_seg",) + cache_key,
    )
    red_launcher = KernelLauncher(
        hsaco=red_hsaco,
        kernel_name=red_kname,
        signature=_reduce_signature(problem.dtype),
        cache_key=("3d_red",) + cache_key,
    )
    pipeline = PipelineLauncher([seg_launcher, red_launcher])
    pool = WorkspacePool()
    _3D_PIPELINES[cache_key] = (pipeline, pool)
    return pipeline, pool


def _select_2d_compile_backend(problem: UnifiedAttentionProblem) -> str:
    """Pick the compile backend (LLVM-direct vs hipcc) for the 2D tiled kernel.

    The HIP path (``hipcc --genco``) is measurably faster than the
    LLVM-direct path on large-batch bf16/fp16 prefill (≈5% on
    ``b4_q1000_kv1000``) because clang's frontend + AMDGPU backend
    pipeline does heavier instruction scheduling for the long
    unrolled loop body. Smaller workloads (decode, small prefill)
    are 5-29% slower via hipcc, so the auto-selector only switches
    when the workload is large enough to amortize hipcc's ~450ms
    compile cost AND benefit from its scheduling.

    The FP8 K/V path uses sync-dequant loaders with intrinsics that
    the HIP debug backend may not fully cover; the auto-selector
    pins FP8 to ``llvm`` until ``hipcc`` is validated for that
    code path.

    See ``/workspace/probe_hip_lowering.py`` for the per-shape sweep.
    """
    if problem.compile_backend in ("llvm", "hipcc"):
        return problem.compile_backend
    # Auto: HIP for large-batch bf16/fp16 prefill workloads where
    # hipcc's heavier scheduler measurably wins. FP8 stays on LLVM
    # until the HIP path is exercised on the dequant-loader kernels.
    if problem.use_fp8:
        return "llvm"
    total_work = problem.num_seqs * max(problem.max_seqlen_q, 1)
    if problem.max_seqlen_q > 512 and total_work > 1024:
        return "hipcc"
    return "llvm"


def _get_2d_launcher(
    problem: UnifiedAttentionProblem,
    cache_key: Tuple,
) -> KernelLauncher:
    if cache_key in _2D_LAUNCHERS:
        return _2D_LAUNCHERS[cache_key]
    if cache_key not in _ATTN_TILED_CACHE:
        spec = _tiled_spec_from_problem(problem)
        kernel = build_unified_attention_2d_tiled(spec)
        backend = _select_2d_compile_backend(problem)
        if backend == "hipcc":
            from ..helpers.compile import compile_kernel_via_hipcc

            artifact = compile_kernel_via_hipcc(kernel)
        else:
            artifact = compile_kernel(kernel, capture_ir_text=False)
        _ATTN_TILED_CACHE[cache_key] = (artifact.hsaco, artifact.kernel_name)
    hsaco, kname = _ATTN_TILED_CACHE[cache_key]
    launcher = KernelLauncher(
        hsaco=hsaco,
        kernel_name=kname,
        signature=_attn_signature(
            problem.dtype,
            include_bt_stride=True,
            include_qq_bias_stride=True,
            kv_dtype=_kv_storage_dtype(problem),
        ),
        cache_key=("2d",) + cache_key,
    )
    _2D_LAUNCHERS[cache_key] = launcher
    return launcher


def _get_scalar_launcher(
    problem: UnifiedAttentionProblem,
    cache_key: Tuple,
) -> KernelLauncher:
    if cache_key in _SCALAR_LAUNCHERS:
        return _SCALAR_LAUNCHERS[cache_key]
    if cache_key not in _ATTN_CACHE:
        spec = UnifiedAttention2DSpec(problem=problem)
        artifact = compile_kernel(
            build_unified_attention_2d(spec), capture_ir_text=False
        )
        _ATTN_CACHE[cache_key] = (artifact.hsaco, artifact.kernel_name)
    hsaco, kname = _ATTN_CACHE[cache_key]
    launcher = KernelLauncher(
        hsaco=hsaco,
        kernel_name=kname,
        signature=_attn_signature(problem.dtype, include_bt_stride=False),
        cache_key=("scalar",) + cache_key,
    )
    _SCALAR_LAUNCHERS[cache_key] = launcher
    return launcher


def run_unified_attention_torch(
    *,
    problem: UnifiedAttentionProblem,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    softmax_scale: float,
    block_table,
    softcap: float,
    sinks=None,
    alibi_slopes=None,
    qq_bias=None,
    qq_bias_stride_0: int = 0,
    warmup: int = 0,
    attempts: int = 1,
    backend: str = "auto",
    stream: int = 0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out_scale: float = 1.0,
):
    """Launch a CK DSL attention kernel on torch tensors.

    Backend selection:
      - `"tiled"`: force the optimized MFMA path; raises if unsupported.
      - `"scalar"`: force the slow correctness kernel.
      - `"auto"`: prefer tiled when supported, else scalar.

    ``alibi_slopes`` is an optional `[num_query_heads]` f32 tensor; when
    supplied, the kernel applies the ALiBi linear bias on each row.
    ``qq_bias`` is an optional 2D f32 query-to-query bias; ``qq_bias_stride_0``
    is its first-axis stride (in elements). Both follow AITER's Triton
    semantics exactly and require the corresponding ``problem.use_alibi`` /
    ``problem.use_qq_bias`` flags to be set.

    ``stream`` is the HIP stream handle (an `int`) to launch on. Pass
    ``torch.cuda.current_stream().cuda_stream`` to make the launches
    visible to ``torch.cuda.graph`` capture; this is how the parity
    harness amortises the segment + reduce launch overhead in the 3D
    path under a hipgraph.
    """
    bt_stride = (
        int(block_table.stride(0))
        if hasattr(block_table, "stride")
        else int(block_table.shape[1])
    )

    # Auto path selection. Historically we *always* preferred 3D when
    # supported because split-KV produces a huge grid that beats Triton
    # 2-6× on decode-shape workloads (small total Q, few sequences).
    # That assumption breaks badly on production traces:
    #
    #   - chunked-prefill (large total Q across many seqs) -- 2D already
    #     saturates the device, so the extra split-KV segments add launch
    #     overhead with no parallelism gain. Triton picks 2D and we used
    #     to pick 3D, losing 30-150×.
    #   - sliding-window with long context -- 2D's per-iter window mask
    #     is much cheaper than 3D's per-segment scan over the full kv.
    #   - short context (k <= 512) -- the split-KV reduce kernel's launch
    #     overhead dominates the few real KV iterations.
    #
    # ``problem.select_path()`` (see UnifiedAttentionProblem.select_path)
    # wraps Triton's ``use_2d_kernel`` selector, which decides exactly
    # the above three cases. Honour it in auto mode unless the user
    # explicitly forces ``backend == "3d"``. This matches Triton's
    # production-tested selector on every shape and recovers the trace
    # buckets (where our old auto picked 3D for chunked prefill at 30-
    # 150× the right path's cost) while keeping the decode-shape 3D
    # wins on the parity harness (those shapes still satisfy the "3D
    # is fine" branch of ``use_2d_kernel``).
    prefer_2d = backend == "auto" and problem.select_path() == "2d"
    if backend == "3d" or (backend == "auto" and not prefer_2d):
        ok_3d, reason_3d = supports_native_unified_attention_3d_tiled(problem)
        if ok_3d:
            return _run_3d_tiled(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                block_table=block_table,
                softcap=softcap,
                sinks=sinks,
                bt_stride=bt_stride,
                warmup=warmup,
                attempts=attempts,
                alibi_slopes=alibi_slopes,
                qq_bias=qq_bias,
                qq_bias_stride_0=qq_bias_stride_0,
                stream=int(stream),
                k_scale=k_scale,
                v_scale=v_scale,
            )
        if backend == "3d":
            raise NotImplementedError(reason_3d)

    if backend in ("tiled", "auto"):
        ok_t, reason_t = supports_native_unified_attention_tiled(problem)
        if ok_t:
            # Hot path: compute the cache key directly from the problem +
            # selectors (skip the 17-field dataclass build). Spec is only
            # built on cache miss inside _get_2d_launcher and for grid
            # math below.
            key = _tiled_cache_key(problem)
            launcher = _get_2d_launcher(problem, key)
            vals = _attn_values(
                problem=problem,
                q=q,
                k=k,
                v=v,
                out=out,
                cu_seqlens_q=cu_seqlens_q,
                seqused_k=seqused_k,
                softmax_scale=softmax_scale,
                block_table=block_table,
                softcap=softcap,
                sinks=sinks,
                bt_stride=bt_stride,
                include_bt_stride=True,
                alibi_slopes=alibi_slopes,
                qq_bias=qq_bias,
                qq_bias_stride_0=qq_bias_stride_0,
                include_qq_bias_stride=True,
                k_scale=k_scale,
                v_scale=v_scale,
                out_scale=out_scale,
            )
            # The dispatcher must compute the grid using the same BLOCK_Q
            # the kernel uses (which depends on `block_m`); a mismatch
            # would launch the wrong number of CTAs and the kernel's
            # q_block_local_idx -> qb_start_pos math would touch the
            # wrong query positions. Compute directly from the selectors
            # (avoiding the full spec construction on the hot path).
            num_warps = _select_2d_num_warps(problem)
            block_m_per_warp = _select_2d_block_m_per_warp(problem)
            block_m = num_warps * block_m_per_warp
            block_q = (
                block_m // problem.num_queries_per_kv
                if problem.num_queries_per_kv <= block_m
                else 1
            )
            total_num_q_blocks = problem.total_q // block_q + problem.num_seqs
            threads_per_block = 64 * num_warps
            return launcher(
                vals,
                config=LaunchConfig(
                    grid=(int(problem.num_kv_heads), int(total_num_q_blocks), 1),
                    block=(threads_per_block, 1, 1),
                    stream=int(stream),
                ),
            )
        if backend == "tiled":
            raise NotImplementedError(reason_t)

    # Scalar fallback. Uses the same KernelLauncher infrastructure as
    # the tiled paths so module load + arg lifetime + stream resolution
    # are handled by-construction.
    ok, reason = supports_native_unified_attention(problem)
    if not ok:
        raise NotImplementedError(reason)
    key = _cache_key(problem)
    launcher = _get_scalar_launcher(problem, key)
    vals = _attn_values(
        problem=problem,
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        softmax_scale=softmax_scale,
        block_table=block_table,
        softcap=softcap,
        sinks=sinks,
        bt_stride=bt_stride,
        include_bt_stride=False,
    )
    return launcher(
        vals,
        config=LaunchConfig(
            grid=(
                int(problem.total_q),
                int(problem.num_query_heads),
                int(problem.head_size),
            ),
            block=(64, 1, 1),
            stream=int(stream),
        ),
    )


@dataclass(frozen=True)
class UnifiedAttention2DSpec:
    problem: UnifiedAttentionProblem
    name: str = "ck_dsl_unified_attention_2d_scalar"

    @property
    def dtype_ir(self) -> Type:
        if self.problem.dtype == "fp16":
            return F16
        if self.problem.dtype == "bf16":
            return BF16
        raise ValueError(
            f"unsupported dtype for scalar 2D kernel: {self.problem.dtype}"
        )

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"q{p.total_q}",
            f"h{p.num_query_heads}",
            f"kv{p.num_kv_heads}",
            f"d{p.head_size}",
            f"b{p.block_size}",
            p.dtype,
            flags={
                "sink": p.use_sinks,
                "sw": p.sliding_window > 0,
                "softcap": p.softcap > 0,
            },
        )


def build_unified_attention_2d(spec: UnifiedAttention2DSpec) -> KernelDef:
    """Build a scalar-correct 2D unified-attention kernel.

    One workgroup computes one output element `(query_token, query_head, dim)`.
    This is deliberately a correctness kernel: it implements the full paged
    online-softmax semantics for fp16/bf16 without relying on Triton. The
    optimized MFMA/tiled kernel will replace this body once parity is locked.
    """
    p = spec.problem
    if p.dtype not in ("fp16", "bf16"):
        raise ValueError("scalar 2D kernel currently supports fp16/bf16")
    dtype = spec.dtype_ir
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64

    output = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    query = b.param(
        "query_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sinks = b.param("sink_ptr", PtrType(dtype, "global"), readonly=True, align=16)
    block_tables = b.param(
        "block_tables_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    _alibi = b.param("alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4)
    _qq_bias = b.param("qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale = b.param("scale", F32)
    _k_scale = b.param("k_scale", F32)
    _v_scale = b.param("v_scale", F32)
    _out_scale = b.param("out_scale", F32)
    softcap = b.param("softcap", F32)
    num_seqs = b.param("num_seqs", I32)

    q_tok = b.block_id_x()
    q_head = b.block_id_y()
    dim = b.block_id_z()
    tid = b.thread_id_x()
    active = b.cmp_eq(tid, b.const_i32(0))

    # Find seq_idx by scanning cu_q: largest i such that cu_q[i] <= q_tok.
    seq_init = b.const_i32(0)
    scan = b.scf_for_iter(
        b.const_i32(0), num_seqs, b.const_i32(1), [("seq_idx", seq_init)], iv_name="si"
    )
    with scan as (si, (seq_idx,)):
        start_i = b.global_load_i32(cu_q, si)
        le = b.cmp_le(start_i, q_tok)
        next_seq = b.select(le, si, seq_idx)
        b.scf_yield(next_seq)
    seq_idx = scan.results[0]

    cu_start = b.global_load_i32(cu_q, seq_idx)
    cu_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    q_len = b.sub(cu_stop, cu_start)
    query_pos = b.sub(q_tok, cu_start)
    kv_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(kv_len, q_len)
    kv_head = b.div(q_head, b.const_i32(p.num_queries_per_kv))

    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    one_f = b.const_f32(1.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)

    if p.use_sinks:
        sink_h = b.global_load(sinks, q_head, dtype, align=2)
        init_m = b.fmul(b.cast_to_f32(sink_h), rcp_ln2)
        init_l = one_f
    else:
        init_m = neg_inf
        init_l = one_f
    init_acc = zero_f

    # Coordinate transforms over the kernel's tensors. Q/output are a
    # naive ``(query_token, query_head, dim)`` layout; the paged KV
    # cache uses ``PagedKvDescriptor`` (in element units, not bytes).
    q_desc = TensorDescriptor.naive(
        "Q",
        lengths=[p.max_seqlen_q + 1, p.num_query_heads, p.head_size],
        coord_names=("token", "head", "dim"),
    )
    kv_desc_elem = PagedKvDescriptor(
        block_size=p.block_size,
        stride_0=p.block_size * p.num_kv_heads * p.head_size,
        stride_1=p.num_kv_heads * p.head_size,
        stride_2=p.head_size,
        stride_3=1,
    )

    loop = b.scf_for_iter(
        b.const_i32(0),
        kv_len,
        b.const_i32(1),
        [("m", init_m), ("l", init_l), ("acc", init_acc)],
        iv_name="kpos",
    )
    with loop as (kpos, (m_val, l_val, acc_val)):
        block_idx = b.div(kpos, b.const_i32(p.block_size))
        token_in_block = b.mod(kpos, b.const_i32(p.block_size))
        physical = b.global_load_i32(
            block_tables,
            b.add(
                b.mul(
                    seq_idx,
                    b.const_i32((p.max_seqlen_k + p.block_size - 1) // p.block_size),
                ),
                block_idx,
            ),
        )

        score = zero_f
        for d in b.unroll(p.head_size):
            d_v = b.const_i32(d)
            q_off, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=d_v)
            k_off = kv_desc_elem.offset(
                b,
                physical_block=physical,
                token_in_block=token_in_block,
                kv_head=kv_head,
                dim=d_v,
            )
            qv = b.cast_to_f32(b.global_load(query, q_off, dtype, align=2))
            kv = b.cast_to_f32(b.global_load(key, k_off, dtype, align=2))
            score = b.fadd(score, b.fmul(qv, kv))

        score = b.fmul(b.fmul(score, scale), rcp_ln2)
        if p.softcap > 0:
            score = b.fmul(apply_softcap_runtime(b, score, softcap), rcp_ln2)

        causal_ok = b.cmp_le(kpos, b.add(context_len, query_pos))
        if p.sliding_window > 0:
            dist = b.sub(b.add(context_len, query_pos), kpos)
            sw_ok = b.cmp_lt(dist, b.const_i32(p.sliding_window))
            causal_ok = b.land(causal_ok, sw_ok)
        score = b.select(causal_ok, score, neg_inf)
        new_m_raw = b.fmax(m_val, score)
        # If both running max and current score are -inf, the row is fully
        # masked; force m to 0 so the resulting alpha/prob are 0 instead of NaN
        # (matches Triton's `m_j = tl.where(m_j > -inf, m_j, 0.0)`).
        is_finite = b.fcmp("ogt", new_m_raw, neg_inf)
        new_m = b.select(is_finite, new_m_raw, zero_f)
        alpha = b.exp2(b.fsub(m_val, new_m))
        prob = b.exp2(b.fsub(score, new_m))
        new_l = b.fadd(b.fmul(l_val, alpha), prob)
        v_off = kv_desc_elem.offset(
            b,
            physical_block=physical,
            token_in_block=token_in_block,
            kv_head=kv_head,
            dim=dim,
        )
        vv = b.cast_to_f32(b.global_load(value, v_off, dtype, align=2))
        new_acc = b.fadd(b.fmul(acc_val, alpha), b.fmul(prob, vv))
        b.scf_yield(new_m, new_l, new_acc)

    out_val = b.fmul(loop.results[2], b.rcp(loop.results[1]))
    out_cast = b.cast_f32_to(out_val, dtype)
    out_off, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=dim)
    valid = b.land(active, b.cmp_lt(dim, b.const_i32(p.head_size)))
    with b.scf_if(valid):
        b.global_store(output, out_off, out_cast, align=2)
    return b.kernel


def apply_softcap_runtime(b: IRBuilder, score_log2: Value, softcap: Value) -> Value:
    """Triton-equivalent softcap on a log2-domain score.

    Computes `softcap * tanh(score_natural / softcap)` using only `exp2` so the
    same primitives that drive the online softmax also handle softcap (matches
    AITER's `apply_softcap`). Given `score_log2 = score_natural * log2(e)`,

        Sdiv = score_log2 / softcap
        p1   = exp2(Sdiv)      = e^(score_natural / softcap)
        p2   = exp2(-Sdiv)     = e^(-score_natural / softcap)
        return softcap * (p1 - p2) / (p1 + p2)

    Returned value is in *natural* domain; the caller is responsible for
    multiplying by `RCP_LN2` to bring it back to log2 for the next `exp2`.
    """
    sdiv = b.fdiv(score_log2, softcap)
    p1 = b.exp2(sdiv)
    p2 = b.exp2(b.fneg(sdiv))
    diff = b.fsub(p1, p2)
    summ = b.fadd(p1, p2)
    return b.fmul(softcap, b.fmul(diff, b.rcp(summ)))


@dataclass(frozen=True)
class UnifiedAttention3DSpec(UnifiedAttention2DSpec):
    name: str = "ck_dsl_unified_attention_3d_scalar"
    num_segments: int = 8

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"q{p.total_q}",
            f"h{p.num_query_heads}",
            f"kv{p.num_kv_heads}",
            f"d{p.head_size}",
            f"b{p.block_size}",
            f"seg{self.num_segments}",
            p.dtype,
        )


def build_unified_attention_3d(spec: UnifiedAttention3DSpec) -> KernelDef:
    """Build scalar-correct split-3D segment attention kernel."""
    p = spec.problem
    dtype = spec.dtype_ir
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64

    segm_output = b.param(
        "segm_output_ptr",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    segm_max = b.param(
        "segm_max_ptr", PtrType(F32, "global"), noalias=True, writeonly=True, align=16
    )
    segm_expsum = b.param(
        "segm_expsum_ptr",
        PtrType(F32, "global"),
        noalias=True,
        writeonly=True,
        align=16,
    )
    query = b.param(
        "query_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr", PtrType(dtype, "global"), noalias=True, readonly=True, align=16
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    _sinks = b.param("sink_ptr", PtrType(dtype, "global"), readonly=True, align=16)
    block_tables = b.param(
        "block_tables_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    _alibi = b.param("alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4)
    _qq_bias = b.param("qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale = b.param("scale", F32)
    _k_scale = b.param("k_scale", F32)
    _v_scale = b.param("v_scale", F32)
    _softcap = b.param("softcap", F32)
    num_seqs = b.param("num_seqs", I32)

    q_tok = b.block_id_x()
    q_head = b.block_id_y()
    zd = b.block_id_z()
    segm_idx = b.div(zd, b.const_i32(p.head_size))
    dim = b.mod(zd, b.const_i32(p.head_size))
    tid = b.thread_id_x()
    active = b.cmp_eq(tid, b.const_i32(0))

    seq_idx = _emit_find_seq_idx_scan(b, cu_q, q_tok, num_seqs)
    cu_start = b.global_load_i32(cu_q, seq_idx)
    cu_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    q_len = b.sub(cu_stop, cu_start)
    query_pos = b.sub(q_tok, cu_start)
    kv_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(kv_len, q_len)
    kv_head = b.div(q_head, b.const_i32(p.num_queries_per_kv))
    tiles_per_segment = b.div(
        b.add(kv_len, b.const_i32(spec.num_segments * p.block_size - 1)),
        b.const_i32(spec.num_segments * p.block_size),
    )
    seg_start = b.mul(segm_idx, b.mul(tiles_per_segment, b.const_i32(p.block_size)))
    seg_stop_i = b.mul(
        b.add(segm_idx, b.const_i32(1)),
        b.mul(tiles_per_segment, b.const_i32(p.block_size)),
    )
    seg_stop_i = b.select(b.cmp_lt(seg_stop_i, kv_len), seg_stop_i, kv_len)

    neg_inf = b.const_f32(float("-inf"))
    zero_f = b.const_f32(0.0)
    rcp_ln2 = b.const_f32(1.4426950408889634)
    init_m = neg_inf
    init_l = zero_f
    init_acc = zero_f

    loop = b.scf_for_iter(
        seg_start,
        seg_stop_i,
        b.const_i32(1),
        [("m", init_m), ("l", init_l), ("acc", init_acc)],
        iv_name="kpos",
    )
    with loop as (kpos, (m_val, l_val, acc_val)):
        score = _emit_qk_score(
            b,
            p,
            dtype,
            query,
            key,
            block_tables,
            seq_idx,
            q_tok,
            q_head,
            kv_head,
            kpos,
            scale,
            rcp_ln2,
        )
        causal_ok = b.cmp_le(kpos, b.add(context_len, query_pos))
        score = b.select(causal_ok, score, neg_inf)
        new_m = b.fmax(m_val, score)
        alpha = b.exp2(b.fsub(m_val, new_m))
        prob = b.exp2(b.fsub(score, new_m))
        new_l = b.fadd(b.fmul(l_val, alpha), prob)
        vv = _emit_v_load(b, p, dtype, value, block_tables, seq_idx, kv_head, kpos, dim)
        new_acc = b.fadd(b.fmul(acc_val, alpha), b.fmul(prob, vv))
        b.scf_yield(new_m, new_l, new_acc)

    ml_desc, out_desc = _segm_descriptors(p, spec.num_segments)
    base, _ = out_desc.offset(b, token=q_tok, head=q_head, seg=segm_idx, dim=dim)
    with b.scf_if(active):
        b.global_store(segm_output, base, loop.results[2], align=4)
        is_dim0 = b.cmp_eq(dim, b.const_i32(0))
        with b.scf_if(is_dim0):
            segm_base, _ = ml_desc.offset(
                b,
                token=q_tok,
                head=q_head,
                seg=segm_idx,
            )
            b.global_store(segm_max, segm_base, loop.results[0], align=4)
            b.global_store(segm_expsum, segm_base, loop.results[1], align=4)
    return b.kernel


@dataclass(frozen=True)
class UnifiedAttentionReduceSpec:
    problem: UnifiedAttentionProblem
    num_segments: int
    name: str = "ck_dsl_unified_attention_reduce_scalar"

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.problem.dtype == "fp16" else BF16

    def kernel_name(self) -> str:
        from ..helpers.spec import kernel_name_join

        p = self.problem
        return kernel_name_join(
            self.name,
            f"q{p.total_q}",
            f"h{p.num_query_heads}",
            f"d{p.head_size}",
            f"seg{self.num_segments}",
            p.dtype,
        )


def build_unified_attention_reduce(spec: UnifiedAttentionReduceSpec) -> KernelDef:
    p = spec.problem
    dtype = spec.dtype_ir
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64
    out = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    segm_output = b.param(
        "segm_output_ptr", PtrType(F32, "global"), readonly=True, align=16
    )
    segm_max = b.param("segm_max_ptr", PtrType(F32, "global"), readonly=True, align=16)
    segm_expsum = b.param(
        "segm_expsum_ptr", PtrType(F32, "global"), readonly=True, align=16
    )
    _seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    _cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    q_tok = b.block_id_x()
    q_head = b.block_id_y()
    dim = b.block_id_z()
    tid = b.thread_id_x()
    active = b.cmp_eq(tid, b.const_i32(0))
    neg_inf = b.const_f32(float("-inf"))
    zero = b.const_f32(0.0)
    ml_desc, seg_out_desc = _segm_descriptors(p, spec.num_segments)
    q_desc = _q_descriptor(p)
    max_loop = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.num_segments),
        b.const_i32(1),
        [("mx", neg_inf)],
        iv_name="seg",
    )
    with max_loop as (seg, (mx,)):
        idx, _ = ml_desc.offset(b, token=q_tok, head=q_head, seg=seg)
        mv = b.global_load_f32(segm_max, idx)
        b.scf_yield(b.fmax(mx, mv))
    overall = max_loop.results[0]
    red = b.scf_for_iter(
        b.const_i32(0),
        b.const_i32(spec.num_segments),
        b.const_i32(1),
        [("den", zero), ("acc", zero)],
        iv_name="seg2",
    )
    with red as (seg, (den, acc)):
        idx, _ = ml_desc.offset(b, token=q_tok, head=q_head, seg=seg)
        mv = b.global_load_f32(segm_max, idx)
        lv = b.global_load_f32(segm_expsum, idx)
        factor = b.exp2(b.fsub(mv, overall))
        den2 = b.fadd(den, b.fmul(lv, factor))
        out_idx, _ = seg_out_desc.offset(
            b,
            token=q_tok,
            head=q_head,
            seg=seg,
            dim=dim,
        )
        ov = b.global_load_f32(segm_output, out_idx)
        acc2 = b.fadd(acc, b.fmul(ov, factor))
        b.scf_yield(den2, acc2)
    result = b.fmul(red.results[1], b.rcp(red.results[0]))
    cast = b.cast_f32_to(result, dtype)
    out_idx, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=dim)
    with b.scf_if(active):
        b.global_store(out, out_idx, cast, align=2)
    return b.kernel


def _emit_find_seq_idx_scan(
    b: IRBuilder, cu_q: Value, q_tok: Value, num_seqs: Value
) -> Value:
    scan = b.scf_for_iter(
        b.const_i32(0),
        num_seqs,
        b.const_i32(1),
        [("seq_idx", b.const_i32(0))],
        iv_name="si",
    )
    with scan as (si, (seq_idx,)):
        start_i = b.global_load_i32(cu_q, si)
        le = b.cmp_le(start_i, q_tok)
        b.scf_yield(b.select(le, si, seq_idx))
    return scan.results[0]


def _q_descriptor(p: UnifiedAttentionProblem) -> TensorDescriptor:
    """Element-unit Q/output descriptor: ``(token, head, dim)``."""
    return TensorDescriptor.naive(
        "Q",
        lengths=[p.max_seqlen_q + 1, p.num_query_heads, p.head_size],
        coord_names=("token", "head", "dim"),
    )


def _paged_kv_descriptor(p: UnifiedAttentionProblem) -> PagedKvDescriptor:
    """Element-unit paged-KV descriptor for the scalar kernels."""
    return PagedKvDescriptor(
        block_size=p.block_size,
        stride_0=p.block_size * p.num_kv_heads * p.head_size,
        stride_1=p.num_kv_heads * p.head_size,
        stride_2=p.head_size,
        stride_3=1,
    )


def _segm_descriptors(
    p: UnifiedAttentionProblem,
    num_segments: int,
) -> Tuple[TensorDescriptor, TensorDescriptor]:
    """``(segm_ml, segm_output)`` descriptors used by 3D + reduce kernels.

    Layouts:

      ``segm_ml``      : ``[total_q, num_query_heads, num_segments]``
      ``segm_output``  : ``[total_q, num_query_heads, num_segments, head_size]``

    Both are produced by ``build_unified_attention_3d`` and consumed by
    ``build_unified_attention_reduce``. Encoding them as descriptors
    means every offset becomes ``desc.offset(token=..., head=..., seg=...,
    dim=...)`` instead of the original ``add(mul, mul)`` ladder.
    """
    ml_desc = TensorDescriptor.naive(
        "segm_ml",
        lengths=[p.max_seqlen_q + 1, p.num_query_heads, num_segments],
        coord_names=("token", "head", "seg"),
    )
    out_desc = TensorDescriptor.naive(
        "segm_output",
        lengths=[
            p.max_seqlen_q + 1,
            p.num_query_heads,
            num_segments,
            p.head_size,
        ],
        coord_names=("token", "head", "seg", "dim"),
    )
    return ml_desc, out_desc


def _emit_qk_score(
    b: IRBuilder,
    p: UnifiedAttentionProblem,
    dtype: Type,
    query: Value,
    key: Value,
    block_tables: Value,
    seq_idx: Value,
    q_tok: Value,
    q_head: Value,
    kv_head: Value,
    kpos: Value,
    scale: Value,
    rcp_ln2: Value,
) -> Value:
    score = b.const_f32(0.0)
    physical, token_in_block = _physical_block_and_token(
        b, p, block_tables, seq_idx, kpos
    )
    q_desc = _q_descriptor(p)
    kv_desc = _paged_kv_descriptor(p)
    for d in b.unroll(p.head_size):
        d_v = b.const_i32(d)
        q_off, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=d_v)
        k_off = kv_desc.offset(
            b,
            physical_block=physical,
            token_in_block=token_in_block,
            kv_head=kv_head,
            dim=d_v,
        )
        qv = b.cast_to_f32(b.global_load(query, q_off, dtype, align=2))
        kv = b.cast_to_f32(b.global_load(key, k_off, dtype, align=2))
        score = b.fadd(score, b.fmul(qv, kv))
    return b.fmul(b.fmul(score, scale), rcp_ln2)


def _emit_v_load(
    b: IRBuilder,
    p: UnifiedAttentionProblem,
    dtype: Type,
    value: Value,
    block_tables: Value,
    seq_idx: Value,
    kv_head: Value,
    kpos: Value,
    dim: Value,
) -> Value:
    physical, token_in_block = _physical_block_and_token(
        b, p, block_tables, seq_idx, kpos
    )
    v_off = _paged_kv_descriptor(p).offset(
        b,
        physical_block=physical,
        token_in_block=token_in_block,
        kv_head=kv_head,
        dim=dim,
    )
    return b.cast_to_f32(b.global_load(value, v_off, dtype, align=2))


def _physical_block_and_token(
    b: IRBuilder,
    p: UnifiedAttentionProblem,
    block_tables: Value,
    seq_idx: Value,
    kpos: Value,
) -> Tuple[Value, Value]:
    block_idx = b.div(kpos, b.const_i32(p.block_size))
    token_in_block = b.mod(kpos, b.const_i32(p.block_size))
    max_blocks = (p.max_seqlen_k + p.block_size - 1) // p.block_size
    physical = b.global_load_i32(
        block_tables, b.add(b.mul(seq_idx, b.const_i32(max_blocks)), block_idx)
    )
    return physical, token_in_block
