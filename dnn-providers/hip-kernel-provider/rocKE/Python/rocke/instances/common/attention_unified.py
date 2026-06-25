# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Native CK DSL entry points for AITER unified attention.

This module intentionally separates *feature selection* from *kernel emission*.
The selector mirrors AITER's Triton wrapper exactly; kernel emission is gated
until every required primitive and correctness/perf path is present.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, Dict, Optional, Tuple

from ...core.ir import (
    BF16,
    F16,
    F32,
    FP8E4M3,
    I32,
    IRBuilder,
    KernelDef,
    PtrType,
    Type,
    Value,
)
from ...helpers.compile import compile_kernel
from ...runtime.launcher import (
    KernelLauncher,
    LaunchConfig,
    LaunchSummary,
    PipelineLauncher,
    WorkspaceSpec,
    WorkspacePool,
    _resolved_fence,
    no_fence,
    wait_stream_and_release,
)

from ...helpers.attention import (
    Attention2DConfig,
    Attention3DConfig,
    PagedKvDescriptor,
    apply_softcap_log2,
    select_2d_config,
    select_3d_config,
    use_2d_kernel,
)

from ...helpers.transforms import (
    TensorDescriptor,
    calculate_magic_numbers,
    do_magic_division,
)

# Alias the promoted log2-domain softcap helper under the local name used by
# the scalar reference kernels (matches the gfx950 tiled modules' convention).
_apply_softcap = apply_softcap_log2

# NOTE: this dispatcher lives in ``instances/common/`` (arch-neutral) but the
# optimized tiled kernels it dispatches to are arch-specific. To keep
# ``common/`` from importing an arch package at module top -- so that
# ``import rocke.instances`` does not pull in the gfx950 (MFMA) implementation
# regardless of the running device, and so the future gfx1151 (WMMA) tiled impl
# can be added the same way -- the gfx950 tiled specs/builders/gates are
# imported *lazily* inside the dispatch functions that actually use them
# (``supports_native_unified_attention_tiled``,
# ``supports_native_unified_attention_3d_tiled``, ``_tiled_spec_from_problem``,
# ``_tiled_3d_spec_from_problem``, ``_get_2d_launcher``, ``_get_3d_pipeline``).
# Return-type annotations that name the tiled spec classes are strings at
# runtime (``from __future__ import annotations`` above), so they need no
# import. The arch is resolved via ``_resolve_attention_arch`` at dispatch
# time, which is the natural seam for routing to a gfx1151 WMMA tiled impl once
# the unify phase lands it.


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
    # Number of physical blocks in the paged KV cache (``k.shape[0]``). Used
    # only to decide whether the i32 buffer voffset can address the whole
    # cache; 0 means "unknown" (assume small / fast i32 path). The
    # dispatcher fills this from the K tensor when available.
    num_kv_blocks: int = 0

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


def _magic_div(b: IRBuilder, dividend: Value, divisor: int) -> Value:
    """``dividend // divisor`` via CK Tile's magic mul-hi division.

    ``divisor`` is a loop-invariant compile-time constant so
    ``(multiplier, shift)`` fold to immediates and the runtime cost is one
    ``v_mul_hi_u32`` + add + shift instead of the AMDGPU integer divider's
    ~20-cycle Newton-Raphson sequence. ``dividend`` is a non-negative i32
    (a grid axis or a KV position), well inside the documented 31-bit
    unsigned range of the magic sequence -- so the unsigned magic quotient
    equals the signed ``b.div`` it replaces. This is the device lowering of
    ``merge_v2_magic_division`` (``transforms.do_magic_division``).
    """
    mult, shift = calculate_magic_numbers(divisor)
    return do_magic_division(b, dividend, mult, shift)


def _magic_div_mod(b: IRBuilder, dividend: Value, divisor: int) -> Tuple[Value, Value]:
    """Split ``dividend`` into ``(dividend // divisor, dividend % divisor)``.

    The quotient is the magic-division mul-hi sequence (:func:`_magic_div`);
    the remainder is reconstructed as ``dividend - quotient * divisor`` --
    exactly how CK Tile's ``merge_v2_magic_division`` recovers the low
    coordinate, dropping the hardware ``mod`` (a second integer divide).
    """
    quotient = _magic_div(b, dividend, divisor)
    remainder = b.sub(dividend, b.mul(quotient, b.const_i32(divisor)))
    return quotient, remainder


_RESOLVED_ATTENTION_ARCH: Optional[str] = None


def _resolve_attention_arch() -> str:
    """Pick the build/launch target for the tiled attention kernels.

    These launch paths compile-then-run on the local device, so the target
    must match the running GPU. Query the runtime; fall back to gfx950 (the
    only arch the tiled MFMA path supports today) when the device arch is
    unavailable (e.g. CPU-only static tests / cross-compile harnesses).

    **Memoized process-wide.** The device arch never changes within a process,
    yet the launch hot path resolves it ~20x per call (every selector that
    feeds ``_tiled_cache_key`` consults it). Re-importing + re-calling
    ``get_device_arch`` that many times is pure host overhead that dominates
    tiny-shape latency, so the result is cached after the first resolution.
    Tests that monkeypatch this function replace it wholesale, so the cache
    does not interfere with them.
    """
    global _RESOLVED_ATTENTION_ARCH
    if _RESOLVED_ATTENTION_ARCH is not None:
        return _RESOLVED_ATTENTION_ARCH
    try:
        from ...runtime.hip_module import get_device_arch

        arch = get_device_arch()
    except Exception:
        arch = None
    _RESOLVED_ATTENTION_ARCH = arch or "gfx950"
    return _RESOLVED_ATTENTION_ARCH


def _tiled_2d_impl(arch: str):
    """Lazily import the arch-specific tiled-2D spec / builder / gate.

    Returns ``(UnifiedAttention2DTiledSpec, build_unified_attention_2d_tiled,
    supports_tiled_2d)``. The import is performed here, inside the dispatch
    seam, so the arch-neutral ``instances/common`` package never imports an
    arch implementation at module top. This is the single place that routes the
    tiled-2D backend on ``arch``:

    * ``gfx1250`` (gfx1250) -> the wave32 WMMA ``16x16x32`` variant
      (``instances/gfx1250``).
    * ``gfx942`` (CDNA3) -> the narrow ``16x16x16`` strided-V variant
      (``instances/gfx942``).
    * everything else (default ``gfx950`` / CDNA4) -> the wide-K transpose-read
      variant (``instances/gfx950``).

    Routing gfx942 to its own variant is what keeps a gfx942 request off the
    gfx950 builder after ``attention_arch.validate_tiled_attention_arch`` was
    relaxed to admit gfx942 -- otherwise the gfx950 builder would emit
    gfx950-only ISA on gfx942 and crash comgr.
    """
    if arch == "gfx1250":
        from ..gfx1250.attention_tiled_2d import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
            supports_tiled_2d,
        )

        return (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
            supports_tiled_2d,
        )

    if arch == "gfx942":
        from ..gfx942.attention_tiled_2d import (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
            supports_tiled_2d,
        )

        return (
            UnifiedAttention2DTiledSpec,
            build_unified_attention_2d_tiled,
            supports_tiled_2d,
        )

    from ..gfx950.attention_tiled_2d import (
        UnifiedAttention2DTiledSpec,
        build_unified_attention_2d_tiled,
        supports_tiled_2d,
    )

    return (
        UnifiedAttention2DTiledSpec,
        build_unified_attention_2d_tiled,
        supports_tiled_2d,
    )


def _tiled_3d_impl(arch: str):
    """Lazily import the arch-specific tiled-3D split-KV spec / builders / gate.

    Returns ``(UnifiedAttention3DTiledSpec, UnifiedAttentionReduceTiledSpec,
    build_unified_attention_3d_tiled, build_unified_attention_reduce_tiled,
    supports_tiled_3d)``. See :func:`_tiled_2d_impl` for why this import is
    lazy and arch-keyed.
    """
    if arch == "gfx1250":
        from ..gfx1250.attention_tiled_3d import (
            UnifiedAttention3DTiledSpec,
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_3d_tiled,
            build_unified_attention_reduce_tiled,
            supports_tiled_3d,
        )

        return (
            UnifiedAttention3DTiledSpec,
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_3d_tiled,
            build_unified_attention_reduce_tiled,
            supports_tiled_3d,
        )

    if arch == "gfx942":
        from ..gfx942.attention_tiled_3d import (
            UnifiedAttention3DTiledSpec,
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_3d_tiled,
            build_unified_attention_reduce_tiled,
            supports_tiled_3d,
        )

        return (
            UnifiedAttention3DTiledSpec,
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_3d_tiled,
            build_unified_attention_reduce_tiled,
            supports_tiled_3d,
        )

    from ..gfx950.attention_tiled_3d import (
        UnifiedAttention3DTiledSpec,
        UnifiedAttentionReduceTiledSpec,
        build_unified_attention_3d_tiled,
        build_unified_attention_reduce_tiled,
        supports_tiled_3d,
    )

    return (
        UnifiedAttention3DTiledSpec,
        UnifiedAttentionReduceTiledSpec,
        build_unified_attention_3d_tiled,
        build_unified_attention_reduce_tiled,
        supports_tiled_3d,
    )


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
    if problem.use_fp8:
        if problem.q_dtype is not None and problem.q_dtype not in ("fp16", "bf16"):
            return False, f"scalar 2D kernel: unsupported q_dtype {problem.q_dtype!r}"
    elif problem.q_dtype is not None:
        return False, "scalar 2D q_dtype override requires use_fp8=True"
    if problem.use_alibi:
        return False, "ALiBi slopes are not enabled in CK DSL attention yet"
    if problem.use_qq_bias:
        return False, "QQ bias is not enabled in CK DSL attention yet"
    return True, "supported by scalar CK DSL 2D attention backend"


def supports_native_unified_attention_tiled(
    problem: UnifiedAttentionProblem,
) -> Tuple[bool, str]:
    """Return whether the optimized tiled MFMA path can run this problem."""
    arch = _resolve_attention_arch()
    if arch == "gfx1250" and problem.softcap > 0:
        return False, "gfx1250 tiled 2D does not support softcap yet"
    _, _, supports_tiled_2d = _tiled_2d_impl(arch)
    gfx942_bf16_wide = _enable_gfx942_bf16_flash(problem)
    if gfx942_bf16_wide:
        # bf16 wide-K (32x32x8) transposed flash path mirrors the build spec
        # in _tiled_spec_from_problem (T=64, mw=32, transposed-x8 bf16; D64 nw=4
        # double-K, D128 nw=2 single-K).
        nw, single_k = _gfx942_bf16_wide_geometry(problem)
        use_cfvst = _gfx942_bf16_wide_use_cfvst(problem)
        return supports_tiled_2d(
            head_size=problem.head_size,
            block_size=problem.block_size,
            dtype=problem.dtype,
            num_queries_per_kv=problem.num_queries_per_kv,
            use_alibi=problem.use_alibi,
            use_qq_bias=problem.use_qq_bias,
            use_fp8=problem.use_fp8,
            q_dtype=problem.q_dtype,
            num_warps=nw,
            block_m_per_warp=32,
            kv_storage_dtype=_kv_storage_dtype(problem),
            tile_size=_gfx942_bf16_wide_tile_size(problem),
            arch=arch,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_k_single_buffer=single_k,
            use_conflict_free_v_store=use_cfvst,
        )
    gfx942_flash = _enable_gfx942_fp16_flash(problem)
    num_warps = (
        _select_gfx942_flash_num_warps(problem)
        if gfx942_flash
        else _select_2d_num_warps(problem)
    )
    return supports_tiled_2d(
        head_size=problem.head_size,
        block_size=problem.block_size,
        dtype=problem.dtype,
        num_queries_per_kv=problem.num_queries_per_kv,
        use_alibi=problem.use_alibi,
        use_qq_bias=problem.use_qq_bias,
        use_fp8=problem.use_fp8,
        q_dtype=problem.q_dtype,
        num_warps=num_warps,
        block_m_per_warp=_select_2d_block_m_per_warp(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size=_select_2d_tile_size(problem),
        arch=arch,
        use_mfma_32x32x8=gfx942_flash,
        use_transposed_qk_32x32=gfx942_flash,
        use_k_single_buffer=gfx942_flash and _gfx942_flash_use_single_buffer(problem),
        use_conflict_free_v_store=gfx942_flash and _gfx942_flash_use_cfvst(problem),
        use_k_sliced_ring=_enable_gfx942_flash_k_sliced_ring(problem),
    )


def supports_native_unified_attention_3d_tiled(
    problem: UnifiedAttentionProblem,
) -> Tuple[bool, str]:
    """Return whether the optimized tiled MFMA 3D split-KV path can run this."""
    arch = _resolve_attention_arch()
    *_, supports_tiled_3d = _tiled_3d_impl(arch)
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
        arch=arch,
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
        problem.q_dtype,
        problem.sliding_window,
        bool(problem.use_sinks),
        bool(problem.softcap > 0),
        bool(problem.use_fp8),
    )


def _enable_d128_small_tile(problem: UnifiedAttentionProblem) -> bool:
    """d128 occupancy lever: select T = block_size (small tile) + nw=2 for
    the single-batch d128 combo so the kernel drops from 1 -> 2 WG/CU.

    Diagnosis (gfx950 MI355X, llvm22+comgr7.2, measured same-session; re-verified
    on the merged tree): the d128 combo at num_warps=2 is purely LDS-bound --
    K_lds[2,T,HD] + V_lds[1,T,HD] = 48 KB at T=2*block_size -> only 1 workgroup
    fits the 64 KB/CU budget; the register file already admits 2 waves/SIMD
    (V+A=229 <= 256). Halving the tile to T=block_size makes LDS=24 KB and
    VGPR=173 -> 2 WG/CU (measured via llvm-readelf on the produced HSACO:
    OFF VGPR=229/256, LDS=48 KB -> 1 WG/CU; ON VGPR=173, AGPR=0, LDS=24 KB
    -> 2 WG/CU, for S=1024/2048/4096 alike). Same-session vs torch SDPA FLASH
    (apples-to-apples, OFF vs ON on this exact tree):
      bf16: S1024 1.333x->1.437x, S2048 0.968x->1.059x (CROSS 1.0),
            S4096 0.916x->0.974x (DSL 0.789->0.741 ms, +6% faster kernel).
      fp16: S1024 1.312x->1.436x, S2048 0.973x->1.061x (CROSS 1.0),
            S4096 0.918x->0.975x. Correctness rel ~4.4e-3 bf16 / ~5.5e-4 fp16
            (IDENTICAL to baseline at every S).

    **RECONCILIATION + DEFAULT-ON.** The small-tile win REQUIRES nw=2 for
    ALL seqlens (the prior nw=4-at-S>=2048 rule + T=32 is occupancy-WORSE: 56 KB
    LDS -> back to 1 WG/CU), so when this gate fires the num_warps selector
    forces nw=2. Verified C-twin selector output == Python for the whole cohort
    and that NO non-cohort shape changes (3826 trace records across the three
    canonical aiter_ua* sets: 0 selector changes). It is therefore enabled by
    DEFAULT for the gfx950 single-batch d128 no-FP8 combo. This is a production
    ROUTING change (the dispatched d128 prefill kernel changes: T 64->32,
    nw S>=2048 4->2); per-spec golden EMIT is unchanged (T=block_size is an
    existing tile_size value, byte-identical to the old code path).

    ESCAPE HATCH: ``HIPDNN_GFX950_D128_SMALL_TILE=0`` (or off/no/false) force-
    DISABLES the lever, restoring the prior routing (T=64, nw=2/4). Any other
    value (or unset) -> default-ON for the cohort.
    """
    env = (
        __import__("os")
        .environ.get("HIPDNN_GFX950_D128_SMALL_TILE", "")
        .strip()
        .lower()
    )
    if env in ("0", "false", "no", "off"):
        return False
    return (
        problem.head_size == 128
        and not problem.use_fp8
        and _enable_single_batch_combo(problem)
    )


def _enable_k_single_buffer(problem: UnifiedAttentionProblem) -> bool:
    """d128 long-context lever: K single-buffer at T=64 (== 2*block_size).

    For the same single-batch d128 cohort that ``_enable_d128_small_tile`` gates,
    keep the LARGER T=64 tile (better long-context per-iter amortisation) but
    halve K_lds via K single-buffer so LDS stays at 32 KB -> 2 WG/CU. The
    next-K prefetch is re-issued after the PV-wait barrier (no WAR race). This
    crosses flash at the S4096 holdout (0.976x -> 1.020x bf16+fp16) and beats
    the small-tile T=32 pick at every measured shape (gfx950 MI355X, same-session
    vs torch SDPA FLASH, GPU-idle; see the long-context analysis). Occupancy proof
    (llvm-readelf):
    T=64 + K-single -> VGPR=215 AGPR=0 LDS=32 KB -> 2 WG/CU.

    Scoped to the EXACT V-single-buffer combo path the gfx950 emitter wires K
    single-buffer on (no ring, no grouped_kv2, no FP8, V single-buffer). The
    cohort already satisfies these; the spec __post_init__ re-validates loudly.

    GEOMETRY GUARD: K-single needs Q to fit the lone K slot, i.e.
    ``block_m <= tile_size``. The cohort runs num_warps=2 / block_m_per_warp=32
    -> block_m=64, with tile_size = 2*block_size, so this only holds for
    ``block_size >= 32`` (T>=64). At block_size=16 (T=32) Q (64 rows) cannot fit
    the single 32-token K slot and the __post_init__ validator rejects it
    (use_q_direct_reg would lift this but is not wired on this path). Fall back
    to the no-K-single small-tile config there.
    """
    return _enable_d128_small_tile(problem) and problem.block_size >= 32


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
    ~24% (measured by an out-of-tree ``probe_prefill_sweep.py``: decode_b1
    drops 34 µs → 26 µs).

    The kernel's own gate (``supports_tiled_2d``) re-validates the
    choice against the per-wave-tokens / LDS-budget constraints.

    **FP8 sliding-window long-prefill exception** (round-2 cluster-B
    sweep, ``prod_nw_sweep.log``): when the
    sliding window prunes the kv-loop to a handful of iters per CTA,
    bigger T over-allocates LDS without amortising any per-iter cost.
    For ``use_fp8 + sliding_window > 0 + max_seqlen_q > 256``, drop to
    ``T = block_size`` (single block per iter, 32 tokens) — measured
    1.15-1.30× win on every FP8 SW long-prefill shape.
    """
    if _resolve_attention_arch() == "gfx1250":
        # gfx1250 v1 consumes exactly one 32-token paged-KV block per WMMA
        # iteration; wider T needs separate multi-block block-table handling.
        return problem.block_size
    # Sliding-window long-prefill FP8 exception. The latest broad sweep
    # confirmed this should stay FP8-only: for bf16 SW long-prefill the
    # correctness-clean winner was T=64, not T=32
    # (``trace_bench/sweep_attention2d_configs.json``:
    # bf16_sw_n16_q1000_k1050 best T=64/mw16/hipcc at ~257us; T=32
    # variants were >=258us and often incorrect under hipcc).
    if problem.use_fp8 and problem.sliding_window > 0 and problem.max_seqlen_q > 256:
        return problem.block_size
    # gfx942 D64. The flash/L4 regime (use_mfma_32x32x8 + sliced-K ring) requires
    # T in {64,128} and a multiple of 32, so force T=64 there (mirrors the D128
    # flash rule below); otherwise paged block_size in {16,32} would yield T=16/32
    # and the spec validator would reject the build (reachable via the public
    # backend="auto" dispatcher for a head_size=64 fp16 long-prefill problem with
    # a 16/32-token paged KV cache). The narrow path keeps the single-KV-block
    # oracle (the generic ``2*block_size`` over-allocates LDS on MI300X with mw=32).
    if _resolve_attention_arch() == "gfx942" and problem.head_size == 64:
        if _enable_gfx942_fp16_flash(problem):
            return 64
        return problem.block_size
    # gfx942 D128 (ALL dtypes): T=64. fp16 flash/L4 wants it for the overlapped
    # sliced-K ring -- an exhaustive ~1200-kernel sweep (exhaustive_sweep.py)
    # showed T=64+ring+mask-limit is 21-23% faster than the T=128 ring and beats
    # Torch (S2048 221 vs 230us, S4096 680 vs 976us). And bf16 / fp16-with-extras
    # D128 *require* T=64: the generic default (T=2*BS=128) makes the
    # double-buffered K alone 64 KB, overflowing the LDS budget so the tiled-2D
    # gate rejects it and the dispatcher drops to the (1000x slower) scalar
    # kernel. T=64 fits the narrow 16x16x16 path (nw<=2 after the LDS step-down).
    if _resolve_attention_arch() == "gfx942" and problem.head_size == 128:
        return 64
    # bf16 transposed-combo sliding-window. Paired with the combo's
    # ``num_warps=2`` (BLOCK_M=64) prelude-light geometry, the smaller
    # ``T = block_size`` tile prunes the window finer (less compute on
    # partially-masked tiles) and dramatically cuts the per-CTA prelude:
    # measured SW geomean 0.67x -> ~1.04x vs Triton-2d (bit-exact). This
    # supersedes the older T=64 SW choice, which assumed num_warps=4 / no
    # combo and a much heavier prelude.
    if _enable_combo_2d(problem) and problem.sliding_window > 0:
        return problem.block_size
    # Single-batch (num_seqs == 1) d128/d64 prefill full-combo cohort.
    # Autotuner-proven KV tile per shape (gfx950, no-SW):
    #   * d128 -> T = 2 * block_size (32). The d128 winners all kept the
    #     default 2x tile; wider tiles over-allocated LDS without amortising.
    #   * d64  -> T = 128 (8 * block_size). Paired with num_warps=4 the wider
    #     tile feeds the d64 KV loop and was the winner for both MHA and
    #     GQA-8 d64 S2048 (1.19-2.68x over prod).
    if _enable_single_batch_combo(problem):
        if problem.head_size == 64:
            return 128
        # d128 occupancy lever (supersedes the small-tile pick for the
        # LONG-context holdout): at num_warps=2 the d128 combo is LDS-bound
        # (K_lds[2,T,HD]+V_lds[1,T,HD] = 48 KB at T=2*BS=64 -> 1 WG/CU). The
        # small-tile lever halved the TILE (T=block_size=32) to drop LDS
        # 48->24 KB -> 2 WG/CU, but at T=32 the long-context S4096 path runs
        # 128 outer iters whose per-iter overhead under-amortises -> only
        # 0.976x flash. The K-single-buffer lever instead
        # keeps the LARGER T=64 tile (64 iters at S4096) and halves K_lds via
        # ``use_k_single_buffer`` (K_lds[1,T,HD]=16 KB + V_lds[1,T,HD]=16 KB =
        # 32 KB -> 2 WG/CU, VGPR=215 AGPR=0). This crosses flash at S4096
        # (0.976x -> 1.020x bf16+fp16) AND beats the T=32 pick at every shape
        # (S1024 1.451->1.476x, S2048 1.065->1.099x; same correctness rel).
        # The single K slot re-issues the next-K prefetch AFTER the PV-wait
        # barrier (all QK reads drained) so it cannot WAR-race -- avoiding the
        # documented gfx942 naive-single-buffer hazard. Default-OFF behind the
        # same env opt-in (golden/production routing unchanged) until landed.
        if _enable_d128_small_tile(problem):
            return 2 * problem.block_size
        return 2 * problem.block_size
    # Qwen3-30B-A3B prefill specialization (bf16, hd64, BS=16, num_seqs=1,
    # num_queries_per_kv=8). At BS=16 the default ``T=2*BS=32`` gives the
    # async DMA loader only 32*64 = 2048 bytes per iter, which under-feeds
    # the long-prefill kv loop. Measured on MI355X with the bench at
    # ``/tmp/bench_prefill_sweep2.py``:
    #   - q=128:   T=64 best (29µs vs 37µs)
    #   - q=256:   T=64 best
    #   - q=512:   T=128 best (37µs vs 56µs)
    #   - q>=1024: T=64 with mfma_32x32+half_local_pv best (the
    #     ``_select_2d_block_m_per_warp`` gate picks mw=32 for this case)
    if (
        problem.head_size == 64
        and problem.block_size == 16
        and problem.num_seqs <= 1
        and not problem.use_fp8
        and problem.dtype == "bf16"
        and problem.num_queries_per_kv >= 4
    ):
        if problem.max_seqlen_q >= 512 and problem.max_seqlen_q <= 768:
            return 128
        if problem.max_seqlen_q > 64:
            return 64
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
    an out-of-tree ``trace_bench_report.md`` on MI355X / gfx950; the
    ``warps``-sweep harness is an out-of-tree ``probe_prefill_sweep.py``
    and the prefill-time harness is an out-of-tree
    ``probe_prefill_time.py``):

    **Post Q-in-registers + single-buffer-V refactor** (measured with
    ``probe_prefill_sweep.py`` on MI355X):

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
    if _resolve_attention_arch() == "gfx1250":
        # A gfx1250 workgroup is one wave32 in the v1 WMMA tiled path.
        return 1
    # The validated combo family (d64/b32/GQA-8 bf16, with sinks).
    #   * full attention: num_warps=4 (BLOCK_M=128) amortises the per-CTA
    #     prelude over the many KV tiles a long no-SW context produces.
    #   * sliding window: num_warps=2 (BLOCK_M=64). The window prunes the
    #     KV loop to a handful of tiles, so the shape is prelude-bound;
    #     halving BLOCK_M halves the per-CTA Q-load prelude and doubles the
    #     CTA count for better latency hiding. Paired with T=block_size
    #     (see _select_2d_tile_size) this takes SW from 0.67x to ~1.04x vs
    #     Triton-2d (bit-exact).
    # Small/medium gfx942 prefill light narrow path: nw=1 for MHA, nw=2 for GQA
    # (loser_sweep.py graph-timed winners). Must precede the D64/L4 rules below,
    # which target the heavy flash/ring geometry.
    if _enable_gfx942_small_q_narrow(problem):
        return 1 if problem.num_queries_per_kv == 1 else 2
    # gfx942 D128 fp16 flash/L4 (shipped): the flash kernel runs at
    # ``_select_gfx942_flash_num_warps`` (wide, default 4 -> BLOCK_M = 4 * 32),
    # not the legacy num_warps=1 geometry. Every production flash site reads the
    # flash selector directly, so this branch is not on the live grid path
    # today; defer to that selector so a future caller can't read a count that
    # disagrees with the built kernel (num_warps is not part of the JitCache
    # key, so a drift would silently launch the wrong CTA count, not rebuild).
    if _enable_gfx942_l4(problem):
        return _select_gfx942_flash_num_warps(problem)
    # gfx942 D64 oracle: paired with ``T=block_size`` and mw=32, four waves fit
    # in the MI300X 64 KB LDS budget and match the direct gfx942 harness.
    if _resolve_attention_arch() == "gfx942" and problem.head_size == 64:
        return 4
    if _enable_combo_2d(problem):
        # bf16 sliding-window is prelude-bound -> nw2 (lighter prelude). But
        # fp8 SW is dequant-bound, not prelude-bound, so it wants nw4 to
        # spread the fp8->bf16 dequant across more warps (nw2 concentrates
        # it and regresses). No-SW (compute-bound) is nw4 for both.
        target = 2 if (problem.sliding_window > 0 and not problem.use_fp8) else 4
        HD = problem.head_size
        BS = problem.block_size
        T = _select_2d_tile_size(problem)
        while target > 1:
            if (T * HD) < 64 * target * 8:
                target //= 2
                continue
            if (64 * 8) // HD > BS:
                target //= 2
                continue
            break
        return max(1, target)
    if _enable_single_batch_combo(problem):
        # Single-batch (num_seqs == 1) d128/d64 prefill full-combo cohort
        # (re-tuned by the joint num_warps x schedule sweep). Per shape
        # (gfx950, no-SW), same-session vs flash:
        #   * d128 (BLOCK_M = 32 * nw): S<=1024 -> nw=2 (nw=1 is occupancy-
        #     starved on the tiny single-seq grid; nw=2 with V-double-buffer
        #     OFF = 1.30x flash vs nw=1's 0.89x). S>=2048 -> nw=4 (more
        #     resident waves hide the long-KV-loop latency; nw=4 wpe=2 is +9%
        #     over nw=2 and lifts S=4096 from 0.81 to 0.885x).
        #   * d64: nw=4 paired with T=128 (the wider tile + 4 warps was the
        #     winner for both MHA and GQA-8 d64 S2048).
        #
        # RECONCILIATION: the d128 SMALL-TILE occupancy win (T=block_size
        # -> 2 WG/CU) REQUIRES num_warps=2 for ALL seqlens. The prior nw=4-at-
        # S>=2048 choice combined with T=block_size is occupancy-WORSE
        # (nw=4 + T=32 -> 56 KB LDS -> back to 1 WG/CU); the measured 2-WG/CU
        # crossing (S=2048 1.06x, S=4096 1.02x flash) only holds at nw=2 + T=32
        # (V+A=173, LDS=24 KB). So when the small-tile cohort is active, force
        # nw=2 here, overriding the S>=2048 -> nw=4 rule. When the small-tile
        # gate is OFF this branch is byte-for-byte the prior behavior.
        if problem.head_size == 64:
            target = 4
        elif _enable_d128_small_tile(problem):
            target = 2
        elif problem.max_seqlen_q <= 1024:
            target = 2
        else:
            target = 4
        HD = problem.head_size
        BS = problem.block_size
        T = _select_2d_tile_size(problem)
        while target > 1:
            if (T * HD) < 64 * target * 8:
                target //= 2
                continue
            if (64 * 8) // HD > BS:
                target //= 2
                continue
            break
        return max(1, target)
    # Qwen3-30B-A3B prefill specialization (bf16, hd64, BS=16, num_seqs=1,
    # num_queries_per_kv>=4). The default heuristic was tuned for HD=128 /
    # multi-batch prefill; for this hd64 single-seq path the per-CTA cost
    # is much higher than necessary because (a) q<=128 had NW=2 which
    # over-allocated LDS/VGPR for a tiny grid, and (b) q>256 num_seqs<=1
    # kept NW=2 which left perf on the table on long single-seq prefill.
    # Measured (``/tmp/bench_prefill_sweep2.py``, MI355X bf16 hd64):
    #   q=128:   NW=1 best  (28.7µs vs 37µs default)
    #   q=256:   NW=2 best  (32µs vs 43µs default)
    #   q>=512:  NW=2 (mw=16) or NW=4 (mw=32) best
    if (
        problem.head_size == 64
        and problem.block_size == 16
        and problem.num_seqs <= 1
        and not problem.use_fp8
        and problem.dtype == "bf16"
        and problem.num_queries_per_kv >= 4
    ):
        if problem.max_seqlen_q <= 128:
            target = 1
        elif problem.max_seqlen_q <= 768:
            target = 2
        else:
            # Long single-seq prefill: NW=4 wins paired with mw=32+mfma_32x32
            target = 4
    elif problem.max_seqlen_q <= 64:
        target = 1
    elif problem.max_seqlen_q <= 128:
        target = 2
    elif problem.max_seqlen_q <= 256:
        target = 4
    elif problem.num_seqs <= 1:
        # Long prefill with num_seqs <= 1: the production-shape sweep
        # (an out-of-tree ``prefill_n_sweep.log``) shows
        # nw=2 beats nw=4 by 3-5% at n=1 (fewer CTAs hurt under-saturated
        # GPU at single-seq). The crossover happens at n=2 where nw=4
        # takes over (4-16% faster than nw=2 from n=2 up to bench-cap n=16).
        # Single-seq prefill is the only production regime where nw=2 wins.
        target = 2
    elif (
        problem.num_queries_per_kv == 1
        and problem.head_size == 64
        and not _enable_combo_2d(problem)
    ):
        # MHA (num_queries_per_kv == 1) d64 prefill is a DIFFERENT family
        # than the GQA-8 cluster-A sweep that set the generic nw=4 rule
        # below (that sweep is BLOCK_Q=8 => 8 q-rows per kv, a GQA shape).
        # The perf_attn2d_sweep (B2 H8 d64 fp16, gfx950) measured the plain
        # ``fallback`` geometry at both num_warps and nw=2 (BLOCK_M=32) wins
        # at the short and long ends, nw=4 (BLOCK_M=64) only in the middle:
        #   s512 : nw2 72.1 vs nw4 67.7  -> nw2  (+6%, more CTAs fill the
        #          small grid)
        #   s1024: nw2 130  vs nw4 148   -> nw4  (mid grid already saturates;
        #          BLOCK_M=64 halves dispatch overhead)
        #   s2048: nw2 272  vs nw4 219   -> nw2  (+25%, the long KV loop wants
        #          the doubled CTA count for latency hiding)
        # Encode that U-shape: nw=2 at the short (<=512) and long (>=1536)
        # ends, nw=4 in the middle. Verified via direct-launch (no harness
        # overhead) on gfx950.
        target = (
            2 if (problem.max_seqlen_q <= 512 or problem.max_seqlen_q >= 1536) else 4
        )
    else:
        # Long prefill (q > 256) with num_seqs >= 2: nw=4 wins.
        # Round-1 cluster-A sweep
        # (an out-of-tree ``cluster_a_sweep.md``)
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
        _resolve_attention_arch(),
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
        (
            _select_gfx942_flash_num_warps(problem)
            if _enable_gfx942_fp16_flash(problem)
            else _select_2d_num_warps(problem)
        ),
        _kv_storage_dtype(problem),
        _select_2d_tile_size(problem),
        _select_2d_waves_per_eu(problem),
        _select_2d_block_m_per_warp(problem),
        _enable_mfma_32x32(problem),
        _enable_transposed_qk_32x32(problem),
        _enable_transposed_half_local_pv(problem),
        # Transposed-softmax VALU sub-flags + V-prefetch schedule. These
        # vary with max_seqlen_q within the single-batch cohort independently of
        # num_warps/wpe/tile, so they MUST be in the key to avoid a launcher
        # collision between two seqlens that share the same geometry but differ
        # on the V schedule (e.g. S=1500 vs S=2048).
        _enable_transposed_subflags(problem),
        _enable_v_double_buffer(problem),
        _enable_early_v_schedule(problem),
        _enable_register_pv(problem),
        _enable_i64_kv_addr(problem),
        _select_2d_compile_backend(problem),
        _enable_gfx942_fp16_flash(problem),
        _gfx942_flash_wide_setting() if _enable_gfx942_fp16_flash(problem) else None,
        (
            _gfx942_flash_kv_cache_policy(problem)
            if _enable_gfx942_fp16_flash(problem)
            else None
        ),
        _enable_gfx942_flash_q_direct(problem),
        _enable_gfx942_flash_mask_limit(problem),
        _enable_gfx942_flash_k_sliced_ring(problem),
        _enable_gfx942_flash_k_sliced_ldsseq(problem),
    )


def _select_2d_waves_per_eu(problem: UnifiedAttentionProblem) -> Optional[int]:
    """Choose ``waves_per_eu`` for the tiled 2D kernel.

    Triton's ``select_2d_config`` uses ``waves_per_eu=2`` for every config
    (verified against the aiter Triton reference
    ``aiter/ops/triton/attention/unified_attention.py``).
    We match that: it gives the LLVM backend more VGPR headroom per wave
    (less risk of spill to scratch / LDS) while still meeting the
    occupancy targets (the double-buffered K/V kernel runs at 2-3 WGs/CU
    on MI355X depending on shape; pushing for wpe=3 was a marginal
    +5% on isolated workloads but no consistent gain over the full
    workload spectrum once we A/B'd it against Triton's choice).

    If the problem itself pinned ``waves_per_eu`` (via the public
    ``UnifiedAttentionProblem.waves_per_eu`` field), respect that.
    """
    if _resolve_attention_arch() == "gfx1250":
        return problem.waves_per_eu
    if problem.waves_per_eu is not None:
        return problem.waves_per_eu
    # Single-batch (num_seqs == 1) d128/d64 prefill full-combo cohort
    # (re-tuned by the joint num_warps x schedule sweep): wpe=2 across all
    # seqlens. The prior wpe=3-at-S>=4096 rule REGRESSED the single-batch
    # cohort (d128 S4096: nw=4 wpe=2 = 0.881 vs nw=4 wpe=3 = 0.856). Checked
    # before the combo wpe=4 rule because the single-batch geometry is
    # smaller-grid and prefers fewer waves. (The small-tile lever uses nw=2 + T=32
    # which is 2 WG/CU at wpe=2; wpe=2 is correct for that cohort too.)
    if _enable_single_batch_combo(problem):
        return 2
    # Combo (incl. fp8 combo) wants wpe=4 -- handled below; check it before
    # the generic fp8 wpe=3 rule so fp8 combo prefill gets wpe=4, not 3.
    if _enable_combo_2d(problem):
        return 4
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
    # (The transposed-32x32 combo is handled at the top of this function:
    # wpe=4 reaches 4 WG/CU on both the nw4/T64 and nw2/T32 geometries and
    # is a consistent win over wpe=2/3 on the d64 prefill-2D cohort.)
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
      - long prefill (many KV iters * many MFMAs), no-SW *and* SW: loses
        ~10% (per-MFMA in-register dequant accumulates faster than the
        loader LDS writes we replaced). Measured on the AmirFix fp8 SW
        prefill cohort (q~8000, k~10000, SW=128): fp8qk=True 0.60x vs
        sync-dequant 0.67x -- so prefill stays on the sync path.
    Gate on (sliding-window OR ``max_seqlen_k <= 16 * T``) AND a small
    ``max_seqlen_q`` (decode / short prefill), plus the loader fitness
    check.
    """
    if not problem.use_fp8:
        return False
    # The 32x32 combo reads bf16 K from LDS, so it MUST use the sync-dequant
    # loader (bf16 K_lds), not this in-LDS-fp8 path. Never combine them.
    if _enable_combo_2d(problem):
        return False
    if not _fp8_qk_loader_fits(problem):
        return False
    # Large prefill (long per-CTA KV loop) prefers the sync-dequant path
    # regardless of sliding window; the K-in-LDS loader only wins when the
    # KV loop is short (decode / short prefill).
    if problem.max_seqlen_q > 256:
        return False
    T_eff = _select_2d_tile_size(problem)
    return problem.sliding_window > 0 or problem.max_seqlen_k <= 16 * T_eff


def _enable_single_batch_combo(problem: UnifiedAttentionProblem) -> bool:
    """Single-batch (num_seqs == 1) d128/d64 prefill -> full 32x32 combo.

    The combinatorial autotuner proved
    that for SINGLE-BATCH (num_seqs == 1) long prefill the production
    selector was routing d128 and d64 to the legacy 16x16x32 path, which is
    ~1.5-2.7x SLOWER than the full 32x32 transposed "combo" and the combo is
    correctness-equal (winner ``max_abs`` == flash's on every shape; bf16 +
    fp16; S in {1024,2048,4096}).

    The old ``_enable_transposed_qk_32x32`` / ``_select_2d_block_m_per_warp``
    gates only picked mw=32 for ``num_seqs >= 2`` (the docstrings justified
    that with a measurement of the PLAIN transposed path at 0.74-0.85x
    slower at num_seqs == 1). That measurement was of the bare transposed
    path -- the FULL combo (scalar-state + mask-once + mask-limit +
    skip-legacy-qreg + half-local-PV, plus a V prefetch schedule) was never
    built for num_seqs == 1, and it is 1.5-2.7x FASTER, not slower. So the
    num_seqs >= 2 restriction was stale for the full combo.

    Cohort (matches the autotuner's proven-winner shapes):
      * gfx950 only (the wide-K 32x32 MFMA + transpose reads are gfx950-only;
        gfx942 / RDNA stay on their narrow / flash paths).
      * num_seqs == 1 (single-batch). Multi-batch keeps its existing combo /
        transposed routing.
      * dtype in {bf16, fp16} (the transposed softmax keeps m/l/acc in f32;
        fp16 winners were as accurate as flash).
      * no FP8 K/V (the combo reads bf16 K from LDS; the fp8 cache path uses
        the sync-dequant loader and its own routing).
      * no ALiBi / QQ bias / softcap / sinks (not wired into the transposed
        softmax VALU opts; the spec validator rejects them).
      * no sliding window (the mask-once / mask-limit opts require no-SW).
      * head_size in {64, 128}.
      * max_seqlen_q > 256 (long prefill; decode-class shapes route to the 3D
        split-KV path via ``select_path``, and the autotuner's win starts at
        S=1024 -- short prefill stays on the legacy path where it is a tie).
    """
    if _resolve_attention_arch() != "gfx950":
        return False
    if problem.num_seqs != 1:
        return False
    if problem.dtype not in ("bf16", "fp16"):
        return False
    if problem.use_fp8:
        return False
    if problem.use_alibi or problem.use_qq_bias:
        return False
    if problem.softcap > 0 or problem.use_sinks:
        return False
    if problem.sliding_window > 0:
        return False
    if problem.head_size not in (64, 128):
        return False
    if problem.max_seqlen_q <= 256:
        return False
    return True


def _enable_v_double_buffer(problem: UnifiedAttentionProblem) -> bool:
    """Enable the V[i+1] double-buffer prefetch on the single-batch combo.

    SHORT single-batch combo prefill (max_seqlen_q <= 1024) on **d64** stacks
    ``use_v_double_buffer`` on top of the combo: the extra in-flight V prefetch
    hides the V async DMA latency the short KV loop cannot otherwise amortise.
    Bit-identical to the no-flag path (a pure schedule rewrite); the spec
    validator allows it on the bf16/fp16 no-FP8 combo.

    **d128: V double-buffer is a NET DRAG and is OFF.** The joint num_warps x
    schedule sweep found that on d128 removing the prefetch goes 0.89 -> 1.08x
    even at nw=1, and the actual win path is nw=2 with NO V prefetch = 1.30x
    flash (the earlier ~6-8% "win" measured the prefetch against an already
    occupancy-starved nw=1 baseline). Turning vdbuf off for d128 also
    auto-disables the lever-3 ``use_sched_barrier`` (it gates on this
    predicate), which was only patching the symptom of the unwanted prefetch.
    (The small-tile lever keeps d128 vdbuf OFF too: its 2-WG/CU LDS budget = K_lds[2]
    24 KB + V_lds[1] 8 KB; a V[i+1] prefetch would re-inflate LDS and drop back
    to 1 WG/CU.)

    LONG prefill (>=2048) gets NO V double-buffer regardless of head_size: the
    extra prefetch REGRESSES long prefill (the long KV loop already hides V
    latency). d64 long uses ``use_early_v_schedule`` instead (mutually
    exclusive).
    """
    if not _enable_single_batch_combo(problem):
        return False
    if _enable_early_v_schedule(problem):
        return False
    # d128: vdbuf is a net drag (see docstring) -- OFF; the nw=2 / no-prefetch
    # path is the winner. This also auto-disables sched_barrier.
    if problem.head_size == 128:
        return False
    return problem.max_seqlen_q <= 1024


def _enable_sched_barrier(problem: UnifiedAttentionProblem) -> bool:
    """Enable the lever-3 sched_barrier fence (CK-Tile-derived).

    A ``__builtin_amdgcn_sched_barrier`` is placed between the QK MFMA cluster
    and the post-QK async prefetch issue so the LLVM post-RA scheduler keeps the
    QK MFMAs packed instead of interleaving the next-tile ``buffer_load_lds``
    into the MFMA window. Bit-identical numerics (pure scheduler hint).

    Same-session A/B on MI355X (gfx950, LLVM backend):
      * SINGLE-BATCH d128 S<=1024 (num_warps==1 + V-double-buffer): +35.6-36.0%
        on bf16 AND fp16 (0.152 -> 0.112 ms) -> DSL/flash ~0.67x -> ~0.92x. The
        lone resident wave cannot hide the prefetch-in-MFMA-window cost, so
        packing the MFMAs is the dominant lever. ON.
      * num_warps>=2 d128 (S>=2048): REGRESSES 2.5-4.6% (more ILP, the fence
        over-constrains the scheduler) -> OFF.

    Scoped to exactly the V-double-buffer cohort (single-batch combo,
    max_seqlen_q <= 1024) where num_warps==1: ``_enable_v_double_buffer`` already
    pins that cohort, so reuse it. Restricted to d128 (the measured win); d64
    short single-batch combo uses num_warps==4 (BLOCK_M=128, ample ILP), where
    the fence is the over-constrained num_warps>=2 regime, so leave it OFF.
    """
    if not _enable_v_double_buffer(problem):
        return False
    return problem.head_size == 128


def _enable_early_v_schedule(problem: UnifiedAttentionProblem) -> bool:
    """Enable the early-V issue schedule on the single-batch combo.

    LONG d64 single-batch combo prefill (head_size == 64, max_seqlen_q >= 2048)
    issues the V load earlier in the iteration so the long KV loop overlaps the
    V DMA with the QK MFMA + softmax window. Bit-identical to the no-flag path.

    Scoped to d64-LONG by same-session ablation (MI355X gfx950 LLVM):
      * d64 GQA-8 S2048: T128 + early-V 104.3us is the best (vs T128 none
        108.1, T64 none 112.2) -> ON for d64 long.
      * d128 S2048/S4096: early-V REGRESSES (e.g. fp16 S2048 187us vs none
        143us) -> OFF; d128 long gets no V schedule flag.
    """
    if not _enable_single_batch_combo(problem):
        return False
    return problem.head_size == 64 and problem.max_seqlen_q >= 2048


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
    multi-batch branch requires ``max_seqlen_q > 256 AND num_seqs >= 2 AND
    not (sw > 0 AND not use_fp8)``; the single-batch combo branch
    (``_enable_single_batch_combo``) requires ``num_seqs == 1 AND
    max_seqlen_q > 256`` for d64/d128 no-SW no-FP8. Adding extra gates
    beyond those silently breaks the post-init validation in the spec
    dataclass.

    **Single-batch (num_seqs == 1) update:** the combinatorial
    autotuner proved the FULL combo is 1.5-2.7x FASTER than the legacy
    16x16x32 path for single-batch d128/d64 long prefill (and
    correctness-equal). The old gate refused num_seqs == 1 based on a
    measurement of the bare transposed path (0.74-0.85x); that was stale for
    the full combo, so ``_enable_single_batch_combo`` now short-circuits to
    True here for that cohort.

    Beyond the mw=32 prereq we gate on:

      * dtype in {bf16, fp16} (the transposed softmax/PV path is dtype-
        generic -- it casts the f32 softmax probabilities to the working
        dtype at the MFMA boundary and keeps m/l/acc in f32, so fp16 gets
        the same fp32-accumulated online softmax bf16 does; the legacy
        16x16 path it replaces lost accuracy on long-KV d128 fp16). The
        fp8 K/V cache path still uses the default kernel.
      * no FP8 K/V (transposed path doesn't dequant K/V from fp8 yet)
      * no ALiBi or QQ bias (transposed mask block doesn't fold them yet)
      * head_size in {64, 128} (hd=256 not benchmarked yet)
      * no softcap / sinks (not wired into transposed softmax yet)

    The validated ``_enable_combo_2d`` family is a superset that DOES wire
    sinks (and sliding window) through the transposed softmax, so it
    short-circuits to True here.

    The transposed/wide-K stack (``use_mfma_32x32`` + the gfx950 transpose
    reads) is gfx950-only: the gfx942 spec validator rejects ``use_mfma_32x32``
    and its dependents outright (on gfx942 this heuristic firing always built
    an invalid spec). Gate the whole thing to gfx950 so gfx942 (and any other
    arch) stays on its narrow 16x16x16 / flash path.
    """
    if _resolve_attention_arch() != "gfx950":
        return False
    if _enable_combo_2d(problem):
        return True
    # Single-batch d128/d64 prefill full-combo cohort. This is the
    # routing fix: num_seqs == 1 now gets the 32x32 transposed combo.
    if _enable_single_batch_combo(problem):
        return True
    if problem.dtype not in ("bf16", "fp16"):
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
    multi_batch = problem.max_seqlen_q > 256 and problem.num_seqs >= 2
    single_seq_hd64 = (
        problem.head_size == 64
        and problem.block_size == 16
        and problem.num_seqs <= 1
        and problem.num_queries_per_kv >= 4
        and problem.max_seqlen_q > 768
    )
    if not (multi_batch or single_seq_hd64):
        return False
    if problem.sliding_window > 0 and not problem.use_fp8:
        return False
    return True


def _enable_gfx942_small_q_narrow(problem: UnifiedAttentionProblem) -> bool:
    """Small/medium gfx942 prefill -> light narrow geometry, not the heavy ring.

    A graph-timed exhaustive sweep (loser_sweep.py) showed that for short
    contexts (``max_seqlen_q <= 768``) a LIGHT narrow-atom geometry --
    16x16x16, ``num_warps`` 1 (MHA) / 2 (GQA), ``block_m_per_warp=16``, T=64 --
    beats both the sliced-K ring AND Torch on every D64/D128 fp16+bf16 shape
    (e.g. D64 S64 0.37x, D128 GQA S512 0.74x). The ring's prelude (256 threads,
    3-slot K staging, Q-direct, mask-limit, mw=32) is pure overhead when the KV
    loop is only 1-2 tiles; it only amortises for long context, where it stays
    the default. Decode (q==1) routes to the 3D path, so it is excluded here.
    """
    return (
        _resolve_attention_arch() == "gfx942"
        and problem.dtype in ("fp16", "bf16")
        and not problem.use_fp8
        and problem.head_size in (64, 128)
        and 1 < problem.max_seqlen_q <= 768
        and problem.sliding_window == 0
        and not problem.use_sinks
        and problem.softcap == 0
        and not problem.use_alibi
        and not problem.use_qq_bias
    )


def _enable_gfx942_fp16_flash(problem: UnifiedAttentionProblem) -> bool:
    """Gate the gfx942 fp16 transposed-x8 attention family.

    The baseline L4 geometry is transposed-x8 + K single-buffer (WG=64). The
    production provider defaults D128 to wide4 (WG=256). D64 uses the same
    gfx942-legal 32x32x8 atom plus cfvst stack; measured on MI300X it improves
    S2048 D64 from ~142 TFLOPS to ~199 TFLOPS with identical correctness.
    """
    return (
        _resolve_attention_arch() == "gfx942"
        and problem.head_size in (64, 128)
        and problem.dtype == "fp16"
        and not problem.use_fp8
        and problem.sliding_window == 0
        and not problem.use_sinks
        and problem.softcap == 0
        and not problem.use_alibi
        and not problem.use_qq_bias
        # short context wins on the light narrow path instead (see
        # _enable_gfx942_small_q_narrow); the ring only amortises for long q.
        and not _enable_gfx942_small_q_narrow(problem)
    )


def _enable_gfx942_bf16_flash(problem: UnifiedAttentionProblem) -> bool:
    """Gate the gfx942 **bf16** wide-K (32x32x8) transposed flash path.

    The wide bf16 MFMA atom that is actually legal on CDNA3/gfx942 is the K=8
    ``mfma_f32_32x32x8_bf16`` (the ``.1k`` intrinsic ->
    ``v_mfma_f32_32x32x8_bf16``; selectable on ROCm 7.2 gfx942). The K=16
    ``mfma_f32_32x32x16_bf16`` advertised by the catalog is in fact CDNA4/gfx950-
    only -- the gfx942 backend ``Cannot select`` it -- so bf16 wide flash uses the
    SAME 32x32x8 path the fp16 flash family uses, just with the bf16 atom.

    The transposed 32x32 orientation feeds QK from strided LDS + a register P^T
    operand and PV from ordinary strided LDS reads, so it needs NONE of the
    gfx950-only transpose reads. Routing bf16 prefill onto this wide atom replaces
    the narrow 16x16x16 path (half-rate compute, ~63 KB LDS => 1 WG/CU) with a
    wide path (the transposed PV consumes P from registers, dropping the P_lds
    round-trip), addressing both throughput and occupancy.

    OPT-IN / DEFAULT-OFF: gated behind ``HIPDNN_GFX942_BF16_WIDE=1`` so the
    default dispatch (and every shipped kernel's byte-identity) is unchanged
    while the path is being validated/tuned. The fp16 flash path
    (``use_mfma_32x32x8``) is unaffected and takes precedence for fp16.
    """
    env = __import__("os").environ.get("HIPDNN_GFX942_BF16_WIDE", "").strip().lower()
    if env not in ("1", "on", "enable", "enabled", "yes", "true"):
        return False
    return (
        _resolve_attention_arch() == "gfx942"
        and problem.head_size in (64, 128)
        and problem.dtype == "bf16"
        and not problem.use_fp8
        and problem.sliding_window == 0
        and not problem.use_sinks
        and problem.softcap == 0
        and not problem.use_alibi
        and not problem.use_qq_bias
        # Prefill only (the wide 32x32 atom processes a 32-row M tile; decode
        # q=1 has no rows to fill and routes to the 3D split-KV / narrow path).
        and problem.max_seqlen_q > 1
        and not _enable_gfx942_small_q_narrow(problem)
    )


def _gfx942_bf16_wide_use_cfvst(problem: UnifiedAttentionProblem) -> bool:
    """Enable the conflict-free V store (cfvst) on the gfx942 bf16 wide path.

    The fp16 transposed-x8 flash family feeds the PV B-operand (V^T) through a
    conflict-free LDS layout: V is stored TRANSPOSED ``[HD, T+pad]`` via an
    in-register 2x2 ``perm_b32`` transpose + one contiguous ``ds_write``, then
    read with a wide bank-spread ``ds_read_b64`` (vs the naive 4x strided
    ``ds_read_u16`` that collapses 32 lanes onto one bank set). This is
    byte-size driven, not fp16-specific (bf16 is also 2 bytes, the perm_b32
    transpose works on raw i32 words, and the K=8 32x32x8 atom is gfx942-legal
    for bf16), so it ports unchanged to bf16.

    MEASURED (MI300X, graph, same-session steady-state, 3% median over 6 runs):
    cfvst is a small consistent WIN on D64 (S1024 ~0.106->0.104ms, S2048
    ~0.309->0.301ms, S4096 ~1.027->0.997ms = ~3% each). On D128 cfvst-only
    cannot fit the nw=4/BLOCK_M=128 geometry within the 64 KB LDS cap (it needs
    the sliced-K ring to fit, and the ring's per-slice sync overhead is NOT
    amortised by the bf16 K=8 atom on this path -- it regressed every config
    that compiled), so D128 stays on the naive-V wide geometry. So cfvst is
    enabled for D64 prefill only.

    OPT-IN: only fires under HIPDNN_GFX942_BF16_WIDE (the caller already gated
    that). HIPDNN_GFX942_BF16_CFVST=0 forces the legacy naive-V feed for A/B.
    """
    env = __import__("os").environ.get("HIPDNN_GFX942_BF16_CFVST", "").strip().lower()
    if env in ("0", "off", "disable", "disabled", "no", "false"):
        return False
    # D64 prefill only (the measured win); D128 nw=4 cfvst overflows LDS and the
    # ring that would make it fit regresses. q==1 routes to the 3D/narrow path.
    return problem.head_size == 64 and problem.max_seqlen_q > 1


def _gfx942_bf16_wide_geometry(problem: UnifiedAttentionProblem) -> Tuple[int, bool]:
    """(num_warps, use_k_single_buffer) for the gfx942 bf16 wide-K path.

    D64: nw=4 (BLOCK_M=128). With cfvst (the default; see
    _gfx942_bf16_wide_use_cfvst) the conflict-free transposed V_lds replaces the
    naive strided-V feed; K stays double-buffered (single_buffer=False).
    D128: nw=2 (BLOCK_M=64 == T=64) + K single-buffer; the nw=4 double-buffered
    form is 80 KB (> 64 KB cap), so D128 trades the second K buffer + half the
    BLOCK_M to land at 48 KB. K single-buffer requires BLOCK_M <= tile_size.
    cfvst is not enabled on D128 (it needs nw=4, which only fits with the
    sliced-K ring, and the ring regressed -- see _gfx942_bf16_wide_use_cfvst).
    """
    if problem.head_size == 128:
        # gfx942 bf16 D128 small-tile double-K (synthesized from the gfx942
        # prefill campaign; ex-T2). At T=32 (BLOCK_M = nw*32 = 64) the K_lds
        # DOUBLE buffer is 2*32*128*2 = 16 KB + V 8 KB = 24 KB -> still 2 WG/CU,
        # so restore the K double-buffer (single_k=False) instead of the T=64
        # K-single-buffer. Same-session MEASURED on MI300X (gfx942): xflash vs
        # torch FLASH 0.24->0.57-0.65 (~2.3-2.4x the shipped T=64 single-K
        # baseline) on bf16 D128 S1024/S2048/S4096, occupancy unchanged at
        # 2 WG/CU (the win is K-double-buffer prefetch overlap, not occupancy --
        # matches the CK Tile ground truth that D128 is already 2 WG/CU and the
        # gap is inner-loop scheduling). Default-ON for the bf16 D128 prefill
        # cohort with a paged cache block_size==32; env escape
        # HIPDNN_GFX942_D128_SMALLTILE_DK=0 reverts to the shipped config. T=32
        # is an EXISTING tile_size value so the per-spec emit is byte-identical
        # (C-twin parity GREEN); only the selector routing changes.
        if _enable_gfx942_d128_smalltile_doublek(problem):
            return 2, False
        return 2, True
    return 4, False


def _gfx942_bf16_wide_tile_size(problem: UnifiedAttentionProblem) -> int:
    """tile_size (T) for the gfx942 bf16 wide-K path. Default T=64; the D128
    small-tile double-K lever halves D128 to T=block_size(==32) to restore the
    K double-buffer prefetch at the same 2 WG/CU. T=32 is an EXISTING tile_size
    value -> per-spec EMIT byte-identical; this is a pure ROUTING change."""
    if _enable_gfx942_d128_smalltile_doublek(problem):
        return problem.block_size
    return 64


def _enable_gfx942_d128_smalltile_doublek(
    problem: UnifiedAttentionProblem,
) -> bool:
    """gfx942 bf16 D128 small-tile (T=32) double-K geometry.

    DEFAULT-ON for the gfx942 bf16 D128 prefill cohort with a paged cache
    block_size==32 (T=32 must be a multiple of block_size). Env escape
    HIPDNN_GFX942_D128_SMALLTILE_DK in {0,off,...} disables and reverts to the
    shipped T=64 K-single-buffer config (byte-identical OFF path). T=32 is an
    EXISTING tile_size value so the per-spec emit is byte-identical; only the
    selector routing changes. Restricted to the bf16 wide path (D128); fp16
    keeps its sliced-K ring (small-tile regressed fp16 prefill)."""
    env = (
        __import__("os")
        .environ.get("HIPDNN_GFX942_D128_SMALLTILE_DK", "")
        .strip()
        .lower()
    )
    if env in ("0", "off", "disable", "disabled", "no", "false"):
        return False
    if _resolve_attention_arch() != "gfx942":
        return False
    if problem.head_size != 128:
        return False
    if problem.block_size != 32:
        return False
    return _enable_gfx942_bf16_flash(problem)


def _enable_gfx942_d128_fp16_flash(problem: UnifiedAttentionProblem) -> bool:
    """D128 subset used by the legacy L4 geometry helpers."""
    return _enable_gfx942_fp16_flash(problem) and problem.head_size == 128


def _gfx942_flash_wide_setting() -> int:
    env = __import__("os").environ.get("HIPDNN_GFX942_FLASH_WIDE", "").strip().lower()
    if env in ("0", "off", "disable", "disabled", "no", "false"):
        return 0
    if env in ("2", "4"):
        return int(env)
    return 4


def _select_gfx942_flash_num_warps(problem: UnifiedAttentionProblem) -> int:
    # D64 and D128 prefill now share the wide (num_warps=4) sliced-K ring path
    # (the ring superseded the prior D64 nw2/single-buffer config: 13-17% faster,
    # beats Torch at S2048). See _enable_gfx942_flash_k_sliced_ring.
    wide = _gfx942_flash_wide_setting()
    return wide if wide in (2, 4) else 1


def _gfx942_flash_use_cfvst(problem: UnifiedAttentionProblem) -> bool:
    # cfvst (conflict-free V store) is required by the ring and used by both
    # D64 and D128 prefill under the wide ring geometry.
    return _gfx942_flash_wide_setting() in (2, 4)


def _gfx942_flash_use_single_buffer(problem: UnifiedAttentionProblem) -> bool:
    # Ring (D64/D128 prefill) stages K in a 3-slot ring, not single/double buffer.
    return not _gfx942_flash_use_cfvst(problem)


def _gfx942_flash_kv_cache_policy(problem: UnifiedAttentionProblem) -> str:
    return "all"


def _enable_gfx942_flash_q_direct(problem: UnifiedAttentionProblem) -> bool:
    return _enable_gfx942_fp16_flash(problem) and problem.head_size == 64


def _enable_gfx942_flash_mask_limit(problem: UnifiedAttentionProblem) -> bool:
    if not _enable_gfx942_fp16_flash(problem):
        return False
    env = __import__("os").environ.get("HIPDNN_GFX942_FLASH_MLIM", "").strip().lower()
    if env in ("0", "off", "disable", "disabled", "no", "false"):
        return False
    if env in ("1", "on", "enable", "enabled", "yes", "true", "all"):
        return True
    if env in ("d64", "64"):
        return problem.head_size == 64
    if env in ("d128", "128"):
        return problem.head_size == 128
    # Default ON for both D64 and D128. D128 mask-limit regressed on the older
    # T=128 / non-ring hot loop, but the exhaustive sweep (exhaustive_sweep.py)
    # shows it is a small positive on the now-default T=64 sliced-K ring path and
    # is part of the validated best D128 config (T=64+ring+mask-limit), which
    # beats Torch at S2048/S4096. D64 remains a measured win.
    return True


def _enable_gfx942_flash_k_sliced_ring(problem: UnifiedAttentionProblem) -> bool:
    # The sliced-K ring (32-wide K slices -> k_groups = HD/32) wins on BOTH head
    # sizes: D128 (k_groups=4) and D64 (k_groups=2). Measured T=64+ring+cfvst+
    # mask-limit (nw4) vs the prior per-head bests: D64 13-17% faster (beats Torch
    # at S2048, ~parity elsewhere); D128 beats Torch S2048/S4096. So D64 and D128
    # prefill now share the ring path.
    if not (_enable_gfx942_fp16_flash(problem) and problem.head_size in (64, 128)):
        return False
    env = (
        __import__("os").environ.get("HIPDNN_GFX942_K_SLICED_RING", "").strip().lower()
    )
    if env in ("0", "off", "disable", "disabled", "no", "false"):
        return False
    if env in ("1", "on", "enable", "enabled", "yes", "true"):
        return True
    # Default ON for prefill. Decode-ish fp16 normally routes to the 3D split-KV
    # path; keep the rare 2D-decode case on the legacy geometry until validated.
    return problem.max_seqlen_q > 1


def _enable_gfx942_flash_k_sliced_ldsseq(problem: UnifiedAttentionProblem) -> bool:
    if not _enable_gfx942_flash_k_sliced_ring(problem):
        return False
    env = __import__("os").environ.get("HIPDNN_GFX942_K_LDSSEQ", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true", "ck")


def _enable_gfx942_l4(problem: UnifiedAttentionProblem) -> bool:
    """Compatibility alias: true for the gfx942 D128 fp16 flash/L4 family."""
    return _enable_gfx942_d128_fp16_flash(problem)


def _enable_transposed_half_local_pv(problem: UnifiedAttentionProblem) -> bool:
    """Enable the half-local PV optimization for the transposed 32x32 path.

    When the transposed 32x32 kernel is selected, ``use_transposed_half_local_pv``
    rewrites the PV phase so each 32-lane half consumes only P rows it owns,
    avoiding the cross-half ``lane^32`` P fetches. Measured on Qwen3-30B-A3B
    prefill (bf16 hd64, num_seqs=1, q=2048): 125µs → 101µs = 1.24× win on top
    of plain mfma_32x32 (``/tmp/bench_prefill_min32.py``).

    This is bit-identical to the without-flag path; it is a pure VALU schedule
    rewrite. The existing prod heuristic never enabled it because the original
    transposed-32x32 calibration was done before this opt landed. Enable it
    whenever the transposed-32x32 path itself fires.
    """
    return _enable_transposed_qk_32x32(problem)


def _enable_register_pv(problem: UnifiedAttentionProblem) -> bool:
    """Enable register-resident P for the existing 16x16x32 2D path.

    P73: enable for ``dtype == "bf16"`` when the gate conditions hold
    (no sinks, no sliding window, no softcap, no alibi, no qq_bias,
    no kv_storage_dtype). The lane-XOR + bit-transpose register
    permutation in ``attention_tiled_2d.py:1020-1078`` is now stable
    enough to flip on for bf16; the f16 path stays disabled until the
    same parity sweep validates it. Other configurations stay on the
    P_LDS publish path.
    """
    if problem.dtype != "bf16":
        return False
    if problem.use_sinks:
        return False
    if problem.sliding_window > 0:
        return False
    if problem.softcap > 0:
        return False
    if problem.use_alibi:
        return False
    if problem.use_qq_bias:
        return False
    if _kv_storage_dtype(problem) is not None:
        return False
    # use_register_pv requires the 16x16x32 MFMA path; it conflicts with
    # use_mfma_32x32. When the 32x32 path is selected we leave it disabled
    # (the 32x32 path has its own in-register P pipeline).
    if _enable_mfma_32x32(problem):
        return False
    return True


def _enable_combo_2d(problem: UnifiedAttentionProblem) -> bool:
    """The full transposed-32x32 "combo" stack for the validated 2D family.

    This wires the kernel config that the parity + trace benchmarks proved
    fastest-and-correct for the AITER prefill-2D trace family (d64 / b32 /
    GQA-8 / bf16, with attention sinks) into production. The combo stacks,
    on top of ``use_mfma_32x32`` + ``use_transposed_qk_32x32``:

      * ``use_transposed_scalar_state``  (one m/l per lane + broadcast alpha)
      * ``use_transposed_mask_once``     (mask invariants once / KV iter; no-SW)
      * ``use_transposed_half_local_pv`` (avoid cross-half lane^32 P fetches)
      * ``use_mfma32_skip_legacy_qreg``  (drop the dead 16x16 Q gather)
      * ``use_transposed_mask_limit``    (single min() softmax mask; no-SW)
      * ``use_fast_paged_kv_desc``       (fast multi-block KV descriptor)

    Crucially this stack supports attention **sinks** in the transposed
    softmax path (the kernel folds the sink as the per-lane running-max
    init), which the older ``_enable_transposed_qk_32x32`` gate refused.

    Gating mirrors the validated cohort. ``max_seqlen_q > 256`` keeps it on
    chunked-prefill / long-prefill (decode-class shapes prefer the 3D path
    via ``select_path``). The half-local-PV transpose regresses on the
    very-high-num-seq sliding-window tail (per-CTA work shrinks below the
    transpose's fixed overhead), so we hand those back to the plain path.

    gfx950-only: the combo stack is built on the gfx950 wide-K 32x32 MFMA +
    transpose reads, which the gfx942 spec validator rejects. Gate to gfx950
    so gfx942 (and any other arch) never builds the combo spec.
    """
    if _resolve_attention_arch() != "gfx950":
        return False
    if problem.dtype != "bf16":
        return False
    # FP8 KV is supported via the *sync-dequant* loader, which writes bf16
    # into K_lds/V_lds (k_scale folded in) -- exactly what the 32x32 combo
    # reads. ``_enable_fp8_mfma_qk`` is forced off for the combo so the
    # in-LDS-fp8 mode (incompatible with the bf16 32x32 reads) never fires.
    # This takes the fp8 prefill cohort from ~0.5x to ~0.9x vs Triton-2d.
    if problem.use_alibi or problem.use_qq_bias or problem.softcap > 0:
        return False
    if problem.head_size != 64 or problem.block_size != 32:
        return False
    if problem.num_queries_per_kv != 8:
        return False
    if problem.max_seqlen_q <= 256:
        return False
    # (Historically the half-local-PV transpose regressed on the very-high
    # num-seqs sliding-window tail, so it was handed back to the plain path
    # at num_seqs >= 450. With waves_per_eu=3 unlocking the 4th WG/CU the
    # combo now wins across the whole SW range -- measured +22% over the
    # plain fallback at num_seqs=464 -- so the cutoff is removed.)
    return True


def _enable_transposed_subflags(problem: UnifiedAttentionProblem) -> bool:
    """Whether to stack the no-SW transposed-softmax VALU sub-flags.

    The sub-flags (``use_transposed_scalar_state`` + ``use_transposed_mask_once``
    + ``use_transposed_mask_limit`` + ``use_mfma32_skip_legacy_qreg``) are the
    correctness-equal, bit-identical VALU rewrites that turn the bare
    transposed-32x32 path into the FULL combo. They require the transposed
    path AND no sliding window (spec ``__post_init__`` enforces both).

    Historically these only fired for the narrow ``_enable_combo_2d`` family
    (bf16 / block_size==32 / GQA-8). The combinatorial autotuner showed
    the SAME stack is the proven winner for:
      * single-batch (num_seqs == 1) d128/d64 prefill (``_enable_single_batch_combo``);
      * the multi-batch (num_seqs >= 2) transposed d128/d64 prefill path -- the
        prod selector enabled mw=32 + transposed + half-local-PV there but
        LEFT the sub-flags on the table (the autotuner's ~1.19x multi-batch
        miss); they are correctness-equal and strictly faster.

    So enable them whenever the transposed-32x32 path is on and there is no
    sliding window (the SW combo keeps its own nw2/T32 mask handling).
    """
    if problem.sliding_window > 0:
        return False
    return _enable_transposed_qk_32x32(problem)


def _enable_i64_kv_addr(problem: UnifiedAttentionProblem) -> bool:
    """Enable 64-bit paged-KV addressing when the cache may exceed ~2 GiB.

    The default load path puts the full byte offset (incl.
    ``physical_block * block_stride``) in a 32-bit buffer voffset, which
    overflows -- and silently corrupts -- once the paged cache exceeds
    2 GiB. When the cache is that large we fold the per-block offset into a
    64-bit buffer base (a tiny, wave-uniform ``make_buffer_rsrc`` per
    block; measured within ~1-2% of the i32 path). Below the cap we keep
    the fast i32 path. ``num_kv_blocks == 0`` means the cache size is
    unknown (caller did not supply it) -> assume small / fast path.
    """
    if problem.num_kv_blocks <= 0:
        return False
    elem_bytes = 1 if problem.use_fp8 else 2
    block_stride = (
        problem.block_size * problem.num_kv_heads * problem.head_size * elem_bytes
    )
    cache_bytes = problem.num_kv_blocks * block_stride
    # The within-block voffset is always < block_stride, so the i32 offset
    # overflows exactly when cache_bytes > 2^31 (the last block's base alone
    # reaches 2^31). Keep the fast i32 path at/below that; switch to i64
    # strictly above. (Verified: cap=65536 bf16 = 2^31 bytes, max offset
    # 2147483646 < 2^31 -> still correct on i32.)
    return cache_bytes > 0x8000_0000


def _tiled_spec_from_problem(
    problem: UnifiedAttentionProblem,
):
    arch = _resolve_attention_arch()
    UnifiedAttention2DTiledSpec, _, _ = _tiled_2d_impl(arch)
    if arch == "gfx1250":
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
            num_warps=1,
            waves_per_eu=_select_2d_waves_per_eu(problem),
            kv_storage_dtype=_kv_storage_dtype(problem),
            tile_size=_select_2d_tile_size(problem),
            block_m_per_warp=16,
        )
    if _enable_gfx942_bf16_flash(problem):
        # gfx942 bf16 wide-K (32x32x8) transposed flash path. OPT-IN
        # (HIPDNN_GFX942_BF16_WIDE=1); default dispatch is byte-identical.
        # Uses the CDNA3-legal mfma_f32_32x32x8_bf16 atom (the K=16 bf16 atom is
        # gfx950-only). The transposed orientation consumes V from strided LDS +
        # P^T from registers (no P_lds, no gfx950-only transpose reads). Geometry:
        #   * D64  -> nw=4 (BLOCK_M=128), double-buffered K: LDS=32 KB => 2 WG/CU.
        #   * D128 -> nw=2 (BLOCK_M=64=T) + K single-buffer: LDS=48 KB (the
        #     double-buffered nw=4 form is 80 KB and overflows the 64 KB cap).
        nw, single_k = _gfx942_bf16_wide_geometry(problem)
        use_cfvst = _gfx942_bf16_wide_use_cfvst(problem)
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
            num_warps=nw,
            waves_per_eu=_select_2d_waves_per_eu(problem),
            kv_storage_dtype=_kv_storage_dtype(problem),
            tile_size=_gfx942_bf16_wide_tile_size(problem),
            block_m_per_warp=32,
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_k_single_buffer=single_k,
            # Port the fp16 flash family's conflict-free V store (cfvst) to bf16
            # (byte-size driven: bf16 == 2 bytes == fp16, the perm_b32 transpose
            # rides raw i32 words, the K=8 atom is gfx942-legal). Measured ~3%
            # win on D64 prefill (the residual naive-V bottleneck); D128 and
            # decode keep the naive-V feed (cfvst nw=4 overflows LDS there and
            # the sliced-K ring that would make it fit regressed -- see
            # _gfx942_bf16_wide_use_cfvst).
            use_conflict_free_v_store=use_cfvst,
        )
    if _enable_gfx942_fp16_flash(problem):
        num_warps = _select_gfx942_flash_num_warps(problem)
        use_cfvst = _gfx942_flash_use_cfvst(problem)
        use_single = _gfx942_flash_use_single_buffer(problem)
        use_mask_limit = _enable_gfx942_flash_mask_limit(problem)
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
            num_warps=num_warps,
            waves_per_eu=_select_2d_waves_per_eu(problem),
            kv_storage_dtype=_kv_storage_dtype(problem),
            tile_size=_select_2d_tile_size(problem),
            block_m_per_warp=_select_2d_block_m_per_warp(problem),
            use_mfma_32x32x8=True,
            use_transposed_qk_32x32=True,
            use_transposed_scalar_state=use_mask_limit,
            use_transposed_invariant_hoist=use_mask_limit,
            use_transposed_mask_once=use_mask_limit,
            use_transposed_mask_limit=use_mask_limit,
            use_conflict_free_v_store=use_cfvst,
            use_k_single_buffer=use_single,
            use_k_sliced_ring=_enable_gfx942_flash_k_sliced_ring(problem),
            use_k_sliced_ldsseq=_enable_gfx942_flash_k_sliced_ldsseq(problem),
            use_q_direct_global=_enable_gfx942_flash_q_direct(problem),
            kv_cache_policy=_gfx942_flash_kv_cache_policy(problem),
            use_i64_kv_addr=_enable_i64_kv_addr(problem),
        )
    combo = _enable_combo_2d(problem)
    combo_no_sw = combo and problem.sliding_window == 0
    # The transposed-softmax VALU sub-flags now fire for the WHOLE no-SW
    # transposed-32x32 cohort (the narrow _enable_combo_2d family, the
    # single-batch d128/d64 prefill cohort, AND the multi-batch transposed
    # d128/d64 path that previously left them on the table -- the autotuner's
    # ~1.19x multi-batch miss). ``_enable_transposed_subflags`` already
    # excludes sliding window, so OR-ing it with the existing combo gates
    # preserves the SW-combo behaviour byte-for-byte:
    #   * scalar_state / skip_legacy_qreg : old ``combo``  -> ``combo OR sub``
    #     (SW combo: combo=True keeps them True; sub=False under SW.)
    #   * mask_once / mask_limit          : old ``combo_no_sw`` -> ``combo_no_sw OR sub``
    #     (SW combo: both stay False.)
    subflags = _enable_transposed_subflags(problem)
    scalar_state = combo or subflags
    skip_legacy_qreg = combo or subflags
    mask_opts = combo_no_sw or subflags
    # gfx950-only schedule fields: the gfx942 2D spec class does not declare
    # ``use_v_double_buffer`` / ``use_sched_barrier``, and the default gfx942
    # forward reaches this shared return (no flash opt-in). Pass them only when
    # the resolved spec class actually declares the field -- gfx950 keeps the
    # exact same construction (byte-identical), while gfx942 no longer raises
    # ``TypeError: unexpected keyword argument`` on the unknown kwarg.
    _spec_field_names = {f.name for f in fields(UnifiedAttention2DTiledSpec)}
    _gfx950_schedule_fields = {}
    if "use_v_double_buffer" in _spec_field_names:
        _gfx950_schedule_fields["use_v_double_buffer"] = _enable_v_double_buffer(
            problem
        )
    if "use_sched_barrier" in _spec_field_names:
        _gfx950_schedule_fields["use_sched_barrier"] = _enable_sched_barrier(problem)
    # d128 long-context lever: K single-buffer lets the larger T=64 tile fit
    # the 2-WG/CU LDS budget at HD=128 (see _select_2d_tile_size). Gated on the
    # same d128 small-tile cohort + opt-in env so default/production routing is
    # byte-identical. Field-presence guarded (gfx942/gfx1250 spec classes lack
    # it). _enable_k_single_buffer also re-asserts the T=64 / V-single-buffer /
    # no-fp8 preconditions so it can never fire on an incompatible spec.
    if "use_k_single_buffer" in _spec_field_names and _enable_k_single_buffer(problem):
        _gfx950_schedule_fields["use_k_single_buffer"] = True
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
        use_transposed_half_local_pv=_enable_transposed_half_local_pv(problem),
        # Full combo stack (fires for the validated _enable_combo_2d family,
        # the single-batch d128/d64 prefill cohort, and the multi-batch
        # transposed d128/d64 path; a strict superset of the plain transposed
        # path). See the ``subflags`` reconciliation above.
        use_transposed_scalar_state=scalar_state,
        use_transposed_mask_once=mask_opts,
        use_transposed_mask_limit=mask_opts,
        use_mfma32_skip_legacy_qreg=skip_legacy_qreg,
        # Single-batch combo V-prefetch schedule (autotuner winners): short
        # prefill -> V double-buffer; long prefill -> early-V issue. Mutually
        # exclusive; both bit-identical to the no-flag path. Off for the
        # multi-batch combo family (its winners did not stack a V schedule).
        # (``use_v_double_buffer`` is injected via ``_gfx950_schedule_fields``
        # below -- gfx942's spec class does not declare it.)
        use_early_v_schedule=_enable_early_v_schedule(problem),
        # The fast paged-KV descriptor is specialised for bf16 / T=64 /
        # num_warps=4, which only the bf16 no-SW combo geometry uses (SW
        # combo is nw2 / T=32; fp8 combo uses the sync-dequant loader). The
        # gfx950 spec restricts it further to the exact 64-query / 8-kv head
        # cohort it was built for; `_enable_combo_2d` only checks the GQA-8
        # *ratio*, so a tensor-parallel-sharded GQA-8 model (e.g. 16/2) would
        # otherwise enable it and trip the spec validator. Match the validator's
        # absolute head-count restriction so non-64/8 GQA-8 combo shapes keep
        # the rest of the combo stack without the fast descriptor.
        use_fast_paged_kv_desc=(
            combo_no_sw
            and not problem.use_fp8
            and problem.num_query_heads == 64
            and problem.num_kv_heads == 8
        ),
        use_register_pv=_enable_register_pv(problem),
        use_fp8_mfma_qk=_enable_fp8_mfma_qk(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
        # CK-Tile-derived sched_barrier steering (lever 3 from the CK Tile ISA analysis). Fences the
        # QK MFMA cluster from the post-QK async prefetch VMEM so the LLVM
        # scheduler keeps the MFMAs packed. Additive perf knob (no routing
        # change); enabled only for the single-batch d128 short-prefill cohort
        # (num_warps==1 + V-double-buffer) where the single resident wave cannot
        # otherwise hide the prefetch-in-MFMA-window cost.
        # (``use_sched_barrier`` is injected via ``_gfx950_schedule_fields``
        # below -- gfx942's spec class does not declare it.)
        **_gfx950_schedule_fields,
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
    if _resolve_attention_arch() == "gfx1250":
        return 16
    # mw=32 (BLOCK_M = 32 * num_warps) only pays off when a path actually
    # exploits the doubled M rows:
    #   * the transposed-32x32 / combo path (32x32 MFMA atoms), or
    #   * the fp8 path (amortises the per-K-byte dequant over 2x the rows).
    # Picking mw=32 with plain 16x16 atoms (no transpose, no fp8) was a
    # latent trap: it doubled BLOCK_M without the atom benefit and ran
    # ~1.4x slower than mw=16 on the sinks trace family. ``select_path``
    # has already routed decode-class shapes to 3D before we get here.
    # gfx942 (CDNA3 / MI300X) head_size=64: the oracle is always mw=32 with a
    # 1x tile (tile_size == block_size) on the plain 16x16 atom path. gfx942 is
    # LDS-bound at one CTA/CU on the 64 KB part, so the mw=32 + 2x-tile combo is
    # rejected while the 1x-tile mw=32 combo fits, yielding 1.7-2.0x over mw=16.
    # Arch-gated so the gfx950 selection below is byte-identical (gfx950 never
    # enters this branch). The C++ SdpaCandidateSelector is the shipping +
    # measured path on gfx942; this mirrors it for DSL-side consistency.
    # Small/medium gfx942 prefill light narrow path uses one M=16 atom/warp
    # (BLOCK_M=16*nw); mw=32 is pure overhead for the 1-2-tile KV loop. Precedes
    # the D64/L4 mw=32 rules.
    if _enable_gfx942_small_q_narrow(problem):
        return 16
    if _resolve_attention_arch() == "gfx942" and problem.head_size == 64:
        return 32
    # gfx942 D128 fp16 L4 (shipped): one M=32 atom per warp (BLOCK_M=32).
    if _enable_gfx942_l4(problem):
        return 32
    if _enable_transposed_qk_32x32(problem):  # includes _enable_combo_2d
        return 32
    if problem.use_fp8 and problem.max_seqlen_q > 256 and problem.num_seqs >= 2:
        return 32
    # Qwen3-30B-A3B prefill specialization (bf16, hd64, BS=16, num_seqs=1,
    # num_queries_per_kv>=4). Long single-seq prefill is the dominant
    # bucket for chunked-prefill production traces with one decode-class
    # request at a time. Measured at ``/tmp/bench_prefill_sweep2.py``:
    # mw=32 with NW=4 + transposed-32x32 + half_local_pv beats the default
    # mw=16 by 1.5-1.8× at q in {1024, 2048}. Below q=1024 the per-CTA
    # prelude is small enough that mw=16 wins.
    if (
        problem.head_size == 64
        and problem.block_size == 16
        and problem.num_seqs <= 1
        and not problem.use_fp8
        and problem.dtype == "bf16"
        and problem.num_queries_per_kv >= 4
        and problem.max_seqlen_q > 768
        and problem.sliding_window == 0
        and problem.softcap == 0
        and not problem.use_sinks
        and not problem.use_alibi
        and not problem.use_qq_bias
    ):
        return 32
    return 16


def _num_segments(problem: UnifiedAttentionProblem) -> int:
    """Mirror AITER ``select_3d_config`` num_segments derivation exactly."""
    attn_cfg, _ = problem.select_3d()
    segments = attn_cfg.NUM_SEGMENTS_PER_SEQ
    # AITER's Triton path scales decode splits with the full MI300X CU count
    # (typically 128 segments here). The gfx942 rocke segment+reduce kernels
    # are not the same implementation; after host-overhead cleanup, 128-way
    # split regresses these q=1/kv=2048 shapes while 64 segments is stable.
    if (
        _resolve_attention_arch() == "gfx942"
        and problem.max_seqlen_q == 1
        and problem.max_seqlen_k <= 2048
        and problem.sliding_window == 0
    ):
        if problem.head_size == 64:
            return min(segments, 32)
        if problem.head_size == 128:
            return min(segments, 16)
        return min(segments, 64)
    return segments


def _gfx942_3d_tile_size_override(problem: UnifiedAttentionProblem) -> Optional[int]:
    arch = _resolve_attention_arch()
    if not (arch == "gfx942" and problem.head_size >= 128 and problem.block_size >= 32):
        return None
    return problem.block_size // 2


def _select_3d_waves_per_eu(problem: UnifiedAttentionProblem) -> Optional[int]:
    if problem.waves_per_eu is not None:
        return problem.waves_per_eu
    if _resolve_attention_arch() == "gfx1250":
        return 2
    return None


def _enable_gfx942_3d_invariant_hoist(problem: UnifiedAttentionProblem) -> bool:
    if _resolve_attention_arch() != "gfx942":
        return False
    env = __import__("os").environ.get("HIPDNN_GFX942_3D_HOIST", "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true")


def _enable_gfx942_3d_wide_kv_load(problem: UnifiedAttentionProblem) -> bool:
    """Wide 128-bit direct global->register->LDS KV feed for the 3D decode
    segment kernel (replaces the gfx942 1-DWORD async DMA). fp16/bf16 only.

    Opt-in via HIPDNN_GFX942_3D_WKV while it is A/B'd against the async path;
    promoted to default once the win is confirmed.
    """
    if _resolve_attention_arch() != "gfx942":
        return False
    if _kv_storage_dtype(problem) is not None:  # fp8 path already loads wide
        return False
    env = __import__("os").environ.get("HIPDNN_GFX942_3D_WKV", "").strip().lower()
    if env in ("0", "off", "disable", "disabled", "no", "false"):
        return False
    # Default ON: validated bit-identical and ~8% faster on D128 decode, neutral
    # on D64, never slower (decode_ab.py A/B). Strict improvement over the 1-DWORD
    # async DMA. (The remaining decode gap vs Torch is structural -- 64-thread /
    # narrow-MFMA / LDS round-trip -- and needs the segment-kernel restructure.)
    return True


def _env_enabled_true(var: str) -> bool:
    env = __import__("os").environ.get(var, "").strip().lower()
    return env in ("1", "on", "enable", "enabled", "yes", "true")


def _env_disabled(var: str) -> bool:
    env = __import__("os").environ.get(var, "").strip().lower()
    return env in ("0", "off", "disable", "disabled", "no", "false")


def _gfx1250_3d_num_waves() -> int:
    """gfx1250 3D decode cooperative multi-wave32 CTA width override
    (``HIPDNN_GFX1250_3D_WAVES`` in {1,2,4,8}).

    Returns ``-1`` (the default when unset / invalid) to mean *auto* — the
    tuning policy (:func:`_resolve_gfx1250_tiled3d`) picks the wave count from
    the problem shape. An explicit value in {1,2,4,8} pins it (overrides auto)."""
    env = __import__("os").environ.get("HIPDNN_GFX1250_3D_WAVES", "").strip()
    if env == "":
        return -1
    try:
        w = int(env)
    except ValueError:
        return -1
    return w if w in (1, 2, 4, 8) else -1


# ---------------------------------------------------------------------------
# gfx1250 3D decode tuning policy
# ---------------------------------------------------------------------------
# Single source of truth for *which config* the gfx1250 split-KV decode kernel
# runs for a given problem. Both the spec builder (kernel emission) and the
# cache key flow from one ``_resolve_gfx1250_tiled3d(problem)`` call, so the
# built kernel and its cache identity can never drift. This is the production
# routing path (not examples): the runtime calls it to auto-select per live
# shape. Empirical "best config per shape" knowledge from the examples sweeps is
# *distilled here* as policy, rather than living as logic in the harnesses.
# Hardened selection (occupancy gating, lever pairing) lands in this one place.


@dataclass(frozen=True)
class _ResolvedTiled3D:
    """Resolved gfx1250 3D decode knobs for one problem (see policy note above)."""

    num_segments: int
    num_waves: int
    waves_per_eu: Optional[int]
    kv_storage_dtype: Optional[str]
    tile_size_override: Optional[int]
    use_invariant_hoist: bool
    use_wide_kv_load: bool
    use_register_p: bool
    use_wide_lds_reads: bool
    use_dtla_prefetch: bool
    use_ds_tr_reads: bool
    use_fused_reduce: bool
    use_dpp_softmax: bool


def _resolve_gfx1250_tiled3d(problem: UnifiedAttentionProblem) -> _ResolvedTiled3D:
    """Resolve the gfx1250 3D decode config for ``problem`` (the tuning policy).

    Env knobs (``HIPDNN_GFX1250_3D_*``) override the policy for A/B work; with no
    env set this returns the production defaults. Behaviour-preserving extract of
    the former inline ``_tiled_3d_spec_from_problem`` gfx1250 branch.
    """
    dtla = _env_enabled_true("HIPDNN_GFX1250_3D_DTLA")
    dstr = _env_enabled_true("HIPDNN_GFX1250_3D_DSTR")
    wkv = _env_enabled_true("HIPDNN_GFX1250_3D_WKV")
    regp = _env_enabled_true("HIPDNN_GFX1250_3D_REGP")

    num_waves = _gfx1250_3d_num_waves()
    if num_waves == -1:  # auto: policy picks the wave count from the shape
        num_waves = 1
    # lever 1, default-on (transposed V + wide ds_load); auto-off when an incompatible
    # lever (multi-wave / double-buffer / register-P) is on.
    use_wide_lds_reads = (
        num_waves == 1
        and not wkv
        and not regp
        and not dtla
        and not dstr
        and not _env_disabled("HIPDNN_GFX1250_3D_WLDS")
    )
    return _ResolvedTiled3D(
        num_segments=_num_segments(problem),
        num_waves=num_waves,
        waves_per_eu=_select_3d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size_override=_gfx942_3d_tile_size_override(problem),  # None on gfx1250
        use_invariant_hoist=_env_enabled_true("HIPDNN_GFX1250_3D_HOIST"),
        use_wide_kv_load=wkv,
        use_register_p=regp,
        use_wide_lds_reads=use_wide_lds_reads,
        use_dtla_prefetch=dtla,
        use_ds_tr_reads=dstr,
        use_fused_reduce=_env_enabled_true("HIPDNN_GFX1250_3D_FRED"),
        # DPP row_xmask softmax reduction (VALU, not LDS port); default on,
        # disable via HIPDNN_GFX1250_3D_DPP=0.
        use_dpp_softmax=not _env_disabled("HIPDNN_GFX1250_3D_DPP"),
    )


def _tiled_3d_spec_from_problem(
    problem: UnifiedAttentionProblem,
):
    arch = _resolve_attention_arch()
    UnifiedAttention3DTiledSpec, *_ = _tiled_3d_impl(arch)
    tile_size_override = _gfx942_3d_tile_size_override(problem)
    if arch == "gfx1250":
        r = _resolve_gfx1250_tiled3d(problem)
        return UnifiedAttention3DTiledSpec(
            head_size=problem.head_size,
            block_size=problem.block_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            use_sinks=problem.use_sinks,
            sliding_window=problem.sliding_window,
            has_softcap=problem.softcap > 0,
            num_segments=r.num_segments,
            use_alibi=problem.use_alibi,
            use_qq_bias=problem.use_qq_bias,
            num_seqs=problem.num_seqs,
            waves_per_eu=r.waves_per_eu,
            kv_storage_dtype=r.kv_storage_dtype,
            tile_size_override=r.tile_size_override,
            use_invariant_hoist=r.use_invariant_hoist,
            use_wide_kv_load=r.use_wide_kv_load,
            use_register_p=r.use_register_p,
            num_waves=r.num_waves,
            use_wide_lds_reads=r.use_wide_lds_reads,
            use_dtla_prefetch=r.use_dtla_prefetch,
            use_ds_tr_reads=r.use_ds_tr_reads,
            use_fused_reduce=r.use_fused_reduce,
            use_dpp_softmax=r.use_dpp_softmax,
        )
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
        waves_per_eu=_select_3d_waves_per_eu(problem),
        kv_storage_dtype=_kv_storage_dtype(problem),
        tile_size_override=tile_size_override,
        use_invariant_hoist=_enable_gfx942_3d_invariant_hoist(problem),
        use_wide_kv_load=_enable_gfx942_3d_wide_kv_load(problem),
        use_i64_kv_addr=_enable_i64_kv_addr(problem),
    )


def _tiled_3d_cache_key(problem: UnifiedAttentionProblem) -> Tuple:
    base = (
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
        _gfx942_3d_tile_size_override(problem),
        _select_3d_waves_per_eu(problem),
        _enable_gfx942_3d_invariant_hoist(problem),
        _enable_gfx942_3d_wide_kv_load(problem),
        _kv_storage_dtype(problem),
        _enable_i64_kv_addr(problem),
    )
    if _resolve_attention_arch() == "gfx1250":
        sp = _tiled_3d_spec_from_problem(problem)
        return base + (
            sp.use_invariant_hoist,
            sp.use_wide_kv_load,
            sp.use_register_p,
            getattr(sp, "wmma_spacing", 0),
            getattr(sp, "num_waves", 1),
            getattr(sp, "use_wide_lds_reads", False),
            getattr(sp, "use_dtla_prefetch", False),
            getattr(sp, "use_ds_tr_reads", False),
            getattr(sp, "use_fused_reduce", False),
            getattr(sp, "use_dpp_softmax", False),
        )
    return base


def _3d_signature(dtype: str, *, kv_dtype: Optional[str] = None):
    from ...helpers.spec import SignatureBuilder

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
    from ...helpers.spec import SignatureBuilder

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
    from ...helpers.spec import SignatureBuilder

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
    use_graph: bool = True,
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
    if use_graph and _enable_3d_graph_replay(problem) and not _torch_stream_capturing():
        graph_key = (
            cache_key,
            int(problem.total_q),
            int(stream),
            id(q),
            id(k),
            id(v),
            id(out),
            id(cu_seqlens_q),
            id(seqused_k),
            id(block_table),
            id(sinks) if sinks is not None else 0,
            id(alibi_slopes) if alibi_slopes is not None else 0,
            id(qq_bias) if qq_bias is not None else 0,
            float(softmax_scale),
            float(k_scale),
            float(v_scale),
            float(softcap),
            int(bt_stride),
            int(qq_bias_stride_0),
        )
        graph = _3D_GRAPHS.get(graph_key)
        if graph is None:
            import torch

            # Build/load launchers and allocate workspace outside capture.
            _run_3d_tiled(
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
                stream=stream,
                k_scale=k_scale,
                v_scale=v_scale,
                use_graph=False,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with no_fence():
                    _run_3d_tiled(
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
                        stream=stream,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        use_graph=False,
                    )
            _3D_GRAPHS[graph_key] = graph
            _3D_GRAPH_REFS[graph_key] = (
                q,
                k,
                v,
                out,
                cu_seqlens_q,
                seqused_k,
                block_table,
                sinks,
                alibi_slopes,
                qq_bias,
            )
        graph.replay()
        if _resolved_fence(True):
            wait_stream_and_release(int(stream))
        return LaunchSummary(launches=2)

    # Lazily build (and cache) the PipelineLauncher + WorkspacePool for
    # this problem shape. This single object owns: the compiled HSACO
    # blobs, the loaded HIP module handles, the kernel function
    # handles, and the segm_* workspace tensors. All five
    # categories of lifetime / race / overhead bugs documented in
    # ``rocke/runtime/launcher.py`` are removed by construction; the
    # only remaining per-call cost is packing args and issuing two
    # ``hipModuleLaunchKernel`` calls on the caller's stream.
    prepared = _get_3d_pipeline(problem, cache_key, num_segments)
    segm_output, segm_max, segm_expsum = prepared.workspace(
        problem, num_segments, q.device
    )
    if _resolve_attention_arch() == "gfx1250":
        import torch

        segm_max.fill_(-1e30)
        segm_expsum.zero_()
        segm_output.zero_()

    bound_key = (
        cache_key,
        int(problem.total_q),
        str(q.device),
        id(q),
        id(k),
        id(v),
        id(cu_seqlens_q),
        id(seqused_k),
        id(block_table),
        id(sinks) if sinks is not None else 0,
        id(alibi_slopes) if alibi_slopes is not None else 0,
        id(qq_bias) if qq_bias is not None else 0,
        float(softmax_scale),
        float(k_scale),
        float(v_scale),
        float(softcap),
        int(problem.num_seqs),
        int(bt_stride),
        int(qq_bias_stride_0),
    )
    cached_values = _3D_BOUND_VALUES.get(bound_key)
    if cached_values is None:
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
        _3D_BOUND_VALUES[bound_key] = (seg_vals, red_vals)
    else:
        seg_vals, red_vals = cached_values
        # Output is commonly a fresh tensor per request; workspace and inputs are
        # fixed by the bound key.
        red_vals["output_ptr"] = out
    return prepared.pipeline(
        (seg_vals, red_vals),
        (prepared.seg_config, prepared.red_config),
        stream=int(stream),
    )


@dataclass
class _Attention3DPrepared:
    pipeline: PipelineLauncher
    pool: WorkspacePool
    seg_config: LaunchConfig
    red_config: LaunchConfig
    workspace_specs: Dict[Any, Tuple[WorkspaceSpec, WorkspaceSpec, WorkspaceSpec]]
    workspace_tensors: Dict[Any, Tuple[Any, Any, Any]]
    seg_values: Dict[str, Any]
    red_values: Dict[str, Any]

    def workspace(self, problem: UnifiedAttentionProblem, num_segments: int, device):
        key = device
        if key not in self.workspace_tensors:
            specs = self.workspace_specs.get(key)
            if specs is None:
                specs = _attention_3d_workspace_specs(problem, num_segments, device)
                self.workspace_specs[key] = specs
            segm_output = self.pool.get_spec(specs[0])
            segm_max = self.pool.get_spec(specs[1])
            segm_expsum = self.pool.get_spec(specs[2])
            self.workspace_tensors[key] = (segm_output, segm_max, segm_expsum)
        return self.workspace_tensors[key]


@dataclass(frozen=True)
class _Attention2DLaunchMeta:
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]


# Per-cache-key prepared 3D launch state. Built lazily at first dispatch for a
# given problem shape; reused across every subsequent dispatch and timing-loop
# iteration. This is the same shape as CK Tile's `fmha_bwd_launcher` (one object
# per problem instance, owns kernels + workspace, survives every launch).
_3D_PIPELINES: Dict[Tuple, _Attention3DPrepared] = {}
_3D_BOUND_VALUES: Dict[Tuple, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
_3D_GRAPHS: Dict[Tuple, Any] = {}
_3D_GRAPH_REFS: Dict[Tuple, Tuple[Any, ...]] = {}
_2D_LAUNCHERS: Dict[Tuple, KernelLauncher] = {}
_2D_LAUNCH_META: Dict[Tuple, _Attention2DLaunchMeta] = {}
_2D_GRAPHS: Dict[Tuple, Any] = {}
_2D_GRAPH_REFS: Dict[Tuple, Tuple[Any, ...]] = {}
_SCALAR_LAUNCHERS: Dict[Tuple, KernelLauncher] = {}


def _recommend_graph_replay(problem: UnifiedAttentionProblem) -> bool:
    """Graph-vs-ungraph heuristic.

    CUDA-graph capture only removes *per-launch host overhead* (the Python
    dispatch + kernarg pack + ``hipModuleLaunchKernel``), so it pays off exactly
    when that overhead is a large fraction of the kernel time:

      * **decode** (``max_seqlen_q == 1``)  -- the kernel is tiny, overhead
        dominates; tensors are stable across steps so the captured graph
        replays. (The 3D split-KV decode path uses this.)
      * **short prefill** (``max_seqlen_q <= 768``, the light-narrow regime)
        -- 1-2 KV tiles, so overhead is a big fraction; graphing wins 3-4x.

    **Long prefill** (``q > 768``) is kernel-bound: the overhead is noise and
    its tensors usually differ per call, so graphing would only add recapture
    cost. -> ungraph. Feature-flagged shapes (sinks/alibi/qq/softcap/sw/fp8) are
    excluded for now. The dispatcher only engages an internal graph when the
    caller is not already capturing (frameworks that graph the whole forward
    take precedence)."""
    if problem.use_sinks or problem.use_alibi or problem.use_qq_bias:
        return False
    if problem.softcap > 0 or problem.sliding_window > 0 or problem.use_fp8:
        return False
    return problem.max_seqlen_q <= 768


def _graph_env_enabled(var: str) -> bool:
    env = __import__("os").environ.get(var, "").strip().lower()
    return env not in ("0", "off", "disable", "disabled", "no", "false")


def _enable_3d_graph_replay(problem: UnifiedAttentionProblem) -> bool:
    arch = _resolve_attention_arch()
    if arch == "gfx942":
        if not _recommend_graph_replay(problem):
            return False
        return _graph_env_enabled("HIPDNN_GFX942_3D_GRAPH")
    if arch == "gfx1250":
        # Decode / short-q 3D is launch-overhead bound. CUDA-graph capture is
        # gfx942-validated; keep gfx1250 opt-in until the ROCm graph path is
        # exercised (``HIPDNN_GFX1250_3D_GRAPH=1``).
        if problem.max_seqlen_q > 768:
            return False
        if problem.use_alibi or problem.use_qq_bias or problem.softcap > 0:
            return False
        if problem.sliding_window > 0:
            return False
        env = (
            __import__("os").environ.get("HIPDNN_GFX1250_3D_GRAPH", "").strip().lower()
        )
        return env in ("1", "on", "enable", "enabled", "yes", "true")
    if arch == "gfx950":
        # The 3D split-KV decode launch is host/dispatch bound here: the two
        # kernels (segment + reduce) device-execute in ~26-35us but the eager
        # Python dispatch adds a ~17us flat floor, so the wall time is ~43us and
        # does not scale with context length. Capturing the (segment + reduce)
        # launch sequence into a hipGraph and replaying it removes that
        # per-launch host cost and roughly halves decode latency (43->21us at
        # k=2048), producing bitwise-identical output. Gated to the decode /
        # short-q regime (``max_seqlen_q <= 768``) so prefill routing is never
        # affected, and feature-flagged shapes are excluded. Opt-in via
        # ``HIPDNN_GFX950_3D_GRAPH=1`` until broadly validated across traces.
        if problem.max_seqlen_q > 768:
            return False
        if problem.use_alibi or problem.use_qq_bias or problem.softcap > 0:
            return False
        if problem.sliding_window > 0 or problem.use_fp8:
            return False
        env = __import__("os").environ.get("HIPDNN_GFX950_3D_GRAPH", "").strip().lower()
        return env in ("1", "on", "enable", "enabled", "yes", "true")
    return False


def _enable_2d_graph_replay(problem: UnifiedAttentionProblem) -> bool:
    """Auto-graph the 2D prefill launch for the short-context regime where the
    light-narrow kernel is host-overhead-bound (see _recommend_graph_replay)."""
    if _resolve_attention_arch() != "gfx942":
        return False
    if problem.max_seqlen_q <= 1 or not _recommend_graph_replay(problem):
        return False
    return _graph_env_enabled("HIPDNN_GFX942_2D_GRAPH")


# Sentinel: 2D graph path could not handle the problem -> caller falls through.
_GRAPH_FALLBACK = object()


def _cheap_2d_sig(problem) -> Tuple:
    """Cheap shape-signature that determines the 2D kernel in a fixed-env
    process (the kernel is a deterministic function of these problem fields).
    Lets the graph fast path skip the ~16us supports + cache_key selector work
    on replay. Env/monkeypatch changes (sweeps) invalidate via ``_2D_GRAPHS``
    being cleared, so the signature need not encode the env knobs."""
    return (
        "2dg",
        problem.head_size,
        problem.num_query_heads,
        problem.num_kv_heads,
        problem.block_size,
        problem.dtype,
        problem.max_seqlen_q,
        problem.max_seqlen_k,
        problem.num_seqs,
        problem.sliding_window,
        bool(problem.use_sinks),
        float(problem.softcap),
        bool(problem.use_alibi),
        bool(problem.use_qq_bias),
        bool(problem.use_fp8),
        int(problem.total_q),
    )


def _run_2d_graphed(
    problem,
    *,
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    block_table,
    softcap,
    sinks,
    bt_stride,
    alibi_slopes,
    qq_bias,
    qq_bias_stride_0,
    k_scale,
    v_scale,
    out_scale,
    stream,
):
    """Look up (or build once) + replay a CUDA graph around the single 2D launch.

    The graph cache is keyed on a cheap shape signature + tensor identities +
    scalar args, looked up BEFORE any selector work -- so a replay (the steady
    state: benchmarks, frameworks that reuse KV/IO buffers) skips supports +
    cache_key + the kernarg pack entirely, leaving only the ~us key build +
    ``graph.replay()``. A new buffer set rebuilds (correctness). Returns
    ``_GRAPH_FALLBACK`` if the shape isn't tiled-supported so the caller can take
    the non-graph path. Tensors + kernarg pack are held alive in
    ``_2D_GRAPH_REFS``."""
    graph_key = _cheap_2d_sig(problem) + (
        int(stream),
        id(q),
        id(k),
        id(v),
        id(out),
        id(cu_seqlens_q),
        id(seqused_k),
        id(block_table),
        id(sinks) if sinks is not None else 0,
        id(alibi_slopes) if alibi_slopes is not None else 0,
        id(qq_bias) if qq_bias is not None else 0,
        float(softmax_scale),
        float(k_scale),
        float(v_scale),
        float(softcap),
        int(bt_stride),
        int(qq_bias_stride_0),
    )
    graph = _2D_GRAPHS.get(graph_key)
    if graph is not None:
        graph.replay()
        return None
    # Miss: the full build path (selectors + launcher + kernarg pack + capture).
    ok_t, _reason = supports_native_unified_attention_tiled(problem)
    if not ok_t:
        return _GRAPH_FALLBACK
    import torch

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
    meta = _get_2d_launch_meta(problem, key)
    cfg = LaunchConfig(grid=meta.grid, block=meta.block, stream=int(stream))

    def _do():
        launcher(vals, config=cfg)

    # Warm OUTSIDE capture (compile already happened in _get_2d_launcher; this
    # primes the module + produces the correct `out`).
    with no_fence():
        _do()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with no_fence():
            _do()
    _2D_GRAPHS[graph_key] = graph
    _2D_GRAPH_REFS[graph_key] = (
        vals,
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
        block_table,
        sinks,
        alibi_slopes,
        qq_bias,
    )
    graph.replay()
    return None


def _torch_stream_capturing() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


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
) -> _Attention3DPrepared:
    prepared_key = cache_key + ("total_q", int(problem.total_q))
    if prepared_key in _3D_PIPELINES:
        return _3D_PIPELINES[prepared_key]
    if cache_key not in _ATTN_3D_TILED_CACHE:
        arch = _resolve_attention_arch()
        (
            _,
            UnifiedAttentionReduceTiledSpec,
            build_unified_attention_3d_tiled,
            build_unified_attention_reduce_tiled,
            _,
        ) = _tiled_3d_impl(arch)
        seg_spec = _tiled_3d_spec_from_problem(problem)
        reduce_spec = UnifiedAttentionReduceTiledSpec(
            head_size=problem.head_size,
            num_query_heads=problem.num_query_heads,
            num_kv_heads=problem.num_kv_heads,
            dtype=problem.dtype,
            num_segments=num_segments,
            waves_per_eu=_select_3d_waves_per_eu(problem),
        )
        seg_art = compile_kernel(
            build_unified_attention_3d_tiled(seg_spec, arch=arch),
            arch=arch,
            capture_ir_text=False,
        )
        red_art = compile_kernel(
            build_unified_attention_reduce_tiled(reduce_spec, arch=arch),
            arch=arch,
            capture_ir_text=False,
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
    block_q = (
        16 // problem.num_queries_per_kv if problem.num_queries_per_kv <= 16 else 1
    )
    total_num_q_blocks = problem.total_q // block_q + problem.num_seqs
    # gfx1250 (gfx1250) runs the split-KV segment + reduce as one wave32 CTA; the
    # CDNA wave64 archs (gfx950/gfx942) use a wave64 CTA.
    wave_size = 32 if _resolve_attention_arch() == "gfx1250" else 64
    # Must match the wave count the seg spec was built with -> resolve through the
    # same tuning policy (not the raw env reader, which now returns -1 for auto).
    seg_waves = (
        _resolve_gfx1250_tiled3d(problem).num_waves
        if _resolve_attention_arch() == "gfx1250"
        else 1
    )
    prepared = _Attention3DPrepared(
        pipeline=pipeline,
        pool=pool,
        seg_config=LaunchConfig(
            grid=(
                int(total_num_q_blocks),
                int(problem.num_kv_heads),
                int(num_segments),
            ),
            block=(wave_size * seg_waves, 1, 1),
        ),
        red_config=LaunchConfig(
            grid=(int(problem.total_q), int(problem.num_query_heads), 1),
            block=(wave_size, 1, 1),
        ),
        workspace_specs={},
        workspace_tensors={},
        seg_values={},
        red_values={},
    )
    _3D_PIPELINES[prepared_key] = prepared
    return prepared


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

    See the out-of-tree ``probe_hip_lowering.py`` for the per-shape sweep.
    """
    if _resolve_attention_arch() == "gfx1250":
        return "llvm"
    if problem.compile_backend in ("llvm", "hipcc"):
        return problem.compile_backend
    # Auto: HIP for large-batch bf16/fp16 prefill workloads where
    # hipcc's heavier scheduler measurably wins. FP8 stays on LLVM
    # until the HIP path is exercised on the dequant-loader kernels.
    if problem.use_fp8:
        return "llvm"
    # gfx942 kernels are validated through LLVM-direct. The HIP debug backend has
    # emitted code objects that hipModuleLoadData rejects for both cfvst D128 and
    # narrow D64 MI300X kernels, so keep gfx942 on the proven direct path unless
    # the caller explicitly overrides ``compile_backend``.
    if _resolve_attention_arch() == "gfx942":
        return "llvm"
    # MHA (num_queries_per_kv == 1) head_size-64 prefill REGRESSES under
    # hipcc: the perf_attn2d sweep (B2 H8 d64 fp16, gfx950) measured the
    # hipcc auto path at 95/186 TFLOPS for s1024/s2048 vs 131/218 on the
    # LLVM-direct path (the byte-identical fallback geometry), i.e. hipcc
    # is 17-27% SLOWER here, not the ~5% faster the b4 GQA shape showed.
    # The fast-path 16x16 MHA loop body does not benefit from clang's
    # heavier unrolled-loop scheduler the way the GQA-8 / combo geometry
    # does, so pin this family to LLVM-direct. (The combo / GQA path below
    # keeps its hipcc eligibility.)
    if (
        problem.num_queries_per_kv == 1
        and problem.head_size == 64
        and not _enable_combo_2d(problem)
    ):
        return "llvm"
    # Single-batch (num_seqs == 1) d128/d64 prefill combo cohort: pin to
    # LLVM-direct. The hipcc scheduler's heavier unrolled-loop pass only pays
    # off on the LARGE multi-batch grid; for the single-seq grid it is a big
    # REGRESSION (same-session A/B: d128
    # S1024 none_hipcc 95us vs none_llvm 49us; d128 S2048 hipcc ~174us vs llvm
    # 143us). The total_work rule below would otherwise pick hipcc for these
    # long single-seq shapes, so gate them to LLVM explicitly.
    if _enable_single_batch_combo(problem):
        return "llvm"
    # P_LDS transposed-read PV path correctness pin. When the 2D
    # kernel lands on the legacy 16x16x32 path with neither the register-P
    # migration (``_enable_register_pv``) nor the 32x32 transposed combo
    # (``_enable_mfma_32x32``), the PV stage publishes the softmax
    # probabilities to LDS as 16-bit and reads them back transposed via
    # ``ds_read_tr16``. The shipped system hipcc (ROCm 7.0 / LLVM 20)
    # MISCOMPILES that transposed-LDS-read sequence: the output is ~0.84
    # max-abs wrong (bad != 0) versus the fp32 reference, while the
    # LLVM-direct (comgr / LLVM 22) build of the *same* IR is correct
    # (rounding tol, bad == 0). The miscompile is path-specific, not
    # dtype-specific -- it reproduces on bf16 too when register-P is forced
    # off -- but the only production config that still routes onto this path
    # is head_size==256 (the 32x32 combo excludes hd256 and register-P is
    # bf16-only), so in practice this pins fp16 d256 long-prefill (which auto
    # would otherwise compile via hipcc). Pin it to the proven LLVM-direct
    # backend until the system hipcc is upgraded / the path is moved onto
    # register-P for hd256. The perf delta is the ~5% hipcc-scheduler edge,
    # which correctness strictly outranks.
    if not _enable_register_pv(problem) and not _enable_mfma_32x32(problem):
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
        arch = _resolve_attention_arch()
        _, build_unified_attention_2d_tiled, _ = _tiled_2d_impl(arch)
        spec = _tiled_spec_from_problem(problem)
        kernel = build_unified_attention_2d_tiled(spec, arch=arch)
        backend = _select_2d_compile_backend(problem)
        if backend == "hipcc":
            from ...helpers.compile import compile_kernel_via_hipcc

            artifact = compile_kernel_via_hipcc(kernel)
        else:
            artifact = compile_kernel(kernel, arch=arch, capture_ir_text=False)
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


def _get_2d_launch_meta(
    problem: UnifiedAttentionProblem,
    cache_key: Tuple,
) -> _Attention2DLaunchMeta:
    meta_key = cache_key + ("total_q", int(problem.total_q))
    if meta_key in _2D_LAUNCH_META:
        return _2D_LAUNCH_META[meta_key]
    arch = _resolve_attention_arch()
    num_warps = (
        _select_gfx942_flash_num_warps(problem)
        if _enable_gfx942_fp16_flash(problem)
        else _select_2d_num_warps(problem)
    )
    block_m_per_warp = _select_2d_block_m_per_warp(problem)
    block_m = num_warps * block_m_per_warp
    block_q = (
        block_m // problem.num_queries_per_kv
        if problem.num_queries_per_kv <= block_m
        else 1
    )
    total_num_q_blocks = problem.total_q // block_q + problem.num_seqs
    wave_size = 32 if arch == "gfx1250" else 64
    meta = _Attention2DLaunchMeta(
        grid=(int(problem.num_kv_heads), int(total_num_q_blocks), 1),
        block=(int(wave_size * num_warps), 1, 1),
    )
    _2D_LAUNCH_META[meta_key] = meta
    return meta


def _get_scalar_launcher(
    problem: UnifiedAttentionProblem,
    cache_key: Tuple,
) -> KernelLauncher:
    if cache_key in _SCALAR_LAUNCHERS:
        return _SCALAR_LAUNCHERS[cache_key]
    if cache_key not in _ATTN_CACHE:
        arch = _resolve_attention_arch()
        spec = UnifiedAttention2DSpec(problem=problem)
        artifact = compile_kernel(
            build_unified_attention_2d(spec), arch=arch, capture_ir_text=False
        )
        _ATTN_CACHE[cache_key] = (artifact.hsaco, artifact.kernel_name)
    hsaco, kname = _ATTN_CACHE[cache_key]
    launcher = KernelLauncher(
        hsaco=hsaco,
        kernel_name=kname,
        signature=_attn_signature(
            problem.dtype,
            include_bt_stride=False,
            kv_dtype=_kv_storage_dtype(problem),
        ),
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

    # Fill the paged-cache block count from the K tensor (k.shape[0]) so the
    # 2D dispatcher can switch to 64-bit KV addressing when the cache exceeds
    # the ~2 GiB i32-voffset cap. ``num_kv_blocks`` has exactly ONE consumer
    # (``_enable_i64_kv_addr``), and only caches >2 GiB need i64 -- so only pay
    # the (~30us, fields()-introspecting) ``dataclasses.replace`` when the cache
    # is actually that large. Small caches (the common case + every test shape)
    # keep ``num_kv_blocks=0`` -> the correct fast i32 path -> no per-call
    # replace, which otherwise dominates tiny-shape host latency.
    if problem.num_kv_blocks <= 0 and hasattr(k, "shape") and len(k.shape) >= 1:
        _eb = 1 if problem.use_fp8 else 2
        _blk_stride = (
            problem.block_size * problem.num_kv_heads * problem.head_size * _eb
        )
        if int(k.shape[0]) * _blk_stride > 0x8000_0000:
            problem = replace(problem, num_kv_blocks=int(k.shape[0]))

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
        # Graphed fast path FIRST: for the short-context regime the launch is
        # auto-graphed (heuristic: _recommend_graph_replay). The graph is looked
        # up by a cheap shape signature + tensor ids BEFORE any selector work, so
        # a replay skips supports + cache_key + the kernarg pack -- the host
        # overhead that otherwise dominates tiny-shape latency. Skipped when the
        # caller is already capturing the forward (they take precedence).
        if _enable_2d_graph_replay(problem) and not _torch_stream_capturing():
            graphed = _run_2d_graphed(
                problem,
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
                alibi_slopes=alibi_slopes,
                qq_bias=qq_bias,
                qq_bias_stride_0=qq_bias_stride_0,
                k_scale=k_scale,
                v_scale=v_scale,
                out_scale=out_scale,
                stream=stream,
            )
            if graphed is not _GRAPH_FALLBACK:
                return graphed
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
            # The dispatcher must launch with the same BLOCK_Q/threads the
            # kernel was built for. Cache that fixed metadata per kernel key so
            # repeated same-shape calls avoid selector math on the hot path.
            meta = _get_2d_launch_meta(problem, key)
            return launcher(
                vals,
                config=LaunchConfig(
                    grid=meta.grid,
                    block=meta.block,
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
        k_scale=k_scale,
        v_scale=v_scale,
        out_scale=out_scale,
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
    name: str = "rocke_unified_attention_2d_scalar"

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
        from ...helpers.spec import kernel_name_join

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
                "fp8kv": p.use_fp8,
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
    kv_dtype = FP8E4M3 if p.use_fp8 else dtype
    kv_vec_align = 8 if p.use_fp8 else 16
    kv_scalar_align = 1 if p.use_fp8 else 2
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = 64

    output = b.param(
        "output_ptr", PtrType(dtype, "global"), noalias=True, writeonly=True, align=16
    )
    abi = _declare_scalar_attn_params(b, dtype, kv_dtype=kv_dtype)
    query = abi["query"]
    key = abi["key"]
    value = abi["value"]
    sinks = abi["sink"]
    block_tables = abi["block_tables"]
    seq_lens = abi["seq_lens"]
    cu_q = abi["cu_q"]
    scale = abi["scale"]
    k_scale = abi["k_scale"]
    v_scale = abi["v_scale"]
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
    kv_head = _magic_div(b, q_head, p.num_queries_per_kv)

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
    # Both share the descriptor builders the 3D/reduce scalar kernels use.
    q_desc = _q_descriptor(p)
    kv_desc_elem = _paged_kv_descriptor(p)

    loop = b.scf_for_iter(
        b.const_i32(0),
        kv_len,
        b.const_i32(1),
        [("m", init_m), ("l", init_l), ("acc", init_acc)],
        iv_name="kpos",
    )
    with loop as (kpos, (m_val, l_val, acc_val)):
        block_idx, token_in_block = _magic_div_mod(b, kpos, p.block_size)
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

        # Vectorised QK dot product. Both Q[q_tok, q_head, :] and
        # K[physical, token_in_block, kv_head, :] are stride-1 along the
        # head_size axis, so we read them in ``vec8`` chunks (one
        # ds_read_b128 / buffer_load_b128 each). The inner accumulation
        # order ``score += q[d] * k[d]`` is preserved by walking the
        # vec lanes in order, so the result is bit-identical to the
        # prior per-element unroll. ``head_size`` is restricted to
        # {64, 128, 256} by :func:`supports_native_unified_attention`,
        # so it always divides cleanly by 8.
        score = zero_f
        q_off_base, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=b.const_i32(0))
        k_off_base = kv_desc_elem.offset(
            b,
            physical_block=physical,
            token_in_block=token_in_block,
            kv_head=kv_head,
            dim=b.const_i32(0),
        )
        VEC = 8
        for d8 in b.unroll(p.head_size // VEC):
            d_base = b.const_i32(d8 * VEC)
            qv_vec = b.global_load_vN(
                query, b.add(q_off_base, d_base), dtype, VEC, align=16
            )
            kv_vec = b.global_load_vN(
                key, b.add(k_off_base, d_base), kv_dtype, VEC, align=kv_vec_align
            )
            for i in range(VEC):
                qv = b.cast_to_f32(b.vec_extract(qv_vec, i))
                kv = b.vec_extract(kv_vec, i)
                kv_f = b.cvt_fp8_to_f32(kv) if p.use_fp8 else b.cast_to_f32(kv)
                if p.use_fp8:
                    kv_f = b.fmul(kv_f, k_scale)
                score = b.fadd(
                    score,
                    b.fmul(qv, kv_f),
                )
        # Defensive tail scalar fold for head_size % 8 != 0; in
        # production this loop is empty.
        for d in b.unroll((p.head_size // VEC) * VEC, p.head_size):
            d_v = b.const_i32(d)
            q_off, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=d_v)
            k_off = kv_desc_elem.offset(
                b,
                physical_block=physical,
                token_in_block=token_in_block,
                kv_head=kv_head,
                dim=d_v,
            )
            qv_s = b.cast_to_f32(b.global_load(query, q_off, dtype, align=2))
            kv_raw = b.global_load(key, k_off, kv_dtype, align=kv_scalar_align)
            kv_s = b.cvt_fp8_to_f32(kv_raw) if p.use_fp8 else b.cast_to_f32(kv_raw)
            if p.use_fp8:
                kv_s = b.fmul(kv_s, k_scale)
            score = b.fadd(score, b.fmul(qv_s, kv_s))

        score = b.fmul(b.fmul(score, scale), rcp_ln2)
        if p.softcap > 0:
            score = b.fmul(_apply_softcap(b, score, softcap), rcp_ln2)

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
        vv_raw = b.global_load(value, v_off, kv_dtype, align=kv_scalar_align)
        vv = b.cvt_fp8_to_f32(vv_raw) if p.use_fp8 else b.cast_to_f32(vv_raw)
        if p.use_fp8:
            vv = b.fmul(vv, v_scale)
        new_acc = b.fadd(b.fmul(acc_val, alpha), b.fmul(prob, vv))
        b.scf_yield(new_m, new_l, new_acc)

    out_val = b.fmul(loop.results[2], b.rcp(loop.results[1]))
    out_cast = b.cast_f32_to(out_val, dtype)
    out_off, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=dim)
    valid = b.land(active, b.cmp_lt(dim, b.const_i32(p.head_size)))
    with b.scf_if(valid):
        b.global_store(output, out_off, out_cast, align=2)
    return b.kernel


@dataclass(frozen=True)
class UnifiedAttention3DSpec(UnifiedAttention2DSpec):
    name: str = "rocke_unified_attention_3d_scalar"
    num_segments: int = 8

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

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
    abi = _declare_scalar_attn_params(b, dtype)
    query = abi["query"]
    key = abi["key"]
    value = abi["value"]
    block_tables = abi["block_tables"]
    seq_lens = abi["seq_lens"]
    cu_q = abi["cu_q"]
    scale = abi["scale"]
    _softcap = b.param("softcap", F32)
    num_seqs = b.param("num_seqs", I32)

    q_tok = b.block_id_x()
    q_head = b.block_id_y()
    zd = b.block_id_z()
    segm_idx, dim = _magic_div_mod(b, zd, p.head_size)
    tid = b.thread_id_x()
    active = b.cmp_eq(tid, b.const_i32(0))

    seq_idx = _emit_find_seq_idx_scan(b, cu_q, q_tok, num_seqs)
    cu_start = b.global_load_i32(cu_q, seq_idx)
    cu_stop = b.global_load_i32(cu_q, b.add(seq_idx, b.const_i32(1)))
    q_len = b.sub(cu_stop, cu_start)
    query_pos = b.sub(q_tok, cu_start)
    kv_len = b.global_load_i32(seq_lens, seq_idx)
    context_len = b.sub(kv_len, q_len)
    kv_head = _magic_div(b, q_head, p.num_queries_per_kv)
    tiles_per_segment = _magic_div(
        b,
        b.add(kv_len, b.const_i32(spec.num_segments * p.block_size - 1)),
        spec.num_segments * p.block_size,
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
    name: str = "rocke_unified_attention_reduce_scalar"

    @property
    def dtype_ir(self) -> Type:
        return F16 if self.problem.dtype == "fp16" else BF16

    def kernel_name(self) -> str:
        from ...helpers.spec import kernel_name_join

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
    """Find ``seq_idx`` for the given Q token via a binary search.

    P74: replaces the historical linear scan (``O(num_seqs)``) with
    the shared ``binary_search_seq_idx`` helper in per-token mode
    (``cu_q[s] <= q_tok < cu_q[s+1]``). Cosmetic on the scalar
    oracle but matches the tiled kernels' shape and saves
    ``O(num_seqs - log2(num_seqs))`` linear-scan iterations on
    high-batch decode workloads.
    """
    from ...helpers.attention import binary_search_seq_idx

    return binary_search_seq_idx(
        b,
        cu_q,
        q_tok,
        num_seqs,
        block_q=1,  # unused in per_token mode
        iterations=32,  # bounded scf.for trip count; 32 covers num_seqs <= 2^32
        per_token=True,
    )


def _declare_scalar_attn_params(
    b: IRBuilder, dtype_ir: Type, *, kv_dtype: Optional[Type] = None
) -> dict:
    """Declare the shared scalar-attention ABI prefix (Q/K/V + aux + scales).

    The 2D and 3D scalar reference kernels declare an identical leading run
    of params -- ``query/key/value``, the ``sink/block_tables/seq_lens/alibi/
    qq_bias/cu_q`` aux pointers, then the ``scale/k_scale/v_scale`` f32
    scalars -- in exactly this order. Param declaration order is load-bearing
    (it fixes the kernel arg layout / ABI), so this helper emits the same
    ``b.param`` calls in the same order, returning the SSA values by name. The
    builders append their own kernel-specific tail params (``out_scale``,
    ``softcap``, ``num_seqs``, ...) after calling this, preserving the overall
    ABI byte-for-byte.

    The ``sink`` and ``alibi``/``qq_bias`` pointers are unused by the scalar
    oracle bodies (they only model causal/sliding masking + softcap), but they
    are part of the AITER ABI and must occupy their slots.
    """
    kv_dtype = kv_dtype or dtype_ir
    query = b.param(
        "query_ptr", PtrType(dtype_ir, "global"), noalias=True, readonly=True, align=16
    )
    key = b.param(
        "key_cache_ptr",
        PtrType(kv_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    value = b.param(
        "value_cache_ptr",
        PtrType(kv_dtype, "global"),
        noalias=True,
        readonly=True,
        align=16,
    )
    sink = b.param("sink_ptr", PtrType(dtype_ir, "global"), readonly=True, align=16)
    block_tables = b.param(
        "block_tables_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    seq_lens = b.param("seq_lens_ptr", PtrType(I32, "global"), readonly=True, align=4)
    alibi = b.param("alibi_slopes_ptr", PtrType(F32, "global"), readonly=True, align=4)
    qq_bias = b.param("qq_bias_ptr", PtrType(F32, "global"), readonly=True, align=4)
    cu_q = b.param(
        "query_start_len_ptr", PtrType(I32, "global"), readonly=True, align=4
    )
    scale = b.param("scale", F32)
    k_scale = b.param("k_scale", F32)
    v_scale = b.param("v_scale", F32)
    return {
        "query": query,
        "key": key,
        "value": value,
        "sink": sink,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
        "alibi": alibi,
        "qq_bias": qq_bias,
        "cu_q": cu_q,
        "scale": scale,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


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
    """Per-(query_token, query_head) QK dot product for the scalar kernels.

    The dot product sums ``head_size`` half-precision element pairs into
    one f32 score. Both Q[q_tok, q_head, :] and
    K[physical_block, token_in_block, kv_head, :] are contiguous along
    the head_size dimension (the innermost stride-1 axis of their
    respective descriptors), so the kernel reads them in
    ``vec8`` chunks via :meth:`IRBuilder.global_load_vN` instead of
    one 16-bit ``b.global_load`` per element. The inner accumulation
    order is preserved (``score += q[d] * k[d]`` for ``d`` increasing
    monotonically), so the result is bit-identical to the prior
    per-element form.

    Head sizes that aren't a multiple of 8 fall back to a tail loop;
    in production the scalar kernels are only built for
    ``head_size in {64, 128, 256}`` (see
    :func:`supports_native_unified_attention`), all multiples of 8.
    """
    score = b.const_f32(0.0)
    physical, token_in_block = _physical_block_and_token(
        b, p, block_tables, seq_idx, kpos
    )
    q_desc = _q_descriptor(p)
    kv_desc = _paged_kv_descriptor(p)
    q_off_base, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=b.const_i32(0))
    k_off_base = kv_desc.offset(
        b,
        physical_block=physical,
        token_in_block=token_in_block,
        kv_head=kv_head,
        dim=b.const_i32(0),
    )
    VEC = 8
    n_vec = p.head_size // VEC
    for d8 in b.unroll(n_vec):
        d_base = b.const_i32(d8 * VEC)
        qv = b.global_load_vN(query, b.add(q_off_base, d_base), dtype, VEC, align=16)
        kv = b.global_load_vN(key, b.add(k_off_base, d_base), dtype, VEC, align=16)
        for i in range(VEC):
            score = b.fadd(
                score,
                b.fmul(
                    b.cast_to_f32(b.vec_extract(qv, i)),
                    b.cast_to_f32(b.vec_extract(kv, i)),
                ),
            )
    # Tail scalar fold for head_size values not a multiple of VEC (defensive;
    # the supported set in supports_native_unified_attention is {64,128,256}
    # — all multiples of 8 — so this loop is empty for any compiled kernel).
    for d in b.unroll(n_vec * VEC, p.head_size):
        d_v = b.const_i32(d)
        q_off, _ = q_desc.offset(b, token=q_tok, head=q_head, dim=d_v)
        k_off = kv_desc.offset(
            b,
            physical_block=physical,
            token_in_block=token_in_block,
            kv_head=kv_head,
            dim=d_v,
        )
        qv_s = b.cast_to_f32(b.global_load(query, q_off, dtype, align=2))
        kv_s = b.cast_to_f32(b.global_load(key, k_off, dtype, align=2))
        score = b.fadd(score, b.fmul(qv_s, kv_s))
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
    block_idx, token_in_block = _magic_div_mod(b, kpos, p.block_size)
    max_blocks = (p.max_seqlen_k + p.block_size - 1) // p.block_size
    physical = b.global_load_i32(
        block_tables, b.add(b.mul(seq_idx, b.const_i32(max_blocks)), block_idx)
    )
    return physical, token_in_block
