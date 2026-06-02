// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace ck_dsl_provider {

/// Performance knobs the provider selects for the unified paged/varlen
/// tiled-2D attention kernel (``UnifiedAttention2DTiledSpec`` in
/// ``instances/gfx950/attention_tiled_2d.py``).
///
/// This is the interface the scorer-driven selection produces and the
/// Python compile path consumes: the chosen combo lands here and is
/// later marshalled into ``build_unified_attention_2d_tiled`` keyword
/// arguments. It is a pure value type (no owning pointers, no
/// allocation) so it can be enumerated, copied, and compared freely.
///
/// The continuous performance axes the dispatcher actually scores are
/// ``num_warps``, ``block_m_per_warp`` (together forming BLOCK_M), and
/// ``tile_size``. The remaining boolean variant flags either steer the
/// MFMA atom / schedule (which DO affect the kernel-key mapping and
/// therefore the score, e.g. ``use_mfma_32x32`` and the schedule flags
/// that pick the pipeline string) or are problem-driven variant lanes
/// (sinks / sliding-window / fp8) that Phase 2 sets from the graph. They
/// are carried here with safe defaults so every emitted combo is
/// complete and buildable.
///
/// There is deliberately NO ``compile_backend`` member: the kernel's
/// ``UnifiedAttention2DTiledSpec`` has no such field; the JIT backend is
/// chosen later by ``compile_kernel(arch=...)``.
struct SdpaPerfKnobs {
    // --- Continuous scored axes --------------------------------------

    /// Number of wave64 warps per CTA. Valid: {1, 2, 4, 8}. Each warp
    /// owns a 16-row slice of BLOCK_M; no cross-warp reduction.
    std::int32_t num_warps{1};

    /// Per-warp M-dimension tile size. Valid: {16, 32}. With 32 each
    /// warp stacks two MFMA-M=16 atoms; 32 requires num_warps in
    /// {1, 2, 4} (CTA thread cap).
    std::int32_t block_m_per_warp{16};

    /// KV tokens consumed per outer-loop iter (``T``). A positive
    /// multiple of ``block_size``. 0 means "unset" -> the kernel
    /// defaults T to block_size; the enumerator always emits explicit
    /// positive values so candidates are deterministic.
    std::int32_t tile_size{0};

    // --- Occupancy hint ----------------------------------------------

    /// AMDGPU ``amdgpu-waves-per-eu`` occupancy hint. 0 means "unset"
    /// (let the LLVM heuristic decide); >0 forces tighter VGPR
    /// allocation for higher occupancy.
    std::int32_t waves_per_eu{0};

    // --- Curated MFMA-atom / schedule flags (affect key mapping) -----

    /// Use the 32x32x16 MFMA geometry instead of the default 16x16x32.
    /// Requires block_m_per_warp == 32. Drives the k0/k1 mapping.
    bool use_mfma_32x32{false};

    /// Transposed QK orientation for the 32x32 path (requires
    /// use_mfma_32x32). Part of the schedule-flag -> pipeline mapping.
    bool use_transposed_qk_32x32{false};

    /// Keep softmax P in registers across the PV MFMA instead of the
    /// P_lds round-trip (16x16x32 path only; bf16-only in the kernel).
    /// Part of the schedule-flag -> pipeline mapping.
    bool use_register_pv{false};

    /// Issue the V async copy early (after the iter-start K drain),
    /// giving V the whole QK+softmax window to arrive. Part of the
    /// schedule-flag -> pipeline mapping.
    bool use_early_v_schedule{false};

    /// Fast paged-KV byte descriptor for the hot R4 geometry
    /// (bf16 h64kv8 HD=64 BS=32 T=64 num_warps=4). Heavily constrained
    /// in the kernel; default off and not enumerated.
    bool use_fast_paged_kv_desc{false};

    // --- Problem-driven variant lanes (Phase 2 sets from the graph) --

    /// Attention sinks. Driven by the graph's Sink_token tensor.
    bool use_sinks{false};

    /// Sliding-window span (tokens). 0 == no window. Driven by the
    /// graph's left_bound. Carried here so the analytic policy and the
    /// kernel-key mapping see it consistently.
    std::int32_t sliding_window{0};

    /// Derived launch BLOCK_M = num_warps * block_m_per_warp. This is
    /// the value mapped to ``tile_shape.m0`` and used to recompute the
    /// launch grid. (Distinct from the kernel's GQA pre-check row count
    /// ``16 * num_warps`` used inside ``supports_tiled_2d``.)
    [[nodiscard]] std::int32_t block_m() const {
        return num_warps * block_m_per_warp;
    }
};

}  // namespace ck_dsl_provider
