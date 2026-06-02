// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ck_tile/dispatcher/fmha_kernel_key.hpp>
#include <ck_tile/dispatcher/fmha_problem.hpp>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "SdpaPerfKnobs.hpp"

namespace ck_dsl_provider {

/// Problem-shape inputs the candidate enumerator and scoring mapping
/// need. These mirror the fields of ``SdpaSpec``/``SdpaProblem`` plus
/// the variant flags the kernel-key signature and the validity gate
/// read. Pure value type; supplied by the plan builder in Phase 2 and
/// by the tests directly in Phase 1.
///
/// ``dtype`` follows the kernel's spelling ("fp16" / "bf16") -- note
/// the kernel uses "fp16" whereas ``SdpaSpec::dtype`` uses "f16"; the
/// caller normalises before populating this struct.
struct SdpaSelectionProblem {
    std::int32_t batch{1};
    std::int32_t num_query_heads{0};  // Hq
    std::int32_t num_kv_heads{0};     // Hkv
    std::int32_t seqlen_q{0};         // Sq
    std::int32_t seqlen_k{0};         // Skv
    std::int32_t head_size{0};        // D (Dqk == Dv)
    std::int32_t block_size{0};       // paged KV block size {16,32,64}
    std::string dtype{"fp16"};        // "fp16" | "bf16"

    // Variant flags that the kernel-key signature mirrors and the
    // analytic policy reads. Set by Phase 2 from the graph.
    bool use_sinks{false};
    std::int32_t sliding_window{0};  // 0 == none; >0 == left_bound span
    bool use_paged_kv{false};        // real paged or degenerate paged
    bool skip_min_seqlen_q{false};
    int mask_type{1};  // fmha mask_enum int (1 == top-left causal)
    int bias_type{0};  // fmha bias_enum int (0 == no_bias)

    /// GQA query-heads-per-kv-head. The kernel requires Hq % Hkv == 0;
    /// the caller guarantees that before constructing this struct.
    [[nodiscard]] std::int32_t num_queries_per_kv() const {
        return num_kv_heads > 0 ? num_query_heads / num_kv_heads : 0;
    }
};

/// Result of ``supports_tiled_2d`` C++ mirror: a verdict plus a
/// structured reason (empty when supported).
struct SupportsResult {
    bool supported{false};
    std::string reason;
};

/// C++ mirror of P's ``supports_tiled_2d`` validity gate
/// (``instances/gfx950/attention_tiled_2d.py:588-690``). Ported exactly
/// so the enumerator only emits buildable combos and so Phase 3 can
/// cross-validate the mirror against the real Python predicate.
///
/// NOTE on the two BLOCK_M forms: this predicate uses the kernel's
/// base-row ``block_m = 16 * num_warps`` for the GQA divisibility check
/// (matching the Python at :634). The launch BLOCK_M
/// (``num_warps * block_m_per_warp``) is a DIFFERENT quantity used by
/// the kernel-key mapping and grid recompute; do not conflate them.
[[nodiscard]] SupportsResult supportsTiled2d(const SdpaSelectionProblem& problem,
                                             const SdpaPerfKnobs& knobs);

/// Enumerate kernel-knob combos for the problem, pre-filtered by
/// ``supportsTiled2d``. Only buildable combos are returned. Each
/// returned combo is complete (the OI-B default-fill table is applied to
/// the non-enumerated variant flags) and carries the problem-driven
/// variant lanes (sinks / sliding-window) copied from ``problem``.
///
/// The enumeration order is deterministic (nested loops over num_warps,
/// block_m_per_warp, tile_size, then the curated atom/schedule flag
/// set), which fixes the tie-break order used by ``selectArgmax``.
[[nodiscard]] std::vector<SdpaPerfKnobs> enumerateCandidates(const SdpaSelectionProblem& problem);

/// Deterministic forward mapping from a knob combo to the dispatcher's
/// ``FmhaKernelKey``. The scored fields are set per §2.6:
///   * tile_shape.m0 = block_m() = num_warps * block_m_per_warp
///   * tile_shape.n0 = tile_size
///   * tile_shape.k0 / k1 from the MFMA-atom flag
///   * pipeline string from the schedule flags
///   * pad_* default-filled (true)
///   * signature.* matched to the problem
/// Non-scored fields (wave_shape, warp_tile_shape, alignments,
/// block_per_cu, selection_rank, gfx_arch) are left at sensible
/// defaults; they do not affect the score.
[[nodiscard]] ck_tile::dispatcher::FmhaKernelKey knobsToKernelKey(
    const SdpaSelectionProblem& problem, const SdpaPerfKnobs& knobs);

/// Build the dispatcher ``FmhaProblem`` from the selection problem so
/// ``signature.*`` is consistent across all scored candidates. Uses
/// ``FmhaProblemBuilder`` with gfx_arch == "gfx950".
[[nodiscard]] ck_tile::dispatcher::FmhaProblem problemToFmhaProblem(
    const SdpaSelectionProblem& problem);

/// Map a knob combo's schedule flags to the dispatcher pipeline string.
/// Returns one of the valid pipeline names recognised by the heuristic
/// feature encoder: "qr", "qr_async", "qr_async_trload",
/// "qr_async_trload_v3", "qr_pagedkv".
[[nodiscard]] std::string pipelineForKnobs(const SdpaSelectionProblem& problem,
                                           const SdpaPerfKnobs& knobs);

/// Argmax selection over a set of candidate knob combos using an
/// injected score callable. The callable receives the candidate's
/// ``FmhaKernelKey`` (built via ``knobsToKernelKey``) and returns a
/// scalar score (higher is better). The highest-scoring combo is
/// returned; ties are broken by enumeration order (the first combo with
/// the maximal score wins), which is stable because ``enumerateCandidates``
/// is deterministic.
///
/// ``candidates`` must be non-empty (the caller guarantees at least one
/// buildable combo, or falls back to the analytic pick).
[[nodiscard]] SdpaPerfKnobs selectArgmax(
    const SdpaSelectionProblem& problem, const std::vector<SdpaPerfKnobs>& candidates,
    const std::function<double(const ck_tile::dispatcher::FmhaKernelKey&)>& score);

/// Analytic fallback pick over the SAME enumerated+filtered combo set,
/// approximating P's ``production_dispatch`` / ``_select_2d_*`` policy
/// (``instances/common/attention_unified.py``). This is the
/// MODEL-LOAD-FAILURE fallback -- it is NOT a trivial first-fit; it
/// scores each candidate by an explicit analytic ordering and returns
/// the best. ``candidates`` must be non-empty.
[[nodiscard]] SdpaPerfKnobs selectAnalyticFallback(const SdpaSelectionProblem& problem,
                                                   const std::vector<SdpaPerfKnobs>& candidates);

}  // namespace ck_dsl_provider
