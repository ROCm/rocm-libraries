// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

#include "SdpaPerfKnobs.hpp"

namespace ck_dsl_provider {

/// C++ mirror of the CK DSL FMHA-forward problem description.
///
/// The fields split into two groups with distinct cache semantics:
///
///   * **Codegen inputs** (B, Hq, Hkv, Sq, Skv, D plus the spec's dtype
///     and mask_mode) determine which kernel binary and launch grid the
///     DSL emits. They ARE folded into the JIT cache signature
///     (``GraphSignature::computeForSpec``): any change must produce a
///     distinct cached module.
///
///   * **Launch-time scalars** (the eight stride_* fields and
///     scale_log2) are passed to the kernel as runtime arguments. They
///     are carried on the problem for convenience -- the plan builder
///     reads them straight off the spec when constructing the plan --
///     but are DELIBERATELY excluded from the cache signature: the
///     compiled kernel and its grid are identical regardless of stride
///     or scale, so folding them would thrash the cache (a different
///     stride or scale would force a redundant recompile of a
///     byte-identical kernel).
///
/// Tensor layout convention (rank-4): [B, H, S, D]. The token stride is
/// the sequence-dim stride (strides[2]); the head stride is the
/// head-dim stride (strides[1]).
///
/// Supported memory layout is BSHD-compatible: the head-dim (last axis)
/// is unit-stride, and for batch>1 the batch stride equals seqlen *
/// sequence-stride (heads interleaved within each sequence position).
/// The kernel has no batch stride and folds batch as
/// batch_idx*seqlen*stride_token; a contiguous BHSD tensor with B>1 and
/// H>1 is NOT supported and is rejected by the adapter.
struct SdpaProblem {
    std::int32_t B{0};    // batch
    std::int32_t Hq{0};   // num_query_heads
    std::int32_t Hkv{0};  // num_kv_heads
    std::int32_t Sq{0};   // seqlen_q
    std::int32_t Skv{0};  // seqlen_k
    std::int32_t D{0};    // head_size (Dqk == Dv enforced)

    // Launch-time scalars (NOT folded into the cache signature; the
    // kernel binary + grid are identical regardless of stride/scale).
    std::int32_t stride_q_token{0};
    std::int32_t stride_q_head{0};
    std::int32_t stride_k_token{0};
    std::int32_t stride_k_head{0};
    std::int32_t stride_v_token{0};
    std::int32_t stride_v_head{0};
    std::int32_t stride_o_token{0};
    std::int32_t stride_o_head{0};
    float scale_log2{0.0f};
};

/// C++ mirror of the CK DSL FMHA-forward spec.
///
/// ``name`` keeps a provider-specific prefix for kernel identification
/// in profiles. ``dtype`` and ``mask_mode`` are codegen-relevant (folded
/// into the cache key alongside the problem's shape fields). ``dtype``
/// holds an unenumerated type string -- the supported values are "f16"
/// and "bf16" (the unified paged kernel accepts both); other types are
/// declined by the capability gate.
struct SdpaSpec {
    SdpaProblem problem;
    std::string name{"ck_dsl_fmha_fwd_mfma"};
    std::string dtype{"f16"};       // "f16" | "bf16"; codegen-relevant
    std::string mask_mode{"none"};  // "none" | "causal"; codegen-relevant

    // Opt-in forward-training stats (LSE) output. When true the kernel
    // appends one f32 ``LSE_out`` pointer at ABI position 16 (after the
    // 15 base args) and writes natural-log LSE in head-major [B, Hq, Sq]
    // layout (consumed by the backward). Codegen-relevant: stats-on and
    // stats-off emit distinct kernels, so this is folded into the cache
    // signature.
    bool generate_stats{false};

    // --- Unified paged/varlen problem lanes --------------------------
    // These describe which marshalling path the spec takes and the KV
    // layout the kernel sees. They are codegen-relevant: a paged build,
    // a varlen build, and a windowed build emit distinct kernels and
    // grids, so all are folded into the cache signature.

    /// Paged KV layout in effect -- either a real paged graph
    /// (Page_table_K/V present) or the dense-degenerate one-block-per-
    /// sequence layout the unified kernel always runs.
    bool is_paged{false};

    /// Paged KV block size (tokens per block). One of {16, 32, 64}.
    /// 0 means "unset" -- the dense-degenerate default is chosen later
    /// during marshalling.
    std::int32_t block_size{0};

    /// Variable-length sequence path (cu_seqlens / seqused KV). When
    /// false the fixed-length dense layout synthesizes trivial
    /// cu_seqlens during marshalling.
    bool is_varlen{false};

    /// Sliding-window span (the graph's ``left_bound``), in tokens.
    /// 0 means no window (full causal context).
    std::int32_t sliding_window{0};

    /// Attention sinks in effect (the graph's Sink_token tensor).
    bool use_sinks{false};

    // --- Chosen perf config ------------------------------------------

    /// Performance knobs the scorer-driven selection picks for this
    /// problem (Phase 2b writes this). Defaults to the Phase-1 POD
    /// defaults so an unscored spec is still complete. Codegen-relevant:
    /// distinct scored configs must cache distinctly, so the knob fields
    /// are folded into the cache signature.
    SdpaPerfKnobs knobs{};
};

}  // namespace ck_dsl_provider
