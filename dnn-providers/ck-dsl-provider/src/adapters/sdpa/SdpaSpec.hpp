// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

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
/// into the cache key alongside the problem's shape fields); the M1 path
/// is FP16-only with top-left causal or no mask.
struct SdpaSpec {
    SdpaProblem problem;
    std::string name{"ck_dsl_fmha_fwd_mfma"};
    std::string dtype{"f16"};       // codegen-relevant
    std::string mask_mode{"none"};  // "none" | "causal"; codegen-relevant

    // Opt-in forward-training stats (LSE) output. When true the kernel
    // appends one f32 ``LSE_out`` pointer at ABI position 16 (after the
    // 15 base args) and writes natural-log LSE in head-major [B, Hq, Sq]
    // layout (consumed by the backward). Codegen-relevant: stats-on and
    // stats-off emit distinct kernels, so this is folded into the cache
    // signature.
    bool generate_stats{false};
};

}  // namespace ck_dsl_provider
