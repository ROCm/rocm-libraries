// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

namespace ck_dsl_provider {

/// C++ mirror of the CK DSL FMHA-backward problem description.
///
/// The fields split into two groups with distinct cache semantics:
///
///   * **Codegen inputs** (B, Hq, Hkv, Sq, Skv, D plus the spec's dtype
///     and mask_mode) determine which kernel binaries and launch grids
///     the DSL emits. They ARE folded into the JIT cache signature
///     (``GraphSignature::computeForSpec`` for the bwd kernel and
///     ``GraphSignature::computeForSdpaLsePrep`` for the LSE-prep
///     kernel): any change must produce a distinct cached module.
///
///   * **Launch-time scalars** (the eleven stride_* fields and the two
///     scale_* values) are passed to the kernels as runtime arguments.
///     They are carried on the problem for convenience -- the plan
///     builder reads them straight off the spec when constructing the
///     plan -- but are DELIBERATELY excluded from the cache signature:
///     the compiled kernels and their grids are identical regardless of
///     stride or scale, so folding them would thrash the cache (a
///     different stride or scale would force a redundant recompile of a
///     byte-identical kernel).
///
/// Tensor layout convention (rank-4): [B, H, S, D]. The token stride is
/// the sequence-dim stride (strides[2]); the head stride is the
/// head-dim stride (strides[1]). The gradient tensors reuse the matching
/// input head stride, so only their token strides are recorded here.
///
/// Supported memory layout is BSHD-compatible: the head-dim (last axis)
/// is unit-stride, and for batch>1 the batch stride equals seqlen *
/// sequence-stride (heads interleaved within each sequence position).
/// The kernels have no batch stride and fold batch as
/// batch_idx*seqlen*stride_token; a contiguous BHSD tensor with B>1 and
/// H>1 is NOT supported and is rejected by the adapter. This is the same
/// BSHD contract the forward vertical enforces.
struct SdpaBwdProblem {
    std::int32_t B{0};    // batch
    std::int32_t Hq{0};   // num_query_heads
    std::int32_t Hkv{0};  // num_kv_heads
    std::int32_t Sq{0};   // seqlen_q
    std::int32_t Skv{0};  // seqlen_k
    std::int32_t D{0};    // head_size (Dqk == Dv enforced)

    // Input strides (token = sequence-dim stride; head = head-dim
    // stride). Launch-time scalars (NOT folded into the cache signature).
    std::int32_t stride_q_token{0};
    std::int32_t stride_q_head{0};
    std::int32_t stride_k_token{0};
    std::int32_t stride_k_head{0};
    std::int32_t stride_v_token{0};
    std::int32_t stride_v_head{0};
    std::int32_t stride_do_token{0};
    std::int32_t stride_do_head{0};

    // Gradient token strides. The head stride for each gradient is
    // implicitly the matching input head stride (the kernel reuses it),
    // which the adapter validates before recording these.
    std::int32_t stride_dq_token{0};
    std::int32_t stride_dk_token{0};
    std::int32_t stride_dv_token{0};

    // Attention scale, in two forms the kernels consume. ``scale_log2``
    // is attn_scale * log2(e) (the kernel computes exp2 in the softmax);
    // ``scale_inv`` is the raw attn_scale (== 1/sqrt(D) by default).
    float scale_log2{0.0f};
    float scale_inv{0.0f};
};

/// C++ mirror of the CK DSL FMHA-backward spec.
///
/// ``name`` keeps a provider-specific prefix for kernel identification
/// in profiles. ``dtype`` and ``mask_mode`` are codegen-relevant (folded
/// into the cache key alongside the problem's shape fields); the bwd
/// path is FP16-only with top-left causal or no mask.
struct SdpaBwdSpec {
    SdpaBwdProblem problem;
    std::string name{"ck_dsl_fmha_bwd"};
    std::string dtype{"f16"};       // codegen-relevant
    std::string mask_mode{"none"};  // "none" | "causal"; codegen-relevant
};

}  // namespace ck_dsl_provider
