// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2FwdPlanBuilder: IPlanBuilder for Flash-Attention 2 V7 forward pass.
// Wraps our V7 HIP kernel (rocWMMA MFMA + causal tile skip) as a hipDNN SDPA engine.
// Mirrors the asm_sdpa_engine::SdpaFwdPlanBuilder pattern.

#pragma once

#include <cstdint>
#include <string>

namespace hip_flash2_engine
{

// =============================================================================
// Flash2FwdParams — extracted from the hipDNN graph at buildPlan() time.
// Holds everything execute() needs to dispatch the kernel.
// =============================================================================
struct Flash2FwdParams
{
    // Tensor UIDs (used to look up device pointers in the variant pack)
    int64_t qUid = 0;
    int64_t kUid = 0;
    int64_t vUid = 0;
    int64_t oUid = 0;

    // Attention geometry — BHSD layout: [B, H, S, D]
    int batch       = 1;
    int num_heads_q = 32;
    int num_heads_k = 32;   // GQA: num_heads_q / num_heads_k = gqa_ratio
    int seq_len_q   = 2048;
    int seq_len_kv  = 2048;
    int head_dim    = 128;  // head_dim_qk (== head_dim_v for our kernel)

    // Attention scale (0 → use 1/sqrt(head_dim) at runtime)
    float attn_scale = 0.0f;

    // Causal mask flag
    bool causal = false;

    // Strides (in elements, not bytes) — BHSD: dim0=B, dim1=H, dim2=S, dim3=D
    int64_t q_stride_batch = 0;
    int64_t q_stride_head  = 0;
    int64_t q_stride_seq   = 0;
    int64_t k_stride_batch = 0;
    int64_t k_stride_head  = 0;
    int64_t k_stride_seq   = 0;
    int64_t v_stride_batch = 0;
    int64_t v_stride_head  = 0;
    int64_t v_stride_seq   = 0;
    int64_t o_stride_batch = 0;
    int64_t o_stride_head  = 0;
    int64_t o_stride_seq   = 0;

    // Architecture string determined at buildPlan() time
    std::string archString;
};

// =============================================================================
// Dispatch heuristic: Flash2 is profitable for prefill shapes.
// Matches UseFlash2ForROCm() from the original FlashInfer benchmark.
// =============================================================================
inline bool useFlash2ForShape(int seq_len_q, int seq_len_kv)
{
    // Decode (seq_q == 1): Flash2 brings no benefit, use batched GEMM instead
    if(seq_len_q <= 1)
        return false;
    const uint32_t cta_q_blocks = (static_cast<uint32_t>(seq_len_q) + 63u) / 64u;
    return (static_cast<uint64_t>(seq_len_q) * static_cast<uint64_t>(seq_len_kv))
           > (static_cast<uint64_t>(cta_q_blocks) * 6000u);
}

} // namespace hip_flash2_engine
