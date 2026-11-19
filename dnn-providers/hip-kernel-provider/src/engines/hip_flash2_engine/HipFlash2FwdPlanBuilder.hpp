// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2FwdPlanBuilder: IPlanBuilder for Flash-Attention 2 V7 forward pass.
// Wraps our V7 HIP kernel (rocWMMA MFMA + causal tile skip) as a hipDNN SDPA
// engine. Mirrors the asm_sdpa_engine::SdpaFwdPlanBuilder pattern.

#pragma once

#include <cstddef>

#include <cstdint>
#include <string>

namespace hip_flash2_engine
{

// =============================================================================
// Flash2FwdParams -- extracted from the hipDNN graph at buildPlan() time.
// Holds everything execute() needs to dispatch the kernel.
// =============================================================================
struct Flash2FwdParams
{
    // Tensor UIDs (used to look up device pointers in the variant pack)
    int64_t qUid = 0;
    int64_t kUid = 0;
    int64_t vUid = 0;
    int64_t oUid = 0;

    // Attention geometry -- BHSD layout: [B, H, S, D]
    int batch = 1;
    int numHeadsQ = 32;
    int numHeadsK = 32; // GQA: numHeadsQ / numHeadsK = gqa_ratio
    int seqLenQ = 2048;
    int seqLenKv = 2048;
    int headDim = 128; // head_dim_qk (== head_dim_v for our kernel)

    // Softmax scale: attn_scale_value, 1.0 (no scaling) when the graph leaves it unset
    float attnScale = 1.0f;

    // Causal mask flag
    bool causal = false;

    // Strides (in elements, not bytes) -- BHSD: dim0=B, dim1=H, dim2=S, dim3=D
    int64_t qStrideBatch = 0;
    int64_t qStrideHead = 0;
    int64_t qStrideSeq = 0;
    int64_t kStrideBatch = 0;
    int64_t kStrideHead = 0;
    int64_t kStrideSeq = 0;
    int64_t vStrideBatch = 0;
    int64_t vStrideHead = 0;
    int64_t vStrideSeq = 0;
    int64_t oStrideBatch = 0;
    int64_t oStrideHead = 0;
    int64_t oStrideSeq = 0;

    // ---- Kernel-variant selection (filled by selectFlash2Config) ------------
    // The engine ships several tilings of the same S-transpose kernel; which is
    // fastest depends on how many CTAs the shape produces relative to the CU
    // count, so launch geometry is a per-plan property, not a constant.
    // blockDim/qPerCta MUST match how the selected .co was compiled.
    std::string variantTag; // "" = legacy single-kernel object
    unsigned int blockDim = 64; // threads per CTA for the selected variant
    unsigned int qPerCta = 64; // query rows covered by one CTA

    // ---- Split-K (flash-decoding) ------------------------------------------
    // Selected for grid-starved shapes. Execution is not yet plumbed through
    // execute(); the fields record the decision so the follow-up does not have
    // to re-derive it.
    int splitK = 1; ///< always 1 today: selection computes a split factor but
        ///< execute() is not wired for it, so buildPlan forces 1.
    size_t workspaceBytes = 0; ///< 0 today; sized only when split-K executes.

    // Architecture string determined at buildPlan() time
    std::string archString;
};

// =============================================================================
// Dispatch heuristic: Flash2 is profitable for prefill shapes.
// Matches UseFlash2ForROCm() from the original FlashInfer benchmark.
// =============================================================================
inline bool useFlash2ForShape(int seqLenQ, int seqLenKv)
{
    // Decode (seq_q == 1): Flash2 brings no benefit, use batched GEMM instead
    if(seqLenQ <= 1)
    {
        return false;
    }
    const uint32_t ctaQBlocks = (static_cast<uint32_t>(seqLenQ) + 63u) / 64u;
    return (static_cast<uint64_t>(seqLenQ) * static_cast<uint64_t>(seqLenKv))
           > (static_cast<uint64_t>(ctaQBlocks) * 6000u);
}

// =============================================================================
// KV sequence-length ceiling.
// =============================================================================
//
// Both kernels count KV tiles in signed int: HipFlash2FwdPlan.hip forms
// (seq_len_kv + 63) / 64 and HipFlash2FwdPlanVariant.hip forms
// (seq_len_kv + F2_BK - 1) / F2_BK, where F2_BK = F2_KG * 16 and F2_KG is 2 or
// 4, so 64 is the largest KV tile this engine dispatches. One full tile of
// headroom below INT32_MAX keeps both additions in range. isApplicable
// declines anything above this rather than dispatch a kernel whose KV loop
// bound has already overflowed.
constexpr int64_t K_FLASH2_MAX_KV_TILE = 64;
constexpr int64_t K_FLASH2_MAX_SEQ_LEN_KV = static_cast<int64_t>(INT32_MAX) - K_FLASH2_MAX_KV_TILE;

} // namespace hip_flash2_engine
