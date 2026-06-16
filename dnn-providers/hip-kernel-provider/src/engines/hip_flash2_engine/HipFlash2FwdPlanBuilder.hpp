// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// HipFlash2FwdPlanBuilder: IPlanBuilder for Flash-Attention 2 forward pass
// Wraps our V7 HIP kernel (rocWMMA MFMA + causal skip) as a hipDNN SDPA engine.

#pragma once

#include <cstdint>
#include <string>

namespace hip_flash2_engine
{

// Parameters extracted from the hipDNN graph for kernel launch
struct Flash2FwdParams
{
    // Tensor UIDs (for variant pack lookup)
    int64_t qUid = 0;
    int64_t kUid = 0;
    int64_t vUid = 0;
    int64_t oUid = 0;

    // Attention geometry
    int batch = 1;
    int num_heads_q = 32;
    int num_heads_k = 32; // num_heads_q / num_heads_k = GQA ratio
    int seq_len_q = 1;
    int seq_len_kv = 2048;
    int head_dim = 128;
    float attn_scale = 0.0f; // 0 → computed as 1/sqrt(head_dim)
    bool causal = false;

    // Layout strides (in elements, not bytes)
    // Q layout: [B, Lq, H, D] — stride_seq = H*D, stride_head = D, stride_batch = Lq*H*D
    int64_t q_stride_batch = 0;
    int64_t q_stride_seq = 0;
    int64_t q_stride_head = 0;
    int64_t k_stride_batch = 0;
    int64_t k_stride_seq = 0;
    int64_t k_stride_head = 0;
    int64_t v_stride_batch = 0;
    int64_t v_stride_seq = 0;
    int64_t v_stride_head = 0;
    int64_t o_stride_batch = 0;
    int64_t o_stride_seq = 0;
    int64_t o_stride_head = 0;
};

// Dispatch heuristic: use Flash2 for prefill, batched GEMM for decode
// Matches our UseFlash2ForROCm() from prefill_dispatch_hip.cuh
inline bool useFlash2ForShape(int qo_len, int kv_len)
{
    if(qo_len <= 1)
        return false; // decode: caller should use batched GEMM instead
    uint32_t cta_q_blocks = ((uint32_t)qo_len + 63u) / 64u;
    return (uint64_t)qo_len * (uint64_t)kv_len > (uint64_t)cta_q_blocks * 6000u;
}

class HipFlash2FwdPlanBuilder
{
public:
    // Check if this builder handles the given graph
    // Returns true for:
    //   - Single SDPA node, FP16 input, gfx942
    //   - head_dim in {64, 128}
    //   - seq_len_q * seq_len_kv > 6000 * ceil(seq_len_q/64) (Flash2 crossover)
    bool isApplicable(const std::string& deviceString,
                      int seq_len_q,
                      int seq_len_kv,
                      int head_dim,
                      const std::string& dtype,
                      bool has_dropout,
                      bool has_alibi) const
    {
        if(deviceString != "gfx942" && deviceString != "gfx950")
            return false;
        if(dtype != "fp16")
            return false;
        if(head_dim != 64 && head_dim != 128)
            return false;
        if(has_dropout || has_alibi)
            return false;
        if(!useFlash2ForShape(seq_len_q, seq_len_kv))
            return false;
        return true;
    }

    // Workspace: 0 bytes (Flash2 uses registers + LDS only)
    size_t getMaxWorkspaceSize() const
    {
        return 0;
    }
};

} // namespace hip_flash2_engine
