// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace asm_sdpa_engine
{

/**
 * @brief Parameters for SDPA backward kernel execution.
 *
 * Holds tensor UIDs, dimensions, strides, and attention scale
 * extracted from the operation graph.  All strides are in elements
 * (converted to bytes at kernel-launch time).
 */
struct SdpaBwdParams
{
    // --- Tensor UIDs ---
    // Inputs
    int64_t qUid;
    int64_t kUid;
    int64_t vUid;
    int64_t oUid;
    int64_t doUid;
    int64_t statsUid; // LSE from forward pass

    // Outputs
    int64_t dqUid;
    int64_t dkUid;
    int64_t dvUid;

    // --- Tensor dimensions ---
    unsigned int batchSize; // B
    unsigned int numHeadsQ; // H_q
    unsigned int numHeadsKv; // H_kv
    unsigned int seqLenQ; // S_q
    unsigned int seqLenKv; // S_kv (= S_k)
    unsigned int headDimQk; // D_qk (128 for POC)
    unsigned int headDimV; // D_v  (128 for POC)

    // --- Tensor strides (in elements) ---
    // Q: [B, H_q, S_q, D_qk]
    unsigned int qStrideSeq;
    unsigned int qStrideHead;
    unsigned int qStrideBatch;

    // K: [B, H_kv, S_kv, D_qk]
    unsigned int kStrideSeq;
    unsigned int kStrideHead;
    unsigned int kStrideBatch;

    // V: [B, H_kv, S_kv, D_v]
    unsigned int vStrideSeq;
    unsigned int vStrideHead;
    unsigned int vStrideBatch;

    // O: [B, H_q, S_q, D_v]
    unsigned int oStrideSeq;
    unsigned int oStrideHead;
    unsigned int oStrideBatch;

    // dO: [B, H_q, S_q, D_v]
    unsigned int doStrideSeq;
    unsigned int doStrideHead;
    unsigned int doStrideBatch;

    // dQ: [B, H_q, S_q, D_qk]
    unsigned int dqStrideSeq;
    unsigned int dqStrideHead;
    unsigned int dqStrideBatch;

    // dK: [B, H_kv, S_kv, D_qk]
    unsigned int dkStrideSeq;
    unsigned int dkStrideHead;
    unsigned int dkStrideBatch;

    // dV: [B, H_kv, S_kv, D_v]
    unsigned int dvStrideSeq;
    unsigned int dvStrideHead;
    unsigned int dvStrideBatch;

    // Stats (LSE): [B, H_q, S_q] in FP32
    unsigned int statsStrideHead;
    unsigned int statsStrideBatch;

    // Attention scale
    float attnScale;
};

} // namespace asm_sdpa_engine
