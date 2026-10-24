// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shared argument structs for GPU reference SDPA kernels.
// Included by both device code (HipRTC) and host launch code.
// Only POD types allowed — no host or device includes.

#pragma once

// --- Stride struct for stride-based indexing ---
// Distinct from Strides4 (defined in GpuRefConvArgs.h) to avoid an ODR clash:
// the SDPA kernel pulls in GpuRefConvArgs.h transitively via GpuRefTypes.h.

// NOLINTBEGIN(modernize-avoid-c-arrays)
struct SdpaStrides
{
    long long s[4];
};
// NOLINTEND(modernize-avoid-c-arrays)

// --- SDPA forward argument struct ---
// Shared between device kernels and host launch code for ABI compatibility.

// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming,
//             modernize-avoid-c-arrays)
struct SdpaFwdArgs
{
    const void* q;
    const void* k;
    const void* v;
    const void* mask;
    void* o;
    // Optional log-sum-exp output [B, H, Sq], always float. nullptr disables it.
    void* lse;
    SdpaStrides qStr;
    SdpaStrides kStr;
    SdpaStrides vStr;
    SdpaStrides oStr;
    SdpaStrides maskStr;
    SdpaStrides lseStr;
    long long batch, numHeads, numHeadsK, numHeadsV;
    long long seqQ, seqKv, headDim, headDimV;
    int maskRank;
    long long maskDims[4];
    float scale;
    long long leftBound, rightBound;
    int topLeftAlignment;
    // Optional padding (cuDNN seq_len_q / seq_len_kv): one INT32 per batch, read at
    // b * stride. nullptr means every batch uses the full seqQ / seqKv.
    const int* seqLenQ;
    const int* seqLenKv;
    long long seqLenQStride, seqLenKvStride;
    // Optional paged K / V (cuDNN): the k / v tensors are containers [blocks, H, blockSize, D]
    // and logical key j of batch b is block table[b * tableStrB + (j / blockSize) * tableStrT],
    // slot j % blockSize. nullptr means that side is dense [B, H, seqKv, D].
    const int* pageTableK;
    const int* pageTableV;
    long long pageTableKStrB, pageTableKStrT, blockSizeK;
    long long pageTableVStrB, pageTableVStrT, blockSizeV;
    // Optional attention sink (cuDNN): one fp32 logit per query head, read at h * sinkStride.
    const float* sink;
    long long sinkStride;
    // Optional explicit-mask dropout (cuDNN custom dropout): FLOAT mask [B|1, H|1, Sq, Skv]
    // (a size-1 dim has stride 0 here) and scale; O uses P * dropoutScale * mask.
    const float* dropoutMask;
    SdpaStrides dropoutMaskStr;
    float dropoutScale;
    // Optional logit softcap c (> 0 enables): scores become c * tanh(s / c) after the attention
    // scale and before bias and masks. Reference-only extension (SEM-020).
    float softcap;
    // Optional cuDNN Max / Sum_exp row statistics, float [B, H, Sq]; nullptr disables each.
    void* rowMax;
    void* rowSumExp;
    SdpaStrides rowMaxStr;
    SdpaStrides rowSumExpStr;
    // cuDNN ALiBi: nonzero adds (skv - sq) * slope(h) after the bias (SEM-021).
    int alibi;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)

// --- SDPA backward argument struct ---
// Mirrors CpuFpReferenceSdpa::backward. lse == nullptr selects the softmax-recompute path. rowD,
// rowM and rowL are fp32 [B, H, Sq] contiguous scratch written by the row-stats kernel: D =
// sum(dO * O), and either the LSE (rowM) or the recomputed row max and exp-sum.
// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming,
//             modernize-avoid-c-arrays)
struct SdpaBwdArgs
{
    const void* q;
    const void* k;
    const void* v;
    const void* o;
    const void* dO;
    const void* mask;
    const float* lse;
    void* dQ;
    void* dK;
    void* dV;
    void* dBias;
    float* rowD;
    float* rowM;
    float* rowL;
    SdpaStrides qStr, kStr, vStr, oStr, dOStr, dQStr, dKStr, dVStr, maskStr, dBiasStr, lseStr;
    long long batch, numHeads, numHeadsK, numHeadsV;
    long long seqQ, seqKv, headDim, headDimV;
    int maskRank;
    long long maskDims[4];
    float scale;
    long long leftBound, rightBound;
    int topLeftAlignment;
    const int* seqLenQ;
    const int* seqLenKv;
    long long seqLenQStride, seqLenKvStride;
    // Optional explicit-mask dropout (see SdpaFwdArgs): D is scaled by dropoutScaleInv, dP is
    // masked, dS is scaled by dropoutScale, and dV uses P * dropoutScale * mask.
    const float* dropoutMask;
    SdpaStrides dropoutMaskStr;
    float dropoutScale;
    float dropoutScaleInv;
    float softcap; // > 0 enables the logit softcap, as in SdpaFwdArgs
    int alibi; // nonzero enables cuDNN ALiBi, as in SdpaFwdArgs
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)

// --- Ragged SDPA forward args (RFC-0014: packed BSHD + ragged_offset) ---
// Tensors are logical [B, H, S, D] with BSHD strides, packed by token with no per-batch padding:
// q=[B,H,Sq,D], k=[B,Hk,Skv,D], v=[B,Hv,Skv,Dv], o=[B,H,Sq,Dv].
// Only Q and K offsets are passed. o shares Q's token boundaries and v shares K's. The host
// checks this against the o/v ragged_offset tables before launch.
// No additive mask: bias is gated off on the ASM v3 path.

// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming,
//             modernize-avoid-c-arrays)
struct SdpaRaggedFwdArgs
{
    const void* q;
    const void* k;
    const void* v;
    void* o;
    // Optional log-sum-exp output, float, [B, H, Sq, 1]. nullptr disables it.
    void* lse;
    // Ragged LSE offsets, length batch+1. nullptr means a dense LSE addressed by lseStr.
    const int* raggedOffsetLse;
    // Cumulative offsets (RFC-0014 ragged_offset), int32, length batch+1. Element offset =
    // offset * offsetMultiplier (1 for element tables, H*D for AITER's token tables).
    // Q's offsets also give o's token boundaries.
    const int* raggedOffsetQ;
    const int* raggedOffsetKv;
    long long offsetMultiplierQ;
    long long offsetMultiplierKv;
    long long offsetMultiplierLse;
    // Elements per token: H*D for Q, Hk*D for K. element offset / seqStride = token boundary.
    long long seqStrideQ;
    long long seqStrideKv;
    // Optional fp8 descale (nullptr = none), indexed through the batch/head strides below.
    // Per-tensor descale uses zero strides. Q and K use the K head, V uses the V head.
    // No softmax or output requant (AITER fp8 fwd contract).
    const float* descaleQ;
    const float* descaleK;
    const float* descaleV;
    long long descaleQBatchStride, descaleQHeadStride;
    long long descaleKBatchStride, descaleKHeadStride;
    long long descaleVBatchStride, descaleVHeadStride;
    SdpaStrides qStr;
    SdpaStrides kStr;
    SdpaStrides vStr;
    SdpaStrides oStr;
    SdpaStrides lseStr;
    long long batch, totalQ, numHeads, numHeadsK, numHeadsV;
    long long headDim, headDimV;
    float scale;
    long long leftBound, rightBound;
    int topLeftAlignment;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)
