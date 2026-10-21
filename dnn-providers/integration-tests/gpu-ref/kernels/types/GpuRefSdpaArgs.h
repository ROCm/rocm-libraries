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
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)

// --- Ragged SDPA forward args (RFC-0014: packed BSHD + ragged_offset), also paged KV ---
// Tensors are logical [B, H, S, D] with BSHD strides, packed by token with no per-batch padding:
// q=[B,H,Sq,D], k=[B,Hk,Skv,D], v=[B,Hv,Skv,Dv], o=[B,H,Sq,Dv].
// Only Q and K offsets are passed. o shares Q's token boundaries and v shares K's. The host
// checks this against the o/v ragged_offset tables before launch.
// Paged variant: raggedOffsetQ == nullptr means dense Q/O (seqQMax tokens per batch), and
// pageTableK != nullptr means K/V are block containers [num_blocks, H, block_size, D] addressed
// through the page tables, with seqLenKv giving each batch's key count.
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
    // Paged / dense-Q extension. All nullptr for the plain ragged path.
    // seqLenQ/seqLenKv: int32 [B], element b at b * stride. seqLenQ overrides the Q length.
    const int* seqLenQ;
    const int* seqLenKv;
    long long seqLenQStride, seqLenKvStride;
    // Page tables, int32 [B, 1, M, 1]: block id of entry (b, j) at b * str0 + j * str2.
    const int* pageTableK;
    const int* pageTableV;
    long long pageTableKStr0, pageTableKStr2;
    long long pageTableVStr0, pageTableVStr2;
    long long blockSize;
    // Dense Q/O (raggedOffsetQ == nullptr): Sq, the per-batch Q extent.
    long long seqQMax;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)
