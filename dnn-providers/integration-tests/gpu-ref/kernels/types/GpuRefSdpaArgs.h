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

// --- Ragged (RFC-0014: packed [B,H,S,D] + ragged_offset) SDPA forward argument struct ---
// Logical rank-4 dims [B, H, S, D] with BSHD-layout strides (seq stride = H*D): q=[B,H,Sq,D],
// k=[B,Hk,Skv,D], v=[B,Hv,Skv,Dv], o=[B,H,Sq,Dv]. The physical buffer is packed (no per-batch
// padding): batch b's data begins at element offset ragged_offset[b]. raggedOffsetQ/raggedOffsetKv
// are the cumulative ELEMENT offsets (RFC-0014), int32, length batch+1, for the Q and K tensors
// respectively (o shares Q token boundaries; v shares K token boundaries). The per-tensor sequence
// stride (seqStrideQ = qStr.s[2], seqStrideKv = kStr.s[2]) converts an element offset to a token
// count: tokenBoundary[b] = ragged_offset[b] / seqStride. Global-token addressing
// (globalToken * seqStride + h * headStride + d) lands in the packed buffer for every tensor, so
// only the Q- and K-side offsets are needed. No additive mask (bias is gated off on the ASM v3
// path). lseStr is rank-4 for a packed [B,H,Sq,1] LSE (seq stride = H).

// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming,
//             modernize-avoid-c-arrays)
struct SdpaRaggedFwdArgs
{
    const void* q;
    const void* k;
    const void* v;
    void* o;
    // Optional log-sum-exp output, packed [B, H, Sq, 1], always float. nullptr disables it.
    void* lse;
    // Cumulative ELEMENT offsets (RFC-0014 ragged_offset), int32, length batch+1. raggedOffsetQ is
    // the Q tensor's offset (also gives o's token boundaries); raggedOffsetKv is the K tensor's.
    const int* raggedOffsetQ;
    const int* raggedOffsetKv;
    // Per-tensor sequence-axis strides (elements per token): H*D for Q, Hk*D for K. Divide a
    // ragged_offset by these to recover token boundaries.
    long long seqStrideQ;
    long long seqStrideKv;
    // Optional fp8 Q/K/V descale (float; nullptr = none). Indexed by (batch, head) via the
    // batch/head strides below; per-tensor [1] descale uses zero strides. descaleQ is indexed by
    // the Q head; descaleK/descaleV by the KV head. Applied as: score *= descaleQ*descaleK,
    // output *= descaleV. No softmax/output requant (AITER fp8 fwd contract).
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
