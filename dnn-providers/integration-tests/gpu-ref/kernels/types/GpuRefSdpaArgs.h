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

// --- Score pipeline parameters shared by the forward and backward kernels ---
// GpuRefSdpaCommon.hpp applies them: (q . k) * scale, softcap, + bias, + ALiBi, then the band.
// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming,
//             modernize-avoid-c-arrays)
struct SdpaScoreParams
{
    // Optional additive bias (MASK_TYPE), right-aligned onto [B, H, Sq, Skv]; nullptr disables it.
    const void* mask;
    SdpaStrides maskStr;
    int maskRank;
    long long maskDims[4];
    float scale;
    // > 0 enables the logit softcap c * tanh(s / c) (reference-only extension, SEM-020).
    float softcap;
    // Nonzero enables cuDNN ALiBi: + (skv - sq) * slope(h) after the bias (SEM-021).
    int alibi;
    // cuDNN diagonal band (SEM-001); -1 is unbounded. Bottom-right measures the diagonal from the
    // batch's valid lengths.
    long long leftBound, rightBound;
    int topLeftAlignment;
    // cuDNN block-sparse mask (SEM-025), UINT8 [B, H, ceil(Sq / 128), ceil(ceil(Skv / 128) / 8)]:
    // a clear bit masks a 128 x 128 score tile. nullptr disables it (the backward has none).
    const unsigned char* blockMask;
    SdpaStrides blockMaskStr;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)

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
    void* o;
    // Optional log-sum-exp output [B, H, Sq], always float. nullptr disables it.
    void* lse;
    SdpaStrides qStr;
    SdpaStrides kStr;
    SdpaStrides vStr;
    SdpaStrides oStr;
    SdpaStrides lseStr;
    long long batch, numHeads, numHeadsK, numHeadsV;
    long long seqQ, seqKv, headDim, headDimV;
    SdpaScoreParams score;
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
    // Optional cuDNN Max / Sum_exp row statistics, float [B, H, Sq]; nullptr disables each.
    void* rowMax;
    void* rowSumExp;
    SdpaStrides rowMaxStr;
    SdpaStrides rowSumExpStr;
    // cuDNN sdpa_fp8 softmax scaling (SEM-024): P @ V uses
    // fp8(exp(s - Max) * scaleS) * descaleS / Sum_exp, in format pFp8Format (0 = off, else a
    // GPU_REF_FP8_* format from GpuRefTypes.h). amaxS (nullptr = off, zeroed by the host) gets max exp(s - Max).
    int pFp8Format;
    float scaleS;
    float descaleS;
    float* amaxS;
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
    const float* lse;
    void* dQ;
    void* dK;
    void* dV;
    void* dBias;
    float* rowD;
    float* rowM;
    float* rowL;
    SdpaStrides qStr, kStr, vStr, oStr, dOStr, dQStr, dKStr, dVStr, dBiasStr, lseStr;
    long long batch, numHeads, numHeadsK, numHeadsV;
    long long seqQ, seqKv, headDim, headDimV;
    SdpaScoreParams score;
    const int* seqLenQ;
    const int* seqLenKv;
    long long seqLenQStride, seqLenKvStride;
    // Optional explicit-mask dropout (see SdpaFwdArgs): D is scaled by dropoutScaleInv, dP is
    // masked, dS is scaled by dropoutScale, and dV uses P * dropoutScale * mask.
    const float* dropoutMask;
    SdpaStrides dropoutMaskStr;
    float dropoutScale;
    float dropoutScaleInv;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)

// --- Ragged SDPA forward args (RFC-0014: ragged_offset origins + per-batch lengths) ---
// Tensors are logical [B, H, S, D] with arbitrary non-aliasing strides:
// q=[B,H,Sq,D], k=[B,Hk,Skv,D], v=[B,Hv,Skv,Dv], o=[B,H,Sq,Dv], lse=[B,H,Sq,1].
// Every ragged tensor carries its own offset table. Batch b of a tensor starts at element
// offset[b] * multiplier and holds seqLen[b * stride] token rows (seqLenQ for Q, O and LSE,
// seqLenKv for K and V), or the offset block's width capped at the sequence extent when seqLen is nullptr
// (RaggedBatchMapping.hpp). The host validates the layouts and that tensors sharing a length
// agree on it before launch.
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
    // Cumulative offsets (RFC-0014 ragged_offset), int32, length batch+1. Element offset =
    // offset * multiplier (1 for element tables, H*D for AITER's token tables).
    const int* raggedOffsetQ;
    const int* raggedOffsetK;
    const int* raggedOffsetV;
    const int* raggedOffsetO;
    // Ragged LSE offsets, length batch+1. nullptr means a dense LSE addressed by lseStr.
    const int* raggedOffsetLse;
    long long offsetMultiplierQ;
    long long offsetMultiplierK;
    long long offsetMultiplierV;
    long long offsetMultiplierO;
    long long offsetMultiplierLse;
    // Batch-axis strides (elements) of the offset tables: entry i sits at offset[i * stride].
    long long offsetStrideQ;
    long long offsetStrideK;
    long long offsetStrideV;
    long long offsetStrideO;
    long long offsetStrideLse;
    // Optional per-batch lengths, int32 [batch], element b at b * stride. nullptr: derived from
    // the offsets.
    const int* seqLenQ;
    const int* seqLenKv;
    long long seqLenQStride;
    long long seqLenKvStride;
    // Sequence extents (S_max) of Q and of K/V.
    long long seqMaxQ;
    long long seqMaxKv;
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
    long long batch, numHeads, numHeadsK, numHeadsV;
    long long headDim, headDimV;
    float scale;
    long long leftBound, rightBound;
    int topLeftAlignment;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)

// --- fp8 dequantize (the SDPA forward's dequantize-first fp8 wrapper) ---
// out (packed float, x's dims) = float(x) * descale(b, h / headsPerKv). The descale is a scalar
// (both strides zero) or [B, H_kv, 1, 1]; nullptr means 1. dims is the logical [B, H, S, D]: x's
// own dims when dense, [B, H, max_seq_kv, D] for a paged container, whose logical row s of batch b
// lives at block pageTable[b * pageTableStrB + (s / blockSize) * pageTableStrT], slot
// s % blockSize (out keeps the container's packed layout). Rows s >= seqLen[b * seqLenStride]
// (padding; seqLen nullptr = none) are neither read nor written (SEM-008).
// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming,
//             modernize-avoid-c-arrays)
struct SdpaDequantArgs
{
    const void* in;
    float* out;
    SdpaStrides inStr;
    long long dims[4];
    const float* descale;
    long long descaleBatchStride, descaleHeadStride;
    long long headsPerKv;
    const int* seqLen;
    long long seqLenStride;
    const int* pageTable;
    long long pageTableStrB, pageTableStrT, blockSize;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)

// --- fp8 output scaling (cuDNN sdpa_fp8 scale_o / amax_o) ---
// out (O_TYPE, outStr) = in (packed float, dims) * scale; amax (nullptr = off, zeroed by the
// host) gets max |in|.
// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming,
//             modernize-avoid-c-arrays)
struct SdpaScaleOutArgs
{
    const float* in;
    void* out;
    SdpaStrides outStr;
    long long dims[4];
    float scale;
    float* amax;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming,
//           modernize-avoid-c-arrays)
