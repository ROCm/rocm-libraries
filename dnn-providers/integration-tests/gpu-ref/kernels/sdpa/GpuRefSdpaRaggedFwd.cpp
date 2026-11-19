// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference SDPA forward for ragged tensors (RFC-0014: ragged_offset origins + per-batch
// lengths). Compiled via HipRTC with -DQ_TYPE, -DK_TYPE, -DV_TYPE, -DO_TYPE and -DCOMPUTE_TYPE.
//
// Tensors are logical [B, H, S, D] with arbitrary non-aliasing strides:
// q=[B,H,Sq,D], k=[B,Hk,Skv,D], v=[B,Hv,Skv,Dv], o=[B,H,Sq,Dv].
// Each of Q, K, V, O (and a ragged LSE) has its own offset table, so batch b of each tensor
// starts at its own origin; RaggedBatchMapping.hpp gives the origin and live row count per
// (tensor, batch). One thread per output element (b, sq, h, dv); threads past the batch's
// query length exit. Each thread uses its batch's own seqQ/seqKv for the key loop and mask
// alignment.
// Numerics match GpuRefSdpaFwd.cpp.

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"
#include "RaggedBatchMapping.hpp"

using namespace gpu_ref;

// expf, -__builtin_huge_valf() and the softmax below assume float. Fail at compile time
// rather than silently truncate.
static_assert(__is_same(COMPUTE_TYPE, float), "GpuRefSdpaRaggedFwd requires COMPUTE_TYPE == float");

#define SDPA_SOFTMAX_PROBABILITY_FLOAT 0
#define SDPA_SOFTMAX_PROBABILITY_BFLOAT16_RTNE 1
#define SDPA_SOFTMAX_PROBABILITY_BFLOAT16_RTZ 2

#ifndef SDPA_SOFTMAX_PROBABILITY_MODE
#define SDPA_SOFTMAX_PROBABILITY_MODE SDPA_SOFTMAX_PROBABILITY_FLOAT
#endif

namespace
{

__device__ inline float truncatePositiveFloatToBfloat16(float value)
{
    // Softmax probabilities are non-negative, so clearing the low 16 bits rounds toward zero.
    unsigned int bits = __builtin_bit_cast(unsigned int, value) & 0xFFFF0000U;
    return __builtin_bit_cast(float, bits);
}

__device__ inline COMPUTE_TYPE storeSoftmaxProbability(COMPUTE_TYPE probability)
{
#if SDPA_SOFTMAX_PROBABILITY_MODE == SDPA_SOFTMAX_PROBABILITY_FLOAT
    return probability;
#elif SDPA_SOFTMAX_PROBABILITY_MODE == SDPA_SOFTMAX_PROBABILITY_BFLOAT16_RTNE
    return static_cast<COMPUTE_TYPE>(static_cast<__bf16>(probability));
#elif SDPA_SOFTMAX_PROBABILITY_MODE == SDPA_SOFTMAX_PROBABILITY_BFLOAT16_RTZ
    return static_cast<COMPUTE_TYPE>(truncatePositiveFloatToBfloat16(probability));
#else
#error "Unsupported SDPA_SOFTMAX_PROBABILITY_MODE"
#endif
}

// Per-batch Q and K/V rows. All per-batch addressing goes through the helpers below, so
// paged KV only has to add a page-table lookup here.
struct BatchRange
{
    long long b;
    hipdnn_gpu_ref::RaggedBatchRows q;
    hipdnn_gpu_ref::RaggedBatchRows k;
    hipdnn_gpu_ref::RaggedBatchRows v;
    hipdnn_gpu_ref::RaggedBatchRows o;
};

__device__ inline BatchRange batchRange(const SdpaRaggedFwdArgs& args, long long b)
{
    using hipdnn_gpu_ref::raggedBatchRows;
    return {b,
            raggedBatchRows(args.raggedOffsetQ,
                            args.offsetStrideQ,
                            args.seqLenQ,
                            args.seqLenQStride,
                            b,
                            args.offsetMultiplierQ,
                            args.qStr.s[2],
                            args.seqMaxQ),
            raggedBatchRows(args.raggedOffsetK,
                            args.offsetStrideK,
                            args.seqLenKv,
                            args.seqLenKvStride,
                            b,
                            args.offsetMultiplierK,
                            args.kStr.s[2],
                            args.seqMaxKv),
            raggedBatchRows(args.raggedOffsetV,
                            args.offsetStrideV,
                            args.seqLenKv,
                            args.seqLenKvStride,
                            b,
                            args.offsetMultiplierV,
                            args.vStr.s[2],
                            args.seqMaxKv),
            raggedBatchRows(args.raggedOffsetO,
                            args.offsetStrideO,
                            args.seqLenQ,
                            args.seqLenQStride,
                            b,
                            args.offsetMultiplierO,
                            args.oStr.s[2],
                            args.seqMaxQ)};
}

// Element offset of key/value row skv (batch-relative) in the K and V buffers.
__device__ inline long long kRow(const SdpaRaggedFwdArgs& args, const BatchRange& r, long long skv)
{
    return r.k.origin + skv * args.kStr.s[2];
}

__device__ inline long long vRow(const SdpaRaggedFwdArgs& args, const BatchRange& r, long long skv)
{
    return r.v.origin + skv * args.vStr.s[2];
}

} // namespace

extern "C" __global__ void sdpaRaggedFwdRef(SdpaRaggedFwdArgs args)
{
    auto* q = static_cast<const Q_TYPE*>(args.q);
    auto* k = static_cast<const K_TYPE*>(args.k);
    auto* v = static_cast<const V_TYPE*>(args.v);
    auto* o = static_cast<O_TYPE*>(args.o);
    // LSE is float, [B, H, Sq, 1]. nullptr disables it.
    auto* lse = static_cast<float*>(args.lse);

    long long totalOutputElements = args.batch * args.seqMaxQ * args.numHeads * args.headDimV;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Linear index -> (b, sq, h, dv).
    long long dv = idx % args.headDimV;
    long long tmp = idx / args.headDimV;
    long long h = tmp % args.numHeads;
    tmp /= args.numHeads;
    const long long sq = tmp % args.seqMaxQ; // query position within the batch
    const long long b = tmp / args.seqMaxQ;

    const BatchRange range = batchRange(args, b);
    const long long seqQ = range.q.length;
    const long long seqKv = range.k.length;
    if(sq >= seqQ)
    {
        return; // not a live row of this batch
    }

    // GQA/MQA: K and V head counts are independent.
    long long kvHeadK = h / (args.numHeads / args.numHeadsK);
    long long kvHeadV = h / (args.numHeads / args.numHeadsV);

    // fp8 descale, 1 when absent. Q and K use the K head, V uses the V head (AITER [B, H_kv]
    // contract). No softmax or output requant.
    const COMPUTE_TYPE descaleQ
        = args.descaleQ != nullptr
              ? args.descaleQ[b * args.descaleQBatchStride + kvHeadK * args.descaleQHeadStride]
              : static_cast<COMPUTE_TYPE>(1);
    const COMPUTE_TYPE descaleK
        = args.descaleK != nullptr
              ? args.descaleK[b * args.descaleKBatchStride + kvHeadK * args.descaleKHeadStride]
              : static_cast<COMPUTE_TYPE>(1);
    const COMPUTE_TYPE descaleV
        = args.descaleV != nullptr
              ? args.descaleV[b * args.descaleVBatchStride + kvHeadV * args.descaleVHeadStride]
              : static_cast<COMPUTE_TYPE>(1);
    const COMPUTE_TYPE descaleQK = descaleQ * descaleK;

    // Window alignment uses this batch's seqQ/seqKv, as in CpuFpReferenceSdpa step 3.
    long long windowOffset = args.topLeftAlignment ? 0 : (seqKv - seqQ);

    // INFINITY from <math.h> is not available under HipRTC, so use the clang builtin.
    const COMPUTE_TYPE negInf = -__builtin_huge_valf();

    // Masked, scaled score for within-batch key skv. Recomputed in each pass to keep the
    // reference simple. Strides: s[1] head, s[2] token, s[3] dim.
    auto score = [&](long long skv) -> COMPUTE_TYPE {
        const long long kRowBase = kRow(args, range, skv);
        COMPUTE_TYPE dot = static_cast<COMPUTE_TYPE>(0);
        for(long long d = 0; d < args.headDim; ++d)
        {
            long long qIdx
                = range.q.origin + sq * args.qStr.s[2] + h * args.qStr.s[1] + d * args.qStr.s[3];
            long long kIdx = kRowBase + kvHeadK * args.kStr.s[1] + d * args.kStr.s[3];
            dot += toAccum(q[qIdx]) * toAccum(k[kIdx]);
        }
        // Fold in the fp8 Q/K descale.
        COMPUTE_TYPE s = dot * descaleQK * static_cast<COMPUTE_TYPE>(args.scale);

        // Diagonal band, aligned per batch: keep keys in (sq + off - leftBound, sq + off + rightBound],
        // so leftBound = L keeps L keys including the diagonal (cuDNN).
        // No additive bias: it is gated off on the ASM v3 path.
        if(args.rightBound >= 0)
        {
            long long startKv = sq + 1 + windowOffset + args.rightBound;
            if(startKv < 0)
            {
                startKv = 0;
            }
            if(skv >= startKv)
            {
                s = negInf;
            }
        }
        if(args.leftBound >= 0)
        {
            if(skv <= sq + windowOffset - args.leftBound)
            {
                s = negInf;
            }
        }
        return s;
    };

    // Whether the band removes key skv (the rule score() applies). A removed key's V is never
    // read: 0 * NaN would leak whatever it holds (SEM-027).
    auto bandRemoves = [&](long long skv) -> bool {
        return (args.rightBound >= 0 && skv >= sq + 1 + windowOffset + args.rightBound)
               || (args.leftBound >= 0 && skv <= sq + windowOffset - args.leftBound);
    };

    // Pass 1: row max for a stable softmax.
    COMPUTE_TYPE maxVal = negInf;
    for(long long skv = 0; skv < seqKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        if(s > maxVal)
        {
            maxVal = s;
        }
    }

    long long oIdx
        = range.o.origin + sq * args.oStr.s[2] + h * args.oStr.s[1] + dv * args.oStr.s[3];
    O_TYPE* tag = nullptr;

    // Only the dv == 0 thread writes LSE, so each (token, h) has one writer. A ragged LSE
    // starts batch b at its own origin, a dense one at b * lseStr.s[0].
    const long long lseBatchBase = args.raggedOffsetLse != nullptr
                                       ? hipdnn_gpu_ref::raggedBatchRows(args.raggedOffsetLse,
                                                                         args.offsetStrideLse,
                                                                         args.seqLenQ,
                                                                         args.seqLenQStride,
                                                                         b,
                                                                         args.offsetMultiplierLse,
                                                                         args.lseStr.s[2],
                                                                         args.seqMaxQ)
                                             .origin
                                       : b * args.lseStr.s[0];
    long long lseIdx = lseBatchBase + h * args.lseStr.s[1] + sq * args.lseStr.s[2];

    // Fully masked row, including seqKv == 0: write zero to match CpuFpReferenceSdpa and
    // avoid a 0/0 NaN.
    if(maxVal == negInf)
    {
        o[oIdx] = fromAccum(static_cast<COMPUTE_TYPE>(0), tag);
        // Matches the CPU's -inf + log(0) = -inf.
        if(lse != nullptr && dv == 0)
        {
            lse[lseIdx] = negInf;
        }
        return;
    }

    // Pass 2: softmax denominator.
    COMPUTE_TYPE sumExp = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < seqKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        // Device expf and host std::exp<float> agree within test tolerance, not bit-for-bit.
        sumExp += expf(s - maxVal);
    }

    // Pass 3: P @ V. Provider-attuned modes round P to bf16 first, as matrix-core kernels do
    // before the second matmul.
    COMPUTE_TYPE weighted = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < seqKv; ++skv)
    {
        if(bandRemoves(skv))
        {
            continue;
        }
        COMPUTE_TYPE s = score(skv);
        COMPUTE_TYPE probability = expf(s - maxVal) / sumExp;
        probability = storeSoftmaxProbability(probability);

        long long vIdx = vRow(args, range, skv) + kvHeadV * args.vStr.s[1] + dv * args.vStr.s[3];
        weighted += probability * toAccum(v[vIdx]);
    }

    // Fold in the fp8 V descale.
    weighted *= descaleV;

    o[oIdx] = fromAccum(weighted, tag);

    // LSE = maxVal + log(sumExp), as in CpuFpReferenceSdpa. sumExp >= 1 because the max term
    // contributes exp(0).
    if(lse != nullptr && dv == 0)
    {
        lse[lseIdx] = static_cast<float>(maxVal) + logf(sumExp);
    }
}
