// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference SDPA forward for ragged tensors (RFC-0014: packed BSHD + ragged_offset), and for
// paged K/V (vLLM block tables) with dense or ragged Q.
// Compiled via HipRTC with -DQ_TYPE, -DK_TYPE, -DV_TYPE, -DO_TYPE and -DCOMPUTE_TYPE.
//
// Tensors are logical [B, H, S, D] with BSHD strides, packed by token with no per-batch padding:
// q=[B,H,Sq,D], k=[B,Hk,Skv,D], v=[B,Hv,Skv,Dv], o=[B,H,Sq,Dv].
// raggedOffsetQ/raggedOffsetKv are cumulative offsets; times their multiplier they are element
// offsets, and dividing by the seq stride gives token boundaries. One thread per output element
// (tokenGlobalQ, h, dv). Each thread finds its batch and uses that batch's own seqQ/seqKv for
// the key loop and mask alignment.
// Paged: K/V are block containers [num_blocks, H, block_size, D]. Key row skv of batch b lives in
// block pageTable[b, skv / block_size] at row skv % block_size, and seqLenKv[b] gives the key
// count. Dense Q (no raggedOffsetQ) has B * Sq tokens; seqLenQ[b] limits the live rows, and rows
// at or past seqQ are left untouched.
// Numerics match GpuRefSdpaFwd.cpp.

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

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

// Per-batch Q and K/V token ranges. All per-batch addressing goes through the helpers below.
struct BatchRange
{
    long long b;
    long long qBase; // first Q token of the batch (global token index)
    long long seqQ;
    long long kvBase; // first K/V token of the batch (ragged K/V only)
    long long seqKv;
};

// Token boundary i of an offset table: element offset (stored * multiplier) / seq stride.
__device__ inline long long
    tokenAt(const int* offsets, long long i, long long multiplier, long long seqStride)
{
    return static_cast<long long>(offsets[i]) * multiplier / seqStride;
}

__device__ inline long long qTokenAt(const SdpaRaggedFwdArgs& args, long long i)
{
    return tokenAt(args.raggedOffsetQ, i, args.offsetMultiplierQ, args.seqStrideQ);
}

__device__ inline long long kvTokenAt(const SdpaRaggedFwdArgs& args, long long i)
{
    return tokenAt(args.raggedOffsetKv, i, args.offsetMultiplierKv, args.seqStrideKv);
}

// Batch that owns a global Q token. A linear scan is fine for a reference. Empty batches own
// no tokens, so they are never returned. Dense Q owns seqQMax tokens per batch.
__device__ inline long long findBatch(const SdpaRaggedFwdArgs& args, long long tokenGlobalQ)
{
    if(args.raggedOffsetQ == nullptr)
    {
        return tokenGlobalQ / args.seqQMax;
    }
    long long b = 0;
    while(b + 1 < args.batch && tokenGlobalQ >= qTokenAt(args, b + 1))
    {
        ++b;
    }
    return b;
}

__device__ inline BatchRange batchRange(const SdpaRaggedFwdArgs& args, long long b)
{
    BatchRange r{b, 0, 0, 0, 0};
    if(args.raggedOffsetQ != nullptr)
    {
        r.qBase = qTokenAt(args, b);
        r.seqQ = qTokenAt(args, b + 1) - r.qBase;
    }
    else
    {
        r.qBase = b * args.seqQMax;
        r.seqQ = args.seqQMax;
    }
    if(args.seqLenQ != nullptr)
    {
        r.seqQ = static_cast<long long>(args.seqLenQ[b * args.seqLenQStride]);
    }
    if(args.pageTableK != nullptr)
    {
        r.seqKv = static_cast<long long>(args.seqLenKv[b * args.seqLenKvStride]);
    }
    else
    {
        r.kvBase = kvTokenAt(args, b);
        r.seqKv = kvTokenAt(args, b + 1) - r.kvBase;
    }
    return r;
}

// Element offset of query/output row sq (batch-relative). Ragged rows are packed by token, dense
// rows use the batch stride.
__device__ inline long long
    qoRow(const SdpaRaggedFwdArgs& args, const SdpaStrides& str, const BatchRange& r, long long sq)
{
    if(args.raggedOffsetQ != nullptr)
    {
        return (r.qBase + sq) * str.s[2];
    }
    return r.b * str.s[0] + sq * str.s[2];
}

// Element offset of paged row skv: the batch's page-table entry picks the block.
__device__ inline long long pagedRow(const SdpaRaggedFwdArgs& args,
                                     const int* pageTable,
                                     long long ptStr0,
                                     long long ptStr2,
                                     const SdpaStrides& str,
                                     long long b,
                                     long long skv)
{
    const long long block
        = static_cast<long long>(pageTable[b * ptStr0 + (skv / args.blockSize) * ptStr2]);
    return block * str.s[0] + (skv % args.blockSize) * str.s[2];
}

// Element offset of key/value row skv (batch-relative) in the K and V buffers.
__device__ inline long long kRow(const SdpaRaggedFwdArgs& args, const BatchRange& r, long long skv)
{
    if(args.pageTableK != nullptr)
    {
        return pagedRow(
            args, args.pageTableK, args.pageTableKStr0, args.pageTableKStr2, args.kStr, r.b, skv);
    }
    return (r.kvBase + skv) * args.kStr.s[2];
}

__device__ inline long long vRow(const SdpaRaggedFwdArgs& args, const BatchRange& r, long long skv)
{
    if(args.pageTableV != nullptr)
    {
        return pagedRow(
            args, args.pageTableV, args.pageTableVStr0, args.pageTableVStr2, args.vStr, r.b, skv);
    }
    return (r.kvBase + skv) * args.vStr.s[2];
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

    long long totalOutputElements = args.totalQ * args.numHeads * args.headDimV;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Linear index -> (tokenGlobalQ, h, dv) in the packed [total_q, H, Dv] output.
    long long dv = idx % args.headDimV;
    long long tmp = idx / args.headDimV;
    long long h = tmp % args.numHeads;
    long long tokenGlobalQ = tmp / args.numHeads;

    const BatchRange range = batchRange(args, findBatch(args, tokenGlobalQ));
    const long long b = range.b;
    const long long seqQ = range.seqQ;
    const long long seqKv = range.seqKv;
    const long long sq = tokenGlobalQ - range.qBase; // query position within the batch

    // Padding rows (dense Q with seq_len_q) and gap rows (ragged Q with seq_len_q) are left
    // untouched.
    if(sq >= seqQ)
    {
        return;
    }
    const long long qRowBase = qoRow(args, args.qStr, range, sq);

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
            long long qIdx = qRowBase + h * args.qStr.s[1] + d * args.qStr.s[3];
            long long kIdx = kRowBase + kvHeadK * args.kStr.s[1] + d * args.kStr.s[3];
            dot += toAccum(q[qIdx]) * toAccum(k[kIdx]);
        }
        // Fold in the fp8 Q/K descale.
        COMPUTE_TYPE s = dot * descaleQK * static_cast<COMPUTE_TYPE>(args.scale);

        // Sliding-window mask, aligned per batch. Only the right bound gets the +1.
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
            if(skv < sq + windowOffset - args.leftBound)
            {
                s = negInf;
            }
        }
        return s;
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

    long long oIdx = qoRow(args, args.oStr, range, sq) + h * args.oStr.s[1] + dv * args.oStr.s[3];
    O_TYPE* tag = nullptr;

    // Only the dv == 0 thread writes LSE, so each (token, h) has one writer. A ragged LSE
    // starts batch b at element raggedOffsetLse[b] * offsetMultiplierLse, a dense one at
    // b * lseStr.s[0].
    const long long lseBatchBase
        = args.raggedOffsetLse != nullptr
              ? static_cast<long long>(args.raggedOffsetLse[b]) * args.offsetMultiplierLse
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
