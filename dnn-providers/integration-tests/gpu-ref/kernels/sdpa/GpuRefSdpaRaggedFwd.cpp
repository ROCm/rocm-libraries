// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference ragged (RFC-0014: packed [B,H,S,D] + ragged_offset) SDPA forward kernel.
// Compiled via HipRTC with -DQ_TYPE=<type> -DK_TYPE=<type> -DV_TYPE=<type>
// -DO_TYPE=<type> -DCOMPUTE_TYPE=<type>.
//
// Logical rank-4 dims [B, H, S, D] with BSHD-layout strides (seq stride = s[2] = H*D):
//   q=[B,H,Sq,D]  k=[B,Hk,Skv,D]  v=[B,Hv,Skv,Dv]  o=[B,H,Sq,Dv].
// The physical buffer is packed in global-token order, so globalToken*seqStride + h*headStride + d
// addresses every tensor. raggedOffsetQ / raggedOffsetKv are cumulative ELEMENT offsets (RFC-0014);
// dividing by the per-tensor seq stride recovers token boundaries. One thread per output element
// (tokenGlobalQ, h, dv). Each thread maps its global Q token to a batch via raggedOffsetQ and
// bounds/aligns the key loop by that batch's own seqQ_b / seqKv_b. Numerics (fp32 softmax,
// provider-attuned P storage) mirror GpuRefSdpaFwd.cpp exactly.

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

using namespace gpu_ref;

// The kernel computes in float: expf, -__builtin_huge_valf(), and the std::exp-matching
// softmax all assume it. COMPUTE_TYPE is float by design (see buildSdpaDefines); enforce it
// so a non-float compute path fails loudly at compile time instead of silently truncating.
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
    // Softmax probabilities are non-negative, so clearing the low 16 mantissa bits
    // is exactly round-toward-zero for the bf16 P-storage cast.
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
    // Softmax probabilities are non-negative, so truncating the low 16 mantissa bits
    // implements round-toward-zero for the provider's P-storage cast.
    return static_cast<COMPUTE_TYPE>(truncatePositiveFloatToBfloat16(probability));
#else
#error "Unsupported SDPA_SOFTMAX_PROBABILITY_MODE"
#endif
}

} // namespace

extern "C" __global__ void sdpaRaggedFwdRef(SdpaRaggedFwdArgs args)
{
    auto* q = static_cast<const Q_TYPE*>(args.q);
    auto* k = static_cast<const K_TYPE*>(args.k);
    auto* v = static_cast<const V_TYPE*>(args.v);
    auto* o = static_cast<O_TYPE*>(args.o);
    // LSE is always float, packed [B, H, Sq, 1]; nullptr disables it. Written once per
    // (tokenGlobalQ, h) by the dv == 0 thread (see below).
    auto* lse = static_cast<float*>(args.lse);

    long long totalOutputElements = args.totalQ * args.numHeads * args.headDimV;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (tokenGlobalQ, h, dv) for the packed [total_q, H, Dv] output.
    long long dv = idx % args.headDimV;
    long long tmp = idx / args.headDimV;
    long long h = tmp % args.numHeads;
    long long tokenGlobalQ = tmp / args.numHeads;

    // Map the global Q token to its batch via the cumulative ELEMENT offsets, converted to token
    // boundaries by dividing by the Q sequence stride. Linear scan: fine for a reference, and
    // batch counts are small. Batches with seqQ_b == 0 own no Q tokens, so no thread lands in them.
    long long b = 0;
    for(long long batchIdx = 0; batchIdx < args.batch; ++batchIdx)
    {
        const long long lo = static_cast<long long>(args.raggedOffsetQ[batchIdx]) / args.seqStrideQ;
        const long long hi
            = static_cast<long long>(args.raggedOffsetQ[batchIdx + 1]) / args.seqStrideQ;
        if(tokenGlobalQ >= lo && tokenGlobalQ < hi)
        {
            b = batchIdx;
            break;
        }
    }

    // Per-batch token boundaries derived from the element offsets (exact: offsets are whole
    // multiples of the sequence stride).
    const long long qBase = static_cast<long long>(args.raggedOffsetQ[b]) / args.seqStrideQ;
    const long long kvBase = static_cast<long long>(args.raggedOffsetKv[b]) / args.seqStrideKv;
    const long long seqQ
        = static_cast<long long>(args.raggedOffsetQ[b + 1]) / args.seqStrideQ - qBase;
    const long long seqKv
        = static_cast<long long>(args.raggedOffsetKv[b + 1]) / args.seqStrideKv - kvBase;
    const long long sq = tokenGlobalQ - qBase; // within-batch query position

    // GQA/MQA: K and V head counts are independent.
    long long kvHeadK = h / (args.numHeads / args.numHeadsK);
    long long kvHeadV = h / (args.numHeads / args.numHeadsV);

    // Optional fp8 Q/K/V descale, constant per thread (b, h, kvHead* are fixed). 1 when absent.
    // Q is indexed by the Q head; K/V by the KV head. Applied as score *= descaleQ*descaleK and
    // output *= descaleV (AITER fp8 fwd contract: Q/K/V descale only, no softmax/output requant).
    const COMPUTE_TYPE descaleQ
        = args.descaleQ != nullptr
              ? args.descaleQ[b * args.descaleQBatchStride + h * args.descaleQHeadStride]
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

    // Per-batch sliding-window offset (matches CpuFpReferenceSdpa Step 3), using this
    // batch's own seqQ/seqKv rather than global lengths.
    long long windowOffset = args.topLeftAlignment ? 0 : (seqKv - seqQ);

    // Negative infinity sentinel for masked scores. INFINITY (a <math.h> macro)
    // is unavailable under HipRTC's self-contained preinclude, so use the clang
    // builtin (matches the __builtin_* idiom in GpuRefTypes.h).
    const COMPUTE_TYPE negInf = -__builtin_huge_valf();

    // Lambda computing the masked, scaled score for a single within-batch kv position.
    // Recomputed in both softmax passes (correctness over speed for a reference). Rank-4 BSHD
    // strides: s[2] = seq/token, s[1] = head, s[3] = dim.
    auto score = [&](long long skv) -> COMPUTE_TYPE {
        COMPUTE_TYPE dot = static_cast<COMPUTE_TYPE>(0);
        for(long long d = 0; d < args.headDim; ++d)
        {
            long long qIdx
                = tokenGlobalQ * args.qStr.s[2] + h * args.qStr.s[1] + d * args.qStr.s[3];
            long long kIdx
                = (kvBase + skv) * args.kStr.s[2] + kvHeadK * args.kStr.s[1] + d * args.kStr.s[3];
            dot += toAccum(q[qIdx]) * toAccum(k[kIdx]);
        }
        // descaleQK folds the fp8 Q/K dequant scalars into the score (both constant over d).
        COMPUTE_TYPE s = dot * descaleQK * static_cast<COMPUTE_TYPE>(args.scale);

        // Sliding-window mask, per-batch aligned. Asymmetric: +1 on the right bound,
        // none on the left bound. No additive bias (gated off on the ASM v3 path).
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

    // PASS 1: numerically stable softmax maximum over this batch's key range.
    COMPUTE_TYPE maxVal = negInf;
    for(long long skv = 0; skv < seqKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        if(s > maxVal)
        {
            maxVal = s;
        }
    }

    long long oIdx = tokenGlobalQ * args.oStr.s[2] + h * args.oStr.s[1] + dv * args.oStr.s[3];
    O_TYPE* tag = nullptr;

    // LSE is per (tokenGlobalQ, h); only the dv == 0 thread writes it, so every output
    // row has exactly one writer and there is no contention. Packed [B,H,Sq,1]:
    // seq stride = lseStr.s[2] (== H), head stride = lseStr.s[1] (== 1).
    long long lseIdx = tokenGlobalQ * args.lseStr.s[2] + h * args.lseStr.s[1];

    // Fully-masked row (no keys in range, incl. seqKv == 0): probabilities are all zero,
    // so the output is zero. Matches CpuFpReferenceSdpa (avoids a 0/0 NaN).
    if(maxVal == negInf)
    {
        o[oIdx] = fromAccum(static_cast<COMPUTE_TYPE>(0), tag);
        // CPU writes maxVal + log(sumExp) = -inf + log(0) = -inf for masked rows.
        if(lse != nullptr && dv == 0)
        {
            lse[lseIdx] = negInf;
        }
        return;
    }

    // PASS 2: softmax denominator.
    COMPUTE_TYPE sumExp = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < seqKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        // COMPUTE_TYPE is float (enforced by the static_assert above), so expf is the
        // correct-precision call; device expf and the oracle's host std::exp<float> agree
        // to within the test tolerance, not bit-for-bit.
        sumExp += expf(s - maxVal);
    }

    // PASS 3: weighted sum over V. Provider-attuned modes round the normalized
    // softmax probability before P@V, matching matrix-core SDPA kernels that
    // materialize P in bf16 before the second matmul.
    COMPUTE_TYPE weighted = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < seqKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        COMPUTE_TYPE probability = expf(s - maxVal) / sumExp;
        probability = storeSoftmaxProbability(probability);

        long long vIdx
            = (kvBase + skv) * args.vStr.s[2] + kvHeadV * args.vStr.s[1] + dv * args.vStr.s[3];
        weighted += probability * toAccum(v[vIdx]);
    }

    // descaleV folds the fp8 V dequant scalar into the output (constant over the P@V sum).
    weighted *= descaleV;

    o[oIdx] = fromAccum(weighted, tag);

    // LSE = maxVal + log(sumExp), matching CpuFpReferenceSdpa. sumExp is the
    // pre-normalization softmax denominator (>= 1, since exp(maxVal-maxVal)=1).
    if(lse != nullptr && dv == 0)
    {
        lse[lseIdx] = static_cast<float>(maxVal) + logf(sumExp);
    }
}
