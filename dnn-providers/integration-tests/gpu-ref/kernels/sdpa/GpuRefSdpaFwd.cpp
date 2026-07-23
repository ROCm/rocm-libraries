// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference scaled-dot-product-attention forward kernel.
// Compiled via HipRTC with -DQ_TYPE=<type> -DK_TYPE=<type> -DV_TYPE=<type>
// -DO_TYPE=<type> -DCOMPUTE_TYPE=<type>.
// One thread per output element (b, h, sq, dv). Uses stride-based indexing.
// Default mode reproduces the fp32-softmax algorithm of CpuFpReferenceSdpa::forward.
// Provider-attuned modes can model lower-precision softmax-probability storage before P@V.
// FMA contraction is enabled so the matmuls round like provider asm kernels
// (fused multiply-add), which makes the GPU reference diverge from the CPU oracle by
// both FMA-vs-separate-multiply/add noise and host-libm-vs-device-math differences.

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

using namespace gpu_ref;

// The kernel computes in float: expf, -__builtin_huge_valf(), and the std::exp-matching
// softmax all assume it. COMPUTE_TYPE is float by design (see buildSdpaDefines); enforce it
// so a non-float compute path fails loudly at compile time instead of silently truncating.
static_assert(__is_same(COMPUTE_TYPE, float), "GpuRefSdpaFwd requires COMPUTE_TYPE == float");

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

extern "C" __global__ void sdpaFwdRef(SdpaFwdArgs args)
{
    auto* q = static_cast<const Q_TYPE*>(args.q);
    auto* k = static_cast<const K_TYPE*>(args.k);
    auto* v = static_cast<const V_TYPE*>(args.v);
    auto* o = static_cast<O_TYPE*>(args.o);
    auto* mask = static_cast<const COMPUTE_TYPE*>(args.mask);
    // LSE is always float [B, H, Sq]; nullptr disables it. Written once per
    // (b, h, sq) by the dv == 0 thread (see below).
    auto* lse = static_cast<float*>(args.lse);

    long long totalOutputElements = args.batch * args.numHeads * args.seqQ * args.headDimV;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalOutputElements)
    {
        return;
    }

    // Decompose linear index into (b, h, sq, dv)
    long long dv = idx % args.headDimV;
    long long tmp = idx / args.headDimV;
    long long sq = tmp % args.seqQ;
    tmp = tmp / args.seqQ;
    long long h = tmp % args.numHeads;
    long long b = tmp / args.numHeads;

    // GQA/MQA: K and V head counts are independent.
    long long kvHeadK = h / (args.numHeads / args.numHeadsK);
    long long kvHeadV = h / (args.numHeads / args.numHeadsV);

    // Sliding-window offset (matches CpuFpReferenceSdpa Step 3).
    long long windowOffset = args.topLeftAlignment ? 0 : (args.seqKv - args.seqQ);

    // Negative infinity sentinel for masked scores. INFINITY (a <math.h> macro)
    // is unavailable under HipRTC's self-contained preinclude, so use the clang
    // builtin (matches the __builtin_* idiom in GpuRefTypes.h).
    const COMPUTE_TYPE negInf = -__builtin_huge_valf();

    // Lambda computing the masked, scaled score for a single kv position.
    // Recomputed in both softmax passes (correctness over speed for a reference).
    auto score = [&](long long skv) -> COMPUTE_TYPE {
        COMPUTE_TYPE dot = static_cast<COMPUTE_TYPE>(0);
        for(long long d = 0; d < args.headDim; ++d)
        {
            long long qIdx = b * args.qStr.s[0] + h * args.qStr.s[1] + sq * args.qStr.s[2]
                             + d * args.qStr.s[3];
            long long kIdx = b * args.kStr.s[0] + kvHeadK * args.kStr.s[1] + skv * args.kStr.s[2]
                             + d * args.kStr.s[3];
            dot += toAccum(q[qIdx]) * toAccum(k[kIdx]);
        }
        COMPUTE_TYPE s = dot * static_cast<COMPUTE_TYPE>(args.scale);

        // (a) Additive attention mask (right-aligned, broadcast on size-1 dims).
        if(mask != nullptr)
        {
            long long ctxIdxs[4] = {b, h, sq, skv};
            long long maskOffset = 0;
            for(int i = 0; i < args.maskRank; ++i)
            {
                long long ctxIdx = ctxIdxs[4 - args.maskRank + i];
                long long idxI = (args.maskDims[i] == 1) ? 0 : ctxIdx;
                maskOffset += idxI * args.maskStr.s[i];
            }
            s += toAccum(mask[maskOffset]);
        }

        // (b) Sliding-window mask (applied after the additive mask).
        // Asymmetric: +1 on the right bound, none on the left bound.
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

    // PASS 1: numerically stable softmax maximum.
    COMPUTE_TYPE maxVal = negInf;
    for(long long skv = 0; skv < args.seqKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        if(s > maxVal)
        {
            maxVal = s;
        }
    }

    long long oIdx
        = b * args.oStr.s[0] + h * args.oStr.s[1] + sq * args.oStr.s[2] + dv * args.oStr.s[3];
    O_TYPE* tag = nullptr;

    // LSE is per (b, h, sq); only the dv == 0 thread writes it, so every output
    // row has exactly one writer and there is no contention.
    long long lseIdx = b * args.lseStr.s[0] + h * args.lseStr.s[1] + sq * args.lseStr.s[2];

    // Fully-masked row: probabilities are all zero, so the output is zero.
    // Matches CpuFpReferenceSdpa (avoids a 0/0 NaN from a sumExp==0 guard).
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
    for(long long skv = 0; skv < args.seqKv; ++skv)
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
    for(long long skv = 0; skv < args.seqKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        COMPUTE_TYPE probability = expf(s - maxVal) / sumExp;
        probability = storeSoftmaxProbability(probability);

        long long vIdx = b * args.vStr.s[0] + kvHeadV * args.vStr.s[1] + skv * args.vStr.s[2]
                         + dv * args.vStr.s[3];
        weighted += probability * toAccum(v[vIdx]);
    }

    o[oIdx] = fromAccum(weighted, tag);

    // LSE = maxVal + log(sumExp), matching CpuFpReferenceSdpa. sumExp is the
    // pre-normalization softmax denominator (>= 1, since exp(maxVal-maxVal)=1).
    if(lse != nullptr && dv == 0)
    {
        lse[lseIdx] = static_cast<float>(maxVal) + logf(sumExp);
    }
}
