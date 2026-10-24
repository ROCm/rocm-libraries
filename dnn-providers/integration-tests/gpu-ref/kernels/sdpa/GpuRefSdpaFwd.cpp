// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference scaled-dot-product-attention forward kernel.
// Compiled via HipRTC with -DQ_TYPE=<type> -DK_TYPE=<type> -DV_TYPE=<type>
// -DO_TYPE=<type> -DCOMPUTE_TYPE=<type> -DMASK_TYPE=<type>.
// MASK_TYPE is the dtype of the additive attention mask (float, half or bfloat16);
// it is converted to float on read and is independent of COMPUTE_TYPE.
// One thread per output element (b, h, sq, dv). Uses stride-based indexing.
// Default mode reproduces the fp32-softmax algorithm of CpuFpReferenceSdpa::forward.
// Provider-attuned modes can model lower-precision softmax-probability storage before P@V.
// FMA contraction is enabled so the matmuls round like provider asm kernels
// (fused multiply-add), which makes the GPU reference diverge from the CPU oracle by
// both FMA-vs-separate-multiply/add noise and host-libm-vs-device-math differences.

#include "GpuRefSdpaArgs.h"
#include "GpuRefSdpaCommon.hpp"
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

// cuDNN Max / Sum_exp outputs (CpuFpReferenceSdpa rowMax / rowSumExp), written by the dv == 0
// thread of each row like the LSE.
__device__ inline void writeRowStats(const SdpaFwdArgs& args,
                                     long long b,
                                     long long h,
                                     long long sq,
                                     long long dv,
                                     COMPUTE_TYPE maxVal,
                                     COMPUTE_TYPE sumExp)
{
    if(dv != 0)
    {
        return;
    }
    if(args.rowMax != nullptr)
    {
        static_cast<float*>(args.rowMax)[b * args.rowMaxStr.s[0] + h * args.rowMaxStr.s[1]
                                         + sq * args.rowMaxStr.s[2]]
            = static_cast<float>(maxVal);
    }
    if(args.rowSumExp != nullptr)
    {
        static_cast<float*>(args.rowSumExp)[b * args.rowSumExpStr.s[0] + h * args.rowSumExpStr.s[1]
                                            + sq * args.rowSumExpStr.s[2]]
            = static_cast<float>(sumExp);
    }
}

} // namespace

extern "C" __global__ void sdpaFwdRef(SdpaFwdArgs args)
{
    auto* q = static_cast<const Q_TYPE*>(args.q);
    auto* k = static_cast<const K_TYPE*>(args.k);
    auto* v = static_cast<const V_TYPE*>(args.v);
    auto* o = static_cast<O_TYPE*>(args.o);
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

    // Padding (cuDNN): this batch's valid lengths.
    const long long liveQ = sdpa_common::liveLength(args.seqLenQ, args.seqLenQStride, b, args.seqQ);
    const long long liveKv
        = sdpa_common::liveLength(args.seqLenKv, args.seqLenKvStride, b, args.seqKv);

    // Element offset of logical key `skv` (row base, without head and dim terms) in a dense or
    // paged K / V. Only called for skv < liveKv, so unused table entries are never read.
    auto rowOffset = [&](const int* table,
                         long long strB,
                         long long strT,
                         long long blockSize,
                         const SdpaStrides& str,
                         long long skv) -> long long {
        if(table == nullptr)
        {
            return b * str.s[0] + skv * str.s[2];
        }
        long long block = static_cast<long long>(table[b * strB + (skv / blockSize) * strT]);
        return block * str.s[0] + (skv % blockSize) * str.s[2];
    };

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
            long long kIdx = rowOffset(args.pageTableK,
                                       args.pageTableKStrB,
                                       args.pageTableKStrT,
                                       args.blockSizeK,
                                       args.kStr,
                                       skv)
                             + kvHeadK * args.kStr.s[1] + d * args.kStr.s[3];
            dot += toAccum(q[qIdx]) * toAccum(k[kIdx]);
        }
        // The shared score pipeline (scale, softcap, bias, ALiBi), then the band.
        if(sdpa_common::bandMasked(args.score, sq, skv, liveQ, liveKv))
        {
            return negInf;
        }
        return sdpa_common::scoreFromDot(args.score, dot, b, h, sq, skv, args.numHeads);
    };

    // PASS 1: numerically stable softmax maximum. Keys past liveKv are padding: never read.
    // An attention sink joins the max and the denominator but has no value row.
    const COMPUTE_TYPE sink
        = args.sink != nullptr ? static_cast<COMPUTE_TYPE>(args.sink[h * args.sinkStride]) : negInf;
    COMPUTE_TYPE maxVal = sink;
    for(long long skv = 0; skv < liveKv; ++skv)
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

    // Fully-masked row, or a padded query row past liveQ: probabilities are all zero, so the
    // output is zero and LSE is -inf. Matches CpuFpReferenceSdpa (avoids a 0/0 NaN).
    if(sq >= liveQ || maxVal == negInf)
    {
        o[oIdx] = fromAccum(static_cast<COMPUTE_TYPE>(0), tag);
        // CPU writes maxVal + log(sumExp) = -inf + log(0) = -inf for masked rows.
        if(lse != nullptr && dv == 0)
        {
            lse[lseIdx] = negInf;
        }
        writeRowStats(args, b, h, sq, dv, negInf, static_cast<COMPUTE_TYPE>(0));
        return;
    }

    // PASS 2: softmax denominator.
    COMPUTE_TYPE sumExp = sink == negInf ? static_cast<COMPUTE_TYPE>(0) : expf(sink - maxVal);
    for(long long skv = 0; skv < liveKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        // COMPUTE_TYPE is float (enforced by the static_assert above), so expf is the
        // correct-precision call; device expf and the oracle's host std::exp<float> agree
        // to within the test tolerance, not bit-for-bit.
        sumExp += expf(s - maxVal);
    }

    // PASS 3: weighted sum over V. Provider-attuned modes round the normalized
    // softmax probability before P@V, matching matrix-core SDPA kernels that
    // materialize P in bf16 before the second matmul. cuDNN sdpa_fp8 instead stores
    // exp(s - Max) * scaleS in fp8 and rescales by descaleS (CpuFpReferenceSdpa::forward).
    COMPUTE_TYPE weighted = static_cast<COMPUTE_TYPE>(0);
    COMPUTE_TYPE rowAmax = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < liveKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        COMPUTE_TYPE e = expf(s - maxVal);
        rowAmax = e > rowAmax ? e : rowAmax;
        if(args.pFp8Format != 0)
        {
            e = roundToFp8E4m3(e * args.scaleS, args.pFp8Format) * args.descaleS;
        }
        COMPUTE_TYPE probability = e / sumExp;
        probability = storeSoftmaxProbability(probability);
        if(args.dropoutMask != nullptr)
        {
            probability
                *= static_cast<COMPUTE_TYPE>(args.dropoutScale)
                   * sdpa_common::dropoutAt(args.dropoutMask, args.dropoutMaskStr, b, h, sq, skv);
        }

        long long vIdx = rowOffset(args.pageTableV,
                                   args.pageTableVStrB,
                                   args.pageTableVStrT,
                                   args.blockSizeV,
                                   args.vStr,
                                   skv)
                         + kvHeadV * args.vStr.s[1] + dv * args.vStr.s[3];
        weighted += probability * toAccum(v[vIdx]);
    }

    o[oIdx] = fromAccum(weighted, tag);
    // amaxS: e >= 0, so its float bits order like unsigned integers.
    if(args.amaxS != nullptr && dv == 0)
    {
        atomicMax(reinterpret_cast<unsigned int*>(args.amaxS),
                  __builtin_bit_cast(unsigned int, static_cast<float>(rowAmax)));
    }

    // LSE = maxVal + log(sumExp), matching CpuFpReferenceSdpa. sumExp is the
    // pre-normalization softmax denominator (>= 1, since exp(maxVal-maxVal)=1).
    if(lse != nullptr && dv == 0)
    {
        lse[lseIdx] = static_cast<float>(maxVal) + logf(sumExp);
    }
    writeRowStats(args, b, h, sq, dv, maxVal, sumExp);
}
