// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference scaled-dot-product-attention forward kernel.
// Compiled via HipRTC with -DQ_TYPE=<type> -DK_TYPE=<type> -DV_TYPE=<type>
// -DO_TYPE=<type> -DCOMPUTE_TYPE=<type>.
// One thread per output element (b, h, sq, dv). Uses stride-based indexing.
// Numerically mirrors CpuFpReferenceSdpa::forward (float compute path).

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

using namespace gpu_ref;

extern "C" __global__ void sdpaFwdRef(SdpaFwdArgs args)
{
    auto* q = static_cast<const Q_TYPE*>(args.q);
    auto* k = static_cast<const K_TYPE*>(args.k);
    auto* v = static_cast<const V_TYPE*>(args.v);
    auto* o = static_cast<O_TYPE*>(args.o);
    auto* mask = static_cast<const COMPUTE_TYPE*>(args.mask);

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

    // Fully-masked row: probabilities are all zero, so the output is zero.
    // Matches CpuFpReferenceSdpa (avoids a 0/0 NaN from a sumExp==0 guard).
    if(maxVal == negInf)
    {
        o[oIdx] = fromAccum(static_cast<COMPUTE_TYPE>(0), tag);
        return;
    }

    // PASS 2: weighted sum over V.
    COMPUTE_TYPE sumExp = static_cast<COMPUTE_TYPE>(0);
    COMPUTE_TYPE weighted = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < args.seqKv; ++skv)
    {
        COMPUTE_TYPE s = score(skv);
        // COMPUTE_TYPE is float by design (see buildSdpaDefines), so the
        // float-precision expf matches the oracle's std::exp<float> exactly.
        COMPUTE_TYPE e = expf(s - maxVal);
        sumExp += e;
        long long vIdx = b * args.vStr.s[0] + kvHeadV * args.vStr.s[1] + skv * args.vStr.s[2]
                         + dv * args.vStr.s[3];
        weighted += e * toAccum(v[vIdx]);
    }

    o[oIdx] = fromAccum(weighted / sumExp, tag);
}
