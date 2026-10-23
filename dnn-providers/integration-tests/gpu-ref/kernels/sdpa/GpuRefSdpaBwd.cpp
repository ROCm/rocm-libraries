// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference scaled-dot-product-attention backward kernel.
// Compiled via HipRTC with -DQ_TYPE=<type> -DK_TYPE=<type> -DV_TYPE=<type>
// -DO_TYPE=<type> -DDO_TYPE=<type> -DDQ_TYPE=<type> -DDK_TYPE=<type> -DDV_TYPE=<type>
// -DCOMPUTE_TYPE=<type> -DMASK_TYPE=<type>.

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

using namespace gpu_ref;

// The kernel computes in float: expf, -__builtin_huge_valf(), and the std::exp-matching
// softmax all assume it. COMPUTE_TYPE is float by design (see buildSdpaBwdDefines); enforce
// it so a non-float compute path fails loudly at compile time instead of silently truncating.
static_assert(__is_same(COMPUTE_TYPE, float), "GpuRefSdpaBwd requires COMPUTE_TYPE == float");

// Negative infinity sentinel for masked scores. INFINITY (a <math.h> macro)
// is unavailable under HipRTC's self-contained preinclude, so use the clang
// builtin (matches the __builtin_* idiom in GpuRefTypes.h).
const COMPUTE_TYPE negInf = -__builtin_huge_valf();

// Computes the masked score for a single (b, h, sq, skv) position
__device__ inline COMPUTE_TYPE computeScore(const Q_TYPE* q,
                                            const K_TYPE* k,
                                            const MASK_TYPE* mask,
                                            long long b,
                                            long long h,
                                            long long sq,
                                            long long skv,
                                            const SdpaBwdArgs& args)
{
    // GQA/MQA: K head count is independent.
    long long kvHeadK = h / (args.numHeads / args.numHeadsK);

    // Sliding-window offset
    long long windowOffset = args.topLeftAlignment ? 0 : (args.seqKv - args.seqQ);

    // Compute Q·K^T dot product
    COMPUTE_TYPE dot = static_cast<COMPUTE_TYPE>(0);
    for(long long d = 0; d < args.headDim; ++d)
    {
        long long qIdx
            = b * args.qStr.s[0] + h * args.qStr.s[1] + sq * args.qStr.s[2] + d * args.qStr.s[3];
        long long kIdx = b * args.kStr.s[0] + kvHeadK * args.kStr.s[1] + skv * args.kStr.s[2]
                         + d * args.kStr.s[3];
        dot += toAccum(q[qIdx]) * toAccum(k[kIdx]);
    }
    COMPUTE_TYPE s = dot * static_cast<COMPUTE_TYPE>(args.scale);

    // Additive attention mask (right-aligned, broadcast on size-1 dims).
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

    // Sliding-window mask (applied after the additive mask).
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
}

// Computes the log-sum-exp for a single query position
__device__ inline COMPUTE_TYPE computeRowLse(const Q_TYPE* q,
                                             const K_TYPE* k,
                                             const MASK_TYPE* mask,
                                             const float* lse,
                                             long long b,
                                             long long h,
                                             long long sq,
                                             const SdpaBwdArgs& args)
{
    if(lse != nullptr)
    {
        long long lseIdx = b * args.lseStr.s[0] + h * args.lseStr.s[1] + sq * args.lseStr.s[2];
        return lse[lseIdx];
    }

    // PASS 1: numerically stable softmax maximum.
    COMPUTE_TYPE maxVal = negInf;
    for(long long skv = 0; skv < args.seqKv; ++skv)
    {
        COMPUTE_TYPE s = computeScore(args, q, k, mask, b, h, sq, skv);
        if(s > maxVal)
        {
            maxVal = s;
        }
    }

    // Fully-masked row: no valid keys, so every probability is 0.
    if(maxVal == negInf)
    {
        return negInf;
    }

    // PASS 2: softmax denominator.
    COMPUTE_TYPE sumExp = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < args.seqKv; ++skv)
    {
        COMPUTE_TYPE s = computeScore(args, q, k, mask, b, h, sq, skv);
        sumExp += expf(s - maxVal);
    }

    return maxVal + logf(sumExp);
}

// Computes the probability for a single kv position
__device__ inline COMPUTE_TYPE computeProbability(COMPUTE_TYPE s, COMPUTE_TYPE rowLse)
{
    if(s == negInf || rowLse == negInf)
    {
        return static_cast<COMPUTE_TYPE>(0);
    }
    return expf(s - rowLse);
}

// Computes the dS for a single (b, h, sq, skv) position
__device__ inline COMPUTE_TYPE computedS(COMPUTE_TYPE p,
                                         const DO_TYPE* dO,
                                         const V_TYPE* v,
                                         const COMPUTE_TYPE* dBuff,
                                         long long b,
                                         long long h,
                                         long long sq,
                                         long long skv,
                                         const SdpaBwdArgs& args)
{
    // GQA/MQA: V head count is independent.
    long long kvHeadV = h / (args.numHeads / args.numHeadsV);

    // dP = dO @ V^T
    COMPUTE_TYPE dP = static_cast<COMPUTE_TYPE>(0);
    for(long long dv = 0; dv < args.headDimV; ++dv)
    {
        long long dOIdx = b * args.dOStr.s[0] + h * args.dOStr.s[1] + sq * args.dOStr.s[2]
                          + dv * args.dOStr.s[3];
        long long vIdx = b * args.vStr.s[0] + kvHeadV * args.vStr.s[1] + skv * args.vStr.s[2]
                         + dv * args.vStr.s[3];
        dP += toAccum(dO[dOIdx]) * toAccum(v[vIdx]);
    }

    long long dBuffIdx = b * args.dBuffStr.s[0] + h * args.dBuffStr.s[1] + sq * args.dBuffStr.s[2];

    // dS = P * (dP - D)
    return p * (dP - dBuff[dBuffIdx]);
}

// Each thread computes the dBuff value for a single (b, h, sq) position.
// dBuff is the sum of dO[b, h, sq, dv] * o[b, h, sq, dv] over dv.
extern "C" __global__ void sdpaBwdRefD(SdpaBwdArgs args)
{
    auto* o = static_cast<const O_TYPE*>(args.o);
    auto* dO = static_cast<const DO_TYPE*>(args.dO);
    auto* dBuff = static_cast<COMPUTE_TYPE*>(args.dBuff);

    long long totalElements = args.batch * args.numHeads * args.seqQ;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (b, h, sq)
    long long sq = idx % args.seqQ;
    long long tmp = idx / args.seqQ;
    long long h = tmp % args.numHeads;
    long long b = tmp / args.numHeads;

    COMPUTE_TYPE dVal = static_cast<COMPUTE_TYPE>(0);
    for(long long dv = 0; dv < args.headDimV; ++dv)
    {
        long long oIdx
            = b * args.oStr.s[0] + h * args.oStr.s[1] + sq * args.oStr.s[2] + dv * args.oStr.s[3];
        long long dOIdx = b * args.dOStr.s[0] + h * args.dOStr.s[1] + sq * args.dOStr.s[2]
                          + dv * args.dOStr.s[3];
        dVal += toAccum(dO[dOIdx]) * toAccum(o[oIdx]);
    }

    long long dBuffIdx = b * args.dBuffStr.s[0] + h * args.dBuffStr.s[1] + sq * args.dBuffStr.s[2];
    dBuff[dBuffIdx] = dVal;
}

// Each thread computes the dQ value for a single (b, h, sq, d) position.
// dQ = scale * sum_kv dS[b, h, sq, kv] * K[b, kvHeadK, kv, d].
// Call this kernel only after sdpaBwdRefD to compute the dBuff first!
extern "C" __global__ void sdpaBwdRefdQ(SdpaBwdArgs args)
{
    auto* q = static_cast<const Q_TYPE*>(args.q);
    auto* k = static_cast<const K_TYPE*>(args.k);
    auto* v = static_cast<const V_TYPE*>(args.v);
    auto* dO = static_cast<const DO_TYPE*>(args.dO);
    auto* dQ = static_cast<DQ_TYPE*>(args.dQ);
    auto* mask = static_cast<const MASK_TYPE*>(args.mask);
    auto* lse = static_cast<const float*>(args.lse);
    auto* dBuff = static_cast<const COMPUTE_TYPE*>(args.dBuff);

    long long totalElements = args.batch * args.numHeads * args.seqQ * args.headDim;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (b, h, sq, d)
    long long d = idx % args.headDim;
    long long tmp = idx / args.headDim;
    long long sq = tmp % args.seqQ;
    tmp = tmp / args.seqQ;
    long long h = tmp % args.numHeads;
    long long b = tmp / args.numHeads;

    // GQA/MQA: K head count is independent.
    long long kvHeadK = h / (args.numHeads / args.numHeadsK);

    COMPUTE_TYPE rowLse = computeRowLse(q, k, mask, lse, b, h, sq, args);

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < args.seqKv; ++skv)
    {
        COMPUTE_TYPE s = computeScore(q, k, mask, b, h, sq, skv, args);
        COMPUTE_TYPE p = computeProbability(s, rowLse);
        if(p == static_cast<COMPUTE_TYPE>(0))
        {
            continue; // dS = 0 for masked positions
        }

        COMPUTE_TYPE dS = computedS(p, dO, v, dBuff, b, h, sq, skv, args);

        long long kIdx = b * args.kStr.s[0] + kvHeadK * args.kStr.s[1] + skv * args.kStr.s[2]
                         + d * args.kStr.s[3];
        acc += dS * toAccum(k[kIdx]);
    }

    DQ_TYPE* tag = nullptr;
    long long dQIdx
        = b * args.dQStr.s[0] + h * args.dQStr.s[1] + sq * args.dQStr.s[2] + d * args.dQStr.s[3];
    dQ[dQIdx] = fromAccum(acc * static_cast<COMPUTE_TYPE>(args.scale), tag);
}

// Each thread computes the dK value for a single (b, hK, skv, d) position.
// dK = scale * sum_q dS[b, h, sq, skv] * Q[b, h, sq, d].
// All Q heads in [hK * headsPerK, (hK + 1) * headsPerK) share this K head,
// so their contributions are summed for the same dK entry.
// Call this kernel only after sdpaBwdRefD to compute the dBuff first!
extern "C" __global__ void sdpaBwdRefdK(SdpaBwdArgs args)
{
    auto* q = static_cast<const Q_TYPE*>(args.q);
    auto* k = static_cast<const K_TYPE*>(args.k);
    auto* v = static_cast<const V_TYPE*>(args.v);
    auto* dO = static_cast<const DO_TYPE*>(args.dO);
    auto* dK = static_cast<DK_TYPE*>(args.dK);
    auto* mask = static_cast<const MASK_TYPE*>(args.mask);
    auto* lse = static_cast<const float*>(args.lse);
    auto* dBuff = static_cast<const COMPUTE_TYPE*>(args.dBuff);

    long long totalElements = args.batch * args.numHeadsK * args.seqKv * args.headDim;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (b, hK, skv, d)
    long long d = idx % args.headDim;
    long long tmp = idx / args.headDim;
    long long skv = tmp % args.seqKv;
    tmp = tmp / args.seqKv;
    long long hK = tmp % args.numHeadsK;
    long long b = tmp / args.numHeadsK;

    // GQA/MQA: all Q heads in [hK * headsPerK, (hK + 1) * headsPerK) share this K head.
    long long headsPerK = args.numHeads / args.numHeadsK;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);
    for(long long h = hK * headsPerK; h < (hK + 1) * headsPerK; ++h)
    {
        for(long long sq = 0; sq < args.seqQ; ++sq)
        {
            COMPUTE_TYPE rowLse = computeRowLse(q, k, mask, lse, b, h, sq, args);
            COMPUTE_TYPE s = computeScore(q, k, mask, b, h, sq, skv, args);
            COMPUTE_TYPE p = computeProbability(s, rowLse);
            if(p == static_cast<COMPUTE_TYPE>(0))
            {
                continue; // dS = 0 for masked positions
            }

            COMPUTE_TYPE dS = computedS(p, dO, v, dBuff, b, h, sq, skv, args);

            long long qIdx = b * args.qStr.s[0] + h * args.qStr.s[1] + sq * args.qStr.s[2]
                             + d * args.qStr.s[3];
            acc += dS * toAccum(q[qIdx]);
        }
    }

    DK_TYPE* tag = nullptr;
    long long dKIdx
        = b * args.dKStr.s[0] + hK * args.dKStr.s[1] + skv * args.dKStr.s[2] + d * args.dKStr.s[3];
    dK[dKIdx] = fromAccum(acc * static_cast<COMPUTE_TYPE>(args.scale), tag);
}

// Each thread computes the dV value for a single (b, hV, skv, dv) position.
// dV = sum_q P[b, h, sq, skv] * dO[b, h, sq, dv].
// All Q heads in [hV * headsPerV, (hV + 1) * headsPerV) share this V head,
// so their contributions are summed for the same dV entry.
extern "C" __global__ void sdpaBwdRefdV(SdpaBwdArgs args)
{
    auto* q = static_cast<const Q_TYPE*>(args.q);
    auto* k = static_cast<const K_TYPE*>(args.k);
    auto* dO = static_cast<const DO_TYPE*>(args.dO);
    auto* dV = static_cast<DV_TYPE*>(args.dV);
    auto* mask = static_cast<const MASK_TYPE*>(args.mask);
    auto* lse = static_cast<const float*>(args.lse);

    long long totalElements = args.batch * args.numHeadsV * args.seqKv * args.headDimV;
    long long idx = static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x)
                    + static_cast<long long>(threadIdx.x);
    if(idx >= totalElements)
    {
        return;
    }

    // Decompose linear index into (b, hV, skv, dv)
    long long dv = idx % args.headDimV;
    long long tmp = idx / args.headDimV;
    long long skv = tmp % args.seqKv;
    tmp = tmp / args.seqKv;
    long long hV = tmp % args.numHeadsV;
    long long b = tmp / args.numHeadsV;

    // GQA/MQA: all Q heads in [hV * headsPerV, (hV + 1) * headsPerV) share this V head.
    long long headsPerV = args.numHeads / args.numHeadsV;

    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);
    for(long long h = hV * headsPerV; h < (hV + 1) * headsPerV; ++h)
    {
        for(long long sq = 0; sq < args.seqQ; ++sq)
        {
            COMPUTE_TYPE rowLse = computeRowLse(q, k, mask, lse, b, h, sq, args);
            COMPUTE_TYPE s = computeScore(q, k, mask, b, h, sq, skv, args);
            COMPUTE_TYPE p = computeProbability(s, rowLse);
            if(p == static_cast<COMPUTE_TYPE>(0))
            {
                continue; // dS = 0 for masked positions
            }

            long long dOIdx = b * args.dOStr.s[0] + h * args.dOStr.s[1] + sq * args.dOStr.s[2]
                              + dv * args.dOStr.s[3];
            acc += p * toAccum(dO[dOIdx]);
        }
    }

    DV_TYPE* tag = nullptr;
    long long dVIdx
        = b * args.dVStr.s[0] + hV * args.dVStr.s[1] + skv * args.dVStr.s[2] + dv * args.dVStr.s[3];
    dV[dVIdx] = fromAccum(acc, tag);
}
