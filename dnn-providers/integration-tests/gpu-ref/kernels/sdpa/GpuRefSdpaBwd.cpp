// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// GPU reference scaled-dot-product-attention backward kernels.
// Compiled via HipRTC with -DQ_TYPE -DK_TYPE -DV_TYPE -DO_TYPE -DDO_TYPE -DDQ_TYPE -DDK_TYPE
// -DDV_TYPE -DMASK_TYPE -DCOMPUTE_TYPE (float).
//
// Mirrors CpuFpReferenceSdpa::backward exactly, without atomics: every output element is owned
// by one thread, which sums its terms in the CPU's (b, hQ, sq, skv) order.
//   sdpaBwdRowStatsRef  one thread per (b, h, sq): D = sum(dO * O), and the row's LSE (given)
//                       or recomputed max / exp-sum.
//   sdpaBwdDqRef        one thread per dQ element.
//   sdpaBwdDkRef        one thread per dK element (loops the query heads of its KV group).
//   sdpaBwdDvRef        one thread per dV element.
//   sdpaBwdDBiasRef     one thread per dBias element (loops the dims the bias broadcasts).
// Padding (seq_len_q / seq_len_kv) is never read; padded rows and keys get exactly 0. A row with
// no live key (LSE +-inf, or every score -inf) contributes nothing.

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

using namespace gpu_ref;

static_assert(__is_same(COMPUTE_TYPE, float), "GpuRefSdpaBwd requires COMPUTE_TYPE == float");

namespace
{

// cuDNN's ALiBi slope for query head h (CpuFpReferenceSdpa::alibiSlope): 2^(-8 (h + 1) / n) for
// the closest power of two n <= numHeads, then 2^(-4 (2 j + 1) / n) for the remaining heads.
__device__ inline float alibiSlope(long long h, long long numHeads)
{
    long long n = 1;
    while(n * 2 <= numHeads)
    {
        n *= 2;
    }
    if(h < n)
    {
        return exp2f(-8.0f * static_cast<float>(h + 1) / static_cast<float>(n));
    }
    return exp2f(-4.0f * static_cast<float>(2 * (h - n) + 1) / static_cast<float>(n));
}

__device__ inline long long
    liveLength(const int* lens, long long stride, long long b, long long full)
{
    if(lens == nullptr)
    {
        return full;
    }
    long long len = static_cast<long long>(lens[b * stride]);
    return len < 0 ? 0 : (len > full ? full : len);
}

// CpuFpReferenceSdpa::isMasked with the batch's valid lengths.
__device__ inline bool isMaskedBand(
    const SdpaBwdArgs& a, long long sq, long long skv, long long liveQ, long long liveKv)
{
    const long long offset = a.topLeftAlignment ? 0 : liveKv - liveQ;
    if(a.rightBound >= 0 && skv >= sq + 1 + offset + a.rightBound)
    {
        return true;
    }
    return a.leftBound >= 0 && skv <= sq + offset - a.leftBound;
}

__device__ inline long long rowIndex(const SdpaBwdArgs& a, long long b, long long h, long long sq)
{
    return (b * a.numHeads + h) * a.seqQ + sq;
}

// Scaled score plus additive bias for an unmasked live (sq, skv).
__device__ COMPUTE_TYPE
    score(const SdpaBwdArgs& a, long long b, long long h, long long sq, long long skv)
{
    const auto* q = static_cast<const Q_TYPE*>(a.q);
    const auto* k = static_cast<const K_TYPE*>(a.k);
    const long long kvHeadK = h / (a.numHeads / a.numHeadsK);
    COMPUTE_TYPE dot = static_cast<COMPUTE_TYPE>(0);
    for(long long d = 0; d < a.headDim; ++d)
    {
        dot += toAccum(q[b * a.qStr.s[0] + h * a.qStr.s[1] + sq * a.qStr.s[2] + d * a.qStr.s[3]])
               * toAccum(k[b * a.kStr.s[0] + kvHeadK * a.kStr.s[1] + skv * a.kStr.s[2]
                           + d * a.kStr.s[3]]);
    }
    COMPUTE_TYPE s = dot * static_cast<COMPUTE_TYPE>(a.scale);
    if(a.softcap > 0.0f)
    {
        s = a.softcap * tanhf(s / a.softcap);
    }
    if(a.mask != nullptr)
    {
        const auto* mask = static_cast<const MASK_TYPE*>(a.mask);
        long long ctx[4] = {b, h, sq, skv};
        long long offset = 0;
        for(int i = 0; i < a.maskRank; ++i)
        {
            long long idx = (a.maskDims[i] == 1) ? 0 : ctx[4 - a.maskRank + i];
            offset += idx * a.maskStr.s[i];
        }
        s += toAccum(mask[offset]);
    }
    if(a.alibi != 0)
    {
        s += static_cast<COMPUTE_TYPE>(skv - sq)
             * static_cast<COMPUTE_TYPE>(alibiSlope(h, a.numHeads));
    }
    return s;
}

// Softmax probability from the row's stats (LSE, or recomputed max and exp-sum).
__device__ COMPUTE_TYPE prob(const SdpaBwdArgs& a, long long row, COMPUTE_TYPE s)
{
    const COMPUTE_TYPE negInf = -__builtin_huge_valf();
    if(a.lse != nullptr)
    {
        const COMPUTE_TYPE lse = a.rowM[row];
        return (lse == negInf || lse == -negInf) ? static_cast<COMPUTE_TYPE>(0) : expf(s - lse);
    }
    if(s == negInf)
    {
        return static_cast<COMPUTE_TYPE>(0);
    }
    const COMPUTE_TYPE e = expf(s - a.rowM[row]);
    return a.rowL[row] > static_cast<COMPUTE_TYPE>(0) ? e / a.rowL[row] : e;
}

// dP = sum_dv dO * V for (b, h, sq, skv).
__device__ COMPUTE_TYPE
    dP(const SdpaBwdArgs& a, long long b, long long h, long long sq, long long skv)
{
    const auto* dO = static_cast<const DO_TYPE*>(a.dO);
    const auto* v = static_cast<const V_TYPE*>(a.v);
    const long long kvHeadV = h / (a.numHeads / a.numHeadsV);
    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);
    for(long long dv = 0; dv < a.headDimV; ++dv)
    {
        acc += toAccum(
                   dO[b * a.dOStr.s[0] + h * a.dOStr.s[1] + sq * a.dOStr.s[2] + dv * a.dOStr.s[3]])
               * toAccum(v[b * a.vStr.s[0] + kvHeadV * a.vStr.s[1] + skv * a.vStr.s[2]
                           + dv * a.vStr.s[3]]);
    }
    return acc;
}

// Derivative of the logit softcap at (b, h, sq, skv): 1 - tanh^2(s / c) for the scaled score s
// (before bias), or 1 without softcap. dQ and dK take it; dBias and dV do not.
__device__ COMPUTE_TYPE
    softcapGrad(const SdpaBwdArgs& a, long long b, long long h, long long sq, long long skv)
{
    if(!(a.softcap > 0.0f))
    {
        return static_cast<COMPUTE_TYPE>(1);
    }
    const auto* q = static_cast<const Q_TYPE*>(a.q);
    const auto* k = static_cast<const K_TYPE*>(a.k);
    const long long kvHeadK = h / (a.numHeads / a.numHeadsK);
    COMPUTE_TYPE dot = static_cast<COMPUTE_TYPE>(0);
    for(long long d = 0; d < a.headDim; ++d)
    {
        dot += toAccum(q[b * a.qStr.s[0] + h * a.qStr.s[1] + sq * a.qStr.s[2] + d * a.qStr.s[3]])
               * toAccum(k[b * a.kStr.s[0] + kvHeadK * a.kStr.s[1] + skv * a.kStr.s[2]
                           + d * a.kStr.s[3]]);
    }
    const COMPUTE_TYPE t = tanhf(dot * static_cast<COMPUTE_TYPE>(a.scale) / a.softcap);
    return static_cast<COMPUTE_TYPE>(1) - t * t;
}

// Explicit dropout mask value for (b, h, sq, skv); 1 without dropout.
__device__ inline COMPUTE_TYPE
    dropoutMaskAt(const SdpaBwdArgs& a, long long b, long long h, long long sq, long long skv)
{
    if(a.dropoutMask == nullptr)
    {
        return static_cast<COMPUTE_TYPE>(1);
    }
    return a.dropoutMask[b * a.dropoutMaskStr.s[0] + h * a.dropoutMaskStr.s[1]
                         + sq * a.dropoutMaskStr.s[2] + skv * a.dropoutMaskStr.s[3]];
}

// dS = dropoutScale * P * (mask * dP - D), before the attention scale, for an unmasked live
// (sq, skv). Without dropout this is P * (dP - D).
__device__ COMPUTE_TYPE
    dS(const SdpaBwdArgs& a, long long b, long long h, long long sq, long long skv)
{
    const long long row = rowIndex(a, b, h, sq);
    const COMPUTE_TYPE p = prob(a, row, score(a, b, h, sq, skv));
    return static_cast<COMPUTE_TYPE>(a.dropoutScale) * p
           * (dropoutMaskAt(a, b, h, sq, skv) * dP(a, b, h, sq, skv) - a.rowD[row]);
}

} // namespace

extern "C" __global__ void sdpaBwdRowStatsRef(SdpaBwdArgs a)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= a.batch * a.numHeads * a.seqQ)
    {
        return;
    }
    const long long sq = idx % a.seqQ;
    const long long h = (idx / a.seqQ) % a.numHeads;
    const long long b = idx / (a.seqQ * a.numHeads);
    const long long liveQ = liveLength(a.seqLenQ, a.seqLenQStride, b, a.seqQ);
    const long long liveKv = liveLength(a.seqLenKv, a.seqLenKvStride, b, a.seqKv);
    const COMPUTE_TYPE negInf = -__builtin_huge_valf();
    a.rowD[idx] = 0.0f;
    a.rowM[idx] = negInf;
    a.rowL[idx] = 0.0f;
    if(sq >= liveQ)
    {
        return;
    }
    const auto* dO = static_cast<const DO_TYPE*>(a.dO);
    const auto* o = static_cast<const O_TYPE*>(a.o);
    COMPUTE_TYPE dVal = static_cast<COMPUTE_TYPE>(0);
    for(long long dv = 0; dv < a.headDimV; ++dv)
    {
        dVal += toAccum(
                    dO[b * a.dOStr.s[0] + h * a.dOStr.s[1] + sq * a.dOStr.s[2] + dv * a.dOStr.s[3]])
                * toAccum(
                    o[b * a.oStr.s[0] + h * a.oStr.s[1] + sq * a.oStr.s[2] + dv * a.oStr.s[3]]);
    }
    a.rowD[idx] = dVal * static_cast<COMPUTE_TYPE>(a.dropoutScaleInv);
    if(a.lse != nullptr)
    {
        a.rowM[idx] = a.lse[b * a.lseStr.s[0] + h * a.lseStr.s[1] + sq * a.lseStr.s[2]];
        return;
    }
    COMPUTE_TYPE maxVal = negInf;
    for(long long skv = 0; skv < liveKv; ++skv)
    {
        if(!isMaskedBand(a, sq, skv, liveQ, liveKv))
        {
            const COMPUTE_TYPE s = score(a, b, h, sq, skv);
            maxVal = s > maxVal ? s : maxVal;
        }
    }
    COMPUTE_TYPE sumExp = static_cast<COMPUTE_TYPE>(0);
    for(long long skv = 0; skv < liveKv; ++skv)
    {
        if(!isMaskedBand(a, sq, skv, liveQ, liveKv))
        {
            const COMPUTE_TYPE s = score(a, b, h, sq, skv);
            if(s != negInf)
            {
                sumExp += expf(s - maxVal);
            }
        }
    }
    a.rowM[idx] = maxVal;
    a.rowL[idx] = sumExp;
}

extern "C" __global__ void sdpaBwdDqRef(SdpaBwdArgs a)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= a.batch * a.numHeads * a.seqQ * a.headDim)
    {
        return;
    }
    const long long d = idx % a.headDim;
    long long t = idx / a.headDim;
    const long long sq = t % a.seqQ;
    t /= a.seqQ;
    const long long h = t % a.numHeads;
    const long long b = t / a.numHeads;
    const long long liveQ = liveLength(a.seqLenQ, a.seqLenQStride, b, a.seqQ);
    const long long liveKv = liveLength(a.seqLenKv, a.seqLenKvStride, b, a.seqKv);
    const auto* k = static_cast<const K_TYPE*>(a.k);
    const long long kvHeadK = h / (a.numHeads / a.numHeadsK);
    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);
    if(sq < liveQ)
    {
        for(long long skv = 0; skv < liveKv; ++skv)
        {
            if(isMaskedBand(a, sq, skv, liveQ, liveKv))
            {
                continue;
            }
            acc += dS(a, b, h, sq, skv) * softcapGrad(a, b, h, sq, skv)
                   * static_cast<COMPUTE_TYPE>(a.scale)
                   * toAccum(k[b * a.kStr.s[0] + kvHeadK * a.kStr.s[1] + skv * a.kStr.s[2]
                               + d * a.kStr.s[3]]);
        }
    }
    DQ_TYPE* dQ = static_cast<DQ_TYPE*>(a.dQ);
    dQ[b * a.dQStr.s[0] + h * a.dQStr.s[1] + sq * a.dQStr.s[2] + d * a.dQStr.s[3]]
        = fromAccum(acc, dQ);
}

extern "C" __global__ void sdpaBwdDkRef(SdpaBwdArgs a)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= a.batch * a.numHeadsK * a.seqKv * a.headDim)
    {
        return;
    }
    const long long d = idx % a.headDim;
    long long t = idx / a.headDim;
    const long long skv = t % a.seqKv;
    t /= a.seqKv;
    const long long hk = t % a.numHeadsK;
    const long long b = t / a.numHeadsK;
    const long long liveQ = liveLength(a.seqLenQ, a.seqLenQStride, b, a.seqQ);
    const long long liveKv = liveLength(a.seqLenKv, a.seqLenKvStride, b, a.seqKv);
    const long long group = a.numHeads / a.numHeadsK;
    const auto* q = static_cast<const Q_TYPE*>(a.q);
    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);
    if(skv < liveKv)
    {
        for(long long h = hk * group; h < (hk + 1) * group; ++h)
        {
            for(long long sq = 0; sq < liveQ; ++sq)
            {
                if(isMaskedBand(a, sq, skv, liveQ, liveKv))
                {
                    continue;
                }
                acc += dS(a, b, h, sq, skv) * softcapGrad(a, b, h, sq, skv)
                       * static_cast<COMPUTE_TYPE>(a.scale)
                       * toAccum(q[b * a.qStr.s[0] + h * a.qStr.s[1] + sq * a.qStr.s[2]
                                   + d * a.qStr.s[3]]);
            }
        }
    }
    DK_TYPE* dK = static_cast<DK_TYPE*>(a.dK);
    dK[b * a.dKStr.s[0] + hk * a.dKStr.s[1] + skv * a.dKStr.s[2] + d * a.dKStr.s[3]]
        = fromAccum(acc, dK);
}

extern "C" __global__ void sdpaBwdDvRef(SdpaBwdArgs a)
{
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= a.batch * a.numHeadsV * a.seqKv * a.headDimV)
    {
        return;
    }
    const long long dv = idx % a.headDimV;
    long long t = idx / a.headDimV;
    const long long skv = t % a.seqKv;
    t /= a.seqKv;
    const long long hv = t % a.numHeadsV;
    const long long b = t / a.numHeadsV;
    const long long liveQ = liveLength(a.seqLenQ, a.seqLenQStride, b, a.seqQ);
    const long long liveKv = liveLength(a.seqLenKv, a.seqLenKvStride, b, a.seqKv);
    const long long group = a.numHeads / a.numHeadsV;
    const auto* dO = static_cast<const DO_TYPE*>(a.dO);
    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);
    if(skv < liveKv)
    {
        for(long long h = hv * group; h < (hv + 1) * group; ++h)
        {
            for(long long sq = 0; sq < liveQ; ++sq)
            {
                if(isMaskedBand(a, sq, skv, liveQ, liveKv))
                {
                    continue;
                }
                const COMPUTE_TYPE p = prob(a, rowIndex(a, b, h, sq), score(a, b, h, sq, skv))
                                       * static_cast<COMPUTE_TYPE>(a.dropoutScale)
                                       * dropoutMaskAt(a, b, h, sq, skv);
                acc += p
                       * toAccum(dO[b * a.dOStr.s[0] + h * a.dOStr.s[1] + sq * a.dOStr.s[2]
                                    + dv * a.dOStr.s[3]]);
            }
        }
    }
    DV_TYPE* dV = static_cast<DV_TYPE*>(a.dV);
    dV[b * a.dVStr.s[0] + hv * a.dVStr.s[1] + skv * a.dVStr.s[2] + dv * a.dVStr.s[3]]
        = fromAccum(acc, dV);
}

extern "C" __global__ void sdpaBwdDBiasRef(SdpaBwdArgs a)
{
    // Element of the bias tensor (rank maskRank, right-aligned onto (b, h, sq, skv)).
    long long count = 1;
    for(int i = 0; i < a.maskRank; ++i)
    {
        count *= a.maskDims[i];
    }
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if(idx >= count)
    {
        return;
    }
    long long maskIdx[4] = {0, 0, 0, 0};
    long long rem = idx;
    for(int i = a.maskRank - 1; i >= 0; --i)
    {
        maskIdx[i] = rem % a.maskDims[i];
        rem /= a.maskDims[i];
    }
    // Per context dim: a fixed index, or the full range when the bias broadcasts over it.
    const long long extent[4] = {a.batch, a.numHeads, a.seqQ, a.seqKv};
    long long lo[4];
    long long hi[4];
    for(int c = 0; c < 4; ++c)
    {
        const int m = c - (4 - a.maskRank);
        if(m >= 0 && a.maskDims[m] != 1)
        {
            lo[c] = maskIdx[m];
            hi[c] = maskIdx[m] + 1;
        }
        else
        {
            lo[c] = 0;
            hi[c] = extent[c];
        }
    }
    COMPUTE_TYPE acc = static_cast<COMPUTE_TYPE>(0);
    for(long long b = lo[0]; b < hi[0]; ++b)
    {
        const long long liveQ = liveLength(a.seqLenQ, a.seqLenQStride, b, a.seqQ);
        const long long liveKv = liveLength(a.seqLenKv, a.seqLenKvStride, b, a.seqKv);
        for(long long h = lo[1]; h < hi[1]; ++h)
        {
            for(long long sq = lo[2]; sq < hi[2] && sq < liveQ; ++sq)
            {
                for(long long skv = lo[3]; skv < hi[3] && skv < liveKv; ++skv)
                {
                    if(!isMaskedBand(a, sq, skv, liveQ, liveKv))
                    {
                        acc += dS(a, b, h, sq, skv);
                    }
                }
            }
        }
    }
    MASK_TYPE* dBias = static_cast<MASK_TYPE*>(a.dBias);
    long long offset = 0;
    for(int i = 0; i < a.maskRank; ++i)
    {
        offset += maskIdx[i] * a.dBiasStr.s[i];
    }
    dBias[offset] = fromAccum(acc, dBias);
}
