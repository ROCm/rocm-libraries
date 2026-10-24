// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The SDPA score pipeline the forward and backward kernels share, so each feature is written once
// (the device counterpart of CpuFpReferenceSdpa::scoreRow). Include after the kernel's
// MASK_TYPE / COMPUTE_TYPE defines. For key skv of row (b, h, sq), in order:
// (q . k) * scale, softcap, + bias, + ALiBi; then the diagonal band masks it to -inf.

#pragma once

#include "GpuRefSdpaArgs.h"
#include "GpuRefTypes.h"

namespace sdpa_common
{

// This batch's valid length (cuDNN seq_len_q / seq_len_kv). Lengths outside [0, full] are an
// invalid graph that the CPU reference rejects; a kernel cannot throw, so it clamps.
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

// CpuFpReferenceSdpa::isMasked with the batch's valid lengths (cuDNN band, SEM-001):
// leftBound = L keeps L keys up to and including the diagonal, rightBound = R keeps R past it.
__device__ inline bool bandMasked(
    const SdpaScoreParams& p, long long sq, long long skv, long long liveQ, long long liveKv)
{
    const long long offset = p.topLeftAlignment ? 0 : liveKv - liveQ;
    if(p.rightBound >= 0 && skv >= sq + 1 + offset + p.rightBound)
    {
        return true;
    }
    return p.leftBound >= 0 && skv <= sq + offset - p.leftBound;
}

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

// The score of an unmasked live key from its dot product: scale, softcap, + bias, + ALiBi.
__device__ inline COMPUTE_TYPE scoreFromDot(const SdpaScoreParams& p,
                                            COMPUTE_TYPE dot,
                                            long long b,
                                            long long h,
                                            long long sq,
                                            long long skv,
                                            long long numHeads)
{
    COMPUTE_TYPE s = dot * static_cast<COMPUTE_TYPE>(p.scale);
    if(p.softcap > 0.0f)
    {
        s = p.softcap * tanhf(s / p.softcap);
    }
    // Additive bias, right-aligned onto (b, h, sq, skv) and broadcast on size-1 dims.
    if(p.mask != nullptr)
    {
        const auto* mask = static_cast<const MASK_TYPE*>(p.mask);
        long long ctx[4] = {b, h, sq, skv};
        long long offset = 0;
        for(int i = 0; i < p.maskRank; ++i)
        {
            long long idx = (p.maskDims[i] == 1) ? 0 : ctx[4 - p.maskRank + i];
            offset += idx * p.maskStr.s[i];
        }
        s += gpu_ref::toAccum(mask[offset]);
    }
    if(p.alibi != 0)
    {
        s += static_cast<COMPUTE_TYPE>(skv - sq)
             * static_cast<COMPUTE_TYPE>(alibiSlope(h, numHeads));
    }
    return s;
}

// d(softcap)/d(scaled score) from the dot product: 1 - tanh^2(dot * scale / c), or 1 without
// softcap. dQ and dK take it; dBias and dV do not.
__device__ inline COMPUTE_TYPE softcapGradFromDot(const SdpaScoreParams& p, COMPUTE_TYPE dot)
{
    if(!(p.softcap > 0.0f))
    {
        return static_cast<COMPUTE_TYPE>(1);
    }
    const COMPUTE_TYPE t = tanhf(dot * static_cast<COMPUTE_TYPE>(p.scale) / p.softcap);
    return static_cast<COMPUTE_TYPE>(1) - t * t;
}

// Explicit dropout mask value for (b, h, sq, skv), [B|1, H|1, Sq, Skv] with stride 0 on broadcast
// dims; 1 without dropout.
__device__ inline COMPUTE_TYPE dropoutAt(const float* mask,
                                         const SdpaStrides& str,
                                         long long b,
                                         long long h,
                                         long long sq,
                                         long long skv)
{
    if(mask == nullptr)
    {
        return static_cast<COMPUTE_TYPE>(1);
    }
    return mask[b * str.s[0] + h * str.s[1] + sq * str.s[2] + skv * str.s[3]];
}

} // namespace sdpa_common
