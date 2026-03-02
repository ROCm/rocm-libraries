// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "FloatTypes.h"
#include "VectorTypes.hpp"

namespace hip_kernel_plugin
{
namespace batchnorm
{

__forceinline__ __device__ unsigned int getStashIndex(unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrp_sz,
                                                      unsigned int xgrp_id,
                                                      unsigned int xlid,
                                                      unsigned int xstride,
                                                      unsigned int nstride)
{
#if(HIP_PLUGIN_USE_FPMIX || HIP_PLUGIN_USE_BFPMIX)
    if constexpr(HIP_PLUGIN_LAYOUT_NHWC)
    {
        if constexpr(HIP_PLUGIN_BN_C % 2 == 0)
        {
            return zgroupoffset * (HIP_PLUGIN_BN_C / HIP_PLUGIN_BN_VEC_SIZE * HIP_PLUGIN_BN_HW) +
                   (vindex * 2 + xlid % 2) * nstride + ygroupoffset * ystride +
                   (xgrp_sz * xgrp_id + xlid / 2 * 2) * xstride;
        }
        else
        {
            return zgroupoffset * (HIP_PLUGIN_BN_C / HIP_PLUGIN_BN_VEC_SIZE * HIP_PLUGIN_BN_HW) +
                   (vindex * 2) * nstride + ygroupoffset * ystride +
                   (xgrp_sz * xgrp_id + xlid) * xstride;
        }
    }
    else
    {
        return zgroupoffset * (HIP_PLUGIN_BN_C / HIP_PLUGIN_BN_VEC_SIZE * HIP_PLUGIN_BN_HW) +
               ((vindex * 2) * nstride + ygroupoffset * ystride +
                (xgrp_sz * xgrp_id + xlid) * xstride + 1) /
                   2 * 2;
    }
#else
    return zgroupoffset * (HIP_PLUGIN_BN_C / HIP_PLUGIN_BN_VEC_SIZE * HIP_PLUGIN_BN_HW) +
           vindex * nstride + ygroupoffset * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
#endif
}

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ FpPrecType_C loadFromStash(const FpType_C* stash,
                                                      unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrp_sz,
                                                      unsigned int xgrp_id,
                                                      unsigned int xlid,
                                                      unsigned int xstride)
{
    constexpr unsigned int nstride_hw =
        HIP_PLUGIN_BN_C / HIP_PLUGIN_BN_VEC_SIZE * HIP_PLUGIN_BN_HW;
    unsigned int nstride = HIP_PLUGIN_BN_STASH_METHOD == 0 ? ystride : nstride_hw;

    unsigned int index = getStashIndex(
        vindex, zgroupoffset, ygroupoffset, ystride, xgrp_sz, xgrp_id, xlid, xstride, nstride);

    if constexpr(HIP_PLUGIN_BN_STASH_METHOD == 0 || HIP_PLUGIN_BN_STASH_METHOD == 1)
    {
        return *((const FpPrecType_C*)(stash + index));
    }
    else
    {
        FpPrecType_C value;
        *(reinterpret_cast<FpType_C*>(&value))     = stash[index];
        *(reinterpret_cast<FpType_C*>(&value) + 1) = stash[index + nstride];
        return value;
    }
}

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ void storeToStash(FpPrecType_C value,
                                             FpType_C* stash,
                                             unsigned int vindex,
                                             unsigned int zgroupoffset,
                                             unsigned int ygroupoffset,
                                             unsigned int ystride,
                                             unsigned int xgrp_sz,
                                             unsigned int xgrp_id,
                                             unsigned int xlid,
                                             unsigned int xstride)
{
    constexpr unsigned int nstride_hw =
        HIP_PLUGIN_BN_C / HIP_PLUGIN_BN_VEC_SIZE * HIP_PLUGIN_BN_HW;
    unsigned int nstride = HIP_PLUGIN_BN_STASH_METHOD == 0 ? ystride : nstride_hw;

    unsigned int index = getStashIndex(
        vindex, zgroupoffset, ygroupoffset, ystride, xgrp_sz, xgrp_id, xlid, xstride, nstride);

    if constexpr(HIP_PLUGIN_BN_STASH_METHOD == 0 || HIP_PLUGIN_BN_STASH_METHOD == 1)
    {
        *(reinterpret_cast<FpPrecType_C*>(stash + index)) = value;
    }
    else
    {
        stash[index]           = *(reinterpret_cast<FpType_C*>(&value));
        stash[index + nstride] = *(reinterpret_cast<FpType_C*>(&value) + 1);
    }
}

} // namespace batchnorm
} // namespace hip_kernel_plugin
