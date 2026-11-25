// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

namespace ck {

// One bit per lane of a wave: wave64 on gfx9 (and on the host, as for ck::WarpSize), wave32
// otherwise.
#if defined(__GFX9__) || !defined(__HIP_DEVICE_COMPILE__)
using lane_mask_t = uint64_t;
#else
using lane_mask_t = uint32_t;
#endif

// Lane mask of the lanes of the wave for which pred is true. It must be called by all the lanes of
// the wave. Uses the ballot of the wave size of the architecture (w64 on gfx9, w32 otherwise).
__device__ inline lane_mask_t amd_wave_ballot(bool pred)
{
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__GFX9__)
    return __builtin_amdgcn_ballot_w64(pred);
#else
    return __builtin_amdgcn_ballot_w32(pred);
#endif
#else
    ignore = pred;
    return 0;
#endif
}

} // namespace ck
