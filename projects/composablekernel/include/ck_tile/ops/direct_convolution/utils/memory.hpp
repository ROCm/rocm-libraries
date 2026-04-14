// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

namespace ck_tile::direct_conv
{

// Compute the vmcnt flags for __builtin_amdgcn_s_waitcnt
//
// Set vmcnt field in [3:0] and [15:14].
// Set [11:4] to all 1's to max out expcnt and lgkmcnt.
inline constexpr auto vmcnt(int count) -> uint16_t
{ return ((count & 0x30) << (14 - 4)) | (count & 0xF) | 0xFF0; }

// Wait for all but the last "count" loads.
//
// One BUFFER_LOADS_*_LDS instruction counts as vmcnt of 1, regardless
// of how many memory transactions it generates.
template <int Count>
__device__ inline void wait_vmcnt()
{
    static_assert(Count >= 0 && Count <= 63, "vmcnt must be in range [0, 63] (6-bit field)");
    __builtin_amdgcn_s_waitcnt(vmcnt(Count));
}

// Convenience alias for common case
__device__ __forceinline__ void wait_vmcnt_all()
{ wait_vmcnt<0>(); }

} // namespace ck_tile::direct_conv
