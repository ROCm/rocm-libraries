#pragma once

/// This file implements some utilities for memory operations.
///
/// Implements functions for constructing buffer resource descriptors with bounds checking
/// and for waiting for memory operations to complete.

#include <hip/hip_runtime.h>

// Compute the vmcnt flags for __builtin_amdgcn_s_waitcnt
//
// Set vmcnt field in [3:0] and [15:14].
// Set [11:4] to all 1's to max out expcnt and lgkmcnt.
inline constexpr auto vmcnt(int count) -> uint16_t
{
    return ((count & 0x30) << (14 - 4)) | (count & 0xF) | 0xFF0;
}

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
{
    wait_vmcnt<0>();
}

// Compute the lgkmcnt flags for __builtin_amdgcn_s_waitcnt.
//
// lgkmcnt occupies bits [11:8] (4 bits). Set vmcnt (bits [3:0] and [15:14])
// and expcnt (bits [6:4]) to their max (15 / 7) so the wait fires only on
// the LDS/scalar-memory counter, not on global vector loads.
inline constexpr auto lgkmcnt(int count) -> uint16_t
{
    return 0xC000 | 0x000F | 0x0070 | ((count & 0xF) << 8);
}

// Wait for all but the last "count" outstanding LDS or scalar-memory operations.
template <int Count>
__device__ inline void wait_lgkmcnt()
{
    static_assert(Count >= 0 && Count <= 15, "lgkmcnt must be in range [0, 15] (4-bit field)");
    __builtin_amdgcn_s_waitcnt(lgkmcnt(Count));
}

// Convenience alias for common case
__device__ __forceinline__ void wait_lgkmcnt_all()
{
    wait_lgkmcnt<0>();
}
