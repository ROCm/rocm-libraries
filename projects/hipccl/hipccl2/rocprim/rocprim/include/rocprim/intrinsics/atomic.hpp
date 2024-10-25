// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_INTRINSICS_ATOMIC_HPP_
#define ROCPRIM_INTRINSICS_ATOMIC_HPP_

#include "../config.hpp"

BEGIN_ROCPRIM_NAMESPACE

namespace detail
{
    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_add(unsigned int * address, unsigned int value)
    {
        return ::atomicAdd(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    int atomic_add(int * address, int value)
    {
        return ::atomicAdd(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    float atomic_add(float * address, float value)
    {
        return ::atomicAdd(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE unsigned long atomic_add(unsigned long* address,
                                                           unsigned long  value)
    {
        return ::atomicAdd(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long long atomic_add(unsigned long long * address, unsigned long long value)
    {
        return ::atomicAdd(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_min(unsigned int* address, unsigned int value)
    {
        return ::atomicMin(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long atomic_min(unsigned long* address, unsigned long value)
    {
        return ::atomicMin(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long long atomic_min(unsigned long long* address, unsigned long long value)
    {
        return ::atomicMin(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_wrapinc(unsigned int* address, unsigned int value)
    {
        return ::atomicInc(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_exch(unsigned int * address, unsigned int value)
    {
        return ::atomicExch(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long long atomic_exch(unsigned long long * address, unsigned long long value)
    {
        return ::atomicExch(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE unsigned char atomic_load(const unsigned char* address)
    {
        return __hip_atomic_load(address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE unsigned short atomic_load(const unsigned short* address)
    {
        return __hip_atomic_load(address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE unsigned int atomic_load(const unsigned int* address)
    {
        return __hip_atomic_load(address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE unsigned long atomic_load(const unsigned long* address)
    {
        return __hip_atomic_load(address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE unsigned long long atomic_load(const unsigned long long* address)
    {
        return __hip_atomic_load(address, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void atomic_store(unsigned char* address, unsigned char value)
    {
        __hip_atomic_store(address, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void atomic_store(unsigned short* address, unsigned short value)
    {
        __hip_atomic_store(address, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void atomic_store(unsigned int* address, unsigned int value)
    {
        __hip_atomic_store(address, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void atomic_store(unsigned long* address, unsigned long value)
    {
        __hip_atomic_store(address, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE void atomic_store(unsigned long long* address,
                                                    unsigned long long  value)
    {
        __hip_atomic_store(address, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    /// \brief Wait for all vector memory operations to complete
    ///
    /// This ensures that previous visible writes to vector memory have completed before the function
    /// returns. Atomic operations following the call are guaranteed to be visible
    /// to other threads in the device after vmem writes preceding the call.
    ///
    /// Provides no guarantees about visibility, only ordering, i.e. caches are not flushed.
    /// Visibility has to be enforced in another way (e.g. writing *through* cache)
    ///
    /// This is a dangerous internal function not meant for users, and only meant to be used by
    /// developers that know what they are doing.
    ROCPRIM_DEVICE ROCPRIM_INLINE void atomic_fence_release_vmem_order_only()
    {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
        // Wait until all vmem operations complete (s_waitcnt vmcnt(0))
        __builtin_amdgcn_s_waitcnt(/*vmcnt*/ 0 | (/*exp_cnt*/ 0x7 << 4) | (/*lgkmcnt*/ 0xf << 8));
    }

    /// \brief Make sure visible operations are complete
    ///
    /// Ensure that following visible reads are not reordered before preceding atomic operations
    /// Similarly to atomic_fence_release_vmem_order_only() this function provides no visibility
    /// guarantees, visiblity of reads must be guaranteed in other wise (like reading *through*
    /// caches)
    ///
    /// This is a dangerous internal function not meant for users, and only meant to be used by
    /// developers that know what they are doing.
    ROCPRIM_DEVICE ROCPRIM_INLINE void atomic_fence_acquire_order_only()
    {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    }
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_ATOMIC_HPP_
