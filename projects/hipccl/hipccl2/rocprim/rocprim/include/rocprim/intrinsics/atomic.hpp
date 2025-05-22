// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

    ROCPRIM_DEVICE ROCPRIM_INLINE
    double atomic_add(double * address, double value)
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
    unsigned int atomic_wrapinc(unsigned int* address, unsigned int value)
    {
        return ::atomicInc(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_max(unsigned int* address, unsigned int value)
    {
        return ::atomicMax(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long atomic_max(unsigned long* address, unsigned long value)
    {
        return ::atomicMax(address, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long long atomic_max(unsigned long long* address, unsigned long long value)
    {
        return ::atomicMax(address, value);
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
    unsigned int atomic_cas(unsigned int* address, unsigned int compare, unsigned int value)
    {
        return ::atomicCAS(address, compare, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long atomic_cas(unsigned long* address, unsigned long compare, unsigned long value)
    {
        return ::atomicCAS(address, compare, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned long long atomic_cas(unsigned long long* address,
                                  unsigned long long  compare,
                                  unsigned long long  value)
    {
        return ::atomicCAS(address, compare, value);
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    unsigned int atomic_exch(unsigned int* address, unsigned int value)
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

    ROCPRIM_DEVICE ROCPRIM_INLINE
    __uint128_t atomic_load(const __uint128_t* address)
    {
        __uint128_t result;

#define ROCPRIM_ATOMIC_LOAD(inst, mod, wait, ptr) \
    asm volatile(inst " %0, %1 " mod "\t\n" wait : "=v"(result) : "v"(ptr) : "memory")

#if ROCPRIM_TARGET_CDNA3
    #define ROCPRIM_ATOMIC_LOAD_FLAT(ptr) \
        ROCPRIM_ATOMIC_LOAD("flat_load_dwordx4", "sc1", "s_waitcnt vmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_LOAD_SHARED(ptr) \
        ROCPRIM_ATOMIC_LOAD("ds_read_b128", "", "s_waitcnt lgkmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_LOAD_GLOBAL(ptr) \
        ROCPRIM_ATOMIC_LOAD("global_load_dwordx4", "off sc1", "s_waitcnt vmcnt(0)", ptr)
#elif ROCPRIM_TARGET_RDNA4
    #define ROCPRIM_ATOMIC_LOAD_FLAT(ptr) \
        ROCPRIM_ATOMIC_LOAD("flat_load_b128", "scope:SCOPE_DEV", "s_wait_loadcnt_dscnt 0x0", ptr)
    #define ROCPRIM_ATOMIC_LOAD_SHARED(ptr) \
        ROCPRIM_ATOMIC_LOAD("ds_load_b128", "", "s_wait_dscnt 0x0", ptr)
    #define ROCPRIM_ATOMIC_LOAD_GLOBAL(ptr) \
        ROCPRIM_ATOMIC_LOAD("global_load_b128", "off scope:SCOPE_DEV", "s_wait_loadcnt 0x0", ptr)
#elif ROCPRIM_TARGET_RDNA2 || ROCPRIM_TARGET_RDNA1
    #define ROCPRIM_ATOMIC_LOAD_FLAT(ptr) \
        ROCPRIM_ATOMIC_LOAD("flat_load_dwordx4", "glc dlc", "s_waitcnt vmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_LOAD_SHARED(ptr) \
        ROCPRIM_ATOMIC_LOAD("ds_read_b128", "", "s_waitcnt lgkmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_LOAD_GLOBAL(ptr) \
        ROCPRIM_ATOMIC_LOAD("global_load_dwordx4", "off glc dlc", "s_waitcnt vmcnt(0)", ptr)
#elif ROCPRIM_TARGET_GCN3
    #define ROCPRIM_ATOMIC_LOAD_FLAT(ptr) \
        ROCPRIM_ATOMIC_LOAD("flat_load_dwordx4", "glc", "s_waitcnt vmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_LOAD_SHARED(ptr) \
        ROCPRIM_ATOMIC_LOAD("ds_read_b128", "", "s_waitcnt lgkmcnt(0)", ptr)
    // This architecture doesn't support atomics on the global AS.
    #define ROCPRIM_ATOMIC_LOAD_GLOBAL(ptr) ROCPRIM_ATOMIC_LOAD_FLAT(ptr)
#elif ROCPRIM_TARGET_RDNA3 || ROCPRIM_TARGET_CDNA2 || ROCPRIM_TARGET_CDNA1 || ROCPRIM_TARGET_GCN5 \
    || ROCPRIM_TARGET_SPIRV
    // We don't really know what architecture we are on when targeting
    // SPIR-V. Lets just assume it's one of these.
    #define ROCPRIM_ATOMIC_LOAD_FLAT(ptr) \
        ROCPRIM_ATOMIC_LOAD("flat_load_dwordx4", "glc", "s_waitcnt vmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_LOAD_SHARED(ptr) \
        ROCPRIM_ATOMIC_LOAD("ds_read_b128", "", "s_waitcnt lgkmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_LOAD_GLOBAL(ptr) \
        ROCPRIM_ATOMIC_LOAD("global_load_dwordx4", "off glc", "s_waitcnt vmcnt(0)", ptr)
#elif defined(__HIP_DEVICE_COMPILE__)
    // Please submit an issue or pull request!
    #error support for 128-bit atomics not implemented for current architecture
#endif

#ifdef __HIP_DEVICE_COMPILE__
    #if !ROCPRIM_TARGET_SPIRV && defined(__has_builtin) \
        && __has_builtin(__builtin_amdgcn_is_shared) && __has_builtin(__builtin_amdgcn_is_private)

        auto* ptr = (const __attribute__((address_space(0 /*flat*/))) __uint128_t*)address;
        if(__builtin_amdgcn_is_shared(ptr))
        {
            auto* shared_ptr
                = (const __attribute__((address_space(3 /*lds*/))) __uint128_t*)address;
            ROCPRIM_ATOMIC_LOAD_SHARED(shared_ptr);
        }
        else if(__builtin_amdgcn_is_private(ptr))
        {
            ROCPRIM_ATOMIC_LOAD_FLAT(address);
        }
        else
        {
            auto* global_ptr
                = (const __attribute__((address_space(1 /*global*/))) __uint128_t*)address;
            ROCPRIM_ATOMIC_LOAD_GLOBAL(global_ptr);
        }
    #else
        // SPIR-V does not like the address-space checks. For now
        // lets just do flat loading/storing.
        ROCPRIM_ATOMIC_LOAD_FLAT(address);
    #endif
#else
        (void)address;
        result = 0;
#endif

        return result;

#undef ROCPRIM_ATOMIC_LOAD
#undef ROCPRIM_ATOMIC_LOAD_FLAT
#undef ROCPRIM_ATOMIC_LOAD_SHARED
#undef ROCPRIM_ATOMIC_LOAD_GLOBAL
    }

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void atomic_store(unsigned char* address, unsigned char value)
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

    ROCPRIM_DEVICE ROCPRIM_INLINE
    void atomic_store(__uint128_t* address, const __uint128_t value)
    {
#define ROCPRIM_ATOMIC_STORE(inst, mod, wait, ptr) \
    asm volatile(inst " %0, %1 " mod "\t\n" wait : : "v"(ptr), "v"(value) : "memory")

#if ROCPRIM_TARGET_CDNA3
    #define ROCPRIM_ATOMIC_STORE_FLAT(ptr) \
        ROCPRIM_ATOMIC_STORE("flat_store_dwordx4", "sc1", "s_waitcnt vmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_STORE_SHARED(ptr) \
        ROCPRIM_ATOMIC_STORE("ds_write_b128", "", "s_waitcnt lgkmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_STORE_GLOBAL(ptr) \
        ROCPRIM_ATOMIC_STORE("global_store_dwordx4", "off sc1", "s_waitcnt vmcnt(0)", ptr)
#elif ROCPRIM_TARGET_RDNA4
    #define ROCPRIM_ATOMIC_STORE_FLAT(ptr) \
        ROCPRIM_ATOMIC_STORE("flat_store_b128", "scope:SCOPE_DEV", "s_wait_storecnt_dscnt 0x0", ptr)
    #define ROCPRIM_ATOMIC_STORE_SHARED(ptr) \
        ROCPRIM_ATOMIC_STORE("ds_store_b128", "", "s_wait_dscnt 0x0", ptr)
    #define ROCPRIM_ATOMIC_STORE_GLOBAL(ptr) \
        ROCPRIM_ATOMIC_STORE("global_store_b128", "off scope:SCOPE_DEV", "s_wait_storecnt 0x0", ptr)
#elif ROCPRIM_TARGET_GCN3
    #define ROCPRIM_ATOMIC_STORE_FLAT(ptr) \
        ROCPRIM_ATOMIC_STORE("flat_store_dwordx4", "", "s_waitcnt vmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_STORE_SHARED(ptr) \
        ROCPRIM_ATOMIC_STORE("ds_write_b128", "", "s_waitcnt lgkmcnt(0)", ptr)
    // This architecture doesn't support atomics on the global AS.
    #define ROCPRIM_ATOMIC_STORE_GLOBAL(ptr) ROCPRIM_ATOMIC_STORE_FLAT(ptr)
#elif ROCPRIM_TARGET_RDNA3 || ROCPRIM_TARGET_RDNA2 || ROCPRIM_TARGET_RDNA1 || ROCPRIM_TARGET_CDNA2 \
    || ROCPRIM_TARGET_CDNA1 || ROCPRIM_TARGET_GCN5 || ROCPRIM_TARGET_SPIRV
    // We don't really know what architecture we are on when targeting
    // SPIR-V. Lets just assume it's one of these.
    #define ROCPRIM_ATOMIC_STORE_FLAT(ptr) \
        ROCPRIM_ATOMIC_STORE("flat_store_dwordx4", "", "s_waitcnt vmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_STORE_SHARED(ptr) \
        ROCPRIM_ATOMIC_STORE("ds_write_b128", "", "s_waitcnt lgkmcnt(0)", ptr)
    #define ROCPRIM_ATOMIC_STORE_GLOBAL(ptr) \
        ROCPRIM_ATOMIC_STORE("global_store_dwordx4", "off", "s_waitcnt vmcnt(0)", ptr)
#elif defined(__HIP_DEVICE_COMPILE__)
    // Please submit an issue or pull request!
    #error support for 128-bit atomics not implemented for current architecture
#endif

#ifdef __HIP_DEVICE_COMPILE__
    #if !ROCPRIM_TARGET_SPIRV && defined(__has_builtin) \
        && __has_builtin(__builtin_amdgcn_is_shared) && __has_builtin(__builtin_amdgcn_is_private)

        auto* ptr = (__attribute__((address_space(0 /*flat*/))) __uint128_t*)address;
        if(__builtin_amdgcn_is_shared(ptr))
        {
            auto* shared_ptr = (__attribute__((address_space(3 /*lds*/))) __uint128_t*)address;
            ROCPRIM_ATOMIC_STORE_SHARED(shared_ptr);
        }
        else if(__builtin_amdgcn_is_private(ptr))
        {
            ROCPRIM_ATOMIC_STORE_FLAT(address);
        }
        else
        {
            auto* global_ptr = (__attribute__((address_space(1 /*global*/))) __uint128_t*)address;
            ROCPRIM_ATOMIC_STORE_GLOBAL(global_ptr);
        }
    #else
        // SPIR-V does not like the address-space checks. For now
        // lets just do flat loading/storing.
        ROCPRIM_ATOMIC_STORE_FLAT(address);
    #endif
#else
        (void)address;
        (void)value;
#endif

#undef ROCPRIM_ATOMIC_STORE
#undef ROCPRIM_ATOMIC_STORE_FLAT
#undef ROCPRIM_ATOMIC_STORE_SHARED
#undef ROCPRIM_ATOMIC_STORE_GLOBAL
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
        __builtin_amdgcn_s_waitcnt(/*vmcnt*/ 0 | (/*exp_cnt*/ 0x7 << 4) | (/*lgkmcnt*/ 0xf << 8));
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    }
}

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_INTRINSICS_ATOMIC_HPP_
