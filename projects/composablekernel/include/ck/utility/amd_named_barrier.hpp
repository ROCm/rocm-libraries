// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef CK_AMD_NAMED_BARRIER_HPP
#define CK_AMD_NAMED_BARRIER_HPP

#include "ck/ck.hpp"

// #define CK_USE_AMD_NAMED_BARRIER_ASM 1

namespace ck {

#ifdef CK_USE_AMD_NAMED_BARRIER_ASM
template <unsigned NameId, unsigned Count>
class NamedBarrier
{
    public:
    __device__ void init()
    {
        constexpr uint32_t imm = NameId | (Count << 12);
        asm("s_movk_i32 m0, %0\n\ts_barrier_init m0" : : "n"(imm) : "memory");
    }
    __device__ void join()
    {
        constexpr uint32_t imm = NameId;
        asm("s_barrier_join %0" : : "n"(imm) : "memory");
    }
    __device__ void wait()
    {
        constexpr uint32_t imm = NameId;
        asm("s_barrier_wait %0" : : "n"(imm) : "memory");
    }
    __device__ void signal()
    {
        constexpr uint32_t imm = NameId;
        asm("s_movk_i32 m0, %0\n\ts_barrier_signal m0" : : "n"(imm) : "memory");
    }

    template <bool async>
    __device__ void sync_lds()
    {
        if constexpr(async)
        {
            asm volatile("s_wait_asynccnt 0x0 " ::);
        }
        else
        {
            asm volatile("s_wait_dscnt 0x0 " ::);
        }
        signal();
        wait();
    }
};
#else
template <unsigned Count>
class NamedBarrier
{
    public:
    __device__ NamedBarrier() {}
    __device__ void init()
    {
#if defined(__gfx13__)
        __builtin_amdgcn_s_barrier_init(&bar_, Count);
#endif
    }
    __device__ void join()
    {
#if defined(__gfx13__)
        __builtin_amdgcn_s_barrier_join(&bar_);
#endif
    }
    __device__ void wait()
    {
#if defined(__gfx13__)
        __builtin_amdgcn_s_barrier_wait(bar_id_);
#endif
    }
    __device__ void signal()
    {
#if defined(__gfx13__)
        __builtin_amdgcn_s_barrier_signal_var(&bar_, 0);
#endif
    }
    template <bool async>
    __device__ void sync_lds()
    {
#if defined(__gfx13__)
        if constexpr(async)
        {
            asm volatile("s_wait_asynccnt 0x0 " ::);
        }
        else
        {
            asm volatile("s_wait_dscnt 0x0 " ::);
        }

        signal();
        wait();
#endif
    }

    private:
#if defined(__gfx13__)
    __amdgpu_named_workgroup_barrier_t bar_;
    static constexpr uint32_t bar_id_ = 1;
#endif
};
#endif

} // namespace ck

#endif // CK_AMD_NAMED_BARRIER_HPP
