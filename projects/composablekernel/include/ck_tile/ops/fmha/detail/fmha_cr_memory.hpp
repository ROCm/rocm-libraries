// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/arch/amd_cluster_load.hpp"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

namespace ck_tile::detail::fmha_cr {
template <class T>
using V8 = T __attribute__((ext_vector_type(8)));
template <class T>
using V16 = T __attribute__((ext_vector_type(16)));
using U4  = uint32_t __attribute__((ext_vector_type(4)));
using I4  = int32_t __attribute__((ext_vector_type(4)));
using I8  = int32_t __attribute__((ext_vector_type(8)));

// TDM descriptor for a native BSHD slice; row padding matches the WMMA LDS readers.
// A [sequence, 128] slice of native BSHD memory. Source row_stride is in
// halfword elements; the destination remains densely packed with LDS padding.
template <int Rows, int CacheHint = 0, int PadBytes = 16, int Width = 128, class T>
__device__ __forceinline__ void
load_strided(T* lds, const T* global, int remaining, uint32_t row_stride)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    static_assert(sizeof(T) == 2);
    static_assert(PadBytes == 16 || PadBytes == 32);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(global);
    const uint32_t local = uint32_t(reinterpret_cast<uintptr_t>(lds));
    U4 g0                = {1, local, uint32_t(addr), uint32_t(addr >> 32) | (2u << 30)};
    I8 g1                = {int((1u << 16) | (1u << 20) | ((Width == 128 ? 5u : 4u) << 22) |
                 ((PadBytes / 4 - 1u) << 25)),
                            Width << 16,
                            int(uint32_t(remaining) << 16),
                            (Width << 16) | (remaining >> 16),
                            Rows,
                            int(row_stride),
                            0,
                            0};
#pragma unroll
    for(int i = 0; i < 4; ++i)
        g0[i] = __builtin_amdgcn_readfirstlane(g0[i]);
#pragma unroll
    for(int i = 0; i < 8; ++i)
        g1[i] = __builtin_amdgcn_readfirstlane(g1[i]);
    __builtin_amdgcn_tensor_load_to_lds(g0, g1, I4{}, I4{}, I8{}, CacheHint);
#else
    (void)lds;
    (void)global;
    (void)remaining;
    (void)row_stride;
#endif
}

// Keep store address operands live until every instruction in the burst has XACKed.
// Preserve the address registers across replay, as in the CK async store helper.
__device__ __forceinline__ void
store_async_x8(const void* base, const uint32_t (&offset)[8], const uint32_t (&local)[8])
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    asm volatile("global_store_async_from_lds_b128 %0, %8, %16\n\t"
                 "global_store_async_from_lds_b128 %1, %9, %16\n\t"
                 "global_store_async_from_lds_b128 %2, %10, %16\n\t"
                 "global_store_async_from_lds_b128 %3, %11, %16\n\t"
                 "global_store_async_from_lds_b128 %4, %12, %16\n\t"
                 "global_store_async_from_lds_b128 %5, %13, %16\n\t"
                 "global_store_async_from_lds_b128 %6, %14, %16\n\t"
                 "global_store_async_from_lds_b128 %7, %15, %16\n\t"
                 "s_wait_xcnt 0"
                 :
                 : "v"(offset[0]),
                   "v"(offset[1]),
                   "v"(offset[2]),
                   "v"(offset[3]),
                   "v"(offset[4]),
                   "v"(offset[5]),
                   "v"(offset[6]),
                   "v"(offset[7]),
                   "v"(local[0]),
                   "v"(local[1]),
                   "v"(local[2]),
                   "v"(local[3]),
                   "v"(local[4]),
                   "v"(local[5]),
                   "v"(local[6]),
                   "v"(local[7]),
                   "s"(base)
                 : "memory");
#else
    (void)base;
    (void)offset;
    (void)local;
#endif
}

// A 15-row fragment contains seven full pairs of rows and one final row.
// The upper half-wave must not store the missing sixteenth row.
__device__ __forceinline__ void
store_async_15(const void* base, const uint32_t (&offset)[8], const uint32_t (&local)[8])
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    asm volatile("global_store_async_from_lds_b128 %0, %7, %14\n\t"
                 "global_store_async_from_lds_b128 %1, %8, %14\n\t"
                 "global_store_async_from_lds_b128 %2, %9, %14\n\t"
                 "global_store_async_from_lds_b128 %3, %10, %14\n\t"
                 "global_store_async_from_lds_b128 %4, %11, %14\n\t"
                 "global_store_async_from_lds_b128 %5, %12, %14\n\t"
                 "global_store_async_from_lds_b128 %6, %13, %14\n\t"
                 "s_wait_xcnt 0"
                 :
                 : "v"(offset[0]),
                   "v"(offset[1]),
                   "v"(offset[2]),
                   "v"(offset[3]),
                   "v"(offset[4]),
                   "v"(offset[5]),
                   "v"(offset[6]),
                   "v"(local[0]),
                   "v"(local[1]),
                   "v"(local[2]),
                   "v"(local[3]),
                   "v"(local[4]),
                   "v"(local[5]),
                   "v"(local[6]),
                   "s"(base)
                 : "memory");
    if(threadIdx.x % 32 < 16)
        asm volatile("global_store_async_from_lds_b128 %0, %1, %2\n\t"
                     "s_wait_xcnt 0"
                     :
                     : "v"(offset[7]), "v"(local[7]), "s"(base)
                     : "memory");
#else
    (void)base;
    (void)offset;
    (void)local;
#endif
}

__device__ __forceinline__ void
store_async_x4(const void* base, const uint32_t (&offset)[4], const uint32_t (&local)[4])
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    asm volatile("global_store_async_from_lds_b128 %0, %4, %8\n\t"
                 "global_store_async_from_lds_b128 %1, %5, %8\n\t"
                 "global_store_async_from_lds_b128 %2, %6, %8\n\t"
                 "global_store_async_from_lds_b128 %3, %7, %8\n\t"
                 "s_wait_xcnt 0"
                 :
                 : "v"(offset[0]),
                   "v"(offset[1]),
                   "v"(offset[2]),
                   "v"(offset[3]),
                   "v"(local[0]),
                   "v"(local[1]),
                   "v"(local[2]),
                   "v"(local[3]),
                   "s"(base)
                 : "memory");
#else
    (void)base;
    (void)offset;
    (void)local;
#endif
}

__device__ __forceinline__ void wait()
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    __builtin_amdgcn_s_wait_tensorcnt(0);
#endif
}

// ds_load_tr16 consumes an 8x16 patch per half-wave. Input lane supplies a
// contiguous vector at (lane%8, lane/8*8); output lane receives a column of 8.
template <class T>
__device__ __forceinline__ V8<T> transpose8(const T* lds)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    using Native = __bf16 __attribute__((vector_size(16)));
    auto p       = ck_tile::to_lds(reinterpret_cast<Native*>(const_cast<T*>(lds)));
    return __builtin_bit_cast(V8<T>, __builtin_amdgcn_ds_load_tr16_b128_v8bf16(p));
#else
    (void)lds;
    return {};
#endif
}

template <int Pitch = 136, class T>
__device__ __forceinline__ V16<T> read_v32(const T* lds, int d, int k)
{
    const int lane = threadIdx.x % 32;
    const int row  = k + lane / 16 * 8 + lane % 8;
    const int col  = d + (lane % 16) / 8 * 8;
    V8<T> a        = transpose8(lds + row * Pitch + col);
    V8<T> b        = transpose8(lds + (row + 16) * Pitch + col);
    V16<T> result{};
#pragma unroll
    for(int i = 0; i < 8; ++i)
    {
        result[i]     = a[i];
        result[8 + i] = b[i];
    }
    return result;
}
} // namespace ck_tile::detail::fmha_cr
