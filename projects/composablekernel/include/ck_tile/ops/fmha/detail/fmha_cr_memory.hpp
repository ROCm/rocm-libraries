// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "ck_tile/core/arch/amd_cluster_load.hpp"
#include "ck_tile/core/arch/amd_tdm_descriptor.hpp"
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

template <int Width, int PadBytes>
struct LdsLayout
{
    // Preserve the tuned D128 row layout; other widths pad each 64- or
    // 32-element chunk so the WMMA fragments never straddle padding.
    static constexpr int kChunk       = Width == 128 ? 128 : Width % 64 == 0 ? 64 : 32;
    static constexpr int kPitch       = Width + Width / kChunk * (PadBytes / 2);
    static constexpr int kPadInterval = kChunk == 128 ? 5 : kChunk == 64 ? 4 : 3;

    CK_TILE_HOST_DEVICE static constexpr int Column(int column)
    {
        return column + column / kChunk * (PadBytes / 2);
    }
};

// TDM descriptor for a native BSHD slice; row padding matches the WMMA LDS readers.
// A [sequence, Width] slice of native BSHD memory. Source row_stride is in
// halfword elements; the destination remains densely packed with LDS padding.
// TDM zero-fills columns beyond valid_width. Q/K head tails rely on this even
// when Q and K reuse LDS containing values from an earlier transfer.
template <int Rows, int CacheHint = 0, int PadBytes = 16, int Width = 128, class T>
__device__ __forceinline__ void
load_strided(T* lds, const T* global, int remaining, uint32_t row_stride, int valid_width = Width)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    static_assert(sizeof(T) == 2);
    static_assert(PadBytes == 16 || PadBytes == 32);
    const uintptr_t addr = reinterpret_cast<uintptr_t>(global);
    const uint32_t local = uint32_t(reinterpret_cast<uintptr_t>(lds));
    U4 g0                = {1, local, uint32_t(addr), uint32_t(addr >> 32) | (2u << 30)};
    I8 g1 = {int((1u << 16) | (1u << 20) | (LdsLayout<Width, PadBytes>::kPadInterval << 22) |
                 ((PadBytes / 4 - 1u) << 25)),
             valid_width << 16,
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
    (void)valid_width;
#endif
}

// Store one row per TDM iteration so the LDS pitch need not match a dense tensor.
// The issuing wave owns these rows; descriptor setup overlaps its pending LDS writes.
// remaining is positive; both row strides are measured in 16-bit elements.
template <int Rows, int Width, int LdsPitch, class T>
__device__ __forceinline__ void store_strided(T* lds, T* global, int remaining, uint32_t row_stride)
{
#if defined(__gfx1250__) || defined(__gfx1250_strict__)
    static_assert(sizeof(T) == 2);
    static_assert(sizeof(TDM_GROUP0) == 16 && sizeof(TDM_GROUP1) == 32);
    static_assert(Rows > 0 && Rows <= 16);
    static_assert(Width >= 32 && Width <= 160 && Width % 32 == 0);
    static_assert(LdsPitch >= Width);
    const int count = remaining < Rows ? remaining : Rows;
    TDM_GROUP0 address(reinterpret_cast<uintptr_t>(lds), reinterpret_cast<uintptr_t>(global), 0, 0);
    TDM_GROUP1 shape;
    // Use 64-byte subrows, avoiding the intermittent D64 corruption observed
    // with a single 128-byte TDM row. Preserve the tuned 256-byte D128 transfer.
    constexpr int Chunk  = Width == 128 ? Width : 32;
    shape.data_size      = 1; // 16-bit elements
    shape.iterate_enable = 1;
    shape.tensorDim0(Chunk);
    shape.tensorDim1(Width / Chunk);
    shape.tile_dim0 = Chunk;
    shape.tile_dim1 = Width / Chunk;
    shape.tensorDim0Stride(Chunk); // One iteration covers a complete output row.
    U4 g0 = bit_cast<U4>(address.bitfield);
    I8 g1 = bit_cast<I8>(shape.bitfield);
    // Iteration uses exactly four dwords. Pack the count in dword 3's high half;
    // TDM_GROUP2_MODE0's nested unions do not place iterate_count there.
    // The count excludes the first row; increments use 16-bit element units.
    I4 g2 = {0, LdsPitch, int(row_stride), (count - 1) << 16};
    g0    = amd_wave_read_first_lane(g0);
    g1    = amd_wave_read_first_lane(g1);
    g2    = amd_wave_read_first_lane(g2);
    // Keep descriptor preparation and LDS writes ahead of the wait/issue pair.
    __builtin_amdgcn_sched_barrier(0);
    ck_tile::s_wait_dscnt<0>();
    __builtin_amdgcn_tensor_store_from_lds(g0, g1, g2, I4{}, I8{}, 0);
#else
    (void)lds;
    (void)global;
    (void)remaining;
    (void)row_stride;
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

template <int Pitch = 136, int Width = 128, class T>
__device__ __forceinline__ V16<T> read_v32(const T* lds, int d, int k)
{
    const int lane = threadIdx.x % 32;
    const int row  = k + lane / 16 * 8 + lane % 8;
    const int col  = LdsLayout<Width, 32>::Column(d + (lane % 16) / 8 * 8);
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
