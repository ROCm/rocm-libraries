// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "data_type.hpp"
#include "amd_address_space.hpp"
#include "ck/utility/amd_buffer_coherence.hpp"

namespace ck {

template <typename T>
union BufferResource
{
    __device__ constexpr BufferResource() : content{} {}

    // 128 bit SGPRs to supply buffer resource in buffer instructions
    // https://rocm-documentation.readthedocs.io/en/latest/GCN_ISA_Manuals/testdocbook.html#vector-memory-buffer-instructions
    int32x4_t content;
    StaticallyIndexedArray<T*, 2> address;
    StaticallyIndexedArray<int32_t, 4> range;
    StaticallyIndexedArray<int32_t, 4> config;
};

template <typename T>
__device__ int32x4_t make_wave_buffer_resource(T* p_wave, index_t element_space_size)
{
    BufferResource<T> wave_buffer_resource;

#ifdef __gfx13__
    // wavewise base address (57 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (45 bit)
    uint64_t num_records = element_space_size * sizeof(T);
    wave_buffer_resource.range(Number<1>{}) |= (num_records & 0x7f) << 25;
    wave_buffer_resource.range(Number<2>{}) = (num_records >> 7);
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;
#else
    // wavewise base address (64 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(Number<2>{}) = element_space_size * sizeof(T);
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;
#endif
    return wave_buffer_resource.content;
}

template <typename T>
__device__ int32x4_t make_wave_buffer_resource_with_default_range(T* p_wave)
{
    BufferResource<T> wave_buffer_resource;
#ifdef __gfx13__
    // wavewise base address (57 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (45 bit)
    wave_buffer_resource.range(Number<2>{}) = 0xffffffff; // max possible range
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;
#else
    // wavewise base address (64 bit)
    wave_buffer_resource.address(Number<0>{}) = const_cast<remove_cv_t<T>*>(p_wave);
    // wavewise range (32 bit)
    wave_buffer_resource.range(Number<2>{}) = 0xffffffff; // max possible range
    // wavewise setting (32 bit)
    wave_buffer_resource.config(Number<3>{}) = CK_BUFFER_RESOURCE_3RD_DWORD;
#endif
    return wave_buffer_resource.content;
}

// buffer load i8
__device__ int8_t
llvm_amdgcn_raw_buffer_load_i8(int32x4_t srsrc,
                               index_t voffset,
                               index_t soffset,
                               index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i8");

__device__ int8x2_t
llvm_amdgcn_raw_buffer_load_i8x2(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i8");

__device__ int8x4_t
llvm_amdgcn_raw_buffer_load_i8x4(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i8");

// buffer load i16
__device__ bhalf_t
llvm_amdgcn_raw_buffer_load_i16(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i16");

__device__ bhalf2_t
llvm_amdgcn_raw_buffer_load_i16x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i16");

__device__ bhalf4_t
llvm_amdgcn_raw_buffer_load_i16x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i16");

// buffer load i32
__device__ int32_t
llvm_amdgcn_raw_buffer_load_i32(int32x4_t srsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.i32");

__device__ int32x2_t
llvm_amdgcn_raw_buffer_load_i32x2(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2i32");

__device__ int32x4_t
llvm_amdgcn_raw_buffer_load_i32x4(int32x4_t srsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

// buffer load fp16
__device__ half_t
llvm_amdgcn_raw_buffer_load_fp16(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f16");

__device__ half2_t
llvm_amdgcn_raw_buffer_load_fp16x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f16");

__device__ half4_t
llvm_amdgcn_raw_buffer_load_fp16x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f16");

// buffer load fp32
__device__ float
llvm_amdgcn_raw_buffer_load_fp32(int32x4_t srsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.f32");

__device__ float2_t
llvm_amdgcn_raw_buffer_load_fp32x2(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

__device__ float4_t
llvm_amdgcn_raw_buffer_load_fp32x4(int32x4_t srsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

// buffer store i8
__device__ void
llvm_amdgcn_raw_buffer_store_i8(int8_t vdata,
                                int32x4_t rsrc,
                                index_t voffset,
                                index_t soffset,
                                index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i8");

__device__ void
llvm_amdgcn_raw_buffer_store_i8x2(int8x2_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i8");

__device__ void
llvm_amdgcn_raw_buffer_store_i8x4(int8x4_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i8");

// buffer store i16
__device__ void
llvm_amdgcn_raw_buffer_store_i16(bhalf_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i16");

__device__ void
llvm_amdgcn_raw_buffer_store_i16x2(bhalf2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i16");

__device__ void
llvm_amdgcn_raw_buffer_store_i16x4(bhalf4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i16");

// buffer store i32
__device__ void
llvm_amdgcn_raw_buffer_store_i32(int32_t vdata,
                                 int32x4_t rsrc,
                                 index_t voffset,
                                 index_t soffset,
                                 index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.i32");

__device__ void
llvm_amdgcn_raw_buffer_store_i32x2(int32x2_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2i32");

__device__ void
llvm_amdgcn_raw_buffer_store_i32x4(int32x4_t vdata,
                                   int32x4_t rsrc,
                                   index_t voffset,
                                   index_t soffset,
                                   index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

// buffer store fp16
__device__ void
llvm_amdgcn_raw_buffer_store_fp16(half_t vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f16");

__device__ void
llvm_amdgcn_raw_buffer_store_fp16x2(half2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f16");

__device__ void
llvm_amdgcn_raw_buffer_store_fp16x4(half4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f16");

// buffer store fp32
__device__ void
llvm_amdgcn_raw_buffer_store_fp32(float vdata,
                                  int32x4_t rsrc,
                                  index_t voffset,
                                  index_t soffset,
                                  index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x2(float2_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

__device__ void
llvm_amdgcn_raw_buffer_store_fp32x4(float4_t vdata,
                                    int32x4_t rsrc,
                                    index_t voffset,
                                    index_t soffset,
                                    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

#if defined(__gfx13__)
// buffer atomic-add fp16
__device__ half2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    half2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16.v4i32");

__device__ int32_t llvm_amdgcn_raw_buffer_atomic_add_i32(
    int32_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32.v4i32");

__device__ float llvm_amdgcn_raw_buffer_atomic_add_fp32(
    float vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32.v4i32");

__device__ double llvm_amdgcn_raw_buffer_atomic_max_fp64(
    double vdata,
    int32x4_t rsrc, // dst_wave_buffer_resource
    int voffset,    // dst_thread_addr_offset
    int soffset,    // dst_wave_addr_offset
    int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fmax.f64.v4i32");
#else
__device__ half2_t llvm_amdgcn_raw_buffer_atomic_add_fp16x2(
    half2_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.v2f16");

// buffer atomic-add i32
__device__ int32_t llvm_amdgcn_raw_buffer_atomic_add_i32(
    int32_t vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.add.i32");

__device__ float llvm_amdgcn_raw_buffer_atomic_add_fp32(
    float vdata,
    int32x4_t rsrc,
    index_t voffset,
    index_t soffset,
    index_t glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fadd.f32");

__device__ double
llvm_amdgcn_raw_buffer_atomic_max_fp64(double vdata,
                                       int32x4_t rsrc, // dst_wave_buffer_resource
                                       int voffset,    // dst_thread_addr_offset
                                       int soffset,    // dst_wave_addr_offset
                                       int glc_slc) __asm("llvm.amdgcn.raw.buffer.atomic.fmax.f64");
#endif

// memory coherency bit for buffer store/load instruction
// check ISA manual for each GFX target
// e.g. for
// https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf,
// page 67~68
enum struct AmdBufferCoherenceEnum
{
    DefaultCoherence = 0, // default value
    GLC              = 1,
    SLC              = 2,
    GLC_SLC          = 3,
    // gfx94: bit 0 = sc0, bit 1 = nt, bit 3 = swz, bit 4 = sc1
    // SC[1:0] System Cache level: 0=wave, 1=group, 2=device, 3=system
    // NT Non-Temporal: 0=expect temporal reuse; 1=do not expect temporal reuse
    WAVE_NT0   = 0,
    WAVE_NT1   = 2,
    GROUP_NT0  = 1,
    GROUP_NT1  = 3,
    DEVICE_NT0 = 8,
    DEVICE_NT1 = 10,
    SYSTEM_NT0 = 9,
    SYSTEM_NT1 = 11,
};

// To do: need refactor as a tuple for combination load
enum struct TensorLoadOption
{
    DEFAULT_LOAD                     = 0,
    CLUSTER_MULTICAST_LOAD           = 1,
    WGP_MULTICAST_LOAD               = 2,
    CLUSTER_DDS_LOAD                 = 3,
    CLUSTER_ASYNC_MULTICAST_LDS_LOAD = 4,
    DS_TILED_LOAD                    = 5,
};

template <index_t N, AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type<int8_t, N>::type
amd_buffer_load_impl_raw(int32x4_t src_wave_buffer_resource,
                         index_t src_thread_addr_offset,
                         index_t src_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        return llvm_amdgcn_raw_buffer_load_i8(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset,
                                              static_cast<index_t>(coherence));
    }
    else if constexpr(N == 2)
    {

        int16_t tmp = llvm_amdgcn_raw_buffer_load_i16(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<int8x2_t>(tmp);
    }
    else if constexpr(N == 4)
    {
        int32_t tmp = llvm_amdgcn_raw_buffer_load_i32(src_wave_buffer_resource,
                                                      src_thread_addr_offset,
                                                      src_wave_addr_offset,
                                                      static_cast<index_t>(coherence));

        return bit_cast<int8x4_t>(tmp);
    }
    else if constexpr(N == 8)
    {
        int32x2_t tmp = llvm_amdgcn_raw_buffer_load_i32x2(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));

        return bit_cast<int8x8_t>(tmp);
    }
    else if constexpr(N == 16)
    {
        int32x4_t tmp = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                          src_thread_addr_offset,
                                                          src_wave_addr_offset,
                                                          static_cast<index_t>(coherence));
        return bit_cast<int8x16_t>(tmp);
    }
    else if constexpr(N == 32)
    {
        int32x4_t tmp0 = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 4 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        vector_type<int32_t, 8> tmp;

        tmp.AsType<int32x4_t>()(Number<0>{}) = tmp0;
        tmp.AsType<int32x4_t>()(Number<1>{}) = tmp1;

        return bit_cast<int8x32_t>(tmp);
    }
    else if constexpr(N == 64)
    {
        int32x4_t tmp0 = llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                                           src_thread_addr_offset,
                                                           src_wave_addr_offset,
                                                           static_cast<index_t>(coherence));
        int32x4_t tmp1 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 4 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        int32x4_t tmp2 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 8 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));
        int32x4_t tmp3 =
            llvm_amdgcn_raw_buffer_load_i32x4(src_wave_buffer_resource,
                                              src_thread_addr_offset,
                                              src_wave_addr_offset + 12 * sizeof(int32_t),
                                              static_cast<index_t>(coherence));

        vector_type<int32_t, 16> tmp;

        tmp.AsType<int32x4_t>()(Number<0>{}) = tmp0;
        tmp.AsType<int32x4_t>()(Number<1>{}) = tmp1;
        tmp.AsType<int32x4_t>()(Number<2>{}) = tmp2;
        tmp.AsType<int32x4_t>()(Number<3>{}) = tmp3;

        return bit_cast<int8x64_t>(tmp);
    }
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type<T, N>::type amd_buffer_load_impl(int32x4_t src_wave_buffer_resource,
                                                                 index_t src_thread_addr_offset,
                                                                 index_t src_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, f8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bf8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, uint8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, pk_i4_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, f4x2_pk_t::type>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    using r_t     = typename vector_type<T, N>::type;
    auto raw_data = amd_buffer_load_impl_raw<sizeof(T) * N, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, src_wave_addr_offset);
    return bit_cast<r_t>(raw_data);
}

template <index_t N, AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void
amd_buffer_store_impl_raw(const typename vector_type<int8_t, N>::type src_thread_data,
                          int32x4_t dst_wave_buffer_resource,
                          index_t dst_thread_addr_offset,
                          index_t dst_wave_addr_offset)
{
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32 || N == 64,
                  "wrong! not implemented");

    if constexpr(N == 1)
    {
        llvm_amdgcn_raw_buffer_store_i8(src_thread_data,
                                        dst_wave_buffer_resource,
                                        dst_thread_addr_offset,
                                        dst_wave_addr_offset,
                                        static_cast<index_t>(coherence));
    }
    else if constexpr(N == 2)
    {

        llvm_amdgcn_raw_buffer_store_i16(bit_cast<int16_t>(src_thread_data),
                                         dst_wave_buffer_resource,
                                         dst_thread_addr_offset,
                                         dst_wave_addr_offset,
                                         static_cast<index_t>(coherence));
    }
    else if constexpr(N == 4)
    {
        llvm_amdgcn_raw_buffer_store_i32(bit_cast<int32_t>(src_thread_data),
                                         dst_wave_buffer_resource,
                                         dst_thread_addr_offset,
                                         dst_wave_addr_offset,
                                         static_cast<index_t>(coherence));
    }
    else if constexpr(N == 8)
    {
        llvm_amdgcn_raw_buffer_store_i32x2(bit_cast<int32x2_t>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 16)
    {
        llvm_amdgcn_raw_buffer_store_i32x4(bit_cast<int32x4_t>(src_thread_data),
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 32)
    {
        vector_type<int32_t, 8> tmp{bit_cast<int32x8_t>(src_thread_data)};

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[Number<0>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[Number<1>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset + sizeof(int32_t) * 4,
                                           static_cast<index_t>(coherence));
    }
    else if constexpr(N == 64)
    {
        vector_type<int32_t, 16> tmp{bit_cast<int32x16_t>(src_thread_data)};

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[Number<0>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset,
                                           static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[Number<1>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset + sizeof(int32_t) * 4,
                                           static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[Number<2>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset + sizeof(int32_t) * 8,
                                           static_cast<index_t>(coherence));

        llvm_amdgcn_raw_buffer_store_i32x4(tmp.template AsType<int32x4_t>()[Number<3>{}],
                                           dst_wave_buffer_resource,
                                           dst_thread_addr_offset,
                                           dst_wave_addr_offset + sizeof(int32_t) * 12,
                                           static_cast<index_t>(coherence));
    }
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_buffer_store_impl(const typename vector_type<T, N>::type src_thread_data,
                                      int32x4_t dst_wave_buffer_resource,
                                      index_t dst_thread_addr_offset,
                                      index_t dst_wave_addr_offset)
{
    static_assert(
        (is_same<T, double>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
            (is_same<T, float>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, half_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bhalf_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, f8_fnuz_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, bf8_fnuz_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, fp8_storage_t>::value &&
             (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
            (is_same<T, int8_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
        "wrong! not implemented");

    using r_t = typename vector_type<int8_t, sizeof(T) * N>::type;

    amd_buffer_store_impl_raw<sizeof(T) * N, coherence>(bit_cast<r_t>(src_thread_data),
                                                        dst_wave_buffer_resource,
                                                        dst_thread_addr_offset,
                                                        dst_wave_addr_offset);
}

template <typename T, index_t N>
__device__ void amd_global_atomic_add_impl(const typename vector_type<T, N>::type src_thread_data,
                                           T* addr)
{
    static_assert((is_same<T, bhalf_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, half_t>::value && (N == 2 || N == 4 || N == 8)),
                  "wrong! not implemented");

    if constexpr(is_same<T, half_t>::value)
    {
        vector_type<half_t, N> tmp{src_thread_data};
        static_for<0, N / 2, 1>{}([&](auto i) {
            __builtin_amdgcn_global_atomic_fadd_v2f16(bit_cast<half2_t*>(addr) + i,
                                                      tmp.template AsType<half2_t>()[i]);
        });
    }
#if defined(__gfx942__) || defined(__gfx950__) || defined(__gfx12__) || defined(__gfx13__)
    else if constexpr(is_same<T, bhalf_t>::value)
    {
        vector_type<bhalf_t, N> tmp{src_thread_data};
        static_for<0, N / 2, 1>{}([&](auto i) {
            __builtin_amdgcn_global_atomic_fadd_v2bf16(bit_cast<bhalf2_t*>(addr) + i,
                                                       tmp.template AsType<bhalf2_t>()[i]);
        });
    }
#endif
}

template <typename T, index_t N>
__device__ void amd_buffer_atomic_add_impl(const typename vector_type<T, N>::type src_thread_data,
                                           int32x4_t dst_wave_buffer_resource,
                                           index_t dst_thread_addr_offset,
                                           index_t dst_wave_addr_offset)
{
    static_assert((is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (is_same<T, half_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");

    if constexpr(is_same<T, float>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp32(src_thread_data,
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            vector_type<float, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(float),
                                                   0);
        }
        else if constexpr(N == 4)
        {
            vector_type<float, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(float),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<2>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 2 * sizeof(float),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_add_fp32(tmp.AsType<float>()[Number<3>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 3 * sizeof(float),
                                                   0);
        }
    }
    else if constexpr(is_same<T, half_t>::value)
    {
        if constexpr(N == 2)
        {
            llvm_amdgcn_raw_buffer_atomic_add_fp16x2(src_thread_data,
                                                     dst_wave_buffer_resource,
                                                     dst_thread_addr_offset,
                                                     dst_wave_addr_offset,
                                                     0);
        }
        else if constexpr(N == 4)
        {
            vector_type<half_t, 4> tmp{src_thread_data};

            static_for<0, 2, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(tmp.AsType<half2_t>()[i],
                                                         dst_wave_buffer_resource,
                                                         dst_thread_addr_offset,
                                                         dst_wave_addr_offset + i * sizeof(half2_t),
                                                         0);
            });
        }
        else if constexpr(N == 8)
        {
            vector_type<half_t, 8> tmp{src_thread_data};

            static_for<0, 4, 1>{}([&](auto i) {
                llvm_amdgcn_raw_buffer_atomic_add_fp16x2(tmp.AsType<half2_t>()[i],
                                                         dst_wave_buffer_resource,
                                                         dst_thread_addr_offset,
                                                         dst_wave_addr_offset + i * sizeof(half2_t),
                                                         0);
            });
        }
    }
    else if constexpr(is_same<T, int32_t>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_add_i32(src_thread_data,
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);
        }
        else if constexpr(N == 2)
        {
            vector_type<int32_t, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<0>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<1>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + sizeof(int32_t),
                                                  0);
        }
        else if constexpr(N == 4)
        {
            vector_type<int32_t, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<0>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset,
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<1>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + sizeof(int32_t),
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<2>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + 2 * sizeof(int32_t),
                                                  0);

            llvm_amdgcn_raw_buffer_atomic_add_i32(tmp.AsType<int32_t>()[Number<3>{}],
                                                  dst_wave_buffer_resource,
                                                  dst_thread_addr_offset,
                                                  dst_wave_addr_offset + 3 * sizeof(int32_t),
                                                  0);
        }
    }
}

template <typename T, index_t N>
__device__ void amd_buffer_atomic_max_impl(const typename vector_type<T, N>::type src_thread_data,
                                           int32x4_t dst_wave_buffer_resource,
                                           index_t dst_thread_addr_offset,
                                           index_t dst_wave_addr_offset)
{
    static_assert((is_same<T, double>::value && (N == 1 || N == 2 || N == 4)),
                  "wrong! not implemented");
    if constexpr(is_same<T, double>::value)
    {
        if constexpr(N == 1)
        {
            llvm_amdgcn_raw_buffer_atomic_max_fp64(src_thread_data,
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);
        }
        else if constexpr(N == 2)
        {
            vector_type<double, 2> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(double),
                                                   0);
        }
        else if constexpr(N == 4)
        {
            vector_type<double, 4> tmp{src_thread_data};

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<0>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset,
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<1>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + sizeof(double),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<2>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 2 * sizeof(double),
                                                   0);

            llvm_amdgcn_raw_buffer_atomic_max_fp64(tmp.AsType<double>()[Number<3>{}],
                                                   dst_wave_buffer_resource,
                                                   dst_thread_addr_offset,
                                                   dst_wave_addr_offset + 3 * sizeof(double),
                                                   0);
        }
    }
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_zero(const T* p_src_wave,
                                            index_t src_thread_element_offset,
                                            bool src_thread_element_valid,
                                            index_t src_element_space_size)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_LOAD_OOB_CHECK_OFFSET_TRICK
    uint32_t src_addr_shift = src_thread_element_valid ? 0 : 0x80000000;
    return amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_addr_shift + src_thread_addr_offset, 0);

#else

    vector_t tmp{amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0)};
    return src_thread_element_valid ? tmp : vector_t(0);
#endif
}

// buffer_load requires:
//   1) p_src_wave must point to global memory space
//   2) p_src_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ typename vector_type_maker<T, N>::type::type
amd_buffer_load_invalid_element_return_customized_value(const T* p_src_wave,
                                                        index_t src_thread_element_offset,
                                                        bool src_thread_element_valid,
                                                        index_t src_element_space_size,
                                                        T customized_value)
{
    const int32x4_t src_wave_buffer_resource =
        make_wave_buffer_resource(p_src_wave, src_element_space_size);

    index_t src_thread_addr_offset = src_thread_element_offset * sizeof(T);

    using vector_t = typename vector_type_maker<T, N>::type::type;
    using scalar_t = typename scalar_type<vector_t>::type;

    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    vector_t tmp{amd_buffer_load_impl<scalar_t, vector_size, coherence>(
        src_wave_buffer_resource, src_thread_addr_offset, 0)};

    return src_thread_element_valid ? tmp : vector_t(customized_value);
}

// buffer_store requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_buffer_store(const typename vector_type_maker<T, N>::type::type src_thread_data,
                                 T* p_dst_wave,
                                 const index_t dst_thread_element_offset,
                                 const bool dst_thread_element_valid,
                                 const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_STORE_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;
    amd_buffer_store_impl<scalar_t, vector_size, coherence>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_store_impl<scalar_t, vector_size, coherence>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// buffer_atomic_add requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ void
amd_buffer_atomic_add(const typename vector_type_maker<T, N>::type::type src_thread_data,
                      T* p_dst_wave,
                      const index_t dst_thread_element_offset,
                      const bool dst_thread_element_valid,
                      const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

    if constexpr(is_same<T, bhalf_t>::value)
    {
        if(dst_thread_element_valid)
        {
            amd_global_atomic_add_impl<scalar_t, vector_size>(
                src_thread_data, p_dst_wave + dst_thread_element_offset);
        }
    }
    else
    {
#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_ADD_OOB_CHECK_OFFSET_TRICK
        uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

        amd_buffer_atomic_add_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
        if(dst_thread_element_valid)
        {
            amd_buffer_atomic_add_impl<scalar_t, vector_size>(
                src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
        }
#endif
    }
}

// buffer_atomic_max requires:
//   1) p_dst_wave must point to global memory
//   2) p_dst_wave must be a wavewise pointer.
// It is user's responsibility to make sure that is true.
template <typename T, index_t N>
__device__ void
amd_buffer_atomic_max(const typename vector_type_maker<T, N>::type::type src_thread_data,
                      T* p_dst_wave,
                      const index_t dst_thread_element_offset,
                      const bool dst_thread_element_valid,
                      const index_t dst_element_space_size)
{
    const int32x4_t dst_wave_buffer_resource =
        make_wave_buffer_resource(p_dst_wave, dst_element_space_size);

    index_t dst_thread_addr_offset = dst_thread_element_offset * sizeof(T);

    using vector_t                = typename vector_type_maker<T, N>::type::type;
    using scalar_t                = typename scalar_type<vector_t>::type;
    constexpr index_t vector_size = scalar_type<vector_t>::vector_size;

#if CK_EXPERIMENTAL_USE_BUFFER_ATOMIC_MAX_OOB_CHECK_OFFSET_TRICK
    uint32_t dst_addr_shift = dst_thread_element_valid ? 0 : 0x80000000;

    amd_buffer_atomic_max_impl<scalar_t, vector_size>(
        src_thread_data, dst_wave_buffer_resource, dst_addr_shift + dst_thread_addr_offset, 0);
#else
    if(dst_thread_element_valid)
    {
        amd_buffer_atomic_max_impl<scalar_t, vector_size>(
            src_thread_data, dst_wave_buffer_resource, dst_thread_addr_offset, 0);
    }
#endif
}

// Direct loads from global to LDS.
__device__ void
llvm_amdgcn_raw_buffer_load_lds(int32x4_t rsrc,
                                __attribute__((address_space(3))) uint32_t* lds_ptr,
                                index_t size,
                                index_t voffset,
                                index_t soffset,
                                index_t offset,
                                index_t aux) __asm("llvm.amdgcn.raw.buffer.load.lds");

#ifndef __HIPCC_RTC__
template <typename T, index_t NumElemsPerThread>
__device__ void amd_direct_load_global_to_lds(const T* global_base_ptr,
                                              const index_t global_offset,
                                              T* lds_base_ptr,
                                              const index_t lds_offset,
                                              const bool is_valid,
                                              const index_t src_element_space_size)
{
    // Direct loads require that each thread reads and writes exactly a single DWORD.
    constexpr auto bytes_per_thread = sizeof(T) * NumElemsPerThread;
#if defined(__gfx950__)
    constexpr auto dword_bytes = 4;
    static_assert(bytes_per_thread == dword_bytes || bytes_per_thread == dword_bytes * 3 ||
                  bytes_per_thread == dword_bytes * 4);
#elif defined(__gfx942__)
    constexpr auto dword_bytes = 4;
    static_assert(bytes_per_thread == dword_bytes);
#endif

    const int32x4_t src_resource =
        make_wave_buffer_resource(global_base_ptr, src_element_space_size);
    const index_t global_offset_bytes = is_valid ? global_offset * sizeof(T) : 0x80000000;

#if CK_USE_AMD_LDS_DIRECT_LOAD_INLINE_ASM
    T* lds_ptr = lds_base_ptr + lds_offset;
#ifndef CK_CODE_GEN_RTC
    auto const lds_ptr_sgpr =
        __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(lds_ptr)));
#else
    auto const lds_ptr_sgpr = __builtin_amdgcn_readfirstlane((reinterpret_cast<size_t>(lds_ptr)));
#endif
    asm volatile("s_mov_b32 m0, %0; \n\t"
                 "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
                 "v"(global_offset_bytes),
                 "s"(src_resource)
                 : "memory");
#else
    // LDS pointer must be attributed with the LDS address space.
    __attribute__((address_space(3))) uint32_t* lds_ptr =
#ifndef CK_CODE_GEN_RTC
        reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(
            reinterpret_cast<uintptr_t>(lds_base_ptr + lds_offset));
#else
        reinterpret_cast<__attribute__((address_space(3))) uint32_t*>(
            reinterpret_cast<size_t>(lds_base_ptr + lds_offset));
#endif

    llvm_amdgcn_raw_buffer_load_lds(
        src_resource, lds_ptr, bytes_per_thread, global_offset_bytes, 0, 0, 0);
#endif
}
#endif

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
          bool multicast                   = false>
__device__ void amd_async_copy_to_lds_impl_raw(__attribute__((address_space(1))) const T* src_ptr,
                                               __attribute__((address_space(3))) T* dst_ptr)
{
#if defined(__gfx13__)
    if constexpr(N == 1)
    {
        __attribute__((address_space(1))) char* global_ptr =
            const_cast<__attribute__((address_space(1))) char*>(
                reinterpret_cast<const __attribute__((address_space(1))) char*>(src_ptr));
        __attribute__((address_space(3))) char* lds_ptr =
            reinterpret_cast<__attribute__((address_space(3))) char*>(dst_ptr);

        if constexpr(multicast)
        {
            __builtin_amdgcn_cluster_load_async_to_lds_b8(global_ptr, lds_ptr, 0, 0, 0xf);
        }
        else
        {
            __builtin_amdgcn_global_load_async_to_lds_b8(
                global_ptr, lds_ptr, 0, static_cast<index_t>(coherence));
        }
        return;
    }

    if constexpr(N == 4)
    {
        __attribute__((address_space(1))) int* global_ptr =
            const_cast<__attribute__((address_space(1))) int*>(
                reinterpret_cast<const __attribute__((address_space(1))) int*>(src_ptr));
        __attribute__((address_space(3))) int* lds_ptr =
            reinterpret_cast<__attribute__((address_space(3))) int*>(dst_ptr);

        if constexpr(multicast)
        {
            __builtin_amdgcn_cluster_load_async_to_lds_b32(global_ptr, lds_ptr, 0, 0, 0xf);
        }
        else
        {
            __builtin_amdgcn_global_load_async_to_lds_b32(
                global_ptr, lds_ptr, 0, static_cast<index_t>(coherence));
        }
        return;
    }

    if constexpr(N == 8)
    {
        __attribute__((address_space(1))) int32x2_t* global_ptr =
            const_cast<__attribute__((address_space(1))) int32x2_t*>(
                reinterpret_cast<const __attribute__((address_space(1))) int32x2_t*>(src_ptr));
        __attribute__((address_space(3))) int32x2_t* lds_ptr =
            reinterpret_cast<__attribute__((address_space(3))) int32x2_t*>(dst_ptr);

        if constexpr(multicast)
        {
            __builtin_amdgcn_cluster_load_async_to_lds_b64(global_ptr, lds_ptr, 0, 0, 0xf);
        }
        else
        {
            __builtin_amdgcn_global_load_async_to_lds_b64(
                global_ptr, lds_ptr, 0, static_cast<index_t>(coherence));
        }
        return;
    }

    if constexpr(N == 16)
    {
        __attribute__((address_space(1))) int32x4_t* global_ptr =
            const_cast<__attribute__((address_space(1))) int32x4_t*>(
                reinterpret_cast<const __attribute__((address_space(1))) int32x4_t*>(src_ptr));
        __attribute__((address_space(3))) int32x4_t* lds_ptr =
            reinterpret_cast<__attribute__((address_space(3))) int32x4_t*>(dst_ptr);
        if constexpr(multicast)
        {
            __builtin_amdgcn_cluster_load_async_to_lds_b128(global_ptr, lds_ptr, 0, 0, 0xf);
        }
        else
        {
            __builtin_amdgcn_global_load_async_to_lds_b128(
                global_ptr, lds_ptr, 0, static_cast<index_t>(coherence));
        }
        return;
    }
#else
    ignore = src_ptr;
    ignore = dst_ptr;
#endif
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_async_store_to_global_impl_raw(__attribute__((address_space(3)))
                                                   const T* src_ptr,
                                                   __attribute__((address_space(1))) T* dst_ptr)
{
#if defined(__gfx13__)
    if constexpr(N == 1)
    {
        __attribute__((address_space(3))) char* lds_ptr =
            const_cast<__attribute__((address_space(3))) char*>(
                reinterpret_cast<const __attribute__((address_space(3))) char*>(src_ptr));
        __attribute__((address_space(1))) char* global_ptr =
            reinterpret_cast<__attribute__((address_space(1))) char*>(dst_ptr);
        __builtin_amdgcn_global_store_async_from_lds_b8(
            global_ptr, lds_ptr, 0, static_cast<index_t>(coherence));
        return;
    }

    if constexpr(N == 4)
    {
        __attribute__((address_space(3))) int* lds_ptr =
            const_cast<__attribute__((address_space(3))) int*>(
                reinterpret_cast<const __attribute__((address_space(3))) int*>(src_ptr));
        __attribute__((address_space(1))) int* global_ptr =
            reinterpret_cast<__attribute__((address_space(1))) int*>(dst_ptr);
        __builtin_amdgcn_global_store_async_from_lds_b32(
            global_ptr, lds_ptr, 0, static_cast<index_t>(coherence));
        return;
    }

    if constexpr(N == 8)
    {
        __attribute__((address_space(3))) int32x2_t* lds_ptr =
            const_cast<__attribute__((address_space(3))) int32x2_t*>(
                reinterpret_cast<const __attribute__((address_space(3))) int32x2_t*>(src_ptr));
        __attribute__((address_space(1))) int32x2_t* global_ptr =
            reinterpret_cast<__attribute__((address_space(1))) int32x2_t*>(dst_ptr);
        __builtin_amdgcn_global_store_async_from_lds_b64(
            global_ptr, lds_ptr, 0, static_cast<index_t>(coherence));
        return;
    }

    if constexpr(N == 16)
    {
        __attribute__((address_space(3))) int32x4_t* lds_ptr =
            const_cast<__attribute__((address_space(3))) int32x4_t*>(
                reinterpret_cast<const __attribute__((address_space(3))) int32x4_t*>(src_ptr));
        __attribute__((address_space(1))) int32x4_t* global_ptr =
            reinterpret_cast<__attribute__((address_space(1))) int32x4_t*>(dst_ptr);
        __builtin_amdgcn_global_store_async_from_lds_b128(
            global_ptr, lds_ptr, 0, static_cast<index_t>(coherence));
        return;
    }
#else
    ignore = src_ptr;
    ignore = dst_ptr;
#endif
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
          bool multicast                   = false>
__device__ void amd_async_copy_to_lds_impl(__attribute__((address_space(1))) const T* src_ptr,
                                           __attribute__((address_space(3))) T* dst_ptr)
{
    // currently only support to b8, b32, b64, b128 when one async copy
    static_assert((is_same<T, double>::value && (N == 1 || N == 2)) ||
                      (is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)) ||
                      (is_same<T, half_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, bhalf_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, f8_t>::value && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (is_same<T, bf8_t>::value && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (is_same<T, int8_t>::value && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (is_same<T, uint8_t>::value && (N == 1 || N == 4 || N == 8 || N == 16)),
                  "wrong! not implemented");

    amd_async_copy_to_lds_impl_raw<T, sizeof(T) * N, coherence, multicast>(src_ptr, dst_ptr);
    return;
}

template <typename T,
          index_t N,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_async_store_to_global_impl(__attribute__((address_space(3))) const T* src_ptr,
                                               __attribute__((address_space(1))) T* dst_ptr)
{
    // currently only support to b8, b32, b64, b128 when one async copy
    static_assert((is_same<T, double>::value && (N == 1 || N == 2)) ||
                      (is_same<T, float>::value && (N == 1 || N == 2 || N == 4)) ||
                      (is_same<T, int32_t>::value && (N == 1 || N == 2 || N == 4)) ||
                      (is_same<T, half_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, bhalf_t>::value && (N == 2 || N == 4 || N == 8)) ||
                      (is_same<T, f8_t>::value && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (is_same<T, bf8_t>::value && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (is_same<T, int8_t>::value && (N == 1 || N == 4 || N == 8 || N == 16)) ||
                      (is_same<T, uint8_t>::value && (N == 1 || N == 4 || N == 8 || N == 16)),
                  "wrong! not implemented");

    amd_async_store_to_global_impl_raw<T, sizeof(T) * N, coherence>(src_ptr, dst_ptr);
    return;
}

template <typename T,
          index_t NumElemsPerThread,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence,
          bool multicast                   = false>
__device__ void amd_async_load_global_to_lds(const T* global_base_ptr,
                                             const index_t global_offset,
                                             T* lds_base_ptr,
                                             const index_t lds_offset,
                                             const bool is_src_valid,
                                             const bool is_dst_valid)
{
    if(is_src_valid && is_dst_valid)
    {
#if defined(__gfx13__)
        const index_t in_global_offset = global_offset;
        __attribute__((address_space(1))) const T* global_ptr =
            reinterpret_cast<__attribute__((address_space(1))) T*>(
                reinterpret_cast<uintptr_t>(global_base_ptr + in_global_offset));
        __attribute__((address_space(3))) T* lds_ptr =
            reinterpret_cast<__attribute__((address_space(3))) T*>(
                reinterpret_cast<uintptr_t>(lds_base_ptr + lds_offset));
        amd_async_copy_to_lds_impl<T, NumElemsPerThread, coherence, multicast>(global_ptr, lds_ptr);
#else
        ignore = global_base_ptr;
        ignore = global_offset;
#endif
    }
    else
    {
        if(is_dst_valid)
        {
            using DstVecType    = typename vector_type_maker<T, NumElemsPerThread>::type;
            DstVecType* lds_ptr = reinterpret_cast<DstVecType*>(lds_base_ptr + lds_offset);
            *lds_ptr            = {};
        }
        else
        {
            return; // do nothing
        }
    }
}

template <typename T,
          index_t NumElemsPerThread,
          AmdBufferCoherenceEnum coherence = AmdBufferCoherenceEnum::DefaultCoherence>
__device__ void amd_async_store_lds_to_global(const T* lds_base_ptr,
                                              const index_t lds_offset,
                                              T* global_base_ptr,
                                              const index_t global_offset,
                                              const bool is_src_valid,
                                              const bool is_dst_valid)
{
    if(is_src_valid && is_dst_valid)
    {
#if defined(__gfx13__)
        __attribute__((address_space(3))) const T* lds_ptr =
            reinterpret_cast<__attribute__((address_space(3))) T*>(
                reinterpret_cast<uintptr_t>(lds_base_ptr + lds_offset));
        __attribute__((address_space(1))) T* global_ptr =
            reinterpret_cast<__attribute__((address_space(1))) T*>(
                reinterpret_cast<uintptr_t>(global_base_ptr + global_offset));
        amd_async_store_to_global_impl<T, NumElemsPerThread, coherence>(lds_ptr, global_ptr);
#else
        ignore = lds_base_ptr;
        ignore = lds_offset;
        ignore = global_base_ptr;
        ignore = global_offset;
#endif
    }
}

template <typename T, index_t N, index_t NumThreadsPerTile, index_t NumVgprsPerTile>
__device__ auto amd_tiled_load_to_vgpr(__attribute__((address_space(1))) const T* in_ptr,
                                       bool is_src_valid)
{
    using vector_t = typename vector_type_maker<T, N>::type::type;
#if defined(__gfx13__)
    if(is_src_valid)
    {
        if constexpr(NumThreadsPerTile == 2)
        {
            __attribute__((address_space(1))) int32_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32_t*>(in_ptr));
            return bit_cast<vector_t>(__builtin_amdgcn_global_tiled_load_half_b64(global_ptr));
        }
        else if constexpr(NumThreadsPerTile == 4 && NumVgprsPerTile == 1)
        {
            __attribute__((address_space(1))) int32_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32_t*>(in_ptr));
            return bit_cast<vector_t>(__builtin_amdgcn_global_tiled_load_qtr_b128(global_ptr));
        }
        /* Remove for padding 4x4 + normal 4x4 */
        /*
        else if constexpr(NumThreadsPerTile == 2 && NumVgprsPerTile == 2)
        {
            __attribute__((address_space(1))) int32x2_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32x2_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32x2_t*>(in_ptr));
            return bit_cast<vector_t>(__builtin_amdgcn_global_tiled_load_b64(global_ptr));
        }
        */
        else
        {
            static_assert(0, "wrong! not implemented");
        }
    }
    else
    {
        return vector_t{0};
    }
#else
    ignore = in_ptr;
    ignore = is_src_valid;
    return vector_t{0};
#endif
}

template <typename T, index_t N, index_t NumThreadsPerTile, index_t NumVgprsPerTile>
__device__ auto amd_ds_tiled_load_to_vgpr(__attribute__((address_space(3))) const T* in_ptr,
                                          bool is_src_valid)
{
    using vector_t = typename vector_type_maker<T, N>::type::type;
#if defined(__gfx13__)
    if(is_src_valid)
    {
        if constexpr(NumThreadsPerTile == 2)
        {
            __attribute__((address_space(3))) int32_t* global_ptr =
                const_cast<__attribute__((address_space(3))) int32_t*>(
                    reinterpret_cast<const __attribute__((address_space(3))) int32_t*>(in_ptr));
            return bit_cast<vector_t>(__builtin_amdgcn_ds_tiled_load_half_b64(global_ptr));
        }
        else if constexpr(NumThreadsPerTile == 4 && NumVgprsPerTile == 4)
        {
            __attribute__((address_space(3))) int32x4_t* global_ptr =
                const_cast<__attribute__((address_space(3))) int32x4_t*>(
                    reinterpret_cast<const __attribute__((address_space(3))) int32x4_t*>(in_ptr));
            return bit_cast<vector_t>(__builtin_amdgcn_ds_tiled_load_b128(global_ptr));
        }
        else
        {
            static_assert(0, "wrong! not implemented");
        }
    }
    else
    {
        return vector_t{0};
    }
#else
    ignore = in_ptr;
    ignore = is_src_valid;
    return vector_t{0};
#endif
}

template <typename T, index_t N, index_t NumThreadsPerTile, index_t NumVgprsPerTile>
__device__ void
amd_tile_store_to_buffer(const typename vector_type_maker<T, N>::type::type src_thread_data,
                         __attribute__((address_space(1))) const T* in_ptr)
{
#if defined(__gfx13__)
    if constexpr((NumThreadsPerTile == 2) && (NumVgprsPerTile == 2))
    {
        // 8X4X8
        if constexpr((is_same<T, half_t>::value && (N == 8)) ||
                     (is_same<T, bhalf_t>::value && (N == 8)))
        {
            vector_type<int32_t, 4> tmp{bit_cast<int32x4_t>(src_thread_data)};
            int32x3_t store_value_up;
            store_value_up[0] = tmp.AsType<int32_t>()[Number<0>{}];
            store_value_up[1] = tmp.AsType<int32_t>()[Number<1>{}];
            store_value_up[2] = tmp.AsType<int32_t>()[Number<2>{}];
            __attribute__((address_space(1))) int32x3_t* global_ptr_up =
                const_cast<__attribute__((address_space(1))) int32x3_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32x3_t*>(in_ptr));
            __builtin_amdgcn_global_tiled_store_vst2_b64(store_value_up, global_ptr_up);

            int32x3_t store_value_down;
            store_value_down[0] = tmp.AsType<int32_t>()[Number<1>{}];
            store_value_down[1] = tmp.AsType<int32_t>()[Number<2>{}];
            store_value_down[2] = tmp.AsType<int32_t>()[Number<3>{}];
            __attribute__((address_space(1)))
            int32x3_t* global_ptr_down = const_cast<__attribute__((address_space(1))) int32x3_t*>(
                reinterpret_cast<const __attribute__((address_space(1))) int32x3_t*>(in_ptr + 4));
            __builtin_amdgcn_global_tiled_store_vst2_b64(store_value_down, global_ptr_down);
        }
        else if constexpr((is_same<T, f8_t>::value && (N == 8)) ||
                          (is_same<T, bf8_t>::value && (N == 8)) ||
                          (is_same<T, int8_t>::value && (N == 8)))
        {
            __attribute__((address_space(1))) int32x2_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32x2_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32x2_t*>(in_ptr));
            __builtin_amdgcn_global_tiled_store_b64(bit_cast<int32x2_t>(src_thread_data),
                                                    global_ptr);
        }
        else
        {
            static_assert(0, "wrong! not implemented");
        }
    }
    else if constexpr((NumThreadsPerTile == 2) && (NumVgprsPerTile == 1))
    {
        if constexpr((is_same<T, half_t>::value && (N == 8)) ||
                     (is_same<T, bhalf_t>::value && (N == 8)))
        {
            vector_type<int32_t, 4> tmp{bit_cast<int32x4_t>(src_thread_data)};
            static_for<0, 4, 1>{}([&](auto i) {
                __attribute__((address_space(1))) int32_t* global_ptr =
                    const_cast<__attribute__((address_space(1))) int32_t*>(
                        reinterpret_cast<const __attribute__((address_space(1))) int32_t*>(in_ptr +
                                                                                           i * 4));
                __builtin_amdgcn_global_tiled_store_half_b64(tmp.AsType<int32_t>()[i], global_ptr);
            });
        }
        else if constexpr((is_same<T, f8_t>::value && (N == 8)) ||
                          (is_same<T, bf8_t>::value && (N == 8)) ||
                          (is_same<T, int8_t>::value && (N == 8)))
        {
            vector_type<int32_t, 2> tmp{bit_cast<int32x2_t>(src_thread_data)};
            static_for<0, 2, 1>{}([&](auto i) {
                __attribute__((address_space(1))) int32_t* global_ptr =
                    const_cast<__attribute__((address_space(1))) int32_t*>(
                        reinterpret_cast<const __attribute__((address_space(1))) int32_t*>(in_ptr +
                                                                                           i * 8));
                __builtin_amdgcn_global_tiled_store_half_b64(tmp.AsType<int32_t>()[i], global_ptr);
            });
        }
        else
        {
            static_assert(0, "wrong! not implemented");
        }
    }
    else if constexpr((NumThreadsPerTile == 4) && (NumVgprsPerTile == 1))
    {
        // 4x2x16
        if constexpr(is_same<T, half_t>::value || is_same<T, bhalf_t>::value)
        {
            vector_type<int32_t, 2> tmp{bit_cast<int32x2_t>(src_thread_data)};
            static_for<0, 2, 1>{}([&](auto i) {
                __attribute__((address_space(1))) int32_t* global_ptr =
                    const_cast<__attribute__((address_space(1))) int32_t*>(
                        reinterpret_cast<const __attribute__((address_space(1))) int32_t*>(in_ptr +
                                                                                           i * 8));
                __builtin_amdgcn_global_tiled_store_qtr_b128(tmp.AsType<int32_t>()[i], global_ptr);
            });
        }
        else if constexpr(is_same<T, f8_t>::value || is_same<T, bf8_t>::value ||
                          is_same<T, int8_t>::value)
        {
            __attribute__((address_space(1))) int32_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32_t*>(in_ptr));
            __builtin_amdgcn_global_tiled_store_qtr_b128(bit_cast<int32_t>(src_thread_data),
                                                         global_ptr);
        }
        else
        {
            static_assert(0, "wrong! not implemented");
        }
    }
    else
    {
        static_assert(0, "wrong! The shape is not supported yet.");
    }
#else
    ignore = src_thread_data;
    ignore = in_ptr;
#endif
}

template <typename T, index_t N, AddressSpaceEnum BufferAddressSpace>
__device__ auto
amd_wgp_multicast_load_to_vgpr(__attribute__((address_space(1))) const T* in_ptr,
                               typename vector_type<remove_cvref_t<T>, N>::type& out,
                               bool is_src_valid)
{
#if defined(__gfx13__)
    if(is_src_valid)
    {
        if constexpr(is_same_v<remove_cvref_t<T>, ck::half_t> && N == 2)
        {
            __attribute__((address_space(1))) int32_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32_t*>(in_ptr));

            __attribute__((address_space(10))) int32_t* lane_vgpr_ptr =
                reinterpret_cast<__attribute__((address_space(10))) int32_t*>(
                    reinterpret_cast<uintptr_t>(&out));
            __builtin_amdgcn_load_mcast_b32(lane_vgpr_ptr, global_ptr, 0, 0x3 << 18);
        }
        else if constexpr(is_same_v<remove_cvref_t<T>, ck::half_t> && N == 4)
        {
            __attribute__((address_space(1))) int32x2_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32x2_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32x2_t*>(in_ptr));

            __attribute__((address_space(10))) int32x2_t* lane_vgpr_ptr =
                reinterpret_cast<__attribute__((address_space(10))) int32x2_t*>(
                    reinterpret_cast<uintptr_t>(&out));
            __builtin_amdgcn_load_mcast_b64(lane_vgpr_ptr, global_ptr, 0, 0x3 << 18);
        }
        else if constexpr(is_same_v<remove_cvref_t<T>, ck::half_t> && N == 8)
        {
            __attribute__((address_space(1))) int32x4_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32x4_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32x4_t*>(in_ptr));

            __attribute__((address_space(10))) int32x4_t* lane_vgpr_ptr =
                reinterpret_cast<__attribute__((address_space(10))) int32x4_t*>(
                    reinterpret_cast<uintptr_t>(&out));
            __builtin_amdgcn_load_mcast_b128(lane_vgpr_ptr, global_ptr, 0, 0x3 << 18);
        }
        else
        {
            // To add for other types.
        }
    }
    else
    {
        using vector_t = typename vector_type<remove_cvref_t<T>, N>::type;
        out            = vector_t{0};
    }
#else
    ignore = in_ptr;
    ignore = out;
    ignore = is_src_valid;
#endif
    return;
}

template <typename T, index_t N, AddressSpaceEnum BufferAddressSpace>
__device__ auto amd_cluster_multicast_load_to_vgpr(__attribute__((address_space(1)))
                                                   const T* in_ptr,
                                                   bool is_src_valid)
{
    using vector_t = typename vector_type_maker<T, N>::type::type;
#if defined(__gfx13__)
    if(is_src_valid)
    {
        if constexpr(is_same_v<remove_cvref_t<T>, ck::half_t> && N == 2)
        {
            __attribute__((address_space(1))) int32_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32_t*>(in_ptr));
            return bit_cast<vector_t>(__builtin_amdgcn_cluster_load_b32(global_ptr, 0, 0xf));
        }
        else if constexpr(is_same_v<remove_cvref_t<T>, ck::half_t> && N == 4)
        {
            __attribute__((address_space(1))) int32x2_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32x2_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32x2_t*>(in_ptr));
            return bit_cast<vector_t>(__builtin_amdgcn_cluster_load_b64(global_ptr, 0, 0xf));
        }
        else if constexpr(is_same_v<remove_cvref_t<T>, ck::half_t> && N == 8)
        {
            __attribute__((address_space(1))) int32x4_t* global_ptr =
                const_cast<__attribute__((address_space(1))) int32x4_t*>(
                    reinterpret_cast<const __attribute__((address_space(1))) int32x4_t*>(in_ptr));
            return bit_cast<vector_t>(__builtin_amdgcn_cluster_load_b128(global_ptr, 0, 0xf));
        }
        else
        {
            // To add for other types
            return vector_t{0};
        }
    }
    else
    {
        return vector_t{0};
    }
#else
    ignore = in_ptr;
    ignore = is_src_valid;
    return vector_t{0};
#endif
}
} // namespace ck
