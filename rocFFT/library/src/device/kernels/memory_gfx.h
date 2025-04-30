/******************************************************************************
 * Copyright 2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *****************************************************************************/
/*! \file
    \brief Architecture-specific operators on memory added for GFX9
*/
// reference:
//   https://github.com/llvm/llvm-project/blob/main/llvm/test/CodeGen/AMDGPU/llvm.amdgcn.raw.buffer.load.ll

#ifndef INTRINSIC_MEM_ACCESS_H
#define INTRINSIC_MEM_ACCESS_H

#if defined(__clang__) && defined(__HIP__)

#if(defined(__NVCC__) || defined(__HIPCC__)) \
    || (defined(__clang__) && (defined(__CUDA__)) || defined(__HIP__))
#define ROCFFT_DEVICE __forceinline__ __device__
#elif defined(__CUDACC_RTC__)
#define ROCFFT_DEVICE __forceinline__ __device__
#else
#define ROCFFT_DEVICE inline
#endif

#if defined(__gfx803__) || defined(__gfx900__) || defined(__gfx906__) || defined(__gfx908__) \
    || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__)                     \
    || defined(__gfx942__) // test device
#define USE_GFX_BUFFER_INTRINSIC
#define BUFFER_RESOURCE_3RD_DWORD 0x00020000
#elif defined(__gfx1030__) // special device
#define USE_GFX_BUFFER_INTRINSIC
#define BUFFER_RESOURCE_3RD_DWORD 0x31014000
#else // not support
#define BUFFER_RESOURCE_3RD_DWORD -1
#endif

/// Controls AMD gfx arch cache operations
struct CacheOperation
{
    enum Kind
    {
        /// Cache at all levels - accessed again
        Always,
        /// Cache at global level; glc = 1
        Global,
        /// Streaming - likely to be accessed once; slc = 1
        Streaming,
        /// Indicates the line will not be used again, glc = 1; slc = 1
        LastUse
    };
};

using float16_t = rocfft_fp16;
using float32_t = float;

template <typename T, int N>
struct NativeVector
{
    using type = T __attribute__((ext_vector_type(N)));
};

// template <int N>
// struct NativeVector<cutlass::half_t, N>
// {
//   using type = typename NativeVector<float16_t, N>::type;
// };

// template <int N>
// struct NativeVector<cutlass::bfloat16_t, N>
// {
//   using type = typename NativeVector<float16_t, N>::type;
// };

using float32x2_t = NativeVector<float, 2>::type;
using float32x4_t = NativeVector<float, 4>::type;

using int32x4_t = NativeVector<int, 4>::type;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct alignas(16) BufferResource
{
    union Desc
    {
        int32x4_t d128;
        void*     d64[2];
        uint32_t  d32[4];
    };

    ROCFFT_DEVICE
    BufferResource(void const* base_addr, uint32_t num_records = (0xFFFFFFFF - 1))
    {
        // Reference:
        //   For CDNA: see section 9.1.8 in the AMD resources
        //   https://developer.amd.com/wp-content/resources/CDNA1_Shader_ISA_14December2020.pdf
        //   For RDNA: see section 8.1.8 in the AMD resources
        //   https://developer.amd.com/wp-content/resources/RDNA2_Shader_ISA_November2020.pdf
        //   The d32[3] field represents the 0x[127] ~ [96]

        // 64-bit base address
        desc_.d64[0] = const_cast<void*>(base_addr);
        // 32-bit number of records in bytes which is used to guard against out-of-range access
        desc_.d32[2] = num_records;
        // 32-bit buffer resource descriptor
        desc_.d32[3] = BUFFER_RESOURCE_3RD_DWORD;
    }

    ROCFFT_DEVICE
    operator int32x4_t()
    {
        // return desc_.d128; // NOTE HIP: Crashes compiler; see below

        /// This hack is to enforce scalarization of the variable "base_addr", where in some
        /// circumstances it becomes vectorized and then in turn causes illegal lowering to GCN ISA
        /// since compiler effectively tries to stuff VGPRs in slots where it only accepts SGPRs
        Desc ret;
        ret.d32[0] = __builtin_amdgcn_readfirstlane(desc_.d32[0]);
        ret.d32[1] = __builtin_amdgcn_readfirstlane(desc_.d32[1]);
        ret.d64[1] = desc_.d64[1];
        return ret.d128;
        ///
    }

    Desc desc_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

///
/// Load
///

// 1 byte
__device__ char
    llvm_amdgcn_raw_buffer_load_i8(int32x4_t buffer_resource,
                                   uint32_t  voffset,
                                   uint32_t  soffset,
                                   int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.load.i8");

// 2 bytes
__device__ float16_t
    llvm_amdgcn_raw_buffer_load_f16(int32x4_t buffer_resource,
                                    uint32_t  voffset,
                                    uint32_t  soffset,
                                    int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.load.f16");

// 4 bytes
__device__ float32_t
    llvm_amdgcn_raw_buffer_load_f32(int32x4_t buffer_resource,
                                    uint32_t  voffset,
                                    uint32_t  soffset,
                                    int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.load.f32");

// 8 bytes
__device__ float32x2_t
    llvm_amdgcn_raw_buffer_load_f32x2(int32x4_t buffer_resource,
                                      uint32_t  voffset,
                                      uint32_t  soffset,
                                      int32_t cache_op) __asm("llvm.amdgcn.raw.buffer.load.v2f32");

// 16 bytes
__device__ float32x4_t
    llvm_amdgcn_raw_buffer_load_f32x4(int32x4_t buffer_resource,
                                      uint32_t  voffset,
                                      uint32_t  soffset,
                                      int32_t cache_op) __asm("llvm.amdgcn.raw.buffer.load.v4f32");

///
/// Store
///

// 1 byte
__device__ void
    llvm_amdgcn_raw_buffer_store_i8(char      data,
                                    int32x4_t buffer_resource,
                                    uint32_t  voffset,
                                    uint32_t  soffset,
                                    int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.store.i8");

// 2 bytes
__device__ void
    llvm_amdgcn_raw_buffer_store_f16(float16_t data,
                                     int32x4_t buffer_resource,
                                     uint32_t  voffset,
                                     uint32_t  soffset,
                                     int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.store.f16");

// 4 bytes
__device__ void
    llvm_amdgcn_raw_buffer_store_f32(float32_t data,
                                     int32x4_t buffer_resource,
                                     uint32_t  voffset,
                                     uint32_t  soffset,
                                     int32_t   cache_op) __asm("llvm.amdgcn.raw.buffer.store.f32");

// 8 bytes
__device__ void llvm_amdgcn_raw_buffer_store_f32x2(
    float32x2_t data,
    int32x4_t   buffer_resource,
    uint32_t    voffset,
    uint32_t    soffset,
    int32_t     cache_op) __asm("llvm.amdgcn.raw.buffer.store.v2f32");

// 16 bytes
__device__ void llvm_amdgcn_raw_buffer_store_f32x4(
    float32x4_t data,
    int32x4_t   buffer_resource,
    uint32_t    voffset,
    uint32_t    soffset,
    int32_t     cache_op) __asm("llvm.amdgcn.raw.buffer.store.v4f32");

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The bytes of loading
    int LoadBytes,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct buffer_load;

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 1, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset  = pred_guard ? voffset : -1;
        char ret = llvm_amdgcn_raw_buffer_load_i8(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset  = pred_guard ? voffset : -1;
        char ret = llvm_amdgcn_raw_buffer_load_i8(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 2, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset       = pred_guard ? voffset : -1;
        float16_t ret = llvm_amdgcn_raw_buffer_load_f16(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset       = pred_guard ? voffset : -1;
        float16_t ret = llvm_amdgcn_raw_buffer_load_f16(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 4, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset       = pred_guard ? voffset : -1;
        float32_t ret = llvm_amdgcn_raw_buffer_load_f32(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset       = pred_guard ? voffset : -1;
        float32_t ret = llvm_amdgcn_raw_buffer_load_f32(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 8, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset         = pred_guard ? voffset : -1;
        float32x2_t ret = llvm_amdgcn_raw_buffer_load_f32x2(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset         = pred_guard ? voffset : -1;
        float32x2_t ret = llvm_amdgcn_raw_buffer_load_f32x2(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_load<AccessType, 16, cache_op>
{
    ROCFFT_DEVICE
    buffer_load() {}

    ROCFFT_DEVICE
    buffer_load(
        AccessType& D, void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset         = pred_guard ? voffset : -1;
        float32x4_t ret = llvm_amdgcn_raw_buffer_load_f32x4(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        D = *reinterpret_cast<AccessType*>(&ret);
    }

    ROCFFT_DEVICE
    AccessType load(void const* base_ptr, uint32_t voffset, uint32_t soffset, bool pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset         = pred_guard ? voffset : -1;
        float32x4_t ret = llvm_amdgcn_raw_buffer_load_f32x4(
            buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
        return *reinterpret_cast<AccessType*>(&ret);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Fragment type to store loaded data
    typename AccessType,
    /// The width of loading
    int NumElements,
    /// Cache operation
    CacheOperation::Kind cache_op = CacheOperation::Always>
struct buffer_store;

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 1, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset   = pred_guard ? voffset : -1;
        char data = *reinterpret_cast<char const*>(&D);
        llvm_amdgcn_raw_buffer_store_i8(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 2, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset        = pred_guard ? voffset : -1;
        float16_t data = *reinterpret_cast<float16_t const*>(&D);
        llvm_amdgcn_raw_buffer_store_f16(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 4, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset        = pred_guard ? voffset : -1;
        float32_t data = *reinterpret_cast<float32_t const*>(&D);
        llvm_amdgcn_raw_buffer_store_f32(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 8, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset          = pred_guard ? voffset : -1;
        float32x2_t data = *reinterpret_cast<float32x2_t const*>(&D);
        llvm_amdgcn_raw_buffer_store_f32x2(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

template <typename AccessType, CacheOperation::Kind cache_op>
struct buffer_store<AccessType, 16, cache_op>
{
    ROCFFT_DEVICE
    buffer_store(const AccessType& D,
                 void const*       base_ptr,
                 uint32_t          voffset,
                 uint32_t          soffset,
                 bool              pred_guard)
    {
        BufferResource buffer_rsc(base_ptr);
        voffset          = pred_guard ? voffset : -1;
        float32x4_t data = *reinterpret_cast<float32x4_t const*>(&D);
        llvm_amdgcn_raw_buffer_store_f32x4(
            data, buffer_rsc, voffset, __builtin_amdgcn_readfirstlane(soffset), cache_op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

#endif // defined(__clang__) && defined(__HIP__)

#endif // INTRINSIC_MEM_ACCESS_H
