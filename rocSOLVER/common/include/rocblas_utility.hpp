/* **************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <cmath>
#include <complex>
#include <exception>
#include <new>
#include <type_traits>

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

// concaternate the two arguments, evaluating them first if they are macros
#define ROCSOLVER_CONCAT2_HELPER(a, b) a##b
#define ROCSOLVER_CONCAT2(a, b) ROCSOLVER_CONCAT2_HELPER(a, b)

#define ROCSOLVER_CONCAT4_HELPER(a, b, c, d) a##b##c##d
#define ROCSOLVER_CONCAT4(a, b, c, d) ROCSOLVER_CONCAT4_HELPER(a, b, c, d)

#if ROCSOLVER_VERSION_MINOR < 10
#define ROCSOLVER_VERSION_MINOR_PADDED ROCSOLVER_CONCAT2(0, ROCSOLVER_VERSION_MINOR)
#else
#define ROCSOLVER_VERSION_MINOR_PADDED ROCSOLVER_VERSION_MINOR
#endif

#if ROCSOLVER_VERSION_PATCH < 10
#define ROCSOLVER_VERSION_PATCH_PADDED ROCSOLVER_CONCAT2(0, ROCSOLVER_VERSION_PATCH)
#else
#define ROCSOLVER_VERSION_PATCH_PADDED ROCSOLVER_VERSION_PATCH
#endif

#ifndef ROCSOLVER_BEGIN_NAMESPACE
#define ROCSOLVER_BEGIN_NAMESPACE                                      \
    namespace rocsolver                                                \
    {                                                                  \
    inline namespace ROCSOLVER_CONCAT4(v,                              \
                                       ROCSOLVER_VERSION_MAJOR,        \
                                       ROCSOLVER_VERSION_MINOR_PADDED, \
                                       ROCSOLVER_VERSION_PATCH_PADDED) \
    {
#define ROCSOLVER_END_NAMESPACE \
    }                           \
    }
#endif

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_BEGIN_NAMESPACE
#endif

#define ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES \
    (ROCBLAS_VERSION_MAJOR >= 4 || (ROCBLAS_VERSION_MAJOR == 3 && ROCBLAS_VERSION_MINOR >= 1))

#pragma STDC CX_LIMITED_RANGE ON

// half vectors
typedef rocblas_half rocblas_half8 __attribute__((ext_vector_type(8)));
typedef rocblas_half rocblas_half2 __attribute__((ext_vector_type(2)));

#ifndef ROCSOLVER_CLIENTS_TEST
extern "C" __device__ rocblas_half2 llvm_fma_v2f16(rocblas_half2,
                                                   rocblas_half2,
                                                   rocblas_half2) __asm("llvm.fma.v2f16");

__device__ inline rocblas_half2
    rocblas_fmadd_half2(rocblas_half2 multiplier, rocblas_half2 multiplicand, rocblas_half2 addend)
{
    return llvm_fma_v2f16(multiplier, multiplicand, addend);
}

// Conjugate a value. For most types, simply return argument; for
// rocblas_float_complex and rocblas_double_complex, return std::conj(z)
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ inline T conj(const T& z)
{
    return z;
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __host__ inline T conj(const T& z)
{
    return std::conj(z);
}

// Load a scalar. If the argument is a pointer, dereference it; otherwise copy
// it. Allows the same kernels to be used for host and device scalars.

// For host scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(T x)
{
    return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(const T* xp)
{
    return *xp;
}

// For rocblas_half2, we broadcast a fp16 across two halves
template <>
__forceinline__ __device__ __host__ rocblas_half2 load_scalar(const rocblas_half2* xp)
{
    auto x = *reinterpret_cast<const rocblas_half*>(xp);
    return {x, x};
}

// Load a batched scalar. This only works on the device. Used for batched
// functions which may pass an array of scalars rather than a single scalar.

// For device side array of scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(T* x, rocblas_int idx, rocblas_int inc)
{
    return x[idx * inc];
}

// Overload for single scalar value
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(T x, rocblas_int idx, rocblas_int inc)
{
    return x;
}

// Load a pointer from a batch. If the argument is a T**, use block to index it
// and add the offset, if the argument is a T*, add block * stride to pointer
// and add offset.

// For device array of device pointers

// For device pointers
template <typename T, typename I>
__forceinline__ __device__ __host__ T*
    load_ptr_batch(T* p, I block, rocblas_stride offset, rocblas_stride stride)
{
    return p + block * stride + offset;
}

// For device array of device pointers
template <typename T, typename I>
__forceinline__ __device__ __host__ T*
    load_ptr_batch(T* const* p, I block, rocblas_stride offset, rocblas_stride stride)
{
    return p[block] + offset;
}

template <typename T, typename I>
__forceinline__ __device__ __host__ T*
    load_ptr_batch(T** p, I block, rocblas_stride offset, rocblas_stride stride)
{
    return p[block] + offset;
}
/*
// Helper for batched functions with temporary memory, currently just trsm and
// trsv. Copys addresses to array of pointers for batched versions.
template <typename T>
ROCSOLVER_KERNEL void setup_batched_array_kernel(T* src, rocblas_stride src_stride, T* dst[])
{
    dst[hipBlockIdx_x] = src + hipBlockIdx_x * src_stride;
}

template <rocblas_int BLOCK, typename T>
void setup_batched_array(hipStream_t stream,
                         T* src,
                         rocblas_stride src_stride,
                         T* dst[],
                         rocblas_int batch_count)
{
    dim3 grid(batch_count);
    dim3 threads(BLOCK);

    hipLaunchKernelGGL(setup_batched_array_kernel<T>, grid, threads, 0, stream, src, src_stride, dst);
}

template <typename T>
ROCSOLVER_KERNEL void setup_device_pointer_array_kernel(T* src,
                                                  rocblas_stride src_stride,
                                                  T* dst[],
                                                  rocblas_int batch_count)
{
    ptrdiff_t tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(tid < batch_count)
        dst[tid] = src + tid * src_stride;
}

template <typename T>
void setup_device_pointer_array(hipStream_t stream,
                                T* src,
                                rocblas_stride src_stride,
                                T* dst[],
                                rocblas_int batch_count)
{
    int NB = 256;
    dim3 grid((batch_count - 1) / NB + 1);
    dim3 threads(NB);
    hipLaunchKernelGGL(setup_device_pointer_array_kernel<T>, grid, threads, 0, stream, src,
                       src_stride, dst, batch_count);
}
*/
#endif // ROCSOLVER_CLIENTS_TEST

inline bool isAligned(const void* pointer, size_t byte_count)
{
    return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

// clang-format off
// // return letter N,T,C in place of rocblas_operation enum
// constexpr char rocblas_transpose_letter(rocblas_operation trans)
// {
//     switch(trans)
//     {
//     case rocblas_operation_none:                return 'N';
//     case rocblas_operation_transpose:           return 'T';
//     case rocblas_operation_conjugate_transpose: return 'C';
//     }
//     return ' ';
// }

// // return letter L, R, B in place of rocblas_side enum
// constexpr char rocblas_side_letter(rocblas_side side)
// {
//     switch(side)
//     {
//     case rocblas_side_left:  return 'L';
//     case rocblas_side_right: return 'R';
//     case rocblas_side_both:  return 'B';
//     }
//     return ' ';
// }

// // return letter U, L, B in place of rocblas_fill enum
// constexpr char rocblas_fill_letter(rocblas_fill fill)
// {
//     switch(fill)
//     {
//     case rocblas_fill_upper: return 'U';
//     case rocblas_fill_lower: return 'L';
//     case rocblas_fill_full:  return 'F';
//     }
//     return ' ';
// }

// // return letter N, U in place of rocblas_diagonal enum
// constexpr char rocblas_diag_letter(rocblas_diagonal diag)
// {
//     switch(diag)
//     {
//     case rocblas_diagonal_non_unit: return 'N';
//     case rocblas_diagonal_unit:     return 'U';
//     }
//     return ' ';
// }

// return precision string for rocblas_datatype
constexpr const char* rocblas_datatype_string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:   return "f16_r";
    case rocblas_datatype_f32_r:   return "f32_r";
    case rocblas_datatype_f64_r:   return "f64_r";
    case rocblas_datatype_f16_c:   return "f16_c";
    case rocblas_datatype_f32_c:   return "f32_c";
    case rocblas_datatype_f64_c:   return "f64_c";
    case rocblas_datatype_i8_r:    return "i8_r";
    case rocblas_datatype_u8_r:    return "u8_r";
    case rocblas_datatype_i32_r:   return "i32_r";
    case rocblas_datatype_u32_r:   return "u32_r";
    case rocblas_datatype_i8_c:    return "i8_c";
    case rocblas_datatype_u8_c:    return "u8_c";
    case rocblas_datatype_i32_c:   return "i32_c";
    case rocblas_datatype_u32_c:   return "u32_c";
    case rocblas_datatype_bf16_r:  return "bf16_r";
    case rocblas_datatype_bf16_c:  return "bf16_c";
    case rocblas_datatype_invalid: return "invalid";
#if ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES
    case rocblas_datatype_f8_r:    return "f8_r";
    case rocblas_datatype_bf8_r:   return "bf8_r";
#endif
    }
    return "invalid";
}

// return sizeof rocblas_datatype
constexpr size_t rocblas_sizeof_datatype(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:   return 2;
    case rocblas_datatype_f32_r:   return 4;
    case rocblas_datatype_f64_r:   return 8;
    case rocblas_datatype_f16_c:   return 4;
    case rocblas_datatype_f32_c:   return 8;
    case rocblas_datatype_f64_c:   return 16;
    case rocblas_datatype_i8_r:    return 1;
    case rocblas_datatype_u8_r:    return 1;
    case rocblas_datatype_i32_r:   return 4;
    case rocblas_datatype_u32_r:   return 4;
    case rocblas_datatype_i8_c:    return 2;
    case rocblas_datatype_u8_c:    return 2;
    case rocblas_datatype_i32_c:   return 8;
    case rocblas_datatype_u32_c:   return 8;
    case rocblas_datatype_bf16_r:  return 2;
    case rocblas_datatype_bf16_c:  return 4;
    case rocblas_datatype_invalid: return 4;
#if ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES
    case rocblas_datatype_f8_r:    return 1;
    case rocblas_datatype_bf8_r:   return 1;
#endif
    }
    return 0;
}

// return rocblas_datatype from type
template <typename> static constexpr rocblas_datatype rocblas_datatype_from_type     = rocblas_datatype_invalid;
template <> static constexpr auto rocblas_datatype_from_type<rocblas_half>           = rocblas_datatype_f16_r;
template <> static constexpr auto rocblas_datatype_from_type<float>                  = rocblas_datatype_f32_r;
template <> static constexpr auto rocblas_datatype_from_type<double>                 = rocblas_datatype_f64_r;
template <> static constexpr auto rocblas_datatype_from_type<rocblas_float_complex>  = rocblas_datatype_f32_c;
template <> static constexpr auto rocblas_datatype_from_type<rocblas_double_complex> = rocblas_datatype_f64_c;
template <> static constexpr auto rocblas_datatype_from_type<int8_t>                 = rocblas_datatype_i8_r;
template <> static constexpr auto rocblas_datatype_from_type<uint8_t>                = rocblas_datatype_u8_r;
template <> static constexpr auto rocblas_datatype_from_type<int32_t>                = rocblas_datatype_i32_r;
template <> static constexpr auto rocblas_datatype_from_type<uint32_t>               = rocblas_datatype_u32_r;
template <> static constexpr auto rocblas_datatype_from_type<rocblas_bfloat16>       = rocblas_datatype_bf16_r;
#if ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES
template <> static constexpr auto rocblas_datatype_from_type<rocblas_f8>             = rocblas_datatype_f8_r;
template <> static constexpr auto rocblas_datatype_from_type<rocblas_bf8>            = rocblas_datatype_bf8_r;
#endif

// return precision string for data type
template <typename> static constexpr char rocblas_precision_string                [] = "invalid";
template <> static constexpr char rocblas_precision_string<rocblas_bfloat16      >[] = "bf16_r";
template <> static constexpr char rocblas_precision_string<rocblas_half          >[] = "f16_r";
template <> static constexpr char rocblas_precision_string<float                 >[] = "f32_r";
template <> static constexpr char rocblas_precision_string<double                >[] = "f64_r";
template <> static constexpr char rocblas_precision_string<int8_t                >[] = "i8_r";
template <> static constexpr char rocblas_precision_string<uint8_t               >[] = "u8_r";
template <> static constexpr char rocblas_precision_string<int32_t               >[] = "i32_r";
template <> static constexpr char rocblas_precision_string<uint32_t              >[] = "u32_r";
template <> static constexpr char rocblas_precision_string<rocblas_float_complex >[] = "f32_c";
template <> static constexpr char rocblas_precision_string<rocblas_double_complex>[] = "f64_c";
#if 0 // Not implemented
template <> static constexpr char rocblas_precision_string<rocblas_half_complex  >[] = "f16_c";
template <> static constexpr char rocblas_precision_string<rocblas_i8_complex    >[] = "i8_c";
template <> static constexpr char rocblas_precision_string<rocblas_u8_complex    >[] = "u8_c";
template <> static constexpr char rocblas_precision_string<rocblas_i32_complex   >[] = "i32_c";
template <> static constexpr char rocblas_precision_string<rocblas_u32_complex   >[] = "u32_c";
#if ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES
template <> static constexpr char rocblas_precision_string<rocblas_f8            >[] = "f8_r";
template <> static constexpr char rocblas_precision_string<rocblas_bf8           >[] = "bf8_r";
#endif
#endif

// clang-format on

/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 * TODO - enumerate library calls to hip runtime, enumerate possible errors from
 *those calls
 ******************************************************************************/
constexpr rocblas_status get_rocblas_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
    // success
    case hipSuccess: return rocblas_status_success;

    // internal hip memory allocation
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources: return rocblas_status_memory_error;

    // user-allocated hip memory
    case hipErrorInvalidDevicePointer: // hip memory
        return rocblas_status_invalid_pointer;

    // user-allocated device, stream, event
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle: return rocblas_status_invalid_handle;

    // library using hip incorrectly
    case hipErrorInvalidValue: return rocblas_status_internal_error;

    // hip runtime failing
    case hipErrorNoDevice: // no hip devices
    case hipErrorUnknown:
    default: return rocblas_status_internal_error;
    }
}

// Absolute value
template <typename T, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__device__ __host__ inline T rocblas_abs(T x)
{
    return x < 0 ? -x : x;
}

// For complex, we have defined a __device__ __host__ compatible std::abs
template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__device__ __host__ inline auto rocblas_abs(T x)
{
    return std::abs(x);
}

// rocblas_bfloat16 is handled specially
__device__ __host__ inline rocblas_bfloat16 rocblas_abs(rocblas_bfloat16 x)
{
    x.data &= 0x7fff;
    return x;
}

// rocblas_half
__device__ __host__ inline rocblas_half rocblas_abs(rocblas_half x)
{
    union
    {
        rocblas_half x;
        uint16_t data;
    } t = {x};
    t.data &= 0x7fff;
    return t.x;
}

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_END_NAMESPACE
#endif

// Get base types from complex types. These should be moved into the rocsolver
// namespace, but they are used by rocBLAS headers.
template <typename T, typename = void>
struct rocblas_real_t_impl
{
    using type = T;
};

template <typename T>
struct rocblas_real_t_impl<T, std::enable_if_t<rocblas_is_complex<T>>>
{
    using type = decltype(std::real(T{}));
};

template <typename T>
struct rocblas_real_t_impl<std::complex<T>>
{
    using type = T;
};

template <typename T>
using real_t = typename rocblas_real_t_impl<T>::type;

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_BEGIN_NAMESPACE
#endif

// Output rocblas_half value
inline std::ostream& operator<<(std::ostream& os, rocblas_half x)
{
    return os << float(x);
}

// Convert the current C++ exception to rocblas_status
// This allows extern "C" functions to return this function in a catch(...)
// block while converting all C++ exceptions to an equivalent rocblas_status
// here
inline rocblas_status exception_to_rocblas_status(std::exception_ptr e = std::current_exception())
try
{
    if(e)
        std::rethrow_exception(e);
    return rocblas_status_success;
}
catch(const rocblas_status& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return rocblas_status_memory_error;
}
catch(...)
{
    return rocblas_status_internal_error;
}

#undef ROCSOLVER_ROCBLAS_HAS_F8_DATATYPES

#ifdef ROCSOLVER_LIBRARY
ROCSOLVER_END_NAMESPACE
#endif
