/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "definitions.hpp"
#include "rocblas.h"
#include <cmath>
#include <complex>
#include <exception>
#include <hip/hip_runtime.h>
#include <new>
#include <type_traits>

constexpr int c_YZ_grid_launch_limit = 1 << 16; // on some gfx

#pragma STDC CX_LIMITED_RANGE ON

// half vectors
typedef rocblas_half rocblas_half8 __attribute__((ext_vector_type(8)));
typedef rocblas_half rocblas_half2 __attribute__((ext_vector_type(2)));
typedef rocblas_half rocblas_half4 __attribute__((ext_vector_type(4)));

#ifndef GOOGLE_TEST
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

template <bool CONJ, typename T>
__device__ __host__ inline T conj_if_true(const T& z)
{
    return CONJ ? conj(z) : z;
}

#endif

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

// Load a batched scalar. This only works on the device. Used for batched functions which may
// pass an array of scalars rather than a single scalar.

// For device side array of scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(const T* x, uint32_t idx, rocblas_stride inc)
{
    return x[idx * inc];
}

// Overload for single scalar value
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(T x, uint32_t idx, rocblas_stride inc)
{
    return x;
}

// Load a pointer from a batch. If the argument is a T**, use block to index it and
// add the offset, if the argument is a T*, add block * stride to pointer and add offset.
//
// ----- for batched use offset else use stride, do not use both stride and offset -----

// For device pointers (used by non-batched and _strided_batched functions)
// clang-format off
template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T* p, uint32_t block, rocblas_stride stride)
{
    return p + block * stride;
}

// For device array of device pointers (used by _batched functions)
template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T* const* p, uint32_t block, rocblas_stride offset)
{
    return p[block] + offset;
}

template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T** p, uint32_t block, rocblas_stride offset)
{
    return p[block] + offset;
}

// ----- use both stride and offset -----
// For device pointers (used by non-batched and _strided_batched functions)
template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T* p, uint32_t block, rocblas_stride offset, rocblas_stride stride)
{
    return p + block * stride + offset;
}

// For device array of device pointers (used by _batched functions)
template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T* const* p, uint32_t block, rocblas_stride offset, rocblas_stride stride)
{
    return p[block] + offset;
}

template <typename T>
__forceinline__ __device__ __host__ T*
                                    load_ptr_batch(T** p, uint32_t block, rocblas_stride offset, rocblas_stride stride)
{
    return p[block] + offset;
}

// guarded by condition
template <typename C, typename T>
__forceinline__ __device__ __host__ T*
                                    cond_load_ptr_batch(C cond, T* p, uint32_t block, rocblas_stride offset, rocblas_stride stride)
{
    // safe to offset pointer regardless of condition as not dereferenced
    return load_ptr_batch( p, block, offset, stride);
}

// For device array of device pointers array is dereferenced, e.g. alpha, if !alpha don't dereference pointer array as we allow it to be null
template <typename C, typename T>
__forceinline__ __device__ __host__ T*
                                    cond_load_ptr_batch(C cond, T* const* p, uint32_t block, rocblas_stride offset, rocblas_stride stride)
{
    return cond ? load_ptr_batch( p, block, offset, stride) : nullptr;
}

template <typename C, typename T>
__forceinline__ __device__ __host__ T*
                                    cond_load_ptr_batch(C cond, T** p, uint32_t block, rocblas_stride offset, rocblas_stride stride)
{
    return cond ? load_ptr_batch( p, block, offset, stride) : nullptr;
}
// clang-format on

/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 ******************************************************************************/
ROCBLAS_INTERNAL_EXPORT rocblas_status
    rocblas_internal_convert_hip_to_rocblas_status(hipError_t status);

ROCBLAS_INTERNAL_EXPORT rocblas_status
    rocblas_internal_convert_hip_to_rocblas_status_and_log(hipError_t status);

#ifndef GOOGLE_TEST

// Helper for batched functions with temporary memory, currently just trsm and trsv.
// Copys addresses to array of pointers for batched versions.
template <rocblas_int NB, typename T>
ROCBLAS_KERNEL(NB)
setup_batched_array_kernel(T* src, rocblas_stride src_stride, T* dst[])
{
    dst[blockIdx.x] = src + blockIdx.x * src_stride;
}

template <rocblas_int BLOCK, typename T>
rocblas_status setup_batched_array(
    hipStream_t stream, T* src, rocblas_stride src_stride, T* dst[], rocblas_int batch_count)
{
    dim3 grid(batch_count);
    dim3 threads(BLOCK);

    ROCBLAS_LAUNCH_KERNEL(
        (setup_batched_array_kernel<BLOCK, T>), grid, threads, 0, stream, src, src_stride, dst);

    return rocblas_status_success;
}

template <rocblas_int NB, typename T>
ROCBLAS_KERNEL(NB)
setup_device_pointer_array_kernel(T*             src,
                                  rocblas_stride src_stride,
                                  T*             dst[],
                                  rocblas_int    batch_count)
{
    ptrdiff_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < batch_count)
        dst[tid] = src + tid * src_stride;
}

template <typename T>
rocblas_status setup_device_pointer_array(
    hipStream_t stream, T* src, rocblas_stride src_stride, T* dst[], rocblas_int batch_count)
{
    int  NB = 256;
    dim3 grid((batch_count - 1) / NB + 1);
    dim3 threads(NB);
    ROCBLAS_LAUNCH_KERNEL((setup_device_pointer_array_kernel<NB, T>),
                          grid,
                          threads,
                          0,
                          stream,
                          src,
                          src_stride,
                          dst,
                          batch_count);

    return rocblas_status_success;
}

#endif // GOOGLE_TEST

inline bool isAligned(const void* pointer, size_t byte_count)
{
    return reinterpret_cast<uintptr_t>(pointer) % byte_count == 0;
}

// clang-format off
// return letter N,T,C in place of rocblas_operation enum
constexpr char rocblas_transpose_letter(rocblas_operation trans)
{
    switch(trans)
    {
    case rocblas_operation_none:                return 'N';
    case rocblas_operation_transpose:           return 'T';
    case rocblas_operation_conjugate_transpose: return 'C';
    }
    return ' ';
}

// return letter L, R, B in place of rocblas_side enum
constexpr char rocblas_side_letter(rocblas_side side)
{
    switch(side)
    {
    case rocblas_side_left:  return 'L';
    case rocblas_side_right: return 'R';
    case rocblas_side_both:  return 'B';
    }
    return ' ';
}

// return letter U, L, B in place of rocblas_fill enum
constexpr char rocblas_fill_letter(rocblas_fill fill)
{
    switch(fill)
    {
    case rocblas_fill_upper: return 'U';
    case rocblas_fill_lower: return 'L';
    case rocblas_fill_full:  return 'F';
    }
    return ' ';
}

// return letter N, U in place of rocblas_diagonal enum
constexpr char rocblas_diag_letter(rocblas_diagonal diag)
{
    switch(diag)
    {
    case rocblas_diagonal_non_unit: return 'N';
    case rocblas_diagonal_unit:     return 'U';
    }
    return ' ';
}

// return precision string for rocblas_datatype
constexpr const char* rocblas_datatype_string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:  return "f16_r";
    case rocblas_datatype_f32_r:  return "f32_r";
    case rocblas_datatype_f64_r:  return "f64_r";
    case rocblas_datatype_f16_c:  return "f16_c";
    case rocblas_datatype_f32_c:  return "f32_c";
    case rocblas_datatype_f64_c:  return "f64_c";
    case rocblas_datatype_i8_r:   return "i8_r";
    case rocblas_datatype_u8_r:   return "u8_r";
    case rocblas_datatype_i32_r:  return "i_r32";
    case rocblas_datatype_u32_r:  return "u32_r";
    case rocblas_datatype_i8_c:   return "i8_c";
    case rocblas_datatype_u8_c:   return "u8_c";
    case rocblas_datatype_i32_c:  return "i32_c";
    case rocblas_datatype_u32_c:  return "u32_c";
    case rocblas_datatype_bf16_r: return "bf16_r";
    case rocblas_datatype_bf16_c: return "bf16_c";
    case rocblas_datatype_invalid: return "invalid";
    }
    return "invalid";
}

// return sizeof rocblas_datatype
constexpr size_t rocblas_sizeof_datatype(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:  return 2;
    case rocblas_datatype_f32_r:  return 4;
    case rocblas_datatype_f64_r:  return 8;
    case rocblas_datatype_f16_c:  return 4;
    case rocblas_datatype_f32_c:  return 8;
    case rocblas_datatype_f64_c:  return 16;
    case rocblas_datatype_i8_r:   return 1;
    case rocblas_datatype_u8_r:   return 1;
    case rocblas_datatype_i32_r:  return 4;
    case rocblas_datatype_u32_r:  return 4;
    case rocblas_datatype_i8_c:   return 2;
    case rocblas_datatype_u8_c:   return 2;
    case rocblas_datatype_i32_c:  return 8;
    case rocblas_datatype_u32_c:  return 8;
    case rocblas_datatype_bf16_r: return 2;
    case rocblas_datatype_bf16_c: return 4;
    case rocblas_datatype_invalid: return 4;
    }
    return 0;
}

// Convert atomics mode to string
constexpr const char* rocblas_atomics_mode_to_string(rocblas_atomics_mode mode)
{
    return mode != rocblas_atomics_not_allowed ? "atomics_allowed" : "atomics_not_allowed";
}

// Convert gemm flags to string
constexpr const char* rocblas_gemm_flags_to_string(rocblas_gemm_flags type)
{
    switch(type)
    {
    case rocblas_gemm_flags_none:                 return "none";
    case rocblas_gemm_flags_use_cu_efficiency:    return "use_cu_efficiency";
    case rocblas_gemm_flags_fp16_alt_impl:        return "fp16_alt_impl";
    case rocblas_gemm_flags_fp16_alt_impl_rnz:  return "fp16_alt_impl_round";
    case rocblas_gemm_flags_check_solution_index: return "check_solution_index";
    case rocblas_gemm_flags_stochastic_rounding:  return "stochastic_rounding";
    }
    return "invalid";
}

// return rocblas_datatype from type
template <typename> inline constexpr rocblas_datatype rocblas_datatype_from_type                   = rocblas_datatype_invalid;
template <> inline constexpr auto rocblas_datatype_from_type<rocblas_half>           = rocblas_datatype_f16_r;
template <> inline constexpr auto rocblas_datatype_from_type<float>                  = rocblas_datatype_f32_r;
template <> inline constexpr auto rocblas_datatype_from_type<double>                 = rocblas_datatype_f64_r;
template <> inline constexpr auto rocblas_datatype_from_type<rocblas_float_complex>  = rocblas_datatype_f32_c;
template <> inline constexpr auto rocblas_datatype_from_type<rocblas_double_complex> = rocblas_datatype_f64_c;
template <> inline constexpr auto rocblas_datatype_from_type<int8_t>                 = rocblas_datatype_i8_r;
template <> inline constexpr auto rocblas_datatype_from_type<uint8_t>                = rocblas_datatype_u8_r;
template <> inline constexpr auto rocblas_datatype_from_type<int32_t>                = rocblas_datatype_i32_r;
template <> inline constexpr auto rocblas_datatype_from_type<uint32_t>               = rocblas_datatype_u32_r;
template <> inline constexpr auto rocblas_datatype_from_type<rocblas_bfloat16>       = rocblas_datatype_bf16_r;

// return precision string for data type
template <typename> inline constexpr char rocblas_precision_string                              [] = "invalid";
template <> inline constexpr char rocblas_precision_string<rocblas_bfloat16      >[] = "bf16_r";
template <> inline constexpr char rocblas_precision_string<rocblas_half          >[] = "f16_r";
template <> inline constexpr char rocblas_precision_string<float                 >[] = "f32_r";
template <> inline constexpr char rocblas_precision_string<double                >[] = "f64_r";
template <> inline constexpr char rocblas_precision_string<int8_t                >[] = "i8_r";
template <> inline constexpr char rocblas_precision_string<uint8_t               >[] = "u8_r";
template <> inline constexpr char rocblas_precision_string<int32_t               >[] = "i32_r";
template <> inline constexpr char rocblas_precision_string<uint32_t              >[] = "u32_r";
template <> inline constexpr char rocblas_precision_string<rocblas_float_complex >[] = "f32_c";
template <> inline constexpr char rocblas_precision_string<rocblas_double_complex>[] = "f64_c";
#if 0 // Not implemented
template <> inline constexpr char rocblas_precision_string<rocblas_half_complex  >[] = "f16_c";
template <> inline constexpr char rocblas_precision_string<rocblas_i8_complex    >[] = "i8_c";
template <> inline constexpr char rocblas_precision_string<rocblas_u8_complex    >[] = "u8_c";
template <> inline constexpr char rocblas_precision_string<rocblas_i32_complex   >[] = "i32_c";
template <> inline constexpr char rocblas_precision_string<rocblas_u32_complex   >[] = "u32_c";
#endif

// clang-format on

/*************************************************************************************************************************
 * \brief The main structure for Numerical checking to detect numerical abnormalities such as NaN/zero/Inf/denormal values
 ************************************************************************************************************************/
typedef struct rocblas_check_numerics_s
{
    // Set to true if there is a NaN in the vector/matrix
    bool has_NaN = false;

    // Set to true if there is a zero in the vector/matrix
    bool has_zero = false;

    // Set to true if there is an Infinity in the vector/matrix
    bool has_Inf = false;

    // Set to true if there is an denormal/subnormal value in the vector/matrix
    bool has_denorm = false;

} rocblas_check_numerics_t;

/*************************************************************************************************************************
//! @brief enum to check the type of matrix
 ************************************************************************************************************************/
typedef enum rocblas_check_matrix_type_
{
    // General matrix
    rocblas_client_general_matrix,

    // Hermitian matrix
    rocblas_client_hermitian_matrix,

    // Symmetric matrix
    rocblas_client_symmetric_matrix,

    // Triangular matrix
    rocblas_client_triangular_matrix,

    // Diagonally dominant triangular matrix
    rocblas_client_diagonally_dominant_triangular_matrix,

} rocblas_check_matrix_type;

/*******************************************************************************
* \brief  returns true if arg is NaN
********************************************************************************/
template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool rocblas_isnan(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{} && !rocblas_is_complex<T>, int> = 0>
__host__ __device__ inline bool rocblas_isnan(T arg)
{
    return std::isnan(arg);
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__host__ __device__ inline bool rocblas_isnan(const T& arg)
{
    return rocblas_isnan(std::real(arg)) || rocblas_isnan(std::imag(arg));
}

__host__ __device__ inline bool rocblas_isnan(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) != 0;
}

/*******************************************************************************
* \brief  returns true if arg is Infinity
********************************************************************************/

template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool rocblas_isinf(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{} && !rocblas_is_complex<T>, int> = 0>
__host__ __device__ inline bool rocblas_isinf(T arg)
{
    return std::isinf(arg);
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__host__ __device__ inline bool rocblas_isinf(const T& arg)
{
    return rocblas_isinf(std::real(arg)) || rocblas_isinf(std::imag(arg));
}

__host__ __device__ inline bool rocblas_isinf(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) == 0;
}

/*******************************************************************************
* \brief  returns max value for type
********************************************************************************/

template <typename T>
__host__ __device__ inline void rocblas_set_max_value(T& val)
{
    val = std::numeric_limits<T>::max();
}

__host__ __device__ inline void rocblas_set_max_value(rocblas_half& val)
{
    *((short*)(&val)) = 0x7c00;
}

/*******************************************************************************
* \brief  returns true if arg is zero
********************************************************************************/

template <typename T>
__host__ __device__ inline bool rocblas_iszero(T arg)
{
    return arg == 0;
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
        uint16_t     data;
    } t = {x};
    t.data &= 0x7fff;
    return t.x;
}

/*******************************************************************************
* \brief  returns true if arg is denormal/subnormal
********************************************************************************/

template <typename T, std::enable_if_t<std::is_integral<T>{}, int> = 0>
__host__ __device__ inline bool rocblas_isdenorm(T)
{
    return false;
}

template <typename T, std::enable_if_t<!std::is_integral<T>{} && !rocblas_is_complex<T>, int> = 0>
__host__ __device__ inline bool rocblas_isdenorm(T arg)
{
    return ((rocblas_abs(arg) >= std::numeric_limits<T>::denorm_min())
            && (rocblas_abs(arg) < std::numeric_limits<T>::min()));
}

template <typename T, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__host__ __device__ inline bool rocblas_isdenorm(const T& arg)
{
    return rocblas_isdenorm(std::real(arg)) || rocblas_isdenorm(std::imag(arg));
}

__host__ __device__ inline bool rocblas_isdenorm(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {rocblas_abs(arg)};
    return (
        (x.data >= 0x0001)
        && (x.data
            < 0x0400)); //0x0001 is the smallest positive subnormal number and 0x0400 is the smallest positive normal number represented by rocblas_half
}

__host__ __device__ inline bool rocblas_isdenorm(rocblas_bfloat16 arg)
{
    union
    {
        rocblas_bfloat16 fp;
        uint16_t         data;
    } x = {rocblas_abs(arg)};
    return (
        (x.data >= 0x0001)
        && (x.data
            < 0x0080)); //0x0001 is the smallest positive subnormal number and 0x0080 is the smallest positive normal number represented by rocblas_bfloat16
}

// Is power of two
__device__ __host__ constexpr bool rocblas_is_po2(rocblas_int x)
{
    return (x && !(x & (x - 1)));
}

// Return previous power of two
__device__ __host__ constexpr rocblas_int rocblas_previous_po2(rocblas_int x)
{
    return x ? decltype(x){1} << (8 * sizeof(x) - 1 - __builtin_clz(x)) : 0;
}

// Get base types from complex types.
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

// Batched datatype
template <typename T, bool BATCHED, typename = void>
struct rocblas_batched_t_impl
{
    using type = T*;
};

template <typename T>
struct rocblas_batched_t_impl<T, true>
{
    using type = T* const*;
};

template <typename T, bool BATCHED>
using rocblas_batched_t = typename rocblas_batched_t_impl<T, BATCHED>::type;

// Const Batched datatype
template <typename T, bool BATCHED, typename = void>
struct rocblas_const_batched_t_impl
{
    using type = const T*;
};

template <typename T>
struct rocblas_const_batched_t_impl<T, true>
{
    using type = const T* const*;
};

template <typename T, bool BATCHED>
using rocblas_const_batched_t = typename rocblas_const_batched_t_impl<T, BATCHED>::type;

// Batched datatype
template <typename T, bool BATCHED, typename = void>
struct rocblas_type_from_ptr_t_impl
{
    // T should be either a const T* or a T* here
    using type = typename std::remove_const<std::remove_pointer_t<T>>::type;
};

template <typename T>
struct rocblas_type_from_ptr_t_impl<T, true>
{
    // T should be either a const T* const* or a T* const* here
    using T_tmp = typename std::remove_const<std::remove_pointer_t<T>>::type;
    using type  = typename rocblas_type_from_ptr_t_impl<T_tmp, false>::type;
};

template <typename T, bool BATCHED>
using rocblas_type_from_ptr_t = typename rocblas_type_from_ptr_t_impl<T, BATCHED>::type;

// Get array2 types from base type
template <typename T, typename = void>
struct rocblas_array2_t_impl
{
    using type = T;
};

template <>
struct rocblas_array2_t_impl<float>
{
    using type = float2;
};

template <>
struct rocblas_array2_t_impl<double>
{
    using type = double2;
};

template <>
struct rocblas_array2_t_impl<rocblas_half>
{
    using type = rocblas_half2;
};

template <typename T>
using array2_t = typename rocblas_array2_t_impl<T>::type;

// rocblas_is_array2<T> returns true iff T is hip vector type of size 2
template <typename T>
inline constexpr bool rocblas_is_array2 = false;

template <>
inline constexpr bool rocblas_is_array2<rocblas_half2> = true;

template <>
inline constexpr bool rocblas_is_array2<float2> = true;

template <>
inline constexpr bool rocblas_is_array2<double2> = true;

// Output rocblas_half value
inline std::ostream& operator<<(std::ostream& os, rocblas_half x)
{
    return os << float(x);
}

// Convert the current C++ exception to rocblas_status
// This allows extern "C" functions to return this function in a catch(...) block
// while converting all C++ exceptions to an equivalent rocblas_status here
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

// Return the value category for a value, as a double precision value, such
// such as whether it's 0, 1, -1 or some other value. Tensile uses a double
// precision value to express the category of beta. This function is to
// convert complex or other types to a double representing the category.
template <typename T>
constexpr double rocblas_internal_value_category(const T& beta)
{
    return beta == T(0) ? 0.0 : beta == T(1) ? 1.0 : beta == T(-1) ? -1.0 : 2.0;
}

// Internal use
int rocblas_internal_get_arch(rocblas_handle handle);

// Internal use, whether Tensile supports ldc != ldd
// We assume true if the value is greater than or equal to 906
bool rocblas_internal_tensile_supports_ldc_ne_ldd(rocblas_handle handle);

// Internal use, whether Device supports xDL math op
// We assume true if the value is between 942 to 1000
ROCBLAS_INTERNAL_EXPORT bool rocblas_internal_tensile_supports_xdl_math_op(rocblas_math_mode mode);

// for internal use during testing, fetch arch name
ROCBLAS_INTERNAL_EXPORT std::string rocblas_internal_get_arch_name();

// for internal use, fetch xnack mode
std::string rocblas_internal_get_xnack_mode();

// for internal use during testing, whether to skip actual kernel launch
ROCBLAS_INTERNAL_EXPORT bool rocblas_internal_tensile_debug_skip_launch();

template <typename T>
struct rocblas_internal_val_ptr
{
    union
    {
        T        value;
        const T* ptr;
    };

    inline rocblas_internal_val_ptr(bool host_mode, const T* val_ptr)
    {
        if(host_mode)
            value = *val_ptr;
        else
            ptr = val_ptr;
    }
};
