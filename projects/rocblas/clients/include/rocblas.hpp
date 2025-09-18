/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************ */

#pragma once

/* library headers */

// we still test deprecated API so don't want warnings
#ifndef ROCBLAS_NO_DEPRECATED_WARNINGS
#define ROCBLAS_NO_DEPRECATED_WARNINGS
#endif

#include "rocblas.h"
#include <cstdint>

#ifdef CLIENTS_NO_FORTRAN
#include "rocblas_no_fortran.hpp"
#else
#include "rocblas_fortran.hpp"
#endif

#include "../../library/src/include/utility.hpp"

#if not defined(__clang_major__)
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

#define MAP2CF(...) GET_MACRO(__VA_ARGS__, MAP2CF5, MAP2CF4, MAP2CF3, dum2, dum1)(__VA_ARGS__)
// dual API C and FORTRAN
#define MAP2CF_D64(...) \
    GET_MACRO(__VA_ARGS__, MAP2DCF5, MAP2DCF4, MAP2DCF3, dum2, dum1)(__VA_ARGS__)

#ifndef CLIENTS_NO_FORTRAN
#define MAP2CF3(FN, A, PFN)         \
    template <>                     \
    inline auto FN<A, false> = PFN; \
    template <>                     \
    inline auto FN<A, true> = PFN##_fortran
#define MAP2CF4(FN, A, B, PFN)         \
    template <>                        \
    inline auto FN<A, B, false> = PFN; \
    template <>                        \
    inline auto FN<A, B, true> = PFN##_fortran
#define MAP2CF5(FN, A, B, C, PFN)         \
    template <>                           \
    inline auto FN<A, B, C, false> = PFN; \
    template <>                           \
    inline auto FN<A, B, C, true> = PFN##_fortran
// dual API C and FORTRAN
#define MAP2DCF3(FN, A, PFN)                  \
    template <>                               \
    inline auto FN<A, false> = PFN;           \
    template <>                               \
    inline auto FN<A, true> = PFN##_fortran;  \
    template <>                               \
    inline auto FN##_64<A, false> = PFN##_64; \
    template <>                               \
    inline auto FN##_64<A, true> = PFN##_64_fortran
#define MAP2DCF4(FN, A, B, PFN)                  \
    template <>                                  \
    inline auto FN<A, B, false> = PFN;           \
    template <>                                  \
    inline auto FN<A, B, true> = PFN##_fortran;  \
    template <>                                  \
    inline auto FN##_64<A, B, false> = PFN##_64; \
    template <>                                  \
    inline auto FN##_64<A, B, true> = PFN##_64_fortran
#define MAP2DCF5(FN, A, B, C, PFN)                  \
    template <>                                     \
    inline auto FN<A, B, C, false> = PFN;           \
    template <>                                     \
    inline auto FN<A, B, C, true> = PFN##_fortran;  \
    template <>                                     \
    inline auto FN##_64<A, B, C, false> = PFN##_64; \
    template <>                                     \
    inline auto FN##_64<A, B, C, true> = PFN##_64_fortran
#else
// mapping fortran and C to C API
#define MAP2CF3(FN, A, PFN)         \
    template <>                     \
    inline auto FN<A, false> = PFN; \
    template <>                     \
    inline auto FN<A, true> = PFN
#define MAP2CF4(FN, A, B, PFN)         \
    template <>                        \
    inline auto FN<A, B, false> = PFN; \
    template <>                        \
    inline auto FN<A, B, true> = PFN
#define MAP2CF5(FN, A, B, C, PFN)         \
    template <>                           \
    inline auto FN<A, B, C, false> = PFN; \
    template <>                           \
    inline auto FN<A, B, C, true> = PFN
// dual API C and FORTRAN
#define MAP2DCF3(FN, A, PFN)                  \
    template <>                               \
    inline auto FN<A, false> = PFN;           \
    template <>                               \
    inline auto FN<A, true> = PFN;            \
    template <>                               \
    inline auto FN##_64<A, false> = PFN##_64; \
    template <>                               \
    inline auto FN##_64<A, true> = PFN##_64
#define MAP2DCF4(FN, A, B, PFN)                  \
    template <>                                  \
    inline auto FN<A, B, false> = PFN;           \
    template <>                                  \
    inline auto FN<A, B, true> = PFN;            \
    template <>                                  \
    inline auto FN##_64<A, B, false> = PFN##_64; \
    template <>                                  \
    inline auto FN##_64<A, B, true> = PFN##_64
#define MAP2DCF5(FN, A, B, C, PFN)                  \
    template <>                                     \
    inline auto FN<A, B, C, false> = PFN;           \
    template <>                                     \
    inline auto FN<A, B, C, true> = PFN;            \
    template <>                                     \
    inline auto FN##_64<A, B, C, false> = PFN##_64; \
    template <>                                     \
    inline auto FN##_64<A, B, C, true> = PFN##_64
#endif

/*!\file
 *  This file exposes C++ templated BLAS interface with only the precision templated.
 */

/*
 * ===========================================================================
 *   README: Please follow the naming convention
 *   Big case for matrix, e.g. matrix A, B, C   GEMM (C = A*B)
 *   Lower case for vector, e.g. vector x, y    GEMV (y = A*x)
 * ===========================================================================
 */

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

// scal
template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_scal)(
    rocblas_handle handle, rocblas_int n, const U* alpha, T* x, rocblas_int incx);

template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_scal_64)(
    rocblas_handle handle, int64_t n, const U* alpha, T* x, int64_t incx);

MAP2CF_D64(rocblas_scal, float, float, rocblas_sscal);
MAP2CF_D64(rocblas_scal, double, double, rocblas_dscal);
MAP2CF_D64(rocblas_scal, rocblas_float_complex, rocblas_float_complex, rocblas_cscal);
MAP2CF_D64(rocblas_scal, rocblas_double_complex, rocblas_double_complex, rocblas_zscal);
MAP2CF_D64(rocblas_scal, rocblas_float_complex, float, rocblas_csscal);
MAP2CF_D64(rocblas_scal, rocblas_double_complex, double, rocblas_zdscal);

/*
MAP2CF_D64 maps the dual (D) LP64 and ILP64 _64 fuction forms of both C and FORTRAN (CF) mappings
e.g. MAP2CF_D64(rocblas_scal, float, float, rocblas_sscal);
instantiates the above two template function pointer prototypes twice each as:
template <>
inline auto rocblas_scal<float, float, false> = rocblas_sscal;
template <>
inline auto rocblas_scal<float, float, true> = rocblas_sscal_fortran;
template <>
inline auto rocblas_scal_64<float, float, false> = rocblas_sscal_64;
template <>
inline auto rocblas_scal_64<float, float, true> = rocblas_sscal_64_fortran;
*/

// scal_batched
template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_scal_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const U*       alpha,
                                              T* const       x[],
                                              rocblas_int    incx,
                                              rocblas_int    batch_count);

template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_scal_batched_64)(rocblas_handle handle,
                                                 int64_t        n,
                                                 const U*       alpha,
                                                 T* const       x[],
                                                 int64_t        incx,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_scal_batched, float, float, rocblas_sscal_batched);
MAP2CF_D64(rocblas_scal_batched, double, double, rocblas_dscal_batched);
MAP2CF_D64(rocblas_scal_batched,
           rocblas_float_complex,
           rocblas_float_complex,
           rocblas_cscal_batched);
MAP2CF_D64(rocblas_scal_batched,
           rocblas_double_complex,
           rocblas_double_complex,
           rocblas_zscal_batched);
MAP2CF_D64(rocblas_scal_batched, rocblas_float_complex, float, rocblas_csscal_batched);
MAP2CF_D64(rocblas_scal_batched, rocblas_double_complex, double, rocblas_zdscal_batched);

// scal_strided_batched
template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_scal_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const U*       alpha,
                                                      T*             x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      rocblas_int    batch_count);

template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_scal_strided_batched_64)(rocblas_handle handle,
                                                         int64_t        n,
                                                         const U*       alpha,
                                                         T*             x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_scal_strided_batched, float, float, rocblas_sscal_strided_batched);
MAP2CF_D64(rocblas_scal_strided_batched, double, double, rocblas_dscal_strided_batched);
MAP2CF_D64(rocblas_scal_strided_batched,
           rocblas_float_complex,
           rocblas_float_complex,
           rocblas_cscal_strided_batched);
MAP2CF_D64(rocblas_scal_strided_batched,
           rocblas_double_complex,
           rocblas_double_complex,
           rocblas_zscal_strided_batched);
MAP2CF_D64(rocblas_scal_strided_batched,
           rocblas_float_complex,
           float,
           rocblas_csscal_strided_batched);
MAP2CF_D64(rocblas_scal_strided_batched,
           rocblas_double_complex,
           double,
           rocblas_zdscal_strided_batched);

// copy
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_copy)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, T* y, rocblas_int incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_copy_64)(
    rocblas_handle handle, int64_t n, const T* x, int64_t incx, T* y, int64_t incy);

MAP2CF_D64(rocblas_copy, float, rocblas_scopy);
MAP2CF_D64(rocblas_copy, double, rocblas_dcopy);
MAP2CF_D64(rocblas_copy, rocblas_float_complex, rocblas_ccopy);
MAP2CF_D64(rocblas_copy, rocblas_double_complex, rocblas_zcopy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_copy_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_copy_batched_64)(rocblas_handle handle,
                                                 int64_t        n,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_copy_batched, float, rocblas_scopy_batched);
MAP2CF_D64(rocblas_copy_batched, double, rocblas_dcopy_batched);
MAP2CF_D64(rocblas_copy_batched, rocblas_float_complex, rocblas_ccopy_batched);
MAP2CF_D64(rocblas_copy_batched, rocblas_double_complex, rocblas_zcopy_batched);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_copy_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_copy_strided_batched_64)(rocblas_handle handle,
                                                         int64_t        n,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stridex,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stridey,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_copy_strided_batched, float, rocblas_scopy_strided_batched);
MAP2CF_D64(rocblas_copy_strided_batched, double, rocblas_dcopy_strided_batched);
MAP2CF_D64(rocblas_copy_strided_batched, rocblas_float_complex, rocblas_ccopy_strided_batched);
MAP2CF_D64(rocblas_copy_strided_batched, rocblas_double_complex, rocblas_zcopy_strided_batched);

// swap
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_swap)(
    rocblas_handle handle, rocblas_int n, T* x, rocblas_int incx, T* y, rocblas_int incy);

// swap_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_swap_64)(
    rocblas_handle handle, int64_t n, T* x, int64_t incx, T* y, int64_t incy);

MAP2CF_D64(rocblas_swap, float, rocblas_sswap);
MAP2CF_D64(rocblas_swap, double, rocblas_dswap);
MAP2CF_D64(rocblas_swap, rocblas_float_complex, rocblas_cswap);
MAP2CF_D64(rocblas_swap, rocblas_double_complex, rocblas_zswap);

// swap_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_swap_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              T* const       x[],
                                              rocblas_int    incx,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

// swap_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_swap_batched_64)(rocblas_handle handle,
                                                 int64_t        n,
                                                 T* const       x[],
                                                 int64_t        incx,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_swap_batched, float, rocblas_sswap_batched);
MAP2CF_D64(rocblas_swap_batched, double, rocblas_dswap_batched);
MAP2CF_D64(rocblas_swap_batched, rocblas_float_complex, rocblas_cswap_batched);
MAP2CF_D64(rocblas_swap_batched, rocblas_double_complex, rocblas_zswap_batched);

// swap_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_swap_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      T*             x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      rocblas_int    batch_count);

// swap_strided_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_swap_strided_batched_64)(rocblas_handle handle,
                                                         int64_t        n,
                                                         T*             x,
                                                         int64_t        incx,
                                                         rocblas_stride stridex,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stridey,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_swap_strided_batched, float, rocblas_sswap_strided_batched);
MAP2CF_D64(rocblas_swap_strided_batched, double, rocblas_dswap_strided_batched);
MAP2CF_D64(rocblas_swap_strided_batched, rocblas_float_complex, rocblas_cswap_strided_batched);
MAP2CF_D64(rocblas_swap_strided_batched, rocblas_double_complex, rocblas_zswap_strided_batched);

// dot
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dot)(rocblas_handle handle,
                                     rocblas_int    n,
                                     const T*       x,
                                     rocblas_int    incx,
                                     const T*       y,
                                     rocblas_int    incy,
                                     T*             result);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dot_64)(rocblas_handle handle,
                                        int64_t        n,
                                        const T*       x,
                                        int64_t        incx,
                                        const T*       y,
                                        int64_t        incy,
                                        T*             result);

MAP2CF_D64(rocblas_dot, float, rocblas_sdot);
MAP2CF_D64(rocblas_dot, double, rocblas_ddot);
MAP2CF_D64(rocblas_dot, rocblas_half, rocblas_hdot);
MAP2CF_D64(rocblas_dot, rocblas_bfloat16, rocblas_bfdot);
MAP2CF_D64(rocblas_dot, rocblas_float_complex, rocblas_cdotu);
MAP2CF_D64(rocblas_dot, rocblas_double_complex, rocblas_zdotu);

// dotc
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dotc)(rocblas_handle handle,
                                      rocblas_int    n,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             result);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dotc_64)(rocblas_handle handle,
                                         int64_t        n,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       y,
                                         int64_t        incy,
                                         T*             result);

MAP2CF_D64(rocblas_dotc, rocblas_float_complex, rocblas_cdotc);
MAP2CF_D64(rocblas_dotc, rocblas_double_complex, rocblas_zdotc);

// dot_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dot_batched)(rocblas_handle handle,
                                             rocblas_int    n,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             const T* const y[],
                                             rocblas_int    incy,
                                             rocblas_int    batch_count,
                                             T*             result);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dot_batched_64)(rocblas_handle handle,
                                                int64_t        n,
                                                const T* const x[],
                                                int64_t        incx,
                                                const T* const y[],
                                                int64_t        incy,
                                                int64_t        batch_count,
                                                T*             result);

MAP2CF_D64(rocblas_dot_batched, float, rocblas_sdot_batched);
MAP2CF_D64(rocblas_dot_batched, double, rocblas_ddot_batched);
MAP2CF_D64(rocblas_dot_batched, rocblas_half, rocblas_hdot_batched);
MAP2CF_D64(rocblas_dot_batched, rocblas_bfloat16, rocblas_bfdot_batched);
MAP2CF_D64(rocblas_dot_batched, rocblas_float_complex, rocblas_cdotu_batched);
MAP2CF_D64(rocblas_dot_batched, rocblas_double_complex, rocblas_zdotu_batched);

// dotc_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dotc_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count,
                                              T*             result);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dotc_batched_64)(rocblas_handle handle,
                                                 int64_t        n,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T* const y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count,
                                                 T*             result);

MAP2CF_D64(rocblas_dotc_batched, rocblas_float_complex, rocblas_cdotc_batched);
MAP2CF_D64(rocblas_dotc_batched, rocblas_double_complex, rocblas_zdotc_batched);

// dot_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dot_strided_batched)(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     const T*       y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stridey,
                                                     rocblas_int    batch_count,
                                                     T*             result);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dot_strided_batched_64)(rocblas_handle handle,
                                                        int64_t        n,
                                                        const T*       x,
                                                        int64_t        incx,
                                                        rocblas_stride stridex,
                                                        const T*       y,
                                                        int64_t        incy,
                                                        rocblas_stride stridey,
                                                        int64_t        batch_count,
                                                        T*             result);

MAP2CF_D64(rocblas_dot_strided_batched, float, rocblas_sdot_strided_batched);
MAP2CF_D64(rocblas_dot_strided_batched, double, rocblas_ddot_strided_batched);
MAP2CF_D64(rocblas_dot_strided_batched, rocblas_half, rocblas_hdot_strided_batched);
MAP2CF_D64(rocblas_dot_strided_batched, rocblas_bfloat16, rocblas_bfdot_strided_batched);
MAP2CF_D64(rocblas_dot_strided_batched, rocblas_float_complex, rocblas_cdotu_strided_batched);
MAP2CF_D64(rocblas_dot_strided_batched, rocblas_double_complex, rocblas_zdotu_strided_batched);

// dotc
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dotc_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      rocblas_int    batch_count,
                                                      T*             result);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dotc_strided_batched_64)(rocblas_handle handle,
                                                         int64_t        n,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stridex,
                                                         const T*       y,
                                                         int64_t        incy,
                                                         rocblas_stride stridey,
                                                         int64_t        batch_count,
                                                         T*             result);

MAP2CF_D64(rocblas_dotc_strided_batched, rocblas_float_complex, rocblas_cdotc_strided_batched);
MAP2CF_D64(rocblas_dotc_strided_batched, rocblas_double_complex, rocblas_zdotc_strided_batched);

// asum
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_asum)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, real_t<T>* result);

// asum_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_asum_64)(
    rocblas_handle handle, int64_t n, const T* x, int64_t incx, real_t<T>* result);

MAP2CF_D64(rocblas_asum, float, rocblas_sasum);
MAP2CF_D64(rocblas_asum, double, rocblas_dasum);
MAP2CF_D64(rocblas_asum, rocblas_float_complex, rocblas_scasum);
MAP2CF_D64(rocblas_asum, rocblas_double_complex, rocblas_dzasum);

// asum_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_asum_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              rocblas_int    batch_count,
                                              real_t<T>*     result);
// asum_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_asum_batched_64)(rocblas_handle handle,
                                                 int64_t        n,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 int64_t        batch_count,
                                                 real_t<T>*     result);

MAP2CF_D64(rocblas_asum_batched, float, rocblas_sasum_batched);
MAP2CF_D64(rocblas_asum_batched, double, rocblas_dasum_batched);
MAP2CF_D64(rocblas_asum_batched, rocblas_float_complex, rocblas_scasum_batched);
MAP2CF_D64(rocblas_asum_batched, rocblas_double_complex, rocblas_dzasum_batched);

// asum_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_asum_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      rocblas_int    batch_count,
                                                      real_t<T>*     result);

// asum_strided_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_asum_strided_batched_64)(rocblas_handle handle,
                                                         int64_t        n,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stridex,
                                                         int64_t        batch_count,
                                                         real_t<T>*     result);

MAP2CF_D64(rocblas_asum_strided_batched, float, rocblas_sasum_strided_batched);
MAP2CF_D64(rocblas_asum_strided_batched, double, rocblas_dasum_strided_batched);
MAP2CF_D64(rocblas_asum_strided_batched, rocblas_float_complex, rocblas_scasum_strided_batched);
MAP2CF_D64(rocblas_asum_strided_batched, rocblas_double_complex, rocblas_dzasum_strided_batched);

// nrm2
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_nrm2)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, real_t<T>* result);

//nrm2_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_nrm2_64)(
    rocblas_handle handle, int64_t n, const T* x, int64_t incx, real_t<T>* result);

MAP2CF_D64(rocblas_nrm2, float, rocblas_snrm2);
MAP2CF_D64(rocblas_nrm2, double, rocblas_dnrm2);
MAP2CF_D64(rocblas_nrm2, rocblas_float_complex, rocblas_scnrm2);
MAP2CF_D64(rocblas_nrm2, rocblas_double_complex, rocblas_dznrm2);

// nrm2_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_nrm2_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              rocblas_int    batch_count,
                                              real_t<T>*     results);

// nrm2_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_nrm2_batched_64)(rocblas_handle handle,
                                                 int64_t        n,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 int64_t        batch_count,
                                                 real_t<T>*     results);

MAP2CF_D64(rocblas_nrm2_batched, float, rocblas_snrm2_batched);
MAP2CF_D64(rocblas_nrm2_batched, double, rocblas_dnrm2_batched);
MAP2CF_D64(rocblas_nrm2_batched, rocblas_float_complex, rocblas_scnrm2_batched);
MAP2CF_D64(rocblas_nrm2_batched, rocblas_double_complex, rocblas_dznrm2_batched);

// nrm2_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_nrm2_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      rocblas_int    batch_count,
                                                      real_t<T>*     results);

// nrm2_strided_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_nrm2_strided_batched_64)(rocblas_handle handle,
                                                         int64_t        n,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stridex,
                                                         int64_t        batch_count,
                                                         real_t<T>*     results);

MAP2CF_D64(rocblas_nrm2_strided_batched, float, rocblas_snrm2_strided_batched);
MAP2CF_D64(rocblas_nrm2_strided_batched, double, rocblas_dnrm2_strided_batched);
MAP2CF_D64(rocblas_nrm2_strided_batched, rocblas_float_complex, rocblas_scnrm2_strided_batched);
MAP2CF_D64(rocblas_nrm2_strided_batched, rocblas_double_complex, rocblas_dznrm2_strided_batched);

// iamax and iamin need to be full functions rather than references, in order
// to allow them to be passed as template arguments
//
// iamax and iamin.
//

//
// Define the signature type.
//
template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_t = rocblas_status (*)(
    rocblas_handle handle, rocblas_int n, const T* x, rocblas_int incx, rocblas_int* result);

template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_64_t = rocblas_status (*)(
    rocblas_handle handle, int64_t n, const T* x, int64_t incx, int64_t* result);

//
// iamax
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_t<T, FORTRAN> rocblas_iamax;
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_64_t<T, FORTRAN> rocblas_iamax_64;

MAP2CF_D64(rocblas_iamax, float, rocblas_isamax);
MAP2CF_D64(rocblas_iamax, double, rocblas_idamax);
MAP2CF_D64(rocblas_iamax, rocblas_float_complex, rocblas_icamax);
MAP2CF_D64(rocblas_iamax, rocblas_double_complex, rocblas_izamax);

//
// iamin
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_t<T, FORTRAN> rocblas_iamin;
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_64_t<T, FORTRAN> rocblas_iamin_64;

MAP2CF_D64(rocblas_iamin, float, rocblas_isamin);
MAP2CF_D64(rocblas_iamin, double, rocblas_idamin);
MAP2CF_D64(rocblas_iamin, rocblas_float_complex, rocblas_icamin);
MAP2CF_D64(rocblas_iamin, rocblas_double_complex, rocblas_izamin);

//
// Define the signature type for the iamax_iamin batched.
//
template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_batched_t = rocblas_status (*)(rocblas_handle  handle,
                                                         rocblas_int     n,
                                                         const T* const* x,
                                                         rocblas_int     incx,
                                                         rocblas_int     batch_count,
                                                         rocblas_int*    result);
template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_batched_64_t = rocblas_status (*)(rocblas_handle  handle,
                                                            int64_t         n,
                                                            const T* const* x,
                                                            int64_t         incx,
                                                            int64_t         batch_count,
                                                            int64_t*        result);

//
// iamax
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_batched_t<T, FORTRAN> rocblas_iamax_batched;
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_batched_64_t<T, FORTRAN> rocblas_iamax_batched_64;

MAP2CF_D64(rocblas_iamax_batched, float, rocblas_isamax_batched);
MAP2CF_D64(rocblas_iamax_batched, double, rocblas_idamax_batched);
MAP2CF_D64(rocblas_iamax_batched, rocblas_float_complex, rocblas_icamax_batched);
MAP2CF_D64(rocblas_iamax_batched, rocblas_double_complex, rocblas_izamax_batched);

//
// iamin
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_batched_t<T, FORTRAN> rocblas_iamin_batched;
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_batched_64_t<T, FORTRAN> rocblas_iamin_batched_64;

MAP2CF_D64(rocblas_iamin_batched, float, rocblas_isamin_batched);
MAP2CF_D64(rocblas_iamin_batched, double, rocblas_idamin_batched);
MAP2CF_D64(rocblas_iamin_batched, rocblas_float_complex, rocblas_icamin_batched);
MAP2CF_D64(rocblas_iamin_batched, rocblas_double_complex, rocblas_izamin_batched);

//
// Define the signature type for the iamax_iamin strided batched.
//
template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_strided_batched_t = rocblas_status (*)(rocblas_handle handle,
                                                                 rocblas_int    n,
                                                                 const T*       x,
                                                                 rocblas_int    incx,
                                                                 rocblas_stride stridex,
                                                                 rocblas_int    batch_count,
                                                                 rocblas_int*   result);
template <typename T, bool FORTRAN = false>
using rocblas_iamax_iamin_strided_batched_64_t = rocblas_status (*)(rocblas_handle handle,
                                                                    int64_t        n,
                                                                    const T*       x,
                                                                    int64_t        incx,
                                                                    rocblas_stride stridex,
                                                                    int64_t        batch_count,
                                                                    int64_t*       result);

//
// iamax
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_strided_batched_t<T, FORTRAN> rocblas_iamax_strided_batched;
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_strided_batched_64_t<T, FORTRAN> rocblas_iamax_strided_batched_64;

MAP2CF_D64(rocblas_iamax_strided_batched, float, rocblas_isamax_strided_batched);
MAP2CF_D64(rocblas_iamax_strided_batched, double, rocblas_idamax_strided_batched);
MAP2CF_D64(rocblas_iamax_strided_batched, rocblas_float_complex, rocblas_icamax_strided_batched);
MAP2CF_D64(rocblas_iamax_strided_batched, rocblas_double_complex, rocblas_izamax_strided_batched);

//
// iamin
//
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_strided_batched_t<T, FORTRAN> rocblas_iamin_strided_batched;
template <typename T, bool FORTRAN = false>
rocblas_iamax_iamin_strided_batched_64_t<T, FORTRAN> rocblas_iamin_strided_batched_64;

MAP2CF_D64(rocblas_iamin_strided_batched, float, rocblas_isamin_strided_batched);
MAP2CF_D64(rocblas_iamin_strided_batched, double, rocblas_idamin_strided_batched);
MAP2CF_D64(rocblas_iamin_strided_batched, rocblas_float_complex, rocblas_icamin_strided_batched);
MAP2CF_D64(rocblas_iamin_strided_batched, rocblas_double_complex, rocblas_izamin_strided_batched);

// axpy
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_axpy)(rocblas_handle handle,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      T*             y,
                                      rocblas_int    incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_axpy_64)(
    rocblas_handle handle, int64_t n, const T* alpha, const T* x, int64_t incx, T* y, int64_t incy);

MAP2CF_D64(rocblas_axpy, float, rocblas_saxpy);
MAP2CF_D64(rocblas_axpy, double, rocblas_daxpy);
MAP2CF_D64(rocblas_axpy, rocblas_half, rocblas_haxpy);
MAP2CF_D64(rocblas_axpy, rocblas_float_complex, rocblas_caxpy);
MAP2CF_D64(rocblas_axpy, rocblas_double_complex, rocblas_zaxpy);

// axpy batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_axpy_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_axpy_batched_64)(rocblas_handle handle,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_axpy_batched, float, rocblas_saxpy_batched);
MAP2CF_D64(rocblas_axpy_batched, double, rocblas_daxpy_batched);
MAP2CF_D64(rocblas_axpy_batched, rocblas_half, rocblas_haxpy_batched);
MAP2CF_D64(rocblas_axpy_batched, rocblas_float_complex, rocblas_caxpy_batched);
MAP2CF_D64(rocblas_axpy_batched, rocblas_double_complex, rocblas_zaxpy_batched);

// axpy strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_axpy_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_axpy_strided_batched_64)(rocblas_handle handle,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stridex,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stridey,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_axpy_strided_batched, float, rocblas_saxpy_strided_batched);
MAP2CF_D64(rocblas_axpy_strided_batched, double, rocblas_daxpy_strided_batched);
MAP2CF_D64(rocblas_axpy_strided_batched, rocblas_half, rocblas_haxpy_strided_batched);
MAP2CF_D64(rocblas_axpy_strided_batched, rocblas_float_complex, rocblas_caxpy_strided_batched);
MAP2CF_D64(rocblas_axpy_strided_batched, rocblas_double_complex, rocblas_zaxpy_strided_batched);

// rot
template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rot)(rocblas_handle handle,
                                     rocblas_int    n,
                                     T*             x,
                                     rocblas_int    incx,
                                     T*             y,
                                     rocblas_int    incy,
                                     const U*       c,
                                     const V*       s);

template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rot_64)(rocblas_handle handle,
                                        int64_t        n,
                                        T*             x,
                                        int64_t        incx,
                                        T*             y,
                                        int64_t        incy,
                                        const U*       c,
                                        const V*       s);

MAP2CF_D64(rocblas_rot, float, float, float, rocblas_srot);
MAP2CF_D64(rocblas_rot, double, double, double, rocblas_drot);
MAP2CF_D64(rocblas_rot, rocblas_float_complex, float, rocblas_float_complex, rocblas_crot);
MAP2CF_D64(rocblas_rot, rocblas_float_complex, float, float, rocblas_csrot);
MAP2CF_D64(rocblas_rot, rocblas_double_complex, double, rocblas_double_complex, rocblas_zrot);
MAP2CF_D64(rocblas_rot, rocblas_double_complex, double, double, rocblas_zdrot);

// rot_batched
template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rot_batched)(rocblas_handle handle,
                                             rocblas_int    n,
                                             T* const       x[],
                                             rocblas_int    incx,
                                             T* const       y[],
                                             rocblas_int    incy,
                                             const U*       c,
                                             const V*       s,
                                             rocblas_int    batch_count);

template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rot_batched_64)(rocblas_handle handle,
                                                int64_t        n,
                                                T* const       x[],
                                                int64_t        incx,
                                                T* const       y[],
                                                int64_t        incy,
                                                const U*       c,
                                                const V*       s,
                                                int64_t        batch_count);

MAP2CF_D64(rocblas_rot_batched, float, float, float, rocblas_srot_batched);
MAP2CF_D64(rocblas_rot_batched, double, double, double, rocblas_drot_batched);
MAP2CF_D64(
    rocblas_rot_batched, rocblas_float_complex, float, rocblas_float_complex, rocblas_crot_batched);
MAP2CF_D64(rocblas_rot_batched, rocblas_float_complex, float, float, rocblas_csrot_batched);
MAP2CF_D64(rocblas_rot_batched,
           rocblas_double_complex,
           double,
           rocblas_double_complex,
           rocblas_zrot_batched);
MAP2CF_D64(rocblas_rot_batched, rocblas_double_complex, double, double, rocblas_zdrot_batched);

// rot_strided_batched
template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rot_strided_batched)(rocblas_handle handle,
                                                     rocblas_int    n,
                                                     T*             x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     T*             y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stride_y,
                                                     const U*       c,
                                                     const V*       s,
                                                     rocblas_int    batch_count);

template <typename T, typename U = T, typename V = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rot_strided_batched_64)(rocblas_handle handle,
                                                        int64_t        n,
                                                        T*             x,
                                                        int64_t        incx,
                                                        rocblas_stride stride_x,
                                                        T*             y,
                                                        int64_t        incy,
                                                        rocblas_stride stride_y,
                                                        const U*       c,
                                                        const V*       s,
                                                        int64_t        batch_count);

MAP2CF_D64(rocblas_rot_strided_batched, float, float, float, rocblas_srot_strided_batched);
MAP2CF_D64(rocblas_rot_strided_batched, double, double, double, rocblas_drot_strided_batched);
MAP2CF_D64(rocblas_rot_strided_batched,
           rocblas_float_complex,
           float,
           rocblas_float_complex,
           rocblas_crot_strided_batched);
MAP2CF_D64(rocblas_rot_strided_batched,
           rocblas_float_complex,
           float,
           float,
           rocblas_csrot_strided_batched);
MAP2CF_D64(rocblas_rot_strided_batched,
           rocblas_double_complex,
           double,
           rocblas_double_complex,
           rocblas_zrot_strided_batched);
MAP2CF_D64(rocblas_rot_strided_batched,
           rocblas_double_complex,
           double,
           double,
           rocblas_zdrot_strided_batched);

// rotg
template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotg)(rocblas_handle handle, T* a, T* b, U* c, T* s);

template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotg_64)(rocblas_handle handle, T* a, T* b, U* c, T* s);

MAP2CF_D64(rocblas_rotg, float, float, rocblas_srotg);
MAP2CF_D64(rocblas_rotg, double, double, rocblas_drotg);
MAP2CF_D64(rocblas_rotg, rocblas_float_complex, float, rocblas_crotg);
MAP2CF_D64(rocblas_rotg, rocblas_double_complex, double, rocblas_zrotg);

// rotg_batched
template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotg_batched)(rocblas_handle handle,
                                              T* const       a[],
                                              T* const       b[],
                                              U* const       c[],
                                              T* const       s[],
                                              rocblas_int    batch_count);

template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotg_batched_64)(rocblas_handle handle,
                                                 T* const       a[],
                                                 T* const       b[],
                                                 U* const       c[],
                                                 T* const       s[],
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_rotg_batched, float, float, rocblas_srotg_batched);
MAP2CF_D64(rocblas_rotg_batched, double, double, rocblas_drotg_batched);
MAP2CF_D64(rocblas_rotg_batched, rocblas_float_complex, float, rocblas_crotg_batched);
MAP2CF_D64(rocblas_rotg_batched, rocblas_double_complex, double, rocblas_zrotg_batched);

//rotg_strided_batched
template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotg_strided_batched)(rocblas_handle handle,
                                                      T*             a,
                                                      rocblas_stride stride_a,
                                                      T*             b,
                                                      rocblas_stride stride_b,
                                                      U*             c,
                                                      rocblas_stride stride_c,
                                                      T*             s,
                                                      rocblas_stride stride_s,
                                                      rocblas_int    batch_count);

template <typename T, typename U = T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotg_strided_batched_64)(rocblas_handle handle,
                                                         T*             a,
                                                         rocblas_stride stride_a,
                                                         T*             b,
                                                         rocblas_stride stride_b,
                                                         U*             c,
                                                         rocblas_stride stride_c,
                                                         T*             s,
                                                         rocblas_stride stride_s,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_rotg_strided_batched, float, float, rocblas_srotg_strided_batched);
MAP2CF_D64(rocblas_rotg_strided_batched, double, double, rocblas_drotg_strided_batched);
MAP2CF_D64(rocblas_rotg_strided_batched,
           rocblas_float_complex,
           float,
           rocblas_crotg_strided_batched);
MAP2CF_D64(rocblas_rotg_strided_batched,
           rocblas_double_complex,
           double,
           rocblas_zrotg_strided_batched);

//rotm
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotm)(rocblas_handle handle,
                                      rocblas_int    n,
                                      T*             x,
                                      rocblas_int    incx,
                                      T*             y,
                                      rocblas_int    incy,
                                      const T*       param);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotm_64)(
    rocblas_handle handle, int64_t n, T* x, int64_t incx, T* y, int64_t incy, const T* param);

MAP2CF_D64(rocblas_rotm, float, rocblas_srotm);
MAP2CF_D64(rocblas_rotm, double, rocblas_drotm);

// rotm_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotm_batched)(rocblas_handle handle,
                                              rocblas_int    n,
                                              T* const       x[],
                                              rocblas_int    incx,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              const T* const param[],
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotm_batched_64)(rocblas_handle handle,
                                                 int64_t        n,
                                                 T* const       x[],
                                                 int64_t        incx,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 const T* const param[],
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_rotm_batched, float, rocblas_srotm_batched);
MAP2CF_D64(rocblas_rotm_batched, double, rocblas_drotm_batched);

// rotm_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotm_strided_batched)(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      T*             x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      const T*       param,
                                                      rocblas_stride stride_param,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotm_strided_batched_64)(rocblas_handle handle,
                                                         int64_t        n,
                                                         T*             x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         const T*       param,
                                                         rocblas_stride stride_param,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_rotm_strided_batched, float, rocblas_srotm_strided_batched);
MAP2CF_D64(rocblas_rotm_strided_batched, double, rocblas_drotm_strided_batched);

//rotmg
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotmg)(
    rocblas_handle handle, T* d1, T* d2, T* x1, const T* y1, T* param);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotmg_64)(
    rocblas_handle handle, T* d1, T* d2, T* x1, const T* y1, T* param);

MAP2CF_D64(rocblas_rotmg, float, rocblas_srotmg);
MAP2CF_D64(rocblas_rotmg, double, rocblas_drotmg);

//rotmg_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotmg_batched)(rocblas_handle handle,
                                               T* const       d1[],
                                               T* const       d2[],
                                               T* const       x1[],
                                               const T* const y1[],
                                               T* const       param[],
                                               rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotmg_batched_64)(rocblas_handle handle,
                                                  T* const       d1[],
                                                  T* const       d2[],
                                                  T* const       x1[],
                                                  const T* const y1[],
                                                  T* const       param[],
                                                  int64_t        batch_count);

MAP2CF_D64(rocblas_rotmg_batched, float, rocblas_srotmg_batched);
MAP2CF_D64(rocblas_rotmg_batched, double, rocblas_drotmg_batched);

//rotmg_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotmg_strided_batched)(rocblas_handle handle,
                                                       T*             d1,
                                                       rocblas_stride stride_d1,
                                                       T*             d2,
                                                       rocblas_stride stride_d2,
                                                       T*             x1,
                                                       rocblas_stride stride_x1,
                                                       const T*       y1,
                                                       rocblas_stride stride_y1,
                                                       T*             param,
                                                       rocblas_stride stride_param,
                                                       rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_rotmg_strided_batched_64)(rocblas_handle handle,
                                                          T*             d1,
                                                          rocblas_stride stride_d1,
                                                          T*             d2,
                                                          rocblas_stride stride_d2,
                                                          T*             x1,
                                                          rocblas_stride stride_x1,
                                                          const T*       y1,
                                                          rocblas_stride stride_y1,
                                                          T*             param,
                                                          rocblas_stride stride_param,
                                                          int64_t        batch_count);

MAP2CF_D64(rocblas_rotmg_strided_batched, float, rocblas_srotmg_strided_batched);
MAP2CF_D64(rocblas_rotmg_strided_batched, double, rocblas_drotmg_strided_batched);

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// ger
template <typename T, bool CONJ, bool FORTRAN = false>
inline rocblas_status (*rocblas_ger)(rocblas_handle handle,
                                     rocblas_int    m,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       x,
                                     rocblas_int    incx,
                                     const T*       y,
                                     rocblas_int    incy,
                                     T*             A,
                                     rocblas_int    lda);

template <typename T, bool CONJ, bool FORTRAN = false>
inline rocblas_status (*rocblas_ger_64)(rocblas_handle handle,
                                        int64_t        m,
                                        int64_t        n,
                                        const T*       alpha,
                                        const T*       x,
                                        int64_t        incx,
                                        const T*       y,
                                        int64_t        incy,
                                        T*             A,
                                        int64_t        lda);

MAP2CF_D64(rocblas_ger, float, false, rocblas_sger);
MAP2CF_D64(rocblas_ger, double, false, rocblas_dger);
MAP2CF_D64(rocblas_ger, rocblas_float_complex, false, rocblas_cgeru);
MAP2CF_D64(rocblas_ger, rocblas_double_complex, false, rocblas_zgeru);
MAP2CF_D64(rocblas_ger, rocblas_float_complex, true, rocblas_cgerc);
MAP2CF_D64(rocblas_ger, rocblas_double_complex, true, rocblas_zgerc);

template <typename T, bool CONJ, bool FORTRAN = false>
inline rocblas_status (*rocblas_ger_batched)(rocblas_handle handle,
                                             rocblas_int    m,
                                             rocblas_int    n,
                                             const T*       alpha,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             const T* const y[],
                                             rocblas_int    incy,
                                             T* const       A[],
                                             rocblas_int    lda,
                                             rocblas_int    batch_count);

template <typename T, bool CONJ, bool FORTRAN = false>
inline rocblas_status (*rocblas_ger_batched_64)(rocblas_handle handle,
                                                int64_t        m,
                                                int64_t        n,
                                                const T*       alpha,
                                                const T* const x[],
                                                int64_t        incx,
                                                const T* const y[],
                                                int64_t        incy,
                                                T* const       A[],
                                                int64_t        lda,
                                                int64_t        batch_count);

MAP2CF_D64(rocblas_ger_batched, float, false, rocblas_sger_batched);
MAP2CF_D64(rocblas_ger_batched, double, false, rocblas_dger_batched);
MAP2CF_D64(rocblas_ger_batched, rocblas_float_complex, false, rocblas_cgeru_batched);
MAP2CF_D64(rocblas_ger_batched, rocblas_double_complex, false, rocblas_zgeru_batched);
MAP2CF_D64(rocblas_ger_batched, rocblas_float_complex, true, rocblas_cgerc_batched);
MAP2CF_D64(rocblas_ger_batched, rocblas_double_complex, true, rocblas_zgerc_batched);

template <typename T, bool CONJ, bool FORTRAN = false>
inline rocblas_status (*rocblas_ger_strided_batched)(rocblas_handle handle,
                                                     rocblas_int    m,
                                                     rocblas_int    n,
                                                     const T*       alpha,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     const T*       y,
                                                     rocblas_int    incy,
                                                     rocblas_stride stride_y,
                                                     T*             A,
                                                     rocblas_int    lda,
                                                     rocblas_stride stride_a,
                                                     rocblas_int    batch_count);

template <typename T, bool CONJ, bool FORTRAN = false>
inline rocblas_status (*rocblas_ger_strided_batched_64)(rocblas_handle handle,
                                                        int64_t        m,
                                                        int64_t        n,
                                                        const T*       alpha,
                                                        const T*       x,
                                                        int64_t        incx,
                                                        rocblas_stride stride_x,
                                                        const T*       y,
                                                        int64_t        incy,
                                                        rocblas_stride stride_y,
                                                        T*             A,
                                                        int64_t        lda,
                                                        rocblas_stride stride_a,
                                                        int64_t        batch_count);

MAP2CF_D64(rocblas_ger_strided_batched, float, false, rocblas_sger_strided_batched);
MAP2CF_D64(rocblas_ger_strided_batched, double, false, rocblas_dger_strided_batched);
MAP2CF_D64(rocblas_ger_strided_batched,
           rocblas_float_complex,
           false,
           rocblas_cgeru_strided_batched);
MAP2CF_D64(rocblas_ger_strided_batched,
           rocblas_double_complex,
           false,
           rocblas_zgeru_strided_batched);
MAP2CF_D64(rocblas_ger_strided_batched, rocblas_float_complex, true, rocblas_cgerc_strided_batched);
MAP2CF_D64(rocblas_ger_strided_batched,
           rocblas_double_complex,
           true,
           rocblas_zgerc_strided_batched);

// spr
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr)(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       x,
                                     rocblas_int    incx,
                                     T*             AP);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr_64)(rocblas_handle handle,
                                        rocblas_fill   uplo,
                                        int64_t        n,
                                        const T*       alpha,
                                        const T*       x,
                                        int64_t        incx,
                                        T*             AP);

MAP2CF_D64(rocblas_spr, float, rocblas_sspr);
MAP2CF_D64(rocblas_spr, double, rocblas_dspr);
MAP2CF_D64(rocblas_spr, rocblas_float_complex, rocblas_cspr);
MAP2CF_D64(rocblas_spr, rocblas_double_complex, rocblas_zspr);

// spr_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr_batched)(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             const T*       alpha,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             T* const       AP[],
                                             rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr_batched_64)(rocblas_handle handle,
                                                rocblas_fill   uplo,
                                                int64_t        n,
                                                const T*       alpha,
                                                const T* const x[],
                                                int64_t        incx,
                                                T* const       AP[],
                                                int64_t        batch_count);

MAP2CF_D64(rocblas_spr_batched, float, rocblas_sspr_batched);
MAP2CF_D64(rocblas_spr_batched, double, rocblas_dspr_batched);
MAP2CF_D64(rocblas_spr_batched, rocblas_float_complex, rocblas_cspr_batched);
MAP2CF_D64(rocblas_spr_batched, rocblas_double_complex, rocblas_zspr_batched);

// spr_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr_strided_batched)(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     rocblas_int    n,
                                                     const T*       alpha,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stride_x,
                                                     T*             AP,
                                                     rocblas_stride stride_A,
                                                     rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr_strided_batched_64)(rocblas_handle handle,
                                                        rocblas_fill   uplo,
                                                        int64_t        n,
                                                        const T*       alpha,
                                                        const T*       x,
                                                        int64_t        incx,
                                                        rocblas_stride stride_x,
                                                        T*             AP,
                                                        rocblas_stride stride_A,
                                                        int64_t        batch_count);

MAP2CF_D64(rocblas_spr_strided_batched, float, rocblas_sspr_strided_batched);
MAP2CF_D64(rocblas_spr_strided_batched, double, rocblas_dspr_strided_batched);
MAP2CF_D64(rocblas_spr_strided_batched, rocblas_float_complex, rocblas_cspr_strided_batched);
MAP2CF_D64(rocblas_spr_strided_batched, rocblas_double_complex, rocblas_zspr_strided_batched);

// spr2
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr2)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             AP);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr2_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       y,
                                         int64_t        incy,
                                         T*             AP);

MAP2CF_D64(rocblas_spr2, float, rocblas_sspr2);
MAP2CF_D64(rocblas_spr2, double, rocblas_dspr2);

// spr2_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr2_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              T* const       AP[],
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr2_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T* const y[],
                                                 int64_t        incy,
                                                 T* const       AP[],
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_spr2_batched, float, rocblas_sspr2_batched);
MAP2CF_D64(rocblas_spr2_batched, double, rocblas_dspr2_batched);

// spr2_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr2_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      T*             AP,
                                                      rocblas_stride stride_A,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spr2_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         T*             AP,
                                                         rocblas_stride stride_A,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_spr2_strided_batched, float, rocblas_sspr2_strided_batched);
MAP2CF_D64(rocblas_spr2_strided_batched, double, rocblas_dspr2_strided_batched);

// syr
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr)(rocblas_handle handle,
                                     rocblas_fill   uplo,
                                     rocblas_int    n,
                                     const T*       alpha,
                                     const T*       x,
                                     rocblas_int    incx,
                                     T*             A,
                                     rocblas_int    lda);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr_64)(rocblas_handle handle,
                                        rocblas_fill   uplo,
                                        int64_t        n,
                                        const T*       alpha,
                                        const T*       x,
                                        int64_t        incx,
                                        T*             A,
                                        int64_t        lda);

MAP2CF_D64(rocblas_syr, float, rocblas_ssyr);
MAP2CF_D64(rocblas_syr, double, rocblas_dsyr);
MAP2CF_D64(rocblas_syr, rocblas_float_complex, rocblas_csyr);
MAP2CF_D64(rocblas_syr, rocblas_double_complex, rocblas_zsyr);

// syr batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr_batched)(rocblas_handle handle,
                                             rocblas_fill   uplo,
                                             rocblas_int    n,
                                             const T*       alpha,
                                             const T* const x[],
                                             rocblas_int    incx,
                                             T* const       A[],
                                             rocblas_int    lda,
                                             rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr_batched_64)(rocblas_handle handle,
                                                rocblas_fill   uplo,
                                                int64_t        n,
                                                const T*       alpha,
                                                const T* const x[],
                                                int64_t        incx,
                                                T* const       A[],
                                                int64_t        lda,
                                                int64_t        batch_count);

MAP2CF_D64(rocblas_syr_batched, float, rocblas_ssyr_batched);
MAP2CF_D64(rocblas_syr_batched, double, rocblas_dsyr_batched);
MAP2CF_D64(rocblas_syr_batched, rocblas_float_complex, rocblas_csyr_batched);
MAP2CF_D64(rocblas_syr_batched, rocblas_double_complex, rocblas_zsyr_batched);

// syr strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr_strided_batched)(rocblas_handle handle,
                                                     rocblas_fill   uplo,
                                                     rocblas_int    n,
                                                     const T*       alpha,
                                                     const T*       x,
                                                     rocblas_int    incx,
                                                     rocblas_stride stridex,
                                                     T*             A,
                                                     rocblas_int    lda,
                                                     rocblas_stride strideA,
                                                     rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr_strided_batched_64)(rocblas_handle handle,
                                                        rocblas_fill   uplo,
                                                        int64_t        n,
                                                        const T*       alpha,
                                                        const T*       x,
                                                        int64_t        incx,
                                                        rocblas_stride stridex,
                                                        T*             A,
                                                        int64_t        lda,
                                                        rocblas_stride strideA,
                                                        int64_t        batch_count);
MAP2CF_D64(rocblas_syr_strided_batched, float, rocblas_ssyr_strided_batched);
MAP2CF_D64(rocblas_syr_strided_batched, double, rocblas_dsyr_strided_batched);
MAP2CF_D64(rocblas_syr_strided_batched, rocblas_float_complex, rocblas_csyr_strided_batched);
MAP2CF_D64(rocblas_syr_strided_batched, rocblas_double_complex, rocblas_zsyr_strided_batched);

// syr2
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             A,
                                      rocblas_int    lda);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       y,
                                         rocblas_int    incy,
                                         T*             A,
                                         int64_t        lda);
MAP2CF_D64(rocblas_syr2, float, rocblas_ssyr2);
MAP2CF_D64(rocblas_syr2, double, rocblas_dsyr2);
MAP2CF_D64(rocblas_syr2, rocblas_float_complex, rocblas_csyr2);
MAP2CF_D64(rocblas_syr2, rocblas_double_complex, rocblas_zsyr2);

// syr2 batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              T* const       A[],
                                              rocblas_int    lda,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T* const y[],
                                                 rocblas_int    incy,
                                                 T* const       A[],
                                                 int64_t        lda,
                                                 int64_t        batch_count);
MAP2CF_D64(rocblas_syr2_batched, float, rocblas_ssyr2_batched);
MAP2CF_D64(rocblas_syr2_batched, double, rocblas_dsyr2_batched);
MAP2CF_D64(rocblas_syr2_batched, rocblas_float_complex, rocblas_csyr2_batched);
MAP2CF_D64(rocblas_syr2_batched, rocblas_double_complex, rocblas_zsyr2_batched);

// syr2 strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stridex,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stridey,
                                                      T*             A,
                                                      rocblas_int    lda,
                                                      rocblas_stride strideA,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stridex,
                                                         const T*       y,
                                                         rocblas_int    incy,
                                                         rocblas_stride stridey,
                                                         T*             A,
                                                         int64_t        lda,
                                                         rocblas_stride strideA,
                                                         int64_t        batch_count);
MAP2CF_D64(rocblas_syr2_strided_batched, float, rocblas_ssyr2_strided_batched);
MAP2CF_D64(rocblas_syr2_strided_batched, double, rocblas_dsyr2_strided_batched);
MAP2CF_D64(rocblas_syr2_strided_batched, rocblas_float_complex, rocblas_csyr2_strided_batched);
MAP2CF_D64(rocblas_syr2_strided_batched, rocblas_double_complex, rocblas_zsyr2_strided_batched);

// gbmv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gbmv)(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      rocblas_int       kl,
                                      rocblas_int       ku,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          x,
                                      rocblas_int       incx,
                                      const T*          beta,
                                      T*                y,
                                      rocblas_int       incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gbmv_64)(rocblas_handle    handle,
                                         rocblas_operation transA,
                                         int64_t           m,
                                         int64_t           n,
                                         int64_t           kl,
                                         int64_t           ku,
                                         const T*          alpha,
                                         const T*          A,
                                         int64_t           lda,
                                         const T*          x,
                                         int64_t           incx,
                                         const T*          beta,
                                         T*                y,
                                         int64_t           incy);

MAP2CF_D64(rocblas_gbmv, float, rocblas_sgbmv);
MAP2CF_D64(rocblas_gbmv, double, rocblas_dgbmv);
MAP2CF_D64(rocblas_gbmv, rocblas_float_complex, rocblas_cgbmv);
MAP2CF_D64(rocblas_gbmv, rocblas_double_complex, rocblas_zgbmv);

// gbmv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gbmv_batched)(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              rocblas_int       kl,
                                              rocblas_int       ku,
                                              const T*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T* const    x[],
                                              rocblas_int       incx,
                                              const T*          beta,
                                              T* const          y[],
                                              rocblas_int       incy,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gbmv_batched_64)(rocblas_handle    handle,
                                                 rocblas_operation transA,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 int64_t           kl,
                                                 int64_t           ku,
                                                 const T*          alpha,
                                                 const T* const    A[],
                                                 int64_t           lda,
                                                 const T* const    x[],
                                                 int64_t           incx,
                                                 const T*          beta,
                                                 T* const          y[],
                                                 int64_t           incy,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_gbmv_batched, float, rocblas_sgbmv_batched);
MAP2CF_D64(rocblas_gbmv_batched, double, rocblas_dgbmv_batched);
MAP2CF_D64(rocblas_gbmv_batched, rocblas_float_complex, rocblas_cgbmv_batched);
MAP2CF_D64(rocblas_gbmv_batched, rocblas_double_complex, rocblas_zgbmv_batched);

// gbmv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gbmv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      rocblas_int       kl,
                                                      rocblas_int       ku,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_A,
                                                      const T*          x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      const T*          beta,
                                                      T*                y,
                                                      rocblas_int       incy,
                                                      rocblas_stride    stride_y,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gbmv_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_operation transA,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         int64_t           kl,
                                                         int64_t           ku,
                                                         const T*          alpha,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_A,
                                                         const T*          x,
                                                         int64_t           incx,
                                                         rocblas_stride    stride_x,
                                                         const T*          beta,
                                                         T*                y,
                                                         int64_t           incy,
                                                         rocblas_stride    stride_y,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_gbmv_strided_batched, float, rocblas_sgbmv_strided_batched);
MAP2CF_D64(rocblas_gbmv_strided_batched, double, rocblas_dgbmv_strided_batched);
MAP2CF_D64(rocblas_gbmv_strided_batched, rocblas_float_complex, rocblas_cgbmv_strided_batched);
MAP2CF_D64(rocblas_gbmv_strided_batched, rocblas_double_complex, rocblas_zgbmv_strided_batched);

// gemv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemv)(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          x,
                                      rocblas_int       incx,
                                      const T*          beta,
                                      T*                y,
                                      rocblas_int       incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemv_64)(rocblas_handle    handle,
                                         rocblas_operation transA,
                                         int64_t           m,
                                         rocblas_int       n,
                                         const T*          alpha,
                                         const T*          A,
                                         int64_t           lda,
                                         const T*          x,
                                         int64_t           incx,
                                         const T*          beta,
                                         T*                y,
                                         int64_t           incy);

MAP2CF_D64(rocblas_gemv, float, rocblas_sgemv);
MAP2CF_D64(rocblas_gemv, double, rocblas_dgemv);
MAP2CF_D64(rocblas_gemv, rocblas_float_complex, rocblas_cgemv);
MAP2CF_D64(rocblas_gemv, rocblas_double_complex, rocblas_zgemv);

// gemv_batched
template <typename Ti, typename Tex = Ti, typename To = Tex, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemv_batched)(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const Tex*        alpha,
                                              const Ti* const   A[],
                                              rocblas_int       lda,
                                              const Ti* const   x[],
                                              rocblas_int       incx,
                                              const Tex*        beta,
                                              To* const         y[],
                                              rocblas_int       incy,
                                              rocblas_int       batch_count);

template <typename Ti, typename Tex = Ti, typename To = Tex, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemv_batched_64)(rocblas_handle    handle,
                                                 rocblas_operation transA,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 const Tex*        alpha,
                                                 const Ti* const   A[],
                                                 rocblas_int       lda,
                                                 const Ti* const   x[],
                                                 rocblas_int       incx,
                                                 const Tex*        beta,
                                                 To* const         y[],
                                                 int64_t           incy,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_gemv_batched, float, float, float, rocblas_sgemv_batched);
MAP2CF_D64(rocblas_gemv_batched, double, double, double, rocblas_dgemv_batched);
MAP2CF_D64(rocblas_gemv_batched,
           rocblas_float_complex,
           rocblas_float_complex,
           rocblas_float_complex,
           rocblas_cgemv_batched);
MAP2CF_D64(rocblas_gemv_batched,
           rocblas_double_complex,
           rocblas_double_complex,
           rocblas_double_complex,
           rocblas_zgemv_batched);
MAP2CF_D64(rocblas_gemv_batched, rocblas_half, float, rocblas_half, rocblas_hshgemv_batched);
MAP2CF_D64(rocblas_gemv_batched, rocblas_half, float, float, rocblas_hssgemv_batched);
MAP2CF_D64(
    rocblas_gemv_batched, rocblas_bfloat16, float, rocblas_bfloat16, rocblas_tstgemv_batched);
MAP2CF_D64(rocblas_gemv_batched, rocblas_bfloat16, float, float, rocblas_tssgemv_batched);

// gemv_strided_batched
template <typename Ti, typename Tex = Ti, typename To = Tex, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const Tex*        alpha,
                                                      const Ti*         A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const Ti*         x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      const Tex*        beta,
                                                      To*               y,
                                                      rocblas_int       incy,
                                                      rocblas_stride    stride_y,
                                                      rocblas_int       batch_count);

template <typename Ti, typename Tex = Ti, typename To = Tex, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemv_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_operation transA,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         const Tex*        alpha,
                                                         const Ti*         A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         const Ti*         x,
                                                         int64_t           incx,
                                                         rocblas_stride    stride_x,
                                                         const Tex*        beta,
                                                         To*               y,
                                                         int64_t           incy,
                                                         rocblas_stride    stride_y,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_gemv_strided_batched, float, float, float, rocblas_sgemv_strided_batched);
MAP2CF_D64(rocblas_gemv_strided_batched, double, double, double, rocblas_dgemv_strided_batched);
MAP2CF_D64(rocblas_gemv_strided_batched,
           rocblas_float_complex,
           rocblas_float_complex,
           rocblas_float_complex,
           rocblas_cgemv_strided_batched);
MAP2CF_D64(rocblas_gemv_strided_batched,
           rocblas_double_complex,
           rocblas_double_complex,
           rocblas_double_complex,
           rocblas_zgemv_strided_batched);
MAP2CF_D64(rocblas_gemv_strided_batched,
           rocblas_half,
           float,
           rocblas_half,
           rocblas_hshgemv_strided_batched);
MAP2CF_D64(
    rocblas_gemv_strided_batched, rocblas_half, float, float, rocblas_hssgemv_strided_batched);
MAP2CF_D64(rocblas_gemv_strided_batched,
           rocblas_bfloat16,
           float,
           rocblas_bfloat16,
           rocblas_tstgemv_strided_batched);
MAP2CF_D64(
    rocblas_gemv_strided_batched, rocblas_bfloat16, float, float, rocblas_tssgemv_strided_batched);

// tpmv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpmv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      const T*          A,
                                      T*                x,
                                      rocblas_int       incx);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpmv_64)(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         int64_t           m,
                                         const T*          A,
                                         T*                x,
                                         int64_t           incx);
MAP2CF_D64(rocblas_tpmv, float, rocblas_stpmv);
MAP2CF_D64(rocblas_tpmv, double, rocblas_dtpmv);
MAP2CF_D64(rocblas_tpmv, rocblas_float_complex, rocblas_ctpmv);
MAP2CF_D64(rocblas_tpmv, rocblas_double_complex, rocblas_ztpmv);

// tpmv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpmv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              const T* const*   A,
                                              T* const*         x,
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpmv_batched_64)(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           m,
                                                 const T* const*   A,
                                                 T* const*         x,
                                                 int64_t           incx,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_tpmv_batched, float, rocblas_stpmv_batched);
MAP2CF_D64(rocblas_tpmv_batched, double, rocblas_dtpmv_batched);
MAP2CF_D64(rocblas_tpmv_batched, rocblas_float_complex, rocblas_ctpmv_batched);
MAP2CF_D64(rocblas_tpmv_batched, rocblas_double_complex, rocblas_ztpmv_batched);

// tpmv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpmv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      const T*          A,
                                                      rocblas_stride    stridea,
                                                      T*                x,
                                                      rocblas_stride    stridex,
                                                      rocblas_int       incx,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpmv_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         int64_t           m,
                                                         const T*          A,
                                                         rocblas_stride    stridea,
                                                         T*                x,
                                                         rocblas_stride    stridex,
                                                         int64_t           incx,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_tpmv_strided_batched, float, rocblas_stpmv_strided_batched);
MAP2CF_D64(rocblas_tpmv_strided_batched, double, rocblas_dtpmv_strided_batched);
MAP2CF_D64(rocblas_tpmv_strided_batched, rocblas_float_complex, rocblas_ctpmv_strided_batched);
MAP2CF_D64(rocblas_tpmv_strided_batched, rocblas_double_complex, rocblas_ztpmv_strided_batched);

// hbmv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hbmv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      rocblas_int    k,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hbmv_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         int64_t        k,
                                         const T*       alpha,
                                         const T*       A,
                                         int64_t        lda,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       beta,
                                         T*             y,
                                         int64_t        incy);

MAP2CF_D64(rocblas_hbmv, rocblas_float_complex, rocblas_chbmv);
MAP2CF_D64(rocblas_hbmv, rocblas_double_complex, rocblas_zhbmv);

// hbmv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hbmv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              rocblas_int    k,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hbmv_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 int64_t        k,
                                                 const T*       alpha,
                                                 const T* const A[],
                                                 int64_t        lda,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T*       beta,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_hbmv_batched, rocblas_float_complex, rocblas_chbmv_batched);
MAP2CF_D64(rocblas_hbmv_batched, rocblas_double_complex, rocblas_zhbmv_batched);

// hbmv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hbmv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      rocblas_int    k,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_A,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hbmv_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         int64_t        k,
                                                         const T*       alpha,
                                                         const T*       A,
                                                         int64_t        lda,
                                                         rocblas_stride stride_A,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       beta,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_hbmv_strided_batched, rocblas_float_complex, rocblas_chbmv_strided_batched);
MAP2CF_D64(rocblas_hbmv_strided_batched, rocblas_double_complex, rocblas_zhbmv_strided_batched);

// hemv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemv_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       A,
                                         int64_t        lda,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       beta,
                                         T*             y,
                                         int64_t        incy);

MAP2CF_D64(rocblas_hemv, rocblas_float_complex, rocblas_chemv);
MAP2CF_D64(rocblas_hemv, rocblas_double_complex, rocblas_zhemv);

// hemv_batched
template <typename T, bool FOTRAN = false>
inline rocblas_status (*rocblas_hemv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <typename T, bool FOTRAN = false>
inline rocblas_status (*rocblas_hemv_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const A[],
                                                 int64_t        lda,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T*       beta,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_hemv_batched, rocblas_float_complex, rocblas_chemv_batched);
MAP2CF_D64(rocblas_hemv_batched, rocblas_double_complex, rocblas_zhemv_batched);

// hemv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_A,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemv_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       A,
                                                         int64_t        lda,
                                                         rocblas_stride stride_A,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       beta,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_hemv_strided_batched, rocblas_float_complex, rocblas_chemv_strided_batched);
MAP2CF_D64(rocblas_hemv_strided_batched, rocblas_double_complex, rocblas_zhemv_strided_batched);

// her
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her)(rocblas_handle   handle,
                                     rocblas_fill     uplo,
                                     rocblas_int      n,
                                     const real_t<T>* alpha,
                                     const T*         x,
                                     rocblas_int      incx,
                                     T*               A,
                                     rocblas_int      lda);

// her_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her_64)(rocblas_handle   handle,
                                        rocblas_fill     uplo,
                                        int64_t          n,
                                        const real_t<T>* alpha,
                                        const T*         x,
                                        int64_t          incx,
                                        T*               A,
                                        int64_t          lda);

MAP2CF_D64(rocblas_her, rocblas_float_complex, rocblas_cher);
MAP2CF_D64(rocblas_her, rocblas_double_complex, rocblas_zher);

// her_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her_batched)(rocblas_handle   handle,
                                             rocblas_fill     uplo,
                                             rocblas_int      n,
                                             const real_t<T>* alpha,
                                             const T* const   x[],
                                             rocblas_int      incx,
                                             T* const         A[],
                                             rocblas_int      lda,
                                             rocblas_int      batch_count);

// her_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her_batched_64)(rocblas_handle   handle,
                                                rocblas_fill     uplo,
                                                int64_t          n,
                                                const real_t<T>* alpha,
                                                const T* const   x[],
                                                int64_t          incx,
                                                T* const         A[],
                                                int64_t          lda,
                                                int64_t          batch_count);

MAP2CF_D64(rocblas_her_batched, rocblas_float_complex, rocblas_cher_batched);
MAP2CF_D64(rocblas_her_batched, rocblas_double_complex, rocblas_zher_batched);

// her_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her_strided_batched)(rocblas_handle   handle,
                                                     rocblas_fill     uplo,
                                                     rocblas_int      n,
                                                     const real_t<T>* alpha,
                                                     const T*         x,
                                                     rocblas_int      incx,
                                                     rocblas_stride   stride_x,
                                                     T*               A,
                                                     rocblas_int      lda,
                                                     rocblas_stride   stride_A,
                                                     rocblas_int      batch_count);

// her_strided_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her_strided_batched_64)(rocblas_handle   handle,
                                                        rocblas_fill     uplo,
                                                        int64_t          n,
                                                        const real_t<T>* alpha,
                                                        const T*         x,
                                                        int64_t          incx,
                                                        rocblas_stride   stride_x,
                                                        T*               A,
                                                        int64_t          lda,
                                                        rocblas_stride   stride_A,
                                                        int64_t          batch_count);

MAP2CF_D64(rocblas_her_strided_batched, rocblas_float_complex, rocblas_cher_strided_batched);
MAP2CF_D64(rocblas_her_strided_batched, rocblas_double_complex, rocblas_zher_strided_batched);

// her2
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             A,
                                      rocblas_int    lda);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       y,
                                         int64_t        incy,
                                         T*             A,
                                         int64_t        lda);

MAP2CF_D64(rocblas_her2, rocblas_float_complex, rocblas_cher2);
MAP2CF_D64(rocblas_her2, rocblas_double_complex, rocblas_zher2);

// her2_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              T* const       A[],
                                              rocblas_int    lda,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T* const y[],
                                                 int64_t        incy,
                                                 T* const       A[],
                                                 int64_t        lda,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_her2_batched, rocblas_float_complex, rocblas_cher2_batched);
MAP2CF_D64(rocblas_her2_batched, rocblas_double_complex, rocblas_zher2_batched);

// her2_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      T*             A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_A,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         T*             A,
                                                         int64_t        lda,
                                                         rocblas_stride stride_A,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_her2_strided_batched, rocblas_float_complex, rocblas_cher2_strided_batched);
MAP2CF_D64(rocblas_her2_strided_batched, rocblas_double_complex, rocblas_zher2_strided_batched);

// hpmv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpmv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpmv_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       A,
                                         int64_t        lda,
                                         const T*       x,
                                         rocblas_int    incx,
                                         const T*       beta,
                                         T*             y,
                                         int64_t        incy);

MAP2CF_D64(rocblas_hpmv, rocblas_float_complex, rocblas_chpmv);
MAP2CF_D64(rocblas_hpmv, rocblas_double_complex, rocblas_zhpmv);

// hpmv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpmv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpmv_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const A[],
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T*       beta,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_hpmv_batched, rocblas_float_complex, rocblas_chpmv_batched);
MAP2CF_D64(rocblas_hpmv_batched, rocblas_double_complex, rocblas_zhpmv_batched);

// hpmv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpmv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_stride stride_A,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpmv_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       A,
                                                         rocblas_stride stride_A,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       beta,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_hpmv_strided_batched, rocblas_float_complex, rocblas_chpmv_strided_batched);
MAP2CF_D64(rocblas_hpmv_strided_batched, rocblas_double_complex, rocblas_zhpmv_strided_batched);

// hpr
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr)(rocblas_handle   handle,
                                     rocblas_fill     uplo,
                                     rocblas_int      n,
                                     const real_t<T>* alpha,
                                     const T*         x,
                                     rocblas_int      incx,
                                     T*               AP);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr_64)(rocblas_handle   handle,
                                        rocblas_fill     uplo,
                                        int64_t          n,
                                        const real_t<T>* alpha,
                                        const T*         x,
                                        int64_t          incx,
                                        T*               AP);

MAP2CF_D64(rocblas_hpr, rocblas_float_complex, rocblas_chpr);
MAP2CF_D64(rocblas_hpr, rocblas_double_complex, rocblas_zhpr);

// hpr_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr_batched)(rocblas_handle   handle,
                                             rocblas_fill     uplo,
                                             rocblas_int      n,
                                             const real_t<T>* alpha,
                                             const T* const   x[],
                                             rocblas_int      incx,
                                             T* const         AP[],
                                             rocblas_int      batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr_batched_64)(rocblas_handle   handle,
                                                rocblas_fill     uplo,
                                                int64_t          n,
                                                const real_t<T>* alpha,
                                                const T* const   x[],
                                                int64_t          incx,
                                                T* const         AP[],
                                                int64_t          batch_count);

MAP2CF_D64(rocblas_hpr_batched, rocblas_float_complex, rocblas_chpr_batched);
MAP2CF_D64(rocblas_hpr_batched, rocblas_double_complex, rocblas_zhpr_batched);

// hpr_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr_strided_batched)(rocblas_handle   handle,
                                                     rocblas_fill     uplo,
                                                     rocblas_int      n,
                                                     const real_t<T>* alpha,
                                                     const T*         x,
                                                     rocblas_int      incx,
                                                     rocblas_stride   stride_x,
                                                     T*               AP,
                                                     rocblas_stride   stride_A,
                                                     rocblas_int      batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr_strided_batched_64)(rocblas_handle   handle,
                                                        rocblas_fill     uplo,
                                                        int64_t          n,
                                                        const real_t<T>* alpha,
                                                        const T*         x,
                                                        int64_t          incx,
                                                        rocblas_stride   stride_x,
                                                        T*               AP,
                                                        rocblas_stride   stride_A,
                                                        int64_t          batch_count);

MAP2CF_D64(rocblas_hpr_strided_batched, rocblas_float_complex, rocblas_chpr_strided_batched);
MAP2CF_D64(rocblas_hpr_strided_batched, rocblas_double_complex, rocblas_zhpr_strided_batched);

// hpr2
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr2)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       y,
                                      rocblas_int    incy,
                                      T*             AP);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr2_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       y,
                                         int64_t        incy,
                                         T*             AP);

MAP2CF_D64(rocblas_hpr2, rocblas_float_complex, rocblas_chpr2);
MAP2CF_D64(rocblas_hpr2, rocblas_double_complex, rocblas_zhpr2);

// hpr2_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr2_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T* const y[],
                                              rocblas_int    incy,
                                              T* const       AP[],
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr2_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T* const y[],
                                                 int64_t        incy,
                                                 T* const       AP[],
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_hpr2_batched, rocblas_float_complex, rocblas_chpr2_batched);
MAP2CF_D64(rocblas_hpr2_batched, rocblas_double_complex, rocblas_zhpr2_batched);

// hpr2_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr2_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      T*             AP,
                                                      rocblas_stride stride_A,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hpr2_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         T*             AP,
                                                         rocblas_stride stride_A,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_hpr2_strided_batched, rocblas_float_complex, rocblas_chpr2_strided_batched);
MAP2CF_D64(rocblas_hpr2_strided_batched, rocblas_double_complex, rocblas_zhpr2_strided_batched);

// trmv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       n,
                                      const T*          A,
                                      rocblas_int       lda,
                                      T*                x,
                                      rocblas_int       incx);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmv_64)(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         int64_t           n,
                                         const T*          A,
                                         int64_t           lda,
                                         T*                x,
                                         int64_t           incx);

MAP2CF_D64(rocblas_trmv, float, rocblas_strmv);
MAP2CF_D64(rocblas_trmv, double, rocblas_dtrmv);
MAP2CF_D64(rocblas_trmv, rocblas_float_complex, rocblas_ctrmv);
MAP2CF_D64(rocblas_trmv, rocblas_double_complex, rocblas_ztrmv);

// trmv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              const T* const*   A,
                                              rocblas_int       lda,
                                              T* const*         x,
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmv_batched_64)(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           n,
                                                 const T* const*   A,
                                                 int64_t           lda,
                                                 T* const*         x,
                                                 int64_t           incx,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_trmv_batched, float, rocblas_strmv_batched);
MAP2CF_D64(rocblas_trmv_batched, double, rocblas_dtrmv_batched);
MAP2CF_D64(rocblas_trmv_batched, rocblas_float_complex, rocblas_ctrmv_batched);
MAP2CF_D64(rocblas_trmv_batched, rocblas_double_complex, rocblas_ztrmv_batched);

// trmv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       n,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stridea,
                                                      T*                x,
                                                      rocblas_stride    stridex,
                                                      rocblas_int       incx,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmv_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         int64_t           n,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    stridea,
                                                         T*                x,
                                                         rocblas_stride    stridex,
                                                         int64_t           incx,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_trmv_strided_batched, float, rocblas_strmv_strided_batched);
MAP2CF_D64(rocblas_trmv_strided_batched, double, rocblas_dtrmv_strided_batched);
MAP2CF_D64(rocblas_trmv_strided_batched, rocblas_float_complex, rocblas_ctrmv_strided_batched);
MAP2CF_D64(rocblas_trmv_strided_batched, rocblas_double_complex, rocblas_ztrmv_strided_batched);

// tbmv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbmv)(rocblas_fill      uplo,
                                      rocblas_handle    handle,
                                      rocblas_diagonal  diag,
                                      rocblas_operation transA,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const T*          A,
                                      rocblas_int       lda,
                                      T*                x,
                                      rocblas_int       incx);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbmv_64)(rocblas_fill      uplo,
                                         rocblas_handle    handle,
                                         rocblas_diagonal  diag,
                                         rocblas_operation transA,
                                         int64_t           n,
                                         int64_t           k,
                                         const T*          A,
                                         int64_t           lda,
                                         T*                x,
                                         int64_t           incx);

MAP2CF_D64(rocblas_tbmv, float, rocblas_stbmv);
MAP2CF_D64(rocblas_tbmv, double, rocblas_dtbmv);
MAP2CF_D64(rocblas_tbmv, rocblas_float_complex, rocblas_ctbmv);
MAP2CF_D64(rocblas_tbmv, rocblas_double_complex, rocblas_ztbmv);

// tbmv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbmv_batched)(rocblas_fill      uplo,
                                              rocblas_handle    handle,
                                              rocblas_diagonal  diag,
                                              rocblas_operation transA,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              T* const          x[],
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbmv_batched_64)(rocblas_fill      uplo,
                                                 rocblas_handle    handle,
                                                 rocblas_diagonal  diag,
                                                 rocblas_operation transA,
                                                 int64_t           n,
                                                 int64_t           k,
                                                 const T* const    A[],
                                                 int64_t           lda,
                                                 T* const          x[],
                                                 int64_t           incx,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_tbmv_batched, float, rocblas_stbmv_batched);
MAP2CF_D64(rocblas_tbmv_batched, double, rocblas_dtbmv_batched);
MAP2CF_D64(rocblas_tbmv_batched, rocblas_float_complex, rocblas_ctbmv_batched);
MAP2CF_D64(rocblas_tbmv_batched, rocblas_double_complex, rocblas_ztbmv_batched);

// tbmv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbmv_strided_batched)(rocblas_fill      uplo,
                                                      rocblas_handle    handle,
                                                      rocblas_diagonal  diag,
                                                      rocblas_operation transA,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_A,
                                                      T*                x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbmv_strided_batched_64)(rocblas_fill      uplo,
                                                         rocblas_handle    handle,
                                                         rocblas_diagonal  diag,
                                                         rocblas_operation transA,
                                                         int64_t           n,
                                                         int64_t           k,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_A,
                                                         T*                x,
                                                         int64_t           incx,
                                                         rocblas_stride    stride_x,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_tbmv_strided_batched, float, rocblas_stbmv_strided_batched);
MAP2CF_D64(rocblas_tbmv_strided_batched, double, rocblas_dtbmv_strided_batched);
MAP2CF_D64(rocblas_tbmv_strided_batched, rocblas_float_complex, rocblas_ctbmv_strided_batched);
MAP2CF_D64(rocblas_tbmv_strided_batched, rocblas_double_complex, rocblas_ztbmv_strided_batched);

// tbsv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbsv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const T*          A,
                                      rocblas_int       lda,
                                      T*                x,
                                      rocblas_int       incx);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbsv_64)(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         int64_t           n,
                                         int64_t           k,
                                         const T*          A,
                                         int64_t           lda,
                                         T*                x,
                                         int64_t           incx);

MAP2CF_D64(rocblas_tbsv, float, rocblas_stbsv);
MAP2CF_D64(rocblas_tbsv, double, rocblas_dtbsv);
MAP2CF_D64(rocblas_tbsv, rocblas_float_complex, rocblas_ctbsv);
MAP2CF_D64(rocblas_tbsv, rocblas_double_complex, rocblas_ztbsv);

// tbsv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbsv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              T* const          x[],
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbsv_batched_64)(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           n,
                                                 int64_t           k,
                                                 const T* const    A[],
                                                 int64_t           lda,
                                                 T* const          x[],
                                                 int64_t           incx,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_tbsv_batched, float, rocblas_stbsv_batched);
MAP2CF_D64(rocblas_tbsv_batched, double, rocblas_dtbsv_batched);
MAP2CF_D64(rocblas_tbsv_batched, rocblas_float_complex, rocblas_ctbsv_batched);
MAP2CF_D64(rocblas_tbsv_batched, rocblas_double_complex, rocblas_ztbsv_batched);

// tbsv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbsv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      T*                x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tbsv_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         int64_t           n,
                                                         int64_t           k,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         T*                x,
                                                         int64_t           incx,
                                                         rocblas_stride    stride_x,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_tbsv_strided_batched, float, rocblas_stbsv_strided_batched);
MAP2CF_D64(rocblas_tbsv_strided_batched, double, rocblas_dtbsv_strided_batched);
MAP2CF_D64(rocblas_tbsv_strided_batched, rocblas_float_complex, rocblas_ctbsv_strided_batched);
MAP2CF_D64(rocblas_tbsv_strided_batched, rocblas_double_complex, rocblas_ztbsv_strided_batched);

// tpsv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpsv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       n,
                                      const T*          AP,
                                      T*                x,
                                      rocblas_int       incx);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpsv_64)(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         int64_t           n,
                                         const T*          AP,
                                         T*                x,
                                         int64_t           incx);

MAP2CF_D64(rocblas_tpsv, float, rocblas_stpsv);
MAP2CF_D64(rocblas_tpsv, double, rocblas_dtpsv);
MAP2CF_D64(rocblas_tpsv, rocblas_float_complex, rocblas_ctpsv);
MAP2CF_D64(rocblas_tpsv, rocblas_double_complex, rocblas_ztpsv);

// tpsv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpsv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       n,
                                              const T* const    AP[],
                                              T* const          x[],
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpsv_batched_64)(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           n,
                                                 const T* const    AP[],
                                                 T* const          x[],
                                                 int64_t           incx,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_tpsv_batched, float, rocblas_stpsv_batched);
MAP2CF_D64(rocblas_tpsv_batched, double, rocblas_dtpsv_batched);
MAP2CF_D64(rocblas_tpsv_batched, rocblas_float_complex, rocblas_ctpsv_batched);
MAP2CF_D64(rocblas_tpsv_batched, rocblas_double_complex, rocblas_ztpsv_batched);

// tpsv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpsv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       n,
                                                      const T*          AP,
                                                      rocblas_stride    stride_A,
                                                      T*                x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_tpsv_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         int64_t           n,
                                                         const T*          AP,
                                                         rocblas_stride    stride_A,
                                                         T*                x,
                                                         int64_t           incx,
                                                         rocblas_stride    stride_x,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_tpsv_strided_batched, float, rocblas_stpsv_strided_batched);
MAP2CF_D64(rocblas_tpsv_strided_batched, double, rocblas_dtpsv_strided_batched);
MAP2CF_D64(rocblas_tpsv_strided_batched, rocblas_float_complex, rocblas_ctpsv_strided_batched);
MAP2CF_D64(rocblas_tpsv_strided_batched, rocblas_double_complex, rocblas_ztpsv_strided_batched);

// trsv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsv)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      const T*          A,
                                      rocblas_int       lda,
                                      T*                x,
                                      rocblas_int       incx);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsv_64)(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         int64_t           m,
                                         const T*          A,
                                         int64_t           lda,
                                         T*                x,
                                         int64_t           incx);

MAP2CF_D64(rocblas_trsv, float, rocblas_strsv);
MAP2CF_D64(rocblas_trsv, double, rocblas_dtrsv);
MAP2CF_D64(rocblas_trsv, rocblas_float_complex, rocblas_ctrsv);
MAP2CF_D64(rocblas_trsv, rocblas_double_complex, rocblas_ztrsv);

// trsv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsv_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              T* const          x[],
                                              rocblas_int       incx,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsv_batched_64)(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           m,
                                                 const T* const    A[],
                                                 int64_t           lda,
                                                 T* const          x[],
                                                 int64_t           incx,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_trsv_batched, float, rocblas_strsv_batched);
MAP2CF_D64(rocblas_trsv_batched, double, rocblas_dtrsv_batched);
MAP2CF_D64(rocblas_trsv_batched, rocblas_float_complex, rocblas_ctrsv_batched);
MAP2CF_D64(rocblas_trsv_batched, rocblas_double_complex, rocblas_ztrsv_batched);

// trsv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsv_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_A,
                                                      T*                x,
                                                      rocblas_int       incx,
                                                      rocblas_stride    stride_x,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsv_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         int64_t           m,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_A,
                                                         T*                x,
                                                         int64_t           incx,
                                                         rocblas_stride    stride_x,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_trsv_strided_batched, float, rocblas_strsv_strided_batched);
MAP2CF_D64(rocblas_trsv_strided_batched, double, rocblas_dtrsv_strided_batched);
MAP2CF_D64(rocblas_trsv_strided_batched, rocblas_float_complex, rocblas_ctrsv_strided_batched);
MAP2CF_D64(rocblas_trsv_strided_batched, rocblas_double_complex, rocblas_ztrsv_strided_batched);

// spmv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spmv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spmv_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       A,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       beta,
                                         T*             y,
                                         int64_t        incy);

MAP2CF_D64(rocblas_spmv, float, rocblas_sspmv);
MAP2CF_D64(rocblas_spmv, double, rocblas_dspmv);

// spmv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spmv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spmv_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const A[],
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T*       beta,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_spmv_batched, float, rocblas_sspmv_batched);
MAP2CF_D64(rocblas_spmv_batched, double, rocblas_dspmv_batched);

// spmv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spmv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_stride stride_a,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_spmv_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       A,
                                                         rocblas_stride stride_a,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       beta,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_spmv_strided_batched, float, rocblas_sspmv_strided_batched);
MAP2CF_D64(rocblas_spmv_strided_batched, double, rocblas_dspmv_strided_batched);

// sbmv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_sbmv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      rocblas_int    k,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_sbmv_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         int64_t        k,
                                         const T*       alpha,
                                         const T*       A,
                                         int64_t        lda,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       beta,
                                         T*             y,
                                         int64_t        incy);

MAP2CF_D64(rocblas_sbmv, float, rocblas_ssbmv);
MAP2CF_D64(rocblas_sbmv, double, rocblas_dsbmv);

// sbmv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_sbmv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              rocblas_int    k,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_sbmv_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 int64_t        k,
                                                 const T*       alpha,
                                                 const T* const A[],
                                                 int64_t        lda,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T*       beta,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_sbmv_batched, float, rocblas_ssbmv_batched);
MAP2CF_D64(rocblas_sbmv_batched, double, rocblas_dsbmv_batched);

// sbmv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_sbmv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      rocblas_int    k,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_sbmv_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         int64_t        k,
                                                         const T*       alpha,
                                                         const T*       A,
                                                         int64_t        lda,
                                                         rocblas_stride stride_a,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       beta,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_sbmv_strided_batched, float, rocblas_ssbmv_strided_batched);
MAP2CF_D64(rocblas_sbmv_strided_batched, double, rocblas_dsbmv_strided_batched);

// symv
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symv)(rocblas_handle handle,
                                      rocblas_fill   uplo,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      const T*       beta,
                                      T*             y,
                                      rocblas_int    incy);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symv_64)(rocblas_handle handle,
                                         rocblas_fill   uplo,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       A,
                                         int64_t        lda,
                                         const T*       x,
                                         int64_t        incx,
                                         const T*       beta,
                                         T*             y,
                                         int64_t        incy);

MAP2CF_D64(rocblas_symv, float, rocblas_ssymv);
MAP2CF_D64(rocblas_symv, double, rocblas_dsymv);
MAP2CF_D64(rocblas_symv, rocblas_float_complex, rocblas_csymv);
MAP2CF_D64(rocblas_symv, rocblas_double_complex, rocblas_zsymv);

// symv_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symv_batched)(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              const T*       beta,
                                              T* const       y[],
                                              rocblas_int    incy,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symv_batched_64)(rocblas_handle handle,
                                                 rocblas_fill   uplo,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const A[],
                                                 int64_t        lda,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 const T*       beta,
                                                 T* const       y[],
                                                 int64_t        incy,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_symv_batched, float, rocblas_ssymv_batched);
MAP2CF_D64(rocblas_symv_batched, double, rocblas_dsymv_batched);
MAP2CF_D64(rocblas_symv_batched, rocblas_float_complex, rocblas_csymv_batched);
MAP2CF_D64(rocblas_symv_batched, rocblas_double_complex, rocblas_zsymv_batched);

// symv_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symv_strided_batched)(rocblas_handle handle,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    n,
                                                      const T*       alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      const T*       beta,
                                                      T*             y,
                                                      rocblas_int    incy,
                                                      rocblas_stride stride_y,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symv_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_fill   uplo,
                                                         int64_t        n,
                                                         const T*       alpha,
                                                         const T*       A,
                                                         int64_t        lda,
                                                         rocblas_stride stride_a,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         const T*       beta,
                                                         T*             y,
                                                         int64_t        incy,
                                                         rocblas_stride stride_y,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_symv_strided_batched, float, rocblas_ssymv_strided_batched);
MAP2CF_D64(rocblas_symv_strided_batched, double, rocblas_dsymv_strided_batched);
MAP2CF_D64(rocblas_symv_strided_batched, rocblas_float_complex, rocblas_csymv_strided_batched);
MAP2CF_D64(rocblas_symv_strided_batched, rocblas_double_complex, rocblas_zsymv_strided_batched);

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// dgmm
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dgmm)(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_int    m,
                                      rocblas_int    n,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       x,
                                      rocblas_int    incx,
                                      T*             C,
                                      rocblas_int    ldc);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dgmm_64)(rocblas_handle handle,
                                         rocblas_side   side,
                                         int64_t        m,
                                         int64_t        n,
                                         const T*       A,
                                         int64_t        lda,
                                         const T*       x,
                                         int64_t        incx,
                                         T*             C,
                                         int64_t        ldc);

MAP2CF_D64(rocblas_dgmm, float, rocblas_sdgmm);
MAP2CF_D64(rocblas_dgmm, double, rocblas_ddgmm);
MAP2CF_D64(rocblas_dgmm, rocblas_float_complex, rocblas_cdgmm);
MAP2CF_D64(rocblas_dgmm, rocblas_double_complex, rocblas_zdgmm);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dgmm_batched)(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const x[],
                                              rocblas_int    incx,
                                              T* const       C[],
                                              rocblas_int    ldc,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dgmm_batched_64)(rocblas_handle handle,
                                                 rocblas_side   side,
                                                 int64_t        m,
                                                 int64_t        n,
                                                 const T* const A[],
                                                 int64_t        lda,
                                                 const T* const x[],
                                                 int64_t        incx,
                                                 T* const       C[],
                                                 int64_t        ldc,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_dgmm_batched, float, rocblas_sdgmm_batched);
MAP2CF_D64(rocblas_dgmm_batched, double, rocblas_ddgmm_batched);
MAP2CF_D64(rocblas_dgmm_batched, rocblas_float_complex, rocblas_cdgmm_batched);
MAP2CF_D64(rocblas_dgmm_batched, rocblas_double_complex, rocblas_zdgmm_batched);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dgmm_strided_batched)(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       x,
                                                      rocblas_int    incx,
                                                      rocblas_stride stride_x,
                                                      T*             C,
                                                      rocblas_int    ldc,
                                                      rocblas_stride stride_c,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_dgmm_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_side   side,
                                                         int64_t        m,
                                                         int64_t        n,
                                                         const T*       A,
                                                         int64_t        lda,
                                                         rocblas_stride stride_a,
                                                         const T*       x,
                                                         int64_t        incx,
                                                         rocblas_stride stride_x,
                                                         T*             C,
                                                         int64_t        ldc,
                                                         rocblas_stride stride_c,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_dgmm_strided_batched, float, rocblas_sdgmm_strided_batched);
MAP2CF_D64(rocblas_dgmm_strided_batched, double, rocblas_ddgmm_strided_batched);
MAP2CF_D64(rocblas_dgmm_strided_batched, rocblas_float_complex, rocblas_cdgmm_strided_batched);
MAP2CF_D64(rocblas_dgmm_strided_batched, rocblas_double_complex, rocblas_zdgmm_strided_batched);

// geam
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_geam)(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          beta,
                                      const T*          B,
                                      rocblas_int       ldb,
                                      T*                C,
                                      rocblas_int       ldc);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_geam_64)(rocblas_handle    handle,
                                         rocblas_operation transA,
                                         rocblas_operation transB,
                                         int64_t           m,
                                         int64_t           n,
                                         const T*          alpha,
                                         const T*          A,
                                         int64_t           lda,
                                         const T*          beta,
                                         const T*          B,
                                         int64_t           ldb,
                                         T*                C,
                                         int64_t           ldc);

MAP2CF_D64(rocblas_geam, float, rocblas_sgeam);
MAP2CF_D64(rocblas_geam, double, rocblas_dgeam);
MAP2CF_D64(rocblas_geam, rocblas_float_complex, rocblas_cgeam);
MAP2CF_D64(rocblas_geam, rocblas_double_complex, rocblas_zgeam);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_geam_batched)(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T*          beta,
                                              const T* const    B[],
                                              rocblas_int       ldb,
                                              T* const          C[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_geam_batched_64)(rocblas_handle    handle,
                                                 rocblas_operation transA,
                                                 rocblas_operation transB,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 const T*          alpha,
                                                 const T* const    A[],
                                                 int64_t           lda,
                                                 const T*          beta,
                                                 const T* const    B[],
                                                 int64_t           ldb,
                                                 T* const          C[],
                                                 int64_t           ldc,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_geam_batched, float, rocblas_sgeam_batched);
MAP2CF_D64(rocblas_geam_batched, double, rocblas_dgeam_batched);
MAP2CF_D64(rocblas_geam_batched, rocblas_float_complex, rocblas_cgeam_batched);
MAP2CF_D64(rocblas_geam_batched, rocblas_double_complex, rocblas_zgeam_batched);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_geam_strided_batched)(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const T*          beta,
                                                      const T*          B,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      T*                C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_geam_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_operation transA,
                                                         rocblas_operation transB,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         const T*          alpha,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         const T*          beta,
                                                         const T*          B,
                                                         rocblas_int       ldb,
                                                         int64_t           stride_b,
                                                         T*                C,
                                                         int64_t           ldc,
                                                         rocblas_stride    stride_c,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_geam_strided_batched, float, rocblas_sgeam_strided_batched);
MAP2CF_D64(rocblas_geam_strided_batched, double, rocblas_dgeam_strided_batched);
MAP2CF_D64(rocblas_geam_strided_batched, rocblas_float_complex, rocblas_cgeam_strided_batched);
MAP2CF_D64(rocblas_geam_strided_batched, rocblas_double_complex, rocblas_zgeam_strided_batched);

// gemm
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemm)(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      rocblas_operation transB,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          B,
                                      rocblas_int       ldb,
                                      const T*          beta,
                                      T*                C,
                                      rocblas_int       ldc);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemm_64)(rocblas_handle    handle,
                                         rocblas_operation transA,
                                         rocblas_operation transB,
                                         int64_t           m,
                                         int64_t           n,
                                         int64_t           k,
                                         const T*          alpha,
                                         const T*          A,
                                         int64_t           lda,
                                         const T*          B,
                                         int64_t           ldb,
                                         const T*          beta,
                                         T*                C,
                                         int64_t           ldc);

MAP2CF_D64(rocblas_gemm, float, rocblas_sgemm);
MAP2CF_D64(rocblas_gemm, double, rocblas_dgemm);
MAP2CF_D64(rocblas_gemm, rocblas_half, rocblas_hgemm);
MAP2CF_D64(rocblas_gemm, rocblas_float_complex, rocblas_cgemm);
MAP2CF_D64(rocblas_gemm, rocblas_double_complex, rocblas_zgemm);

// gemm_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemm_batched)(rocblas_handle    handle,
                                              rocblas_operation transA,
                                              rocblas_operation transB,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const T*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T* const    B[],
                                              rocblas_int       ldb,
                                              const T*          beta,
                                              T* const          C[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemm_batched_64)(rocblas_handle    handle,
                                                 rocblas_operation transA,
                                                 rocblas_operation transB,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 int64_t           k,
                                                 const T*          alpha,
                                                 const T* const    A[],
                                                 int64_t           lda,
                                                 const T* const    B[],
                                                 int64_t           ldb,
                                                 const T*          beta,
                                                 T* const          C[],
                                                 int64_t           ldc,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_gemm_batched, float, rocblas_sgemm_batched);
MAP2CF_D64(rocblas_gemm_batched, double, rocblas_dgemm_batched);
MAP2CF_D64(rocblas_gemm_batched, rocblas_half, rocblas_hgemm_batched);
MAP2CF_D64(rocblas_gemm_batched, rocblas_float_complex, rocblas_cgemm_batched);
MAP2CF_D64(rocblas_gemm_batched, rocblas_double_complex, rocblas_zgemm_batched);

// gemm_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemm_strided_batched)(rocblas_handle    handle,
                                                      rocblas_operation transA,
                                                      rocblas_operation transB,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const T*          alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    bsa,
                                                      const T*          B,
                                                      rocblas_int       ldb,
                                                      rocblas_int       bsb,
                                                      const T*          beta,
                                                      T*                C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    bsc,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemm_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_operation transA,
                                                         rocblas_operation transB,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         int64_t           k,
                                                         const T*          alpha,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    bsa,
                                                         const T*          B,
                                                         int64_t           ldb,
                                                         int64_t           bsb,
                                                         const T*          beta,
                                                         T*                C,
                                                         int64_t           ldc,
                                                         rocblas_stride    bsc,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_gemm_strided_batched, float, rocblas_sgemm_strided_batched);
MAP2CF_D64(rocblas_gemm_strided_batched, double, rocblas_dgemm_strided_batched);
MAP2CF_D64(rocblas_gemm_strided_batched, rocblas_half, rocblas_hgemm_strided_batched);
MAP2CF_D64(rocblas_gemm_strided_batched, rocblas_float_complex, rocblas_cgemm_strided_batched);
MAP2CF_D64(rocblas_gemm_strided_batched, rocblas_double_complex, rocblas_zgemm_strided_batched);

// gemmt
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemmt)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_operation transB,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const T*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

// gemmt_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemmt_64)(rocblas_handle    handle,
                                          rocblas_fill      uplo,
                                          rocblas_operation transA,
                                          rocblas_operation transB,
                                          int64_t           n,
                                          int64_t           k,
                                          const T*          alpha,
                                          const T*          A,
                                          int64_t           lda,
                                          const T*          B,
                                          int64_t           ldb,
                                          const T*          beta,
                                          T*                C,
                                          int64_t           ldc);

MAP2CF_D64(rocblas_gemmt, float, rocblas_sgemmt);
MAP2CF_D64(rocblas_gemmt, double, rocblas_dgemmt);
MAP2CF_D64(rocblas_gemmt, rocblas_float_complex, rocblas_cgemmt);
MAP2CF_D64(rocblas_gemmt, rocblas_double_complex, rocblas_zgemmt);

// gemmt_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemmt_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_operation transB,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T*          alpha,
                                               const T* const    A[],
                                               rocblas_int       lda,
                                               const T* const    B[],
                                               rocblas_int       ldb,
                                               const T*          beta,
                                               T* const          C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

// gemmt_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemmt_batched_64)(rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation transA,
                                                  rocblas_operation transB,
                                                  int64_t           n,
                                                  int64_t           k,
                                                  const T*          alpha,
                                                  const T* const    A[],
                                                  int64_t           lda,
                                                  const T* const    B[],
                                                  int64_t           ldb,
                                                  const T*          beta,
                                                  T* const          C[],
                                                  int64_t           ldc,
                                                  int64_t           batch_count);

MAP2CF_D64(rocblas_gemmt_batched, float, rocblas_sgemmt_batched);
MAP2CF_D64(rocblas_gemmt_batched, double, rocblas_dgemmt_batched);
MAP2CF_D64(rocblas_gemmt_batched, rocblas_float_complex, rocblas_cgemmt_batched);
MAP2CF_D64(rocblas_gemmt_batched, rocblas_double_complex, rocblas_zgemmt_batched);

// gemmt_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemmt_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_operation transB,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T*          alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    bsa,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_int       bsb,
                                                       const T*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    bsc,
                                                       rocblas_int       batch_count);

// gemmt_strided_batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_gemmt_strided_batched_64)(rocblas_handle    handle,
                                                          rocblas_fill      uplo,
                                                          rocblas_operation transA,
                                                          rocblas_operation transB,
                                                          int64_t           n,
                                                          int64_t           k,
                                                          const T*          alpha,
                                                          const T*          A,
                                                          rocblas_int       lda,
                                                          rocblas_stride    bsa,
                                                          const T*          B,
                                                          int64_t           ldb,
                                                          int64_t           bsb,
                                                          const T*          beta,
                                                          T*                C,
                                                          int64_t           ldc,
                                                          rocblas_stride    bsc,
                                                          int64_t           batch_count);

MAP2CF_D64(rocblas_gemmt_strided_batched, float, rocblas_sgemmt_strided_batched);
MAP2CF_D64(rocblas_gemmt_strided_batched, double, rocblas_dgemmt_strided_batched);
MAP2CF_D64(rocblas_gemmt_strided_batched, rocblas_float_complex, rocblas_cgemmt_strided_batched);
MAP2CF_D64(rocblas_gemmt_strided_batched, rocblas_double_complex, rocblas_zgemmt_strided_batched);

// hemm
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemm)(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      rocblas_int    m,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       B,
                                      rocblas_int    ldb,
                                      const T*       beta,
                                      T*             C,
                                      rocblas_int    ldc);

//hemm_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemm_64)(rocblas_handle handle,
                                         rocblas_side   side,
                                         rocblas_fill   uplo,
                                         int64_t        m,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       A,
                                         int64_t        lda,
                                         const T*       B,
                                         int64_t        ldb,
                                         const T*       beta,
                                         T*             C,
                                         int64_t        ldc);

MAP2CF_D64(rocblas_hemm, rocblas_float_complex, rocblas_chemm);
MAP2CF_D64(rocblas_hemm, rocblas_double_complex, rocblas_zhemm);

// hemm batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemm_batched)(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_fill   uplo,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const B[],
                                              rocblas_int    ldb,
                                              const T*       beta,
                                              T* const       C[],
                                              rocblas_int    ldc,
                                              rocblas_int    batch_count);
// hemm batched_64
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemm_batched_64)(rocblas_handle handle,
                                                 rocblas_side   side,
                                                 rocblas_fill   uplo,
                                                 int64_t        m,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const A[],
                                                 int64_t        lda,
                                                 const T* const B[],
                                                 int64_t        ldb,
                                                 const T*       beta,
                                                 T* const       C[],
                                                 int64_t        ldc,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_hemm_batched, rocblas_float_complex, rocblas_chemm_batched);
MAP2CF_D64(rocblas_hemm_batched, rocblas_double_complex, rocblas_zhemm_batched);

// hemm strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemm_strided_batched)(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      const T* const alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       B,
                                                      rocblas_int    ldb,
                                                      rocblas_stride stride_b,
                                                      const T* const beta,
                                                      T*             C,
                                                      rocblas_int    ldc,
                                                      rocblas_stride stride_c,
                                                      rocblas_int    batch_count);

// hemm strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_hemm_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_side   side,
                                                         rocblas_fill   uplo,
                                                         int64_t        m,
                                                         int64_t        n,
                                                         const T* const alpha,
                                                         const T*       A,
                                                         int64_t        lda,
                                                         rocblas_stride stride_a,
                                                         const T*       B,
                                                         int64_t        ldb,
                                                         rocblas_stride stride_b,
                                                         const T* const beta,
                                                         T*             C,
                                                         int64_t        ldc,
                                                         rocblas_stride stride_c,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_hemm_strided_batched, rocblas_float_complex, rocblas_chemm_strided_batched);
MAP2CF_D64(rocblas_hemm_strided_batched, rocblas_double_complex, rocblas_zhemm_strided_batched);

// herk
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herk)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const U*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const U*          beta,
                                      T*                C,
                                      rocblas_int       ldc);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herk_64)(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         int64_t           n,
                                         int64_t           k,
                                         const U*          alpha,
                                         const T*          A,
                                         int64_t           lda,
                                         const U*          beta,
                                         T*                C,
                                         int64_t           ldc);

MAP2CF_D64(rocblas_herk, rocblas_float_complex, float, rocblas_cherk);
MAP2CF_D64(rocblas_herk, rocblas_double_complex, double, rocblas_zherk);

// herk batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herk_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const U*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const U*          beta,
                                              T* const          C[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herk_batched_64)(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 int64_t           n,
                                                 int64_t           k,
                                                 const U*          alpha,
                                                 const T* const    A[],
                                                 int64_t           lda,
                                                 const U*          beta,
                                                 T* const          C[],
                                                 int64_t           ldc,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_herk_batched, rocblas_float_complex, float, rocblas_cherk_batched);
MAP2CF_D64(rocblas_herk_batched, rocblas_double_complex, double, rocblas_zherk_batched);

// herk strided batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herk_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const U* const    alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const U*          beta,
                                                      T*                C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herk_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         int64_t           n,
                                                         int64_t           k,
                                                         const U* const    alpha,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         const U*          beta,
                                                         T*                C,
                                                         int64_t           ldc,
                                                         rocblas_stride    stride_c,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_herk_strided_batched,
           rocblas_float_complex,
           float,
           rocblas_cherk_strided_batched);
MAP2CF_D64(rocblas_herk_strided_batched,
           rocblas_double_complex,
           double,
           rocblas_zherk_strided_batched);

// her2k
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2k)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const U*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2k_64)(rocblas_handle    handle,
                                          rocblas_fill      uplo,
                                          rocblas_operation transA,
                                          int64_t           n,
                                          int64_t           k,
                                          const T*          alpha,
                                          const T*          A,
                                          int64_t           lda,
                                          const T*          B,
                                          int64_t           ldb,
                                          const U*          beta,
                                          T*                C,
                                          int64_t           ldc);

MAP2CF_D64(rocblas_her2k, rocblas_float_complex, float, rocblas_cher2k);
MAP2CF_D64(rocblas_her2k, rocblas_double_complex, double, rocblas_zher2k);

// her2k batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2k_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T*          alpha,
                                               const T* const    A[],
                                               rocblas_int       lda,
                                               const T* const    B[],
                                               rocblas_int       ldb,
                                               const U*          beta,
                                               T* const          C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2k_batched_64)(rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation transA,
                                                  int64_t           n,
                                                  int64_t           k,
                                                  const T*          alpha,
                                                  const T* const    A[],
                                                  int64_t           lda,
                                                  const T* const    B[],
                                                  int64_t           ldb,
                                                  const U*          beta,
                                                  T* const          C[],
                                                  int64_t           ldc,
                                                  int64_t           batch_count);

MAP2CF_D64(rocblas_her2k_batched, rocblas_float_complex, float, rocblas_cher2k_batched);
MAP2CF_D64(rocblas_her2k_batched, rocblas_double_complex, double, rocblas_zher2k_batched);

// her2k strided batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2k_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T* const    alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_b,
                                                       const U*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_her2k_strided_batched_64)(rocblas_handle    handle,
                                                          rocblas_fill      uplo,
                                                          rocblas_operation transA,
                                                          int64_t           n,
                                                          int64_t           k,
                                                          const T* const    alpha,
                                                          const T*          A,
                                                          int64_t           lda,
                                                          rocblas_stride    stride_a,
                                                          const T*          B,
                                                          int64_t           ldb,
                                                          rocblas_stride    stride_b,
                                                          const U*          beta,
                                                          T*                C,
                                                          int64_t           ldc,
                                                          rocblas_stride    stride_c,
                                                          int64_t           batch_count);

MAP2CF_D64(rocblas_her2k_strided_batched,
           rocblas_float_complex,
           float,
           rocblas_cher2k_strided_batched);
MAP2CF_D64(rocblas_her2k_strided_batched,
           rocblas_double_complex,
           double,
           rocblas_zher2k_strided_batched);

// herkx
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herkx)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const U*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herkx_64)(rocblas_handle    handle,
                                          rocblas_fill      uplo,
                                          rocblas_operation transA,
                                          int64_t           n,
                                          int64_t           k,
                                          const T*          alpha,
                                          const T*          A,
                                          int64_t           lda,
                                          const T*          B,
                                          int64_t           ldb,
                                          const U*          beta,
                                          T*                C,
                                          int64_t           ldc);

MAP2CF_D64(rocblas_herkx, rocblas_float_complex, float, rocblas_cherkx);
MAP2CF_D64(rocblas_herkx, rocblas_double_complex, double, rocblas_zherkx);

// herkx batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herkx_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T*          alpha,
                                               const T* const    A[],
                                               rocblas_int       lda,
                                               const T* const    B[],
                                               rocblas_int       ldb,
                                               const U*          beta,
                                               T* const          C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herkx_batched_64)(rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation transA,
                                                  int64_t           n,
                                                  int64_t           k,
                                                  const T*          alpha,
                                                  const T* const    A[],
                                                  int64_t           lda,
                                                  const T* const    B[],
                                                  int64_t           ldb,
                                                  const U*          beta,
                                                  T* const          C[],
                                                  int64_t           ldc,
                                                  int64_t           batch_count);

MAP2CF_D64(rocblas_herkx_batched, rocblas_float_complex, float, rocblas_cherkx_batched);
MAP2CF_D64(rocblas_herkx_batched, rocblas_double_complex, double, rocblas_zherkx_batched);

// herkx strided batched
template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herkx_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T* const    alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_b,
                                                       const U*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count);

template <typename T, typename U = real_t<T>, bool FORTRAN = false>
inline rocblas_status (*rocblas_herkx_strided_batched_64)(rocblas_handle    handle,
                                                          rocblas_fill      uplo,
                                                          rocblas_operation transA,
                                                          int64_t           n,
                                                          int64_t           k,
                                                          const T* const    alpha,
                                                          const T*          A,
                                                          int64_t           lda,
                                                          rocblas_stride    stride_a,
                                                          const T*          B,
                                                          int64_t           ldb,
                                                          rocblas_stride    stride_b,
                                                          const U*          beta,
                                                          T*                C,
                                                          int64_t           ldc,
                                                          rocblas_stride    stride_c,
                                                          int64_t           batch_count);

MAP2CF_D64(rocblas_herkx_strided_batched,
           rocblas_float_complex,
           float,
           rocblas_cherkx_strided_batched);
MAP2CF_D64(rocblas_herkx_strided_batched,
           rocblas_double_complex,
           double,
           rocblas_zherkx_strided_batched);

// symm
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symm)(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      rocblas_int    m,
                                      rocblas_int    n,
                                      const T*       alpha,
                                      const T*       A,
                                      rocblas_int    lda,
                                      const T*       B,
                                      rocblas_int    ldb,
                                      const T*       beta,
                                      T*             C,
                                      rocblas_int    ldc);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symm_64)(rocblas_handle handle,
                                         rocblas_side   side,
                                         rocblas_fill   uplo,
                                         int64_t        m,
                                         int64_t        n,
                                         const T*       alpha,
                                         const T*       A,
                                         int64_t        lda,
                                         const T*       B,
                                         int64_t        ldb,
                                         const T*       beta,
                                         T*             C,
                                         int64_t        ldc);

MAP2CF_D64(rocblas_symm, float, rocblas_ssymm);
MAP2CF_D64(rocblas_symm, double, rocblas_dsymm);
MAP2CF_D64(rocblas_symm, rocblas_float_complex, rocblas_csymm);
MAP2CF_D64(rocblas_symm, rocblas_double_complex, rocblas_zsymm);

// symm batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symm_batched)(rocblas_handle handle,
                                              rocblas_side   side,
                                              rocblas_fill   uplo,
                                              rocblas_int    m,
                                              rocblas_int    n,
                                              const T*       alpha,
                                              const T* const A[],
                                              rocblas_int    lda,
                                              const T* const B[],
                                              rocblas_int    ldb,
                                              const T*       beta,
                                              T* const       C[],
                                              rocblas_int    ldc,
                                              rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symm_batched_64)(rocblas_handle handle,
                                                 rocblas_side   side,
                                                 rocblas_fill   uplo,
                                                 int64_t        m,
                                                 int64_t        n,
                                                 const T*       alpha,
                                                 const T* const A[],
                                                 int64_t        lda,
                                                 const T* const B[],
                                                 int64_t        ldb,
                                                 const T*       beta,
                                                 T* const       C[],
                                                 int64_t        ldc,
                                                 int64_t        batch_count);

MAP2CF_D64(rocblas_symm_batched, float, rocblas_ssymm_batched);
MAP2CF_D64(rocblas_symm_batched, double, rocblas_dsymm_batched);
MAP2CF_D64(rocblas_symm_batched, rocblas_float_complex, rocblas_csymm_batched);
MAP2CF_D64(rocblas_symm_batched, rocblas_double_complex, rocblas_zsymm_batched);

// symm strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symm_strided_batched)(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_fill   uplo,
                                                      rocblas_int    m,
                                                      rocblas_int    n,
                                                      const T* const alpha,
                                                      const T*       A,
                                                      rocblas_int    lda,
                                                      rocblas_stride stride_a,
                                                      const T*       B,
                                                      rocblas_int    ldb,
                                                      rocblas_stride stride_b,
                                                      const T* const beta,
                                                      T*             C,
                                                      rocblas_int    ldc,
                                                      rocblas_stride stride_c,
                                                      rocblas_int    batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_symm_strided_batched_64)(rocblas_handle handle,
                                                         rocblas_side   side,
                                                         rocblas_fill   uplo,
                                                         int64_t        m,
                                                         int64_t        n,
                                                         const T* const alpha,
                                                         const T*       A,
                                                         int64_t        lda,
                                                         rocblas_stride stride_a,
                                                         const T*       B,
                                                         int64_t        ldb,
                                                         rocblas_stride stride_b,
                                                         const T* const beta,
                                                         T*             C,
                                                         int64_t        ldc,
                                                         rocblas_stride stride_c,
                                                         int64_t        batch_count);

MAP2CF_D64(rocblas_symm_strided_batched, float, rocblas_ssymm_strided_batched);
MAP2CF_D64(rocblas_symm_strided_batched, double, rocblas_dsymm_strided_batched);
MAP2CF_D64(rocblas_symm_strided_batched, rocblas_float_complex, rocblas_csymm_strided_batched);
MAP2CF_D64(rocblas_symm_strided_batched, rocblas_double_complex, rocblas_zsymm_strided_batched);

// syrk
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrk)(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_int       n,
                                      rocblas_int       k,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          beta,
                                      T*                C,
                                      rocblas_int       ldc);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrk_64)(rocblas_handle    handle,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         int64_t           n,
                                         int64_t           k,
                                         const T*          alpha,
                                         const T*          A,
                                         int64_t           lda,
                                         const T*          beta,
                                         T*                C,
                                         int64_t           ldc);

MAP2CF_D64(rocblas_syrk, float, rocblas_ssyrk);
MAP2CF_D64(rocblas_syrk, double, rocblas_dsyrk);
MAP2CF_D64(rocblas_syrk, rocblas_float_complex, rocblas_csyrk);
MAP2CF_D64(rocblas_syrk, rocblas_double_complex, rocblas_zsyrk);

// syrk batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrk_batched)(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_int       n,
                                              rocblas_int       k,
                                              const T*          alpha,
                                              const T* const    A[],
                                              rocblas_int       lda,
                                              const T*          beta,
                                              T* const          C[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrk_batched_64)(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 int64_t           n,
                                                 int64_t           k,
                                                 const T*          alpha,
                                                 const T* const    A[],
                                                 int64_t           lda,
                                                 const T*          beta,
                                                 T* const          C[],
                                                 int64_t           ldc,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_syrk_batched, float, rocblas_ssyrk_batched);
MAP2CF_D64(rocblas_syrk_batched, double, rocblas_dsyrk_batched);
MAP2CF_D64(rocblas_syrk_batched, rocblas_float_complex, rocblas_csyrk_batched);
MAP2CF_D64(rocblas_syrk_batched, rocblas_double_complex, rocblas_zsyrk_batched);

// syrk strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrk_strided_batched)(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_int       n,
                                                      rocblas_int       k,
                                                      const T* const    alpha,
                                                      const T*          A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const T*          beta,
                                                      T*                C,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrk_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         int64_t           n,
                                                         int64_t           k,
                                                         const T* const    alpha,
                                                         const T*          A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         const T*          beta,
                                                         T*                C,
                                                         int64_t           ldc,
                                                         rocblas_stride    stride_c,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_syrk_strided_batched, float, rocblas_ssyrk_strided_batched);
MAP2CF_D64(rocblas_syrk_strided_batched, double, rocblas_dsyrk_strided_batched);
MAP2CF_D64(rocblas_syrk_strided_batched, rocblas_float_complex, rocblas_csyrk_strided_batched);
MAP2CF_D64(rocblas_syrk_strided_batched, rocblas_double_complex, rocblas_zsyrk_strided_batched);

// syr2k
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2k)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const T*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2k_64)(rocblas_handle    handle,
                                          rocblas_fill      uplo,
                                          rocblas_operation transA,
                                          int64_t           n,
                                          int64_t           k,
                                          const T*          alpha,
                                          const T*          A,
                                          int64_t           lda,
                                          const T*          B,
                                          int64_t           ldb,
                                          const T*          beta,
                                          T*                C,
                                          int64_t           ldc);

MAP2CF_D64(rocblas_syr2k, float, rocblas_ssyr2k);
MAP2CF_D64(rocblas_syr2k, double, rocblas_dsyr2k);
MAP2CF_D64(rocblas_syr2k, rocblas_float_complex, rocblas_csyr2k);
MAP2CF_D64(rocblas_syr2k, rocblas_double_complex, rocblas_zsyr2k);

// syr2k batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2k_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T*          alpha,
                                               const T* const    A[],
                                               rocblas_int       lda,
                                               const T* const    B[],
                                               rocblas_int       ldb,
                                               const T*          beta,
                                               T* const          C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2k_batched_64)(rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation transA,
                                                  int64_t           n,
                                                  int64_t           k,
                                                  const T*          alpha,
                                                  const T* const    A[],
                                                  int64_t           lda,
                                                  const T* const    B[],
                                                  int64_t           ldb,
                                                  const T*          beta,
                                                  T* const          C[],
                                                  int64_t           ldc,
                                                  int64_t           batch_count);

MAP2CF_D64(rocblas_syr2k_batched, float, rocblas_ssyr2k_batched);
MAP2CF_D64(rocblas_syr2k_batched, double, rocblas_dsyr2k_batched);
MAP2CF_D64(rocblas_syr2k_batched, rocblas_float_complex, rocblas_csyr2k_batched);
MAP2CF_D64(rocblas_syr2k_batched, rocblas_double_complex, rocblas_zsyr2k_batched);

// syr2k strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2k_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T* const    alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_b,
                                                       const T*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syr2k_strided_batched_64)(rocblas_handle    handle,
                                                          rocblas_fill      uplo,
                                                          rocblas_operation transA,
                                                          int64_t           n,
                                                          int64_t           k,
                                                          const T* const    alpha,
                                                          const T*          A,
                                                          int64_t           lda,
                                                          rocblas_stride    stride_a,
                                                          const T*          B,
                                                          int64_t           ldb,
                                                          rocblas_stride    stride_b,
                                                          const T*          beta,
                                                          T*                C,
                                                          int64_t           ldc,
                                                          rocblas_stride    stride_c,
                                                          int64_t           batch_count);

MAP2CF_D64(rocblas_syr2k_strided_batched, float, rocblas_ssyr2k_strided_batched);
MAP2CF_D64(rocblas_syr2k_strided_batched, double, rocblas_dsyr2k_strided_batched);
MAP2CF_D64(rocblas_syr2k_strided_batched, rocblas_float_complex, rocblas_csyr2k_strided_batched);
MAP2CF_D64(rocblas_syr2k_strided_batched, rocblas_double_complex, rocblas_zsyr2k_strided_batched);

// syrkx
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrkx)(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation transA,
                                       rocblas_int       n,
                                       rocblas_int       k,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_int       lda,
                                       const T*          B,
                                       rocblas_int       ldb,
                                       const T*          beta,
                                       T*                C,
                                       rocblas_int       ldc);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrkx_64)(rocblas_handle    handle,
                                          rocblas_fill      uplo,
                                          rocblas_operation transA,
                                          int64_t           n,
                                          int64_t           k,
                                          const T*          alpha,
                                          const T*          A,
                                          int64_t           lda,
                                          const T*          B,
                                          int64_t           ldb,
                                          const T*          beta,
                                          T*                C,
                                          int64_t           ldc);

MAP2CF_D64(rocblas_syrkx, float, rocblas_ssyrkx);
MAP2CF_D64(rocblas_syrkx, double, rocblas_dsyrkx);
MAP2CF_D64(rocblas_syrkx, rocblas_float_complex, rocblas_csyrkx);
MAP2CF_D64(rocblas_syrkx, rocblas_double_complex, rocblas_zsyrkx);

// syrkx batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrkx_batched)(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation transA,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T*          alpha,
                                               const T* const    A[],
                                               rocblas_int       lda,
                                               const T* const    B[],
                                               rocblas_int       ldb,
                                               const T*          beta,
                                               T* const          C[],
                                               rocblas_int       ldc,
                                               rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrkx_batched_64)(rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation transA,
                                                  int64_t           n,
                                                  int64_t           k,
                                                  const T*          alpha,
                                                  const T* const    A[],
                                                  int64_t           lda,
                                                  const T* const    B[],
                                                  int64_t           ldb,
                                                  const T*          beta,
                                                  T* const          C[],
                                                  int64_t           ldc,
                                                  int64_t           batch_count);

MAP2CF_D64(rocblas_syrkx_batched, float, rocblas_ssyrkx_batched);
MAP2CF_D64(rocblas_syrkx_batched, double, rocblas_dsyrkx_batched);
MAP2CF_D64(rocblas_syrkx_batched, rocblas_float_complex, rocblas_csyrkx_batched);
MAP2CF_D64(rocblas_syrkx_batched, rocblas_double_complex, rocblas_zsyrkx_batched);

// syrkx strided batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrkx_strided_batched)(rocblas_handle    handle,
                                                       rocblas_fill      uplo,
                                                       rocblas_operation transA,
                                                       rocblas_int       n,
                                                       rocblas_int       k,
                                                       const T* const    alpha,
                                                       const T*          A,
                                                       rocblas_int       lda,
                                                       rocblas_stride    stride_a,
                                                       const T*          B,
                                                       rocblas_int       ldb,
                                                       rocblas_stride    stride_b,
                                                       const T*          beta,
                                                       T*                C,
                                                       rocblas_int       ldc,
                                                       rocblas_stride    stride_c,
                                                       rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_syrkx_strided_batched_64)(rocblas_handle    handle,
                                                          rocblas_fill      uplo,
                                                          rocblas_operation transA,
                                                          int64_t           n,
                                                          int64_t           k,
                                                          const T* const    alpha,
                                                          const T*          A,
                                                          int64_t           lda,
                                                          rocblas_stride    stride_a,
                                                          const T*          B,
                                                          int64_t           ldb,
                                                          rocblas_stride    stride_b,
                                                          const T*          beta,
                                                          T*                C,
                                                          int64_t           ldc,
                                                          rocblas_stride    stride_c,
                                                          int64_t           batch_count);

MAP2CF_D64(rocblas_syrkx_strided_batched, float, rocblas_ssyrkx_strided_batched);
MAP2CF_D64(rocblas_syrkx_strided_batched, double, rocblas_dsyrkx_strided_batched);
MAP2CF_D64(rocblas_syrkx_strided_batched, rocblas_float_complex, rocblas_csyrkx_strided_batched);
MAP2CF_D64(rocblas_syrkx_strided_batched, rocblas_double_complex, rocblas_zsyrkx_strided_batched);

// trmm
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmm)(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_int       lda,
                                      const T*          B,
                                      rocblas_int       ldb,
                                      T*                C,
                                      rocblas_int       ldc);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmm_64)(rocblas_handle    handle,
                                         rocblas_side      side,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         int64_t           m,
                                         int64_t           n,
                                         const T*          alpha,
                                         const T*          A,
                                         int64_t           lda,
                                         const T*          B,
                                         int64_t           ldb,
                                         T*                C,
                                         int64_t           ldc);

MAP2CF_D64(rocblas_trmm, float, rocblas_strmm);
MAP2CF_D64(rocblas_trmm, double, rocblas_dtrmm);
MAP2CF_D64(rocblas_trmm, rocblas_float_complex, rocblas_ctrmm);
MAP2CF_D64(rocblas_trmm, rocblas_double_complex, rocblas_ztrmm);

// trmm_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmm_batched)(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transa,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              const T* const    a[],
                                              rocblas_int       lda,
                                              const T* const    b[],
                                              rocblas_int       ldb,
                                              T* const          c[],
                                              rocblas_int       ldc,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmm_batched_64)(rocblas_handle    handle,
                                                 rocblas_side      side,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transa,
                                                 rocblas_diagonal  diag,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 const T*          alpha,
                                                 const T* const    a[],
                                                 int64_t           lda,
                                                 const T* const    b[],
                                                 int64_t           ldb,
                                                 T* const          c[],
                                                 int64_t           ldc,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_trmm_batched, float, rocblas_strmm_batched);
MAP2CF_D64(rocblas_trmm_batched, double, rocblas_dtrmm_batched);
MAP2CF_D64(rocblas_trmm_batched, rocblas_float_complex, rocblas_ctrmm_batched);
MAP2CF_D64(rocblas_trmm_batched, rocblas_double_complex, rocblas_ztrmm_batched);

// trmm_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmm_strided_batched)(rocblas_handle    handle,
                                                      rocblas_side      side,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transa,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      const T*          a,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      const T*          b,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      T*                c,
                                                      rocblas_int       ldc,
                                                      rocblas_stride    stride_c,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trmm_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_side      side,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transa,
                                                         rocblas_diagonal  diag,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         const T*          alpha,
                                                         const T*          a,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         const T*          b,
                                                         int64_t           ldb,
                                                         rocblas_stride    stride_b,
                                                         T*                c,
                                                         int64_t           ldc,
                                                         rocblas_stride    stride_c,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_trmm_strided_batched, float, rocblas_strmm_strided_batched);
MAP2CF_D64(rocblas_trmm_strided_batched, double, rocblas_dtrmm_strided_batched);
MAP2CF_D64(rocblas_trmm_strided_batched, rocblas_float_complex, rocblas_ctrmm_strided_batched);
MAP2CF_D64(rocblas_trmm_strided_batched, rocblas_double_complex, rocblas_ztrmm_strided_batched);

// trsm
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsm)(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      rocblas_int       m,
                                      rocblas_int       n,
                                      const T*          alpha,
                                      T*                A,
                                      rocblas_int       lda,
                                      T*                B,
                                      rocblas_int       ldb);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsm_64)(rocblas_handle    handle,
                                         rocblas_side      side,
                                         rocblas_fill      uplo,
                                         rocblas_operation transA,
                                         rocblas_diagonal  diag,
                                         int64_t           m,
                                         int64_t           n,
                                         const T*          alpha,
                                         T*                A,
                                         int64_t           lda,
                                         T*                B,
                                         int64_t           ldb);

MAP2CF_D64(rocblas_trsm, float, rocblas_strsm);
MAP2CF_D64(rocblas_trsm, double, rocblas_dtrsm);
MAP2CF_D64(rocblas_trsm, rocblas_float_complex, rocblas_ctrsm);
MAP2CF_D64(rocblas_trsm, rocblas_double_complex, rocblas_ztrsm);

// trsm_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsm_batched)(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              T* const          A[],
                                              rocblas_int       lda,
                                              T* const          B[],
                                              rocblas_int       ldb,
                                              rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsm_batched_64)(rocblas_handle    handle,
                                                 rocblas_side      side,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 const T*          alpha,
                                                 T* const          A[],
                                                 int64_t           lda,
                                                 T* const          B[],
                                                 int64_t           ldb,
                                                 int64_t           batch_count);

MAP2CF_D64(rocblas_trsm_batched, float, rocblas_strsm_batched);
MAP2CF_D64(rocblas_trsm_batched, double, rocblas_dtrsm_batched);
MAP2CF_D64(rocblas_trsm_batched, rocblas_float_complex, rocblas_ctrsm_batched);
MAP2CF_D64(rocblas_trsm_batched, rocblas_double_complex, rocblas_ztrsm_batched);

// trsm_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsm_strided_batched)(rocblas_handle    handle,
                                                      rocblas_side      side,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation transA,
                                                      rocblas_diagonal  diag,
                                                      rocblas_int       m,
                                                      rocblas_int       n,
                                                      const T*          alpha,
                                                      T*                A,
                                                      rocblas_int       lda,
                                                      rocblas_stride    stride_a,
                                                      T*                B,
                                                      rocblas_int       ldb,
                                                      rocblas_stride    stride_b,
                                                      rocblas_int       batch_count);

template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trsm_strided_batched_64)(rocblas_handle    handle,
                                                         rocblas_side      side,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         const T*          alpha,
                                                         T*                A,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         T*                B,
                                                         int64_t           ldb,
                                                         rocblas_stride    stride_b,
                                                         int64_t           batch_count);

MAP2CF_D64(rocblas_trsm_strided_batched, float, rocblas_strsm_strided_batched);
MAP2CF_D64(rocblas_trsm_strided_batched, double, rocblas_dtrsm_strided_batched);
MAP2CF_D64(rocblas_trsm_strided_batched, rocblas_float_complex, rocblas_ctrsm_strided_batched);
MAP2CF_D64(rocblas_trsm_strided_batched, rocblas_double_complex, rocblas_ztrsm_strided_batched);

// trtri
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trtri)(rocblas_handle   handle,
                                       rocblas_fill     uplo,
                                       rocblas_diagonal diag,
                                       rocblas_int      n,
                                       T*               A,
                                       rocblas_int      lda,
                                       T*               invA,
                                       rocblas_int      ldinvA);

MAP2CF(rocblas_trtri, float, rocblas_strtri);
MAP2CF(rocblas_trtri, double, rocblas_dtrtri);
MAP2CF(rocblas_trtri, rocblas_float_complex, rocblas_ctrtri);
MAP2CF(rocblas_trtri, rocblas_double_complex, rocblas_ztrtri);

// trtri_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trtri_batched)(rocblas_handle   handle,
                                               rocblas_fill     uplo,
                                               rocblas_diagonal diag,
                                               rocblas_int      n,
                                               T* const         A[],
                                               rocblas_int      lda,
                                               T* const         invA[],
                                               rocblas_int      ldinvA,
                                               rocblas_int      batch_count);

MAP2CF(rocblas_trtri_batched, float, rocblas_strtri_batched);
MAP2CF(rocblas_trtri_batched, double, rocblas_dtrtri_batched);
MAP2CF(rocblas_trtri_batched, rocblas_float_complex, rocblas_ctrtri_batched);
MAP2CF(rocblas_trtri_batched, rocblas_double_complex, rocblas_ztrtri_batched);

// trtri_strided_batched
template <typename T, bool FORTRAN = false>
inline rocblas_status (*rocblas_trtri_strided_batched)(rocblas_handle   handle,
                                                       rocblas_fill     uplo,
                                                       rocblas_diagonal diag,
                                                       rocblas_int      n,
                                                       T*               A,
                                                       rocblas_int      lda,
                                                       rocblas_stride   bsa,
                                                       T*               invA,
                                                       rocblas_int      ldinvA,
                                                       rocblas_stride   bsinvA,
                                                       rocblas_int      batch_count);

MAP2CF(rocblas_trtri_strided_batched, float, rocblas_strtri_strided_batched);
MAP2CF(rocblas_trtri_strided_batched, double, rocblas_dtrtri_strided_batched);
MAP2CF(rocblas_trtri_strided_batched, rocblas_float_complex, rocblas_ctrtri_strided_batched);
MAP2CF(rocblas_trtri_strided_batched, rocblas_double_complex, rocblas_ztrtri_strided_batched);

#undef GET_MACRO
#undef MAP2CF
#undef MAP2CF3
#undef MAP2CF4
#undef MAP2CF5
