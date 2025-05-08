/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_swap_kernels.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_swap.hpp"

template <typename T>
rocblas_status rocblas_swap_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           T              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files swap*.cpp

// clang-format off
#ifdef INSTANTIATE_SWAP_CHECK_NUMERICS
#error INSTANTIATE_SWAP_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_SWAP_CHECK_NUMERICS(T_)                                            \
template rocblas_status rocblas_swap_check_numerics<T_>(const char*    function_name,  \
                                                        rocblas_handle handle,         \
                                                        int64_t    n,              \
                                                        T_             x,              \
                                                        rocblas_stride offset_x,       \
                                                        int64_t    inc_x,          \
                                                        rocblas_stride stride_x,       \
                                                        T_             y,              \
                                                        rocblas_stride offset_y,       \
                                                        int64_t    inc_y,          \
                                                        rocblas_stride stride_y,       \
                                                        int64_t    batch_count,    \
                                                        const int      check_numerics, \
                                                        bool           is_input);

INSTANTIATE_SWAP_CHECK_NUMERICS(float*)
INSTANTIATE_SWAP_CHECK_NUMERICS(double*)
INSTANTIATE_SWAP_CHECK_NUMERICS(rocblas_float_complex*)
INSTANTIATE_SWAP_CHECK_NUMERICS(rocblas_double_complex*)

INSTANTIATE_SWAP_CHECK_NUMERICS(float* const*)
INSTANTIATE_SWAP_CHECK_NUMERICS(double* const*)
INSTANTIATE_SWAP_CHECK_NUMERICS(rocblas_float_complex* const*)
INSTANTIATE_SWAP_CHECK_NUMERICS(rocblas_double_complex* const*)

#undef INSTANTIATE_SWAP_CHECK_NUMERICS

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files copy*.cpp

#ifdef INSTANTIATE_SWAP_LAUNCHER
#error INSTANTIATE_SWAP_LAUNCHER already defined
#endif

#define INSTANTIATE_SWAP_LAUNCHER(NB_, T_)                                            \
    template rocblas_status rocblas_internal_swap_launcher<rocblas_int, NB_, T_>(     \
        rocblas_handle handle,                                                        \
        rocblas_int    n,                                                             \
        T_             x,                                                             \
        rocblas_stride offsetx,                                                       \
        rocblas_int    incx,                                                          \
        rocblas_stride stridex,                                                       \
        T_             y,                                                             \
        rocblas_stride offsety,                                                       \
        rocblas_int    incy,                                                          \
        rocblas_stride stridey,                                                       \
        rocblas_int    batch_count);

// non batched

INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, float*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, double*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, rocblas_float_complex*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, rocblas_double_complex*)

// batched

INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, float* const*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, double* const*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, rocblas_float_complex* const*)
INSTANTIATE_SWAP_LAUNCHER(ROCBLAS_SWAP_NB, rocblas_double_complex* const*)

#undef INSTANTIATE_SWAP_LAUNCHER
// clang-format on
