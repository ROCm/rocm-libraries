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

#include "rocblas_ger_kernels.hpp"
#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "rocblas_ger.hpp"

template <typename T, typename U>
rocblas_status rocblas_ger_check_numerics(const char*    function_name,
                                          rocblas_handle handle,
                                          int64_t        m,
                                          int64_t        n,
                                          U              A,
                                          rocblas_stride offset_a,
                                          int64_t        lda,
                                          rocblas_stride stride_a,
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
        = rocblas_internal_check_numerics_matrix_template(function_name,
                                                          handle,
                                                          rocblas_operation_none,
                                                          rocblas_fill_full,
                                                          rocblas_client_general_matrix,
                                                          m,
                                                          n,
                                                          A,
                                                          offset_a,
                                                          lda,
                                                          stride_a,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    if(is_input)
    {
        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                m,
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
    }
    return check_numerics_status;
}

#ifdef INSTANTIATE_GER_NUMERICS
#error INSTANTIATE_GER_NUMERICS already defined
#endif

#define INSTANTIATE_GER_NUMERICS(T_, U_)                                                      \
    template rocblas_status rocblas_ger_check_numerics<T_, U_>(const char*    function_name,  \
                                                               rocblas_handle handle,         \
                                                               int64_t        m,              \
                                                               int64_t        n,              \
                                                               U_             A,              \
                                                               rocblas_stride offset_a,       \
                                                               int64_t        lda,            \
                                                               rocblas_stride stride_a,       \
                                                               T_             x,              \
                                                               rocblas_stride offset_x,       \
                                                               int64_t        inc_x,          \
                                                               rocblas_stride stride_x,       \
                                                               T_             y,              \
                                                               rocblas_stride offset_y,       \
                                                               int64_t        inc_y,          \
                                                               rocblas_stride stride_y,       \
                                                               int64_t        batch_count,    \
                                                               const int      check_numerics, \
                                                               bool           is_input);

INSTANTIATE_GER_NUMERICS(float const*, float*)
INSTANTIATE_GER_NUMERICS(double const*, double*)
INSTANTIATE_GER_NUMERICS(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_GER_NUMERICS(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_GER_NUMERICS(float const* const*, float* const*)
INSTANTIATE_GER_NUMERICS(double const* const*, double* const*)
INSTANTIATE_GER_NUMERICS(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_GER_NUMERICS(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_GER_NUMERICS

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *ger*.cpp

INSTANTIATE_GER_TEMPLATE(rocblas_int, float)
INSTANTIATE_GER_TEMPLATE(rocblas_int, double)
INSTANTIATE_GER_TEMPLATE(rocblas_int, rocblas_float_complex)
INSTANTIATE_GER_TEMPLATE(rocblas_int, rocblas_double_complex)

INSTANTIATE_GERC_TEMPLATE(rocblas_int, rocblas_float_complex)
INSTANTIATE_GERC_TEMPLATE(rocblas_int, rocblas_double_complex)

INSTANTIATE_GER_BATCHED_TEMPLATE(rocblas_int, float)
INSTANTIATE_GER_BATCHED_TEMPLATE(rocblas_int, double)
INSTANTIATE_GER_BATCHED_TEMPLATE(rocblas_int, rocblas_float_complex)
INSTANTIATE_GER_BATCHED_TEMPLATE(rocblas_int, rocblas_double_complex)

INSTANTIATE_GERC_BATCHED_TEMPLATE(rocblas_int, rocblas_float_complex)
INSTANTIATE_GERC_BATCHED_TEMPLATE(rocblas_int, rocblas_double_complex)
