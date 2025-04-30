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

#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_rotg_64.hpp"

#include "blas1/rocblas_rotg.hpp" // int32 API called
//#include "blas1/rocblas_rotg_kernels.hpp"

// prototype for 32 bit launcher
template <typename API_INT, typename T, typename U>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE rocblas_status
    rocblas_internal_rotg_launcher(rocblas_handle handle,
                                   T              a_in,
                                   rocblas_stride offset_a,
                                   rocblas_stride stride_a,
                                   T              b_in,
                                   rocblas_stride offset_b,
                                   rocblas_stride stride_b,
                                   U              c_in,
                                   rocblas_stride offset_c,
                                   rocblas_stride stride_c,
                                   T              s_in,
                                   rocblas_stride offset_s,
                                   rocblas_stride stride_s,
                                   API_INT        batch_count);

template <typename API_INT, typename T, typename U>
rocblas_status rocblas_internal_rotg_launcher_64(rocblas_handle handle,
                                                 T              a_in,
                                                 rocblas_stride offset_a,
                                                 rocblas_stride stride_a,
                                                 T              b_in,
                                                 rocblas_stride offset_b,
                                                 rocblas_stride stride_b,
                                                 U              c_in,
                                                 rocblas_stride offset_c,
                                                 rocblas_stride stride_c,
                                                 T              s_in,
                                                 rocblas_stride offset_s,
                                                 rocblas_stride stride_s,
                                                 API_INT        batch_count_64)
{
    // Quick returns handled earlier

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto a_ptr = adjust_ptr_batch(a_in, b_base, stride_a);
        auto b_ptr = adjust_ptr_batch(b_in, b_base, stride_b);
        auto c_ptr = adjust_ptr_batch(c_in, b_base, stride_c);
        auto s_ptr = adjust_ptr_batch(s_in, b_base, stride_s);

        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        // 32bit API call
        rocblas_status status = rocblas_internal_rotg_launcher<rocblas_int, T, U>(handle,
                                                                                  a_ptr,
                                                                                  offset_a,
                                                                                  stride_a,
                                                                                  b_ptr,
                                                                                  offset_b,
                                                                                  stride_b,
                                                                                  c_ptr,
                                                                                  offset_c,
                                                                                  stride_c,
                                                                                  s_ptr,
                                                                                  offset_s,
                                                                                  stride_s,
                                                                                  batch_count);
        if(status != rocblas_status_success)
            return status;
    }

    return rocblas_status_success;
}

#ifdef INST_ROTG_LAUNCHER
#error INST_ROTG_LAUNCHER already defined
#endif

#define INST_ROTG_LAUNCHER(T_, U_)                                              \
    template rocblas_status rocblas_internal_rotg_launcher_64<int64_t, T_, U_>( \
        rocblas_handle handle,                                                  \
        T_             a_in,                                                    \
        rocblas_stride offset_a,                                                \
        rocblas_stride stride_a,                                                \
        T_             b_in,                                                    \
        rocblas_stride offset_b,                                                \
        rocblas_stride stride_b,                                                \
        U_             c_in,                                                    \
        rocblas_stride offset_c,                                                \
        rocblas_stride stride_c,                                                \
        T_             s_in,                                                    \
        rocblas_stride offset_s,                                                \
        rocblas_stride stride_s,                                                \
        int64_t        batch_count);

// instantiate for rocblas_Xrotg and rocblas_Xrotg_strided_batched
INST_ROTG_LAUNCHER(float*, float*)
INST_ROTG_LAUNCHER(double*, double*)
INST_ROTG_LAUNCHER(rocblas_float_complex*, float*)
INST_ROTG_LAUNCHER(rocblas_double_complex*, double*)

// instantiate for rocblas_Xrotg_batched
INST_ROTG_LAUNCHER(float* const*, float* const*)
INST_ROTG_LAUNCHER(double* const*, double* const*)
INST_ROTG_LAUNCHER(rocblas_float_complex* const*, float* const*)
INST_ROTG_LAUNCHER(rocblas_double_complex* const*, double* const*)
