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
#include "rocblas_rotmg_64.hpp"

#include "blas1/rocblas_rotmg.hpp" // int32 API called

template <typename API_INT, typename T, typename U>
ROCBLAS_INTERNAL_ONLY_EXPORT_NOINLINE rocblas_status
    rocblas_internal_rotmg_launcher(rocblas_handle handle,
                                    T              d1_in,
                                    rocblas_stride offset_d1,
                                    rocblas_stride stride_d1,
                                    T              d2_in,
                                    rocblas_stride offset_d2,
                                    rocblas_stride stride_d2,
                                    T              x1_in,
                                    rocblas_stride offset_x1,
                                    rocblas_stride stride_x1,
                                    U              y1_in,
                                    rocblas_stride offset_y1,
                                    rocblas_stride stride_y1,
                                    T              param,
                                    rocblas_stride offset_param,
                                    rocblas_stride stride_param,
                                    API_INT        batch_count);

template <typename API_INT, typename T, typename U>
rocblas_status rocblas_internal_rotmg_launcher_64(rocblas_handle handle,
                                                  T              d1_in,
                                                  rocblas_stride offset_d1,
                                                  rocblas_stride stride_d1,
                                                  T              d2_in,
                                                  rocblas_stride offset_d2,
                                                  rocblas_stride stride_d2,
                                                  T              x1_in,
                                                  rocblas_stride offset_x1,
                                                  rocblas_stride stride_x1,
                                                  U              y1_in,
                                                  rocblas_stride offset_y1,
                                                  rocblas_stride stride_y1,
                                                  T              param,
                                                  rocblas_stride offset_param,
                                                  rocblas_stride stride_param,
                                                  API_INT        batch_count_64)
{
    // Quick returns handled earlier

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto d1_ptr = adjust_ptr_batch(d1_in, b_base, stride_d1);
        auto d2_ptr = adjust_ptr_batch(d2_in, b_base, stride_d2);
        auto x1_ptr = adjust_ptr_batch(x1_in, b_base, stride_x1);
        auto y1_ptr = adjust_ptr_batch(y1_in, b_base, stride_y1);
        auto p_ptr  = adjust_ptr_batch(param, b_base, stride_param);

        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        // 32bit API call
        rocblas_status status = rocblas_internal_rotmg_launcher<rocblas_int, T, U>(handle,
                                                                                   d1_ptr,
                                                                                   offset_d1,
                                                                                   stride_d1,
                                                                                   d2_ptr,
                                                                                   offset_d2,
                                                                                   stride_d2,
                                                                                   x1_ptr,
                                                                                   offset_x1,
                                                                                   stride_x1,
                                                                                   y1_ptr,
                                                                                   offset_y1,
                                                                                   stride_y1,
                                                                                   p_ptr,
                                                                                   offset_param,
                                                                                   stride_param,
                                                                                   batch_count);
        if(status != rocblas_status_success)
            return status;
    }

    return rocblas_status_success;
}

#ifdef INST_ROTMG_LAUNCHER
#error INST_ROTMG_LAUNCHER already defined
#endif

#define INST_ROTMG_LAUNCHER(TI_, T_, U_)                                     \
    template rocblas_status rocblas_internal_rotmg_launcher_64<TI_, T_, U_>( \
        rocblas_handle handle,                                               \
        T_             d1_in,                                                \
        rocblas_stride offset_d1,                                            \
        rocblas_stride stride_d1,                                            \
        T_             d2_in,                                                \
        rocblas_stride offset_d2,                                            \
        rocblas_stride stride_d2,                                            \
        T_             x1_in,                                                \
        rocblas_stride offset_x1,                                            \
        rocblas_stride stride_x1,                                            \
        U_             y1_in,                                                \
        rocblas_stride offset_y1,                                            \
        rocblas_stride stride_y1,                                            \
        T_             param,                                                \
        rocblas_stride offset_param,                                         \
        rocblas_stride stride_param,                                         \
        TI_            batch_count);

// instantiate for rocblas_Xrotmg and rocblas_Xrotg_strided_batched
INST_ROTMG_LAUNCHER(int64_t, float*, float const*)
INST_ROTMG_LAUNCHER(int64_t, double*, double const*)

// instantiate for rocblas_Xrotmg_strided_batched
INST_ROTMG_LAUNCHER(int64_t, float* const*, float const* const*)
INST_ROTMG_LAUNCHER(int64_t, double* const*, double const* const*)

#undef INST_ROTMG_LAUNCHER
