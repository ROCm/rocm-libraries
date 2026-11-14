/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    struct cscsv_kernel_constants_t
    {
        static constexpr uint32_t s_blocksize = 256;
        // Entries per lane of a task.
        static constexpr uint32_t s_num_items = 8;
    };

    typedef rocsparse_status (*cscsv_init_launch_t)(rocsparse_handle handle,
                                                    int64_t          m,
                                                    int64_t          batch_count,
                                                    const void*      alpha,
                                                    int64_t          alpha_stride,
                                                    const void*      dependency_count,
                                                    const void*      x,
                                                    int64_t          x_inc,
                                                    int64_t          x_stride,
                                                    void*            rem,
                                                    void*            sum);

    rocsparse_status cscsv_init_launch_find(rocsparse::cscsv_init_launch_t* launch,
                                            rocsparse_indextype             j_type,
                                            rocsparse_datatype              t_type);

    typedef rocsparse_status (*cscsv_solve_launch_t)(rocsparse_handle     handle,
                                                     int64_t              batch_count,
                                                     int64_t              m,
                                                     const void*          csc_col_ptr,
                                                     const void*          csc_row_ind,
                                                     const void*          csc_val,
                                                     int64_t              csc_val_stride,
                                                     const void*          csc_diag_ind,
                                                     int64_t              num_tasks,
                                                     const void*          task_col,
                                                     const void*          task_begin,
                                                     void*                y,
                                                     int64_t              y_inc,
                                                     int64_t              y_stride,
                                                     void*                rem,
                                                     void*                sum,
                                                     int32_t*             ticket,
                                                     void*                zero_pivot,
                                                     rocsparse_index_base idx_base,
                                                     rocsparse_fill_mode  fill_mode,
                                                     rocsparse_diag_type  diag_type);

    rocsparse_status cscsv_solve_launch_find(rocsparse::cscsv_solve_launch_t* launch,
                                             uint32_t                         num_lanes,
                                             rocsparse_indextype              i_type,
                                             rocsparse_indextype              j_type,
                                             rocsparse_datatype               t_type);
}
