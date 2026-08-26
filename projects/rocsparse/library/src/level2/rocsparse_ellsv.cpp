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

#include "rocsparse_control.hpp"
#include "rocsparse_ellsv.hpp"
#include "rocsparse_mat_info.hpp"
#include "rocsparse_one.hpp"
#include "rocsparse_utility.hpp"

#include <string>

namespace rocsparse
{
    void ellsv_select_launch(rocsparse_handle handle, bool* sleep, uint32_t* wfsize)
    {
        const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
        const int         asic_rev      = handle->asic_rev;
        *sleep  = (gcn_arch_name == rocsparse_arch_names::gfx908 && asic_rev < 2);
        *wfsize = (*sleep) ? 64 : handle->wavefront_size;
    }

    rocsparse_fill_mode ellsv_flip_fill(rocsparse_fill_mode fill_mode)
    {
        return (fill_mode == rocsparse_fill_mode_lower) ? rocsparse_fill_mode_upper
                                                        : rocsparse_fill_mode_lower;
    }

    rocsparse_status ellsv_check(rocsparse_const_spmat_descr A)
    {
        rocsparse_mat_descr descr = A->descr;
        ROCSPARSE_CHECKARG(2, A, (A->rows != A->cols), rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(2,
                           descr,
                           (descr->type != rocsparse_matrix_type_general
                            && descr->type != rocsparse_matrix_type_triangular),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(2,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        return rocsparse_status_success;
    }

    rocsparse_status ellsv_zero_pivot(rocsparse_handle     handle,
                                      rocsparse_ellsv_info info,
                                      rocsparse_indextype  indextype,
                                      void*                position)
    {
        ROCSPARSE_ROUTINE_TRACE;

        auto numeric_exact_position = (info) ? info->get_singularity_numeric_exact() : nullptr;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::singularity_get_position_async(handle,
                                                                            1,
                                                                            info,
                                                                            numeric_exact_position,
                                                                            nullptr,
                                                                            handle->pointer_mode,
                                                                            indextype,
                                                                            position));

        switch(indextype)
        {
        case rocsparse_indextype_i32:
        {
            int32_t p;
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                &p, position, sizeof(int32_t), hipMemcpyDefault, handle->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
            if(p != -1)
            {
                return rocsparse_status_zero_pivot;
            }
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            int64_t p;
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                &p, position, sizeof(int64_t), hipMemcpyDefault, handle->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
            if(p != -1)
            {
                return rocsparse_status_zero_pivot;
            }
            return rocsparse_status_success;
        }
            // LCOV_EXCL_START
        case deprecated_rocsparse_indextype_u16:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   "rocsparse_indextype_u16 not supported");
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}
