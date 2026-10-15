/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_sptrsm.h"
#include "rocsparse_control.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "../../conversion/rocsparse_convert_array.hpp"
#include "../../conversion/rocsparse_convert_scalar.hpp"
#include "../rocsparse_sptrsm_descr.hpp"
#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_common.h"
#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsm_output value)
{
    switch(value)
    {
    case rocsparse_sptrsm_output_singularity_position:
    case rocsparse_sptrsm_output_singularity:
    case rocsparse_sptrsm_output_zero_pivot_position:
    {
        return false;
    }
    }
    return true;
};

extern "C" rocsparse_status rocsparse_sptrsm_get_output(rocsparse_handle        handle,
                                                        rocsparse_sptrsm_descr  sptrsm_descr,
                                                        rocsparse_sptrsm_output output,
                                                        void*                   data,
                                                        size_t                  data_size_in_bytes,
                                                        rocsparse_error*        p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsm_descr);
    ROCSPARSE_CHECKARG_ENUM(2, output);
    ROCSPARSE_CHECKARG_POINTER(3, data);
    ROCSPARSE_CHECKARG(
        4, data_size_in_bytes, data_size_in_bytes == 0, rocsparse_status_invalid_size);

    switch(output)
    {
    case rocsparse_sptrsm_output_singularity_position:
    case rocsparse_sptrsm_output_singularity:
    {

        const bool determine_singularity = (output == rocsparse_sptrsm_output_singularity);
        if(determine_singularity)
        {
            ROCSPARSE_CHECKARG(4,
                               data_size_in_bytes,
                               data_size_in_bytes != sizeof(rocsparse_singularity),
                               rocsparse_status_invalid_value);
        }
        else
        {
            ROCSPARSE_CHECKARG(4,
                               data_size_in_bytes,
                               data_size_in_bytes != sizeof(int64_t),
                               rocsparse_status_invalid_value);
        }

        rocsparse::pivot_info_t*    symbolic_pivot{};
        rocsparse::singular_info_t* exact_pivot{};
        rocsparse::singular_info_t* near_pivot{};

        auto csrsm_info = sptrsm_descr->get_csrsm_info();
        if(csrsm_info != nullptr)
        {
            symbolic_pivot = static_cast<rocsparse::pivot_info_t*>(csrsm_info);
            exact_pivot    = csrsm_info->get_singularity_numeric_exact();
        }

        if(determine_singularity)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::singularity_get_async(handle,
                                                 sptrsm_descr->get_batch_count(),
                                                 symbolic_pivot,
                                                 exact_pivot,
                                                 near_pivot,
                                                 handle->pointer_mode,
                                                 data));
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::singularity_get_position_async(handle,
                                                          sptrsm_descr->get_batch_count(),
                                                          symbolic_pivot,
                                                          exact_pivot,
                                                          near_pivot,
                                                          handle->pointer_mode,
                                                          rocsparse_indextype_i64,
                                                          data));
        }

        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_output_zero_pivot_position:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(int64_t),
                           rocsparse_status_invalid_size);

        auto csrsm_info = sptrsm_descr->get_csrsm_info();

        auto status
            = rocsparse::csrsm_zero_pivot(handle, csrsm_info, rocsparse_indextype_i64, data);
        if(status != rocsparse_status_zero_pivot)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
        }

        return status;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
