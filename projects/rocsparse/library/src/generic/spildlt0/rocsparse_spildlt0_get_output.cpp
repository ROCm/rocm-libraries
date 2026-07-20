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

#include "rocsparse_one.hpp"
#include "rocsparse_singularity.hpp"
#include "rocsparse_spildlt0_descr.hpp"
#include "rocsparse_utility.hpp"

#include "internal/generic/rocsparse_spildlt0.h"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spildlt0_output value)
{
    switch(value)
    {
    case rocsparse_spildlt0_output_singularity:
    case rocsparse_spildlt0_output_singularity_position:
    {
        return false;
    }
    }
    return true;
};

extern "C" rocsparse_status rocsparse_spildlt0_get_output(rocsparse_handle          handle,
                                                          rocsparse_spildlt0_descr  spildlt0_descr,
                                                          rocsparse_spildlt0_output output,
                                                          void*                     data,
                                                          size_t           data_size_in_bytes,
                                                          rocsparse_error* p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    //
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    //
    ROCSPARSE_CHECKARG_POINTER(1, spildlt0_descr);
    //
    ROCSPARSE_CHECKARG_ENUM(2, output);
    //
    ROCSPARSE_CHECKARG_POINTER(3, data);
    //
    ROCSPARSE_CHECKARG(
        4, data_size_in_bytes, data_size_in_bytes == 0, rocsparse_status_invalid_size);
    //

    switch(output)
    {
    case rocsparse_spildlt0_output_singularity_position:
    case rocsparse_spildlt0_output_singularity:
    {
        const bool determine_singularity = (output == rocsparse_spildlt0_output_singularity);
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

        const size_t sizeelm
            = (determine_singularity) ? sizeof(rocsparse_singularity) : sizeof(int64_t);

        rocsparse::pivot_info_t*    symbolic_pivot{};
        rocsparse::singular_info_t* exact_pivot{};
        rocsparse::singular_info_t* near_pivot{};

        auto csrildlt0_info = spildlt0_descr->get_csrildlt0_info();
        if(csrildlt0_info != nullptr)
        {
            symbolic_pivot = static_cast<rocsparse::pivot_info_t*>(csrildlt0_info);
            exact_pivot    = csrildlt0_info->get_singularity_numeric_exact();
            near_pivot     = csrildlt0_info->get_singularity_numeric_near();
        }

        ROCSPARSE_CHECKARG(
            4, data_size_in_bytes, data_size_in_bytes != sizeelm, rocsparse_status_invalid_size);

        if(determine_singularity)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::singularity_get_async(handle,
                                                 spildlt0_descr->m_batch_count,
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
                                                          spildlt0_descr->m_batch_count,
                                                          symbolic_pivot,
                                                          exact_pivot,
                                                          near_pivot,
                                                          handle->pointer_mode,
                                                          rocsparse_indextype_i64,
                                                          data));
        }
        return rocsparse_status_success;
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
