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

#include "internal/auxiliary/rocsparse_mat_info_backward.h"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocsparse_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_mat_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, info);
    *info = new _rocsparse_mat_info;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Copy mat info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_mat_info(rocsparse_mat_info dest, const rocsparse_mat_info src)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, dest);
    ROCSPARSE_CHECKARG_POINTER(1, src);
    ROCSPARSE_CHECKARG(1, src, (src == dest), rocsparse_status_invalid_pointer);

    dest->duplicate_trdata(src, 0);

    hipStream_t          default_stream{};
    rocsparse_csrmv_info src_csrmv_info  = src->get_csrmv_info();
    rocsparse_csrmv_info dest_csrmv_info = dest->get_csrmv_info();
    if(src_csrmv_info != nullptr)
    {
        if(dest_csrmv_info == nullptr)
        {
            dest_csrmv_info = new _rocsparse_csrmv_info();
            dest->set_csrmv_info(dest_csrmv_info);
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::copy_csrmv_info(dest_csrmv_info, src_csrmv_info, default_stream));
    }

    rocsparse_bsrmv_info src_bsrmv_info  = src->get_bsrmv_info();
    rocsparse_bsrmv_info dest_bsrmv_info = dest->get_bsrmv_info();
    if(src_bsrmv_info != nullptr)
    {
        if(dest_bsrmv_info == nullptr)
        {
            dest_bsrmv_info = new _rocsparse_bsrmv_info();
            dest->set_bsrmv_info(dest_bsrmv_info);
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::copy_bsrmv_info(dest_bsrmv_info, src_bsrmv_info, default_stream));
    }

    if(src->csrgemm_info != nullptr)
    {
        if(dest->csrgemm_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_csrgemm_info(&dest->csrgemm_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::copy_csrgemm_info(dest->csrgemm_info, src->csrgemm_info));
    }

    if(src->csritsv_info != nullptr)
    {
        if(dest->csritsv_info == nullptr)
        {
            dest->csritsv_info = new _rocsparse_csritsv_info();
        }
        hipStream_t default_stream{};
        dest->csritsv_info->copy(src->csritsv_info, default_stream);
    }
    dest->get_boost()->copy(*src->get_boost());
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Destroy mat info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    hipStream_t default_stream{};
    info->destroy(default_stream);
    delete info;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#ifdef __cplusplus
}
#endif
