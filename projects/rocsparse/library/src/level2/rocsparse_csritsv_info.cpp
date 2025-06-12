/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csritsv_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

/********************************************************************************
 * \brief rocsparse_csritsv_info is a structure holding the rocsparse csritsv
 * info data gathered during csritsv_buffer_size. It must be initialized using
 * the create_csritsv_info() routine. It should be destroyed at the
 * end using destroy_csritsv_info().
 *******************************************************************************/
rocsparse_status rocsparse::create_csritsv_info(rocsparse_csritsv_info* info)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(info == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }
    else
    {
        // Allocate
        try
        {
            *info = new _rocsparse_csritsv_info;
        }
        catch(const rocsparse_status& status)
        {
            return status;
        }
        return rocsparse_status_success;
    }
}

/********************************************************************************
 * \brief Copy csritsv info.
 *******************************************************************************/
rocsparse_status rocsparse::copy_csritsv_info(rocsparse_csritsv_info       dest,
                                              const rocsparse_csritsv_info src)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(dest == nullptr || src == nullptr || dest == src)
    {
        return rocsparse_status_invalid_pointer;
    }
    dest->is_submatrix      = src->is_submatrix;
    dest->ptr_end_size      = src->ptr_end_size;
    dest->ptr_end_indextype = src->ptr_end_indextype;
    dest->ptr_end           = src->ptr_end;
    return rocsparse_status_success;
}

/********************************************************************************
 * \brief Destroy csritsv info.
 *******************************************************************************/
rocsparse_status rocsparse::destroy_csritsv_info(rocsparse_csritsv_info info)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    if(info->ptr_end != nullptr && info->is_submatrix)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(info->ptr_end));
        info->ptr_end = nullptr;
    }

    // Destruct
    try
    {
        delete info;
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }
    return rocsparse_status_success;
}
