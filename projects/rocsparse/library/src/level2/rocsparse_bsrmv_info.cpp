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

#include "rocsparse_bsrmv_info.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

/********************************************************************************
 * \brief Copy csrmv info.
 *******************************************************************************/
rocsparse_status rocsparse::copy_bsrmv_info(rocsparse_bsrmv_info       dest,
                                            const rocsparse_bsrmv_info src)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(dest == nullptr || src == nullptr || dest == src)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }
    rocsparse_csrmv_info src_csrmv_info = src->get_csrmv_info();
    if(src_csrmv_info != nullptr)
    {
        rocsparse_csrmv_info csrmv_info = dest->get_csrmv_info();
        if(csrmv_info == nullptr)
        {
            csrmv_info = new _rocsparse_csrmv_info();
            dest->set_csrmv_info(csrmv_info);
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::copy_csrmv_info(csrmv_info, src_csrmv_info));
    }
    return rocsparse_status_success;
}
