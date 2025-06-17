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

#pragma once

#include "rocsparse_csrmv_info.hpp"

/********************************************************************************
 * \brief rocsparse_csritsv_info is a structure holding the rocsparse csritsv
 * info data gathered during csritsv_buffer_size. It must be initialized using
 * the create_csritsv_info() routine. It should be destroyed at the
 * end using destroy_csritsv_info().
 *******************************************************************************/
typedef struct _rocsparse_csritsv_info
{
protected:
    rocsparse_csrmv_info m_csrmv_info{};

public:
    bool                is_submatrix;
    int64_t             ptr_end_size{};
    rocsparse_indextype ptr_end_indextype{};
    void*               ptr_end{};

    rocsparse_csrmv_info get_csrmv_info()
    {
        return this->m_csrmv_info;
    }
    void set_csrmv_info(rocsparse_csrmv_info value)
    {
        this->m_csrmv_info = value;
    }

    _rocsparse_csritsv_info() = default;

    ~_rocsparse_csritsv_info()
    {
        if(this->m_csrmv_info)
        {
            delete this->m_csrmv_info;
        }
    }
} * rocsparse_csritsv_info;

namespace rocsparse
{
    /********************************************************************************
 * \brief rocsparse_csritsv_info is a structure holding the rocsparse csritsv
 * info data gathered during csritsv_buffer_size. It must be initialized using
 * the create_csritsv_info() routine. It should be destroyed at the
 * end using destroy_csritsv_info().
 *******************************************************************************/
    rocsparse_status create_csritsv_info(rocsparse_csritsv_info* info);

    /********************************************************************************
 * \brief Copy csritsv info.
 *******************************************************************************/
    rocsparse_status copy_csritsv_info(rocsparse_csritsv_info       dest,
                                       const rocsparse_csritsv_info src);

    /********************************************************************************
 * \brief Destroy csritsv info.
 *******************************************************************************/
    rocsparse_status destroy_csritsv_info(rocsparse_csritsv_info info);
}
