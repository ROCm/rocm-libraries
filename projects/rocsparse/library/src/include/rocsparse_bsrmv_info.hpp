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
 * \brief rocsparse_bsrmv_info is a structure holding the rocsparse bsrmv info
 * data gathered during bsrmv_analysis.
 *******************************************************************************/
typedef struct _rocsparse_bsrmv_info
{
protected:
    rocsparse_csrmv_info m_csrmv_info{};

public:
    _rocsparse_bsrmv_info() {}

    ~_rocsparse_bsrmv_info()
    {
        if(this->m_csrmv_info)
        {
            delete this->m_csrmv_info;
            this->m_csrmv_info = nullptr;
        }
    }

    rocsparse_csrmv_info get_csrmv_info()
    {
        return this->m_csrmv_info;
    }
    void set_csrmv_info(rocsparse_csrmv_info value)
    {
        this->m_csrmv_info = value;
    }

} * rocsparse_bsrmv_info;

namespace rocsparse
{
    /********************************************************************************
   * \brief Copy csrmv info.
   *******************************************************************************/
    rocsparse_status copy_bsrmv_info(rocsparse_bsrmv_info dest, const rocsparse_bsrmv_info src);
}
