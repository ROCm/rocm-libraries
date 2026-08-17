/*! \file */
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

#pragma once

#include "rocsparse_control.hpp"
#include "rocsparse_position_t.hpp"

namespace rocsparse
{
    struct pivot_info_t : rocsparse::position_t
    {
    protected:
        pivot_info_t() = default;

    public:
        ~pivot_info_t() = default;

        inline rocsparse_status create_zero_pivot_async(rocsparse_indextype indextype,
                                                        hipStream_t         stream)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                this->create_zero_pivot_async(static_cast<int64_t>(1), indextype, stream));
            return rocsparse_status_success;
        }

        rocsparse_status create_zero_pivot_async(int64_t             batch_count,
                                                 rocsparse_indextype indextype,
                                                 hipStream_t         stream);

        rocsparse_status copy_pivot_info_async(const pivot_info_t* that, hipStream_t stream);
    };

}
