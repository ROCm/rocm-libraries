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

#include "rocsparse-types.h"

namespace rocsparse
{
    struct sorted_coo2csr_info_t
    {
    private:
        int64_t             m_num_rows{};
        rocsparse_indextype m_row_ptr_indextype{};
        void*               m_row_ptr{};

    public:
        const void* get_row_ptr() const;
        sorted_coo2csr_info_t() = delete;

        sorted_coo2csr_info_t(int64_t             num_rows,
                              rocsparse_indextype ptr_indextype,
                              hipStream_t         stream);

        hipError_t free_memory(hipStream_t stream);
        ~sorted_coo2csr_info_t();

        rocsparse_status calculate(rocsparse_handle     handle,
                                   int64_t              coo_row_ind_size,
                                   const void*          coo_row_ind,
                                   rocsparse_indextype  coo_row_ind_indextype,
                                   rocsparse_index_base coo_row_ind_index_base);
    };
}
