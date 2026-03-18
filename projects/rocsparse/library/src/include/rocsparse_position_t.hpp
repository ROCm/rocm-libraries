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

#include "rocsparse-types.h"

namespace rocsparse
{
    struct position_t
    {
    protected:
        rocsparse_indextype m_position_indextype{};
        int64_t             m_batch_count{};
        void*               m_position{};
        position_t();
        ~position_t();

        rocsparse_status create_position_async(int64_t             batch_count,
                                               rocsparse_indextype indextype,
                                               hipStream_t         stream);
        rocsparse_status free_position_async(hipStream_t stream);
        rocsparse_status set_max_position_async(hipStream_t stream);
        rocsparse_status copy_position_async(const position_t* that, hipStream_t stream);

    public:
        int64_t             get_batch_count() const;
        int64_t             get_stride() const;
        rocsparse_indextype get_indextype() const;
        const void*         get_position() const;
        void*               get_position();
    };

}
