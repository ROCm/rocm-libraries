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
    struct position_t
    {
    protected:
        rocsparse_indextype m_position_indextype{};
        void*               m_position{};
        position_t();

        void copy_value(hipStream_t stream, rocsparse_pointer_mode mode, int64_t* value) const;

        void copy_async(hipStream_t            stream,
                        rocsparse_pointer_mode mode,
                        rocsparse_indextype    indextype,
                        void*                  value) const;

        ~position_t();

        void create_position_async(rocsparse_indextype indextype, hipStream_t stream);
        void free_position_async(hipStream_t stream);

        rocsparse_indextype get_position_indextype() const;
        const void*         get_position() const;
        void*               get_position();

        rocsparse_status copy_position_async(rocsparse_pointer_mode pointer_mode,
                                             rocsparse_indextype    position_indextype,
                                             void*                  position,
                                             hipStream_t            stream) const;

        void set_max_position_async(hipStream_t stream);

        void copy_position_async(const position_t* that, hipStream_t stream);
    };

}
