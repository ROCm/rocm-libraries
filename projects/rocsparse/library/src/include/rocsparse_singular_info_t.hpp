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
#include "rocsparse_position_t.hpp"

namespace rocsparse
{
    struct singular_info_t : rocsparse::position_t
    {
    protected:
        double                 m_singular_tol_host_value[1]{};
        const double*          m_singular_tol_pointer = m_singular_tol_host_value;
        rocsparse_pointer_mode m_singular_tol_pointer_mode{rocsparse_pointer_mode_host};
        rocsparse_datatype     m_singular_tol_datatype{rocsparse_datatype_f64_r};

    public:
        singular_info_t() = default;
        ~singular_info_t();

        double get_tolerance_legacy() const;
        void   set_tolerance_legacy(double);

        const void*            get_tolerance_pointer() const;
        rocsparse_pointer_mode get_tolerance_pointer_mode() const;
        rocsparse_datatype     get_tolerance_datatype() const;

        void set_tolerance_pointer(const void*            p,
                                   rocsparse_pointer_mode pointer_mode,
                                   rocsparse_datatype     datatype);

        rocsparse_status create_singular_pivot_async(int64_t             batch_count,
                                                     rocsparse_indextype indextype,
                                                     hipStream_t         stream);

        rocsparse_status copy_singular_info_async(const singular_info_t* that, hipStream_t stream);
    };
}
