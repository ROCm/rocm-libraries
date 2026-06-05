/*! \file */
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

#pragma once

#include "rocsparse-types.h"

namespace rocsparse
{
    struct numeric_boost
    {
    protected:
        int32_t m_enable{};

        rocsparse_pointer_mode m_tol_pointer_mode{};
        rocsparse_datatype     m_tol_datatype{rocsparse_datatype_f32_r};
        const void*            m_tol{};

        rocsparse_pointer_mode m_val_pointer_mode{};
        const void*            m_val{};

    public:
        numeric_boost();
        ~numeric_boost();

        void define(int32_t                enable,
                    rocsparse_pointer_mode tol_pointer_mode,
                    rocsparse_datatype     tol_datatype,
                    const void*            tol,
                    rocsparse_pointer_mode val_pointer_mode,
                    const void*            val);

        const void*            get_tol() const;
        void                   set_tol(const void*);
        const void*            get_val() const;
        void                   set_val(const void*);
        int32_t                get_enable() const;
        void                   set_enable(int32_t);
        rocsparse_datatype     get_tol_datatype() const;
        void                   set_tol_datatype(rocsparse_datatype);
        rocsparse_pointer_mode get_tol_pointer_mode() const;
        void                   set_tol_pointer_mode(rocsparse_pointer_mode);
        rocsparse_pointer_mode get_val_pointer_mode() const;
        void                   set_val_pointer_mode(rocsparse_pointer_mode);
        void                   copy(const rocsparse::numeric_boost& that);
    };
}
