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

#include "rocsparse-types.h"

struct _rocsparse_sptrsv_descr
{
protected:
    rocsparse_sptrsv_stage m_stage;
    rocsparse_sptrsv_alg   m_alg;
    rocsparse_operation    m_operation;
    rocsparse_datatype     m_scalar_datatype;
    rocsparse_datatype     m_compute_datatype;
    int64_t                m_zero_pivot_position;
    const void*            m_scalar_alpha;

public:
    ~_rocsparse_sptrsv_descr() = default;
    _rocsparse_sptrsv_descr();
    int64_t                get_zero_pivot_position() const;
    rocsparse_sptrsv_stage get_stage() const;
    rocsparse_sptrsv_alg   get_alg() const;
    rocsparse_operation    get_operation() const;
    rocsparse_datatype     get_scalar_datatype() const;
    rocsparse_datatype     get_compute_datatype() const;
    void                   set_stage(rocsparse_sptrsv_stage value);
    void                   set_alg(rocsparse_sptrsv_alg value);
    void                   set_operation(rocsparse_operation value);
    void                   set_scalar_datatype(rocsparse_datatype value);
    void                   set_scalar_alpha(const void* value);
    void                   set_compute_datatype(rocsparse_datatype value);
    void                   set_zero_pivot_position(int64_t value);
    const void*            get_scalar_alpha() const;
};
