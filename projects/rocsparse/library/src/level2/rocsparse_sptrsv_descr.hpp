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
#include "rocsparse_csrsv_info.hpp"
#include <memory>
struct _rocsparse_sptrsv_descr
{
    const void* m_scalar_alpha;
    const void* s_scalar_alpha;

protected:
    rocsparse_sptrsv_stage                 m_stage;
    rocsparse_sptrsv_alg                   m_alg;
    rocsparse_operation                    m_operation;
    rocsparse_datatype                     m_scalar_datatype;
    rocsparse_datatype                     m_compute_datatype;
    rocsparse_analysis_policy              m_analysis_policy;
    std::shared_ptr<_rocsparse_csrsv_info> m_csrsv_info;

public:
    ~_rocsparse_sptrsv_descr();
    _rocsparse_sptrsv_descr();

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

    const void*               get_scalar_alpha() const;
    rocsparse_analysis_policy get_analysis_policy() const;
    void                      set_analysis_policy(rocsparse_analysis_policy value);

    rocsparse_csrsv_info get_csrsv_info();
    void                 set_csrsv_info(rocsparse_csrsv_info value);
    void                 set_shared_csrsv_info(std::shared_ptr<_rocsparse_csrsv_info> value);

    float m_local_host_alpha_value[4];

    void* get_local_host_alpha()
    {
        return &this->m_local_host_alpha_value[0];
    }
};
