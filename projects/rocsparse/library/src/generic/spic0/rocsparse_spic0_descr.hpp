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

#include "internal/generic/rocsparse_spic0.h"
#include "rocsparse_bsric0_info.hpp"
#include "rocsparse_csric0_info.hpp"
#include "rocsparse_mat_info.hpp"

struct _rocsparse_spic0_descr
{
protected:
    rocsparse_spic0_stage                   m_stage;
    rocsparse_spic0_alg                     m_alg;
    rocsparse_datatype                      m_compute_datatype;
    rocsparse_analysis_policy               m_analysis_policy;
    std::shared_ptr<_rocsparse_csric0_info> m_csric0_info;
    std::shared_ptr<_rocsparse_bsric0_info> m_bsric0_info;

    rocsparse_format       m_format{};
    const double*          m_tolerance_pointer{};
    rocsparse_pointer_mode m_tolerance_pointer_mode{};

    rocsparse::numeric_boost m_boost{};

public:
    int64_t                   m_batch_count{};
    rocsparse::numeric_boost* get_boost();

    ~_rocsparse_spic0_descr();
    _rocsparse_spic0_descr();

    const double*          get_tolerance_pointer() const;
    rocsparse_pointer_mode get_tolerance_pointer_mode() const;

    void set_tolerance_pointer(const void*);
    void set_tolerance_pointer_mode(rocsparse_pointer_mode);

    rocsparse_spic0_stage get_stage() const;
    rocsparse_spic0_alg   get_alg() const;
    rocsparse_datatype    get_compute_datatype() const;
    void                  set_stage(rocsparse_spic0_stage value);
    void                  set_alg(rocsparse_spic0_alg value);
    void                  set_compute_datatype(rocsparse_datatype value);

    rocsparse_analysis_policy get_analysis_policy() const;
    void                      set_analysis_policy(rocsparse_analysis_policy value);

    rocsparse_csric0_info get_csric0_info();
    void                  set_csric0_info(rocsparse_csric0_info value);
    void                  set_shared_csric0_info(std::shared_ptr<_rocsparse_csric0_info> value);

    rocsparse_bsric0_info get_bsric0_info();
    void                  set_bsric0_info(rocsparse_bsric0_info value);
    void                  set_shared_bsric0_info(std::shared_ptr<_rocsparse_bsric0_info> value);

    rocsparse_format get_format() const;
    void             set_format(rocsparse_format format);
};
