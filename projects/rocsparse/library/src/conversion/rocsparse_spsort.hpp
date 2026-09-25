/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include "internal/generic/rocsparse_spsort.h"
#include "rocsparse_enum_utils.hpp"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_alg value_)
{
    switch(value_)
    {
    case rocsparse_spsort_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_input value_)
{
    switch(value_)
    {
    case rocsparse_spsort_input_alg:
    case rocsparse_spsort_input_direction:
    {
        return false;
    }
    }
    return true;
}

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_spsort_stage value_)
{
    switch(value_)
    {
    case rocsparse_spsort_stage_analysis:
    case rocsparse_spsort_stage_compute:
    {
        return false;
    }
    }
    return true;
}

struct _rocsparse_spsort_descr
{
protected:
    rocsparse_spsort_stage m_stage;
    rocsparse_spsort_alg   m_alg;
    rocsparse_direction    m_dir;

public:
    ~_rocsparse_spsort_descr();
    _rocsparse_spsort_descr();

    rocsparse_spsort_stage get_stage() const;
    rocsparse_spsort_alg   get_alg() const;
    rocsparse_direction    get_dir() const;
    void                   set_stage(rocsparse_spsort_stage value);
    void                   set_alg(rocsparse_spsort_alg value);
    void                   set_dir(rocsparse_direction value);
};

namespace rocsparse
{
    // Validates the descriptor and the matrices shared by rocsparse_spsort_buffer_size and
    // rocsparse_spsort.
    rocsparse_status spsort_check_arguments(rocsparse_spsort_descr      descr,
                                            rocsparse_const_spmat_descr source,
                                            rocsparse_const_spmat_descr target);
}
