/*! \file */
/* ************************************************************************
* Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_test_dispatch_enum.hpp"
#include "rocsparse_test_numeric_types_enum.hpp"

template <rocsparse_test_dispatch_enum::value_type      DISPATCH,
          rocsparse_test_numeric_types_enum::value_type NUMERIC_TYPES>
struct rocsparse_test_config_template
{
    static constexpr rocsparse_test_dispatch_enum::value_type      s_dispatch      = DISPATCH;
    static constexpr rocsparse_test_numeric_types_enum::value_type s_numeric_types = NUMERIC_TYPES;
};

struct rocsparse_test_config
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::t,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_real_only
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::t,
                                     rocsparse_test_numeric_types_enum::real_only>
{
};

struct rocsparse_test_config_complex_only
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::t,
                                     rocsparse_test_numeric_types_enum::complex_only>
{
};

struct rocsparse_test_config_it
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::it,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_it_sparse_to_dense
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::it_sparse_to_dense,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_it_dense_to_sparse
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::it_dense_to_sparse,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_it_real_only
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::it,
                                     rocsparse_test_numeric_types_enum::real_only>
{
};

struct rocsparse_test_config_it_complex_only
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::it,
                                     rocsparse_test_numeric_types_enum::complex_only>
{
};

struct rocsparse_test_config_it_gather_scatter
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::it_gather_scatter,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_ijt
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ijt,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_ijt_sparse_to_dense
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ijt_sparse_to_dense,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_ijt_dense_to_sparse
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ijt_dense_to_sparse,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_ijt_real_only
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ijt,
                                     rocsparse_test_numeric_types_enum::real_only>
{
};

struct rocsparse_test_config_ijt_complex_only
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ijt,
                                     rocsparse_test_numeric_types_enum::complex_only>
{
};

struct rocsparse_test_config_ixyt
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ixyt,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_ixyt_axpby
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ixyt_axpby,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_iaxyt
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::iaxyt,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_ijaxyt
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ijaxyt,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_iabct
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::iabct,
                                     rocsparse_test_numeric_types_enum::all>
{
};
struct rocsparse_test_config_ijabct
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ijabct,
                                     rocsparse_test_numeric_types_enum::all>
{
};

struct rocsparse_test_config_ijabct_sddmm
    : rocsparse_test_config_template<rocsparse_test_dispatch_enum::ijabct_sddmm,
                                     rocsparse_test_numeric_types_enum::all>
{
};
