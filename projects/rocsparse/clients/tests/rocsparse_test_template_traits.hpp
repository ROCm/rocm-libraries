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

#include "rocsparse_test_template.hpp"

template <rocsparse_test_enum::value_type          ROUTINE,
          rocsparse_test_dispatch_enum::value_type DISPATCH>
struct rocsparse_test_template_traits;

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::t>
{
    using filter = typename rocsparse_test_t_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_t_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::it>
{
    using filter = typename rocsparse_test_it_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_it_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::it_sparse_to_dense>
{
    using filter = typename rocsparse_test_it_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_it_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::it_dense_to_sparse>
{
    using filter = typename rocsparse_test_it_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_it_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::it_gather_scatter>
{
    using filter = typename rocsparse_test_it_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_it_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::ijt>
{
    using filter = typename rocsparse_test_ijt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_ijt_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::ijt_sparse_to_dense>
{
    using filter = typename rocsparse_test_ijt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_ijt_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::ijt_dense_to_sparse>
{
    using filter = typename rocsparse_test_ijt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_ijt_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::ixyt>
{
    using filter = typename rocsparse_test_ixyt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_ixyt_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::ixyt_axpby>
{
    using filter = typename rocsparse_test_ixyt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_ixyt_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::iaxyt>
{
    using filter = typename rocsparse_test_iaxyt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_iaxyt_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::ijaxyt>
{
    using filter = typename rocsparse_test_ijaxyt_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_ijaxyt_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::iabct>
{
    using filter = typename rocsparse_test_iabct_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_iabct_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::ijabct>
{
    using filter = typename rocsparse_test_ijabct_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_ijabct_template<ROUTINE>::template test_call<P...>;
};

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_template_traits<ROUTINE, rocsparse_test_dispatch_enum::ijabct_sddmm>
{
    using filter = typename rocsparse_test_ijabct_template<ROUTINE>::test;
    template <typename... P>
    using caller = typename rocsparse_test_ijabct_template<ROUTINE>::template test_call<P...>;
};
