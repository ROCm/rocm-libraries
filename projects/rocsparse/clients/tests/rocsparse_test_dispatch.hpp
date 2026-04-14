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

#include "rocsparse_test.hpp"
#include "rocsparse_test_dispatch_enum.hpp"
#include "type_dispatch.hpp"

template <rocsparse_test_dispatch_enum::value_type TYPE_DISPATCH = rocsparse_test_dispatch_enum::t>
struct rocsparse_test_dispatch;

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::t>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_simple_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::it>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_it_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::it_sparse_to_dense>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_it_sparse_to_dense_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::it_dense_to_sparse>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_it_dense_to_sparse_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::it_gather_scatter>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_it_gather_scatter_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ijt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ijt_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ijt_sparse_to_dense>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ijt_sparse_to_dense_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ijt_dense_to_sparse>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ijt_dense_to_sparse_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ixyt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ixyt_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ixyt_axpby>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ixyt_axpby_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::iaxyt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_iaxyt_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ijaxyt>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ijaxyt_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::iabct>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_iabct_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ijabct>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ijabct_dispatch<TEST>(arg);
    }
};

template <>
struct rocsparse_test_dispatch<rocsparse_test_dispatch_enum::ijabct_sddmm>
{
    template <template <typename...> class TEST>
    static auto dispatch(const Arguments& arg)
    {
        return rocsparse_ijabct_sddmm_dispatch<TEST>(arg);
    }
};
