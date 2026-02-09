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

#include "hipsparse_test_traits.hpp"

template <hipsparse_test_enum::value_type ROUTINE>
struct hipsparse_test_check
{
public:
    template <typename T>
    static constexpr bool is_valid_type()
    {
        switch(hipsparse_test_traits<ROUTINE>::s_numeric_types)
        {
        case hipsparse_test_numeric_types_enum::all:
        {
            return std::is_same<T, float>{} || std::is_same<T, double>{}
                   || std::is_same<T, hipComplex>{} || std::is_same<T, hipDoubleComplex>{};
        }
        case hipsparse_test_numeric_types_enum::real_only:
        {
            return std::is_same<T, float>{} || std::is_same<T, double>{};
        }
        case hipsparse_test_numeric_types_enum::complex_only:
        {
            return std::is_same<T, hipComplex>{} || std::is_same<T, hipDoubleComplex>{};
        }
        }
        return false;
    };

    template <typename I, typename T>
    static constexpr bool is_valid_type()
    {
        return std::is_integral<I>::value && is_valid_type<T>();
    };

    template <typename I, typename J, typename T>
    static constexpr bool is_valid_type()
    {
        return std::is_integral<I>::value && std::is_integral<J>::value && is_valid_type<T>();
    };
};
