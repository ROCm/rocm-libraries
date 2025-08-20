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

#include "rocsparse_test_traits.hpp"

template <rocsparse_test_enum::value_type ROUTINE>
struct rocsparse_test_check
{
private:
    template <typename T>
    static inline constexpr bool is_valid_type()
    {
        switch(rocsparse_test_traits<ROUTINE>::s_numeric_types)
        {
        case rocsparse_test_numeric_types_enum::all:
        {
            return std::is_same<T, int8_t>{} || std::is_same<T, _Float16>{}
                   || std::is_same<T, rocsparse_bfloat16>{} || std::is_same<T, float>{}
                   || std::is_same<T, double>{} || std::is_same<T, rocsparse_float_complex>{}
                   || std::is_same<T, rocsparse_double_complex>{};
        }
        case rocsparse_test_numeric_types_enum::real_only:
        {
            return std::is_same<T, int8_t>{} || std::is_same<T, _Float16>{}
                   || std::is_same<T, rocsparse_bfloat16>{} || std::is_same<T, float>{}
                   || std::is_same<T, double>{};
        }
        case rocsparse_test_numeric_types_enum::complex_only:
        {
            return std::is_same<T, rocsparse_float_complex>{}
                   || std::is_same<T, rocsparse_double_complex>{};
        }
        }
        return false;
    };

    template <typename T, typename... P>
    static inline constexpr bool is_valid_type_list_check()
    {
        constexpr std::size_t n = sizeof...(P);
        if(n == 0)
        {
            //
            // last type name.
            //
            return is_valid_type<T>();
        }
        else
        {
            if(!std::is_same<T, int32_t>{} && !std::is_same<T, int64_t>{})
            {
                return false;
            }
            return is_valid_type_list<P...>();
        }
    }

    template <typename... Targs>
    static inline constexpr bool is_valid_type_list()
    {
        return is_valid_type_list_check<Targs...>();
    }

    template <>
    inline constexpr bool is_valid_type_list<>()
    {
        return false;
    }

public:
    template <typename... P>
    static constexpr bool is_type_valid()
    {
        return is_valid_type_list<P...>();
    }
};
