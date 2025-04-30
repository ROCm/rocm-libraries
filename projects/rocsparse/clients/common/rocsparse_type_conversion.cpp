/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_type_conversion.hpp"
#include <iostream>
template <>
rocsparse_status rocsparse_type_conversion(const size_t& x, size_t& y)
{
    y = x;
    return rocsparse_status_success;
}

template <>
rocsparse_status rocsparse_type_conversion(const int32_t& x, int32_t& y)
{
    y = x;
    return rocsparse_status_success;
}

template <>
rocsparse_status rocsparse_type_conversion(const int64_t& x, int64_t& y)
{
    y = x;
    return rocsparse_status_success;
}

template <>
rocsparse_status rocsparse_type_conversion(const int32_t& x, int64_t& y)
{
    y = x;
    return rocsparse_status_success;
}

template <>
rocsparse_status rocsparse_type_conversion(const int64_t& x, size_t& y)
{
    if(x < 0)
    {
        std::cerr << "corrupted conversion from int64_t to size_t." << std::endl;
        return rocsparse_status_invalid_value;
    }
    else
    {
        y = static_cast<size_t>(x);
        return rocsparse_status_success;
    }
}

template <>
rocsparse_status rocsparse_type_conversion(const int32_t& x, size_t& y)
{
    if(x < 0)
    {
        std::cerr << "corrupted conversion from int32_t to size_t." << std::endl;
        return rocsparse_status_invalid_value;
    }
    else
    {
        y = static_cast<size_t>(x);
        return rocsparse_status_success;
    }
}

template <>
rocsparse_status rocsparse_type_conversion(const int64_t& x, int32_t& y)
{
    static constexpr int32_t int32max = std::numeric_limits<int32_t>::max();
    if(x > int32max)
    {
        std::cerr << "corrupted conversion from int64_t to int32_t." << std::endl;
        return rocsparse_status_invalid_value;
    }
    static constexpr int32_t int32min = std::numeric_limits<int32_t>::min();
    if(x < int32min)
    {
        std::cerr << "corrupted conversion from int64_t to int32_t." << std::endl;
        return rocsparse_status_invalid_value;
    }
    y = static_cast<int32_t>(x);
    return rocsparse_status_success;
}

template <>
rocsparse_status rocsparse_type_conversion(const size_t& x, int32_t& y)
{
    static constexpr int32_t int32max = std::numeric_limits<int32_t>::max();
    if(x > int32max)
    {
        std::cerr << "corrupted conversion from size_t to int32_t." << std::endl;
        return rocsparse_status_invalid_value;
    }
    y = static_cast<int32_t>(x);
    return rocsparse_status_success;
}

template <>
rocsparse_status rocsparse_type_conversion(const size_t& x, int64_t& y)
{
    static constexpr int64_t int64max = std::numeric_limits<int64_t>::max();
    if(x > int64max)
    {
        std::cerr << "corrupted conversion from size_t to int64_t." << std::endl;
        return rocsparse_status_invalid_value;
    }
    y = static_cast<int64_t>(x);
    return rocsparse_status_success;
}

template <>
rocsparse_status rocsparse_type_conversion(const float& x, double& y)
{
    y = static_cast<double>(x);
    return rocsparse_status_success;
}
