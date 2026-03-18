/*! \file */
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

#pragma once

#include "rocsparse-types.h"

namespace rocsparse
{
    template <rocsparse_datatype v>
    struct datatype_traits;

    template <>
    struct datatype_traits<rocsparse_datatype_f16_r>
    {
        using type_t = _Float16;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_bf16_r>
    {
        using type_t = rocsparse_bfloat16;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_f32_r>
    {
        using type_t = float;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_f64_r>
    {
        using type_t = double;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_f32_c>
    {
        using type_t = rocsparse_float_complex;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_f64_c>
    {
        using type_t = rocsparse_double_complex;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_u32_r>
    {
        using type_t = uint32_t;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_i32_r>
    {
        using type_t = int32_t;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_u8_r>
    {
        using type_t = uint8_t;
    };

    template <>
    struct datatype_traits<rocsparse_datatype_i8_r>
    {
        using type_t = int8_t;
    };

    template <typename T>
    rocsparse_datatype get_datatype();

    size_t datatype_sizeof(rocsparse_datatype that);
}
