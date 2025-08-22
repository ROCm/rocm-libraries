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
#include "rocsparse_datatype_utils.hpp"

template <>
rocsparse_datatype rocsparse::get_datatype<_Float16>()
{
    return rocsparse_datatype_f16_r;
}

template <>
rocsparse_datatype rocsparse::get_datatype<rocsparse_bfloat16>()
{
    return rocsparse_datatype_bf16_r;
}

template <>
rocsparse_datatype rocsparse::get_datatype<float>()
{
    return rocsparse_datatype_f32_r;
}

template <>
rocsparse_datatype rocsparse::get_datatype<double>()
{
    return rocsparse_datatype_f64_r;
}

template <>
rocsparse_datatype rocsparse::get_datatype<rocsparse_float_complex>()
{
    return rocsparse_datatype_f32_c;
}

template <>
rocsparse_datatype rocsparse::get_datatype<rocsparse_double_complex>()
{
    return rocsparse_datatype_f64_c;
}

size_t rocsparse::datatype_sizeof(rocsparse_datatype that)
{
    switch(that)
    {
    case rocsparse_datatype_i32_r:
    {
        return sizeof(int32_t);
    }

    case rocsparse_datatype_u32_r:
    {
        return sizeof(uint32_t);
    }

    case rocsparse_datatype_i8_r:
    {
        return sizeof(int8_t);
    }

    case rocsparse_datatype_u8_r:
    {
        return sizeof(uint8_t);
    }

    case rocsparse_datatype_f16_r:
    {
        return sizeof(_Float16);
    }

    case rocsparse_datatype_bf16_r:
    {
        return sizeof(rocsparse_bfloat16);
    }

    case rocsparse_datatype_f32_r:
    {
        return sizeof(float);
    }

    case rocsparse_datatype_f64_r:
    {
        return sizeof(double);
    }

    case rocsparse_datatype_f32_c:
    {
        return sizeof(rocsparse_float_complex);
    }

    case rocsparse_datatype_f64_c:
    {
        return sizeof(rocsparse_double_complex);
    }
    }
}
