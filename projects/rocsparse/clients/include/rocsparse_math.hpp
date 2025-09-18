/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_MATH_HPP
#define ROCSPARSE_MATH_HPP

#include "rocsparse.h"
#include <cmath>

#include "rocsparse_traits.hpp"

/* =================================================================================== */
/*! \brief  returns true if value is NaN */
template <typename T>
inline bool rocsparse_isnan(T arg);

template <>
inline bool rocsparse_isnan(int8_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(uint8_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(uint32_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(int arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(int64_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(uint64_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isnan(_Float16 arg)
{
    return arg != arg;
}

template <>
inline bool rocsparse_isnan(rocsparse_bfloat16 arg)
{
    return arg != arg;
}

template <>
inline bool rocsparse_isnan(double arg)
{
    return std::isnan(arg);
}
template <>
inline bool rocsparse_isnan(float arg)
{
    return std::isnan(arg);
}

template <>
inline bool rocsparse_isnan(rocsparse_float_complex arg)
{
    return std::isnan(std::real(arg)) || std::isnan(std::imag(arg));
}

template <>
inline bool rocsparse_isnan(rocsparse_double_complex arg)
{
    return std::isnan(std::real(arg)) || std::isnan(std::imag(arg));
}

/* =================================================================================== */
/*! \brief  returns true if value is inf */
template <typename T>
inline bool rocsparse_isinf(T arg);

template <>
inline bool rocsparse_isinf(int8_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isinf(int arg)
{
    return false;
}

template <>
inline bool rocsparse_isinf(int64_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isinf(uint64_t arg)
{
    return false;
}

template <>
inline bool rocsparse_isinf(_Float16 arg)
{
    union
    {
        _Float16 fp;
        uint16_t data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) == 0;
}

template <>
inline bool rocsparse_isinf(float arg)
{
    return std::isinf(arg);
}

template <>
inline bool rocsparse_isinf(double arg)
{
    return std::isinf(arg);
}

template <>
inline bool rocsparse_isinf(rocsparse_float_complex arg)
{
    return std::isinf(std::real(arg)) || std::isinf(std::imag(arg));
}

template <>
inline bool rocsparse_isinf(rocsparse_double_complex arg)
{
    return std::isinf(std::real(arg)) || std::isinf(std::imag(arg));
}

/* =================================================================================== */
/*! \brief  returns complex conjugate */
template <typename T>
inline T rocsparse_conj(T arg)
{
    return arg;
}

template <>
inline rocsparse_float_complex rocsparse_conj(rocsparse_float_complex arg)
{
    return std::conj(arg);
}

template <>
inline rocsparse_double_complex rocsparse_conj(rocsparse_double_complex arg)
{
    return std::conj(arg);
}

/* =================================================================================== */
/*! \brief  absolute value */
template <typename T>
inline floating_data_t<T> rocsparse_abs(T arg)
{
    return std::abs(arg);
}

template <>
inline _Float16 rocsparse_abs(_Float16 arg)
{
    return (arg < 0.0) ? -arg : arg;
}

#endif // ROCSPARSE_MATH_HPP
