/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef ROCSPARSE_CHECK_HPP
#define ROCSPARSE_CHECK_HPP

#include "rocsparse_math.hpp"
#include "rocsparse_traits.hpp"

template <typename T>
struct default_tolerance;

template <>
struct default_tolerance<int32_t>
{
    static constexpr int32_t value = 0;
};

template <>
struct default_tolerance<float>
{
    static constexpr float value = 1.0e-3f;
};

template <>
struct default_tolerance<double>
{
    static constexpr double value = 1.0e-10;
};

template <>
struct default_tolerance<rocsparse_float_complex>
{
    static constexpr float value = default_tolerance<float>::value;
};

template <>
struct default_tolerance<rocsparse_double_complex>
{
    static constexpr double value = default_tolerance<double>::value;
};

template <typename T>
void unit_check_general(int64_t m, int64_t n, const T* a, int64_t lda, const T* b, int64_t ldb);

template <typename T>
void unit_check_enum(const T a, const T b);

template <typename T>
inline void unit_check_scalar(const T a, const T b)
{
    unit_check_general(1, 1, &a, 1, &b, 1);
}

template <typename T>
inline void unit_check_segments(size_t n, const T* a, const T* b)
{
    unit_check_general(1, n, a, 1, b, 1);
}

template <typename T>
void near_check_general(int64_t            m,
                        int64_t            n,
                        const T*           a,
                        int64_t            lda,
                        const T*           b,
                        int64_t            ldb,
                        floating_data_t<T> tol = default_tolerance<T>::value);

template <typename T>
inline void
    near_check_scalar(const T* a, const T* b, floating_data_t<T> tol = default_tolerance<T>::value)
{
    near_check_general(1, 1, a, 1, b, 1, tol);
}

template <typename T>
inline void near_check_segments(int64_t            n,
                                const T*           a,
                                const T*           b,
                                floating_data_t<T> tol = default_tolerance<T>::value)
{
    near_check_general(1, n, a, 1, b, 1, tol);
}

void unit_check_garray(rocsparse_datatype val_type,
                       int64_t            size,
                       const void*        source,
                       const void*        target);
void unit_check_garray(rocsparse_indextype ind_type,
                       int64_t             size,
                       const void*         source,
                       const void*         target);

#endif // ROCSPARSE_CHECK_HPP
