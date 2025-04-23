/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#ifndef ROCALUTION_UTILS_TYPE_TRAITS_HPP_
#define ROCALUTION_UTILS_TYPE_TRAITS_HPP_

#include <complex>

namespace rocalution
{
    //
    // Convenience traits.
    //
    template <typename T>
    struct numeric_traits
    {
        using value_type = T;
    };

    template <>
    struct numeric_traits<std::complex<float>>
    {
        using value_type = float;
    };

    template <>
    struct numeric_traits<std::complex<double>>
    {
        using value_type = double;
    };

    template <>
    struct numeric_traits<float>
    {
        using value_type = float;
    };
    template <>
    struct numeric_traits<double>
    {
        using value_type = double;
    };

    template <typename T>
    using numeric_traits_t = typename numeric_traits<T>::value_type;
}

#endif // ROCALUTION_UTILS_TYPE_TRAITS_HPP_
