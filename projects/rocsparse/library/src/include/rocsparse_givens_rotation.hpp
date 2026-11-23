/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.hpp"

#include <type_traits>

// Division without squaring the denominator is rocsparse::scaled_div, in rocsparse_common.hpp.

namespace rocsparse
{
    // cos = conj(f) / hypot(|f|, |g|) and sin = conj(g) / hypot(|f|, |g|), for real and complex T.
    // Divide by the largest component first so the hypotenuse lies in [1, 2].
    // Forming |z|^2 directly overflows or underflows when f and g are scaled far from 1.
    // If f and g are both zero there is nothing to eliminate: cos = 1, sin = 0.
    template <typename T>
    __device__ __forceinline__ void givens_rotation(T f, T g, T& cos_theta, T& sin_theta)
    {
        if constexpr(std::is_same<T, float>() || std::is_same<T, double>())
        {
            T scale = rocsparse::max(rocsparse::abs(f), rocsparse::abs(g));

            if(scale == static_cast<T>(0))
            {
                cos_theta = static_cast<T>(1);
                sin_theta = static_cast<T>(0);
                return;
            }

            f /= scale;
            g /= scale;

            T hypotenuse = rocsparse::sqrt(rocsparse::fma(f, f, g * g));

            cos_theta = f / hypotenuse;
            sin_theta = g / hypotenuse;
        }
        else
        {
            auto fr = rocsparse::real(f);
            auto fi = rocsparse::imag(f);
            auto gr = rocsparse::real(g);
            auto gi = rocsparse::imag(g);

            using real_t = decltype(fr);

            real_t scale = rocsparse::max(rocsparse::max(rocsparse::abs(fr), rocsparse::abs(fi)),
                                          rocsparse::max(rocsparse::abs(gr), rocsparse::abs(gi)));

            if(scale == static_cast<real_t>(0))
            {
                cos_theta = static_cast<T>(1);
                sin_theta = static_cast<T>(0);
                return;
            }

            fr /= scale;
            fi /= scale;
            gr /= scale;
            gi /= scale;

            real_t hypotenuse = rocsparse::sqrt(rocsparse::fma(fr, fr, fi * fi)
                                                + rocsparse::fma(gr, gr, gi * gi));
            real_t inv        = static_cast<real_t>(1) / hypotenuse;

            // Divide the components by the real hypotenuse. Complex operator/ squares its denominator.
            cos_theta = T(fr * inv, -fi * inv);
            sin_theta = T(gr * inv, -gi * inv);
        }
    }
}
