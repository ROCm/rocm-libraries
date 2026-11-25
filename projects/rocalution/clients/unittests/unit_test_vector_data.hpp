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

// Values and references of the vector tests, at the backend and the Local level

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "unit_test_types.hpp"

namespace rocalution_unit
{
    template <typename T>
    std::vector<T> x()
    {
        return from_ints<T>({3, -1, 4, 1, -5, 9, 2, -6, 5, 3, -5, 8});
    }

    template <typename T>
    std::vector<T> y()
    {
        return from_ints<T>({2, 7, -1, 8, 2, -8, 1, 8, -2, 8, 4, -5});
    }

    template <typename T, typename F>
    std::vector<T> elementwise(const std::vector<T>& a, const std::vector<T>& b, F f)
    {
        std::vector<T> r(a.size());
        for(size_t i = 0; i < a.size(); ++i)
        {
            r[i] = f(a[i], b[i]);
        }
        return r;
    }

    // As std::abs, but of type T
    template <typename T>
    T magnitude(const T& v)
    {
        if constexpr(is_complex_v<T>)
            return T(std::abs(v));
        else
            return std::abs(v);
    }

    // Host arrays, which std::vector<bool> does not provide
    template <typename T>
    std::unique_ptr<T[]> array_of(const std::vector<T>& v)
    {
        std::unique_ptr<T[]> a(new T[v.size()]);
        std::copy(v.begin(), v.end(), a.get());
        return a;
    }
}
