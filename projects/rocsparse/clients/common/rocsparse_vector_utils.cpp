/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_vector_utils.hpp"

#include <algorithm>

/*! \brief Perform max norm on an array of complex values. */
template <typename T,
          typename std::enable_if<(std::is_same<T, rocsparse_float_complex>::value
                                   || std::is_same<T, rocsparse_double_complex>::value),
                                  int>::type
          = 0>
static void normalize_array(T* v, size_t v_size)
{
    if(v_size > 0)
    {
        auto max_val = std::abs(v[0]);

        for(size_t i = 1; i < v_size; i++)
        {
            max_val = std::max(std::abs(v[i]), max_val);
        }

        auto factor = static_cast<floating_data_t<T>>(1) / max_val;

        for(size_t i = 0; i < v_size; i++)
        {
            v[i] *= factor;
        }
    }
}

/*! \brief Perform affine transform of bound [-1, 1] on an array of real values. */
template <typename T,
          typename std::enable_if<(std::is_same<T, float>::value || std::is_same<T, double>::value),
                                  int>::type
          = 0>
static void normalize_array(T* v, size_t v_size)
{
    if(v_size > 0)
    {
        auto max_val = v[0];
        auto min_val = v[0];

        for(size_t i = 1; i < v_size; i++)
        {
            max_val = std::max(v[i], max_val);
            min_val = std::min(v[i], min_val);
        }

        if(std::abs(max_val - min_val) > std::numeric_limits<T>::epsilon() * max_val)
        {
            // y = (2x - max - min) / (max - min)
            auto denom = static_cast<T>(1) / (max_val - min_val);

            for(size_t i = 0; i < v_size; i++)
            {
                v[i] = (2 * v[i] - max_val - min_val) * denom;
            }
        }
        else
        {
            auto denom = static_cast<T>(1) / std::max(std::abs(max_val), std::abs(min_val));

            for(size_t i = 0; i < v_size; i++)
            {
                v[i] *= denom;
            }
        }
    }
}

template <typename T>
void rocsparse_vector_utils<T>::normalize(host_dense_vector<T>& v)
{
    normalize_array<T>(v.data(), v.size());
}

template <typename T>
void rocsparse_vector_utils<T>::normalize(host_vector<T>& v)
{
    normalize_array<T>(v.data(), v.size());
}

#define INSTANTIATE(TYPE) template struct rocsparse_vector_utils<TYPE>;

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE
