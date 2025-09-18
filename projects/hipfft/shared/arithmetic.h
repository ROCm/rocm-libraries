/******************************************************************************
* Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#pragma once

#include <numeric>
#include <stddef.h>
#include <tuple>

// arithmetic helper functions

static inline bool IsPo2(size_t u)
{
    return (u != 0) && (0 == (u & (u - 1)));
}

//	help function: Find the smallest power of 2 that is >= n; return its
//  power of 2 factor
//	e.g., CeilPo2 (7) returns 3 : (2^3 >= 7)
static inline size_t CeilPo2(size_t n)
{
    size_t v = 1, t = 0;
    while(v < n)
    {
        v <<= 1;
        t++;
    }

    return t;
}

template <typename T>
static inline T DivRoundingUp(T a, T b)
{
    return (a + (b - 1)) / b;
}

template <typename Titer>
typename Titer::value_type product(Titer begin, Titer end)
{
    return std::accumulate(
        begin, end, typename Titer::value_type(1), std::multiplies<typename Titer::value_type>());
}

// count the number of total iterations for 1-, 2-, and 3-D dimensions
template <typename T1>
static inline size_t count_iters(const T1& i)
{
    return i;
}
template <typename T1>
static inline size_t count_iters(const std::tuple<T1, T1>& i)
{
    return std::get<0>(i) * std::get<1>(i);
}
template <typename T1>
static inline size_t count_iters(const std::tuple<T1, T1, T1>& i)
{
    return std::get<0>(i) * std::get<1>(i) * std::get<2>(i);
}
