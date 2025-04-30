/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2023 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

/*!\file
 * \brief provides Floating point counts of Basic Linear Algebra Subprograms (BLAS) of Level 1, 2,
 * 3. Where possible we are using the values of NOP from the legacy BLAS files [sdcz]blas[23]time.f
 * for flop count.
 */

template <typename T>
constexpr double prune_strip_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 4.0 * 6 * 3 / 1e9;
}

template <typename T>
constexpr double prune_tile_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 16.0 * 90 * (8 + 7) / 1e9;
}

/* \brief floating point counts of GEMM */
template <typename T>
constexpr double gemm_gflop_count(int64_t m, int64_t n, int64_t k)
{
    return (2.0 * m * n * k) / 1e9;
}

template <typename T>
constexpr double relu_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 1e9;
}

template <typename T>
constexpr double clippedrelu_gflop_count(int64_t m, int64_t n)
{
    return 2 * (m * n) / 1e9;
}

template <typename T>
constexpr double gelu_gflop_count(int64_t m, int64_t n, bool scaling = false)
{
    int ops = scaling ? 10.0f : 9.0f;
    return (ops * m * n) / 1e9;
}

template <typename T>
constexpr double abs_gflop_count(int64_t m, int64_t n)
{
    return (m * n) / 1e9;
}

template <typename T>
constexpr double leakyrelu_gflop_count(int64_t m, int64_t n)
{
    return 2 * (m * n) / 1e9;
}

template <typename T>
constexpr double sigmoid_gflop_count(int64_t m, int64_t n)
{
    return 3 * (m * n) / 1e9;
}

template <typename T>
constexpr double tanh_gflop_count(int64_t m, int64_t n)
{
    return 3 * (m * n) / 1e9;
}
