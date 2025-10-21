/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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
#ifndef GUARD_GEMM_HPP
#define GUARD_GEMM_HPP

#include "ford.hpp"
#include <iostream>

/*
    A and B rows and cols should be passed as default values (NxM, MxK), independently of
   a_transponse/b_transpose flag value
    C rows and cols should have correct values based on a_transponse/b_transpose values
    A, B, C strides should have corret values based on a_transponse/b_transpose values
*/
template <typename Dtype>
void gemm_cpu(const Dtype* a_ptr,
              const size_t a_cols,
              const size_t a_rows,
              const size_t a_stride,
              const bool a_transpose,
              const Dtype* b_ptr,
              const size_t b_cols,
              const size_t b_rows,
              const size_t b_stride,
              const bool b_transpose,
              Dtype* c_ptr,
              const size_t c_cols,
              const size_t c_rows,
              const size_t c_stride,
              double d_alpha = 1.0,
              double d_beta  = 1.0)
{
    Dtype alpha = static_cast<Dtype>(d_alpha);
    Dtype beta  = static_cast<Dtype>(d_beta);
    if((!a_transpose && !b_transpose &&
        ((a_cols != b_rows) || (a_rows != c_rows) || (b_cols != c_cols))) ||
       (a_transpose && b_transpose &&
        ((a_rows != b_cols) || (a_cols != c_rows) || (b_rows != c_cols))) ||
       (a_transpose && !b_transpose &&
        ((a_rows != b_rows) || (a_cols != c_rows) || (b_cols != c_cols))) ||
       (!a_transpose && b_transpose &&
        ((a_cols != b_cols) || (a_rows != c_rows) || (b_rows != c_cols))))
    {

        std::cout << "MM_CPU_ERROR. Incompatible matrix size: " << std::endl
                  << "A: " << a_rows << "x" << a_cols << " transpose: " << a_transpose << std::endl
                  << "B: " << b_rows << "x" << b_cols << " transpose: " << b_transpose << std::endl
                  << "C: " << c_rows << "x" << c_cols << std::endl;
        return;
    }

    size_t inner_loop_limit = a_transpose ? a_rows : a_cols;
    auto inner_loop         = [&](int n, int k) {
        Dtype el = static_cast<Dtype>(0.0);
        ford(inner_loop_limit)(
            [&](int m) { el += a_ptr[n * a_stride + m] * b_ptr[m * b_stride + k]; });
        c_ptr[n * c_stride + k] = beta * c_ptr[n * c_stride + k] + alpha * el;
    };

    constexpr size_t iter_margin = 1'048'576;
    if(c_rows * c_cols * inner_loop_limit > iter_margin)
    {
        par_ford(c_rows, c_cols)(inner_loop);
    }
    else
    {
        ford(c_rows, c_cols)(inner_loop);
    }
}

#endif
