/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * Copyright (c) 2012-, King Abdullah University of Science and Technology
 * All rights reserved.

 * Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ************************************************************************ */

#pragma once

#include "device_macros.hpp"
// uses dot shuffle reductions
#include "../blas1/rocblas_reduction.hpp"
// uses recursive folding reduction
#include "../blas1/reduction.hpp"

// Specialized pipelined kernel for float
template <bool CONJ, int TILE_DIM_X, int TILE_DIM_Y>
ROCBLAS_KERNEL_ILF void rocblas_gemvt_row_vectorized_kernel_calc(rocblas_int m,
                                                                 rocblas_int n,
                                                                 float       alpha,
                                                                 const float* __restrict__ A,
                                                                 rocblas_stride lda,
                                                                 const float* __restrict__ x,
                                                                 rocblas_int incx,
                                                                 float       beta,
                                                                 float* __restrict__ y,
                                                                 rocblas_int incy)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int y_col = blockIdx.x * TILE_DIM_Y + ty;

    if(y_col >= n)
        return;

    if(alpha == 0.0f)
    {
        if(tx == 0)
        {
            y[y_col * int64_t(incy)] = (beta == 0.0f) ? 0.0f : y[y_col * int64_t(incy)] * beta;
        }
        return;
    }

    float psum = 0.0f;

    const float* current_A = A + y_col * lda;

    const int stride   = TILE_DIM_X * 8;
    int       row_base = tx * 8;

    float4 a_vec_k0, x_vec_k0, a_vec_k1, x_vec_k1;
    float4 a_vec_n0, x_vec_n0, a_vec_n1, x_vec_n1;

    if(row_base < m)
    {
        if(row_base + 7 < m && incx == 1)
        {
            a_vec_k0 = *reinterpret_cast<const float4*>(current_A + row_base);
            x_vec_k0 = *reinterpret_cast<const float4*>(x + row_base);
            a_vec_k1 = *reinterpret_cast<const float4*>(current_A + row_base + 4);
            x_vec_k1 = *reinterpret_cast<const float4*>(x + row_base + 4);
        }
        else
        {
            for(int i = 0; i < 4; ++i)
            {
                if(row_base + i < m)
                {
                    ((float*)&a_vec_k0)[i] = current_A[row_base + i];
                    ((float*)&x_vec_k0)[i] = x[(row_base + i) * int64_t(incx)];
                }
                else
                {
                    ((float*)&a_vec_k0)[i] = 0.0f;
                    ((float*)&x_vec_k0)[i] = 0.0f;
                }
                if(row_base + 4 + i < m)
                {
                    ((float*)&a_vec_k1)[i] = current_A[row_base + 4 + i];
                    ((float*)&x_vec_k1)[i] = x[(row_base + 4 + i) * int64_t(incx)];
                }
                else
                {
                    ((float*)&a_vec_k1)[i] = 0.0f;
                    ((float*)&x_vec_k1)[i] = 0.0f;
                }
            }
        }
    }
    else
    {
        a_vec_k0 = a_vec_k1 = x_vec_k0 = x_vec_k1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for(row_base += stride; row_base < m; row_base += stride)
    {
        if(row_base + 7 < m && incx == 1)
        {
            a_vec_n0 = *reinterpret_cast<const float4*>(current_A + row_base);
            x_vec_n0 = *reinterpret_cast<const float4*>(x + row_base);
            a_vec_n1 = *reinterpret_cast<const float4*>(current_A + row_base + 4);
            x_vec_n1 = *reinterpret_cast<const float4*>(x + row_base + 4);
        }
        else
        {
            for(int i = 0; i < 4; ++i)
            {
                if(row_base + i < m)
                {
                    ((float*)&a_vec_n0)[i] = current_A[row_base + i];
                    ((float*)&x_vec_n0)[i] = x[(row_base + i) * int64_t(incx)];
                }
                else
                {
                    ((float*)&a_vec_n0)[i] = 0.0f;
                    ((float*)&x_vec_n0)[i] = 0.0f;
                }
                if(row_base + 4 + i < m)
                {
                    ((float*)&a_vec_n1)[i] = current_A[row_base + 4 + i];
                    ((float*)&x_vec_n1)[i] = x[(row_base + 4 + i) * int64_t(incx)];
                }
                else
                {
                    ((float*)&a_vec_n1)[i] = 0.0f;
                    ((float*)&x_vec_n1)[i] = 0.0f;
                }
            }
        }

        psum += a_vec_k0.x * x_vec_k0.x;
        psum += a_vec_k0.y * x_vec_k0.y;
        psum += a_vec_k0.z * x_vec_k0.z;
        psum += a_vec_k0.w * x_vec_k0.w;
        psum += a_vec_k1.x * x_vec_k1.x;
        psum += a_vec_k1.y * x_vec_k1.y;
        psum += a_vec_k1.z * x_vec_k1.z;
        psum += a_vec_k1.w * x_vec_k1.w;

        a_vec_k0 = a_vec_n0;
        x_vec_k0 = x_vec_n0;
        a_vec_k1 = a_vec_n1;
        x_vec_k1 = x_vec_n1;
    }

    psum += a_vec_k0.x * x_vec_k0.x;
    psum += a_vec_k0.y * x_vec_k0.y;
    psum += a_vec_k0.z * x_vec_k0.z;
    psum += a_vec_k0.w * x_vec_k0.w;
    psum += a_vec_k1.x * x_vec_k1.x;
    psum += a_vec_k1.y * x_vec_k1.y;
    psum += a_vec_k1.z * x_vec_k1.z;
    psum += a_vec_k1.w * x_vec_k1.w;

    psum = rocblas_wavefront_reduce<TILE_DIM_X>(psum);

    if(tx == 0)
    {
        y[y_col * int64_t(incy)]
            = (beta == 0.0f) ? alpha * psum : alpha * psum + y[y_col * int64_t(incy)] * beta;
    }
}

// Specialized pipelined kernel for double
template <bool CONJ, int TILE_DIM_X, int TILE_DIM_Y>
ROCBLAS_KERNEL_ILF void rocblas_gemvt_row_vectorized_kernel_calc(rocblas_int m,
                                                                 rocblas_int n,
                                                                 double      alpha,
                                                                 const double* __restrict__ A,
                                                                 rocblas_stride lda,
                                                                 const double* __restrict__ x,
                                                                 rocblas_int incx,
                                                                 double      beta,
                                                                 double* __restrict__ y,
                                                                 rocblas_int incy)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int y_col = blockIdx.x * TILE_DIM_Y + ty;

    if(y_col >= n)
        return;

    if(alpha == 0.0)
    {
        if(tx == 0)
        {
            y[y_col * int64_t(incy)] = (beta == 0.0) ? 0.0 : y[y_col * int64_t(incy)] * beta;
        }
        return;
    }

    double psum = 0.0;

    const double* current_A = A + y_col * lda;

    const int stride   = TILE_DIM_X * 4;
    int       row_base = tx * 4;

    double2 a_vec_k0, x_vec_k0, a_vec_k1, x_vec_k1;
    double2 a_vec_n0, x_vec_n0, a_vec_n1, x_vec_n1;

    if(row_base < m)
    {
        if(row_base + 3 < m && incx == 1)
        {
            a_vec_k0 = *reinterpret_cast<const double2*>(current_A + row_base);
            x_vec_k0 = *reinterpret_cast<const double2*>(x + row_base);
            a_vec_k1 = *reinterpret_cast<const double2*>(current_A + row_base + 2);
            x_vec_k1 = *reinterpret_cast<const double2*>(x + row_base + 2);
        }
        else
        {
            for(int i = 0; i < 2; ++i)
            {
                if(row_base + i < m)
                {
                    ((double*)&a_vec_k0)[i] = current_A[row_base + i];
                    ((double*)&x_vec_k0)[i] = x[(row_base + i) * int64_t(incx)];
                }
                else
                {
                    ((double*)&a_vec_k0)[i] = 0.0;
                    ((double*)&x_vec_k0)[i] = 0.0;
                }
                if(row_base + 2 + i < m)
                {
                    ((double*)&a_vec_k1)[i] = current_A[row_base + 2 + i];
                    ((double*)&x_vec_k1)[i] = x[(row_base + 2 + i) * int64_t(incx)];
                }
                else
                {
                    ((double*)&a_vec_k1)[i] = 0.0;
                    ((double*)&x_vec_k1)[i] = 0.0;
                }
            }
        }
    }
    else
    {
        a_vec_k0 = a_vec_k1 = x_vec_k0 = x_vec_k1 = make_double2(0.0, 0.0);
    }

    for(row_base += stride; row_base < m; row_base += stride)
    {
        if(row_base + 3 < m && incx == 1)
        {
            a_vec_n0 = *reinterpret_cast<const double2*>(current_A + row_base);
            x_vec_n0 = *reinterpret_cast<const double2*>(x + row_base);
            a_vec_n1 = *reinterpret_cast<const double2*>(current_A + row_base + 2);
            x_vec_n1 = *reinterpret_cast<const double2*>(x + row_base + 2);
        }
        else
        {
            for(int i = 0; i < 2; ++i)
            {
                if(row_base + i < m)
                {
                    ((double*)&a_vec_n0)[i] = current_A[row_base + i];
                    ((double*)&x_vec_n0)[i] = x[(row_base + i) * int64_t(incx)];
                }
                else
                {
                    ((double*)&a_vec_n0)[i] = 0.0;
                    ((double*)&x_vec_n0)[i] = 0.0;
                }
                if(row_base + 2 + i < m)
                {
                    ((double*)&a_vec_n1)[i] = current_A[row_base + 2 + i];
                    ((double*)&x_vec_n1)[i] = x[(row_base + 2 + i) * int64_t(incx)];
                }
                else
                {
                    ((double*)&a_vec_n1)[i] = 0.0;
                    ((double*)&x_vec_n1)[i] = 0.0;
                }
            }
        }

        psum += a_vec_k0.x * x_vec_k0.x;
        psum += a_vec_k0.y * x_vec_k0.y;
        psum += a_vec_k1.x * x_vec_k1.x;
        psum += a_vec_k1.y * x_vec_k1.y;

        a_vec_k0 = a_vec_n0;
        x_vec_k0 = x_vec_n0;
        a_vec_k1 = a_vec_n1;
        x_vec_k1 = x_vec_n1;
    }

    psum += a_vec_k0.x * x_vec_k0.x;
    psum += a_vec_k0.y * x_vec_k0.y;
    psum += a_vec_k1.x * x_vec_k1.x;
    psum += a_vec_k1.y * x_vec_k1.y;

    psum = rocblas_wavefront_reduce<TILE_DIM_X>(psum);

    if(tx == 0)
    {
        y[y_col * int64_t(incy)]
            = (beta == 0.0) ? alpha * psum : alpha * psum + y[y_col * int64_t(incy)] * beta;
    }
}

// Specialized pipelined kernel for rocblas_float_complex
template <bool CONJ, int TILE_DIM_X, int TILE_DIM_Y>
ROCBLAS_KERNEL_ILF void
    rocblas_gemvt_row_vectorized_kernel_calc(rocblas_int           m,
                                             rocblas_int           n,
                                             rocblas_float_complex alpha,
                                             const rocblas_float_complex* __restrict__ A,
                                             rocblas_stride lda,
                                             const rocblas_float_complex* __restrict__ x,
                                             rocblas_int           incx,
                                             rocblas_float_complex beta,
                                             rocblas_float_complex* __restrict__ y,
                                             rocblas_int incy)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int y_col = blockIdx.x * TILE_DIM_Y + ty;

    if(y_col >= n)
        return;

    if(alpha.real() == 0.0f && alpha.imag() == 0.0f)
    {
        if(tx == 0)
        {
            bool beta_is_zero        = beta.real() == 0.0f && beta.imag() == 0.0f;
            y[y_col * int64_t(incy)] = beta_is_zero ? rocblas_float_complex(0.0f, 0.0f)
                                                    : y[y_col * int64_t(incy)] * beta;
        }
        return;
    }

    rocblas_float_complex psum(0.0f, 0.0f);

    const rocblas_float_complex* current_A = A + y_col * lda;

    const int stride   = TILE_DIM_X * 4;
    int       row_base = tx * 4;

    float4 a_vec_k0, x_vec_k0, a_vec_k1, x_vec_k1;
    float4 a_vec_n0, x_vec_n0, a_vec_n1, x_vec_n1;

    if(row_base < m)
    {
        if(row_base + 3 < m && incx == 1)
        {
            a_vec_k0 = *reinterpret_cast<const float4*>(current_A + row_base);
            x_vec_k0 = *reinterpret_cast<const float4*>(x + row_base);
            a_vec_k1 = *reinterpret_cast<const float4*>(current_A + row_base + 2);
            x_vec_k1 = *reinterpret_cast<const float4*>(x + row_base + 2);
        }
        else
        {
            for(int i = 0; i < 2; ++i)
            {
                if(row_base + i < m)
                {
                    ((rocblas_float_complex*)&a_vec_k0)[i] = current_A[row_base + i];
                    ((rocblas_float_complex*)&x_vec_k0)[i] = x[(row_base + i) * int64_t(incx)];
                }
                else
                {
                    ((rocblas_float_complex*)&a_vec_k0)[i] = rocblas_float_complex(0.0f, 0.0f);
                    ((rocblas_float_complex*)&x_vec_k0)[i] = rocblas_float_complex(0.0f, 0.0f);
                }
                if(row_base + 2 + i < m)
                {
                    ((rocblas_float_complex*)&a_vec_k1)[i] = current_A[row_base + 2 + i];
                    ((rocblas_float_complex*)&x_vec_k1)[i] = x[(row_base + 2 + i) * int64_t(incx)];
                }
                else
                {
                    ((rocblas_float_complex*)&a_vec_k1)[i] = rocblas_float_complex(0.0f, 0.0f);
                    ((rocblas_float_complex*)&x_vec_k1)[i] = rocblas_float_complex(0.0f, 0.0f);
                }
            }
        }
    }
    else
    {
        a_vec_k0 = a_vec_k1 = x_vec_k0 = x_vec_k1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    for(row_base += stride; row_base < m; row_base += stride)
    {
        if(row_base + 3 < m && incx == 1)
        {
            a_vec_n0 = *reinterpret_cast<const float4*>(current_A + row_base);
            x_vec_n0 = *reinterpret_cast<const float4*>(x + row_base);
            a_vec_n1 = *reinterpret_cast<const float4*>(current_A + row_base + 2);
            x_vec_n1 = *reinterpret_cast<const float4*>(x + row_base + 2);
        }
        else
        {
            for(int i = 0; i < 2; ++i)
            {
                if(row_base + i < m)
                {
                    ((rocblas_float_complex*)&a_vec_n0)[i] = current_A[row_base + i];
                    ((rocblas_float_complex*)&x_vec_n0)[i] = x[(row_base + i) * int64_t(incx)];
                }
                else
                {
                    ((rocblas_float_complex*)&a_vec_n0)[i] = rocblas_float_complex(0.0f, 0.0f);
                    ((rocblas_float_complex*)&x_vec_n0)[i] = rocblas_float_complex(0.0f, 0.0f);
                }
                if(row_base + 2 + i < m)
                {
                    ((rocblas_float_complex*)&a_vec_n1)[i] = current_A[row_base + 2 + i];
                    ((rocblas_float_complex*)&x_vec_n1)[i] = x[(row_base + 2 + i) * int64_t(incx)];
                }
                else
                {
                    ((rocblas_float_complex*)&a_vec_n1)[i] = rocblas_float_complex(0.0f, 0.0f);
                    ((rocblas_float_complex*)&x_vec_n1)[i] = rocblas_float_complex(0.0f, 0.0f);
                }
            }
        }

        const rocblas_float_complex* a_k0
            = reinterpret_cast<const rocblas_float_complex*>(&a_vec_k0);
        const rocblas_float_complex* x_k0
            = reinterpret_cast<const rocblas_float_complex*>(&x_vec_k0);
        const rocblas_float_complex* a_k1
            = reinterpret_cast<const rocblas_float_complex*>(&a_vec_k1);
        const rocblas_float_complex* x_k1
            = reinterpret_cast<const rocblas_float_complex*>(&x_vec_k1);

        for(int i = 0; i < 2; ++i)
        {
            psum += conj_if_true<CONJ>(a_k0[i]) * x_k0[i];
            psum += conj_if_true<CONJ>(a_k1[i]) * x_k1[i];
        }

        a_vec_k0 = a_vec_n0;
        x_vec_k0 = x_vec_n0;
        a_vec_k1 = a_vec_n1;
        x_vec_k1 = x_vec_n1;
    }

    const rocblas_float_complex* a_k0 = reinterpret_cast<const rocblas_float_complex*>(&a_vec_k0);
    const rocblas_float_complex* x_k0 = reinterpret_cast<const rocblas_float_complex*>(&x_vec_k0);
    const rocblas_float_complex* a_k1 = reinterpret_cast<const rocblas_float_complex*>(&a_vec_k1);
    const rocblas_float_complex* x_k1 = reinterpret_cast<const rocblas_float_complex*>(&x_vec_k1);

    for(int i = 0; i < 2; ++i)
    {
        psum += conj_if_true<CONJ>(a_k0[i]) * x_k0[i];
        psum += conj_if_true<CONJ>(a_k1[i]) * x_k1[i];
    }

    psum = rocblas_wavefront_reduce<TILE_DIM_X>(psum);

    if(tx == 0)
    {
        bool beta_is_zero = beta.real() == 0.0f && beta.imag() == 0.0f;
        y[y_col * int64_t(incy)]
            = beta_is_zero ? alpha * psum : alpha * psum + y[y_col * int64_t(incy)] * beta;
    }
}

// Specialized pipelined kernel for rocblas_double_complex
template <bool CONJ, int TILE_DIM_X, int TILE_DIM_Y>
ROCBLAS_KERNEL_ILF void
    rocblas_gemvt_row_vectorized_kernel_calc(rocblas_int            m,
                                             rocblas_int            n,
                                             rocblas_double_complex alpha,
                                             const rocblas_double_complex* __restrict__ A,
                                             rocblas_stride lda,
                                             const rocblas_double_complex* __restrict__ x,
                                             rocblas_int            incx,
                                             rocblas_double_complex beta,
                                             rocblas_double_complex* __restrict__ y,
                                             rocblas_int incy)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int y_col = blockIdx.x * TILE_DIM_Y + ty;

    if(y_col >= n)
        return;

    if(alpha.real() == 0.0 && alpha.imag() == 0.0)
    {
        if(tx == 0)
        {
            bool beta_is_zero = beta.real() == 0.0 && beta.imag() == 0.0;
            y[y_col * int64_t(incy)]
                = beta_is_zero ? rocblas_double_complex(0.0, 0.0) : y[y_col * int64_t(incy)] * beta;
        }
        return;
    }

    rocblas_double_complex psum(0.0, 0.0);

    const rocblas_double_complex* current_A = A + y_col * lda;

    const int stride   = TILE_DIM_X * 2;
    int       row_base = tx * 2;

    rocblas_double_complex a_k0, x_k0, a_k1, x_k1;
    rocblas_double_complex a_n0, x_n0, a_n1, x_n1;

    if(row_base < m)
    {
        a_k0 = current_A[row_base];
        x_k0 = x[row_base * int64_t(incx)];
    }
    else
    {
        a_k0 = rocblas_double_complex(0.0, 0.0);
        x_k0 = rocblas_double_complex(0.0, 0.0);
    }
    if(row_base + 1 < m)
    {
        a_k1 = current_A[row_base + 1];
        x_k1 = x[(row_base + 1) * int64_t(incx)];
    }
    else
    {
        a_k1 = rocblas_double_complex(0.0, 0.0);
        x_k1 = rocblas_double_complex(0.0, 0.0);
    }

    for(row_base += stride; row_base < m; row_base += stride)
    {
        a_n0 = current_A[row_base];
        x_n0 = x[row_base * int64_t(incx)];
        if(row_base + 1 < m)
        {
            a_n1 = current_A[row_base + 1];
            x_n1 = x[(row_base + 1) * int64_t(incx)];
        }
        else
        {
            a_n1 = rocblas_double_complex(0.0, 0.0);
            x_n1 = rocblas_double_complex(0.0, 0.0);
        }

        psum += conj_if_true<CONJ>(a_k0) * x_k0;
        psum += conj_if_true<CONJ>(a_k1) * x_k1;

        a_k0 = a_n0;
        x_k0 = x_n0;
        a_k1 = a_n1;
        x_k1 = x_n1;
    }

    psum += conj_if_true<CONJ>(a_k0) * x_k0;
    psum += conj_if_true<CONJ>(a_k1) * x_k1;

    psum = rocblas_wavefront_reduce<TILE_DIM_X>(psum);

    if(tx == 0)
    {
        bool beta_is_zero = beta.real() == 0.0 && beta.imag() == 0.0;
        y[y_col * int64_t(incy)]
            = beta_is_zero ? alpha * psum : alpha * psum + y[y_col * int64_t(incy)] * beta;
    }
}

// Launcher for the specialized row-vectorized kernels
template <bool CONJ, int TILE_DIM_X, int TILE_DIM_Y, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL(TILE_DIM_X* TILE_DIM_Y)
rocblas_gemvt_row_vectorized_kernel(rocblas_int    m,
                                    rocblas_int    n,
                                    Tex            alpha_device_host,
                                    rocblas_stride stride_alpha,
                                    const Ti*      Aa,
                                    rocblas_stride shifta,
                                    rocblas_int    lda,
                                    rocblas_stride strideA,
                                    const Ti*      xa,
                                    rocblas_stride shiftx,
                                    rocblas_int    incx,
                                    rocblas_stride stridex,
                                    Tex            beta_device_host,
                                    rocblas_stride stride_beta,
                                    To*            ya,
                                    rocblas_stride shifty,
                                    rocblas_int    incy,
                                    rocblas_stride stridey,
                                    rocblas_int    batch_count)
{
// gfx90a gfx942 kernels
#if defined(__gfx90a__) || defined(__gfx942__)

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_gemvt_row_vectorized_kernel_calc<CONJ, TILE_DIM_X, TILE_DIM_Y>(
            m, n, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif

#endif
}

template <int NB, typename Tex, typename To>
ROCBLAS_KERNEL_ILF void rocblas_gemv_scal_kernel_calc(
    rocblas_int n, Tex beta, rocblas_stride stride_beta, To* y, rocblas_int incy)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // bound
    if(tid < n)
    {
        if(beta == 0)
        {
            y[tid * incy] = To(0);
        }
        else
        {
            y[tid * incy] = To(y[tid * incy] * beta);
        }
    }
}

template <int NB, typename Tex, typename To>
ROCBLAS_KERNEL(NB)
rocblas_gemv_scal_kernel(rocblas_int    n,
                         Tex            beta_device_host,
                         rocblas_stride stride_beta,
                         To             ya,
                         rocblas_stride offset_y,
                         rocblas_int    incy,
                         rocblas_stride stride_y,
                         rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto* __restrict__ y = load_ptr_batch(ya, batch, offset_y, stride_y);
        auto beta            = load_scalar(beta_device_host, batch, stride_beta);

        if(beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }
        rocblas_gemv_scal_kernel_calc<NB>(n, beta, stride_beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//
template <int DIM_X, int DIM_Y, int elements_per_thread, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL_ILF void rocblas_gemvn_double_buffered_kernel_calc(rocblas_int rows,
                                                                  rocblas_int cols,
                                                                  Tex         alpha,
                                                                  const Ti* __restrict__ A,
                                                                  rocblas_int lda,
                                                                  const Ti* __restrict__ x,
                                                                  rocblas_int incx,
                                                                  To* __restrict__ y,
                                                                  rocblas_int incy)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);

    Tex res_1_ = Tex(0);
    Tex res_2_ = Tex(0);
    Tex areg_upper[elements_per_thread];
    Tex areg_lower[elements_per_thread];

    __shared__ Tex la[DIM_X * (2 * DIM_Y)];

    int count = (cols / DIM_X) / gridDim.y + (by < (cols / DIM_X) % gridDim.y);
    {
        int start = by * ((cols / DIM_X) / gridDim.y) + min(by, (cols / DIM_X) % gridDim.y);

        // Advance 'A'
        A += DIM_X * bx;
        A += start * DIM_X * size_t(lda);

        // Advance 'x'
        x += start * DIM_X * int64_t(incx);

        // Advance 'y'
        y += (bx * DIM_X) * int64_t(incy);
    }

    if(count == 0)
        return;

    const size_t j = ty_ * elements_per_thread * size_t(lda) + tx_;

// read upper
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        areg_upper[k] = A[j + k * size_t(lda)];

    int Vblocks = 0;
    //#pragma unroll
    for(Vblocks = 0; Vblocks < count; Vblocks++)
    {
// read lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            areg_lower[k] = A[(DIM_X / 2) + j + k * size_t(lda)];

// compute upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            res_1_ += areg_upper[k] * x[(ty_ * elements_per_thread + k) * int64_t(incx)];

        A += DIM_X * size_t(lda);

        // read upper from next block
        if(Vblocks != count - 1)
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                areg_upper[k] = A[j + k * size_t(lda)];
        }

// compute lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            res_2_ += areg_lower[k] * x[(ty_ * elements_per_thread + k) * int64_t(incx)];

        //Should add int64_t(incx) and stress test if newer architecture uses this routine
        x += DIM_X * incx;
    }

    la[ty_ * DIM_X + tx_]               = res_1_;
    la[ty_ * DIM_X + tx_ + (DIM_X / 2)] = res_2_;
    __syncthreads();

    if(ty == 0)
    {
        res_1_ = Tex(0);
#pragma unroll
        for(int k = 0; k < 2 * DIM_Y; k++)
            res_1_ += la[k * DIM_X + tx];

        atomicAdd(&y[tx * int64_t(incy)], (alpha * res_1_));
    }
}

template <bool CONJ, int DIM_X, int elements_per_thread, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL_ILF void rocblas_gemvt_double_buffered_kernel_calc(rocblas_int rows,
                                                                  rocblas_int cols,
                                                                  Tex         alpha,
                                                                  const Ti* __restrict__ A,
                                                                  rocblas_int lda,
                                                                  const Ti* __restrict__ x,
                                                                  rocblas_int incx,
                                                                  To* __restrict__ y,
                                                                  rocblas_int incy)
{
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);

    __shared__ Tex la[DIM_X * (DIM_X / 2)];

    Tex Areg_upper[elements_per_thread];
    Tex Areg_lower[elements_per_thread];
    Tex treg[elements_per_thread] = {Tex(0)};

    int count = (rows / DIM_X) / gridDim.y + (by < (rows / DIM_X) % gridDim.y);
    {
        int start = by * ((rows / DIM_X) / gridDim.y) + min(by, (rows / DIM_X) % gridDim.y);

        // Advance 'A' to start a block column
        A += DIM_X * bx * size_t(lda);
        A += start * DIM_X;

        // Advance 'x'
        x += start * DIM_X * int64_t(incx);

        // Advance 'y'
        y += (bx * DIM_X) * int64_t(incy);
    }

    if(count == 0)
        return;

    const size_t j = ty_ * elements_per_thread * size_t(lda) + tx_;

// read upper
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        Areg_upper[k] = A[j + k * size_t(lda)];

    for(int Vblocks = 0; Vblocks < count; Vblocks++)
    {
// read lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            Areg_lower[k] = A[(DIM_X / 2) + j + k * size_t(lda)];

// compute upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            treg[k] += (CONJ ? conj(Areg_upper[k]) : Areg_upper[k]) * x[tx_ * int64_t(incx)];

        A += DIM_X;

        // read upper from next block
        if(Vblocks != count - 1)
        {
#pragma unroll
            for(int k = 0; k < elements_per_thread; k++)
                Areg_upper[k] = A[j + k * size_t(lda)];
        }

//compute lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            treg[k] += (CONJ ? conj(Areg_lower[k]) : Areg_lower[k])
                       * x[(tx_ + (DIM_X / 2)) * int64_t(incx)];

        x += DIM_X * int64_t(incx);
    }

// final reduction
#pragma unroll
    for(int k = 0; k < elements_per_thread; k++)
        la[(ty_ * elements_per_thread + k) * (DIM_X / 2) + tx_] = treg[k];

    __syncthreads();

    if(ty == 0)
    {
        treg[0] = Tex(0);
#pragma unroll
        for(int k = tx; k < tx + (DIM_X / 2); k++)
            treg[0] += la[tx * (DIM_X / 2) + (k % (DIM_X / 2))];

        atomicAdd(&y[tx * int64_t(incy)], (treg[0] * alpha));
    }
}

template <int DIM_X,
          int DIM_Y,
          typename T_Index,
          typename Ti,
          typename Tex,
          typename To,
          std::enable_if_t<!std::is_same_v<Ti, rocblas_double_complex>, int> = 0>
ROCBLAS_KERNEL_ILF void rocblas_gemvn_kernel_calc(rocblas_int m,
                                                  rocblas_int n,
                                                  Tex         alpha,
                                                  const Ti*   A,
                                                  T_Index     lda,
                                                  const Ti*   x,
                                                  T_Index     incx,
                                                  Tex         beta,
                                                  To*         y,
                                                  T_Index     incy)
{
    rocblas_int thread_id = threadIdx.x + threadIdx.y * DIM_X;

    if(!alpha)
    {
        if(thread_id < DIM_X * 4)
        {
            int64_t ind = blockIdx.x * DIM_X * 4 + thread_id;
            if(ind < m)
                y[ind * T_Index(incy)] = beta ? (To)(beta * y[ind * T_Index(incy)]) : (To)0;
        }
        return;
    }

    // threads are all configurated locally
    rocblas_int tx = threadIdx.x;
    rocblas_int ty = threadIdx.y;

    rocblas_int ind;

    __shared__ Tex sdata[DIM_X * 4 * DIM_Y];

    Tex res_A[4];
    Tex res_x[4];

    res_A[0] = res_A[1] = res_A[2] = res_A[3] = Tex{0};

    ind = blockIdx.x * DIM_X * 4 + tx;

    rocblas_int n_tail = n % (4 * DIM_Y);
    rocblas_int col;

    for(col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y)
    {
        res_x[0] = x[(col + 0) * T_Index(incx)];
        res_x[1] = x[(col + 1) * T_Index(incx)];
        res_x[2] = x[(col + 2) * T_Index(incx)];
        res_x[3] = x[(col + 3) * T_Index(incx)];

        if(ind < m)
        {
            res_A[0] += A[ind + (col + 0) * T_Index(lda)] * res_x[0];
            res_A[0] += A[ind + (col + 1) * T_Index(lda)] * res_x[1];
            res_A[0] += A[ind + (col + 2) * T_Index(lda)] * res_x[2];
            res_A[0] += A[ind + (col + 3) * T_Index(lda)] * res_x[3];

            if(ind + DIM_X < m)
            {
                res_A[1] += A[ind + DIM_X + (col + 0) * T_Index(lda)] * res_x[0];
                res_A[1] += A[ind + DIM_X + (col + 1) * T_Index(lda)] * res_x[1];
                res_A[1] += A[ind + DIM_X + (col + 2) * T_Index(lda)] * res_x[2];
                res_A[1] += A[ind + DIM_X + (col + 3) * T_Index(lda)] * res_x[3];

                if(ind + 2 * DIM_X < m)
                {
                    res_A[2] += A[ind + 2 * DIM_X + (col + 0) * T_Index(lda)] * res_x[0];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 1) * T_Index(lda)] * res_x[1];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 2) * T_Index(lda)] * res_x[2];
                    res_A[2] += A[ind + 2 * DIM_X + (col + 3) * T_Index(lda)] * res_x[3];

                    if(ind + 3 * DIM_X < m)
                    {
                        res_A[3] += A[ind + 3 * DIM_X + (col + 0) * T_Index(lda)] * res_x[0];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 1) * T_Index(lda)] * res_x[1];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 2) * T_Index(lda)] * res_x[2];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 3) * T_Index(lda)] * res_x[3];
                    }
                }
            }
        }
    }

    // if n is not multiple of (DIM_Y * 4)
    if(n_tail > 0)
    {
        res_x[0] = res_x[1] = res_x[2] = res_x[3] = Tex{0};

        if(col + 0 < n)
        {
            res_x[0] = x[(col + 0) * T_Index(incx)];

            if(col + 1 < n)
            {
                res_x[1] = x[(col + 1) * T_Index(incx)];

                if(col + 2 < n)
                {
                    res_x[2] = x[(col + 2) * T_Index(incx)];

                    if(col + 3 < n)
                        res_x[3] = x[(col + 3) * T_Index(incx)];
                }
            }
        }

        if(ind < m)
        {
            res_A[0] += A[ind + (col + 0) * T_Index(lda) * (col + 0 < n)] * res_x[0];
            res_A[0] += A[ind + (col + 1) * T_Index(lda) * (col + 1 < n)] * res_x[1];
            res_A[0] += A[ind + (col + 2) * T_Index(lda) * (col + 2 < n)] * res_x[2];
            res_A[0] += A[ind + (col + 3) * T_Index(lda) * (col + 3 < n)] * res_x[3];

            if(ind + DIM_X < m)
            {
                res_A[1] += A[ind + DIM_X + (col + 0) * T_Index(lda) * (col + 0 < n)] * res_x[0];
                res_A[1] += A[ind + DIM_X + (col + 1) * T_Index(lda) * (col + 1 < n)] * res_x[1];
                res_A[1] += A[ind + DIM_X + (col + 2) * T_Index(lda) * (col + 2 < n)] * res_x[2];
                res_A[1] += A[ind + DIM_X + (col + 3) * T_Index(lda) * (col + 3 < n)] * res_x[3];

                if(ind + 2 * DIM_X < m)
                {
                    res_A[2]
                        += A[ind + 2 * DIM_X + (col + 0) * T_Index(lda) * (col + 0 < n)] * res_x[0];
                    res_A[2]
                        += A[ind + 2 * DIM_X + (col + 1) * T_Index(lda) * (col + 1 < n)] * res_x[1];
                    res_A[2]
                        += A[ind + 2 * DIM_X + (col + 2) * T_Index(lda) * (col + 2 < n)] * res_x[2];
                    res_A[2]
                        += A[ind + 2 * DIM_X + (col + 3) * T_Index(lda) * (col + 3 < n)] * res_x[3];

                    if(ind + 3 * DIM_X < m)
                    {
                        res_A[3] += A[ind + 3 * DIM_X + (col + 0) * T_Index(lda) * (col + 0 < n)]
                                    * res_x[0];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 1) * T_Index(lda) * (col + 1 < n)]
                                    * res_x[1];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 2) * T_Index(lda) * (col + 2 < n)]
                                    * res_x[2];
                        res_A[3] += A[ind + 3 * DIM_X + (col + 3) * T_Index(lda) * (col + 3 < n)]
                                    * res_x[3];
                    }
                }
            }
        }
    }

    sdata[tx + ty * DIM_X * 4]             = res_A[0];
    sdata[tx + DIM_X + ty * DIM_X * 4]     = res_A[1];
    sdata[tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
    sdata[tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

    __syncthreads();

    if(thread_id < DIM_X * 4)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];

        ind = blockIdx.x * DIM_X * 4 + thread_id;

        if(ind < m)
            y[ind * T_Index(incy)]
                = beta ? (To)(alpha * sdata[thread_id] + beta * y[ind * T_Index(incy)])
                       : (To)(alpha * sdata[thread_id]);
    }
}

// Overload for double precision complex numbers. We run out of registers
// if we use the above algorithm.
template <int DIM_X, int DIM_Y, typename T_Index, typename U>
ROCBLAS_KERNEL_ILF void rocblas_gemvn_kernel_calc(rocblas_int                   m,
                                                  rocblas_int                   n,
                                                  U                             alpha,
                                                  const rocblas_double_complex* A,
                                                  T_Index                       lda,
                                                  const rocblas_double_complex* x,
                                                  T_Index                       incx,
                                                  U                             beta,
                                                  rocblas_double_complex*       y,
                                                  T_Index                       incy)
{
    rocblas_int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    if(!alpha)
    {
        if(thread_id < DIM_X)
        {
            int64_t ind = blockIdx.x * DIM_X + thread_id;
            if(ind < m)
                y[ind * T_Index(incy)] = beta ? beta * y[ind * T_Index(incy)] : 0;
        }
        return;
    }

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = blockIdx.x * DIM_X + tx;

    __shared__ rocblas_double_complex sdata[DIM_X * DIM_Y];

    rocblas_double_complex res_A;
    rocblas_double_complex res_x;

    res_A = res_x = rocblas_double_complex{0, 0};

    rocblas_int n_tail = n % (DIM_Y);
    rocblas_int col;

    for(col = ty; col < (n - n_tail); col += DIM_Y)
    {

        if(ind < m)
        {
            res_A += A[ind + col * T_Index(lda)] * x[col * T_Index(incx)];
        }
    }

    // if n  is not multiple of (DIM_Y)
    if(n_tail > 0)
    {
        if(col + 0 < n)
            res_x = x[col * T_Index(incx)];
        else
            res_x = rocblas_double_complex{0, 0};

        if(ind < m)
        {
            res_A += A[ind + col * T_Index(lda) * (col < n)] * res_x;
        }
    }

    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    if(thread_id < DIM_X)
    {
        // always alpha non zero as !alpha quick return
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        ind = blockIdx.x * DIM_X + thread_id;

        if(ind < m)
        {
            y[ind * T_Index(incy)] = beta ? alpha * sdata[thread_id] + beta * y[ind * T_Index(incy)]
                                          : alpha * sdata[thread_id];
        }
    }
}

//Optimized kernel for GEMV transpose case when m or n is less than gemvt_threshold
template <bool CONJ, int NB_X, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL_ILF void rocblas_gemvt_kernel_calc(rocblas_int m,
                                                  rocblas_int n,
                                                  Tex         alpha,
                                                  const Ti*   A,
                                                  rocblas_int lda,
                                                  const Ti*   x,
                                                  rocblas_int incx,
                                                  Tex         beta,
                                                  To*         y,
                                                  rocblas_int incy)
{
    rocblas_int tx  = threadIdx.x;
    rocblas_int col = blockIdx.x;

    if(!alpha)
    {
        if(tx == 0)
            y[col * int64_t(incy)] = beta ? (To)(beta * y[col * int64_t(incy)]) : (To)0;
        return;
    }

    if(tx < m)
        A += tx;

    A += col * size_t(lda);

    Tex res = 0;

    __shared__ Tex sdata[NB_X];

    // partial sums
    rocblas_int m_full = (m / NB_X) * NB_X;

    for(rocblas_int i = 0; i < m_full; i += NB_X)
        res += (CONJ ? conj(A[i]) : A[i]) * x[(tx + i) * int64_t(incx)];

    if(tx + m_full < m)
        res += (CONJ ? conj(A[m_full]) : A[m_full]) * x[(tx + m_full) * int64_t(incx)];

    sdata[tx] = res;

    // tree reduction of partial sums,
    if(NB_X > 16)
    {
        rocblas_sum_reduce<NB_X>(tx, sdata);
    }
    else
    {
        __syncthreads();

        if(tx == 0)
        {
            for(rocblas_int i = 1; i < m && i < NB_X; i++)
                sdata[0] += sdata[i];
        }

        __syncthreads();
    }

    if(tx == 0)
    {
        // !alpha handled earlier by early return
        y[col * int64_t(incy)] = beta ? (To)(alpha * sdata[0] + beta * y[col * int64_t(incy)])
                                      : (To)(alpha * sdata[0]);
    }
}

template <bool CONJ, int NB_X, typename T_Index, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL_ILF void rocblas_gemvt_reduce_kernel_calc(rocblas_int m,
                                                         rocblas_int n,
                                                         Tex         alpha,
                                                         const Ti* __restrict__ A,
                                                         T_Index lda,
                                                         const Ti* __restrict__ x,
                                                         T_Index incx,
                                                         Tex     beta,
                                                         To* __restrict__ y,
                                                         T_Index incy)
{
    rocblas_int tx  = threadIdx.x;
    rocblas_int col = blockIdx.x;

    if(!alpha)
    {
        if(tx == 0)
            y[col * (incy)] = beta ? (To)(beta * y[col * (incy)]) : (To)0;
        return;
    }

    if(tx < m)
        A += tx;

    //Each BlockIdx.x takes care of each column of matrix A
    A += col * (lda);

    Tex res = 0;

    // partial sums
    rocblas_int m_full = (m / NB_X) * NB_X;

    //Each column of Matrix A is multiplied with vector x and the resultant value is stored in res.
    //If m > NB_X, then the threads are reused and the multiplied values will be accumalated.
    for(rocblas_int i = 0; tx + i < m_full; i += NB_X)
        res += (CONJ ? conj(A[i]) : A[i]) * x[(tx + i) * (incx)];

    if(tx + m_full < m)
        res += (CONJ ? conj(A[m_full]) : A[m_full]) * x[(tx + m_full) * (incx)];

    static_assert(NB_X > WARP_32);

    // if ever use smaller NB_X then call directly a faster warp only reduce
    // if(NB_X <= warpSize)
    // {
    //     // shuffle warp reduction if NB_X is less than or equal warpSize
    //     rocblas_wavefront_reduce<NB_X>(res);
    // }
    // else
    // {

    //block shuffle warp reduction
    if(warpSize == WARP_32)
        res = rocblas_dot_block_reduce<WARP_32, NB_X>(res);
    else
        res = rocblas_dot_block_reduce<WARP_64, NB_X>(res);
    //}

    if(tx == 0)
    {
        // !alpha handled earlier by early return
        y[col * (incy)] = beta ? (To)(alpha * res + beta * y[col * (incy)]) : (To)(alpha * res);
    }
}

template <bool CONJ, rocblas_int NB_X, rocblas_int WIN, typename T_Index, typename Ti, typename Tex>
ROCBLAS_KERNEL_ILF void rocblas_gemvt_sn_kernel_calc(rocblas_int m,
                                                     rocblas_int n,
                                                     Tex         alpha,
                                                     const Ti*   A,
                                                     T_Index     lda,
                                                     const Ti*   x,
                                                     T_Index     incx,
                                                     Tex*        workspace,
                                                     uint32_t    batch)
{
    // skinny n kernel

    rocblas_int tx = threadIdx.x;

    // offset blocks * cols * batch
    workspace += size_t(gridDim.x) * n * batch;

    // We need to short-circuit if alpha==0 and not propagate NaNs
    if(!alpha)
    {
        if(tx == 0)
            for(int i = 0; i < n; i++)
                workspace[blockIdx.x + size_t(i) * gridDim.x] = 0;
        return;
    }

    int row = tx * WIN + blockIdx.x * NB_X * WIN;
    A += row;

    constexpr int NC = 4;

    rocblas_int n_tail = n % NC;

    rocblas_int m_tail = m % WIN;

    Tex sum[NC];
    Tex xvec[WIN];

    int i = 0; // col
    for(i = 0; i < n - n_tail; i += NC)
    {
        sum[0] = sum[1] = sum[2] = sum[3] = Tex{0};

        if(row + WIN <= m)
        {
            for(int j = 0; j < WIN; j++)
            {
                xvec[j] = x[(row + j) * T_Index(incx)];
            }
            for(int j = 0; j < WIN; j++)
            {
                for(int k = 0; k < NC; k++)
                    sum[k] += (CONJ ? conj(A[(i + k) * T_Index(lda) + j])
                                    : A[(i + k) * T_Index(lda) + j])
                              * xvec[j];
            }
        }
        else if(row + m_tail <= m)
        {
            for(int j = 0; j < m_tail; j++)
            {
                xvec[j] = x[(row + j) * T_Index(incx)];
            }
            for(int j = 0; j < m_tail; j++)
            {
                for(int k = 0; k < NC; k++)
                    sum[k] += (CONJ ? conj(A[(i + k) * T_Index(lda) + j])
                                    : A[(i + k) * T_Index(lda) + j])
                              * xvec[j];
            }
        }

        for(int k = 0; k < NC; k++)
        {
            if(warpSize == WARP_32)
                sum[k] = rocblas_dot_block_reduce<WARP_32, NB_X>(sum[k]);
            else
                sum[k] = rocblas_dot_block_reduce<WARP_64, NB_X>(sum[k]);
        }

        if(tx == 0)
        {
            for(int k = 0; k < NC; k++)
                workspace[blockIdx.x + T_Index(k + i) * gridDim.x] = alpha * sum[k];
        }
    }
    for(; i < n; i++)
    {
        sum[0] = Tex{0};

        if(row + WIN <= m)
        {
            for(int j = 0; j < WIN; j++)
            {
                xvec[j] = x[(row + j) * T_Index(incx)];
            }
            for(int j = 0; j < WIN; j++)
            {
                sum[0]
                    += (CONJ ? conj(A[(i + 0) * T_Index(lda) + j]) : A[(i + 0) * T_Index(lda) + j])
                       * xvec[j];
            }
        }
        else if(row + m_tail <= m)
        {
            for(int j = 0; j < m_tail; j++)
            {
                xvec[j] = x[(row + j) * T_Index(incx)];
            }
            for(int j = 0; j < m_tail; j++)
            {
                sum[0]
                    += (CONJ ? conj(A[(i + 0) * T_Index(lda) + j]) : A[(i + 0) * T_Index(lda) + j])
                       * xvec[j];
            }
        }
        if(warpSize == WARP_32)
            sum[0] = rocblas_dot_block_reduce<WARP_32, NB_X>(sum[0]);
        else
            sum[0] = rocblas_dot_block_reduce<WARP_64, NB_X>(sum[0]);
        if(tx == 0)
            workspace[blockIdx.x + size_t(i) * gridDim.x] = alpha * sum[0];
    }
}

template <rocblas_int NB, rocblas_int WIN, typename Tex, typename To>
ROCBLAS_KERNEL_ILF void rocblas_gemvt_sn_reduce_calc(rocblas_int n_sums,
                                                     Tex         beta,
                                                     To* __restrict__ y,
                                                     rocblas_int incy,
                                                     Tex* __restrict__ workspace,
                                                     uint32_t batch)
{
    Tex sum{0};

    size_t offset = size_t(n_sums) * (gridDim.y * batch + blockIdx.y);
    workspace += offset;

    int inc = blockDim.x * WIN;

    int i         = threadIdx.x * WIN;
    int remainder = n_sums % WIN;
    int end       = n_sums - remainder;
    for(; i < end; i += inc) // cover all sums as 1 block
    {
        for(int j = 0; j < WIN; j++)
            sum += workspace[i + j];
    }
    if(threadIdx.x < remainder)
    {
        sum += workspace[n_sums - 1 - threadIdx.x];
    }
    if(warpSize == WARP_32)
        sum = rocblas_dot_block_reduce<WARP_32, NB>(sum);
    else
        sum = rocblas_dot_block_reduce<WARP_64, NB>(sum);

    if(threadIdx.x == 0)
    {
        y[blockIdx.y * int64_t(incy)]
            = beta ? (To)(y[blockIdx.y * int64_t(incy)] * beta + sum) : (To)sum;
    }
}

template <bool CONJ, int NB_X, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL_ILF void rocblas_gemvtsm_kernel_calc(rocblas_int m,
                                                    rocblas_int n,
                                                    Tex         alpha,
                                                    const Ti*   A,
                                                    rocblas_int lda,
                                                    const Ti*   x,
                                                    rocblas_int incx,
                                                    Tex         beta,
                                                    To*         y,
                                                    rocblas_int incy)
{
    // small m <= 64 kernel

    rocblas_int tx = threadIdx.x;

    if(!alpha)
    {
        if(beta)
            for(rocblas_int i = 0; i < n; i += NB_X)
            {
                int64_t col = i + tx;
                if(col < n)
                    y[col * incy] = (To)(y[col * incy] * beta);
            }
        else
            for(rocblas_int i = 0; i < n; i += NB_X)
            {
                int64_t col = i + tx;
                if(col < n)
                    y[col * incy] = (To)0;
            }
        return;
    }

    __shared__ Tex shared_x[64];

    if(tx < m)
        shared_x[tx] = alpha * x[tx * int64_t(incx)];
    __syncthreads();

    for(rocblas_int i = 0; i < n; i += NB_X)
    {
        rocblas_int col = i + tx;
        if(col < n)
        {
            int64_t   idx  = col * int64_t(incy);
            Tex       res  = beta ? beta * y[idx] : 0;
            const Ti* Aptr = A + col * size_t(lda);
            for(rocblas_int l = 0; l < m; ++l)
                res += shared_x[l] * (CONJ ? conj(Aptr[l]) : Aptr[l]);
            y[idx] = (To)res;
        }
    }
}

template <int DIM_X, int DIM_Y, int elements_per_thread, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_gemvn_double_buffered_kernel(rocblas_int    m,
                                     rocblas_int    n,
                                     Tex            alpha_device_host,
                                     rocblas_stride stride_alpha,
                                     const Ti*      Aa,
                                     rocblas_stride shifta,
                                     rocblas_int    lda,
                                     rocblas_stride strideA,
                                     const Ti*      xa,
                                     rocblas_stride shiftx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     To*            ya,
                                     rocblas_stride shifty,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);

        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_gemvn_double_buffered_kernel_calc<DIM_X, DIM_Y, elements_per_thread>(
            m, n, alpha, A, lda, x, incx, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <bool CONJ,
          int  DIM_X,
          int  DIM_Y,
          int  elements_per_thread,
          typename Ti,
          typename Tex,
          typename To>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_gemvt_double_buffered_kernel(rocblas_int    m,
                                     rocblas_int    n,
                                     Tex            alpha_device_host,
                                     rocblas_stride stride_alpha,
                                     const Ti*      Aa,
                                     rocblas_stride shifta,
                                     rocblas_int    lda,
                                     rocblas_stride strideA,
                                     const Ti*      xa,
                                     rocblas_stride shiftx,
                                     rocblas_int    incx,
                                     rocblas_stride stridex,
                                     To*            ya,
                                     rocblas_stride shifty,
                                     rocblas_int    incy,
                                     rocblas_stride stridey,
                                     rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);

        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_gemvt_double_buffered_kernel_calc<CONJ, DIM_X, elements_per_thread>(
            m, n, alpha, A, lda, x, incx, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <int DIM_X, int DIM_Y, typename T_Index, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_gemvn_kernel(rocblas_int    m,
                     rocblas_int    n,
                     Tex            alpha_device_host,
                     rocblas_stride stride_alpha,
                     const Ti*      Aa,
                     rocblas_stride shifta,
                     T_Index        lda,
                     rocblas_stride strideA,
                     const Ti*      xa,
                     rocblas_stride shiftx,
                     T_Index        incx,
                     rocblas_stride stridex,
                     Tex            beta_device_host,
                     rocblas_stride stride_beta,
                     To*            ya,
                     rocblas_stride shifty,
                     T_Index        incy,
                     rocblas_stride stridey,
                     rocblas_int    batch_count)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_gemvn_kernel_calc<DIM_X, DIM_Y, T_Index>(
            m, n, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

// lda always cast to size_t so single kernel
//Optimized kernel for GEMV transpose case when m or n is less than gemvt_threshold
template <bool CONJ, int NB_X, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL(NB_X)
rocblas_gemvt_kernel(rocblas_int    m,
                     rocblas_int    n,
                     Tex            alpha_device_host,
                     rocblas_stride stride_alpha,
                     const Ti*      Aa,
                     rocblas_stride shifta,
                     rocblas_int    lda,
                     rocblas_stride strideA,
                     const Ti*      xa,
                     rocblas_stride shiftx,
                     rocblas_int    incx,
                     rocblas_stride stridex,
                     Tex            beta_device_host,
                     rocblas_stride stride_beta,
                     To*            ya,
                     rocblas_stride shifty,
                     rocblas_int    incy,
                     rocblas_stride stridey,
                     rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_gemvt_kernel_calc<CONJ, NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <bool CONJ, int NB_X, typename T_Index, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL(NB_X)
rocblas_gemvt_warp_reduce_kernel(rocblas_int    m,
                                 rocblas_int    n,
                                 Tex            alpha_device_host,
                                 rocblas_stride stride_alpha,
                                 const Ti*      Aa,
                                 rocblas_stride shifta,
                                 T_Index        lda,
                                 rocblas_stride strideA,
                                 const Ti*      xa,
                                 rocblas_stride shiftx,
                                 T_Index        incx,
                                 rocblas_stride stridex,
                                 Tex            beta_device_host,
                                 rocblas_stride stride_beta,
                                 To*            ya,
                                 rocblas_stride shifty,
                                 T_Index        incy,
                                 rocblas_stride stridey,
                                 rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif
        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);

        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        auto* y = load_ptr_batch(ya, batch, shifty, stridey);

        rocblas_gemvt_reduce_kernel_calc<CONJ, NB_X, T_Index>(
            m, n, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <bool CONJ, int NB_X, int WIN, typename T_Index, typename Ti, typename U, typename Tex>
ROCBLAS_KERNEL(NB_X)
rocblas_gemvt_sn_kernel(rocblas_int    m,
                        rocblas_int    n,
                        U              alpha_device_host,
                        rocblas_stride stride_alpha,
                        const Ti*      Aa,
                        rocblas_stride shifta,
                        rocblas_int    lda,
                        rocblas_stride strideA,
                        const Ti*      xa,
                        rocblas_stride shiftx,
                        rocblas_int    incx,
                        rocblas_stride stridex,
                        Tex*           workspace,
                        rocblas_int    batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, strideA);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stridex);

        rocblas_gemvt_sn_kernel_calc<CONJ, NB_X, WIN, T_Index>(
            m, n, alpha, A, lda, x, incx, workspace, batch);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <int NB, int WIN, typename Tex, typename U, typename To>
ROCBLAS_KERNEL(NB)
rocblas_gemvt_sn_reduce(rocblas_int    n_sums,
                        U              beta_device_host,
                        rocblas_stride stride_beta,
                        To*            ya,
                        rocblas_stride shifty,
                        rocblas_int    incy,
                        rocblas_stride stridey,
                        Tex* __restrict__ workspace,
                        rocblas_int batch_count)
{
    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto* y    = load_ptr_batch(ya, batch, shifty, stridey);
        auto  beta = load_scalar(beta_device_host, batch, stride_beta);

        rocblas_gemvt_sn_reduce_calc<NB, WIN>(n_sums, beta, y, incy, workspace, batch);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <bool CONJ, int NB_X, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL(NB_X)
rocblas_gemvtsm_kernel(rocblas_int    m,
                       rocblas_int    n,
                       Tex            alpha_device_host,
                       rocblas_stride stride_alpha,
                       const Ti*      Aa,
                       rocblas_stride shifta,
                       rocblas_int    lda,
                       rocblas_stride strideA,
                       const Ti*      xa,
                       rocblas_stride shiftx,
                       rocblas_int    incx,
                       rocblas_stride stridex,
                       Tex            beta_device_host,
                       rocblas_stride stride_beta,
                       To*            ya,
                       rocblas_stride shifty,
                       rocblas_int    incy,
                       rocblas_stride stridey)
{
    auto alpha = load_scalar(alpha_device_host, blockIdx.x, stride_alpha);
    auto beta  = load_scalar(beta_device_host, blockIdx.x, stride_beta);

    if(!alpha && beta == 1)
        return;

    // batch in blockIdx.x not y
    const auto* A = cond_load_ptr_batch(alpha, Aa, blockIdx.x, shifta, strideA);
    const auto* x = cond_load_ptr_batch(alpha, xa, blockIdx.x, shiftx, stridex);

    auto* y = load_ptr_batch(ya, blockIdx.x, shifty, stridey);

    rocblas_gemvtsm_kernel_calc<CONJ, NB_X>(m, n, alpha, A, lda, x, incx, beta, y, incy);
}

template <int NB_X, int NB_BATCH, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL_ILF void rocblas_gemv_sm_mn_batched_kernel_calc(rocblas_int m,
                                                               rocblas_int n,
                                                               Tex         alpha,
                                                               const Ti*   A,
                                                               rocblas_int lda,
                                                               const Ti*   x,
                                                               rocblas_int incx,
                                                               Tex         beta,
                                                               To*         y,
                                                               rocblas_int incy)
{
    // small m && n <= 32 and large batch kernel

    const int tx = threadIdx.x; // row
    const int ty = threadIdx.y; // batch offset in batch group

    if(!alpha)
    {
        if(beta)
        {
            if(tx < m)
                y[tx * int64_t(incy)] = (To)((Tex)y[tx * int64_t(incy)] * beta);
        }
        else
        {
            if(tx < m)
                y[tx * int64_t(incy)] = (To)0;
        }
        return;
    }

    __shared__ Tex shared_x[NB_X * NB_BATCH];

    Tex* sx = (Tex*)(shared_x);
    sx += ty * NB_X;

    if(tx < n)
        sx[tx] = alpha * x[tx * int64_t(incx)];

    __syncthreads();

    if(tx < m)
    {
        Tex res = beta ? (Tex)(beta * y[tx * int64_t(incy)]) : (Tex)0;
        Tex rA[NB_X];

#pragma unroll
        for(int j = 0; j < NB_X; j++)
            rA[j] = j < n ? (Tex)A[j * size_t(lda) + tx] : (Tex)0;

#pragma unroll
        for(int j = 0; j < NB_X; j++)
            res += j < n ? rA[j] * sx[j] : (Tex)0;

        y[tx * int64_t(incy)] = (To)res;
    }
}

template <int NB_X, int NB_BATCH, typename Ti, typename Tex, typename To>
ROCBLAS_KERNEL(NB_X* NB_BATCH)
rocblas_gemvn_sm_mn_batched_kernel(rocblas_int    m,
                                   rocblas_int    n,
                                   Tex            alpha_device_host,
                                   rocblas_stride stride_alpha,
                                   const Ti*      Aa,
                                   rocblas_stride shifta,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   const Ti*      xa,
                                   rocblas_stride shiftx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   Tex            beta_device_host,
                                   rocblas_stride stride_beta,
                                   To*            ya,
                                   rocblas_stride shifty,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   rocblas_int    batch_count)
{
// gfx90a gfx942 kernels
#if defined(__gfx90a__) || defined(__gfx942__)

    const int b = blockIdx.x * blockDim.y + threadIdx.y;
    if(b >= batch_count)
        return;

    auto alpha = load_scalar(alpha_device_host, b, stride_alpha);
    auto beta  = load_scalar(beta_device_host, b, stride_beta);

    if(!alpha && beta == 1)
        return;

    const auto* A = cond_load_ptr_batch(alpha, Aa, b, shifta, strideA);
    const auto* x = cond_load_ptr_batch(alpha, xa, b, shiftx, stridex);

    auto* y = load_ptr_batch(ya, b, shifty, stridey);

    rocblas_gemv_sm_mn_batched_kernel_calc<NB_X, NB_BATCH>(
        m, n, alpha, A, lda, x, incx, beta, y, incy);

#endif
}
