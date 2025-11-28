/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DLRM_TEST_DEVICE_COMMON_HPP
#define DLRM_TEST_DEVICE_COMMON_HPP

#include <rocwmma/internal/types.hpp>

namespace rocwmma
{

#if !ROCWMMA_TESTS_NO_HALF
    __device__ inline bool is_equal(half a, half b)
    {
        return __heq(a, b);
    }
#endif // !ROCWMMA_NO_HALF

    __device__ inline bool is_equal(float a, float b)
    {
        return a == b;
    }

    template <typename DataT>
    __device__ inline void store(DataT* dst, float* src)
    {
        if(std::is_same<DataT, float16_t>::value)
            *dst = __float2half(*src);
        else
            *dst = *src;
    }

    template <typename DataT>
    __device__ inline void store(DataT* dst, const float src)
    {
        if(std::is_same<DataT, float16_t>::value)
            *dst = __float2half(src);
        else
            *dst = src;
    }

    template <typename T, uint THREADBLOCK_SIZE>
    __global__ __launch_bounds__(THREADBLOCK_SIZE) void allclose_kernel(T*     a,
                                                                        T*     b,
                                                                        size_t num_elm,
                                                                        float* abs_diff,
                                                                        float* rel_diff,
                                                                        float* a_float,
                                                                        float* b_float)
    {
        int    tid      = threadIdx.x;
        int    nthreads = blockDim.x;
        size_t start    = (num_elm * tid) / nthreads;
        size_t end      = (num_elm * (tid + 1)) / nthreads;
        for(size_t i = start; i < end; i++)
        {
            if(!is_equal(a[i], b[i]))
            {
                float a_    = (float)a[i];
                float b_    = (float)b[i];
                a_float[i]  = a_;
                b_float[i]  = b_;
                abs_diff[i] = fabs(a_ - b_);
                if(a_ != 0.0f)
                {
                    rel_diff[i] = abs_diff[i] / fabs(a_);
                }
                else
                {
                    rel_diff[i] = 0.0f;
                }
            }
            else
            {
                abs_diff[i] = 0.0f;
                rel_diff[i] = 0.0f;
            }
        }
    }

} // namespace rocwmma

#endif // DLRM_TEST_DEVICE_COMMON_HPP
