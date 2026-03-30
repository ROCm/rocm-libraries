// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "cpu_ref.hpp"

void cpu_gemm(const float* a,
              const float* b,
              float* c,
              int M,
              int N,
              int K,
              int a_stride_m,
              int a_stride_k,
              int b_stride_k,
              int b_stride_n,
              int c_stride_m,
              int c_stride_n)
{
    for(int m = 0; m < M; ++m)
    {
        for(int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for(int k = 0; k < K; ++k)
            {
                sum += a[m * a_stride_m + k * a_stride_k] * b[k * b_stride_k + n * b_stride_n];
            }
            c[m * c_stride_m + n * c_stride_n] = sum;
        }
    }
}

void cpu_gemm_add_add(float* e, const float* c, const float* d0, const float* d1, int count)
{
    for(int i = 0; i < count; ++i)
        e[i] = c[i] + d0[i] + d1[i];
}

void cpu_gemm_add(float* e, const float* c, const float* d0, int count)
{
    for(int i = 0; i < count; ++i)
        e[i] = c[i] + d0[i];
}

void cpu_gemm_add_relu(float* e, const float* c, const float* d0, int count)
{
    for(int i = 0; i < count; ++i)
    {
        float val = c[i] + d0[i];
        e[i]      = val > 0.f ? val : 0.f;
    }
}
