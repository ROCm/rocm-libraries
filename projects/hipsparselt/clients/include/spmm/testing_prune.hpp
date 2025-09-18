/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#include "flops.hpp"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_init.hpp"
#include "hipsparselt_math.hpp"
#include "hipsparselt_random.hpp"
#include "hipsparselt_test.hpp"
#include "hipsparselt_vector.hpp"
#include "unit.hpp"
#include "utility.hpp"
#include <hipsparselt/hipsparselt.h>
#include <omp.h>

template <typename Ti, typename Tc>
inline Tc norm1(Ti a, Ti b)
{
    auto ac = static_cast<Tc>(a);
    auto bc = static_cast<Tc>(b);

    return static_cast<Tc>(abs(ac) + abs(bc));
}

template <typename Ti, typename Tc>
inline Tc norm1(Ti a, Ti b, Ti c, Ti d, Ti e, Ti f, Ti g, Ti h)
{
    auto ac = static_cast<Tc>(a);
    auto bc = static_cast<Tc>(b);
    auto cc = static_cast<Tc>(c);
    auto dc = static_cast<Tc>(d);
    auto ec = static_cast<Tc>(e);
    auto fc = static_cast<Tc>(f);
    auto gc = static_cast<Tc>(g);
    auto hc = static_cast<Tc>(h);
    return static_cast<Tc>(abs(ac) + abs(bc) + abs(cc) + abs(dc) + abs(ec) + abs(fc) + abs(gc)
                           + abs(hc));
}

template <typename Ti, typename Tc>
void prune_strip(const Ti* in,
                 Ti*       out,
                 int64_t   m,
                 int64_t   n,
                 int64_t   stride1,
                 int64_t   stride2,
                 int       num_batches,
                 int64_t   stride_b)
{
    for(int b = 0; b < num_batches; b++)
#pragma omp parallel for
        for(int i = 0; i < m; i++)
        {
#pragma omp parallel for
            for(int j = 0; j < n; j += 4)
            {
                size_t pos[4];
                for(int k = 0; k < 4; k++)
                {
                    pos[k] = b * stride_b + i * stride1 + (j + k) * stride2;
                }

                auto max_norm1 = static_cast<double>(-1.0f);
                int  pos_a, pos_b;
                for(int a = 0; a < 4; a++)
                {
                    for(int b = a + 1; b < 4; b++)
                    {
                        auto norm1_v = norm1<Ti, double>(in[pos[a]], in[pos[b]]);
                        if(norm1_v > max_norm1)
                        {
                            pos_a     = a;
                            pos_b     = b;
                            max_norm1 = norm1_v;
                        }
                    }
                }

                for(int k = 0; k < 4; k++)
                {
                    if(k == pos_a || k == pos_b)
                    {
                        if(in != out)
                            out[pos[k]] = in[pos[k]];
                    }
                    else
                        out[pos[k]] = static_cast<Ti>(0.0f);
                }
            }
        }
}

typedef std::pair<int, int> pos_t;

// clang-format off
const static pos_t pos_patterns[90][4] = {
    {pos_t(0, 2), pos_t(0, 2), pos_t(1, 3), pos_t(1, 3)},  //ROW#(COL#,COL#), 0(0,2), 1(0,2), 2(1,3), 3(1,3)
    {pos_t(0, 2), pos_t(0, 3), pos_t(1, 3), pos_t(1, 2)},
    {pos_t(0, 2), pos_t(0, 3), pos_t(1, 2), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(0, 1), pos_t(1, 3), pos_t(2, 3)},
    {pos_t(0, 2), pos_t(0, 1), pos_t(2, 3), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(0, 2), pos_t(1, 3), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(0, 2), pos_t(1, 2), pos_t(0, 3), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(1, 2), pos_t(1, 3), pos_t(0, 3)},
    {pos_t(0, 2), pos_t(2, 3), pos_t(0, 1), pos_t(1, 3)},
    {pos_t(0, 2), pos_t(2, 3), pos_t(1, 3), pos_t(0, 1)},
    {pos_t(0, 3), pos_t(0, 2), pos_t(1, 3), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(0, 2), pos_t(1, 2), pos_t(1, 3)},
    {pos_t(0, 3), pos_t(0, 3), pos_t(1, 2), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(0, 1), pos_t(1, 2), pos_t(2, 3)},
    {pos_t(0, 3), pos_t(0, 1), pos_t(2, 3), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(1, 3), pos_t(0, 2), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(1, 3), pos_t(1, 2), pos_t(0, 2)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(0, 3), pos_t(1, 2), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(0, 3), pos_t(2, 3), pos_t(0, 1), pos_t(1, 2)},
    {pos_t(0, 3), pos_t(2, 3), pos_t(1, 2), pos_t(0, 1)},
    {pos_t(0, 1), pos_t(0, 2), pos_t(1, 3), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(0, 2), pos_t(2, 3), pos_t(1, 3)},
    {pos_t(0, 1), pos_t(0, 3), pos_t(1, 2), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(0, 3), pos_t(2, 3), pos_t(1, 2)},
    {pos_t(0, 1), pos_t(0, 1), pos_t(2, 3), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(1, 3), pos_t(0, 2), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(1, 3), pos_t(2, 3), pos_t(0, 2)},
    {pos_t(0, 1), pos_t(1, 2), pos_t(0, 3), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(1, 2), pos_t(2, 3), pos_t(0, 3)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(0, 1), pos_t(2, 3), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(1, 3), pos_t(0, 2), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(1, 3), pos_t(0, 3), pos_t(0, 2), pos_t(1, 2)},
    {pos_t(1, 3), pos_t(0, 3), pos_t(1, 2), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(0, 1), pos_t(0, 2), pos_t(2, 3)},
    {pos_t(1, 3), pos_t(0, 1), pos_t(2, 3), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(1, 3), pos_t(0, 2), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(1, 2), pos_t(0, 2), pos_t(0, 3)},
    {pos_t(1, 3), pos_t(1, 2), pos_t(0, 3), pos_t(0, 2)},
    {pos_t(1, 3), pos_t(2, 3), pos_t(0, 2), pos_t(0, 1)},
    {pos_t(1, 3), pos_t(2, 3), pos_t(0, 1), pos_t(0, 2)},
    {pos_t(1, 2), pos_t(0, 2), pos_t(0, 3), pos_t(1, 3)},
    {pos_t(1, 2), pos_t(0, 2), pos_t(1, 3), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(0, 3), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(1, 2), pos_t(0, 1), pos_t(0, 3), pos_t(2, 3)},
    {pos_t(1, 2), pos_t(0, 1), pos_t(2, 3), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(1, 3), pos_t(0, 2), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(1, 3), pos_t(0, 3), pos_t(0, 2)},
    {pos_t(1, 2), pos_t(1, 2), pos_t(0, 3), pos_t(0, 3)},
    {pos_t(1, 2), pos_t(2, 3), pos_t(0, 3), pos_t(0, 1)},
    {pos_t(1, 2), pos_t(2, 3), pos_t(0, 1), pos_t(0, 3)},
    {pos_t(2, 3), pos_t(0, 2), pos_t(0, 1), pos_t(1, 3)},
    {pos_t(2, 3), pos_t(0, 2), pos_t(1, 3), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(0, 3), pos_t(0, 1), pos_t(1, 2)},
    {pos_t(2, 3), pos_t(0, 3), pos_t(1, 2), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(0, 2), pos_t(1, 3)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(0, 3), pos_t(1, 2)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(0, 1), pos_t(2, 3)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(1, 3), pos_t(0, 2)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(1, 2), pos_t(0, 3)},
    {pos_t(2, 3), pos_t(0, 1), pos_t(2, 3), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(1, 3), pos_t(0, 2), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(1, 3), pos_t(0, 1), pos_t(0, 2)},
    {pos_t(2, 3), pos_t(1, 2), pos_t(0, 3), pos_t(0, 1)},
    {pos_t(2, 3), pos_t(1, 2), pos_t(0, 1), pos_t(0, 3)},
    {pos_t(2, 3), pos_t(2, 3), pos_t(0, 1), pos_t(0, 1)},
};
// clang-format on

template <typename Ti, typename Tc>
void prune_tile(const Ti* in,
                Ti*       out,
                int64_t   m,
                int64_t   n,
                int64_t   stride1,
                int64_t   stride2,
                int       num_batches,
                int64_t   stride_b)
{
    for(int b = 0; b < num_batches; b++)
#pragma omp parallel for
        for(int i = 0; i < m; i += 4)
        {
#pragma omp parallel for
            for(int j = 0; j < n; j += 4)
            {

                Ti value[16];
                for(int x = 0; x < 4; x++)
                {
#pragma omp parallel for
                    for(int y = 0; y < 4; y++)
                    {
                        int64_t pos = b * stride_b + (i + x) * stride1 + (j + y) * stride2;

                        if((i + x) < m && (j + y) < n)
                        {
                            value[x * 4 + y] = in[pos];
                        }
                        else
                            value[x * 4 + y] = static_cast<Ti>(0.0f);
                    }
                }

                float  norm_res[90];
                int    max_norm_idx = 0;
                double max_norm     = -1.0;

#pragma omp parallel for
                for(int pi = 0; pi < 90; pi++)
                {
                    auto pos_pattern = pos_patterns[pi];
                    norm_res[pi]     = norm1<Ti, double>(value[pos_pattern[0].first],
                                                     value[pos_pattern[0].second],
                                                     value[1 * 4 + pos_pattern[1].first],
                                                     value[1 * 4 + pos_pattern[1].second],
                                                     value[2 * 4 + pos_pattern[2].first],
                                                     value[2 * 4 + pos_pattern[2].second],
                                                     value[3 * 4 + pos_pattern[3].first],
                                                     value[3 * 4 + pos_pattern[3].second]);
                }
                for(int pi = 0; pi < 90; pi++)
                {
                    if(max_norm < norm_res[pi])
                    {
                        max_norm     = norm_res[pi];
                        max_norm_idx = pi;
                    }
                }

                auto pos_s = pos_patterns[max_norm_idx];
                for(int x = 0; x < 4; x++)
                {
#pragma omp parallel for
                    for(int y = 0; y < 4; y++)
                    {
                        if((i + x) < m && (j + y) < n)
                        {
                            int64_t pos = b * stride_b + (i + x) * stride1 + (j + y) * stride2;
                            if(pos_s[x].first == y || pos_s[x].second == y)
                            {
                                if(in != out)
                                    out[pos] = in[pos];
                            }
                            else
                                out[pos] = static_cast<Ti>(0.0f);
                        }
                    }
                }
            }
        }
}

template <typename Ti, typename Tc>
void tile_4x4_norm1(const Ti* in,
                    Tc*       out,
                    int64_t   m,
                    int64_t   n,
                    int64_t   stride1,
                    int64_t   stride2,
                    int       num_batches,
                    int64_t   stride_b)
{
    constexpr Tc TC_ZERO = static_cast<Tc>(0);
    for(int b = 0; b < num_batches; b++)
#pragma omp parallel for
        for(int i = 0; i < m; i += 4)
        {
#pragma omp parallel for
            for(int j = 0; j < n; j += 4)
            {

                Tc value = TC_ZERO;
                for(int x = 0; x < 4; x++)
                {
#pragma omp parallel for
                    for(int y = 0; y < 4; y++)
                    {
                        int64_t pos = b * stride_b + (i + x) * stride1 + (j + y) * stride2;

                        if((i + x) < m && (j + y) < n)
                        {
                            Tc tmp = static_cast<Tc>(in[pos]);
                            value += (tmp >= TC_ZERO ? tmp : -tmp);
                        }
                    }
                }
                for(int x = 0; x < 4; x++)
                {
#pragma omp parallel for
                    for(int y = 0; y < 4; y++)
                    {
                        int64_t pos = b * stride_b + (i + x) * stride1 + (j + y) * stride2;

                        if((i + x) < m && (j + y) < n)
                        {
                            if(x == 0 && y == 0)
                            {
                                out[pos] = value;
                            }
                            else
                            {
                                out[pos] = TC_ZERO;
                            }
                        }
                    }
                }
            }
        }
}

template <typename Ti, typename To, typename Tc>
void testing_prune_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const size_t safe_size = N * lda;

    const hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    const hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // allocate memory on device
    device_vector<Ti> dA(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());

    hipsparselt_local_handle handle{arg};
    hipsparseLtHandle_t      handle_;
    hipsparseLtMatmulDescriptor_t matmul_;
    hipsparseLtMatDescriptor_t matA_;
    hipsparseOrder_t            order = HIPSPARSE_ORDER_COL;
    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, order);
    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, order);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, order);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, order);
    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    hipStream_t stream = nullptr;

    if(arg.func_version == 1)
    {
        // test version 1
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(&handle_, matmul, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(nullptr, matmul, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(handle, &matmul_, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(handle, nullptr, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(handle, matmul, nullptr, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(handle, matmul, dA, nullptr, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune(handle, matmul, dA, dA, (hipsparseLtPruneAlg_t)3, stream),
            HIPSPARSE_STATUS_NOT_SUPPORTED);
    }
    else if (arg.func_version == 2)
    {
        // test version 2
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                nullptr, matA, true, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                &handle_, matA, true, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                handle, &matA_, true, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                handle, nullptr, true, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                handle, matB, true, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_NOT_SUPPORTED);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                handle, matA, true, transA, dA, nullptr, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                handle, matA, true, transA, nullptr, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        hipsparselt_local_mat_descr matA_f32(
            hipsparselt_matrix_type_structured, handle, M, K, lda, HIP_R_32F, order);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPrune2(
                handle, matA_f32, true, transA, dA, dA, HIPSPARSELT_PRUNE_SPMMA_STRIP, stream),
            HIPSPARSE_STATUS_NOT_SUPPORTED);
    }
}

template <typename Ti, typename To, typename Tc>
void testing_prune_check_bad_arg(const Arguments& arg)
{
    const int64_t M = 128;
    const int64_t N = 128;
    const int64_t K = 128;

    const int64_t lda = 128;
    const int64_t ldb = 128;
    const int64_t ldc = 128;

    const size_t safe_size = N * lda;

    const hipsparseOperation_t transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    const hipsparseOperation_t transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;

    // allocate memory on device
    device_vector<Ti> dA(safe_size);
    CHECK_DEVICE_ALLOCATION(dA.memcheck());

    device_vector<int> d_valid(1, 1);

    hipsparselt_local_handle handle{arg};
    hipsparseLtHandle_t      handle_;
    hipsparseLtMatmulDescriptor_t matmul_;
    hipsparseLtMatDescriptor_t matA_;
    hipsparseOrder_t            order = HIPSPARSE_ORDER_COL;
    hipsparselt_local_mat_descr matA(
        hipsparselt_matrix_type_structured, handle, M, K, lda, arg.a_type, order);
    hipsparselt_local_mat_descr matB(
        hipsparselt_matrix_type_dense, handle, K, N, ldb, arg.b_type, order);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, order);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.d_type, order);
    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    hipStream_t stream = nullptr;

    if(arg.func_version == 1)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck(&handle_, matmul, dA, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck(nullptr, matmul, dA, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck(handle, &matmul_,  dA, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck(handle, nullptr,  dA, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck(handle, matmul,  nullptr, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck(handle, matmul,  dA, nullptr, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);
    }
    else if (arg.func_version == 2)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck2(&handle_, matA, true, transA, dA, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck2(nullptr, matA, true, transA, dA, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck2(handle, &matA_, true, transA, dA, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck2(handle, nullptr, true, transA, dA, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck2(handle, matB, true, transA, dA, d_valid, stream),
            HIPSPARSE_STATUS_NOT_SUPPORTED);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck2(handle, matA, true, transA, nullptr, d_valid, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);

        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtSpMMAPruneCheck2(handle, matA, true, transA, dA, nullptr, stream),
            HIPSPARSE_STATUS_INVALID_VALUE);
    }
}
template <typename Ti,
          typename To,
          typename Tc,
          hipsparselt_batch_type btype = hipsparselt_batch_type::none>
void testing_prune(const Arguments& arg)
{
    hipsparseLtPruneAlg_t prune_algo = hipsparseLtPruneAlg_t(arg.prune_algo);

    constexpr bool do_batched         = (btype == hipsparselt_batch_type::batched);
    constexpr bool do_strided_batched = (btype == hipsparselt_batch_type::strided_batched);

    void (*prune_cpu)(const Ti* in,
                      Ti*       out,
                      int64_t   m,
                      int64_t   n,
                      int64_t   stride1,
                      int64_t   stride2,
                      int       num_batches,
                      int64_t   stride_b);
    prune_cpu
        = prune_algo == HIPSPARSELT_PRUNE_SPMMA_STRIP ? prune_strip<Ti, Tc> : prune_tile<Ti, Tc>;

    hipsparseOperation_t transA = char_to_hipsparselt_operation(arg.transA);
    hipsparseOperation_t transB = char_to_hipsparselt_operation(arg.transB);

    int64_t M = arg.M;
    int64_t N = arg.N;
    int64_t K = arg.K;

    int64_t lda = arg.lda;
    int64_t ldb = arg.ldb;
    int64_t ldc = arg.ldc;
    int64_t ldd = arg.ldd;

    double gpu_time_used, cpu_time_used;
    gpu_time_used = cpu_time_used              = 0.0;
    double                   hipsparselt_error = 0.0;
    bool                     HMM               = arg.HMM;
    hipsparselt_local_handle handle{arg};
    hipStream_t              stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    hipsparseOrder_t orderA = char_to_hipsparselt_order(arg.orderA);
    hipsparseOrder_t orderB = char_to_hipsparselt_order(arg.orderB);
    hipsparseOrder_t orderC = char_to_hipsparselt_order(arg.orderC);
    hipsparseOrder_t orderD = char_to_hipsparselt_order(arg.orderD);

    int64_t A_row = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? M : K;
    int64_t A_col = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : M;
    int64_t B_row = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? K : N;
    int64_t B_col = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? N : K;

    auto adjust_row_col = [](int64_t& row, int64_t& col, int64_t ld, hipsparseOperation_t& trans) {
        if(row > ld)
        {
            if(col > ld)
                return false;
            std::swap(row, col);
            trans = (trans == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                        ? HIPSPARSE_OPERATION_TRANSPOSE
                        : HIPSPARSE_OPERATION_NON_TRANSPOSE;
        }
        return true;
    };

    if(orderA == HIPSPARSE_ORDER_COL)
    {
        if(!adjust_row_col(A_row, A_col, lda, transA))
            return;
    }
    else
    {
        if(!adjust_row_col(A_col, A_row, lda, transA))
            return;
    }
    if(orderB == HIPSPARSE_ORDER_COL)
    {
        if(!adjust_row_col(B_row, B_col, ldb, transB))
            return;
    }
    else
    {
        if(!adjust_row_col(B_col, B_row, ldb, transB))
            return;
    }

    int64_t stride_1_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : lda;
    int64_t stride_2_a = transA == HIPSPARSE_OPERATION_NON_TRANSPOSE ? lda : 1;

    int64_t stride_1_b = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? 1 : ldb;
    int64_t stride_2_b = transB == HIPSPARSE_OPERATION_NON_TRANSPOSE ? ldb : 1;

    int     num_batches = (do_batched || do_strided_batched ? arg.batch_count : 1);
    int64_t stride_a    = do_strided_batched              ? arg.stride_a
                          : orderA == HIPSPARSE_ORDER_COL ? lda * A_col
                                                          : lda * A_row;
    int64_t stride_b    = do_strided_batched              ? arg.stride_b
                          : orderB == HIPSPARSE_ORDER_COL ? ldb * B_col
                                                          : ldb * B_row;
    int64_t stride_c    = do_strided_batched              ? arg.stride_c
                          : orderC == HIPSPARSE_ORDER_COL ? ldc * N
                                                          : ldc * M;
    int64_t stride_d    = do_strided_batched              ? arg.stride_d
                          : orderD == HIPSPARSE_ORDER_COL ? ldd * N
                                                          : ldd * M;

    hipsparselt_local_mat_descr matA(arg.sparse_b ? hipsparselt_matrix_type_dense
                                                  : hipsparselt_matrix_type_structured,
                                     handle,
                                     A_row,
                                     A_col,
                                     lda,
                                     arg.a_type,
                                     orderA);
    hipsparselt_local_mat_descr matB(arg.sparse_b ? hipsparselt_matrix_type_structured
                                                  : hipsparselt_matrix_type_dense,
                                     handle,
                                     B_row,
                                     B_col,
                                     ldb,
                                     arg.b_type,
                                     orderB);
    hipsparselt_local_mat_descr matAv2(arg.sparse_b ? hipsparselt_matrix_type_dense
                                                  : hipsparselt_matrix_type_structured,
                                     handle,
                                     A_row,
                                     A_col,
                                     lda,
                                     arg.a_type,
                                     orderA);
    hipsparselt_local_mat_descr matBv2(arg.sparse_b ? hipsparselt_matrix_type_structured
                                                  : hipsparselt_matrix_type_dense,
                                     handle,
                                     B_row,
                                     B_col,
                                     ldb,
                                     arg.b_type,
                                     orderB);
    hipsparselt_local_mat_descr matC(
        hipsparselt_matrix_type_dense, handle, M, N, ldc, arg.c_type, orderC);
    hipsparselt_local_mat_descr matD(
        hipsparselt_matrix_type_dense, handle, M, N, ldd, arg.d_type, orderD);

    hipsparseStatus_t eStatusA = expected_hipsparse_status_of_matrix_size(
        arg.a_type, A_row, A_col, lda, orderA, !arg.sparse_b);
    EXPECT_HIPSPARSE_STATUS(matA.status(), eStatusA);
    if(eStatusA != HIPSPARSE_STATUS_SUCCESS)
        return;

    hipsparseStatus_t eStatusB = expected_hipsparse_status_of_matrix_size(
        arg.b_type, B_row, B_col, ldb, orderB, arg.sparse_b);
    EXPECT_HIPSPARSE_STATUS(matB.status(), eStatusB);
    if(eStatusB != HIPSPARSE_STATUS_SUCCESS)
        return;

    hipsparseStatus_t eStatusC
        = expected_hipsparse_status_of_matrix_size(arg.c_type, M, N, ldc, orderC);
    EXPECT_HIPSPARSE_STATUS(matC.status(), eStatusC);
    if(eStatusC != HIPSPARSE_STATUS_SUCCESS)
        return;

    hipsparseStatus_t eStatusD
        = expected_hipsparse_status_of_matrix_size(arg.d_type, M, N, ldd, orderD);
    EXPECT_HIPSPARSE_STATUS(matD.status(), eStatusD);
    if(eStatusD != HIPSPARSE_STATUS_SUCCESS)
        return;

    if(do_batched || do_strided_batched)
    {
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matA, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matB, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matAv2, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matBv2, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matC, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matD, HIPSPARSELT_MAT_NUM_BATCHES, &num_batches, sizeof(int)),
            HIPSPARSE_STATUS_SUCCESS);
    }

    if(do_strided_batched)
    {
        eStatusA = expected_hipsparse_status_of_matrix_stride(stride_a, A_row, A_col, lda, orderA);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matA, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_a, sizeof(int64_t)),
            eStatusA);
        if(eStatusA != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatusB = expected_hipsparse_status_of_matrix_stride(stride_b, B_row, B_col, ldb, orderB);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matB, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_b, sizeof(int64_t)),
            eStatusB);
        if(eStatusB != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatusA = expected_hipsparse_status_of_matrix_stride(stride_a, A_row, A_col, lda, orderA);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matAv2, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_a, sizeof(int64_t)),
            eStatusA);
        if(eStatusA != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatusB = expected_hipsparse_status_of_matrix_stride(stride_b, B_row, B_col, ldb, orderB);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matBv2, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_b, sizeof(int64_t)),
            eStatusB);
        if(eStatusB != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatusC = expected_hipsparse_status_of_matrix_stride(stride_c, M, N, ldc, orderC);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matC, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_c, sizeof(int64_t)),
            eStatusC);
        if(eStatusC != HIPSPARSE_STATUS_SUCCESS)
            return;
        eStatusD = expected_hipsparse_status_of_matrix_stride(stride_d, M, N, ldd, orderD);
        EXPECT_HIPSPARSE_STATUS(
            hipsparseLtMatDescSetAttribute(
                handle, matD, HIPSPARSELT_MAT_BATCH_STRIDE, &stride_d, sizeof(int64_t)),
            eStatusD);
        if(eStatusD != HIPSPARSE_STATUS_SUCCESS)
            return;
    }

    hipsparselt_local_matmul_descr matmul(
        handle, transA, transB, matA, matB, matC, matD, arg.compute_type);

    const size_t size_A           = stride_a == 0
                                        ? (orderA == HIPSPARSE_ORDER_COL ? A_col * lda : A_row * lda)
                                        : num_batches * stride_a;
    const size_t size_A_copy      = arg.unit_check || arg.norm_check ? size_A : 0;
    const size_t size_A_norm_copy = prune_algo == HIPSPARSELT_PRUNE_SPMMA_TILE ? size_A_copy : 0;

    const size_t size_B           = stride_b == 0
                                        ? (orderB == HIPSPARSE_ORDER_COL ? B_col * ldb : B_row * ldb)
                                        : num_batches * stride_b;
    const size_t size_B_copy      = arg.unit_check || arg.norm_check ? size_B : 0;
    const size_t size_B_norm_copy = prune_algo == HIPSPARSELT_PRUNE_SPMMA_TILE ? size_B_copy : 0;

    // allocate memory on device
    device_vector<Ti> dT(arg.sparse_b ? size_B : size_A, 1, HMM);
    device_vector<Ti> dT_pruned(arg.sparse_b ? size_B : size_A, 1, HMM);
    CHECK_DEVICE_ALLOCATION(dT.memcheck());
    CHECK_DEVICE_ALLOCATION(dT_pruned.memcheck());

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    host_vector<Ti>    hT(arg.sparse_b ? size_B : size_A);
    host_vector<Ti>    hT_gold(arg.sparse_b ? size_B_copy : size_A_copy);
    host_vector<Ti>    hT_1(arg.sparse_b ? size_B_copy : size_A_copy);
    host_vector<float> hT_gold_norm(arg.sparse_b ? size_B_norm_copy : size_A_norm_copy);
    host_vector<float> hT_1_norm(arg.sparse_b ? size_B_norm_copy : size_A_norm_copy);

    hipsparselt_seedrand();

    size_t T_row, T_col, ldt, stride_t;

    if(!arg.sparse_b)
    {
        T_row    = orderA == HIPSPARSE_ORDER_COL ? A_row : A_col;
        T_col    = orderA == HIPSPARSE_ORDER_COL ? A_col : A_row;
        ldt      = lda;
        stride_t = stride_a;
    }
    else
    {
        T_row    = orderB == HIPSPARSE_ORDER_COL ? B_row : B_col;
        T_col    = orderB == HIPSPARSE_ORDER_COL ? B_col : B_row;
        ldt      = ldb;
        stride_t = stride_b;
    }

    // Initial Data on CPU
    if(arg.initialization == hipsparselt_initialization::rand_int)
    {
        hipsparselt_init<Ti>(hT, T_row, T_col, ldt, stride_t, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::trig_float)
    {
        hipsparselt_init_sin<Ti>(hT, T_row, T_col, ldt, stride_t, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::hpl)
    {
        hipsparselt_init_hpl<Ti>(hT, T_row, T_col, ldt, stride_t, num_batches);
    }
    else if(arg.initialization == hipsparselt_initialization::special)
    {
        hipsparselt_init_alt_impl_big<Ti>(hT, T_row, T_col, ldt, stride_t, num_batches);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dT.transfer_from(hT));
    if(arg.inEqualOut)
        CHECK_HIP_ERROR(dT_pruned.transfer_from(hT));

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.func_version == 1)
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMAPrune(handle, matmul, arg.inEqualOut ? dT_pruned : dT, dT_pruned, prune_algo, stream),
                HIPSPARSE_STATUS_SUCCESS);
        else if(arg.func_version == 2)
            EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMAPrune2(handle,
                                                           arg.sparse_b ? matBv2 : matAv2,
                                                           !arg.sparse_b,
                                                           arg.sparse_b ? transB : transA,
                                                           arg.inEqualOut ? dT_pruned : dT,
                                                           dT_pruned,
                                                           prune_algo,
                                                           stream),
                                    HIPSPARSE_STATUS_SUCCESS);

        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_HIP_ERROR(hT_1.transfer_from(dT_pruned));

        //print_strided_batched("device", hT_1.data(), M, K, num_batches, stride_1_a, stride_2_a, stride_a);

        device_vector<int> d_valid(1, 1, HMM);
        int                h_valid = 0;
        //check the pruned matrix is sparisty 50 or not.
        if(arg.func_version == 1)
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMAPruneCheck(handle, matmul, dT_pruned, d_valid, stream),
                HIPSPARSE_STATUS_SUCCESS);
        else if(arg.func_version == 2)
            EXPECT_HIPSPARSE_STATUS(hipsparseLtSpMMAPruneCheck2(handle,
                                                                arg.sparse_b ? matBv2 : matAv2,
                                                                !arg.sparse_b,
                                                                arg.sparse_b ? transB : transA,
                                                                dT_pruned,
                                                                d_valid,
                                                                stream),
                                    HIPSPARSE_STATUS_SUCCESS);
        CHECK_HIP_ERROR(
            hipMemcpyAsync(&h_valid, d_valid, sizeof(int), hipMemcpyDeviceToHost, stream));
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        CHECK_SUCCESS(h_valid == 0);

        int64_t row, col, stride_1, stride_2, stride;
        if(!arg.sparse_b)
        {
            row      = M;
            col      = K;
            stride_1 = orderA == HIPSPARSE_ORDER_COL ? stride_1_a : stride_2_a;
            stride_2 = orderA == HIPSPARSE_ORDER_COL ? stride_2_a : stride_1_a;
        }
        else
        {
            row      = N;
            col      = K;
            stride_1 = orderB == HIPSPARSE_ORDER_COL ? stride_2_b : stride_1_b;
            stride_2 = orderB == HIPSPARSE_ORDER_COL ? stride_1_b : stride_2_b;
        }

        // now we can recycle gold matrix for reference purposes
        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync();
        }

        prune_cpu(hT, hT_gold, row, col, stride_1, stride_2, num_batches, stride_t);

        if(arg.timing)
        {
            cpu_time_used = get_time_us_no_sync() - cpu_time_used;
        }

        //releasing already used host memory
        hT = host_vector<Ti>();

        if(prune_algo == HIPSPARSELT_PRUNE_SPMMA_TILE)
        {
            tile_4x4_norm1<Ti, float>(
                hT_1, hT_1_norm, row, col, stride_1, stride_2, num_batches, stride_t);
            tile_4x4_norm1<Ti, float>(
                hT_gold, hT_gold_norm, row, col, stride_1, stride_2, num_batches, stride_t);

            // check host error and norm
            if(arg.unit_check)
            {
                unit_check_general<float>(
                    T_row, T_col, ldt, stride_t, hT_gold_norm, hT_1_norm, num_batches);
            }

            if(arg.norm_check)
            {
                hipsparselt_error = unit_check_diff<float>(
                    T_row, T_col, ldt, stride_t, hT_gold_norm, hT_1_norm, num_batches);
            }
        }
        else if(prune_algo == HIPSPARSELT_PRUNE_SPMMA_STRIP)
        {
            // check host error and norm
            if(arg.unit_check)
            {
                unit_check_general<Ti>(T_row, T_col, ldt, stride_t, hT_gold, hT_1, num_batches);
            }

            if(arg.norm_check)
            {
                hipsparselt_error
                    = unit_check_diff<Ti>(T_row, T_col, ldt, stride_t, hT_gold, hT_1, num_batches);
            }
        }
    }

    if(arg.timing)
    {
        int number_cold_calls = arg.cold_iters;
        int number_hot_calls  = arg.iters;

        for(int i = 0; i < number_cold_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMAPrune(handle, matmul, dT, dT_pruned, prune_algo, stream),
                HIPSPARSE_STATUS_SUCCESS);
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        gpu_time_used = get_time_us_sync(stream); // in microseconds
        for(int i = 0; i < number_hot_calls; i++)
        {
            EXPECT_HIPSPARSE_STATUS(
                hipsparseLtSpMMAPrune(handle, matmul, dT, dT_pruned, prune_algo, stream),
                HIPSPARSE_STATUS_SUCCESS);
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        double (*gflop_count)(int64_t m, int64_t n);
        gflop_count = (prune_algo == HIPSPARSELT_PRUNE_SPMMA_STRIP) ? prune_strip_gflop_count<Ti>
                                                                    : prune_tile_gflop_count<Ti>;

        ArgumentModel<e_transA, e_transB, e_M, e_N, e_K, e_lda, e_stride_a, e_batch_count>{}
            .log_args<float>(hipsparselt_cout,
                             arg,
                             gpu_time_used,
                             arg.sparse_b ? gflop_count(K, N) : gflop_count(M, K),
                             ArgumentLogging::NA_value,
                             cpu_time_used,
                             hipsparselt_error);
    }
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
}
