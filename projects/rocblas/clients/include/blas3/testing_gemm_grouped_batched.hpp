/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * ************************************************************************ */

#pragma once

#include "blas3/rocblas_gemm.hpp"
#include "frequency_monitor.hpp"
#include "testing_common.hpp"

#include <algorithm>
#include <vector>

namespace
{
    constexpr rocblas_int k_max_grouped_gemm_groups = 8;

    inline rocblas_operation grouped_gemm_toggle_n_t(rocblas_operation trans)
    {
        if(trans == rocblas_operation_none)
            return rocblas_operation_transpose;
        if(trans == rocblas_operation_transpose)
            return rocblas_operation_none;
        return trans;
    }

    inline rocblas_int grouped_gemm_lda(rocblas_int       m,
                                        rocblas_int       k,
                                        rocblas_operation transA,
                                        rocblas_int       lda_override)
    {
        if(lda_override > 0)
            return lda_override;
        rocblas_int a_row = transA == rocblas_operation_none ? m : k;
        return std::max(a_row, rocblas_int(1));
    }

    inline rocblas_int grouped_gemm_ldb(rocblas_int       n,
                                        rocblas_int       k,
                                        rocblas_operation transB,
                                        rocblas_int       ldb_override)
    {
        if(ldb_override > 0)
            return ldb_override;
        rocblas_int b_row = transB == rocblas_operation_none ? k : n;
        return std::max(b_row, rocblas_int(1));
    }

    inline rocblas_int grouped_gemm_ldc(rocblas_int m, rocblas_int ldc_override)
    {
        if(ldc_override > 0)
            return ldc_override;
        return std::max(m, rocblas_int(1));
    }

    template <typename T>
    struct grouped_gemm_test_config
    {
        rocblas_int group_count{};
        rocblas_int problem_count{};

        std::vector<rocblas_operation> transa_array;
        std::vector<rocblas_operation> transb_array;
        std::vector<rocblas_int>       m_array;
        std::vector<rocblas_int>       n_array;
        std::vector<rocblas_int>       k_array;
        std::vector<rocblas_int>       lda_array;
        std::vector<rocblas_int>       ldb_array;
        std::vector<rocblas_int>       ldc_array;
        std::vector<rocblas_int>       group_size;
        std::vector<T>                 alpha_array;
        std::vector<T>                 beta_array;

        rocblas_int max_m{};
        rocblas_int max_n{};
        rocblas_int max_k{};
        rocblas_int max_lda{};
        rocblas_int max_ldb{};
        rocblas_int max_ldc{};
        rocblas_int max_a_row{};
        rocblas_int max_a_col{};
        rocblas_int max_b_row{};
        rocblas_int max_b_col{};
    };

    // Group g uses M+g, N+g, K+g (and lda/ldb/ldc+g when set in yaml).
    // Odd-indexed groups toggle transA/transB between N and T.
    template <typename T>
    grouped_gemm_test_config<T> grouped_gemm_test_config_from_arg(const Arguments& arg)
    {
        grouped_gemm_test_config<T> cfg{};
        cfg.group_count
            = std::min(std::max(arg.stride_x, int64_t(1)), int64_t(k_max_grouped_gemm_groups));

        const rocblas_operation base_trans_a = char2rocblas_operation(arg.transA);
        const rocblas_operation base_trans_b = char2rocblas_operation(arg.transB);
        const T                 base_alpha   = arg.get_alpha<T>();
        const T                 base_beta    = arg.get_beta<T>();

        cfg.transa_array.resize(cfg.group_count);
        cfg.transb_array.resize(cfg.group_count);
        cfg.m_array.resize(cfg.group_count);
        cfg.n_array.resize(cfg.group_count);
        cfg.k_array.resize(cfg.group_count);
        cfg.lda_array.resize(cfg.group_count);
        cfg.ldb_array.resize(cfg.group_count);
        cfg.ldc_array.resize(cfg.group_count);
        cfg.group_size.resize(cfg.group_count);
        cfg.alpha_array.resize(cfg.group_count);
        cfg.beta_array.resize(cfg.group_count);

        for(rocblas_int g = 0; g < cfg.group_count; ++g)
        {
            cfg.m_array[g] = rocblas_int(arg.M + g);
            cfg.n_array[g] = rocblas_int(arg.N + g);
            cfg.k_array[g] = rocblas_int(arg.K + g);

            cfg.transa_array[g]
                = (g % 2 == 0) ? base_trans_a : grouped_gemm_toggle_n_t(base_trans_a);
            cfg.transb_array[g]
                = (g % 2 == 0) ? base_trans_b : grouped_gemm_toggle_n_t(base_trans_b);

            const rocblas_int lda_g = arg.lda > 0 ? rocblas_int(arg.lda + g) : 0;
            const rocblas_int ldb_g = arg.ldb > 0 ? rocblas_int(arg.ldb + g) : 0;
            const rocblas_int ldc_g = arg.ldc > 0 ? rocblas_int(arg.ldc + g) : 0;

            cfg.lda_array[g]
                = grouped_gemm_lda(cfg.m_array[g], cfg.k_array[g], cfg.transa_array[g], lda_g);
            cfg.ldb_array[g]
                = grouped_gemm_ldb(cfg.n_array[g], cfg.k_array[g], cfg.transb_array[g], ldb_g);
            cfg.ldc_array[g] = grouped_gemm_ldc(cfg.m_array[g], ldc_g);

            cfg.group_size[g]  = std::max(arg.batch_count, int64_t(0));
            cfg.alpha_array[g] = base_alpha;
            cfg.beta_array[g]  = base_beta;
        }

        cfg.problem_count = 0;
        for(rocblas_int g = 0; g < cfg.group_count; ++g)
            cfg.problem_count += cfg.group_size[g];

        for(rocblas_int g = 0; g < cfg.group_count; ++g)
        {
            cfg.max_m   = std::max(cfg.max_m, cfg.m_array[g]);
            cfg.max_n   = std::max(cfg.max_n, cfg.n_array[g]);
            cfg.max_k   = std::max(cfg.max_k, cfg.k_array[g]);
            cfg.max_lda = std::max(cfg.max_lda, cfg.lda_array[g]);
            cfg.max_ldb = std::max(cfg.max_ldb, cfg.ldb_array[g]);
            cfg.max_ldc = std::max(cfg.max_ldc, cfg.ldc_array[g]);

            rocblas_int a_row
                = cfg.transa_array[g] == rocblas_operation_none ? cfg.m_array[g] : cfg.k_array[g];
            rocblas_int a_col
                = cfg.transa_array[g] == rocblas_operation_none ? cfg.k_array[g] : cfg.m_array[g];
            rocblas_int b_row
                = cfg.transb_array[g] == rocblas_operation_none ? cfg.k_array[g] : cfg.n_array[g];
            rocblas_int b_col
                = cfg.transb_array[g] == rocblas_operation_none ? cfg.n_array[g] : cfg.k_array[g];

            cfg.max_a_row = std::max(cfg.max_a_row, a_row);
            cfg.max_a_col = std::max(cfg.max_a_col, a_col);
            cfg.max_b_row = std::max(cfg.max_b_row, b_row);
            cfg.max_b_col = std::max(cfg.max_b_col, b_col);
        }

        cfg.max_a_row = std::max(cfg.max_a_row, rocblas_int(1));
        cfg.max_a_col = std::max(cfg.max_a_col, rocblas_int(1));
        cfg.max_b_row = std::max(cfg.max_b_row, rocblas_int(1));
        cfg.max_b_col = std::max(cfg.max_b_col, rocblas_int(1));
        cfg.max_m     = std::max(cfg.max_m, rocblas_int(1));
        cfg.max_n     = std::max(cfg.max_n, rocblas_int(1));

        return cfg;
    }

    inline std::vector<int64_t> grouped_gemm_to_int64(const std::vector<rocblas_int>& v)
    {
        return std::vector<int64_t>(v.begin(), v.end());
    }
}

template <typename T>
void testing_gemm_grouped_batched_bad_arg(const Arguments& arg)
{
    auto                        rocblas_gemm_grouped_batched_fn    = arg.api & c_API_FORTRAN
                                                                         ? rocblas_gemm_grouped_batched<T, true>
                                                                         : rocblas_gemm_grouped_batched<T, false>;
    auto                        rocblas_gemm_grouped_batched_fn_64 = arg.api & c_API_FORTRAN
                                                                         ? rocblas_gemm_grouped_batched_64<T, true>
                                                                         : rocblas_gemm_grouped_batched_64<T, false>;
    grouped_gemm_test_config<T> cfg           = grouped_gemm_test_config_from_arg<T>(arg);
    const rocblas_int           group_count   = cfg.group_count;
    const rocblas_int           problem_count = cfg.problem_count;

    const size_t safe_size        = std::max(size_t(cfg.max_a_row) * size_t(cfg.max_a_col),
                                      size_t(cfg.max_b_row) * size_t(cfg.max_b_col));
    const size_t padded_safe_size = std::max(safe_size, size_t(cfg.max_m) * size_t(cfg.max_n));

    rocblas_local_handle handle{arg};

    DEVICE_MEMCHECK(device_batch_vector<T>, dA, (padded_safe_size, 1, std::max(problem_count, 1)));
    DEVICE_MEMCHECK(device_batch_vector<T>, dB, (padded_safe_size, 1, std::max(problem_count, 1)));
    DEVICE_MEMCHECK(device_batch_vector<T>, dC, (padded_safe_size, 1, std::max(problem_count, 1)));

    const std::vector<int64_t> m_array_64    = grouped_gemm_to_int64(cfg.m_array);
    const std::vector<int64_t> n_array_64    = grouped_gemm_to_int64(cfg.n_array);
    const std::vector<int64_t> k_array_64    = grouped_gemm_to_int64(cfg.k_array);
    const std::vector<int64_t> lda_array_64  = grouped_gemm_to_int64(cfg.lda_array);
    const std::vector<int64_t> ldb_array_64  = grouped_gemm_to_int64(cfg.ldb_array);
    const std::vector<int64_t> ldc_array_64  = grouped_gemm_to_int64(cfg.ldc_array);
    const std::vector<int64_t> group_size_64 = grouped_gemm_to_int64(cfg.group_size);
    std::vector<rocblas_int>   bad_m_array(cfg.m_array);
    bad_m_array[0]                            = -1;
    const std::vector<int64_t> bad_m_array_64 = grouped_gemm_to_int64(bad_m_array);
    std::vector<rocblas_int>   bad_group_size(cfg.group_size);
    bad_group_size[0]                            = -1;
    const std::vector<int64_t> bad_group_size_64 = grouped_gemm_to_int64(bad_group_size);

    if(arg.api & c_API_64)
    {
        const int64_t group_count_64 = group_count;
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(nullptr,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 m_array_64.data(),
                                                                 n_array_64.data(),
                                                                 k_array_64.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA.ptr_on_device(),
                                                                 lda_array_64.data(),
                                                                 dB.ptr_on_device(),
                                                                 ldb_array_64.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 ldc_array_64.data(),
                                                                 group_count_64,
                                                                 group_size_64.data()),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 m_array_64.data(),
                                                                 n_array_64.data(),
                                                                 k_array_64.data(),
                                                                 nullptr,
                                                                 dA.ptr_on_device(),
                                                                 lda_array_64.data(),
                                                                 dB.ptr_on_device(),
                                                                 ldb_array_64.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 ldc_array_64.data(),
                                                                 group_count_64,
                                                                 group_size_64.data()),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 bad_m_array_64.data(),
                                                                 n_array_64.data(),
                                                                 k_array_64.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA.ptr_on_device(),
                                                                 lda_array_64.data(),
                                                                 dB.ptr_on_device(),
                                                                 ldb_array_64.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 ldc_array_64.data(),
                                                                 group_count_64,
                                                                 group_size_64.data()),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 m_array_64.data(),
                                                                 n_array_64.data(),
                                                                 k_array_64.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA.ptr_on_device(),
                                                                 lda_array_64.data(),
                                                                 dB.ptr_on_device(),
                                                                 ldb_array_64.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 ldc_array_64.data(),
                                                                 group_count_64,
                                                                 bad_group_size_64.data()),
                              rocblas_status_invalid_size);
    }
    else
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn(nullptr,
                                                              cfg.transa_array.data(),
                                                              cfg.transb_array.data(),
                                                              cfg.m_array.data(),
                                                              cfg.n_array.data(),
                                                              cfg.k_array.data(),
                                                              cfg.alpha_array.data(),
                                                              dA.ptr_on_device(),
                                                              cfg.lda_array.data(),
                                                              dB.ptr_on_device(),
                                                              cfg.ldb_array.data(),
                                                              cfg.beta_array.data(),
                                                              dC.ptr_on_device(),
                                                              cfg.ldc_array.data(),
                                                              group_count,
                                                              cfg.group_size.data()),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn(handle,
                                                              cfg.transa_array.data(),
                                                              cfg.transb_array.data(),
                                                              cfg.m_array.data(),
                                                              cfg.n_array.data(),
                                                              cfg.k_array.data(),
                                                              nullptr,
                                                              dA.ptr_on_device(),
                                                              cfg.lda_array.data(),
                                                              dB.ptr_on_device(),
                                                              cfg.ldb_array.data(),
                                                              cfg.beta_array.data(),
                                                              dC.ptr_on_device(),
                                                              cfg.ldc_array.data(),
                                                              group_count,
                                                              cfg.group_size.data()),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn(handle,
                                                              cfg.transa_array.data(),
                                                              cfg.transb_array.data(),
                                                              bad_m_array.data(),
                                                              cfg.n_array.data(),
                                                              cfg.k_array.data(),
                                                              cfg.alpha_array.data(),
                                                              dA.ptr_on_device(),
                                                              cfg.lda_array.data(),
                                                              dB.ptr_on_device(),
                                                              cfg.ldb_array.data(),
                                                              cfg.beta_array.data(),
                                                              dC.ptr_on_device(),
                                                              cfg.ldc_array.data(),
                                                              group_count,
                                                              cfg.group_size.data()),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn(handle,
                                                              cfg.transa_array.data(),
                                                              cfg.transb_array.data(),
                                                              cfg.m_array.data(),
                                                              cfg.n_array.data(),
                                                              cfg.k_array.data(),
                                                              cfg.alpha_array.data(),
                                                              dA.ptr_on_device(),
                                                              cfg.lda_array.data(),
                                                              dB.ptr_on_device(),
                                                              cfg.ldb_array.data(),
                                                              cfg.beta_array.data(),
                                                              dC.ptr_on_device(),
                                                              cfg.ldc_array.data(),
                                                              group_count,
                                                              bad_group_size.data()),
                              rocblas_status_invalid_size);
    }
}

template <typename T>
void testing_gemm_grouped_batched(const Arguments& arg)
{
    auto                        rocblas_gemm_grouped_batched_fn    = arg.api & c_API_FORTRAN
                                                                         ? rocblas_gemm_grouped_batched<T, true>
                                                                         : rocblas_gemm_grouped_batched<T, false>;
    auto                        rocblas_gemm_grouped_batched_fn_64 = arg.api & c_API_FORTRAN
                                                                         ? rocblas_gemm_grouped_batched_64<T, true>
                                                                         : rocblas_gemm_grouped_batched_64<T, false>;
    grouped_gemm_test_config<T> cfg           = grouped_gemm_test_config_from_arg<T>(arg);
    const rocblas_int           group_count   = cfg.group_count;
    const rocblas_int           problem_count = cfg.problem_count;

    if(group_count <= 0 || problem_count <= 0)
        return;

    rocblas_local_handle handle{arg};

    HOST_MEMCHECK(
        host_batch_matrix<T>, hA, (cfg.max_a_row, cfg.max_a_col, cfg.max_lda, problem_count));
    HOST_MEMCHECK(
        host_batch_matrix<T>, hB, (cfg.max_b_row, cfg.max_b_col, cfg.max_ldb, problem_count));
    HOST_MEMCHECK(host_batch_matrix<T>, hC, (cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count));
    HOST_MEMCHECK(
        host_batch_matrix<T>, hC_init, (cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count));
    HOST_MEMCHECK(
        host_batch_matrix<T>, hC_gold, (cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count));
    HOST_MEMCHECK(host_vector<T>, h_alpha, (group_count));
    HOST_MEMCHECK(host_vector<T>, h_beta, (group_count));
    for(rocblas_int g = 0; g < group_count; ++g)
    {
        h_alpha[g] = cfg.alpha_array[g];
        h_beta[g]  = cfg.beta_array[g];
    }

    DEVICE_MEMCHECK(
        device_batch_matrix<T>, dA, (cfg.max_a_row, cfg.max_a_col, cfg.max_lda, problem_count));
    DEVICE_MEMCHECK(
        device_batch_matrix<T>, dB, (cfg.max_b_row, cfg.max_b_col, cfg.max_ldb, problem_count));
    DEVICE_MEMCHECK(device_batch_matrix<T>, dC, (cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count));
    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (group_count));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (group_count));

    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
    hC_init.copy_from(hC);
    hC_gold.copy_from(hC);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        rocblas_int idx = 0;
        for(rocblas_int g = 0; g < group_count; ++g)
        {
            for(rocblas_int p = 0; p < cfg.group_size[g]; ++p, ++idx)
            {
                ref_gemm<T>(cfg.transa_array[g],
                            cfg.transb_array[g],
                            cfg.m_array[g],
                            cfg.n_array[g],
                            cfg.k_array[g],
                            cfg.alpha_array[g],
                            hA[idx],
                            cfg.lda_array[g],
                            hB[idx],
                            cfg.ldb_array[g],
                            cfg.beta_array[g],
                            hC_gold[idx],
                            cfg.ldc_array[g]);
            }
        }
    }

    const auto run_grouped_gemm = [&](const T* alpha_ptr, const T* beta_ptr) {
        if(arg.api & c_API_64)
        {
            const std::vector<int64_t> m_array_64     = grouped_gemm_to_int64(cfg.m_array);
            const std::vector<int64_t> n_array_64     = grouped_gemm_to_int64(cfg.n_array);
            const std::vector<int64_t> k_array_64     = grouped_gemm_to_int64(cfg.k_array);
            const std::vector<int64_t> lda_array_64   = grouped_gemm_to_int64(cfg.lda_array);
            const std::vector<int64_t> ldb_array_64   = grouped_gemm_to_int64(cfg.ldb_array);
            const std::vector<int64_t> ldc_array_64   = grouped_gemm_to_int64(cfg.ldc_array);
            const std::vector<int64_t> group_size_64  = grouped_gemm_to_int64(cfg.group_size);
            const int64_t              group_count_64 = group_count;
            CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                   cfg.transa_array.data(),
                                                                   cfg.transb_array.data(),
                                                                   m_array_64.data(),
                                                                   n_array_64.data(),
                                                                   k_array_64.data(),
                                                                   alpha_ptr,
                                                                   dA.ptr_on_device(),
                                                                   lda_array_64.data(),
                                                                   dB.ptr_on_device(),
                                                                   ldb_array_64.data(),
                                                                   beta_ptr,
                                                                   dC.ptr_on_device(),
                                                                   ldc_array_64.data(),
                                                                   group_count_64,
                                                                   group_size_64.data()));
        }
        else
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_fn(handle,
                                                                cfg.transa_array.data(),
                                                                cfg.transb_array.data(),
                                                                cfg.m_array.data(),
                                                                cfg.n_array.data(),
                                                                cfg.k_array.data(),
                                                                alpha_ptr,
                                                                dA.ptr_on_device(),
                                                                cfg.lda_array.data(),
                                                                dB.ptr_on_device(),
                                                                cfg.ldb_array.data(),
                                                                beta_ptr,
                                                                dC.ptr_on_device(),
                                                                cfg.ldc_array.data(),
                                                                group_count,
                                                                cfg.group_size.data()));
        }
    };

    const auto check_result = [&] {
        if(arg.unit_check)
        {
            rocblas_int idx = 0;
            for(rocblas_int g = 0; g < group_count; ++g)
            {
                for(rocblas_int p = 0; p < cfg.group_size[g]; ++p, ++idx)
                {
                    unit_check_general<T>(
                        cfg.m_array[g], cfg.n_array[g], cfg.ldc_array[g], hC_gold[idx], hC[idx]);
                }
            }
        }

        if(arg.norm_check)
        {
            double      error = 0;
            rocblas_int idx   = 0;
            for(rocblas_int g = 0; g < group_count; ++g)
            {
                for(rocblas_int p = 0; p < cfg.group_size[g]; ++p, ++idx)
                {
                    error = std::max(error,
                                     std::abs(norm_check_general<T>('F',
                                                                    cfg.m_array[g],
                                                                    cfg.n_array[g],
                                                                    cfg.ldc_array[g],
                                                                    hC_gold[idx],
                                                                    hC[idx])));
                }
            }
            ASSERT_NEAR(error, 0.0, 1e-10);
        }
    };

    if(arg.pointer_mode_host)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        run_grouped_gemm(cfg.alpha_array.data(), cfg.beta_array.data());
        CHECK_HIP_ERROR(hC.transfer_from(dC));
        check_result();
    }

    if(arg.pointer_mode_device)
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC_init));
        CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
        CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));
        run_grouped_gemm(d_alpha, d_beta);
        CHECK_HIP_ERROR(hC.transfer_from(dC));
        check_result();
    }
}
