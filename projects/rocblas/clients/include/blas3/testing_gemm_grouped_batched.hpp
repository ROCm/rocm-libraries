/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
    inline rocblas_operation grouped_gemm_toggle_n_t(rocblas_operation trans)
    {
        if(trans == rocblas_operation_none)
            return rocblas_operation_transpose;
        if(trans == rocblas_operation_transpose)
            return rocblas_operation_none;
        return trans;
    }

    template <typename Ti>
    inline Ti grouped_gemm_lda(Ti m, Ti k, rocblas_operation transA, Ti lda_override)
    {
        if(lda_override > 0)
            return lda_override;
        const Ti a_row = transA == rocblas_operation_none ? m : k;
        return std::max(a_row, Ti(1));
    }

    template <typename Ti>
    inline Ti grouped_gemm_ldb(Ti n, Ti k, rocblas_operation transB, Ti ldb_override)
    {
        if(ldb_override > 0)
            return ldb_override;
        const Ti b_row = transB == rocblas_operation_none ? k : n;
        return std::max(b_row, Ti(1));
    }

    template <typename Ti>
    inline Ti grouped_gemm_ldc(Ti m, Ti ldc_override)
    {
        if(ldc_override > 0)
            return ldc_override;
        return std::max(m, Ti(1));
    }

    template <typename T, typename Ti>
    struct grouped_gemm_test_config
    {
        Ti      group_count{};
        int64_t problem_count{};

        std::vector<rocblas_operation> transa_array;
        std::vector<rocblas_operation> transb_array;
        std::vector<Ti>                m_array;
        std::vector<Ti>                n_array;
        std::vector<Ti>                k_array;
        std::vector<Ti>                lda_array;
        std::vector<Ti>                ldb_array;
        std::vector<Ti>                ldc_array;
        std::vector<Ti>                group_size;
        std::vector<T>                 alpha_array;
        std::vector<T>                 beta_array;

        Ti max_m{};
        Ti max_n{};
        Ti max_k{};
        Ti max_lda{};
        Ti max_ldb{};
        Ti max_ldc{};
        Ti max_a_row{};
        Ti max_a_col{};
        Ti max_b_row{};
        Ti max_b_col{};
    };

    // Group g uses M+g, N+g, K+g (and lda/ldb/ldc+g when set in yaml).
    // Odd-indexed groups toggle transA/transB between N and T.
    template <typename T, typename Ti>
    grouped_gemm_test_config<T, Ti> grouped_gemm_test_config_from_arg(const Arguments& arg)
    {
        grouped_gemm_test_config<T, Ti> cfg{};
        cfg.group_count = Ti(arg.stride_x);

        const rocblas_operation base_trans_a = char2rocblas_operation(arg.transA);
        const rocblas_operation base_trans_b = char2rocblas_operation(arg.transB);
        const T                 base_alpha   = arg.get_alpha<T>();
        const T                 base_beta    = arg.get_beta<T>();

        const Ti group_count = Ti(cfg.group_count);
        cfg.transa_array.resize(group_count);
        cfg.transb_array.resize(group_count);
        cfg.m_array.resize(group_count);
        cfg.n_array.resize(group_count);
        cfg.k_array.resize(group_count);
        cfg.lda_array.resize(group_count);
        cfg.ldb_array.resize(group_count);
        cfg.ldc_array.resize(group_count);
        cfg.group_size.resize(group_count);
        cfg.alpha_array.resize(group_count);
        cfg.beta_array.resize(group_count);

        for(Ti g = 0; g < group_count; ++g)
        {
            int variation = g % 4;

            const Ti m_g = Ti(arg.M + variation);
            const Ti n_g = Ti(arg.N + variation);
            const Ti k_g = Ti(arg.K + variation);

            cfg.m_array[g] = m_g;
            cfg.n_array[g] = n_g;
            cfg.k_array[g] = k_g;

            cfg.transa_array[g]
                = (g % 2 == 0) ? base_trans_a : grouped_gemm_toggle_n_t(base_trans_a);
            cfg.transb_array[g]
                = (g % 2 == 0) ? base_trans_b : grouped_gemm_toggle_n_t(base_trans_b);

            const Ti lda_g = arg.lda > 0 ? Ti(arg.lda + variation) : Ti(0);
            const Ti ldb_g = arg.ldb > 0 ? Ti(arg.ldb + variation) : Ti(0);
            const Ti ldc_g = arg.ldc > 0 ? Ti(arg.ldc + variation) : Ti(0);

            cfg.lda_array[g] = grouped_gemm_lda(m_g, k_g, cfg.transa_array[g], lda_g);
            cfg.ldb_array[g] = grouped_gemm_ldb(n_g, k_g, cfg.transb_array[g], ldb_g);
            cfg.ldc_array[g] = grouped_gemm_ldc(m_g, ldc_g);

            cfg.group_size[g]  = Ti(std::max(arg.batch_count + variation, int64_t(0)));
            cfg.alpha_array[g] = base_alpha;
            cfg.beta_array[g]  = base_beta;
        }

        cfg.problem_count = Ti(0);
        for(Ti g = 0; g < group_count; ++g)
            cfg.problem_count += cfg.group_size[g];

        for(Ti g = 0; g < group_count; ++g)
        {
            cfg.max_m   = std::max(cfg.max_m, cfg.m_array[g]);
            cfg.max_n   = std::max(cfg.max_n, cfg.n_array[g]);
            cfg.max_k   = std::max(cfg.max_k, cfg.k_array[g]);
            cfg.max_lda = std::max(cfg.max_lda, cfg.lda_array[g]);
            cfg.max_ldb = std::max(cfg.max_ldb, cfg.ldb_array[g]);
            cfg.max_ldc = std::max(cfg.max_ldc, cfg.ldc_array[g]);

            const Ti a_row
                = cfg.transa_array[g] == rocblas_operation_none ? cfg.m_array[g] : cfg.k_array[g];
            const Ti a_col
                = cfg.transa_array[g] == rocblas_operation_none ? cfg.k_array[g] : cfg.m_array[g];
            const Ti b_row
                = cfg.transb_array[g] == rocblas_operation_none ? cfg.k_array[g] : cfg.n_array[g];
            const Ti b_col
                = cfg.transb_array[g] == rocblas_operation_none ? cfg.n_array[g] : cfg.k_array[g];

            cfg.max_a_row = std::max(cfg.max_a_row, a_row);
            cfg.max_a_col = std::max(cfg.max_a_col, a_col);
            cfg.max_b_row = std::max(cfg.max_b_row, b_row);
            cfg.max_b_col = std::max(cfg.max_b_col, b_col);
        }

        cfg.max_a_row = std::max(cfg.max_a_row, Ti(1));
        cfg.max_a_col = std::max(cfg.max_a_col, Ti(1));
        cfg.max_b_row = std::max(cfg.max_b_row, Ti(1));
        cfg.max_b_col = std::max(cfg.max_b_col, Ti(1));
        cfg.max_m     = std::max(cfg.max_m, Ti(1));
        cfg.max_n     = std::max(cfg.max_n, Ti(1));

        return cfg;
    }

}

template <typename T>
void testing_gemm_grouped_batched_bad_arg(const Arguments& arg)
{
    auto rocblas_gemm_grouped_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_gemm_grouped_batched<T, true>
                                                  : rocblas_gemm_grouped_batched<T, false>;
    auto rocblas_gemm_grouped_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_gemm_grouped_batched_64<T, true>
                                                  : rocblas_gemm_grouped_batched_64<T, false>;
    grouped_gemm_test_config<T, rocblas_int> cfg
        = grouped_gemm_test_config_from_arg<T, rocblas_int>(arg);
    grouped_gemm_test_config<T, int64_t> cfg_64{};
    if(arg.api & c_API_64)
        cfg_64 = grouped_gemm_test_config_from_arg<T, int64_t>(arg);

    const int64_t problem_count = (cfg.problem_count);

    const size_t safe_size        = std::max(size_t(cfg.max_a_row) * size_t(cfg.max_a_col),
                                      size_t(cfg.max_b_row) * size_t(cfg.max_b_col));
    const size_t padded_safe_size = std::max(safe_size, size_t(cfg.max_m) * size_t(cfg.max_n));

    rocblas_local_handle handle{arg};

    DEVICE_MEMCHECK(device_batch_vector<T>, dA, (padded_safe_size, 1, problem_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dB, (padded_safe_size, 1, problem_count));
    DEVICE_MEMCHECK(device_batch_vector<T>, dC, (padded_safe_size, 1, problem_count));

    std::vector<rocblas_int> bad_m_array(cfg.m_array);
    bad_m_array[0] = -1;
    std::vector<rocblas_int> bad_group_size(cfg.group_size);
    bad_group_size[0] = -1;

    std::vector<int64_t> bad_m_array_64;
    std::vector<int64_t> bad_group_size_64;
    if(arg.api & c_API_64)
    {
        bad_m_array_64       = cfg_64.m_array;
        bad_m_array_64[0]    = -1;
        bad_group_size_64    = cfg_64.group_size;
        bad_group_size_64[0] = -1;
    }

    if(arg.api & c_API_64)
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(nullptr,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 cfg_64.m_array.data(),
                                                                 cfg_64.n_array.data(),
                                                                 cfg_64.k_array.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA.ptr_on_device(),
                                                                 cfg_64.lda_array.data(),
                                                                 dB.ptr_on_device(),
                                                                 cfg_64.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 cfg_64.ldc_array.data(),
                                                                 cfg_64.group_count,
                                                                 cfg_64.group_size.data()),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 cfg_64.m_array.data(),
                                                                 cfg_64.n_array.data(),
                                                                 cfg_64.k_array.data(),
                                                                 nullptr,
                                                                 dA.ptr_on_device(),
                                                                 cfg_64.lda_array.data(),
                                                                 dB.ptr_on_device(),
                                                                 cfg_64.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 cfg_64.ldc_array.data(),
                                                                 cfg_64.group_count,
                                                                 cfg_64.group_size.data()),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 bad_m_array_64.data(),
                                                                 cfg_64.n_array.data(),
                                                                 cfg_64.k_array.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA.ptr_on_device(),
                                                                 cfg_64.lda_array.data(),
                                                                 dB.ptr_on_device(),
                                                                 cfg_64.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 cfg_64.ldc_array.data(),
                                                                 cfg_64.group_count,
                                                                 cfg_64.group_size.data()),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 cfg_64.m_array.data(),
                                                                 cfg_64.n_array.data(),
                                                                 cfg_64.k_array.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA.ptr_on_device(),
                                                                 cfg_64.lda_array.data(),
                                                                 dB.ptr_on_device(),
                                                                 cfg_64.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 cfg_64.ldc_array.data(),
                                                                 cfg_64.group_count,
                                                                 bad_group_size_64.data()),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 cfg_64.m_array.data(),
                                                                 cfg_64.n_array.data(),
                                                                 cfg_64.k_array.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA.ptr_on_device(),
                                                                 cfg_64.lda_array.data(),
                                                                 dB.ptr_on_device(),
                                                                 cfg_64.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC.ptr_on_device(),
                                                                 cfg_64.ldc_array.data(),
                                                                 int64_t(-1),
                                                                 cfg_64.group_size.data()),
                              rocblas_status_invalid_size);

        // If group_count==0, then all pointers can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 int64_t(0),
                                                                 nullptr),
                              rocblas_status_success);
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
                                                              cfg.group_count,
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
                                                              cfg.group_count,
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
                                                              cfg.group_count,
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
                                                              cfg.group_count,
                                                              bad_group_size.data()),
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
                                                              -1,
                                                              cfg.group_size.data()),
                              rocblas_status_invalid_size);

        // If group_count==0, then all pointers can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_fn(handle,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              nullptr,
                                                              0,
                                                              nullptr),
                              rocblas_status_success);
    }
}

template <typename T>
void testing_gemm_grouped_batched(const Arguments& arg)
{
    auto rocblas_gemm_grouped_batched_fn    = arg.api & c_API_FORTRAN
                                                  ? rocblas_gemm_grouped_batched<T, true>
                                                  : rocblas_gemm_grouped_batched<T, false>;
    auto rocblas_gemm_grouped_batched_fn_64 = arg.api & c_API_FORTRAN
                                                  ? rocblas_gemm_grouped_batched_64<T, true>
                                                  : rocblas_gemm_grouped_batched_64<T, false>;
    grouped_gemm_test_config<T, rocblas_int> cfg
        = grouped_gemm_test_config_from_arg<T, rocblas_int>(arg);
    grouped_gemm_test_config<T, int64_t> cfg_64{};
    if(arg.api & c_API_64)
        cfg_64 = grouped_gemm_test_config_from_arg<T, int64_t>(arg);

    // we aren't testing these with > 32bit groups for now
    const int64_t group_count   = (cfg.group_count);
    const int64_t problem_count = (cfg.problem_count);

    double rocblas_error = 0.0, error_hst_ptr = 0.0, error_dev_ptr = 0.0;

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
    for(int64_t g = 0; g < group_count; ++g)
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
        int64_t idx = 0;
        for(int64_t g = 0; g < group_count; ++g)
        {
            for(int64_t p = 0; p < cfg.group_size[g]; ++p, ++idx)
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
            CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                   cfg.transa_array.data(),
                                                                   cfg.transb_array.data(),
                                                                   cfg_64.m_array.data(),
                                                                   cfg_64.n_array.data(),
                                                                   cfg_64.k_array.data(),
                                                                   alpha_ptr,
                                                                   dA.ptr_on_device(),
                                                                   cfg_64.lda_array.data(),
                                                                   dB.ptr_on_device(),
                                                                   cfg_64.ldb_array.data(),
                                                                   beta_ptr,
                                                                   dC.ptr_on_device(),
                                                                   cfg_64.ldc_array.data(),
                                                                   cfg_64.group_count,
                                                                   cfg_64.group_size.data()));
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

    const auto compare_to_gold = [&] {
        if(arg.unit_check)
        {
            int64_t idx = 0;
            for(int64_t g = 0; g < group_count; ++g)
            {
                for(int64_t p = 0; p < cfg.group_size[g]; ++p, ++idx)
                {
                    if(std::is_same_v<
                           T,
                           rocblas_half> && (rocblas_handle(handle)->getArchMajor() == 11))
                    {
                        const double tol = cfg.k_array[g] * sum_error_tolerance_for_gfx11<T, T, T>;
                        near_check_general<T>(cfg.m_array[g],
                                              cfg.n_array[g],
                                              cfg.ldc_array[g],
                                              hC_gold[idx],
                                              hC[idx],
                                              tol);
                    }
                    else if(reduction_requires_near<T>(arg, cfg.k_array[g]))
                    {
                        const double tol = cfg.k_array[g] * sum_error_tolerance<T>;
                        near_check_general<T>(cfg.m_array[g],
                                              cfg.n_array[g],
                                              cfg.ldc_array[g],
                                              hC_gold[idx],
                                              hC[idx],
                                              tol);
                    }
                    else
                    {
                        unit_check_general<T>(cfg.m_array[g],
                                              cfg.n_array[g],
                                              cfg.ldc_array[g],
                                              hC_gold[idx],
                                              hC[idx]);
                    }
                }
            }
        }

        double error = 0;
        if(arg.norm_check)
        {
            int64_t idx = 0;
            for(int64_t g = 0; g < group_count; ++g)
            {
                for(int64_t p = 0; p < cfg.group_size[g]; ++p, ++idx)
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
        }
        return error;
    };

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            run_grouped_gemm(cfg.alpha_array.data(), cfg.beta_array.data());
            CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        if(arg.pointer_mode_device)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dC.transfer_from(hC_init));
            CHECK_HIP_ERROR(d_alpha.transfer_from(h_alpha));
            CHECK_HIP_ERROR(d_beta.transfer_from(h_beta));
            run_grouped_gemm(d_alpha, d_beta);
        }

        if(arg.pointer_mode_host)
        {
            error_hst_ptr = compare_to_gold();
        }
        if(arg.pointer_mode_device)
        {
            CHECK_HIP_ERROR(hC.transfer_from(dC));
            error_dev_ptr = compare_to_gold();
        }
        rocblas_error = error_dev_ptr > error_hst_ptr ? error_dev_ptr : error_hst_ptr;
    }

    if(arg.timing && arg.api != INTERNAL)
    {
        double gpu_time_used     = 0.0;
        int    number_cold_calls = arg.cold_iters;
        int    total_calls       = number_cold_calls + arg.iters;

        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        hipStream_t stream;
        CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

        FrequencyMonitor& freq_monitor = getFrequencyMonitor();
        freq_monitor.start();

        for(int i = 0; i < total_calls; i++)
        {
            if(i == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream); // in microseconds

            if(arg.api & c_API_64)
            {
                CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_fn_64(handle,
                                                                       cfg.transa_array.data(),
                                                                       cfg.transb_array.data(),
                                                                       cfg_64.m_array.data(),
                                                                       cfg_64.n_array.data(),
                                                                       cfg_64.k_array.data(),
                                                                       cfg.alpha_array.data(),
                                                                       dA.ptr_on_device(),
                                                                       cfg_64.lda_array.data(),
                                                                       dB.ptr_on_device(),
                                                                       cfg_64.ldb_array.data(),
                                                                       cfg.beta_array.data(),
                                                                       dC.ptr_on_device(),
                                                                       cfg_64.ldc_array.data(),
                                                                       cfg_64.group_count,
                                                                       cfg_64.group_size.data()));
            }
            else
            {
                CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_fn(handle,
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
                                                                    cfg.group_size.data()));
            }
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        freq_monitor.stop();

        double gflop_count = 0.0;
        for(int64_t g = 0; g < group_count; ++g)
            gflop_count += cfg.group_size[g]
                           * gemm_gflop_count<T>(cfg.m_array[g], cfg.n_array[g], cfg.k_array[g]);

        ArgumentModel<e_transA,
                      e_transB,
                      e_M,
                      e_N,
                      e_K,
                      e_alpha,
                      e_lda,
                      e_beta,
                      e_ldb,
                      e_ldc,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<T>(rocblas_cout,
                         arg,
                         gpu_time_used,
                         gflop_count,
                         ArgumentLogging::NA_value,
                         ArgumentLogging::NA_value,
                         ArgumentLogging::NA_value,
                         ArgumentLogging::NA_value,
                         ArgumentLogging::NA_value,
                         ArgumentLogging::NA_value);
    }
}
