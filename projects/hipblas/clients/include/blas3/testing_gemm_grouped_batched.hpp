/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
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
 * ************************************************************************ */

#pragma once

#include "testing_common.hpp"

#include <algorithm>
#include <vector>

using hipblasGemmGroupedBatchedModel = ArgumentModel<e_a_type,
                                                     e_transA,
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
                                                     e_batch_count>;

inline void testname_gemm_grouped_batched(const Arguments& arg, std::string& name)
{
    hipblasGemmGroupedBatchedModel{}.test_name(arg, name);
}

namespace
{
    constexpr int k_max_grouped_gemm_groups = 8;

    inline hipblasOperation_t grouped_gemm_toggle_n_t(hipblasOperation_t trans)
    {
        if(trans == HIPBLAS_OP_N)
            return HIPBLAS_OP_T;
        if(trans == HIPBLAS_OP_T)
            return HIPBLAS_OP_N;
        return trans;
    }

    template <typename Ti>
    inline Ti grouped_gemm_lda(Ti m, Ti k, hipblasOperation_t transA, Ti lda_override)
    {
        if(lda_override > 0)
            return lda_override;
        const Ti a_row = transA == HIPBLAS_OP_N ? m : k;
        return std::max(a_row, Ti(1));
    }

    template <typename Ti>
    inline Ti grouped_gemm_ldb(Ti n, Ti k, hipblasOperation_t transB, Ti ldb_override)
    {
        if(ldb_override > 0)
            return ldb_override;
        const Ti b_row = transB == HIPBLAS_OP_N ? k : n;
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
        Ti group_count{};
        Ti problem_count{};

        std::vector<hipblasOperation_t> transa_array;
        std::vector<hipblasOperation_t> transb_array;
        std::vector<Ti>                 m_array;
        std::vector<Ti>                 n_array;
        std::vector<Ti>                 k_array;
        std::vector<Ti>                 lda_array;
        std::vector<Ti>                 ldb_array;
        std::vector<Ti>                 ldc_array;
        std::vector<Ti>                 group_size;
        std::vector<T>                  alpha_array;
        std::vector<T>                  beta_array;

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

    template <typename T, typename Ti>
    grouped_gemm_test_config<T, Ti> grouped_gemm_test_config_from_arg(const Arguments& arg)
    {
        grouped_gemm_test_config<T, Ti> cfg{};
        cfg.group_count
            = Ti(std::min(std::max(arg.stride_x, int64_t(1)), int64_t(k_max_grouped_gemm_groups)));

        const hipblasOperation_t base_trans_a = char2hipblas_operation(arg.transA);
        const hipblasOperation_t base_trans_b = char2hipblas_operation(arg.transB);
        const T                  base_alpha   = arg.get_alpha<T>();
        const T                  base_beta    = arg.get_beta<T>();

        const int group_count = int(cfg.group_count);
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

        for(int g = 0; g < group_count; ++g)
        {
            const Ti m_g = Ti(arg.M + g);
            const Ti n_g = Ti(arg.N + g);
            const Ti k_g = Ti(arg.K + g);

            cfg.m_array[g] = m_g;
            cfg.n_array[g] = n_g;
            cfg.k_array[g] = k_g;

            cfg.transa_array[g]
                = (g % 2 == 0) ? base_trans_a : grouped_gemm_toggle_n_t(base_trans_a);
            cfg.transb_array[g]
                = (g % 2 == 0) ? base_trans_b : grouped_gemm_toggle_n_t(base_trans_b);

            const Ti lda_g = arg.lda > 0 ? Ti(arg.lda + g) : Ti(0);
            const Ti ldb_g = arg.ldb > 0 ? Ti(arg.ldb + g) : Ti(0);
            const Ti ldc_g = arg.ldc > 0 ? Ti(arg.ldc + g) : Ti(0);

            cfg.lda_array[g] = grouped_gemm_lda(m_g, k_g, cfg.transa_array[g], lda_g);
            cfg.ldb_array[g] = grouped_gemm_ldb(n_g, k_g, cfg.transb_array[g], ldb_g);
            cfg.ldc_array[g] = grouped_gemm_ldc(m_g, ldc_g);

            cfg.group_size[g]  = Ti(std::max(arg.batch_count + g, int64_t(0)));
            cfg.alpha_array[g] = base_alpha;
            cfg.beta_array[g]  = base_beta;
        }

        cfg.problem_count = Ti(0);
        for(int g = 0; g < group_count; ++g)
            cfg.problem_count += cfg.group_size[g];

        for(int g = 0; g < group_count; ++g)
        {
            cfg.max_m   = std::max(cfg.max_m, cfg.m_array[g]);
            cfg.max_n   = std::max(cfg.max_n, cfg.n_array[g]);
            cfg.max_k   = std::max(cfg.max_k, cfg.k_array[g]);
            cfg.max_lda = std::max(cfg.max_lda, cfg.lda_array[g]);
            cfg.max_ldb = std::max(cfg.max_ldb, cfg.ldb_array[g]);
            cfg.max_ldc = std::max(cfg.max_ldc, cfg.ldc_array[g]);

            const Ti a_row = cfg.transa_array[g] == HIPBLAS_OP_N ? cfg.m_array[g] : cfg.k_array[g];
            const Ti a_col = cfg.transa_array[g] == HIPBLAS_OP_N ? cfg.k_array[g] : cfg.m_array[g];
            const Ti b_row = cfg.transb_array[g] == HIPBLAS_OP_N ? cfg.k_array[g] : cfg.n_array[g];
            const Ti b_col = cfg.transb_array[g] == HIPBLAS_OP_N ? cfg.n_array[g] : cfg.k_array[g];

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
auto get_gemm_grouped_batched_fn(const Arguments& arg)
{
    if constexpr(std::is_same_v<T, float>)
        return arg.api == FORTRAN ? hipblasSgemmGroupedBatchedFortran : hipblasSgemmGroupedBatched;
    else
        return arg.api == FORTRAN ? hipblasDgemmGroupedBatchedFortran : hipblasDgemmGroupedBatched;
}

template <typename T>
auto get_gemm_grouped_batched_fn_64(const Arguments& arg)
{
    if constexpr(std::is_same_v<T, float>)
        return arg.api == FORTRAN_64 ? hipblasSgemmGroupedBatched_64Fortran
                                     : hipblasSgemmGroupedBatched_64;
    else
        return arg.api == FORTRAN_64 ? hipblasDgemmGroupedBatched_64Fortran
                                     : hipblasDgemmGroupedBatched_64;
}

template <typename T>
void testing_gemm_grouped_batched_bad_arg(const Arguments& arg)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    grouped_gemm_test_config<T, int>     cfg = grouped_gemm_test_config_from_arg<T, int>(arg);
    grouped_gemm_test_config<T, int64_t> cfg_64{};
    if(arg.api & c_API_64)
        cfg_64 = grouped_gemm_test_config_from_arg<T, int64_t>(arg);

    auto      fn            = get_gemm_grouped_batched_fn<T>(arg);
    auto      fn_64         = get_gemm_grouped_batched_fn_64<T>(arg);
    const int group_count   = int(cfg.group_count);
    const int problem_count = int(cfg.problem_count);

    const size_t safe_size        = std::max(size_t(cfg.max_a_row) * size_t(cfg.max_a_col),
                                      size_t(cfg.max_b_row) * size_t(cfg.max_b_col));
    const size_t padded_safe_size = std::max(safe_size, size_t(cfg.max_m) * size_t(cfg.max_n));

    hipblasLocalHandle handle(arg);

    device_batch_matrix<T> dA(padded_safe_size, 1, 1, std::max(problem_count, 1));
    device_batch_matrix<T> dB(padded_safe_size, 1, 1, std::max(problem_count, 1));
    device_batch_matrix<T> dC(padded_safe_size, 1, 1, std::max(problem_count, 1));

    std::vector<int> bad_m_array(cfg.m_array);
    bad_m_array[0] = -1;
    std::vector<int> bad_group_size(cfg.group_size);
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
        EXPECT_HIPBLAS_STATUS(fn_64(nullptr,
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
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(fn_64(handle,
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
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(fn_64(handle,
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
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(fn_64(handle,
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
                              HIPBLAS_STATUS_INVALID_VALUE);
    }
    else
    {
        EXPECT_HIPBLAS_STATUS(fn(nullptr,
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
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(fn(handle,
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
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(fn(handle,
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
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(fn(handle,
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
                              HIPBLAS_STATUS_INVALID_VALUE);
    }
}

template <typename T>
void testing_gemm_grouped_batched(const Arguments& arg)
{
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);

    grouped_gemm_test_config<T, int>     cfg = grouped_gemm_test_config_from_arg<T, int>(arg);
    grouped_gemm_test_config<T, int64_t> cfg_64{};
    if(arg.api & c_API_64)
        cfg_64 = grouped_gemm_test_config_from_arg<T, int64_t>(arg);

    auto      fn            = get_gemm_grouped_batched_fn<T>(arg);
    auto      fn_64         = get_gemm_grouped_batched_fn_64<T>(arg);
    const int group_count   = int(cfg.group_count);
    const int problem_count = int(cfg.problem_count);

    if(group_count <= 0 || problem_count <= 0)
        return;

    hipblasLocalHandle handle(arg);

    const int unit_check = arg.unit_check;
    const int norm_check = arg.norm_check;
    const int timing     = arg.timing;

    double hipblas_error_host{0}, hipblas_error_device{0};

    host_batch_matrix<T> hA(cfg.max_a_row, cfg.max_a_col, cfg.max_lda, problem_count);
    host_batch_matrix<T> hB(cfg.max_b_row, cfg.max_b_col, cfg.max_ldb, problem_count);
    host_batch_matrix<T> hC_host(cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count);
    host_batch_matrix<T> hC_device(cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count);
    host_batch_matrix<T> hC_init(cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count);
    host_batch_matrix<T> hC_gold(cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count);

    device_batch_matrix<T> dA(cfg.max_a_row, cfg.max_a_col, cfg.max_lda, problem_count);
    device_batch_matrix<T> dB(cfg.max_b_row, cfg.max_b_col, cfg.max_ldb, problem_count);
    device_batch_matrix<T> dC(cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count);

    device_vector<T> d_alpha(group_count);
    device_vector<T> d_beta(group_count);

    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC_host.memcheck());
    CHECK_HIP_ERROR(hC_device.memcheck());
    CHECK_HIP_ERROR(hC_init.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(
        hB, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, false, true);
    hipblas_init_matrix(hC_host, arg, hipblas_client_beta_sets_nan, hipblas_general_matrix);
    hC_init.copy_from(hC_host);
    hC_gold.copy_from(hC_host);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC_host));

    if(unit_check || norm_check)
    {
        int idx = 0;
        for(int g = 0; g < group_count; ++g)
        {
            for(int p = 0; p < int(cfg.group_size[g]); ++p, ++idx)
            {
                ref_gemm<T>(cfg.transa_array[g],
                            cfg.transb_array[g],
                            int(cfg.m_array[g]),
                            int(cfg.n_array[g]),
                            int(cfg.k_array[g]),
                            cfg.alpha_array[g],
                            hA[idx],
                            int(cfg.lda_array[g]),
                            hB[idx],
                            int(cfg.ldb_array[g]),
                            cfg.beta_array[g],
                            hC_gold[idx],
                            int(cfg.ldc_array[g]));
            }
        }
    }

    const auto run_grouped_gemm = [&](const T* alpha_ptr, const T* beta_ptr) {
        if(arg.api & c_API_64)
        {
            CHECK_HIPBLAS_ERROR(fn_64(handle,
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
            CHECK_HIPBLAS_ERROR(fn(handle,
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

    const auto compare_to_gold = [&](host_batch_matrix<T>& hC) {
        if(unit_check)
        {
            int idx = 0;
            for(int g = 0; g < group_count; ++g)
            {
                for(int p = 0; p < int(cfg.group_size[g]); ++p, ++idx)
                {
                    unit_check_general<T>(int(cfg.m_array[g]),
                                          int(cfg.n_array[g]),
                                          int(cfg.ldc_array[g]),
                                          hC_gold[idx],
                                          hC[idx]);
                }
            }
        }

        double error = 0;
        if(norm_check)
        {
            int idx = 0;
            for(int g = 0; g < group_count; ++g)
            {
                for(int p = 0; p < int(cfg.group_size[g]); ++p, ++idx)
                {
                    error = std::max(error,
                                     std::abs(norm_check_general<T>('F',
                                                                    int(cfg.m_array[g]),
                                                                    int(cfg.n_array[g]),
                                                                    int(cfg.ldc_array[g]),
                                                                    hC_gold[idx],
                                                                    hC[idx])));
                }
            }
        }
        return error;
    };

    if(unit_check || norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(hipMemcpy(
            d_alpha, cfg.alpha_array.data(), group_count * sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            d_beta, cfg.beta_array.data(), group_count * sizeof(T), hipMemcpyHostToDevice));
        run_grouped_gemm(d_alpha, d_beta);
        CHECK_HIP_ERROR(hC_device.transfer_from(dC));

        CHECK_HIP_ERROR(dC.transfer_from(hC_init));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        run_grouped_gemm(cfg.alpha_array.data(), cfg.beta_array.data());
        CHECK_HIP_ERROR(hC_host.transfer_from(dC));

        hipblas_error_host   = compare_to_gold(hC_host);
        hipblas_error_device = compare_to_gold(hC_device);
    }

    if(timing)
    {
        hipStream_t stream;
        CHECK_HIPBLAS_ERROR(hipblasGetStream(handle, &stream));
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

        double    gpu_time_used = 0.0;
        const int runs          = arg.cold_iters + arg.iters;

        for(int iter = 0; iter < runs; ++iter)
        {
            if(iter == arg.cold_iters)
                gpu_time_used = get_time_us_sync(stream);
            run_grouped_gemm(cfg.alpha_array.data(), cfg.beta_array.data());
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        double gflop_count = 0.0;
        for(int g = 0; g < group_count; ++g)
            gflop_count += int(cfg.group_size[g])
                           * gemm_gflop_count<T>(
                               int(cfg.m_array[g]), int(cfg.n_array[g]), int(cfg.k_array[g]));

        hipblasGemmGroupedBatchedModel{}.log_args<T>(
            std::cout, arg, gpu_time_used, gflop_count, hipblas_error_host, hipblas_error_device);
    }
}
