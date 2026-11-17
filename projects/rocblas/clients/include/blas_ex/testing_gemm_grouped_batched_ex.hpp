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

#include "frequency_monitor.hpp"
#include "testing_common.hpp"

#include <algorithm>
#include <vector>

namespace
{
    inline rocblas_operation grouped_gemm_ex_toggle_n_t(rocblas_operation trans)
    {
        if(trans == rocblas_operation_none)
            return rocblas_operation_transpose;
        if(trans == rocblas_operation_transpose)
            return rocblas_operation_none;
        return trans;
    }

    template <typename Ti>
    inline Ti grouped_gemm_ex_lda(Ti m, Ti k, rocblas_operation transA, Ti lda_override)
    {
        if(lda_override > 0)
            return lda_override;
        const Ti a_row = transA == rocblas_operation_none ? m : k;
        return std::max(a_row, Ti(1));
    }

    template <typename Ti>
    inline Ti grouped_gemm_ex_ldb(Ti n, Ti k, rocblas_operation transB, Ti ldb_override)
    {
        if(ldb_override > 0)
            return ldb_override;
        const Ti b_row = transB == rocblas_operation_none ? k : n;
        return std::max(b_row, Ti(1));
    }

    template <typename Ti>
    inline Ti grouped_gemm_ex_ldc(Ti m, Ti ldc_override)
    {
        if(ldc_override > 0)
            return ldc_override;
        return std::max(m, Ti(1));
    }

    template <typename Tc, typename Ti>
    struct grouped_gemm_ex_test_config
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
        std::vector<Ti>                ldd_array;
        std::vector<Ti>                group_size;
        std::vector<Tc>                alpha_array;
        std::vector<Tc>                beta_array;

        Ti max_m{};
        Ti max_n{};
        Ti max_k{};
        Ti max_lda{};
        Ti max_ldb{};
        Ti max_ldc{};
        Ti max_ldd{};
        Ti max_a_row{};
        Ti max_a_col{};
        Ti max_b_row{};
        Ti max_b_col{};
    };

    template <typename Tc, typename Ti>
    grouped_gemm_ex_test_config<Tc, Ti> grouped_gemm_ex_test_config_from_arg(const Arguments& arg)
    {
        grouped_gemm_ex_test_config<Tc, Ti> cfg{};
        cfg.group_count = arg.stride_x;

        const rocblas_operation base_trans_a = char2rocblas_operation(arg.transA);
        const rocblas_operation base_trans_b = char2rocblas_operation(arg.transB);
        const Tc                base_alpha   = arg.get_alpha<Tc>();
        const Tc                base_beta    = arg.get_beta<Tc>();

        const Ti group_count = cfg.group_count;
        cfg.transa_array.resize(group_count);
        cfg.transb_array.resize(group_count);
        cfg.m_array.resize(group_count);
        cfg.n_array.resize(group_count);
        cfg.k_array.resize(group_count);
        cfg.lda_array.resize(group_count);
        cfg.ldb_array.resize(group_count);
        cfg.ldc_array.resize(group_count);
        cfg.ldd_array.resize(group_count);
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
                = (g % 2 == 0) ? base_trans_a : grouped_gemm_ex_toggle_n_t(base_trans_a);
            cfg.transb_array[g]
                = (g % 2 == 0) ? base_trans_b : grouped_gemm_ex_toggle_n_t(base_trans_b);

            const Ti lda_g = arg.lda > 0 ? Ti(arg.lda + variation) : Ti(0);
            const Ti ldb_g = arg.ldb > 0 ? Ti(arg.ldb + variation) : Ti(0);
            const Ti ldc_g = arg.ldc > 0 ? Ti(arg.ldc + variation) : Ti(0);
            const Ti ldd_g = arg.ldd > 0 ? Ti(arg.ldd + variation) : ldc_g;

            cfg.lda_array[g] = grouped_gemm_ex_lda(m_g, k_g, cfg.transa_array[g], lda_g);
            cfg.ldb_array[g] = grouped_gemm_ex_ldb(n_g, k_g, cfg.transb_array[g], ldb_g);
            cfg.ldc_array[g] = grouped_gemm_ex_ldc(m_g, ldc_g);
            cfg.ldd_array[g] = grouped_gemm_ex_ldc(m_g, ldd_g);

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
            cfg.max_ldd = std::max(cfg.max_ldd, cfg.ldd_array[g]);

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

template <typename Ti, typename To, typename Tc>
void testing_gemm_grouped_batched_ex_bad_arg(const Arguments& arg)
{
    auto rocblas_gemm_grouped_batched_ex_fn    = arg.api & c_API_FORTRAN
                                                     ? rocblas_gemm_grouped_batched_ex_fortran
                                                     : rocblas_gemm_grouped_batched_ex;
    auto rocblas_gemm_grouped_batched_ex_fn_64 = arg.api & c_API_FORTRAN
                                                     ? rocblas_gemm_grouped_batched_ex_64_fortran
                                                     : rocblas_gemm_grouped_batched_ex_64;

    grouped_gemm_ex_test_config<Tc, rocblas_int> cfg
        = grouped_gemm_ex_test_config_from_arg<Tc, rocblas_int>(arg);
    grouped_gemm_ex_test_config<Tc, int64_t> cfg_64{};
    if(arg.api & c_API_64)
        cfg_64 = grouped_gemm_ex_test_config_from_arg<Tc, int64_t>(arg);

    // we aren't testing these with > 32bit groups for now
    const int64_t problem_count = (cfg.problem_count);

    const size_t safe_size        = std::max(size_t(cfg.max_a_row) * size_t(cfg.max_a_col),
                                      size_t(cfg.max_b_row) * size_t(cfg.max_b_col));
    const size_t padded_safe_size = std::max(safe_size, size_t(cfg.max_m) * size_t(cfg.max_n));

    rocblas_local_handle handle{arg};
    rocblas_gemm_algo    algo  = rocblas_gemm_algo_standard;
    uint32_t             flags = 0;

    DEVICE_MEMCHECK(device_batch_vector<Ti>, dA, (padded_safe_size, 1, problem_count));
    DEVICE_MEMCHECK(device_batch_vector<Ti>, dB, (padded_safe_size, 1, problem_count));
    DEVICE_MEMCHECK(device_batch_vector<To>, dC, (padded_safe_size, 1, problem_count));
    DEVICE_MEMCHECK(device_batch_vector<To>, dD, (padded_safe_size, 1, problem_count));

    const void* const* dA_ptr = reinterpret_cast<const void* const*>(dA.ptr_on_device());
    const void* const* dB_ptr = reinterpret_cast<const void* const*>(dB.ptr_on_device());
    const void* const* dC_ptr = reinterpret_cast<const void* const*>(dC.ptr_on_device());
    void* const*       dD_ptr
        = const_cast<void* const*>(reinterpret_cast<const void* const*>(dD.ptr_on_device()));

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
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(nullptr,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    cfg_64.m_array.data(),
                                                                    cfg_64.n_array.data(),
                                                                    cfg_64.k_array.data(),
                                                                    cfg.alpha_array.data(),
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    cfg_64.lda_array.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    cfg_64.ldb_array.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    cfg_64.ldc_array.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    cfg_64.ldd_array.data(),
                                                                    cfg_64.group_count,
                                                                    cfg_64.group_size.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    cfg_64.m_array.data(),
                                                                    cfg_64.n_array.data(),
                                                                    cfg_64.k_array.data(),
                                                                    nullptr,
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    cfg_64.lda_array.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    cfg_64.ldb_array.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    cfg_64.ldc_array.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    cfg_64.ldd_array.data(),
                                                                    cfg_64.group_count,
                                                                    cfg_64.group_size.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    bad_m_array_64.data(),
                                                                    cfg_64.n_array.data(),
                                                                    cfg_64.k_array.data(),
                                                                    cfg.alpha_array.data(),
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    cfg_64.lda_array.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    cfg_64.ldb_array.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    cfg_64.ldc_array.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    cfg_64.ldd_array.data(),
                                                                    cfg_64.group_count,
                                                                    cfg_64.group_size.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    cfg_64.m_array.data(),
                                                                    cfg_64.n_array.data(),
                                                                    cfg_64.k_array.data(),
                                                                    cfg.alpha_array.data(),
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    cfg_64.lda_array.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    cfg_64.ldb_array.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    cfg_64.ldc_array.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    cfg_64.ldd_array.data(),
                                                                    cfg_64.group_count,
                                                                    bad_group_size_64.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    cfg_64.m_array.data(),
                                                                    cfg_64.n_array.data(),
                                                                    cfg_64.k_array.data(),
                                                                    cfg.alpha_array.data(),
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    cfg_64.lda_array.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    cfg_64.ldb_array.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    cfg_64.ldc_array.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    cfg_64.ldd_array.data(),
                                                                    int64_t(-1),
                                                                    cfg_64.group_size.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_size);

        // If group_count==0, then all pointers can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    arg.a_type,
                                                                    nullptr,
                                                                    nullptr,
                                                                    arg.b_type,
                                                                    nullptr,
                                                                    nullptr,
                                                                    nullptr,
                                                                    arg.c_type,
                                                                    nullptr,
                                                                    nullptr,
                                                                    arg.d_type,
                                                                    nullptr,
                                                                    int64_t(0),
                                                                    nullptr,
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_success);
    }
    else
    {
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn(nullptr,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 cfg.m_array.data(),
                                                                 cfg.n_array.data(),
                                                                 cfg.k_array.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA_ptr,
                                                                 arg.a_type,
                                                                 cfg.lda_array.data(),
                                                                 dB_ptr,
                                                                 arg.b_type,
                                                                 cfg.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC_ptr,
                                                                 arg.c_type,
                                                                 cfg.ldc_array.data(),
                                                                 dD_ptr,
                                                                 arg.d_type,
                                                                 cfg.ldd_array.data(),
                                                                 cfg.group_count,
                                                                 cfg.group_size.data(),
                                                                 arg.compute_type,
                                                                 algo,
                                                                 flags),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 cfg.m_array.data(),
                                                                 cfg.n_array.data(),
                                                                 cfg.k_array.data(),
                                                                 nullptr,
                                                                 dA_ptr,
                                                                 arg.a_type,
                                                                 cfg.lda_array.data(),
                                                                 dB_ptr,
                                                                 arg.b_type,
                                                                 cfg.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC_ptr,
                                                                 arg.c_type,
                                                                 cfg.ldc_array.data(),
                                                                 dD_ptr,
                                                                 arg.d_type,
                                                                 cfg.ldd_array.data(),
                                                                 cfg.group_count,
                                                                 cfg.group_size.data(),
                                                                 arg.compute_type,
                                                                 algo,
                                                                 flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 bad_m_array.data(),
                                                                 cfg.n_array.data(),
                                                                 cfg.k_array.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA_ptr,
                                                                 arg.a_type,
                                                                 cfg.lda_array.data(),
                                                                 dB_ptr,
                                                                 arg.b_type,
                                                                 cfg.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC_ptr,
                                                                 arg.c_type,
                                                                 cfg.ldc_array.data(),
                                                                 dD_ptr,
                                                                 arg.d_type,
                                                                 cfg.ldd_array.data(),
                                                                 cfg.group_count,
                                                                 cfg.group_size.data(),
                                                                 arg.compute_type,
                                                                 algo,
                                                                 flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 cfg.m_array.data(),
                                                                 cfg.n_array.data(),
                                                                 cfg.k_array.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA_ptr,
                                                                 arg.a_type,
                                                                 cfg.lda_array.data(),
                                                                 dB_ptr,
                                                                 arg.b_type,
                                                                 cfg.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC_ptr,
                                                                 arg.c_type,
                                                                 cfg.ldc_array.data(),
                                                                 dD_ptr,
                                                                 arg.d_type,
                                                                 cfg.ldd_array.data(),
                                                                 cfg.group_count,
                                                                 bad_group_size.data(),
                                                                 arg.compute_type,
                                                                 algo,
                                                                 flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn(handle,
                                                                 cfg.transa_array.data(),
                                                                 cfg.transb_array.data(),
                                                                 cfg.m_array.data(),
                                                                 cfg.n_array.data(),
                                                                 cfg.k_array.data(),
                                                                 cfg.alpha_array.data(),
                                                                 dA_ptr,
                                                                 arg.a_type,
                                                                 cfg.lda_array.data(),
                                                                 dB_ptr,
                                                                 arg.b_type,
                                                                 cfg.ldb_array.data(),
                                                                 cfg.beta_array.data(),
                                                                 dC_ptr,
                                                                 arg.c_type,
                                                                 cfg.ldc_array.data(),
                                                                 dD_ptr,
                                                                 arg.d_type,
                                                                 cfg.ldd_array.data(),
                                                                 -1,
                                                                 cfg.group_size.data(),
                                                                 arg.compute_type,
                                                                 algo,
                                                                 flags),
                              rocblas_status_invalid_size);

        // If group_count==0, then all pointers can be nullptr without issue.
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn(handle,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 arg.a_type,
                                                                 nullptr,
                                                                 nullptr,
                                                                 arg.b_type,
                                                                 nullptr,
                                                                 nullptr,
                                                                 nullptr,
                                                                 arg.c_type,
                                                                 nullptr,
                                                                 nullptr,
                                                                 arg.d_type,
                                                                 nullptr,
                                                                 0,
                                                                 nullptr,
                                                                 arg.compute_type,
                                                                 algo,
                                                                 flags),
                              rocblas_status_success);
    }
}

template <typename Ti, typename To, typename Tc>
void testing_gemm_grouped_batched_ex(const Arguments& arg)
{
    auto rocblas_gemm_grouped_batched_ex_fn    = arg.api & c_API_FORTRAN
                                                     ? rocblas_gemm_grouped_batched_ex_fortran
                                                     : rocblas_gemm_grouped_batched_ex;
    auto rocblas_gemm_grouped_batched_ex_fn_64 = arg.api & c_API_FORTRAN
                                                     ? rocblas_gemm_grouped_batched_ex_64_fortran
                                                     : rocblas_gemm_grouped_batched_ex_64;

    grouped_gemm_ex_test_config<Tc, rocblas_int> cfg
        = grouped_gemm_ex_test_config_from_arg<Tc, rocblas_int>(arg);
    grouped_gemm_ex_test_config<Tc, int64_t> cfg_64{};
    if(arg.api & c_API_64)
        cfg_64 = grouped_gemm_ex_test_config_from_arg<Tc, int64_t>(arg);

    const int64_t group_count   = cfg.group_count;
    const int64_t problem_count = cfg.problem_count;

    double rocblas_error = 0.0, error_hst_ptr = 0.0, error_dev_ptr = 0.0;

    rocblas_local_handle handle{arg};
    rocblas_gemm_algo    algo = rocblas_gemm_algo(arg.algo);
    uint32_t             flags(arg.flags);
    rocblas_datatype     d_type = arg.d_type;

    if(!arg.outofplace)
    {
        d_type = arg.c_type;
    }

    using To_hpa = std::conditional_t<std::is_same_v<To, rocblas_bfloat16>, float, To>;

    HOST_MEMCHECK(
        host_batch_matrix<Ti>, hA, (cfg.max_a_row, cfg.max_a_col, cfg.max_lda, problem_count));
    HOST_MEMCHECK(
        host_batch_matrix<Ti>, hB, (cfg.max_b_row, cfg.max_b_col, cfg.max_ldb, problem_count));
    HOST_MEMCHECK(host_batch_matrix<To>, hC, (cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count));
    HOST_MEMCHECK(
        host_batch_matrix<To>, hC_init, (cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count));
    HOST_MEMCHECK(
        host_batch_matrix<To_hpa>, hD_gold, (cfg.max_m, cfg.max_n, cfg.max_ldd, problem_count));

    DEVICE_MEMCHECK(
        device_batch_matrix<Ti>, dA, (cfg.max_a_row, cfg.max_a_col, cfg.max_lda, problem_count));
    DEVICE_MEMCHECK(
        device_batch_matrix<Ti>, dB, (cfg.max_b_row, cfg.max_b_col, cfg.max_ldb, problem_count));
    DEVICE_MEMCHECK(
        device_batch_matrix<To>, dC, (cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count));
    device_batch_matrix<To> dD
        = arg.outofplace ? device_batch_matrix<To>(cfg.max_m, cfg.max_n, cfg.max_ldd, problem_count)
                         : device_batch_matrix<To>(0, 1, 1, 1);
    CHECK_DEVICE_ALLOCATION(dD.memcheck());
    device_batch_matrix<To>& dDref = arg.outofplace ? dD : dC;

    const void* const* dA_ptr = reinterpret_cast<const void* const*>(dA.ptr_on_device());
    const void* const* dB_ptr = reinterpret_cast<const void* const*>(dB.ptr_on_device());
    const void* const* dC_ptr = reinterpret_cast<const void* const*>(dC.ptr_on_device());
    void* const*       dD_ptr
        = arg.outofplace
              ? const_cast<void* const*>(reinterpret_cast<const void* const*>(dD.ptr_on_device()))
              : const_cast<void* const*>(dC_ptr);
    void* const* dDref_ptr = dD_ptr;

    rocblas_init_matrix(
        hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
    rocblas_init_matrix(
        hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
    rocblas_init_matrix(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);
    hC_init.copy_from(hC);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(arg.unit_check || arg.norm_check)
    {
        copy_matrix_with_different_leading_dimensions(hC, hD_gold);

        int64_t idx = 0;
        for(int64_t g = 0; g < group_count; ++g)
        {
            for(int64_t p = 0; p < cfg.group_size[g]; ++p, ++idx)
            {
                ref_gemm<Ti, To_hpa, Tc>(cfg.transa_array[g],
                                         cfg.transb_array[g],
                                         int64_t(cfg.m_array[g]),
                                         int64_t(cfg.n_array[g]),
                                         int64_t(cfg.k_array[g]),
                                         cfg.alpha_array[g],
                                         hA[idx],
                                         int64_t(cfg.lda_array[g]),
                                         hB[idx],
                                         int64_t(cfg.ldb_array[g]),
                                         cfg.beta_array[g],
                                         hD_gold[idx],
                                         int64_t(cfg.ldd_array[g]));
            }
        }
    }

    const auto run_grouped_gemm_ex = [&](const void* alpha_ptr, const void* beta_ptr) {
        if(arg.api & c_API_64)
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                      cfg.transa_array.data(),
                                                                      cfg.transb_array.data(),
                                                                      cfg_64.m_array.data(),
                                                                      cfg_64.n_array.data(),
                                                                      cfg_64.k_array.data(),
                                                                      alpha_ptr,
                                                                      dA_ptr,
                                                                      arg.a_type,
                                                                      cfg_64.lda_array.data(),
                                                                      dB_ptr,
                                                                      arg.b_type,
                                                                      cfg_64.ldb_array.data(),
                                                                      beta_ptr,
                                                                      dC_ptr,
                                                                      arg.c_type,
                                                                      cfg_64.ldc_array.data(),
                                                                      dDref_ptr,
                                                                      d_type,
                                                                      cfg_64.ldd_array.data(),
                                                                      cfg_64.group_count,
                                                                      cfg_64.group_size.data(),
                                                                      arg.compute_type,
                                                                      algo,
                                                                      flags));
        }
        else
        {
            CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_ex_fn(handle,
                                                                   cfg.transa_array.data(),
                                                                   cfg.transb_array.data(),
                                                                   cfg.m_array.data(),
                                                                   cfg.n_array.data(),
                                                                   cfg.k_array.data(),
                                                                   alpha_ptr,
                                                                   dA_ptr,
                                                                   arg.a_type,
                                                                   cfg.lda_array.data(),
                                                                   dB_ptr,
                                                                   arg.b_type,
                                                                   cfg.ldb_array.data(),
                                                                   beta_ptr,
                                                                   dC_ptr,
                                                                   arg.c_type,
                                                                   cfg.ldc_array.data(),
                                                                   dDref_ptr,
                                                                   d_type,
                                                                   cfg.ldd_array.data(),
                                                                   group_count,
                                                                   cfg.group_size.data(),
                                                                   arg.compute_type,
                                                                   algo,
                                                                   flags));
        }
    };

    HOST_MEMCHECK(host_batch_matrix<To>, hD_1, (cfg.max_m, cfg.max_n, cfg.max_ldd, problem_count));
    HOST_MEMCHECK(host_batch_matrix<To>, hD_2, (cfg.max_m, cfg.max_n, cfg.max_ldd, problem_count));

    {
        int64_t idx = 0;
        for(int64_t g = 0; g < group_count; ++g)
        {
            for(int64_t p = 0; p < cfg.group_size[g]; ++p, ++idx)
            {
                rocblas_init_nan<To>(hD_1[idx], cfg.m_array[g], cfg.n_array[g], cfg.ldd_array[g]);
            }
        }
    }
    hD_2.copy_from(hD_1);

    const auto compare_to_gold = [&](host_batch_matrix<To>& hD) {
        if(arg.unit_check)
        {
            int64_t idx = 0;
            for(int64_t g = 0; g < group_count; ++g)
            {
                for(int64_t p = 0; p < cfg.group_size[g]; ++p, ++idx)
                {
                    if((rocblas_handle(handle)->getArchMajor() == 11) && (sizeof(Ti) == 2))
                    {
                        const double tol
                            = cfg.k_array[g] * sum_error_tolerance_for_gfx11<Tc, Ti, To>;
                        near_check_general<To, To_hpa>(cfg.m_array[g],
                                                       cfg.n_array[g],
                                                       cfg.ldd_array[g],
                                                       hD_gold[idx],
                                                       hD[idx],
                                                       tol);
                    }
                    else if(std::is_same_v<Tc, rocblas_half> && cfg.k_array[g] > 10000)
                    {
                        const double tol = cfg.k_array[g] * sum_error_tolerance<Tc>;
                        near_check_general<To, To_hpa>(cfg.m_array[g],
                                                       cfg.n_array[g],
                                                       cfg.ldd_array[g],
                                                       hD_gold[idx],
                                                       hD[idx],
                                                       tol);
                    }
                    else
                    {
                        unit_check_general<To, To_hpa>(cfg.m_array[g],
                                                       cfg.n_array[g],
                                                       cfg.ldd_array[g],
                                                       hD_gold[idx],
                                                       hD[idx]);
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
                                     std::abs(norm_check_general<To>('F',
                                                                     cfg.m_array[g],
                                                                     cfg.n_array[g],
                                                                     cfg.ldd_array[g],
                                                                     (To_hpa*)hD_gold[idx],
                                                                     (To*)hD[idx])));
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
            handle.pre_test(arg);
            run_grouped_gemm_ex(cfg.alpha_array.data(), cfg.beta_array.data());
            handle.post_test(arg);
            CHECK_HIP_ERROR(hD_1.transfer_from(dDref));
        }

        if(arg.pointer_mode_device)
        {
            DEVICE_MEMCHECK(device_vector<Tc>, d_alpha, (group_count));
            DEVICE_MEMCHECK(device_vector<Tc>, d_beta, (group_count));
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
            CHECK_HIP_ERROR(dC.transfer_from(hC_init));
            CHECK_HIP_ERROR(hipMemcpy(
                d_alpha, cfg.alpha_array.data(), group_count * sizeof(Tc), hipMemcpyHostToDevice));
            CHECK_HIP_ERROR(hipMemcpy(
                d_beta, cfg.beta_array.data(), group_count * sizeof(Tc), hipMemcpyHostToDevice));
            run_grouped_gemm_ex(d_alpha, d_beta);
            CHECK_HIP_ERROR(hD_2.transfer_from(dDref));
        }

        if(arg.pointer_mode_host)
        {
            error_hst_ptr = compare_to_gold(hD_1);
        }
        if(arg.pointer_mode_device)
        {
            error_dev_ptr = compare_to_gold(hD_2);
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
                CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                          cfg.transa_array.data(),
                                                                          cfg.transb_array.data(),
                                                                          cfg_64.m_array.data(),
                                                                          cfg_64.n_array.data(),
                                                                          cfg_64.k_array.data(),
                                                                          cfg.alpha_array.data(),
                                                                          dA_ptr,
                                                                          arg.a_type,
                                                                          cfg_64.lda_array.data(),
                                                                          dB_ptr,
                                                                          arg.b_type,
                                                                          cfg_64.ldb_array.data(),
                                                                          cfg.beta_array.data(),
                                                                          dC_ptr,
                                                                          arg.c_type,
                                                                          cfg_64.ldc_array.data(),
                                                                          dDref_ptr,
                                                                          d_type,
                                                                          cfg_64.ldd_array.data(),
                                                                          cfg_64.group_count,
                                                                          cfg_64.group_size.data(),
                                                                          arg.compute_type,
                                                                          algo,
                                                                          flags));
            }
            else
            {
                CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_ex_fn(handle,
                                                                       cfg.transa_array.data(),
                                                                       cfg.transb_array.data(),
                                                                       cfg.m_array.data(),
                                                                       cfg.n_array.data(),
                                                                       cfg.k_array.data(),
                                                                       cfg.alpha_array.data(),
                                                                       dA_ptr,
                                                                       arg.a_type,
                                                                       cfg.lda_array.data(),
                                                                       dB_ptr,
                                                                       arg.b_type,
                                                                       cfg.ldb_array.data(),
                                                                       cfg.beta_array.data(),
                                                                       dC_ptr,
                                                                       arg.c_type,
                                                                       cfg.ldc_array.data(),
                                                                       dDref_ptr,
                                                                       d_type,
                                                                       cfg.ldd_array.data(),
                                                                       group_count,
                                                                       cfg.group_size.data(),
                                                                       arg.compute_type,
                                                                       algo,
                                                                       flags));
            }
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        freq_monitor.stop();

        double gflop_count = 0.0;
        for(int64_t g = 0; g < group_count; ++g)
            gflop_count += cfg.group_size[g]
                           * gemm_gflop_count<Tc>(cfg.m_array[g], cfg.n_array[g], cfg.k_array[g]);

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
                      e_ldd,
                      e_stride_x,
                      e_batch_count>{}
            .log_args<To>(rocblas_cout,
                          arg,
                          gpu_time_used,
                          gflop_count,
                          ArgumentLogging::NA_value,
                          ArgumentLogging::NA_value,
                          ArgumentLogging::NA_value);
    }
}
