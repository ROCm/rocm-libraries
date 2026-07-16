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

#include "frequency_monitor.hpp"
#include "testing_common.hpp"

#include <algorithm>
#include <vector>

namespace
{
    constexpr rocblas_int k_max_grouped_gemm_groups = 8;

    inline rocblas_operation grouped_gemm_ex_toggle_n_t(rocblas_operation trans)
    {
        if(trans == rocblas_operation_none)
            return rocblas_operation_transpose;
        if(trans == rocblas_operation_transpose)
            return rocblas_operation_none;
        return trans;
    }

    inline rocblas_int grouped_gemm_ex_lda(rocblas_int       m,
                                           rocblas_int       k,
                                           rocblas_operation transA,
                                           rocblas_int       lda_override)
    {
        if(lda_override > 0)
            return lda_override;
        rocblas_int a_row = transA == rocblas_operation_none ? m : k;
        return std::max(a_row, rocblas_int(1));
    }

    inline rocblas_int grouped_gemm_ex_ldb(rocblas_int       n,
                                           rocblas_int       k,
                                           rocblas_operation transB,
                                           rocblas_int       ldb_override)
    {
        if(ldb_override > 0)
            return ldb_override;
        rocblas_int b_row = transB == rocblas_operation_none ? k : n;
        return std::max(b_row, rocblas_int(1));
    }

    inline rocblas_int grouped_gemm_ex_ldc(rocblas_int m, rocblas_int ldc_override)
    {
        if(ldc_override > 0)
            return ldc_override;
        return std::max(m, rocblas_int(1));
    }

    template <typename Tc>
    struct grouped_gemm_ex_test_config
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
        std::vector<rocblas_int>       ldd_array;
        std::vector<rocblas_int>       group_size;
        std::vector<Tc>                alpha_array;
        std::vector<Tc>                beta_array;

        rocblas_int max_m{};
        rocblas_int max_n{};
        rocblas_int max_k{};
        rocblas_int max_lda{};
        rocblas_int max_ldb{};
        rocblas_int max_ldc{};
        rocblas_int max_ldd{};
        rocblas_int max_a_row{};
        rocblas_int max_a_col{};
        rocblas_int max_b_row{};
        rocblas_int max_b_col{};
    };

    template <typename Tc>
    grouped_gemm_ex_test_config<Tc> grouped_gemm_ex_test_config_from_arg(const Arguments& arg)
    {
        grouped_gemm_ex_test_config<Tc> cfg{};
        cfg.group_count
            = std::min(std::max(arg.stride_x, int64_t(1)), int64_t(k_max_grouped_gemm_groups));

        const rocblas_operation base_trans_a = char2rocblas_operation(arg.transA);
        const rocblas_operation base_trans_b = char2rocblas_operation(arg.transB);
        const Tc                base_alpha   = arg.get_alpha<Tc>();
        const Tc                base_beta    = arg.get_beta<Tc>();

        cfg.transa_array.resize(cfg.group_count);
        cfg.transb_array.resize(cfg.group_count);
        cfg.m_array.resize(cfg.group_count);
        cfg.n_array.resize(cfg.group_count);
        cfg.k_array.resize(cfg.group_count);
        cfg.lda_array.resize(cfg.group_count);
        cfg.ldb_array.resize(cfg.group_count);
        cfg.ldc_array.resize(cfg.group_count);
        cfg.ldd_array.resize(cfg.group_count);
        cfg.group_size.resize(cfg.group_count);
        cfg.alpha_array.resize(cfg.group_count);
        cfg.beta_array.resize(cfg.group_count);

        for(rocblas_int g = 0; g < cfg.group_count; ++g)
        {
            // add in g to sizes and batch_count to test different group sizes and batch counts
            cfg.m_array[g] = rocblas_int(arg.M + g);
            cfg.n_array[g] = rocblas_int(arg.N + g);
            cfg.k_array[g] = rocblas_int(arg.K + g);

            cfg.transa_array[g]
                = (g % 2 == 0) ? base_trans_a : grouped_gemm_ex_toggle_n_t(base_trans_a);
            cfg.transb_array[g]
                = (g % 2 == 0) ? base_trans_b : grouped_gemm_ex_toggle_n_t(base_trans_b);

            const rocblas_int lda_g = arg.lda > 0 ? rocblas_int(arg.lda + g) : 0;
            const rocblas_int ldb_g = arg.ldb > 0 ? rocblas_int(arg.ldb + g) : 0;
            const rocblas_int ldc_g = arg.ldc > 0 ? rocblas_int(arg.ldc + g) : 0;
            const rocblas_int ldd_g = arg.ldd > 0 ? rocblas_int(arg.ldd + g) : ldc_g;

            cfg.lda_array[g]
                = grouped_gemm_ex_lda(cfg.m_array[g], cfg.k_array[g], cfg.transa_array[g], lda_g);
            cfg.ldb_array[g]
                = grouped_gemm_ex_ldb(cfg.n_array[g], cfg.k_array[g], cfg.transb_array[g], ldb_g);
            cfg.ldc_array[g] = grouped_gemm_ex_ldc(cfg.m_array[g], ldc_g);
            cfg.ldd_array[g] = grouped_gemm_ex_ldc(cfg.m_array[g], ldd_g);

            cfg.group_size[g]  = std::max(arg.batch_count+g, int64_t(0));
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
            cfg.max_ldd = std::max(cfg.max_ldd, cfg.ldd_array[g]);

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

    inline std::vector<int64_t> grouped_gemm_ex_to_int64(const std::vector<rocblas_int>& v)
    {
        return std::vector<int64_t>(v.begin(), v.end());
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

    grouped_gemm_ex_test_config<Tc> cfg           = grouped_gemm_ex_test_config_from_arg<Tc>(arg);
    const rocblas_int               group_count   = cfg.group_count;
    const rocblas_int               problem_count = cfg.problem_count;

    const size_t safe_size        = std::max(size_t(cfg.max_a_row) * size_t(cfg.max_a_col),
                                      size_t(cfg.max_b_row) * size_t(cfg.max_b_col));
    const size_t padded_safe_size = std::max(safe_size, size_t(cfg.max_m) * size_t(cfg.max_n));

    rocblas_local_handle handle{arg};
    rocblas_gemm_algo    algo  = rocblas_gemm_algo_standard;
    uint32_t             flags = 0;

    DEVICE_MEMCHECK(device_batch_vector<Ti>, dA, (padded_safe_size, 1, std::max(problem_count, 1)));
    DEVICE_MEMCHECK(device_batch_vector<Ti>, dB, (padded_safe_size, 1, std::max(problem_count, 1)));
    DEVICE_MEMCHECK(device_batch_vector<To>, dC, (padded_safe_size, 1, std::max(problem_count, 1)));
    DEVICE_MEMCHECK(device_batch_vector<To>, dD, (padded_safe_size, 1, std::max(problem_count, 1)));

    const void* const* dA_ptr = reinterpret_cast<const void* const*>(dA.ptr_on_device());
    const void* const* dB_ptr = reinterpret_cast<const void* const*>(dB.ptr_on_device());
    const void* const* dC_ptr = reinterpret_cast<const void* const*>(dC.ptr_on_device());
    void* const*       dD_ptr
        = const_cast<void* const*>(reinterpret_cast<const void* const*>(dD.ptr_on_device()));

    const std::vector<int64_t> m_array_64    = grouped_gemm_ex_to_int64(cfg.m_array);
    const std::vector<int64_t> n_array_64    = grouped_gemm_ex_to_int64(cfg.n_array);
    const std::vector<int64_t> k_array_64    = grouped_gemm_ex_to_int64(cfg.k_array);
    const std::vector<int64_t> lda_array_64  = grouped_gemm_ex_to_int64(cfg.lda_array);
    const std::vector<int64_t> ldb_array_64  = grouped_gemm_ex_to_int64(cfg.ldb_array);
    const std::vector<int64_t> ldc_array_64  = grouped_gemm_ex_to_int64(cfg.ldc_array);
    const std::vector<int64_t> ldd_array_64  = grouped_gemm_ex_to_int64(cfg.ldd_array);
    const std::vector<int64_t> group_size_64 = grouped_gemm_ex_to_int64(cfg.group_size);
    std::vector<rocblas_int>   bad_m_array(cfg.m_array);
    bad_m_array[0]                            = -1;
    const std::vector<int64_t> bad_m_array_64 = grouped_gemm_ex_to_int64(bad_m_array);
    std::vector<rocblas_int>   bad_group_size(cfg.group_size);
    bad_group_size[0]                            = -1;
    const std::vector<int64_t> bad_group_size_64 = grouped_gemm_ex_to_int64(bad_group_size);

    if(arg.api & c_API_64)
    {
        const int64_t group_count_64 = group_count;
        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(nullptr,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    m_array_64.data(),
                                                                    n_array_64.data(),
                                                                    k_array_64.data(),
                                                                    cfg.alpha_array.data(),
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    lda_array_64.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    ldb_array_64.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    ldc_array_64.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    ldd_array_64.data(),
                                                                    group_count_64,
                                                                    group_size_64.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_handle);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    m_array_64.data(),
                                                                    n_array_64.data(),
                                                                    k_array_64.data(),
                                                                    nullptr,
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    lda_array_64.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    ldb_array_64.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    ldc_array_64.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    ldd_array_64.data(),
                                                                    group_count_64,
                                                                    group_size_64.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_pointer);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    bad_m_array_64.data(),
                                                                    n_array_64.data(),
                                                                    k_array_64.data(),
                                                                    cfg.alpha_array.data(),
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    lda_array_64.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    ldb_array_64.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    ldc_array_64.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    ldd_array_64.data(),
                                                                    group_count_64,
                                                                    group_size_64.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_size);

        EXPECT_ROCBLAS_STATUS(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                    cfg.transa_array.data(),
                                                                    cfg.transb_array.data(),
                                                                    m_array_64.data(),
                                                                    n_array_64.data(),
                                                                    k_array_64.data(),
                                                                    cfg.alpha_array.data(),
                                                                    dA_ptr,
                                                                    arg.a_type,
                                                                    lda_array_64.data(),
                                                                    dB_ptr,
                                                                    arg.b_type,
                                                                    ldb_array_64.data(),
                                                                    cfg.beta_array.data(),
                                                                    dC_ptr,
                                                                    arg.c_type,
                                                                    ldc_array_64.data(),
                                                                    dD_ptr,
                                                                    arg.d_type,
                                                                    ldd_array_64.data(),
                                                                    group_count_64,
                                                                    bad_group_size_64.data(),
                                                                    arg.compute_type,
                                                                    algo,
                                                                    flags),
                              rocblas_status_invalid_size);
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
                                                                 group_count,
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
                                                                 group_count,
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
                                                                 group_count,
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
                                                                 group_count,
                                                                 bad_group_size.data(),
                                                                 arg.compute_type,
                                                                 algo,
                                                                 flags),
                              rocblas_status_invalid_size);
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

    grouped_gemm_ex_test_config<Tc> cfg           = grouped_gemm_ex_test_config_from_arg<Tc>(arg);
    const rocblas_int               group_count   = cfg.group_count;
    const rocblas_int               problem_count = cfg.problem_count;

    if(group_count <= 0 || problem_count <= 0)
        return;

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

        rocblas_int idx = 0;
        for(rocblas_int g = 0; g < group_count; ++g)
        {
            for(rocblas_int p = 0; p < cfg.group_size[g]; ++p, ++idx)
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
            const std::vector<int64_t> m_array_64     = grouped_gemm_ex_to_int64(cfg.m_array);
            const std::vector<int64_t> n_array_64     = grouped_gemm_ex_to_int64(cfg.n_array);
            const std::vector<int64_t> k_array_64     = grouped_gemm_ex_to_int64(cfg.k_array);
            const std::vector<int64_t> lda_array_64   = grouped_gemm_ex_to_int64(cfg.lda_array);
            const std::vector<int64_t> ldb_array_64   = grouped_gemm_ex_to_int64(cfg.ldb_array);
            const std::vector<int64_t> ldc_array_64   = grouped_gemm_ex_to_int64(cfg.ldc_array);
            const std::vector<int64_t> ldd_array_64   = grouped_gemm_ex_to_int64(cfg.ldd_array);
            const std::vector<int64_t> group_size_64  = grouped_gemm_ex_to_int64(cfg.group_size);
            const int64_t              group_count_64 = group_count;
            CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                      cfg.transa_array.data(),
                                                                      cfg.transb_array.data(),
                                                                      m_array_64.data(),
                                                                      n_array_64.data(),
                                                                      k_array_64.data(),
                                                                      alpha_ptr,
                                                                      dA_ptr,
                                                                      arg.a_type,
                                                                      lda_array_64.data(),
                                                                      dB_ptr,
                                                                      arg.b_type,
                                                                      ldb_array_64.data(),
                                                                      beta_ptr,
                                                                      dC_ptr,
                                                                      arg.c_type,
                                                                      ldc_array_64.data(),
                                                                      dDref_ptr,
                                                                      d_type,
                                                                      ldd_array_64.data(),
                                                                      group_count_64,
                                                                      group_size_64.data(),
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

    HOST_MEMCHECK(host_batch_matrix<To>, hD, (cfg.max_m, cfg.max_n, cfg.max_ldd, problem_count));

    if(arg.unit_check || arg.norm_check)
    {
        if(arg.pointer_mode_host)
        {
            CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
            run_grouped_gemm_ex(cfg.alpha_array.data(), cfg.beta_array.data());
            CHECK_HIP_ERROR(hD.transfer_from(dDref));
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
            CHECK_HIP_ERROR(hD.transfer_from(dDref));
        }

        if(arg.unit_check)
        {
            rocblas_int idx = 0;
            for(rocblas_int g = 0; g < group_count; ++g)
            {
                for(rocblas_int p = 0; p < cfg.group_size[g]; ++p, ++idx)
                {
                    unit_check_general<To, To_hpa>(
                        cfg.m_array[g], cfg.n_array[g], cfg.ldd_array[g], hD_gold[idx], hD[idx]);
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
                                     std::abs(norm_check_general<To>('F',
                                                                     cfg.m_array[g],
                                                                     cfg.n_array[g],
                                                                     cfg.ldd_array[g],
                                                                     (To_hpa*)hD_gold[idx],
                                                                     hD[idx])));
                }
            }
            ASSERT_NEAR(error, 0.0, 1e-10);
        }
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

        const std::vector<int64_t> m_array_64     = grouped_gemm_ex_to_int64(cfg.m_array);
        const std::vector<int64_t> n_array_64     = grouped_gemm_ex_to_int64(cfg.n_array);
        const std::vector<int64_t> k_array_64     = grouped_gemm_ex_to_int64(cfg.k_array);
        const std::vector<int64_t> lda_array_64   = grouped_gemm_ex_to_int64(cfg.lda_array);
        const std::vector<int64_t> ldb_array_64   = grouped_gemm_ex_to_int64(cfg.ldb_array);
        const std::vector<int64_t> ldc_array_64   = grouped_gemm_ex_to_int64(cfg.ldc_array);
        const std::vector<int64_t> ldd_array_64   = grouped_gemm_ex_to_int64(cfg.ldd_array);
        const std::vector<int64_t> group_size_64  = grouped_gemm_ex_to_int64(cfg.group_size);
        const int64_t              group_count_64 = group_count;

        for(int i = 0; i < total_calls; i++)
        {
            if(i == number_cold_calls)
                gpu_time_used = get_time_us_sync(stream); // in microseconds

            if(arg.api & c_API_64)
            {
                CHECK_ROCBLAS_ERROR(rocblas_gemm_grouped_batched_ex_fn_64(handle,
                                                                          cfg.transa_array.data(),
                                                                          cfg.transb_array.data(),
                                                                          m_array_64.data(),
                                                                          n_array_64.data(),
                                                                          k_array_64.data(),
                                                                          cfg.alpha_array.data(),
                                                                          dA_ptr,
                                                                          arg.a_type,
                                                                          lda_array_64.data(),
                                                                          dB_ptr,
                                                                          arg.b_type,
                                                                          ldb_array_64.data(),
                                                                          cfg.beta_array.data(),
                                                                          dC_ptr,
                                                                          arg.c_type,
                                                                          ldc_array_64.data(),
                                                                          dDref_ptr,
                                                                          d_type,
                                                                          ldd_array_64.data(),
                                                                          group_count_64,
                                                                          group_size_64.data(),
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
        for(rocblas_int g = 0; g < group_count; ++g)
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
