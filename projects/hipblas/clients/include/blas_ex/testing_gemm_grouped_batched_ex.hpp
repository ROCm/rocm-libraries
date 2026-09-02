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

using hipblasGemmGroupedBatchedExModel = ArgumentModel<e_a_type,
                                                       e_b_type,
                                                       e_c_type,
                                                       e_d_type,
                                                       e_compute_type,
                                                       e_transA,
                                                       e_transB,
                                                       e_M,
                                                       e_N,
                                                       e_K,
                                                       e_alpha,
                                                       e_lda,
                                                       e_ldb,
                                                       e_beta,
                                                       e_ldc,
                                                       e_ldd,
                                                       e_stride_x,
                                                       e_batch_count,
                                                       e_with_flags,
                                                       e_flags>;

inline void testname_gemm_grouped_batched_ex(const Arguments& arg, std::string& name)
{
    hipblasGemmGroupedBatchedExModel{}.test_name(arg, name);
}

namespace
{
    constexpr int k_max_grouped_gemm_ex_groups = 8;

    inline hipblasOperation_t grouped_gemm_ex_toggle_n_t(hipblasOperation_t trans)
    {
        if(trans == HIPBLAS_OP_N)
            return HIPBLAS_OP_T;
        if(trans == HIPBLAS_OP_T)
            return HIPBLAS_OP_N;
        return trans;
    }

    template <typename Ti>
    inline Ti grouped_gemm_ex_lda(Ti m, Ti k, hipblasOperation_t transA, Ti lda_override)
    {
        if(lda_override > 0)
            return lda_override;
        const Ti a_row = transA == HIPBLAS_OP_N ? m : k;
        return std::max(a_row, Ti(1));
    }

    template <typename Ti>
    inline Ti grouped_gemm_ex_ldb(Ti n, Ti k, hipblasOperation_t transB, Ti ldb_override)
    {
        if(ldb_override > 0)
            return ldb_override;
        const Ti b_row = transB == HIPBLAS_OP_N ? k : n;
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
        std::vector<Ti>                 ldd_array;
        std::vector<Ti>                 group_size;
        std::vector<Tc>                 alpha_array;
        std::vector<Tc>                 beta_array;

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
        cfg.group_count = Ti(
            std::min(std::max(arg.stride_x, int64_t(1)), int64_t(k_max_grouped_gemm_ex_groups)));

        const hipblasOperation_t base_trans_a = char2hipblas_operation(arg.transA);
        const hipblasOperation_t base_trans_b = char2hipblas_operation(arg.transB);
        const Tc                 base_alpha   = arg.get_alpha<Tc>();
        const Tc                 base_beta    = arg.get_beta<Tc>();

        const int group_count = int(cfg.group_count);
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

        for(int g = 0; g < group_count; ++g)
        {
            const Ti m_g = Ti(arg.M + g);
            const Ti n_g = Ti(arg.N + g);
            const Ti k_g = Ti(arg.K + g);

            cfg.m_array[g] = m_g;
            cfg.n_array[g] = n_g;
            cfg.k_array[g] = k_g;

            cfg.transa_array[g]
                = (g % 2 == 0) ? base_trans_a : grouped_gemm_ex_toggle_n_t(base_trans_a);
            cfg.transb_array[g]
                = (g % 2 == 0) ? base_trans_b : grouped_gemm_ex_toggle_n_t(base_trans_b);

            const Ti lda_g = arg.lda > 0 ? Ti(arg.lda + g) : Ti(0);
            const Ti ldb_g = arg.ldb > 0 ? Ti(arg.ldb + g) : Ti(0);
            const Ti ldc_g = arg.ldc > 0 ? Ti(arg.ldc + g) : Ti(0);
            const Ti ldd_g = arg.ldd > 0 ? Ti(arg.ldd + g) : ldc_g;

            cfg.lda_array[g] = grouped_gemm_ex_lda(m_g, k_g, cfg.transa_array[g], lda_g);
            cfg.ldb_array[g] = grouped_gemm_ex_ldb(n_g, k_g, cfg.transb_array[g], ldb_g);
            cfg.ldc_array[g] = grouped_gemm_ex_ldc(m_g, ldc_g);
            cfg.ldd_array[g] = grouped_gemm_ex_ldc(m_g, ldd_g);

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
            cfg.max_ldd = std::max(cfg.max_ldd, cfg.ldd_array[g]);

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

template <typename Ti, typename To = Ti, typename Tc = To>
void testing_gemm_grouped_batched_ex_bad_arg(const Arguments& arg)
{
    auto fn    = arg.api == FORTRAN ? hipblasGemmGroupedBatchedExWithFlagsFortran
                                    : hipblasGemmGroupedBatchedExWithFlags;
    auto fn_64 = arg.api == FORTRAN_64 ? hipblasGemmGroupedBatchedExWithFlags_64Fortran
                                       : hipblasGemmGroupedBatchedExWithFlags_64;

    grouped_gemm_ex_test_config<Tc, int> cfg = grouped_gemm_ex_test_config_from_arg<Tc, int>(arg);
    grouped_gemm_ex_test_config<Tc, int64_t> cfg_64{};
    if(arg.api & c_API_64)
        cfg_64 = grouped_gemm_ex_test_config_from_arg<Tc, int64_t>(arg);

    const int group_count   = int(cfg.group_count);
    const int problem_count = int(cfg.problem_count);

    const size_t safe_size        = std::max(size_t(cfg.max_a_row) * size_t(cfg.max_a_col),
                                      size_t(cfg.max_b_row) * size_t(cfg.max_b_col));
    const size_t padded_safe_size = std::max(safe_size, size_t(cfg.max_m) * size_t(cfg.max_n));

    hipblasLocalHandle handle(arg);

    hipDataType          aType       = arg.a_type;
    hipDataType          bType       = arg.b_type;
    hipDataType          cType       = arg.c_type;
    hipDataType          dType       = cType;
    hipblasComputeType_t computeType = arg.compute_type_gemm;
    hipblasGemmAlgo_t    algo        = HIPBLAS_GEMM_DEFAULT;
    hipblasGemmFlags_t   flags       = HIPBLAS_GEMM_FLAGS_NONE;

    device_batch_matrix<Ti> dA(padded_safe_size, 1, 1, std::max(problem_count, 1));
    device_batch_matrix<Ti> dB(padded_safe_size, 1, 1, std::max(problem_count, 1));
    device_batch_matrix<To> dC(padded_safe_size, 1, 1, std::max(problem_count, 1));
    device_batch_matrix<To> dD(padded_safe_size, 1, 1, std::max(problem_count, 1));

    const void* const* dA_ptr = reinterpret_cast<const void* const*>(dA.ptr_on_device());
    const void* const* dB_ptr = reinterpret_cast<const void* const*>(dB.ptr_on_device());
    const void* const* dC_ptr = reinterpret_cast<const void* const*>(dC.ptr_on_device());
    void* const*       dD_ptr
        = const_cast<void* const*>(reinterpret_cast<const void* const*>(dD.ptr_on_device()));

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
                                    dA_ptr,
                                    aType,
                                    cfg_64.lda_array.data(),
                                    dB_ptr,
                                    bType,
                                    cfg_64.ldb_array.data(),
                                    cfg.beta_array.data(),
                                    dC_ptr,
                                    cType,
                                    cfg_64.ldc_array.data(),
                                    dD_ptr,
                                    dType,
                                    cfg_64.ldd_array.data(),
                                    cfg_64.group_count,
                                    cfg_64.group_size.data(),
                                    computeType,
                                    algo,
                                    flags),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(fn_64(handle,
                                    cfg.transa_array.data(),
                                    cfg.transb_array.data(),
                                    cfg_64.m_array.data(),
                                    cfg_64.n_array.data(),
                                    cfg_64.k_array.data(),
                                    nullptr,
                                    dA_ptr,
                                    aType,
                                    cfg_64.lda_array.data(),
                                    dB_ptr,
                                    bType,
                                    cfg_64.ldb_array.data(),
                                    cfg.beta_array.data(),
                                    dC_ptr,
                                    cType,
                                    cfg_64.ldc_array.data(),
                                    dD_ptr,
                                    dType,
                                    cfg_64.ldd_array.data(),
                                    cfg_64.group_count,
                                    cfg_64.group_size.data(),
                                    computeType,
                                    algo,
                                    flags),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(fn_64(handle,
                                    cfg.transa_array.data(),
                                    cfg.transb_array.data(),
                                    bad_m_array_64.data(),
                                    cfg_64.n_array.data(),
                                    cfg_64.k_array.data(),
                                    cfg.alpha_array.data(),
                                    dA_ptr,
                                    aType,
                                    cfg_64.lda_array.data(),
                                    dB_ptr,
                                    bType,
                                    cfg_64.ldb_array.data(),
                                    cfg.beta_array.data(),
                                    dC_ptr,
                                    cType,
                                    cfg_64.ldc_array.data(),
                                    dD_ptr,
                                    dType,
                                    cfg_64.ldd_array.data(),
                                    cfg_64.group_count,
                                    cfg_64.group_size.data(),
                                    computeType,
                                    algo,
                                    flags),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(fn_64(handle,
                                    cfg.transa_array.data(),
                                    cfg.transb_array.data(),
                                    cfg_64.m_array.data(),
                                    cfg_64.n_array.data(),
                                    cfg_64.k_array.data(),
                                    cfg.alpha_array.data(),
                                    dA_ptr,
                                    aType,
                                    cfg_64.lda_array.data(),
                                    dB_ptr,
                                    bType,
                                    cfg_64.ldb_array.data(),
                                    cfg.beta_array.data(),
                                    dC_ptr,
                                    cType,
                                    cfg_64.ldc_array.data(),
                                    dD_ptr,
                                    dType,
                                    cfg_64.ldd_array.data(),
                                    cfg_64.group_count,
                                    bad_group_size_64.data(),
                                    computeType,
                                    algo,
                                    flags),
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
                                 dA_ptr,
                                 aType,
                                 cfg.lda_array.data(),
                                 dB_ptr,
                                 bType,
                                 cfg.ldb_array.data(),
                                 cfg.beta_array.data(),
                                 dC_ptr,
                                 cType,
                                 cfg.ldc_array.data(),
                                 dD_ptr,
                                 dType,
                                 cfg.ldd_array.data(),
                                 group_count,
                                 cfg.group_size.data(),
                                 computeType,
                                 algo,
                                 flags),
                              HIPBLAS_STATUS_NOT_INITIALIZED);

        EXPECT_HIPBLAS_STATUS(fn(handle,
                                 cfg.transa_array.data(),
                                 cfg.transb_array.data(),
                                 cfg.m_array.data(),
                                 cfg.n_array.data(),
                                 cfg.k_array.data(),
                                 nullptr,
                                 dA_ptr,
                                 aType,
                                 cfg.lda_array.data(),
                                 dB_ptr,
                                 bType,
                                 cfg.ldb_array.data(),
                                 cfg.beta_array.data(),
                                 dC_ptr,
                                 cType,
                                 cfg.ldc_array.data(),
                                 dD_ptr,
                                 dType,
                                 cfg.ldd_array.data(),
                                 group_count,
                                 cfg.group_size.data(),
                                 computeType,
                                 algo,
                                 flags),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(fn(handle,
                                 cfg.transa_array.data(),
                                 cfg.transb_array.data(),
                                 bad_m_array.data(),
                                 cfg.n_array.data(),
                                 cfg.k_array.data(),
                                 cfg.alpha_array.data(),
                                 dA_ptr,
                                 aType,
                                 cfg.lda_array.data(),
                                 dB_ptr,
                                 bType,
                                 cfg.ldb_array.data(),
                                 cfg.beta_array.data(),
                                 dC_ptr,
                                 cType,
                                 cfg.ldc_array.data(),
                                 dD_ptr,
                                 dType,
                                 cfg.ldd_array.data(),
                                 group_count,
                                 cfg.group_size.data(),
                                 computeType,
                                 algo,
                                 flags),
                              HIPBLAS_STATUS_INVALID_VALUE);

        EXPECT_HIPBLAS_STATUS(fn(handle,
                                 cfg.transa_array.data(),
                                 cfg.transb_array.data(),
                                 cfg.m_array.data(),
                                 cfg.n_array.data(),
                                 cfg.k_array.data(),
                                 cfg.alpha_array.data(),
                                 dA_ptr,
                                 aType,
                                 cfg.lda_array.data(),
                                 dB_ptr,
                                 bType,
                                 cfg.ldb_array.data(),
                                 cfg.beta_array.data(),
                                 dC_ptr,
                                 cType,
                                 cfg.ldc_array.data(),
                                 dD_ptr,
                                 dType,
                                 cfg.ldd_array.data(),
                                 group_count,
                                 bad_group_size.data(),
                                 computeType,
                                 algo,
                                 flags),
                              HIPBLAS_STATUS_INVALID_VALUE);
    }
}

template <typename Ti, typename To = Ti, typename Tc = To>
void testing_gemm_grouped_batched_ex(const Arguments& arg)
{
    auto fn_with_flags    = arg.api == FORTRAN ? hipblasGemmGroupedBatchedExWithFlagsFortran
                                               : hipblasGemmGroupedBatchedExWithFlags;
    auto fn_64_with_flags = arg.api == FORTRAN_64 ? hipblasGemmGroupedBatchedExWithFlags_64Fortran
                                                  : hipblasGemmGroupedBatchedExWithFlags_64;
    auto fn = arg.api == FORTRAN ? hipblasGemmGroupedBatchedExFortran : hipblasGemmGroupedBatchedEx;
    auto fn_64 = arg.api == FORTRAN_64 ? hipblasGemmGroupedBatchedEx_64Fortran
                                       : hipblasGemmGroupedBatchedEx_64;

    grouped_gemm_ex_test_config<Tc, int> cfg = grouped_gemm_ex_test_config_from_arg<Tc, int>(arg);
    grouped_gemm_ex_test_config<Tc, int64_t> cfg_64{};
    if(arg.api & c_API_64)
        cfg_64 = grouped_gemm_ex_test_config_from_arg<Tc, int64_t>(arg);

    const int group_count   = int(cfg.group_count);
    const int problem_count = int(cfg.problem_count);

    if(group_count <= 0 || problem_count <= 0)
        return;

    hipblasLocalHandle handle(arg);

    hipDataType          aType       = arg.a_type;
    hipDataType          bType       = arg.b_type;
    hipDataType          cType       = arg.c_type;
    hipDataType          dType       = cType;
    hipblasComputeType_t computeType = arg.compute_type_gemm;
    hipblasGemmAlgo_t    algo        = HIPBLAS_GEMM_DEFAULT;
    hipblasGemmFlags_t   flags       = hipblasGemmFlags_t(arg.flags);

    const int unit_check = arg.unit_check;
    const int norm_check = arg.norm_check;
    const int timing     = arg.timing;

    double hipblas_error_host{0}, hipblas_error_device{0};

    host_batch_matrix<Ti> hA(cfg.max_a_row, cfg.max_a_col, cfg.max_lda, problem_count);
    host_batch_matrix<Ti> hB(cfg.max_b_row, cfg.max_b_col, cfg.max_ldb, problem_count);
    host_batch_matrix<To> hC(cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count);
    host_batch_matrix<To> hC_init(cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count);
    host_batch_matrix<To> hD_gold(cfg.max_m, cfg.max_n, cfg.max_ldd, problem_count);
    host_batch_matrix<To> hD_host(cfg.max_m, cfg.max_n, cfg.max_ldd, problem_count);
    host_batch_matrix<To> hD_device(cfg.max_m, cfg.max_n, cfg.max_ldd, problem_count);

    device_batch_matrix<Ti> dA(cfg.max_a_row, cfg.max_a_col, cfg.max_lda, problem_count);
    device_batch_matrix<Ti> dB(cfg.max_b_row, cfg.max_b_col, cfg.max_ldb, problem_count);
    device_batch_matrix<To> dC(cfg.max_m, cfg.max_n, cfg.max_ldc, problem_count);

    device_vector<Tc> d_alpha(group_count);
    device_vector<Tc> d_beta(group_count);

    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hC_init.memcheck());
    CHECK_HIP_ERROR(hD_gold.memcheck());
    CHECK_HIP_ERROR(hD_host.memcheck());
    CHECK_HIP_ERROR(hD_device.memcheck());
    CHECK_DEVICE_ALLOCATION(dA.memcheck());
    CHECK_DEVICE_ALLOCATION(dB.memcheck());
    CHECK_DEVICE_ALLOCATION(dC.memcheck());
    CHECK_DEVICE_ALLOCATION(d_alpha.memcheck());
    CHECK_DEVICE_ALLOCATION(d_beta.memcheck());

    const void* const* dA_ptr = reinterpret_cast<const void* const*>(dA.ptr_on_device());
    const void* const* dB_ptr = reinterpret_cast<const void* const*>(dB.ptr_on_device());
    const void* const* dC_ptr = reinterpret_cast<const void* const*>(dC.ptr_on_device());
    // C alias D or inplace, rocblas to test out of place
    void* const* dD_ptr = const_cast<void* const*>(dC_ptr);

    hipblas_init_matrix(hA, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, true);
    hipblas_init_matrix(
        hB, arg, hipblas_client_alpha_sets_nan, hipblas_general_matrix, false, true);
    hipblas_init_matrix(hC, arg, hipblas_client_beta_sets_nan, hipblas_general_matrix);
    hC_init.copy_from(hC);

    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

    if(unit_check || norm_check)
    {
        copy_matrix_with_different_leading_dimensions(hC, hD_gold);

        int idx = 0;
        for(int g = 0; g < group_count; ++g)
        {
            for(int p = 0; p < int(cfg.group_size[g]); ++p, ++idx)
            {
                ref_gemm<Ti, To, Tc>(cfg.transa_array[g],
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
                                     hD_gold[idx],
                                     int(cfg.ldd_array[g]));
            }
        }
    }

    const auto run_grouped_gemm_ex = [&](const void* alpha_ptr, const void* beta_ptr) {
        if(arg.api & c_API_64)
        {
            if(arg.with_flags)
            {
                CHECK_HIPBLAS_ERROR(fn_64_with_flags(handle,
                                                     cfg.transa_array.data(),
                                                     cfg.transb_array.data(),
                                                     cfg_64.m_array.data(),
                                                     cfg_64.n_array.data(),
                                                     cfg_64.k_array.data(),
                                                     alpha_ptr,
                                                     dA_ptr,
                                                     aType,
                                                     cfg_64.lda_array.data(),
                                                     dB_ptr,
                                                     bType,
                                                     cfg_64.ldb_array.data(),
                                                     beta_ptr,
                                                     dC_ptr,
                                                     cType,
                                                     cfg_64.ldc_array.data(),
                                                     dD_ptr,
                                                     dType,
                                                     cfg_64.ldd_array.data(),
                                                     cfg_64.group_count,
                                                     cfg_64.group_size.data(),
                                                     computeType,
                                                     algo,
                                                     flags));
            }
            else
            {
                CHECK_HIPBLAS_ERROR(fn_64(handle,
                                          cfg.transa_array.data(),
                                          cfg.transb_array.data(),
                                          cfg_64.m_array.data(),
                                          cfg_64.n_array.data(),
                                          cfg_64.k_array.data(),
                                          alpha_ptr,
                                          dA_ptr,
                                          aType,
                                          cfg_64.lda_array.data(),
                                          dB_ptr,
                                          bType,
                                          cfg_64.ldb_array.data(),
                                          beta_ptr,
                                          dC_ptr,
                                          cType,
                                          cfg_64.ldc_array.data(),
                                          dD_ptr,
                                          dType,
                                          cfg_64.ldd_array.data(),
                                          cfg_64.group_count,
                                          cfg_64.group_size.data(),
                                          computeType,
                                          algo));
            }
        }
        else
        {
            if(arg.with_flags)
            {
                CHECK_HIPBLAS_ERROR(fn_with_flags(handle,
                                                  cfg.transa_array.data(),
                                                  cfg.transb_array.data(),
                                                  cfg.m_array.data(),
                                                  cfg.n_array.data(),
                                                  cfg.k_array.data(),
                                                  alpha_ptr,
                                                  dA_ptr,
                                                  aType,
                                                  cfg.lda_array.data(),
                                                  dB_ptr,
                                                  bType,
                                                  cfg.ldb_array.data(),
                                                  beta_ptr,
                                                  dC_ptr,
                                                  cType,
                                                  cfg.ldc_array.data(),
                                                  dD_ptr,
                                                  dType,
                                                  cfg.ldd_array.data(),
                                                  group_count,
                                                  cfg.group_size.data(),
                                                  computeType,
                                                  algo,
                                                  flags));
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
                                       dA_ptr,
                                       aType,
                                       cfg.lda_array.data(),
                                       dB_ptr,
                                       bType,
                                       cfg.ldb_array.data(),
                                       beta_ptr,
                                       dC_ptr,
                                       cType,
                                       cfg.ldc_array.data(),
                                       dD_ptr,
                                       dType,
                                       cfg.ldd_array.data(),
                                       group_count,
                                       cfg.group_size.data(),
                                       computeType,
                                       algo));
            }
        }
    };

    const auto compare_to_gold = [&](host_batch_matrix<To>& hD) {
        if(unit_check)
        {
            int idx = 0;
            for(int g = 0; g < group_count; ++g)
            {
                for(int p = 0; p < int(cfg.group_size[g]); ++p, ++idx)
                {
                    if((getArchMajor() == 11)
                       && ((std::is_same<Tc, float>{} && std::is_same<Ti, hipblasBfloat16>{})
                           || (std::is_same<Tc, float>{} && std::is_same<Ti, hipblasHalf>{})
                           || (std::is_same<Tc, hipblasHalf>{} && std::is_same<Ti, hipblasHalf>{})))
                    {
                        const double tol
                            = int(cfg.k_array[g]) * sum_error_tolerance_for_gfx11<Tc, Ti, To>;
                        near_check_general<To>(int(cfg.m_array[g]),
                                               int(cfg.n_array[g]),
                                               int(cfg.ldd_array[g]),
                                               hD_gold[idx],
                                               hD[idx],
                                               tol);
                    }
                    else
                    {
                        unit_check_general<To>(int(cfg.m_array[g]),
                                               int(cfg.n_array[g]),
                                               int(cfg.ldd_array[g]),
                                               hD_gold[idx],
                                               hD[idx]);
                    }
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
                                     std::abs(norm_check_general<To>('F',
                                                                     int(cfg.m_array[g]),
                                                                     int(cfg.n_array[g]),
                                                                     int(cfg.ldd_array[g]),
                                                                     hD_gold[idx],
                                                                     hD[idx])));
                }
            }
        }
        return error;
    };

    if(unit_check || norm_check)
    {
        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));
        run_grouped_gemm_ex(cfg.alpha_array.data(), cfg.beta_array.data());
        CHECK_HIP_ERROR(hD_host.transfer_from(dC));

        CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));
        CHECK_HIP_ERROR(dC.transfer_from(hC_init));
        CHECK_HIP_ERROR(hipMemcpy(
            d_alpha, cfg.alpha_array.data(), group_count * sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            d_beta, cfg.beta_array.data(), group_count * sizeof(Tc), hipMemcpyHostToDevice));
        run_grouped_gemm_ex(d_alpha, d_beta);
        CHECK_HIP_ERROR(hD_device.transfer_from(dC));

        hipblas_error_host   = compare_to_gold(hD_host);
        hipblas_error_device = compare_to_gold(hD_device);
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
            run_grouped_gemm_ex(cfg.alpha_array.data(), cfg.beta_array.data());
        }

        gpu_time_used = get_time_us_sync(stream) - gpu_time_used;

        double gflop_count = 0.0;
        for(int g = 0; g < group_count; ++g)
            gflop_count += int(cfg.group_size[g])
                           * gemm_gflop_count<Tc>(
                               int(cfg.m_array[g]), int(cfg.n_array[g]), int(cfg.k_array[g]));

        hipblasGemmGroupedBatchedExModel{}.log_args<Tc>(
            std::cout, arg, gpu_time_used, gflop_count, hipblas_error_host, hipblas_error_device);
    }
}
