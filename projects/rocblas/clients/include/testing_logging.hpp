/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "logging.hpp"
#include "testing_common.hpp"
#include "type_dispatch.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#ifdef WIN32
#include <stdlib.h>
#define setenv(A, B, C) _putenv_s(A, B)
#else
#include <sys/param.h>
#include <unistd.h>
#endif

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error no filesystem found
#endif

template <typename T>
inline constexpr auto precision_letter = "*";
template <>
inline constexpr auto precision_letter<rocblas_half> = "h";
template <>
inline constexpr auto precision_letter<float> = "s";
template <>
inline constexpr auto precision_letter<double> = "d";
template <>
inline constexpr auto precision_letter<rocblas_float_complex> = "c";
template <>
inline constexpr auto precision_letter<rocblas_double_complex> = "z";

#ifdef WIN32
int diff_files(std::string path1, std::string path2)
{
    std::replace(path1.begin(), path1.end(), '/', '\\');
    std::replace(path2.begin(), path2.end(), '/', '\\');
    std::ifstream actual(path1);
    std::ifstream expected(path2);
    std::string   test((std::istreambuf_iterator<char>(actual)), std::istreambuf_iterator<char>());
    std::string gold((std::istreambuf_iterator<char>(expected)), std::istreambuf_iterator<char>());
    actual.close();
    expected.close();
    int cmp = test != gold;
    return cmp;
}
#endif

void erase_letter(std::string& str, char letter)
{
    size_t pos = str.find(letter);
    if(pos != std::string::npos)
    {
        str.erase(pos, 1);
    }
}

// replaces X in string with s, d, c, z or h depending on typename T
template <typename T>
std::string replaceX(rocblas_client_api api, std::string input_string)
{
    char letter(precision_letter<T>[0]);
    std::replace(input_string.begin(), input_string.end(), 'X', letter);
    // exceptions
    if(letter == 'c' || letter == 'z')
    {
        if(letter == 'c')
            std::replace(input_string.begin(), input_string.end(), 'P', 's');
        else
            std::replace(input_string.begin(), input_string.end(), 'P', 'd');

        std::replace(input_string.begin(), input_string.end(), 'U', 'u');
    }
    else
    {
        erase_letter(input_string, 'P');
        erase_letter(input_string, 'U');
    }

    if(api & c_API_64)
    {
        input_string += "_64";
    }
    return input_string;
}

template <typename T>
void testing_logging(const Arguments& arg)
{
    rocblas_pointer_mode test_pointer_mode
        = arg.pointer_mode_host ? rocblas_pointer_mode_host : rocblas_pointer_mode_device;

    // set environment variable ROCBLAS_LAYER to turn on logging. Note that setenv
    // only has scope for this executable, so it is not necessary to save and restore
    // this environment variable
    //
    // ROCBLAS_LAYER is a bit mask (see enum rocblas_layer_mode)
    // ROCBLAS_LAYER = 1 turns on log_trace
    // ROCBLAS_LAYER = 2 turns on log_bench
    // ROCBLAS_LAYER = 4 turns on log_profile
    // ROCBLAS_LAYER = 8 turns on internal function log_trace
    int setenv_status;

    setenv_status = setenv("ROCBLAS_LAYER", "3", true);
    ASSERT_EQ(setenv_status, 0);

    // open files
    static std::string tmp_dir = rocblas_tempname();

    std::string test_run(precision_letter<T>);
    test_run.append(arg.pointer_mode_host ? "_hostptr" : "_devptr");
    if(arg.api & c_API_64)
    {
        test_run.append("_I64");
    }

    const fs::path trace_fspath1 = tmp_dir + std::string("trace_") + test_run + std::string(".csv");
    const fs::path trace_fspath2
        = tmp_dir + std::string("trace_") + test_run + std::string("_gold.csv");
    const fs::path bench_fspath1 = tmp_dir + std::string("bench_") + test_run + std::string(".txt");
    const fs::path bench_fspath2
        = tmp_dir + std::string("bench_") + test_run + std::string("_gold.txt");

    std::string trace_path1 = trace_fspath1.generic_string();
    std::string trace_path2 = trace_fspath2.generic_string();
    std::string bench_path1 = bench_fspath1.generic_string();
    std::string bench_path2 = bench_fspath2.generic_string();

    // set environment variable to give pathname of for log_trace file
    setenv_status = setenv("ROCBLAS_LOG_TRACE_PATH", trace_path1.c_str(), true);
    ASSERT_EQ(setenv_status, 0);

    // set environment variable to give pathname of for log_bench file
    setenv_status = setenv("ROCBLAS_LOG_BENCH_PATH", bench_path1.c_str(), true);
    ASSERT_EQ(setenv_status, 0);

    //
    // call rocBLAS functions with log_trace and log_bench to output log_trace and log_bench files
    //

    int64_t m    = 3;
    int64_t n    = 2;
    int64_t k    = 1;
    int64_t kl   = 1;
    int64_t ku   = 1;
    int64_t incx = 1;
    int64_t incy = 1;

    int64_t        lda         = 4;
    rocblas_stride stride_a    = 40;
    int64_t        ldb         = 5;
    rocblas_stride stride_b    = 50;
    int64_t        ldc         = 6;
    rocblas_stride stride_c    = 60;
    int64_t        ldd         = 7;
    rocblas_stride stride_d    = 70;
    int64_t        batch_count = 1;

    T  h_alpha = 1.5;
    T* alpha   = &h_alpha;

    T  h_beta = 0.5;
    T* beta   = &h_beta;

    real_t<T>  h_real_alpha = 1.5;
    real_t<T>* real_alpha   = &h_real_alpha;

    // for gemm_ex
    float alpha_float = 1.5;
    float beta_float  = 0.5;

    DEVICE_MEMCHECK(device_vector<T>, d_alpha, (1));
    DEVICE_MEMCHECK(device_vector<real_t<T>>, d_real_alpha, (1));
    DEVICE_MEMCHECK(device_vector<T>, d_beta, (1));

    bool device_mode = test_pointer_mode == rocblas_pointer_mode_device;
    if(device_mode)
    {
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, beta, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(d_real_alpha, real_alpha, sizeof(real_t<T>), hipMemcpyHostToDevice));
        alpha      = d_alpha;
        beta       = d_beta;
        real_alpha = d_real_alpha;
    }

    rocblas_operation transA = rocblas_operation_none;
    rocblas_operation transB = rocblas_operation_transpose;
    rocblas_fill      uplo   = rocblas_fill_upper;
    rocblas_diagonal  diag   = rocblas_diagonal_unit;
    rocblas_side      side   = rocblas_side_left;

    // use m as larger
    int64_t size_x = m * incx;
    int64_t size_y = m * incy;

    // strides set above max requirements
    int64_t size_a = stride_a * batch_count;
    int64_t size_b = stride_b * batch_count;
    int64_t size_c = stride_c * batch_count;
    int64_t size_d = stride_d * batch_count;

    // allocate memory on device
    DEVICE_MEMCHECK(device_vector<T>, dx, (size_x));
    DEVICE_MEMCHECK(device_vector<T>, dy, (size_y));
    DEVICE_MEMCHECK(device_vector<T>, da, (size_a));
    DEVICE_MEMCHECK(device_vector<T>, db, (size_b));
    DEVICE_MEMCHECK(device_vector<T>, dc, (size_c));
    DEVICE_MEMCHECK(device_vector<T>, dd, (size_d));

    int64_t     h_i64_result;
    rocblas_int h_i32_result;

    DEVICE_MEMCHECK(device_vector<int64_t>, d_i64_result, (1));
    DEVICE_MEMCHECK(device_vector<rocblas_int>, d_i32_result, (1));

    DEVICE_MEMCHECK(device_vector<T>, d_result, (1));
    DEVICE_MEMCHECK(device_vector<real_t<T>>, d_real_result, (1));

    T            h_result;
    T*           result = &h_result;
    real_t<T>    h_real_result;
    real_t<T>*   real_result = &h_real_result;
    int64_t*     i64_result  = &h_i64_result;
    rocblas_int* i32_result  = &h_i32_result;

    if(device_mode)
    {
        result      = d_result;
        real_result = d_real_result;
        i64_result  = d_i64_result;
        i32_result  = d_i32_result;
    }

    rocblas_pointer_mode mode;

    // enclose in {} so rocblas_local_handle destructor called as it goes out of scope
    {
        // Auxiliary functions
        rocblas_local_handle handle;

        rocblas_set_pointer_mode(handle, test_pointer_mode);
        rocblas_get_pointer_mode(handle, &mode);

        if constexpr(!std::is_same_v<T, rocblas_half>)
        {

            // *************************************************** BLAS1 ***************************************************
            if(arg.api == C_64)
            {
                rocblas_iamax_64<T>(handle, n, dx, incx, i64_result);
                rocblas_iamin_64<T>(handle, n, dx, incx, i64_result);
            }
            else
            {
                rocblas_iamax<T>(handle, n, dx, incx, i32_result);
                rocblas_iamin<T>(handle, n, dx, incx, i32_result);
            }

            auto rocblas_asum_fn    = rocblas_asum<T>;
            auto rocblas_asum_fn_64 = rocblas_asum_64<T>;
            DAPI_CHECK(rocblas_asum_fn, (handle, n, dx, incx, real_result));

            auto rocblas_copy_fn    = rocblas_copy<T>;
            auto rocblas_copy_fn_64 = rocblas_copy_64<T>;
            DAPI_CHECK(rocblas_copy_fn, (handle, n, dx, incx, dy, incy));
        }

        auto rocblas_axpy_fn    = rocblas_axpy<T>;
        auto rocblas_axpy_fn_64 = rocblas_axpy_64<T>;
        DAPI_CHECK(rocblas_axpy_fn, (handle, n, alpha, dx, incx, dy, incy));

        auto rocblas_axpy_ex_fn    = rocblas_axpy_ex;
        auto rocblas_axpy_ex_fn_64 = rocblas_axpy_ex_64;
        // fixed datatype dispatch
        rocblas_datatype dt = rocblas_type2datatype<T>();
        DAPI_CHECK(rocblas_axpy_ex_fn, (handle, n, alpha, dt, dx, dt, incx, dy, dt, incy, dt));

        auto rocblas_dot_fn    = rocblas_dot<T>;
        auto rocblas_dot_fn_64 = rocblas_dot_64<T>;
        DAPI_CHECK(rocblas_dot_fn, (handle, n, dx, incx, dy, incy, result));

        if constexpr(!std::is_same_v<T, rocblas_half>)
        {

            auto rocblas_nrm2_fn    = rocblas_nrm2<T>;
            auto rocblas_nrm2_fn_64 = rocblas_nrm2_64<T>;
            DAPI_CHECK(rocblas_nrm2_fn, (handle, n, dx, incx, real_result));

            auto rocblas_scal_r_fn    = rocblas_scal<T, real_t<T>>;
            auto rocblas_scal_r_fn_64 = rocblas_scal_64<T, real_t<T>>;
            DAPI_CHECK(rocblas_scal_r_fn, (handle, n, real_alpha, dx, incx));

            auto rocblas_scal_fn    = rocblas_scal<T>;
            auto rocblas_scal_fn_64 = rocblas_scal_64<T>;
            DAPI_CHECK(rocblas_scal_fn, (handle, n, alpha, dx, incx));

            auto rocblas_swap_fn    = rocblas_swap<T>;
            auto rocblas_swap_fn_64 = rocblas_swap_64<T>;
            DAPI_CHECK(rocblas_swap_fn, (handle, n, dx, incx, dy, incy));

            // *************************************************** BLAS2 ***************************************************

            auto rocblas_gbmv_fn    = rocblas_gbmv<T>;
            auto rocblas_gbmv_fn_64 = rocblas_gbmv_64<T>;
            DAPI_CHECK(rocblas_gbmv_fn,
                       (handle, transA, m, n, kl, ku, alpha, da, lda, dx, incx, beta, dy, incy));

            auto rocblas_gemv_fn    = rocblas_gemv<T>;
            auto rocblas_gemv_fn_64 = rocblas_gemv_64<T>;
            DAPI_CHECK(rocblas_gemv_fn,
                       (handle, transA, m, n, alpha, da, lda, dx, incx, beta, dy, incy));

            auto rocblas_ger_fn    = rocblas_ger<T, false>;
            auto rocblas_ger_fn_64 = rocblas_ger_64<T, false>;
            DAPI_CHECK(rocblas_ger_fn, (handle, m, n, alpha, dx, incx, dy, incy, da, lda));

            if constexpr(!rocblas_is_complex<T>)
            {
                auto rocblas_sbmv_fn    = rocblas_sbmv<T>;
                auto rocblas_sbmv_fn_64 = rocblas_sbmv_64<T>;
                DAPI_CHECK(rocblas_sbmv_fn,
                           (handle, uplo, n, k, alpha, da, lda, dx, incx, beta, dy, incy));

                auto rocblas_spmv_fn    = rocblas_spmv<T>;
                auto rocblas_spmv_fn_64 = rocblas_spmv_64<T>;
                DAPI_CHECK(rocblas_spmv_fn, (handle, uplo, n, alpha, da, dx, incx, beta, dy, incy));

                auto rocblas_spr_fn    = rocblas_spr<T>;
                auto rocblas_spr_fn_64 = rocblas_spr_64<T>;
                DAPI_CHECK(rocblas_spr_fn, (handle, uplo, n, alpha, dx, incx, da));

                auto rocblas_spr2_fn    = rocblas_spr2<T>;
                auto rocblas_spr2_fn_64 = rocblas_spr2_64<T>;
                DAPI_CHECK(rocblas_spr2_fn, (handle, uplo, n, alpha, dx, incx, dy, incy, da));
            }

            auto rocblas_symv_fn    = rocblas_symv<T>;
            auto rocblas_symv_fn_64 = rocblas_symv_64<T>;
            DAPI_CHECK(rocblas_symv_fn,
                       (handle, uplo, n, alpha, da, lda, dx, incx, beta, dy, incy));

            auto rocblas_syr_fn    = rocblas_syr<T>;
            auto rocblas_syr_fn_64 = rocblas_syr_64<T>;
            DAPI_CHECK(rocblas_syr_fn, (handle, uplo, n, alpha, dx, incx, da, lda));

            auto rocblas_syr2_fn    = rocblas_syr2<T>;
            auto rocblas_syr2_fn_64 = rocblas_syr2_64<T>;
            DAPI_CHECK(rocblas_syr2_fn, (handle, uplo, n, alpha, dx, incx, dy, incy, da, lda));

            auto rocblas_tbmv_fn    = rocblas_tbmv<T>;
            auto rocblas_tbmv_fn_64 = rocblas_tbmv_64<T>;
            DAPI_CHECK(rocblas_tbmv_fn, (handle, uplo, transA, diag, n, k, da, lda, dx, incx));

            auto rocblas_tpmv_fn    = rocblas_tpmv<T>;
            auto rocblas_tpmv_fn_64 = rocblas_tpmv_64<T>;
            DAPI_CHECK(rocblas_tpmv_fn, (handle, uplo, transA, diag, n, da, dx, incx));

            auto rocblas_trmv_fn    = rocblas_trmv<T>;
            auto rocblas_trmv_fn_64 = rocblas_trmv_64<T>;
            DAPI_CHECK(rocblas_trmv_fn, (handle, uplo, transA, diag, n, da, lda, dx, incx));

            auto rocblas_tbsv_fn    = rocblas_tbsv<T>;
            auto rocblas_tbsv_fn_64 = rocblas_tbsv_64<T>;
            DAPI_CHECK(rocblas_tbsv_fn, (handle, uplo, transA, diag, n, k, da, lda, dx, incx));

            auto rocblas_tpsv_fn    = rocblas_tpsv<T>;
            auto rocblas_tpsv_fn_64 = rocblas_tpsv_64<T>;
            DAPI_CHECK(rocblas_tpsv_fn, (handle, uplo, transA, diag, n, da, dx, incx));

            auto rocblas_trsv_fn    = rocblas_trsv<T>;
            auto rocblas_trsv_fn_64 = rocblas_trsv_64<T>;
            DAPI_CHECK(rocblas_trsv_fn, (handle, uplo, transA, diag, n, da, lda, dx, incx));

            // *************************************************** BLAS3 ***************************************************
            auto rocblas_trsm_fn    = rocblas_trsm<T>;
            auto rocblas_trsm_fn_64 = rocblas_trsm_64<T>;
            DAPI_CHECK(rocblas_trsm_fn,
                       (handle, side, uplo, transA, diag, m, n, alpha, da, lda, db, ldb));

            auto rocblas_geam_fn    = rocblas_geam<T>;
            auto rocblas_geam_fn_64 = rocblas_geam_64<T>;
            DAPI_CHECK(rocblas_geam_fn,
                       (handle, transA, transB, m, n, alpha, da, lda, beta, db, ldb, dc, ldc));

            auto rocblas_symm_fn    = rocblas_symm<T>;
            auto rocblas_symm_fn_64 = rocblas_symm_64<T>;
            DAPI_CHECK(rocblas_symm_fn,
                       (handle, side, uplo, m, n, alpha, da, lda, db, ldb, beta, dc, ldc));

            auto rocblas_syrk_fn    = rocblas_syrk<T>;
            auto rocblas_syrk_fn_64 = rocblas_syrk_64<T>;
            DAPI_CHECK(rocblas_syrk_fn,
                       (handle, uplo, transA, n, k, alpha, da, lda, beta, dc, ldc));

            auto rocblas_syr2k_fn    = rocblas_syr2k<T>;
            auto rocblas_syr2k_fn_64 = rocblas_syr2k_64<T>;
            DAPI_CHECK(rocblas_syr2k_fn,
                       (handle, uplo, transA, n, k, alpha, da, lda, db, ldb, beta, dc, ldc));

            auto rocblas_syrkx_fn    = rocblas_syrkx<T>;
            auto rocblas_syrkx_fn_64 = rocblas_syrkx_64<T>;
            DAPI_CHECK(rocblas_syrkx_fn,
                       (handle, uplo, transA, n, k, alpha, da, lda, db, ldb, beta, dc, ldc));

            auto rocblas_trmm_fn    = rocblas_trmm<T>;
            auto rocblas_trmm_fn_64 = rocblas_trmm_64<T>;
            DAPI_CHECK(rocblas_trmm_fn,
                       (handle, side, uplo, transA, diag, m, n, alpha, da, lda, db, ldb, dc, ldc));
        } // !half

        // half allowed
        auto rocblas_gemm_fn    = rocblas_gemm<T>;
        auto rocblas_gemm_fn_64 = rocblas_gemm_64<T>;
        DAPI_CHECK(rocblas_gemm_fn,
                   (handle, transA, transB, m, n, k, alpha, da, lda, db, ldb, beta, dc, ldc));

        auto rocblas_gemm_strided_batched_fn    = rocblas_gemm_strided_batched<T>;
        auto rocblas_gemm_strided_batched_fn_64 = rocblas_gemm_strided_batched_64<T>;
        DAPI_CHECK(rocblas_gemm_strided_batched_fn,
                   (handle,
                    transA,
                    transB,
                    m,
                    n,
                    k,
                    alpha,
                    da,
                    lda,
                    stride_a,
                    db,
                    ldb,
                    stride_b,
                    beta,
                    dc,
                    ldc,
                    stride_c,
                    batch_count));

        // exclude trtri as it is an internal function
        //      rocblas_trtri<T>(handle, uplo, diag, n, da, lda, db, ldb);

        // trmm
        // tritri

        if constexpr(!rocblas_is_complex<T>)
        {

            // BLAS_EX

            void* alpha_gemm_ex = nullptr;
            void* beta_gemm_ex  = nullptr;

            rocblas_half      alpha_half(alpha_float);
            rocblas_half      beta_half(beta_float);
            double            alpha_double(alpha_float);
            double            beta_double(beta_float);
            rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
            int32_t           solution_index = 0;
            uint32_t          flags          = 0;
            rocblas_datatype  a_type;
            rocblas_datatype  b_type;
            rocblas_datatype  c_type;
            rocblas_datatype  d_type;
            rocblas_datatype  compute_type;

            DEVICE_MEMCHECK(device_vector<rocblas_half>, d_alpha_half, (1));
            DEVICE_MEMCHECK(device_vector<rocblas_half>, d_beta_half, (1));

            DEVICE_MEMCHECK(device_vector<float>, d_alpha_float, (1));
            DEVICE_MEMCHECK(device_vector<float>, d_beta_float, (1));

            DEVICE_MEMCHECK(device_vector<double>, d_alpha_double, (1));
            DEVICE_MEMCHECK(device_vector<double>, d_beta_double, (1));

            if constexpr(std::is_same_v<T, rocblas_half>)
            {
                a_type        = rocblas_datatype_f16_r;
                b_type        = rocblas_datatype_f16_r;
                c_type        = rocblas_datatype_f16_r;
                d_type        = rocblas_datatype_f16_r;
                compute_type  = rocblas_datatype_f16_r;
                alpha_gemm_ex = &alpha_half;
                beta_gemm_ex  = &beta_half;
                if(device_mode)
                {
                    CHECK_HIP_ERROR(hipMemcpy(
                        d_alpha_half, alpha_gemm_ex, sizeof(rocblas_half), hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(hipMemcpy(
                        d_beta_half, beta_gemm_ex, sizeof(rocblas_half), hipMemcpyHostToDevice));
                    alpha_gemm_ex = d_alpha_half;
                    beta_gemm_ex  = d_beta_half;
                }
            }
            else if constexpr(std::is_same_v<T, float>)
            {
                a_type        = rocblas_datatype_f32_r;
                b_type        = rocblas_datatype_f32_r;
                c_type        = rocblas_datatype_f32_r;
                d_type        = rocblas_datatype_f32_r;
                compute_type  = rocblas_datatype_f32_r;
                alpha_gemm_ex = &alpha_float;
                beta_gemm_ex  = &beta_float;
                if(device_mode)
                {
                    CHECK_HIP_ERROR(hipMemcpy(
                        d_alpha_float, alpha_gemm_ex, sizeof(float), hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(hipMemcpy(
                        d_beta_float, beta_gemm_ex, sizeof(float), hipMemcpyHostToDevice));
                    alpha_gemm_ex = d_alpha_float;
                    beta_gemm_ex  = d_beta_float;
                }
            }
            else if constexpr(std::is_same_v<T, double>)
            {
                a_type        = rocblas_datatype_f64_r;
                b_type        = rocblas_datatype_f64_r;
                c_type        = rocblas_datatype_f64_r;
                d_type        = rocblas_datatype_f64_r;
                compute_type  = rocblas_datatype_f64_r;
                alpha_gemm_ex = &alpha_double;
                beta_gemm_ex  = &beta_double;
                if(device_mode)
                {
                    CHECK_HIP_ERROR(hipMemcpy(
                        d_alpha_double, alpha_gemm_ex, sizeof(double), hipMemcpyHostToDevice));
                    CHECK_HIP_ERROR(hipMemcpy(
                        d_beta_double, beta_gemm_ex, sizeof(double), hipMemcpyHostToDevice));
                    alpha_gemm_ex = d_alpha_double;
                    beta_gemm_ex  = d_beta_double;
                }
            }

            auto rocblas_gemm_ex_fn    = rocblas_gemm_ex;
            auto rocblas_gemm_ex_fn_64 = rocblas_gemm_ex_64;
            DAPI_CHECK(rocblas_gemm_ex_fn,
                       (handle,
                        transA,
                        transB,
                        m,
                        n,
                        k,
                        alpha_gemm_ex,
                        da,
                        a_type,
                        lda,
                        db,
                        b_type,
                        ldb,
                        beta_gemm_ex,
                        dc,
                        c_type,
                        ldc,
                        dd,
                        d_type,
                        ldd,
                        compute_type,
                        algo,
                        solution_index,
                        flags));

            auto rocblas_gemm_strided_batched_ex_fn    = rocblas_gemm_strided_batched_ex;
            auto rocblas_gemm_strided_batched_ex_fn_64 = rocblas_gemm_strided_batched_ex_64;
            DAPI_CHECK(rocblas_gemm_strided_batched_ex_fn,
                       (handle,
                        transA,
                        transB,
                        m,
                        n,
                        k,
                        alpha_gemm_ex,
                        da,
                        a_type,
                        lda,
                        stride_a,
                        db,
                        b_type,
                        ldb,
                        stride_b,
                        beta_gemm_ex,
                        dc,
                        c_type,
                        ldc,
                        stride_c,
                        dd,
                        d_type,
                        ldd,
                        stride_d,
                        batch_count,
                        compute_type,
                        algo,
                        solution_index,
                        flags));
        } // !complex
    } // local_handle

#ifdef BUILD_WITH_HIPBLASLT

    if(arg.algo == 1)
    {
        if constexpr(!rocblas_is_complex<T>)
        {
            // internal API test requires hipblaslt backend to be used, see hardcoded results section

            // GEMM internal API trace 8
            setenv_status = setenv("ROCBLAS_LAYER", "8", true);
            ASSERT_EQ(setenv_status, 0);

            // Auxiliary functions
            rocblas_local_handle handle;

            rocblas_set_pointer_mode(handle, test_pointer_mode);

            CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
                handle, transA, transB, m, n, k, alpha, da, lda, db, ldb, beta, dc, ldc));

        } // !complex
    }

#endif // BUILD_WITH_HIPBLASLT

    setenv_status = setenv("ROCBLAS_LAYER", "0", true);
    ASSERT_EQ(setenv_status, 0);

    //
    // write "golden file"
    //

    std::ofstream trace_ofs;
    std::ofstream bench_ofs;

    trace_ofs.exceptions(std::ifstream::badbit | std::ifstream::failbit);
    trace_ofs.exceptions(std::ofstream::badbit | std::ofstream::failbit);

    trace_ofs.open(trace_path2);
    bench_ofs.open(bench_path2);

    ASSERT_FALSE(trace_ofs.fail());
    ASSERT_FALSE(trace_ofs.bad());
    ASSERT_FALSE(bench_ofs.fail());
    ASSERT_FALSE(bench_ofs.bad());

    rocblas_internal_ostream trace_ofs2;
    rocblas_internal_ostream bench_ofs2;

    std::string bench(arg.api == C_64 ? "./rocblas-bench --api 1" : "./rocblas-bench");

    // Auxiliary function
    int pmode = (int)test_pointer_mode;
    trace_ofs2 << "rocblas_create_handle,atomics_not_allowed\n";
    trace_ofs2 << "rocblas_set_pointer_mode," << pmode << ",atomics_not_allowed\n";
    trace_ofs2 << "rocblas_get_pointer_mode," << pmode << ",atomics_not_allowed\n";

    auto transA_letter = rocblas2char_operation(transA);
    auto transB_letter = rocblas2char_operation(transB);
    auto side_letter   = rocblas2char_side(side);
    auto uplo_letter   = rocblas2char_fill(uplo);
    auto diag_letter   = rocblas2char_diagonal(diag);

    // *************************************************** BLAS1 ***************************************************

    const char* bench_endl = " \n"; // space for empty atomics enabled default

    if constexpr(!std::is_same_v<T, rocblas_half>)
    {
        //
        // AMAX
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_iXamax") << "," << n << "," << (void*)dx << ","
                   << incx << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f iamax -r " << rocblas_precision_string<T> << " -n " << n
                   << " --incx " << incx << bench_endl;

        //
        // AMIN
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_iXamin") << "," << n << "," << (void*)dx << ","
                   << incx << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f iamin -r " << rocblas_precision_string<T> << " -n " << n
                   << " --incx " << incx << bench_endl;

        //
        // ASUM
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_PXasum") << "," << n << "," << (void*)dx << ","
                   << incx << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f asum -r " << rocblas_precision_string<T> << " -n " << n
                   << " --incx " << incx << bench_endl;

        //
        // COPY
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xcopy") << "," << n << "," << (void*)dx << ","
                   << incx << "," << (void*)dy << "," << incy << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f copy -r " << rocblas_precision_string<T> << " -n " << n
                   << " --incx " << incx << " --incy " << incy << bench_endl;
    }

    //
    // AXPY
    //
    trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xaxpy") << "," << n << "," << h_alpha << ","
               << (void*)dx << "," << incx << "," << (void*)dy << "," << incy
               << ",atomics_not_allowed\n";
    bench_ofs2 << bench.c_str() << " -f axpy -r " << rocblas_precision_string<T> << " -n " << n
               << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --incx "
               << incx << " --incy " << incy << bench_endl;

    //
    // AXPY_EX
    //
    std::string dt_str(rocblas_datatype_string(rocblas_type2datatype<T>()));
    if(test_pointer_mode == rocblas_pointer_mode_host)
    {
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_axpy_ex") << "," << n << "," << h_alpha << ","
                   << dt_str << "," << (void*)dx << "," << dt_str << "," << incx << "," << (void*)dy
                   << "," << dt_str << "," << incy << "," << dt_str << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f axpy_ex"
                   << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --a_type "
                   << dt_str << " --b_type " << dt_str << " --incx " << incx << " --c_type "
                   << dt_str << " --incy " << incy << " --compute_type " << dt_str << bench_endl;
    }
    else
    {
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_axpy_ex") << "," << n << "," << dt_str << ","
                   << (void*)dx << "," << dt_str << "," << incx << "," << (void*)dy << "," << dt_str
                   << "," << incy << "," << dt_str << ",atomics_not_allowed\n";
        // no bench output TODO
    }

    //
    // DOT
    //
    trace_ofs2 << replaceX<T>(arg.api, "rocblas_XdotU") << "," << n << "," << (void*)dx << ","
               << incx << "," << (void*)dy << "," << incy << ",atomics_not_allowed\n";
    bench_ofs2 << bench.c_str() << " -f dot -r " << rocblas_precision_string<T> << " -n " << n
               << " --incx " << incx << " --incy " << incy << bench_endl;

    if constexpr(!std::is_same_v<T, rocblas_half>)
    {

        //
        // NRM2
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_PXnrm2") << "," << n << "," << (void*)dx << ","
                   << incx << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f nrm2 -r " << rocblas_precision_string<T> << " -n " << n
                   << " --incx " << incx << bench_endl;

        //
        // SCAL (real alpha)
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_XPscal") << "," << n << "," << h_real_alpha
                   << "," << (void*)dx << "," << incx << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f scal --a_type "
                   << rocblas_precision_string<T> << " --b_type "
                   << rocblas_precision_string<real_t<T>> << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_real_alpha) << " --incx "
                   << incx << bench_endl;

        //
        // SCAL
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xscal") << "," << n << "," << h_alpha << ","
                   << (void*)dx << "," << incx << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f scal --a_type "
                   << rocblas_precision_string<T> << " --b_type "
                   << rocblas_precision_string<T> << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --incx "
                   << incx << bench_endl;

        //
        // SWAP
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xswap") << "," << n << "," << (void*)dx << ","
                   << incx << "," << (void*)dy << "," << incy << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f swap -r " << rocblas_precision_string<T> << " -n " << n
                   << " --incx " << incx << " --incy " << incy << bench_endl;

        // *************************************************** BLAS2 ***************************************************

        //
        // GBMV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xgbmv") << "," << transA << "," << m << "," << n
                   << "," << kl << "," << ku << "," << h_alpha << "," << (void*)da << "," << lda
                   << "," << (void*)dx << "," << incx << "," << h_beta << "," << (void*)dy << ","
                   << incy << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f gbmv -r "
                   << rocblas_precision_string<T> << " --transposeA " << transA_letter << " -m "
                   << m << " -n " << n << " --kl " << kl << " --ku " << ku << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda " << lda
                   << " --incx " << incx << " "
                   << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --incy " << incy
                   << bench_endl;

        //
        // GEMV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xgemv") << "," << transA << "," << m << "," << n
                   << "," << h_alpha << "," << (void*)da << "," << lda << "," << (void*)dx << ","
                   << incx << "," << h_beta << "," << (void*)dy << "," << incy
                   << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f gemv -r "
                   << rocblas_precision_string<T> << " --transposeA " << transA_letter << " -m "
                   << m << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda " << lda
                   << " --incx " << incx << " "
                   << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --incy " << incy
                   << bench_endl;

        //
        // GER
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_XgerU") << "," << m << "," << n << ","
                   << h_alpha << "," << (void*)dx << "," << incx << "," << (void*)dy << "," << incy
                   << "," << (void*)da << "," << lda << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f " << replaceX<T>(rocblas_client_api::C, "gerU")
                   << " -r " << rocblas_precision_string<T> << " -m " << m << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --incx "
                   << incx << " --incy " << incy << " --lda " << lda << bench_endl;

        if constexpr(!rocblas_is_complex<T>)
        {
            //
            // SBMV
            //
            trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xsbmv") << "," << uplo << "," << n << ","
                       << k << "," << h_alpha << "," << (void*)da << "," << lda << "," << (void*)dx
                       << "," << incx << "," << h_beta << "," << (void*)dy << "," << incy
                       << ",atomics_not_allowed\n";
            bench_ofs2 << bench.c_str() << " -f sbmv -r "
                       << rocblas_precision_string<T> << " --uplo " << uplo_letter << " -n " << n
                       << " -k " << k << " "
                       << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda "
                       << lda << " --incx " << incx << " "
                       << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --incy "
                       << incy << bench_endl;

            //
            // SPMV
            //
            trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xspmv") << "," << uplo << "," << n << ","
                       << h_alpha << "," << (void*)da << "," << (void*)dx << "," << incx << ","
                       << h_beta << "," << (void*)dy << "," << incy << ",atomics_not_allowed\n";
            bench_ofs2 << bench.c_str() << " -f spmv -r "
                       << rocblas_precision_string<T> << " --uplo " << uplo_letter << " -n " << n
                       << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha)
                       << " --incx " << incx << " "
                       << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --incy "
                       << incy << bench_endl;

            //
            // SPR
            //
            trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xspr") << "," << uplo << "," << n << ","
                       << h_alpha << "," << (void*)dx << "," << incx << "," << (void*)da
                       << ",atomics_not_allowed\n";
            bench_ofs2 << bench.c_str() << " -f spr -r "
                       << rocblas_precision_string<T> << " --uplo " << uplo_letter << " -n " << n
                       << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha)
                       << " --incx " << incx << bench_endl;

            //
            // SPR2
            //
            trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xspr2") << "," << uplo << "," << n << ","
                       << h_alpha << "," << (void*)dx << "," << incx << "," << (void*)dy << ","
                       << incy << "," << (void*)da << ",atomics_not_allowed\n";
            bench_ofs2 << bench.c_str() << " -f spr2 -r "
                       << rocblas_precision_string<T> << " --uplo " << uplo_letter << " -n " << n
                       << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha)
                       << " --incx " << incx << " --incy " << incy << bench_endl;
        }

        //
        // SYMV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xsymv") << "," << uplo << "," << n << ","
                   << h_alpha << "," << (void*)da << "," << lda << "," << (void*)dx << "," << incx
                   << "," << h_beta << "," << (void*)dy << "," << incy << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f symv -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda " << lda
                   << " --incx " << incx << " "
                   << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --incy " << incy
                   << bench_endl;

        //
        // SYR
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xsyr") << "," << uplo << "," << n << ","
                   << h_alpha << "," << (void*)dx << "," << incx << "," << (void*)da << "," << lda
                   << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f syr -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --incx "
                   << incx << " --lda " << lda << bench_endl;

        //
        // SYR2
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xsyr2") << "," << uplo << "," << n << ","
                   << h_alpha << "," << (void*)dx << "," << incx << "," << (void*)dy << "," << incy
                   << "," << (void*)da << "," << lda << ",atomics_not_allowed\n";
        bench_ofs2 << bench.c_str() << " -f syr2 -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda " << lda
                   << " --incx " << incx << " --incy " << incy << bench_endl;

        //
        // TBMV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xtbmv") << "," << uplo << "," << transA << ","
                   << diag << "," << n << "," << k << "," << (void*)da << "," << lda << ","
                   << (void*)dx << "," << incx << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f tbmv -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " --transposeA " << transA_letter << " --diag " << diag_letter
                   << " -n " << n << " -k " << k << " --lda " << lda << " --incx " << incx
                   << bench_endl;

        //
        // TPMV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xtpmv") << "," << uplo << "," << transA << ","
                   << diag << "," << n << "," << (void*)da << "," << (void*)dx << "," << incx
                   << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f tpmv -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " --transposeA " << transA_letter << " --diag " << diag_letter
                   << " -n " << n << " --incx " << incx << bench_endl;

        //
        // TRMV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xtrmv") << "," << uplo << "," << transA << ","
                   << diag << "," << n << "," << (void*)da << "," << lda << "," << (void*)dx << ","
                   << incx << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f trmv -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " --transposeA " << transA_letter << " --diag " << diag_letter
                   << " -n " << n << " --lda " << lda << " --incx " << incx << bench_endl;

        //
        // TBSV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xtbsv") << "," << uplo << "," << transA << ","
                   << diag << "," << n << "," << k << "," << (void*)da << "," << lda << ","
                   << (void*)dx << "," << incx << ",atomics_not_allowed\n";
        if(test_pointer_mode == rocblas_pointer_mode_host)
            bench_ofs2 << bench.c_str() << " -f tbsv -r "
                       << rocblas_precision_string<T> << " --uplo " << uplo_letter
                       << " --transposeA " << transA_letter << " --diag " << diag_letter << " -n "
                       << n << " -k " << k << " --lda " << lda << " --incx " << incx << bench_endl;

        //
        // TPSV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xtpsv") << "," << uplo << "," << transA << ","
                   << diag << "," << n << "," << (void*)da << "," << (void*)dx << "," << incx
                   << ",atomics_not_allowed\n";
        if(test_pointer_mode == rocblas_pointer_mode_host)
            bench_ofs2 << bench.c_str() << " -f tpsv -r "
                       << rocblas_precision_string<T> << " --uplo " << uplo_letter
                       << " --transposeA " << transA_letter << " --diag " << diag_letter << " -n "
                       << n << " --incx " << incx << bench_endl;

        //
        // TRSV
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xtrsv") << "," << uplo << "," << transA << ","
                   << diag << "," << n << "," << (void*)da << "," << lda << "," << (void*)dx << ","
                   << incx << ",atomics_not_allowed\n";

        if(test_pointer_mode == rocblas_pointer_mode_host)
            bench_ofs2 << bench.c_str() << " -f trsv -r "
                       << rocblas_precision_string<T> << " --uplo " << uplo_letter
                       << " --transposeA " << transA_letter << " --diag " << diag_letter << " -n "
                       << n << " --lda " << lda << " --incx " << incx << bench_endl;

        // *************************************************** BLAS3 ***************************************************

        //
        // TRSM
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xtrsm") << "," << side << "," << uplo << ","
                   << transA << "," << diag << "," << m << "," << n << "," << h_alpha << ","
                   << (void*)da << "," << lda << "," << (void*)db << "," << ldb
                   << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f trsm -r " << rocblas_precision_string<T> << " --side "
                   << side_letter << " --uplo " << uplo_letter << " --transposeA " << transA_letter
                   << " --diag " << diag_letter << " -m " << m << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda " << lda
                   << " --ldb " << ldb << bench_endl;

        //
        // GEAM
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xgeam") << "," << transA << "," << transB << ","
                   << m << "," << n << "," << h_alpha << "," << (void*)da << "," << lda << ","
                   << h_beta << "," << (void*)db << "," << ldb << "," << (void*)dc << "," << ldc
                   << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f geam -r "
                   << rocblas_precision_string<T> << " --transposeA " << transA_letter
                   << " --transposeB " << transB_letter << " -m " << m << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda " << lda
                   << " " << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --ldb "
                   << ldb << " --ldc " << ldc << bench_endl;

        //
        // SYMM
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xsymm") << "," << side << "," << uplo << ","
                   << m << "," << n << "," << h_alpha << "," << (void*)da << "," << lda << ","
                   << (void*)db << "," << ldb << "," << h_beta << "," << (void*)dc << "," << ldc
                   << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f symm -r " << rocblas_precision_string<T> << " --side "
                   << side_letter << " --uplo " << uplo_letter << " -m " << m << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda " << lda
                   << " --ldb " << ldb << " "
                   << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --ldc " << ldc
                   << bench_endl;

        //
        // SYRK
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xsyrk") << "," << uplo << "," << transA << ","
                   << n << "," << k << "," << h_alpha << "," << (void*)da << "," << lda << ","
                   << h_beta << "," << (void*)dc << "," << ldc << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f syrk -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " --transposeA " << transA_letter << " -n " << n << " -k " << k
                   << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda "
                   << lda << " " << rocblas_internal_log_bench_scalar_value("beta", &h_beta)
                   << " --ldc " << ldc << bench_endl;

        //
        // SYR2K
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xsyr2k") << "," << uplo << "," << transA << ","
                   << n << "," << k << "," << h_alpha << "," << (void*)da << "," << lda << ","
                   << (void*)db << "," << ldb << "," << h_beta << "," << (void*)dc << "," << ldc
                   << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f syr2k -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " --transposeA " << transA_letter << " -n " << n << " -k " << k
                   << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda "
                   << lda << " --ldb " << ldb << " "
                   << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --ldc " << ldc
                   << bench_endl;

        //
        // SYRKX
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xsyrkx") << "," << uplo << "," << transA << ","
                   << n << "," << k << "," << h_alpha << "," << (void*)da << "," << lda << ","
                   << (void*)db << "," << ldb << "," << h_beta << "," << (void*)dc << "," << ldc
                   << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f syrkx -r " << rocblas_precision_string<T> << " --uplo "
                   << uplo_letter << " --transposeA " << transA_letter << " -n " << n << " -k " << k
                   << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda "
                   << lda << " --ldb " << ldb << " "
                   << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --ldc " << ldc
                   << bench_endl;

        //
        // TRMM
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xtrmm") << "," << side << "," << uplo << ","
                   << transA << "," << diag << "," << m << "," << n << "," << h_alpha << ","
                   << (void*)da << "," << lda << "," << (void*)db << "," << ldb << "," << (void*)dc
                   << "," << ldc << ",atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f trmm -r " << rocblas_precision_string<T> << " --side "
                   << side_letter << " --uplo " << uplo_letter << " --transposeA " << transA_letter
                   << " --diag " << diag_letter << " -m " << m << " -n " << n << " "
                   << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda " << lda
                   << " --ldb " << ldb << " --ldc " << ldc << bench_endl;
    } // !half

    //
    // GEMM
    //
    trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xgemm") << "," << transA << "," << transB << ","
               << m << "," << n << "," << k << "," << h_alpha << "," << (void*)da << "," << lda
               << "," << (void*)db << "," << ldb << "," << h_beta << "," << (void*)dc << "," << ldc
               << ",atomics_not_allowed\n";

    bench_ofs2 << bench.c_str() << " -f gemm -r " << rocblas_precision_string<T> << " --transposeA "
               << transA_letter << " --transposeB " << transB_letter << " -m " << m << " -n " << n
               << " -k " << k << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha)
               << " --lda " << lda << " --ldb " << ldb << " "
               << rocblas_internal_log_bench_scalar_value("beta", &h_beta) << " --ldc " << ldc
               << bench_endl;

    //
    // GEMM_STRIDED_BATCHED
    //
    trace_ofs2 << replaceX<T>(arg.api, "rocblas_Xgemm_strided_batched") << "," << transA << ","
               << transB << "," << m << "," << n << "," << k << "," << h_alpha << "," << (void*)da
               << "," << lda << "," << stride_a << "," << (void*)db << "," << ldb << "," << stride_b
               << "," << h_beta << "," << (void*)dc << "," << ldc << "," << stride_c << ","
               << batch_count << ",atomics_not_allowed\n";

    bench_ofs2 << bench.c_str() << " -f gemm_strided_batched -r "
               << rocblas_precision_string<T> << " --transposeA " << transA_letter
               << " --transposeB " << transB_letter << " -m " << m << " -n " << n << " -k " << k
               << " " << rocblas_internal_log_bench_scalar_value("alpha", &h_alpha) << " --lda "
               << lda << " --stride_a " << stride_a << " --ldb " << ldb << " --stride_b "
               << stride_b << " " << rocblas_internal_log_bench_scalar_value("beta", &h_beta)
               << " --ldc " << ldc << " --stride_c " << stride_c << " --batch_count " << batch_count
               << bench_endl;

    if constexpr(!rocblas_is_complex<T>)
    {
        rocblas_datatype a_type, b_type, c_type, d_type, compute_type;

        if(std::is_same_v<T, rocblas_half>)
        {
            a_type       = rocblas_datatype_f16_r;
            b_type       = rocblas_datatype_f16_r;
            c_type       = rocblas_datatype_f16_r;
            d_type       = rocblas_datatype_f16_r;
            compute_type = rocblas_datatype_f16_r;
        }
        else if(std::is_same_v<T, float>)
        {
            a_type       = rocblas_datatype_f32_r;
            b_type       = rocblas_datatype_f32_r;
            c_type       = rocblas_datatype_f32_r;
            d_type       = rocblas_datatype_f32_r;
            compute_type = rocblas_datatype_f32_r;
        }
        if(std::is_same_v<T, double>)
        {
            a_type       = rocblas_datatype_f64_r;
            b_type       = rocblas_datatype_f64_r;
            c_type       = rocblas_datatype_f64_r;
            d_type       = rocblas_datatype_f64_r;
            compute_type = rocblas_datatype_f64_r;
        }

        rocblas_gemm_algo algo           = rocblas_gemm_algo_standard;
        int32_t           solution_index = 0;
        uint32_t          flags          = 0;

        //
        //GEMM_EX
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_gemm_ex") << "," << transA << "," << transB
                   << "," << m << "," << n << "," << k << "," << alpha_float << "," << (void*)da
                   << "," << rocblas_datatype_string(a_type) << "," << lda << "," << (void*)db
                   << "," << rocblas_datatype_string(b_type) << "," << ldb << "," << beta_float
                   << "," << (void*)dc << "," << rocblas_datatype_string(c_type) << "," << ldc
                   << "," << (void*)dd << "," << rocblas_datatype_string(d_type) << "," << ldd
                   << "," << rocblas_datatype_string(compute_type) << "," << algo << ","
                   << solution_index << ",none,atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f gemm_ex"
                   << " --transposeA " << transA_letter << " --transposeB " << transB_letter
                   << " -m " << m << " -n " << n << " -k " << k << " --alpha " << alpha_float
                   << " --a_type " << rocblas_datatype_string(a_type) << " --lda " << lda
                   << " --b_type " << rocblas_datatype_string(b_type) << " --ldb " << ldb
                   << " --beta " << beta_float << " --c_type " << rocblas_datatype_string(c_type)
                   << " --ldc " << ldc << " --d_type " << rocblas_datatype_string(d_type)
                   << " --ldd " << ldd << " --compute_type "
                   << rocblas_datatype_string(compute_type) << " --algo " << algo
                   << " --solution_index " << solution_index << " --flags " << flags << bench_endl;

        //
        //GEMM_STRIDED_BATCHED_EX
        //
        trace_ofs2 << replaceX<T>(arg.api, "rocblas_gemm_strided_batched_ex") << "," << transA
                   << "," << transB << "," << m << "," << n << "," << k << "," << alpha_float << ","
                   << (void*)da << "," << rocblas_datatype_string(a_type) << "," << lda << ","
                   << stride_a << "," << (void*)db << "," << rocblas_datatype_string(b_type) << ","
                   << ldb << "," << stride_b << "," << beta_float << "," << (void*)dc << ","
                   << rocblas_datatype_string(c_type) << "," << ldc << "," << stride_c << ","
                   << (void*)dd << "," << rocblas_datatype_string(d_type) << "," << ldd << ","
                   << stride_d << "," << batch_count << "," << rocblas_datatype_string(compute_type)
                   << "," << algo << "," << solution_index << ",none,atomics_not_allowed\n";

        bench_ofs2 << bench.c_str() << " -f gemm_strided_batched_ex"
                   << " --transposeA " << transA_letter << " --transposeB " << transB_letter
                   << " -m " << m << " -n " << n << " -k " << k << " --alpha " << alpha_float
                   << " --a_type " << rocblas_datatype_string(a_type) << " --lda " << lda
                   << " --stride_a " << stride_a << " --b_type " << rocblas_datatype_string(b_type)
                   << " --ldb " << ldb << " --stride_b " << stride_b << " --beta " << beta_float
                   << " --c_type " << rocblas_datatype_string(c_type) << " --ldc " << ldc
                   << " --stride_c " << stride_c << " --d_type " << rocblas_datatype_string(d_type)
                   << " --ldd " << ldd << " --stride_d " << stride_d << " --batch_count "
                   << batch_count << " --compute_type " << rocblas_datatype_string(compute_type)
                   << " --algo " << algo << " --solution_index " << solution_index << " --flags "
                   << flags << bench_endl;

    } // !complex

    // Auxiliary function
    trace_ofs2 << "rocblas_destroy_handle,atomics_not_allowed\n";

#ifdef BUILD_WITH_HIPBLASLT
    if(arg.algo == 1)
    {
        if constexpr(!rocblas_is_complex<T>)
        {
            //
            // GEMM internal logging only (no bench or profile)
            //
            trace_ofs2 << "rocblas_internal,rocblas_gemm_hipblaslt_backend"
                       << "," << transA << "," << transB << "," << m << "," << n << "," << k << ","
                       << h_alpha << "," << (void*)da << "," << lda << "," << (void*)db << ","
                       << ldb << "," << h_beta << "," << (void*)dc << "," << ldc << "," << (void*)dc
                       << "," << ldc << ",atomics_not_allowed\n"; // logs d info as also for gemm_ex
        } // !complex
    }
#endif // BUILD_WITH_HIPBLASLT

    // excluded trtri as it is an internal function
    //  trace_ofs2 << "\n" << replaceX<T>("rocblas_Xtrtri")  << "," << uplo << "," << diag << "," <<
    //  n
    //  << "," << (void*)da << "," << lda << "," << (void*)db << "," << ldb;

    // Flush the streams
    trace_ofs2.flush();
    bench_ofs2.flush();

    // Transfer the formatted output to the files
    trace_ofs << trace_ofs2;
    bench_ofs << bench_ofs2;

    // Flush the streams
    trace_ofs.flush();
    bench_ofs.flush();

    // Close the files
    trace_ofs.close();
    bench_ofs.close();

//
// check if rocBLAS output files same as "golden files"
//
#ifdef WIN32
    // need all file descriptors closed to allow file removal on windows before process exits
    rocblas_internal_ostream::clear_workers();

    int trace_cmp = diff_files(trace_path1, trace_path2);
//int trace_cmp = system(("fc.exe \"" + trace_path1 + "\" \"" + trace_path2 + "\" | findstr *****").c_str());
#else
    int trace_cmp = system(("/usr/bin/diff " + trace_path1 + " " + trace_path2).c_str());
#endif
    ASSERT_EQ(trace_cmp, 0);

    if(!trace_cmp)
    {
        fs::remove(trace_fspath1);
        fs::remove(trace_fspath2);
    }

#ifdef WIN32
    int bench_cmp = diff_files(bench_path1, bench_path2);
//int bench_cmp = system(("fc.exe \"" + bench_path1 + "\" \"" + bench_path2 + "\" | findstr *****").c_str());
#else
    int bench_cmp = system(("/usr/bin/diff " + bench_path1 + " " + bench_path2).c_str());
#endif
    ASSERT_EQ(bench_cmp, 0);

    if(!bench_cmp)
    {
        fs::remove(bench_fspath1);
        fs::remove(bench_fspath2);
    }
}
