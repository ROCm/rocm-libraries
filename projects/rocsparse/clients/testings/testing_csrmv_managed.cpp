/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_csrmv_managed_bad_arg(const Arguments& arg)
{
    // check managed memory enablement
    if(!is_hmm_enabled())
    {
        std::puts("Managed memory not enabled on device. Skipping test...");
        std::fflush(stdout);
        return;
    }

    static const size_t safe_size = 100;

    const T h_alpha = static_cast<T>(1);
    const T h_beta  = static_cast<T>(1);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    // Create matrix info
    rocsparse_local_mat_info local_info;

    rocsparse_handle          handle            = local_handle;
    rocsparse_operation       trans             = rocsparse_operation_none;
    rocsparse_int             m                 = safe_size;
    rocsparse_int             n                 = safe_size;
    rocsparse_int             nnz               = safe_size;
    const T*                  alpha_device_host = &h_alpha;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  csr_val           = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr       = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind       = (const rocsparse_int*)0x4;
    rocsparse_mat_info        info              = local_info;
    const T*                  x                 = (const T*)0x4;
    const T*                  beta_device_host  = &h_beta;
    T*                        y                 = (T*)0x4;

#define PARAMS_ANALYSIS handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info
    bad_arg_analysis(rocsparse_csrmv_analysis<T>, PARAMS_ANALYSIS);

#define PARAMS                                                                                   \
    handle, trans, m, n, nnz, alpha_device_host, descr, csr_val, csr_row_ptr, csr_col_ind, info, \
        x, beta_device_host, y

    {
        static constexpr int num_exclusions  = 1;
        static constexpr int exclude_args[1] = {10};
        select_bad_arg_analysis(rocsparse_csrmv<T>, num_exclusions, exclude_args, PARAMS);
    }

    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_clear(nullptr, info), rocsparse_status_invalid_handle);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_clear(handle, nullptr),
                            rocsparse_status_invalid_pointer);

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general
           && matrix_type != rocsparse_matrix_type_symmetric
           && matrix_type != rocsparse_matrix_type_triangular)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(PARAMS_ANALYSIS),
                                    rocsparse_status_not_implemented);
            EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv_analysis<T>(PARAMS_ANALYSIS),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csrmv<T>(PARAMS), rocsparse_status_requires_sorted_storage);

#undef PARAMS_ANALYSIS
#undef PARAMS
}

template <typename T>
void testing_csrmv_managed(const Arguments& arg)
{
    // check managed memory enablement
    if(!is_hmm_enabled())
    {
        std::puts("Managed memory not enabled on device. Skipping test...");
        std::fflush(stdout);
        return;
    }

    rocsparse_int        M     = arg.M;
    rocsparse_int        N     = arg.N;
    rocsparse_operation  trans = arg.transA;
    rocsparse_index_base base  = arg.baseA;
    rocsparse_spmv_alg   alg   = arg.spmv_alg;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info_ptr;

    rocsparse_mat_info info = (alg == rocsparse_spmv_alg_csr_adaptive) ? info_ptr : nullptr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Wavefront size
    int dev;
    CHECK_HIP_ERROR(hipGetDevice(&dev));

    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

    bool to_int = false;
    to_int |= (prop.warpSize == 32);
    to_int |= (alg != rocsparse_spmv_alg_csr_rowsplit);

    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, arg.unit_check ? to_int : false, full_rank);

    // Generate matrix
    host_vector<rocsparse_int> trow_ptr;
    host_vector<rocsparse_int> tcol_ind;
    host_vector<T>             tval;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(trow_ptr, tcol_ind, tval, M, N, nnz, base);

    // Allocate host memory for vectors
    host_vector<T> tx(N);
    host_vector<T> ty(M);

    // Initialize data on CPU
    rocsparse_init<T>(tx, 1, N, 1, arg.convert_to_int);
    rocsparse_init<T>(ty, 1, M, 1, arg.convert_to_int);

    // Allocate managed memory
    managed_dense_vector<rocsparse_int> csr_row_ptr(M + 1);
    managed_dense_vector<rocsparse_int> csr_col_ind(nnz);
    managed_dense_vector<T>             csr_val(nnz);
    managed_dense_vector<T>             x(N);
    managed_dense_vector<T>             y_1(M);
    managed_dense_vector<T>             y_2(M);
    managed_scalar<T>                   alpha;
    managed_scalar<T>                   beta;

    // Copy data to managed arrays
    for(rocsparse_int i = 0; i < M + 1; i++)
    {
        csr_row_ptr[i] = trow_ptr[i];
    }

    for(rocsparse_int i = 0; i < nnz; i++)
    {
        csr_col_ind[i] = tcol_ind[i];
        csr_val[i]     = tval[i];
    }

    for(rocsparse_int i = 0; i < N; i++)
    {
        x[i] = tx[i];
    }

    for(rocsparse_int i = 0; i < M; i++)
    {
        y_1[i] = ty[i];
        y_2[i] = ty[i];
    }

    *alpha = arg.get_alpha<T>();
    *beta  = arg.get_beta<T>();

    // If adaptive, run analysis step
    if(alg == rocsparse_spmv_alg_csr_adaptive)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_analysis<T>(
            handle, trans, M, N, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, info));
    }
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrmv<T>(handle,
                                                          trans,
                                                          M,
                                                          N,
                                                          nnz,
                                                          alpha,
                                                          descr,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          info,
                                                          x,
                                                          beta,
                                                          y_1));

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrmv<T>(handle,
                                                          trans,
                                                          M,
                                                          N,
                                                          nnz,
                                                          alpha,
                                                          descr,
                                                          csr_val,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          info,
                                                          x,
                                                          beta,
                                                          y_2));

        CHECK_HIP_ERROR(hipDeviceSynchronize());
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save(
                "Y pointer mode host", y_1, "Y pointer mode device", y_2);
        }

        // CPU y
        host_vector<T> y_gold(M);
        for(rocsparse_int i = 0; i < M; i++)
        {
            y_gold[i] = ty[i];
        }

        // CPU csrmv
        host_csrmv(trans,
                   M,
                   N,
                   nnz,
                   *alpha,
                   trow_ptr.data(),
                   tcol_ind.data(),
                   tval.data(),
                   tx.data(),
                   *beta,
                   y_gold.data(),
                   base,
                   rocsparse_matrix_type_general,
                   alg,
                   false);

        near_check_segments<T>(M, &y_gold[0], y_1);
        near_check_segments<T>(M, &y_gold[0], y_2);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csrmv<T>,
                                                                      handle,
                                                                      trans,
                                                                      M,
                                                                      N,
                                                                      nnz,
                                                                      alpha,
                                                                      descr,
                                                                      csr_val,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      info,
                                                                      x,
                                                                      beta,
                                                                      y_1);

        double gflop_count = spmv_gflop_count(M, nnz, *beta != static_cast<T>(0));
        double gbyte_count = csrmv_gbyte_count<T>(M, N, nnz, *beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::alpha,
                            *alpha,
                            display_key_t::beta,
                            *beta,
                            display_key_t::algorithm,
                            ((alg == rocsparse_spmv_alg_csr_adaptive) ? "adaptive" : "stream"),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    // If adaptive, clear analysis data
    if(alg == rocsparse_spmv_alg_csr_adaptive)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
    }
}

#define INSTANTIATE(TYPE)                                                    \
    template void testing_csrmv_managed_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrmv_managed<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrmv_managed_extra(const Arguments& arg) {}
