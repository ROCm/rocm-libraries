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
void testing_csrmv_bad_arg(const Arguments& arg)
{
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
void testing_csrmv(const Arguments& arg)
{
    auto                  tol                 = get_near_check_tol<T>(arg);
    rocsparse_int         M                   = arg.M;
    rocsparse_int         N                   = arg.N;
    rocsparse_operation   trans               = arg.transA;
    rocsparse_index_base  base                = arg.baseA;
    rocsparse_matrix_type matrix_type         = arg.matrix_type;
    rocsparse_fill_mode   uplo                = arg.uplo;
    rocsparse_spmv_alg    alg                 = arg.spmv_alg;
    const bool            call_stage_analysis = arg.call_stage_analysis;

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    device_scalar<T> d_alpha(h_alpha);
    device_scalar<T> d_beta(h_beta);

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Create matrix info
    rocsparse_local_mat_info info_ptr;

    rocsparse_mat_info info = (call_stage_analysis)
                                  ? ((alg == rocsparse_spmv_alg_csr_adaptive) ? info_ptr : nullptr)
                                  : nullptr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Set matrix type
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));

    // Set fill mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_fill_mode(descr, uplo));

#define PARAMS_ANALYSIS(A_) handle, trans, A_.m, A_.n, A_.nnz, descr, A_.val, A_.ptr, A_.ind, info
#define PARAMS(alpha_, A_, x_, beta_, y_) \
    handle, trans, A_.m, A_.n, A_.nnz, alpha_, descr, A_.val, A_.ptr, A_.ind, info, x_, beta_, y_

    // Allocate host memory for matrix
    // Wavefront size
    int dev;
    CHECK_HIP_ERROR(hipGetDevice(&dev));

    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));
    const bool has_datafile = rocsparse_arguments_has_datafile(arg);

    bool to_int = false;
    to_int |= (prop.warpSize == 32);
    to_int |= (alg != rocsparse_spmv_alg_csr_rowsplit);
    to_int |= (trans != rocsparse_operation_none && has_datafile);
    to_int |= (matrix_type == rocsparse_matrix_type_symmetric && has_datafile);

    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, arg.unit_check ? to_int : false, full_rank);

    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA, M, N);

    // normalize
    rocsparse_vector_utils<T>::normalize(hA.val);

    if((matrix_type == rocsparse_matrix_type_symmetric && M != N)
       || (matrix_type == rocsparse_matrix_type_triangular && M != N))
    {
        return;
    }
    device_csr_matrix<T> dA(hA);

    host_dense_matrix<T> hx(trans == rocsparse_operation_none ? N : M, 1);
    rocsparse_matrix_utils::init_exact(hx);
    device_dense_matrix<T> dx(hx);

    host_dense_matrix<T> hy(trans == rocsparse_operation_none ? M : N, 1);
    rocsparse_matrix_utils::init_exact(hy);
    device_dense_matrix<T> dy(hy);

    // If adaptive, run analysis step
    if(call_stage_analysis)
    {
        if(alg == rocsparse_spmv_alg_csr_adaptive)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_analysis<T>(PARAMS_ANALYSIS(dA)));
        }
    }

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode host", dy);
        }

        host_dense_matrix<T> hy_copy(hy);
        host_csrmv<T, rocsparse_int, rocsparse_int, T, T, T>(trans,
                                                             M,
                                                             N,
                                                             hA.nnz,
                                                             *h_alpha,
                                                             hA.ptr,
                                                             hA.ind,
                                                             hA.val,
                                                             hx,
                                                             *h_beta,
                                                             hy,
                                                             base,
                                                             matrix_type,
                                                             alg,
                                                             false);

        hy.near_check(dy, tol);
        dy = hy_copy;

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csrmv<T>(PARAMS(d_alpha, dA, dx, d_beta, dy)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode device", dy);
        }

        hy.near_check(dy, tol);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_csrmv<T>, PARAMS(h_alpha, dA, dx, h_beta, dy));

        double gflop_count = spmv_gflop_count(M, dA.nnz, *h_beta != static_cast<T>(0));
        double gbyte_count = csrmv_gbyte_count<T>(M, N, dA.nnz, *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            dA.nnz,
                            display_key_t::alpha,
                            *h_alpha,
                            display_key_t::beta,
                            *h_beta,
                            display_key_t::algorithm,
                            ((alg == rocsparse_spmv_alg_csr_adaptive) ? "adaptive" : "stream"),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    if(info != nullptr)
    {
        CHECK_ROCSPARSE_ERROR(rocsparse_csrmv_clear(handle, info));
    }
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_csrmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csrmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csrmv_extra(const Arguments& arg) {}
