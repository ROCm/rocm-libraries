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
void testing_coomv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    const T h_alpha = static_cast<T>(1);
    const T h_beta  = static_cast<T>(1);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    rocsparse_handle          handle            = local_handle;
    rocsparse_operation       trans             = rocsparse_operation_none;
    rocsparse_int             m                 = safe_size;
    rocsparse_int             n                 = safe_size;
    rocsparse_int             nnz               = safe_size;
    const T*                  alpha_device_host = &h_alpha;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  coo_val           = (const T*)0x4;
    const rocsparse_int*      coo_row_ind       = (const rocsparse_int*)0x4;
    const rocsparse_int*      coo_col_ind       = (const rocsparse_int*)0x4;
    const T*                  x                 = (const T*)0x4;
    const T*                  beta_device_host  = &h_beta;
    T*                        y                 = (T*)0x4;

#define PARAMS                                                                                \
    handle, trans, m, n, nnz, alpha_device_host, descr, coo_val, coo_row_ind, coo_col_ind, x, \
        beta_device_host, y

    bad_arg_analysis(rocsparse_coomv<T>, PARAMS);

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_coomv<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }

#undef PARAMS
}

template <typename T>
void testing_coomv(const Arguments& arg)
{
    rocsparse_int          M           = arg.M;
    rocsparse_int          N           = arg.N;
    rocsparse_operation    trans       = arg.transA;
    rocsparse_index_base   base        = arg.baseA;
    rocsparse_matrix_type  matrix_type = arg.matrix_type;
    rocsparse_storage_mode storage     = arg.storage;

    host_scalar<T> h_alpha(arg.get_alpha<T>());
    host_scalar<T> h_beta(arg.get_beta<T>());

    device_scalar<T> d_alpha(h_alpha);
    device_scalar<T> d_beta(h_beta);

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Set matrix type
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));

    // Set storage mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, storage));

#define PARAMS(alpha_, A_, x_, beta_, y_) \
    handle, trans, A_.m, A_.n, A_.nnz, alpha_, descr, A_.val, A_.row_ind, A_.col_ind, x_, beta_, y_

    rocsparse_matrix_factory<T> matrix_factory(arg, arg.timing ? false : true, false);

    host_coo_matrix<T> hA;

    matrix_factory.init_coo(hA, M, N);

    host_dense_matrix<T> hx((trans == rocsparse_operation_none) ? N : M, 1);
    host_dense_matrix<T> hy((trans == rocsparse_operation_none) ? M : N, 1);

    rocsparse_matrix_utils::init(hx);
    rocsparse_matrix_utils::init(hy);

    device_coo_matrix<T>   dA(hA);
    device_dense_matrix<T> dx(hx), dy(hy);

    if(arg.unit_check)
    {
        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_coomv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode host", dy);
        }

        {
            host_dense_matrix<T> hy_copy(hy);
            // CPU coomv
            host_coomv<T, rocsparse_int, T, T, T>(trans,
                                                  hA.m,
                                                  hA.n,
                                                  hA.nnz,
                                                  *h_alpha,
                                                  hA.row_ind,
                                                  hA.col_ind,
                                                  hA.val,
                                                  hx,
                                                  *h_beta,
                                                  hy,
                                                  hA.base);
            hy.near_check(dy);
            dy = hy_copy;
        }

        // Pointer mode device
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_coomv<T>(PARAMS(d_alpha, dA, dx, d_beta, dy)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode device", dy);
        }
        hy.near_check(dy);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_coomv<T>, PARAMS(h_alpha, dA, dx, h_beta, dy));

        double gflop_count = spmv_gflop_count(M, dA.nnz, *h_beta != static_cast<T>(0));
        double gbyte_count = coomv_gbyte_count<T>(M, N, dA.nnz, *h_beta != static_cast<T>(0));

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
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
#undef PARAMS
}

#define INSTANTIATE(TYPE)                                            \
    template void testing_coomv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_coomv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
#undef INSTANTIATE
void testing_coomv_extra(const Arguments& arg) {}
