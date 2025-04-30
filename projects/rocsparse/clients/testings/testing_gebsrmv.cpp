/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "testing.hpp"

#include "rocsparse_enum.hpp"

template <typename T>
void testing_gebsrmv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 10;
    const T             h_alpha   = static_cast<T>(1);
    const T             h_beta    = static_cast<T>(1);
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    rocsparse_handle          handle        = local_handle;
    rocsparse_direction       dir           = rocsparse_direction_column;
    rocsparse_operation       trans         = rocsparse_operation_none;
    rocsparse_int             mb            = safe_size;
    rocsparse_int             nb            = safe_size;
    rocsparse_int             nnzb          = safe_size;
    const T*                  alpha         = (const T*)&h_alpha;
    const rocsparse_mat_descr descr         = local_descr;
    const T*                  bsr_val       = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr   = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind   = (const rocsparse_int*)0x4;
    rocsparse_int             row_block_dim = safe_size;
    rocsparse_int             col_block_dim = safe_size;
    const T*                  x             = (const T*)0x4;
    const T*                  beta          = (const T*)&h_beta;
    T*                        y             = (T*)0x4;

#define PARAMS                                                                         \
    handle, dir, trans, mb, nb, nnzb, alpha, descr, bsr_val, bsr_row_ptr, bsr_col_ind, \
        row_block_dim, col_block_dim, x, beta, y

    bad_arg_analysis(rocsparse_gebsrmv<T>, PARAMS);

    //
    // operation different from rocsparse_operation_none is not implemented.
    //
    for(auto operation : rocsparse_operation_t::values)
    {
        if(operation != rocsparse_operation_none)
        {
            {
                auto tmp = trans;
                trans    = operation;
                for(rocsparse_int i = 1; i <= 17; ++i)
                {
                    row_block_dim = i;
                    col_block_dim = i + 1;
                    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(PARAMS),
                                            rocsparse_status_not_implemented);
                }
                row_block_dim = safe_size;
                col_block_dim = safe_size;
                trans         = tmp;
            }
        }
    }

    //
    // Matrix types different from general.
    //
    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(PARAMS), rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

    // row_block_dim == 0
    row_block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(PARAMS), rocsparse_status_invalid_size);
    row_block_dim = safe_size;

    // col_block_dim == 0
    col_block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(PARAMS), rocsparse_status_invalid_size);
    col_block_dim = safe_size;

    // row_block_dim == 0 && col_block_dim == 0
    row_block_dim = 0;
    col_block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(PARAMS), rocsparse_status_invalid_size);
    row_block_dim = safe_size;
    col_block_dim = safe_size;
#undef PARAMS

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsrmv<T>(handle,
                                                 dir,
                                                 trans,
                                                 mb,
                                                 nb,
                                                 nnzb,
                                                 alpha,
                                                 descr,
                                                 nullptr,
                                                 bsr_row_ptr,
                                                 nullptr,
                                                 row_block_dim,
                                                 col_block_dim,
                                                 x,
                                                 beta,
                                                 y),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_gebsrmv(const Arguments& arg)
{
    auto                 tol   = get_near_check_tol<T>(arg);
    rocsparse_int        M     = arg.M;
    rocsparse_int        N     = arg.N;
    rocsparse_operation  trans = arg.transA;
    rocsparse_index_base base  = arg.baseA;

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

    // Wavefront size
    int dev;
    CHECK_HIP_ERROR(hipGetDevice(&dev));

    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

    host_gebsr_matrix<T> hA;

    {
        bool                  type = (prop.warpSize == 32) ? (arg.timing ? false : true) : false;
        static constexpr bool full_rank = false;
        rocsparse_matrix_factory<T> matrix_factory(arg, type, full_rank);

        matrix_factory.init_gebsr(hA);
    }

    M = hA.mb * hA.row_block_dim;
    N = hA.nb * hA.col_block_dim;

    host_dense_matrix<T> hx(N, 1), hy(M, 1);
    rocsparse_matrix_utils::init(hx);
    rocsparse_matrix_utils::init(hy);

    device_gebsr_matrix<T> dA(hA);
    device_dense_matrix<T> dx(hx), dy(hy);

#define PARAMS(alpha_, A_, x_, beta_, y_)                                                    \
    handle, A_.block_direction, trans, A_.mb, A_.nb, A_.nnzb, alpha_, descr, A_.val, A_.ptr, \
        A_.ind, A_.row_block_dim, A_.col_block_dim, x_, beta_, y_

    if(arg.unit_check)
    {
        // Navi21 on Windows requires weaker tolerance due to different rounding behaviour
#if defined(WIN32)
        int dev;
        CHECK_HIP_ERROR(hipGetDevice(&dev));

        hipDeviceProp_t prop;
        CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

        if(prop.major == 10 && prop.minor == 3)
        {
            tol *= 1e2;
        }
#endif

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_gebsrmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode host", dy);
        }

        {
            host_dense_matrix<T> hy_copy(hy);
            // CPU gebsrmv
            host_gebsrmv<T>(hA.block_direction,
                            trans,
                            hA.mb,
                            hA.nb,
                            hA.nnzb,
                            *h_alpha,
                            hA.ptr,
                            hA.ind,
                            hA.val,
                            hA.row_block_dim,
                            hA.col_block_dim,
                            hx,
                            *h_beta,
                            hy,
                            base);
            hy.near_check(dy, tol);
            dy = hy_copy;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_gebsrmv<T>(PARAMS(d_alpha, dA, dx, d_beta, dy)));
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
            arg, rocsparse_gebsrmv<T>, PARAMS(h_alpha, dA, dx, h_beta, dy));

        double gflop_count = spmv_gflop_count(
            M, dA.nnzb * dA.row_block_dim * dA.col_block_dim, *h_beta != static_cast<T>(0));
        double gbyte_count = gebsrmv_gbyte_count<T>(dA.mb,
                                                    dA.nb,
                                                    dA.nnzb,
                                                    dA.row_block_dim,
                                                    dA.col_block_dim,
                                                    *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnzb,
                            dA.nnzb,
                            display_key_t::rbdim,
                            dA.row_block_dim,
                            display_key_t::cbdim,
                            dA.col_block_dim,
                            display_key_t::bdir,
                            rocsparse_direction2string(dA.block_direction),
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

#define INSTANTIATE(TYPE)                                              \
    template void testing_gebsrmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gebsrmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_gebsrmv_extra(const Arguments& arg) {}
