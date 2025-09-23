/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_bsrxmv_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 10;
    const T             h_alpha   = static_cast<T>(1);
    const T             h_beta    = static_cast<T>(1);
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    rocsparse_handle          handle            = local_handle;
    rocsparse_direction       dir               = rocsparse_direction_column;
    rocsparse_operation       trans             = rocsparse_operation_none;
    rocsparse_int             mb                = safe_size;
    rocsparse_int             nb                = safe_size;
    rocsparse_int             nnzb              = safe_size;
    const T*                  alpha_device_host = (const T*)&h_alpha;
    const rocsparse_mat_descr descr             = local_descr;
    const T*                  bsr_val           = (const T*)0x4;
    rocsparse_int             size_of_mask      = safe_size;
    const rocsparse_int*      bsr_mask_ptr      = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_row_ptr       = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_end_ptr       = (const rocsparse_int*)0x4;
    const rocsparse_int*      bsr_col_ind       = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim         = safe_size;
    const T*                  x                 = (const T*)0x4;
    const T*                  beta_device_host  = (const T*)&h_beta;
    T*                        y                 = (T*)0x4;

#define PARAMS                                                                         \
    handle, dir, trans, size_of_mask, mb, nb, nnzb, alpha_device_host, descr, bsr_val, \
        bsr_mask_ptr, bsr_row_ptr, bsr_end_ptr, bsr_col_ind, block_dim, x, beta_device_host, y

    bad_arg_analysis(rocsparse_bsrxmv<T>, PARAMS);

    {
        auto tmp = trans;
        trans    = rocsparse_operation_transpose;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrxmv<T>(PARAMS), rocsparse_status_not_implemented);
        trans = tmp;
    }

    {
        auto tmp  = block_dim;
        block_dim = 1;
        EXPECT_ROCSPARSE_STATUS(rocsparse_bsrxmv<T>(PARAMS), rocsparse_status_not_implemented);
        block_dim = tmp;
    }

    for(auto matrix_type : rocsparse_matrix_type_t::values)
    {
        if(matrix_type != rocsparse_matrix_type_general)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, matrix_type));
            EXPECT_ROCSPARSE_STATUS(rocsparse_bsrxmv<T>(PARAMS), rocsparse_status_not_implemented);
        }
    }
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_type(descr, rocsparse_matrix_type_general));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrxmv<T>(PARAMS), rocsparse_status_requires_sorted_storage);
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_sorted));

    // block_dim == 0
    block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrxmv<T>(PARAMS), rocsparse_status_invalid_size);
    block_dim = safe_size;
#undef PARAMS

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrxmv<T>(handle,
                                                dir,
                                                trans,
                                                size_of_mask,
                                                mb,
                                                nb,
                                                nnzb,
                                                alpha_device_host,
                                                descr,
                                                nullptr,
                                                bsr_mask_ptr,
                                                bsr_row_ptr,
                                                bsr_end_ptr,
                                                nullptr,
                                                block_dim,
                                                x,
                                                beta_device_host,
                                                y),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_bsrxmv(const Arguments& arg)
{

    rocsparse_int        M            = arg.M;
    rocsparse_int        N            = arg.N;
    rocsparse_direction  dir          = arg.direction;
    rocsparse_operation  trans        = arg.transA;
    rocsparse_index_base base         = arg.baseA;
    rocsparse_int        block_dim    = arg.block_dim;
    rocsparse_int        size_of_mask = 0;

    rocsparse_seedrand();
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

    // BSR dimensions
    rocsparse_int mb = (block_dim > 0) ? (M + block_dim - 1) / block_dim : 0;
    rocsparse_int nb = (block_dim > 0) ? (N + block_dim - 1) / block_dim : 0;

#define PARAMS(alpha_, A_, x_, beta_, y_)                                                          \
    handle, A_.block_direction, trans, size_of_mask, A_.mb, A_.nb, A_.nnzb, alpha_, descr, A_.val, \
        dbsr_mask_ptr, dbsr_row_ptr, dbsr_end_ptr, A_.ind, A_.row_block_dim, x_, beta_, y_

    size_of_mask = (mb > 0) ? random_generator<rocsparse_int>(0, mb - 1) : 0;

    // Wavefront size
    int dev;
    CHECK_HIP_ERROR(hipGetDevice(&dev));

    hipDeviceProp_t prop;
    CHECK_HIP_ERROR(hipGetDeviceProperties(&prop, dev));

    bool                  type      = (prop.warpSize == 32) ? (arg.timing ? false : true) : false;
    static constexpr bool full_rank = false;

    rocsparse_matrix_factory<T> matrix_factory(arg, type, full_rank);

    //
    // Declare and initialize matrices.
    //
    host_gebsr_matrix<T>   hA;
    device_gebsr_matrix<T> dA;
    matrix_factory.init_bsr(hA, dA, mb, nb, base);

    M = dA.mb * dA.row_block_dim;
    N = dA.nb * dA.col_block_dim;

    //
    // Initialization of the row_ptr and end_ptr.
    //
    host_dense_vector<rocsparse_int> hbsr_row_ptr(mb);
    host_dense_vector<rocsparse_int> hbsr_end_ptr(mb);

    for(rocsparse_int i = 0; i < mb; ++i)
    {
        hbsr_row_ptr[i] = (hA.ptr[i + 1] > hA.ptr[i])
                              ? random_generator<rocsparse_int>(hA.ptr[i], hA.ptr[i + 1] - 1)
                              : hA.ptr[i];
    }

    for(rocsparse_int i = 0; i < mb; ++i)
    {
        hbsr_end_ptr[i] = random_generator<rocsparse_int>(hbsr_row_ptr[i], hA.ptr[i + 1]);
    }

    //
    // Initialization of the mask.
    //
    host_dense_vector<rocsparse_int> hbsr_mask_ptr(size_of_mask);

    {
        //
        // Unique random integer values.
        //
        host_dense_vector<rocsparse_int> marker(mb);
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            marker[i] = 0;
        }
        rocsparse_int count = 0;
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            if(count == size_of_mask)
            {
                break;
            }

            marker[i] = random_generator<rocsparse_int>(0, 1);
            if(marker[i] > 0)
            {
                ++count;
            }
        }

        if(count < size_of_mask)
        {
            //
            // If size_of_mask is mb or big enough, we don't want
            // the random generator taking an eternity to fill the vector.
            // So let's fill.
            //
            for(rocsparse_int i = 0; i < mb; ++i)
            {
                if(marker[i] == 0)
                {
                    marker[i] = 1;
                    ++count;
                }
                if(size_of_mask == count)
                {
                    break;
                }
            }
        }

        count = 0;
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            if(marker[i] == 1)
            {
                hbsr_mask_ptr[count++] = i + hA.base;
            }
        }
    }

    device_dense_vector<rocsparse_int> dbsr_mask_ptr(hbsr_mask_ptr);
    device_dense_vector<rocsparse_int> dbsr_row_ptr(hbsr_row_ptr);
    device_dense_vector<rocsparse_int> dbsr_end_ptr(hbsr_end_ptr);

    host_dense_matrix<T> hx(N, 1), hy(M, 1);

    rocsparse_matrix_utils::init(hx);
    rocsparse_matrix_utils::init(hy);

    device_dense_matrix<T> dx(hx), dy(hy);

    if(arg.unit_check)
    {

        // Pointer mode host
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrxmv<T>(PARAMS(h_alpha, dA, dx, h_beta, dy)));
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("Y pointer mode host", dy);
        }

        {
            host_dense_matrix<T> hy_copy(hy);
            // CPU bsrxmv
            host_bsrxmv<T>(dir,
                           trans,
                           size_of_mask,
                           hA.mb,
                           hA.nb,
                           hA.nnzb,
                           *h_alpha,
                           hbsr_mask_ptr,
                           hbsr_row_ptr,
                           hbsr_end_ptr,
                           hA.ind,
                           hA.val,
                           hA.row_block_dim,
                           hx,
                           *h_beta,
                           hy,
                           base);

            hy.near_check(dy);
            dy = hy_copy;
        }

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrxmv<T>(PARAMS(d_alpha, dA, dx, d_beta, dy)));
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
            arg, rocsparse_bsrxmv<T>, PARAMS(h_alpha, dA, dx, h_beta, dy));

        //
        // Re-use bsrmv gflop and gbyte counts but with different parameters
        // mb switched with size_of_mask
        // nnzb switched with updated value.
        //
        rocsparse_int xnnzb = 0;
        for(rocsparse_int i = 0; i < mb; ++i)
        {
            xnnzb += (hbsr_end_ptr[i] - hbsr_row_ptr[i]);
        }
        double gflop_count = spmv_gflop_count(
            M, size_t(xnnzb) * dA.row_block_dim * dA.col_block_dim, *h_beta != static_cast<T>(0));
        double gbyte_count = bsrmv_gbyte_count<T>(
            size_of_mask, dA.nb, xnnzb, dA.row_block_dim, *h_beta != static_cast<T>(0));

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::bdim,
                            dA.row_block_dim,
                            display_key_t::bdir,
                            rocsparse_direction2string(dA.block_direction),
                            display_key_t::mask_size,
                            size_of_mask,
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

#define INSTANTIATE(TYPE)                                             \
    template void testing_bsrxmv_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrxmv<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
#undef INSTANTIATE
void testing_bsrxmv_extra(const Arguments& arg) {}
