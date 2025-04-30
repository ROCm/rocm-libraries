/*! \file */
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

#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_gebsr2gebsc_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle        = local_handle;
    rocsparse_int        mb            = safe_size;
    rocsparse_int        nb            = safe_size;
    rocsparse_int        nnzb          = safe_size;
    const T*             bsr_val       = (const T*)0x4;
    const rocsparse_int* bsr_row_ptr   = (const rocsparse_int*)0x4;
    const rocsparse_int* bsr_col_ind   = (const rocsparse_int*)0x4;
    rocsparse_int        row_block_dim = safe_size;
    rocsparse_int        col_block_dim = safe_size;
    T*                   bsc_val       = (T*)0x4;
    rocsparse_int*       bsc_row_ind   = (rocsparse_int*)0x4;
    rocsparse_int*       bsc_col_ptr   = (rocsparse_int*)0x4;
    rocsparse_action     copy_values   = rocsparse_action_numeric;
    rocsparse_index_base idx_base      = rocsparse_index_base_zero;
    size_t*              p_buffer_size = (size_t*)0x4;
    void*                temp_buffer   = (void*)0x4;

#define PARAMS_BUFFER_SIZE                                                                 \
    handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, \
        p_buffer_size
#define PARAMS                                                                             \
    handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, \
        bsc_val, bsc_row_ind, bsc_col_ptr, copy_values, idx_base, temp_buffer
    bad_arg_analysis(rocsparse_gebsr2gebsc_buffer_size<T>, PARAMS_BUFFER_SIZE);
    bad_arg_analysis(rocsparse_gebsr2gebsc<T>, PARAMS);

    // Check row_block_dim == 0
    row_block_dim = 0;
    col_block_dim = safe_size;
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc<T>(PARAMS), rocsparse_status_invalid_size);

    // Check col_block_dim == 0
    row_block_dim = safe_size;
    col_block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc<T>(PARAMS), rocsparse_status_invalid_size);

#undef PARAMS
#undef PARAMS_BUFFER_SIZE

    // Additional tests for invalid zero matrices
    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc_buffer_size<T>(handle,
                                                                 safe_size,
                                                                 safe_size,
                                                                 safe_size,
                                                                 nullptr,
                                                                 bsr_row_ptr,
                                                                 nullptr,
                                                                 safe_size,
                                                                 safe_size,
                                                                 p_buffer_size),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc<T>(handle,
                                                     safe_size,
                                                     safe_size,
                                                     safe_size,
                                                     nullptr,
                                                     bsr_row_ptr,
                                                     nullptr,
                                                     safe_size,
                                                     safe_size,
                                                     bsc_val,
                                                     bsc_row_ind,
                                                     bsc_col_ptr,
                                                     rocsparse_action_numeric,
                                                     rocsparse_index_base_zero,
                                                     temp_buffer),
                            rocsparse_status_invalid_pointer);

    EXPECT_ROCSPARSE_STATUS(rocsparse_gebsr2gebsc<T>(handle,
                                                     safe_size,
                                                     safe_size,
                                                     safe_size,
                                                     bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     safe_size,
                                                     safe_size,
                                                     nullptr,
                                                     nullptr,
                                                     bsc_col_ptr,
                                                     rocsparse_action_numeric,
                                                     rocsparse_index_base_zero,
                                                     temp_buffer),
                            rocsparse_status_invalid_pointer);
}

template <typename T>
void testing_gebsr2gebsc(const Arguments& arg)
{
    rocsparse_action action = arg.action;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    //
    // Declare the factory.
    //
    rocsparse_matrix_factory<T> factory(arg);

    //
    // Initialize the matrix.
    //
    host_gebsr_matrix<T> hbsr;
    factory.init_gebsr(hbsr);

    //
    // Allocate and transfer to device.
    //
    device_gebsr_matrix<T> dbsr(hbsr);

    //
    // Obtain required buffer size (from host)
    //
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(testing::rocsparse_gebsr2gebsc_buffer_size<T>(handle,
                                                                        dbsr.mb,
                                                                        dbsr.nb,
                                                                        dbsr.nnzb,
                                                                        dbsr.val,
                                                                        dbsr.ptr,
                                                                        dbsr.ind,
                                                                        dbsr.row_block_dim,
                                                                        dbsr.col_block_dim,
                                                                        &buffer_size));

    //
    // Allocate the buffer size.
    //
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    //
    // Allocate device bsc matrix.
    //
    device_gebsc_matrix<T> dbsc(dbsr.block_direction,
                                dbsr.mb,
                                dbsr.nb,
                                dbsr.nnzb,
                                dbsr.row_block_dim,
                                dbsr.col_block_dim,
                                dbsr.base);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_gebsr2gebsc<T>(handle,
                                                                dbsr.mb,
                                                                dbsr.nb,
                                                                dbsr.nnzb,
                                                                dbsr.val,
                                                                dbsr.ptr,
                                                                dbsr.ind,
                                                                dbsr.row_block_dim,
                                                                dbsr.col_block_dim,
                                                                dbsc.val,
                                                                dbsc.ind,
                                                                dbsc.ptr,
                                                                action,
                                                                dbsr.base,
                                                                dbuffer));
        //
        // Transfer to host.
        //
        host_gebsc_matrix<T> hbsc_from_device(dbsc);

        //
        // Allocate host bsc matrix.
        //
        host_gebsc_matrix<T> hbsc(hbsr.block_direction,
                                  hbsr.mb,
                                  hbsr.nb,
                                  hbsr.nnzb,
                                  hbsr.row_block_dim,
                                  hbsr.col_block_dim,
                                  hbsr.base);

        //
        // Now the results need to be validated with 2 steps:
        //
        host_gebsr_to_gebsc<T>(hbsr.mb,
                               hbsr.nb,
                               hbsr.nnzb,
                               hbsr.ptr,
                               hbsr.ind,
                               hbsr.val,
                               hbsr.row_block_dim,
                               hbsr.col_block_dim,
                               hbsc.ind,
                               hbsc.ptr,
                               hbsc.val,
                               action,
                               hbsr.base);

        hbsc.unit_check(hbsc_from_device, action == rocsparse_action_numeric);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_gebsr2gebsc<T>,
                                                                      handle,
                                                                      dbsr.mb,
                                                                      dbsr.nb,
                                                                      dbsr.nnzb,
                                                                      dbsr.val,
                                                                      dbsr.ptr,
                                                                      dbsr.ind,
                                                                      dbsr.row_block_dim,
                                                                      dbsr.col_block_dim,
                                                                      dbsc.val,
                                                                      dbsc.ind,
                                                                      dbsc.ptr,
                                                                      action,
                                                                      dbsr.base,
                                                                      dbuffer);

        double gbyte_count = gebsr2gebsc_gbyte_count<T>(
            dbsr.mb, dbsr.nb, dbsr.nnzb, dbsr.row_block_dim, dbsr.col_block_dim, action);

        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::Mb,
                            dbsr.mb,
                            display_key_t::Nb,
                            dbsr.nb,
                            display_key_t::nnzb,
                            dbsr.nnzb,
                            display_key_t::rbdim,
                            dbsr.row_block_dim,
                            display_key_t::cbdim,
                            dbsr.col_block_dim,
                            display_key_t::action,
                            rocsparse_action2string(action),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                                  \
    template void testing_gebsr2gebsc_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_gebsr2gebsc<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_gebsr2gebsc_extra(const Arguments& arg) {}
