/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_bsrpad_value_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_bsr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_int             m           = safe_size;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             nnzb        = safe_size;
    rocsparse_int             block_dim   = 1;
    T                         value       = 1;
    const rocsparse_mat_descr bsr_descr   = local_bsr_descr;
    T*                        bsr_val     = (T*)0x4;
    rocsparse_int*            bsr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            bsr_col_ind = (rocsparse_int*)0x4;

    int       nargs_to_exclude   = 1;
    const int args_to_exclude[1] = {5};

#define PARAMS handle, m, mb, nnzb, block_dim, value, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind

    select_bad_arg_analysis(rocsparse_bsrpad_value<T>, nargs_to_exclude, args_to_exclude, PARAMS);

    // block_dim * mb > m
    mb = 3;
    m  = block_dim * mb + 1;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS), rocsparse_status_invalid_size);

    // block_dim * mb < m
    mb = 3;
    m  = block_dim * (mb - 1);
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS), rocsparse_status_invalid_size);

    // block_dim == 0
    block_dim = 0;
    mb        = 3;
    m         = block_dim * mb;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsrpad_value<T>(PARAMS), rocsparse_status_invalid_size);

#undef PARAMS
}

template <typename T>
void testing_bsrpad_value(const Arguments& arg)
{
    static constexpr bool       toint     = false;
    static constexpr bool       full_rank = false;
    rocsparse_matrix_factory<T> matrix_factory(arg, toint, full_rank);

    rocsparse_int          M         = arg.M;
    rocsparse_int          block_dim = arg.block_dim;
    rocsparse_index_base   base      = arg.baseA;
    rocsparse_direction    direction = arg.direction;
    rocsparse_storage_mode storage   = arg.storage;
    T                      value     = 1;

    rocsparse_int Mb = (M + block_dim - 1) / block_dim;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr bsr_descr;

    // Create matrix info
    rocsparse_local_mat_info info;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, base));

    // Set matrix storage mode
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(bsr_descr, storage));

    // Allocate device memory
    device_gebsr_matrix<T> dbsr;

    {
        host_csr_matrix<T> hcsrA;

        // Generate a temporary sorted csr matrix to get the correct dimensions
        hcsrA.define(M, M, 0, base);
        matrix_factory.init_csr(hcsrA.ptr,
                                hcsrA.ind,
                                hcsrA.val,
                                hcsrA.m,
                                hcsrA.n,
                                hcsrA.nnz,
                                hcsrA.base,
                                rocsparse_matrix_type_general,
                                rocsparse_fill_mode_lower,
                                rocsparse_storage_mode_sorted);
        M = hcsrA.m;

        device_csr_matrix<T> dcsrA(hcsrA);

        rocsparse_matrix_utils::convert(
            dcsrA, direction, block_dim, base, rocsparse_storage_mode_sorted, dbsr);

        switch(storage)
        {
        case rocsparse_storage_mode_unsorted:
        {
            host_gebsr_matrix<T> hbsr(dbsr);
            rocsparse_matrix_utils::host_gebsrunsort<T>(
                hbsr.ptr.data(), hbsr.ind.data(), hbsr.mb, hbsr.base);
            dbsr(hbsr);
            break;
        }
        case rocsparse_storage_mode_sorted:
        {
            break;
        }
        }

        Mb = dbsr.mb;
    }

    if(arg.unit_check)
    {
        // Allocate host memory for BSR matrix
        host_gebsr_matrix<T> hbsr(dbsr);

        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsrpad_value<T>(
            handle, M, Mb, dbsr.nnzb, block_dim, value, bsr_descr, dbsr.val, dbsr.ptr, dbsr.ind));

        host_gebsr_matrix<T> hbsrC(dbsr);

        // CPU bsrpad_value
        host_bsrpad_value<T>(
            M, Mb, hbsr.nnzb, block_dim, value, hbsr.val, hbsr.ptr, hbsr.ind, hbsr.base);

        hbsrC.unit_check(hbsr);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_bsrpad_value<T>,
                                                                      handle,
                                                                      M,
                                                                      Mb,
                                                                      dbsr.nnzb,
                                                                      block_dim,
                                                                      value,
                                                                      bsr_descr,
                                                                      dbsr.val,
                                                                      dbsr.ptr,
                                                                      dbsr.ind);

        double gbyte_count = 0;
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::Mb,
                            Mb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                   \
    template void testing_bsrpad_value_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsrpad_value<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_bsrpad_value_extra(const Arguments& arg) {}
