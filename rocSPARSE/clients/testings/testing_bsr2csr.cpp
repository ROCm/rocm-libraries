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
void testing_bsr2csr_bad_arg(const Arguments& arg)
{
    static const size_t              safe_size = 1;
    host_dense_vector<rocsparse_int> hbsr_row_ptr(safe_size + 1);
    hbsr_row_ptr[0] = 0;
    hbsr_row_ptr[1] = 1;
    device_dense_vector<rocsparse_int> dbsr_row_ptr(hbsr_row_ptr);
    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create rocsparse decriptors
    rocsparse_local_mat_descr local_bsr_descr;
    rocsparse_local_mat_descr local_csr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_direction       dir         = rocsparse_direction_row;
    rocsparse_int             mb          = safe_size;
    rocsparse_int             nb          = safe_size;
    const rocsparse_mat_descr bsr_descr   = local_bsr_descr;
    const T*                  bsr_val     = (const T*)0x4;
    const rocsparse_int*      bsr_row_ptr = (const rocsparse_int*)dbsr_row_ptr;

    const rocsparse_int*      bsr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim   = safe_size;
    const rocsparse_mat_descr csr_descr   = local_csr_descr;
    T*                        csr_val     = (T*)0x4;
    rocsparse_int*            csr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind = (rocsparse_int*)0x4;

#define PARAMS                                                                               \
    handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, csr_descr, \
        csr_val, csr_row_ptr, csr_col_ind

    bad_arg_analysis(rocsparse_bsr2csr<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(PARAMS), rocsparse_status_requires_sorted_storage);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_sorted));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_sorted));

    // Check block_dim == 0
    block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_bsr2csr<T>(PARAMS), rocsparse_status_invalid_size);

#undef PARAMS
}

template <typename T>
void testing_bsr2csr(const Arguments& arg)
{
    rocsparse_int        M         = arg.M;
    rocsparse_int        N         = arg.N;
    rocsparse_index_base bsr_base  = arg.baseA;
    rocsparse_index_base csr_base  = arg.baseB;
    rocsparse_int        block_dim = arg.block_dim;

    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    rocsparse_local_mat_descr bsr_descr;
    rocsparse_local_mat_descr csr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, bsr_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));

    rocsparse_matrix_factory<T> matrix_factory(arg);

    // Declare and initialize matrices.
    host_gebsr_matrix<T>   hA;
    device_gebsr_matrix<T> dA;

    matrix_factory.init_bsr(hA, dA, Mb, Nb, bsr_base);

    M = dA.mb * dA.row_block_dim;
    N = dA.nb * dA.col_block_dim;

    rocsparse_int nnzb = hA.ind.size();

    // Allocate device memory for output CSR matrix
    device_csr_matrix<T> dC(M, N, size_t(nnzb) * block_dim * block_dim, csr_base);

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_bsr2csr<T>(handle,
                                                            dA.block_direction,
                                                            dA.mb,
                                                            dA.nb,
                                                            bsr_descr,
                                                            dA.val,
                                                            dA.ptr,
                                                            dA.ind,
                                                            block_dim,
                                                            csr_descr,
                                                            dC.val,
                                                            dC.ptr,
                                                            dC.ind));

        host_csr_matrix<T> hC_gold(M, N, size_t(nnzb) * block_dim * block_dim, csr_base);
        host_bsr_to_csr<T>(hA.block_direction,
                           hA.mb,
                           hA.nb,
                           hA.nnzb,
                           hA.val,
                           hA.ptr,
                           hA.ind,
                           block_dim,
                           hA.base,
                           hC_gold.val,
                           hC_gold.ptr,
                           hC_gold.ind,
                           hC_gold.base);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC", dC);
        }

        hC_gold.near_check(dC);
    }

    if(arg.timing)
    {
        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_bsr2csr<T>,
                                                                      handle,
                                                                      dA.block_direction,
                                                                      dA.mb,
                                                                      dA.nb,
                                                                      bsr_descr,
                                                                      dA.val,
                                                                      dA.ptr,
                                                                      dA.ind,
                                                                      block_dim,
                                                                      csr_descr,
                                                                      dC.val,
                                                                      dC.ptr,
                                                                      dC.ind);

        double gbyte_count = bsr2csr_gbyte_count<T>(Mb, block_dim, nnzb);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::Mb,
                            Mb,
                            display_key_t::Nb,
                            Nb,
                            display_key_t::bdim,
                            block_dim,
                            display_key_t::nnzb,
                            nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_bsr2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_bsr2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
#undef INSTANTIATE
void testing_bsr2csr_extra(const Arguments& arg) {}
