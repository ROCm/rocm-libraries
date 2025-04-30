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
void testing_csr2bsr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 1;

    host_dense_vector<rocsparse_int> hptr(safe_size + 1);
    hptr[0] = 0;
    hptr[1] = 1;
    device_dense_vector<rocsparse_int> dcsr_row_ptr(hptr);
    device_dense_vector<rocsparse_int> dbsr_row_ptr(hptr);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptors
    rocsparse_local_mat_descr local_csr_descr;
    rocsparse_local_mat_descr local_bsr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_direction       dir         = rocsparse_direction_row;
    rocsparse_int             m           = safe_size;
    rocsparse_int             n           = safe_size;
    const rocsparse_mat_descr csr_descr   = local_csr_descr;
    const T*                  csr_val     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr = (const rocsparse_int*)dcsr_row_ptr;
    const rocsparse_int*      csr_col_ind = (const rocsparse_int*)0x4;
    rocsparse_int             block_dim   = safe_size;
    const rocsparse_mat_descr bsr_descr   = local_bsr_descr;
    T*                        bsr_val     = (T*)0x4;
    rocsparse_int*            bsr_row_ptr = (rocsparse_int*)dbsr_row_ptr;
    rocsparse_int*            bsr_col_ind = (rocsparse_int*)0x4;
    rocsparse_int*            bsr_nnz     = (rocsparse_int*)0x4;

#define PARAMS_NNZ                                                                             \
    handle, dir, m, n, csr_descr, csr_row_ptr, csr_col_ind, block_dim, bsr_descr, bsr_row_ptr, \
        bsr_nnz
#define PARAMS                                                                             \
    handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, csr_col_ind, block_dim, bsr_descr, \
        bsr_val, bsr_row_ptr, bsr_col_ind
    {
        static constexpr int nargs_to_exclude                  = 1;
        const int            args_to_exclude[nargs_to_exclude] = {6};

        select_bad_arg_analysis(
            rocsparse_csr2bsr_nnz, nargs_to_exclude, args_to_exclude, PARAMS_NNZ);
    }

    {
        static constexpr int nargs_to_exclude                  = 4;
        const int            args_to_exclude[nargs_to_exclude] = {5, 7, 10, 12};
        select_bad_arg_analysis(rocsparse_csr2bsr<T>, nargs_to_exclude, args_to_exclude, PARAMS);
    }

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(PARAMS_NNZ),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(PARAMS), rocsparse_status_requires_sorted_storage);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_sorted));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(bsr_descr, rocsparse_storage_mode_sorted));

    // Check block_dim == 0
    block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr_nnz(PARAMS_NNZ), rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2bsr<T>(PARAMS), rocsparse_status_invalid_size);
#undef PARAMS
#undef PARAMS_NNZ
}

template <typename T>
void testing_csr2bsr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M         = arg.M;
    rocsparse_int               N         = arg.N;
    rocsparse_index_base        csr_base  = arg.baseA;
    rocsparse_index_base        bsr_base  = arg.baseB;
    rocsparse_direction         direction = arg.direction;
    rocsparse_int               block_dim = arg.block_dim;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_local_mat_descr csr_descr;
    rocsparse_local_mat_descr bsr_descr;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(csr_descr, csr_base));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(bsr_descr, bsr_base));

    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA, M, N);

    // M and N can be modified in rocsparse_init_csr_matrix
    rocsparse_int Mb = (M + block_dim - 1) / block_dim;
    rocsparse_int Nb = (N + block_dim - 1) / block_dim;

    device_csr_matrix<T>   dA(hA);
    device_gebsr_matrix<T> dC(direction, Mb, Nb, 0, block_dim, block_dim, bsr_base);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

    host_scalar<rocsparse_int> hbsr_nnzb;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                direction,
                                                dA.m,
                                                dA.n,
                                                csr_descr,
                                                dA.ptr,
                                                dA.ind,
                                                block_dim,
                                                bsr_descr,
                                                dC.ptr,
                                                hbsr_nnzb));

    CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

    device_scalar<rocsparse_int> dbsr_nnzb;
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                direction,
                                                dA.m,
                                                dA.n,
                                                csr_descr,
                                                dA.ptr,
                                                dA.ind,
                                                block_dim,
                                                bsr_descr,
                                                dC.ptr,
                                                dbsr_nnzb));

    dC.define(direction, Mb, Nb, *hbsr_nnzb, block_dim, block_dim, bsr_base);

    if(arg.unit_check)
    {
        dbsr_nnzb.unit_check(hbsr_nnzb);

        // Finish conversion
        CHECK_ROCSPARSE_ERROR(rocsparse_csr2bsr<T>(handle,
                                                   direction,
                                                   dA.m,
                                                   dA.n,
                                                   csr_descr,
                                                   dA.val,
                                                   dA.ptr,
                                                   dA.ind,
                                                   block_dim,
                                                   bsr_descr,
                                                   dC.val,
                                                   dC.ptr,
                                                   dC.ind));

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dC", dC);
        }

        // Host solution
        host_gebsr_matrix<T> hC_gold(direction, Mb, Nb, *hbsr_nnzb, block_dim, block_dim, bsr_base);
        host_csr_to_bsr(direction,
                        hA.m,
                        hA.n,
                        hA.nnz,
                        hA.val,
                        hA.ptr,
                        hA.ind,
                        block_dim,
                        csr_base,
                        hC_gold.val,
                        hC_gold.ptr,
                        hC_gold.ind,
                        bsr_base);

        hC_gold.near_check(dC);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csr2bsr<T>,
                                                                      handle,
                                                                      direction,
                                                                      dA.m,
                                                                      dA.n,
                                                                      csr_descr,
                                                                      dA.val,
                                                                      dA.ptr,
                                                                      dA.ind,
                                                                      block_dim,
                                                                      bsr_descr,
                                                                      dC.val,
                                                                      dC.ptr,
                                                                      dC.ind);

        double gbyte_count = csr2bsr_gbyte_count<T>(M, Mb, hA.nnz, *hbsr_nnzb, block_dim);
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
                            *hbsr_nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2bsr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2bsr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csr2bsr_extra(const Arguments& arg) {}
