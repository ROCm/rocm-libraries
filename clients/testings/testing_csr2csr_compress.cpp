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
void testing_csr2csr_compress_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 1;
    static const T      safe_tol  = static_cast<T>(0);

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr_A;

    rocsparse_handle          handle  = local_handle;
    rocsparse_int             m       = safe_size;
    rocsparse_int             n       = safe_size;
    const rocsparse_mat_descr descr_A = local_descr_A;
    rocsparse_int             nnz_A   = safe_size;

    host_dense_vector<rocsparse_int> hnnz_per_row(safe_size);
    hnnz_per_row[0] = 1;
    device_dense_vector<rocsparse_int> dnnz_per_row(hnnz_per_row);

    //rocsparse_int  nnz_per_row_value = safe_size;
    rocsparse_int* nnz_per_row = (rocsparse_int*)dnnz_per_row;

    host_dense_vector<rocsparse_int> hptr(safe_size + 1);
    hptr[0] = 0;
    hptr[1] = 1;
    device_dense_vector<rocsparse_int> dcsr_row_ptr_C(hptr);

    host_dense_vector<rocsparse_int> hind_C(safe_size);
    hind_C[0] = 0;
    device_dense_vector<rocsparse_int> dcsr_col_ind_C(hind_C);

    host_dense_vector<T> hval_C(safe_size);
    hval_C[0] = static_cast<T>(1);
    device_dense_vector<T> dcsr_val_C(hval_C);

    host_dense_vector<rocsparse_int> hptr_A(safe_size + 1);
    hptr_A[0] = 0;
    hptr_A[1] = 1;
    device_dense_vector<rocsparse_int> dcsr_row_ptr_A(hptr_A);

    host_dense_vector<rocsparse_int> hind_A(safe_size);
    hind_A[0] = 0;
    device_dense_vector<rocsparse_int> dcsr_col_ind_A(hind_A);

    host_dense_vector<T> hval_A(safe_size);
    hval_A[0] = static_cast<T>(1);
    ;
    device_dense_vector<T> dcsr_val_A(hval_A);

    const T*             csr_val_A     = (const T*)dcsr_val_A;
    const rocsparse_int* csr_row_ptr_A = (const rocsparse_int*)dcsr_row_ptr_A;
    const rocsparse_int* csr_col_ind_A = (const rocsparse_int*)dcsr_col_ind_A;

    T*             csr_val_C     = (T*)dcsr_val_C;
    rocsparse_int* csr_row_ptr_C = (rocsparse_int*)dcsr_row_ptr_C;
    rocsparse_int* csr_col_ind_C = (rocsparse_int*)dcsr_col_ind_C;

    rocsparse_int  nnz   = safe_size;
    rocsparse_int* nnz_C = &nnz;

    int       nargs_to_exclude_nnz   = 2;
    const int args_to_exclude_nnz[2] = {3, 7};

    int       nargs_to_exclude   = 1;
    const int args_to_exclude[1] = {12};
    T         tol                = safe_tol;

#define PARAMS_NNZ handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol
#define PARAMS                                                                                     \
    handle, m, n, descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, nnz_A, nnz_per_row, csr_val_C, \
        csr_row_ptr_C, csr_col_ind_C, tol
    select_bad_arg_analysis(
        rocsparse_nnz_compress<T>, nargs_to_exclude_nnz, args_to_exclude_nnz, PARAMS_NNZ);
    select_bad_arg_analysis(
        rocsparse_csr2csr_compress<T>, nargs_to_exclude, args_to_exclude, PARAMS);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr_A, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_nnz_compress<T>(PARAMS_NNZ),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(PARAMS),
                            rocsparse_status_requires_sorted_storage);

#undef PARAMS
#undef PARAMS_NNZ
}

template <typename T>
void testing_csr2csr_compress(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M    = arg.M;
    rocsparse_int               N    = arg.N;
    rocsparse_index_base        base = arg.baseA;

    T tol = arg.get_alpha<T>();

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    rocsparse_local_mat_descr descr_A;

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr_A, base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || std::real(tol) < std::real(static_cast<T>(0)))
    {
        static const size_t safe_size = 100;

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr_A(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_A(safe_size);
        device_vector<T>             dcsr_val_A(safe_size);
        device_vector<rocsparse_int> dcsr_row_ptr_C(safe_size);
        device_vector<rocsparse_int> dcsr_col_ind_C(safe_size);
        device_vector<T>             dcsr_val_C(safe_size);
        device_vector<rocsparse_int> dnnz_per_row(safe_size);

        rocsparse_status status = rocsparse_status_success;
        if(M < 0 || N < 0)
        {
            status = rocsparse_status_invalid_size;
        }
        else if(std::real(tol) < std::real(static_cast<T>(0)))
        {
            status = rocsparse_status_invalid_value;
        }

        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2csr_compress<T>(handle,
                                                              M,
                                                              N,
                                                              descr_A,
                                                              dcsr_val_A,
                                                              dcsr_row_ptr_A,
                                                              dcsr_col_ind_A,
                                                              safe_size,
                                                              dnnz_per_row,
                                                              dcsr_val_C,
                                                              dcsr_row_ptr_C,
                                                              dcsr_col_ind_C,
                                                              tol),
                                status);

        return;
    }

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_A;
    host_vector<rocsparse_int> hcsr_col_ind_A;
    host_vector<T>             hcsr_val_A;
    host_vector<rocsparse_int> hcsr_row_ptr_C_gold;
    host_vector<rocsparse_int> hcsr_col_ind_C_gold;
    host_vector<T>             hcsr_val_C_gold;

    // Sample matrix
    rocsparse_int nnz_A;
    matrix_factory.init_csr(hcsr_row_ptr_A, hcsr_col_ind_A, hcsr_val_A, M, N, nnz_A, base);

    // Allocate host memory for nnz_per_row array
    host_vector<rocsparse_int> hnnz_per_row(M);

    // Allocate host memory for compressed CSR row pointer array
    host_vector<rocsparse_int> hcsr_row_ptr_C(M + 1);

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr_A(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind_A(nnz_A);
    device_vector<T>             dcsr_val_A(nnz_A);
    device_vector<rocsparse_int> dcsr_row_ptr_C(M + 1);
    device_vector<rocsparse_int> dnnz_per_row(M);

    // Copy data from CPU to device
    dcsr_row_ptr_A.transfer_from(hcsr_row_ptr_A);
    dcsr_col_ind_A.transfer_from(hcsr_col_ind_A);
    dcsr_val_A.transfer_from(hcsr_val_A);

    if(arg.unit_check)
    {
        // Obtain compressed CSR nnz twice, first using host pointer for nnz and second using device pointer
        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int hnnz_C;
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, M, descr_A, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &hnnz_C, tol));

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device));

        device_vector<rocsparse_int> dnnz_C(1);
        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, M, descr_A, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, dnnz_C, tol));

        rocsparse_int hnnz_C_copied_from_device = 0;
        CHECK_HIP_ERROR(hipMemcpy(
            &hnnz_C_copied_from_device, dnnz_C, sizeof(rocsparse_int), hipMemcpyDeviceToHost));

        // Confirm that nnz is the same regardless of whether we use host or device pointers
        unit_check_scalar(hnnz_C, hnnz_C_copied_from_device);

        // Allocate device memory for compressed CSR col indices and values array
        device_vector<rocsparse_int> dcsr_col_ind_C(hnnz_C);
        device_vector<T>             dcsr_val_C(hnnz_C);
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csr2csr_compress<T>(handle,
                                                                     M,
                                                                     N,
                                                                     descr_A,
                                                                     dcsr_val_A,
                                                                     dcsr_row_ptr_A,
                                                                     dcsr_col_ind_A,
                                                                     nnz_A,
                                                                     dnnz_per_row,
                                                                     dcsr_val_C,
                                                                     dcsr_row_ptr_C,
                                                                     dcsr_col_ind_C,
                                                                     tol));

        // Allocate host memory for compressed CSR col indices and values array
        host_vector<rocsparse_int> hcsr_col_ind_C(hnnz_C);
        host_vector<T>             hcsr_val_C(hnnz_C);

        hcsr_row_ptr_C.transfer_from(dcsr_row_ptr_C);
        hcsr_col_ind_C.transfer_from(dcsr_col_ind_C);
        hcsr_val_C.transfer_from(dcsr_val_C);

        // CPU csr2csr_compress
        host_csr_to_csr_compress<T>(M,
                                    N,
                                    nnz_A,
                                    hcsr_row_ptr_A,
                                    hcsr_col_ind_A,
                                    hcsr_val_A,
                                    hcsr_row_ptr_C_gold,
                                    hcsr_col_ind_C_gold,
                                    hcsr_val_C_gold,
                                    base,
                                    tol);

        hcsr_row_ptr_C_gold.unit_check(hcsr_row_ptr_C);
        hcsr_col_ind_C_gold.unit_check(hcsr_col_ind_C);
        hcsr_val_C_gold.unit_check(hcsr_val_C);
    }

    if(arg.timing)
    {

        CHECK_ROCSPARSE_ERROR(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host));

        rocsparse_int nnz_C;

        CHECK_ROCSPARSE_ERROR(rocsparse_nnz_compress<T>(
            handle, M, descr_A, dcsr_val_A, dcsr_row_ptr_A, dnnz_per_row, &nnz_C, tol));

        // Allocate device memory for compressed CSR col indices and values array
        device_vector<rocsparse_int> dcsr_col_ind_C(nnz_C);
        device_vector<T>             dcsr_val_C(nnz_C);

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csr2csr_compress<T>,
                                                                      handle,
                                                                      M,
                                                                      N,
                                                                      descr_A,
                                                                      dcsr_val_A,
                                                                      dcsr_row_ptr_A,
                                                                      dcsr_col_ind_A,
                                                                      nnz_A,
                                                                      dnnz_per_row,
                                                                      dcsr_val_C,
                                                                      dcsr_row_ptr_C,
                                                                      dcsr_col_ind_C,
                                                                      tol);

        double gbyte_count = csr2csr_compress_gbyte_count<T>(M, nnz_A, nnz_C);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz_A,
                            nnz_A,
                            display_key_t::nnz_C,
                            nnz_C,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                                       \
    template void testing_csr2csr_compress_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2csr_compress<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csr2csr_compress_extra(const Arguments& arg) {}
