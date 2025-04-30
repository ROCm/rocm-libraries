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
void testing_hyb2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create descriptor
    rocsparse_local_mat_descr local_descr;

    // Create HYB structure
    rocsparse_local_hyb_mat local_hyb;
    rocsparse_hyb_mat       ptr  = local_hyb;
    test_hyb*               dhyb = reinterpret_cast<test_hyb*>(ptr);

    dhyb->m       = safe_size;
    dhyb->n       = safe_size;
    dhyb->ell_nnz = safe_size;
    dhyb->coo_nnz = safe_size;

    rocsparse_handle          handle      = local_handle;
    const rocsparse_mat_descr descr       = local_descr;
    const rocsparse_hyb_mat   hyb         = local_hyb;
    T*                        csr_val     = (T*)0x4;
    rocsparse_int*            csr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind = (rocsparse_int*)0x4;
    size_t*                   buffer_size = (size_t*)0x4;
    void*                     temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE handle, descr, hyb, csr_row_ptr, buffer_size
#define PARAMS handle, descr, hyb, csr_val, csr_row_ptr, csr_col_ind, temp_buffer
    bad_arg_analysis(rocsparse_hyb2csr_buffer_size, PARAMS_BUFFER_SIZE);
    bad_arg_analysis(rocsparse_hyb2csr<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_storage_mode(descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_hyb2csr_buffer_size(PARAMS_BUFFER_SIZE),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_hyb2csr<T>(PARAMS), rocsparse_status_requires_sorted_storage);
#undef PARAMS
#undef PARAMS_BUFFER_SIZE
}

template <typename T>
void testing_hyb2csr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M    = arg.M;
    rocsparse_int               N    = arg.N;
    rocsparse_index_base        base = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Create HYB structure
    rocsparse_local_hyb_mat hyb;

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr_gold;
    host_vector<rocsparse_int> hcsr_col_ind_gold;
    host_vector<T>             hcsr_val_gold;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr_gold, hcsr_col_ind_gold, hcsr_val_gold, M, N, nnz, base);

    // Allocate device memory and convert to HYB
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dcsr_col_ind(nnz);
    device_vector<T>             dcsr_val(nnz);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_row_ptr, hcsr_row_ptr_gold, sizeof(rocsparse_int) * (M + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsr_col_ind, hcsr_col_ind_gold, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val_gold, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Convert CSR to HYB
    CHECK_ROCSPARSE_ERROR(rocsparse_csr2hyb<T>(handle,
                                               M,
                                               N,
                                               descr,
                                               dcsr_val,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               hyb,
                                               0,
                                               rocsparse_hyb_partition_auto));

    // Set all CSR arrays to zero
    CHECK_HIP_ERROR(hipMemset(dcsr_row_ptr, 0, sizeof(rocsparse_int) * (M + 1)));
    CHECK_HIP_ERROR(hipMemset(dcsr_col_ind, 0, sizeof(rocsparse_int) * nnz));
    CHECK_HIP_ERROR(hipMemset(dcsr_val, 0, sizeof(T) * nnz));

    // Obtain required buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(
        testing::rocsparse_hyb2csr_buffer_size(handle, descr, hyb, dcsr_row_ptr, &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_hyb2csr<T>(
            handle, descr, hyb, dcsr_val, dcsr_row_ptr, dcsr_col_ind, dbuffer));

        // Copy output to host
        host_vector<rocsparse_int> hcsr_row_ptr(M + 1);
        host_vector<rocsparse_int> hcsr_col_ind(nnz);
        host_vector<T>             hcsr_val(nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr, dcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind, dcsr_col_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val, dcsr_val, sizeof(T) * nnz, hipMemcpyDeviceToHost));

        hcsr_row_ptr_gold.unit_check(hcsr_row_ptr);
        hcsr_col_ind_gold.unit_check(hcsr_col_ind);
        hcsr_val_gold.unit_check(hcsr_val);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_hyb2csr<T>,
                                                                      handle,
                                                                      descr,
                                                                      hyb,
                                                                      dcsr_val,
                                                                      dcsr_row_ptr,
                                                                      dcsr_col_ind,
                                                                      dbuffer);

        // Initialize pseudo HYB matrix
        rocsparse_hyb_mat ptr  = hyb;
        test_hyb*         dhyb = reinterpret_cast<test_hyb*>(ptr);

        double gbyte_count = hyb2csr_gbyte_count<T>(M, nnz, dhyb->ell_nnz, dhyb->coo_nnz);

        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    // Free buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_hyb2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_hyb2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_hyb2csr_extra(const Arguments& arg) {}
