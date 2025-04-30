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

#include "testing.hpp"

template <typename T>
void testing_cscsort_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor
    rocsparse_local_mat_descr local_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_int             m           = safe_size;
    rocsparse_int             n           = safe_size;
    rocsparse_int             nnz         = safe_size;
    const rocsparse_mat_descr descr       = local_descr;
    rocsparse_int*            csc_col_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            csc_row_ind = (rocsparse_int*)0x4;
    size_t*                   buffer_size = (size_t*)0x4;
    void*                     temp_buffer = (void*)0x4;

    int            nargs_to_exclude   = 1;
    const int      args_to_exclude[1] = {7};
    rocsparse_int* perm               = nullptr;

#define PARAMS_BUFFER_SIZE handle, m, n, nnz, csc_col_ptr, csc_row_ind, buffer_size
#define PARAMS handle, m, n, nnz, descr, csc_col_ptr, csc_row_ind, perm, temp_buffer
    bad_arg_analysis(rocsparse_cscsort_buffer_size, PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(rocsparse_cscsort, nargs_to_exclude, args_to_exclude, PARAMS);
#undef PARAMS_BUFFER_SIZE
#undef PARAMS
}

template <typename T>
void testing_cscsort(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);

    rocsparse_int        M       = arg.M;
    rocsparse_int        N       = arg.N;
    bool                 permute = arg.algo;
    rocsparse_index_base base    = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor
    rocsparse_local_mat_descr descr;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descr, base));

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsc_row_ind;
    host_vector<rocsparse_int> hcsc_col_ptr;
    host_vector<T>             hcsc_val;
    host_vector<rocsparse_int> hcsc_row_ind_gold;
    host_vector<T>             hcsc_val_gold;

    // Sample matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsc_col_ptr, hcsc_row_ind, hcsc_val, N, M, nnz, base);

    // Unsort CSR matrix
    host_vector<rocsparse_int> hperm(nnz);
    hcsc_row_ind_gold = hcsc_row_ind;
    hcsc_val_gold     = hcsc_val;

    for(rocsparse_int i = 0; i < N; ++i)
    {
        rocsparse_int col_begin = hcsc_col_ptr[i] - base;
        rocsparse_int col_end   = hcsc_col_ptr[i + 1] - base;
        rocsparse_int col_nnz   = col_end - col_begin;

        for(rocsparse_int j = col_begin; j < col_end; ++j)
        {
            rocsparse_int rng = col_begin + rand() % col_nnz;
            std::swap(hcsc_row_ind[j], hcsc_row_ind[rng]);
            std::swap(hcsc_val[j], hcsc_val[rng]);
        }
    }

    // Allocate device memory
    device_vector<rocsparse_int> dcsc_row_ind(nnz);
    device_vector<rocsparse_int> dcsc_col_ptr(N + 1);
    device_vector<T>             dcsc_val(nnz);
    device_vector<rocsparse_int> dperm(nnz);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsc_row_ind, hcsc_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(
        dcsc_col_ptr, hcsc_col_ptr, sizeof(rocsparse_int) * (N + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsc_val, hcsc_val, sizeof(T) * nnz, hipMemcpyHostToDevice));

    // Obtain buffer size
    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(
        rocsparse_cscsort_buffer_size(handle, M, N, nnz, dcsc_col_ptr, dcsc_row_ind, &buffer_size));

    // Allocate buffer
    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        // Create permutation vector
        CHECK_ROCSPARSE_ERROR(rocsparse_create_identity_permutation(handle, nnz, dperm));

        // Sort CSR matrix
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_cscsort(handle,
                                                         M,
                                                         N,
                                                         nnz,
                                                         descr,
                                                         dcsc_col_ptr,
                                                         dcsc_row_ind,
                                                         permute ? dperm : nullptr,
                                                         dbuffer));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsc_row_ind, dcsc_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyDeviceToHost));

        hcsc_row_ind_gold.unit_check(hcsc_row_ind);

        // Permute, copy and check values, if requested
        if(permute)
        {
            device_vector<T> dcsc_val_sorted(nnz);

            CHECK_ROCSPARSE_ERROR(rocsparse_gthr<T>(
                handle, nnz, dcsc_val, dcsc_val_sorted, dperm, rocsparse_index_base_zero));
            CHECK_HIP_ERROR(
                hipMemcpy(hcsc_val, dcsc_val_sorted, sizeof(T) * nnz, hipMemcpyDeviceToHost));

            hcsc_val_gold.unit_check(hcsc_val);
        }
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_cscsort,
                                                                      handle,
                                                                      M,
                                                                      N,
                                                                      nnz,
                                                                      descr,
                                                                      dcsc_col_ptr,
                                                                      dcsc_row_ind,
                                                                      permute ? dperm : nullptr,
                                                                      dbuffer);

        double gbyte_count = cscsort_gbyte_count<T>(N, nnz, permute);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::permute,
                            (permute ? "yes" : "no"),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    // Clear buffer
    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_cscsort_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_cscsort<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_cscsort_extra(const Arguments& arg) {}
