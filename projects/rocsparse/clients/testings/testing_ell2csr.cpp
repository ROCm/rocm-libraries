/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
void testing_ell2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create descriptors
    rocsparse_local_mat_descr local_ell_descr;
    rocsparse_local_mat_descr local_csr_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_int             m           = safe_size;
    rocsparse_int             n           = safe_size;
    const rocsparse_mat_descr ell_descr   = local_ell_descr;
    rocsparse_int             ell_width   = safe_size;
    const T*                  ell_val     = (const T*)0x4;
    const rocsparse_int*      ell_col_ind = (const rocsparse_int*)0x4;
    const rocsparse_mat_descr csr_descr   = local_csr_descr;
    T*                        csr_val     = (T*)0x4;
    rocsparse_int*            csr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_int*            csr_col_ind = (rocsparse_int*)0x4;
    rocsparse_int*            csr_nnz     = (rocsparse_int*)0x4;

#define PARAMS_NNZ handle, m, n, ell_descr, ell_width, ell_col_ind, csr_descr, csr_row_ptr, csr_nnz
#define PARAMS                                                                                 \
    handle, m, n, ell_descr, ell_width, ell_val, ell_col_ind, csr_descr, csr_val, csr_row_ptr, \
        csr_col_ind
    bad_arg_analysis(rocsparse_ell2csr_nnz, PARAMS_NNZ);
    bad_arg_analysis(rocsparse_ell2csr<T>, PARAMS);

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(ell_descr, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_unsorted));
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr_nnz(PARAMS_NNZ),
                            rocsparse_status_requires_sorted_storage);
    EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(PARAMS), rocsparse_status_requires_sorted_storage);
#undef PARAMS
#undef PARAMS_NNZ
}

template <typename T>
void testing_ell2csr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M     = arg.M;
    rocsparse_int               N     = arg.N;
    rocsparse_index_base        baseA = arg.baseA;
    rocsparse_index_base        baseB = arg.baseB;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Create matrix descriptor for ELL matrix
    rocsparse_local_mat_descr descrA;

    // Create matrix descriptor for CSR matrix
    rocsparse_local_mat_descr descrB;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0)
    {
        static const size_t safe_size = 100;
        size_t              ptr_size  = std::max(safe_size, static_cast<size_t>(M + 1));

        // Allocate memory on device
        device_vector<rocsparse_int> dcsr_row_ptr(ptr_size);
        device_vector<rocsparse_int> dcsr_col_ind(safe_size);
        device_vector<T>             dcsr_val(safe_size);
        device_vector<rocsparse_int> dell_col_ind(safe_size);
        device_vector<T>             dell_val(safe_size);

        if(!dcsr_row_ptr || !dcsr_col_ind || !dcsr_val || !dell_col_ind || !dell_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        rocsparse_int csr_nnz;
        EXPECT_ROCSPARSE_STATUS(
            rocsparse_ell2csr_nnz(
                handle, M, N, descrA, safe_size, dell_col_ind, descrB, dcsr_row_ptr, &csr_nnz),
            (M < 0 || N < 0) ? rocsparse_status_invalid_size : rocsparse_status_success);
        EXPECT_ROCSPARSE_STATUS(rocsparse_ell2csr<T>(handle,
                                                     M,
                                                     N,
                                                     descrA,
                                                     safe_size,
                                                     dell_val,
                                                     dell_col_ind,
                                                     descrB,
                                                     dcsr_val,
                                                     dcsr_row_ptr,
                                                     dcsr_col_ind),
                                (M < 0 || N < 0) ? rocsparse_status_invalid_size
                                                 : rocsparse_status_success);

        return;
    }

    // Allocate host memory for matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;
    host_vector<rocsparse_int> hcsr_row_ptr_gold;
    host_vector<rocsparse_int> hcsr_col_ind_gold;
    host_vector<T>             hcsr_val_gold;

    // Sample matrix
    {
        rocsparse_int csr_nnz_gold;
        matrix_factory.init_csr(
            hcsr_row_ptr_gold, hcsr_col_ind_gold, hcsr_val_gold, M, N, csr_nnz_gold, baseB);
    }

    // Convert to ELL
    host_vector<rocsparse_int> hell_col_ind;
    host_vector<T>             hell_val;

    rocsparse_int ell_width = 0;
    for(rocsparse_int i = 0; i < M; ++i)
    {
        ell_width = std::max(hcsr_row_ptr_gold[i + 1] - hcsr_row_ptr_gold[i], ell_width);
    }

    rocsparse_int ell_nnz = ell_width * M;

    host_csr_to_ell(M,
                    hcsr_row_ptr_gold,
                    hcsr_col_ind_gold,
                    hcsr_val_gold,
                    hell_col_ind,
                    hell_val,
                    ell_width,
                    baseB,
                    baseA);

    hcsr_row_ptr_gold.clear();
    hcsr_col_ind_gold.clear();
    hcsr_val_gold.clear();

    // Allocate device memory
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);
    device_vector<rocsparse_int> dell_col_ind(ell_nnz);
    device_vector<T>             dell_val(ell_nnz);

    if(!dcsr_row_ptr || !dell_col_ind || !dell_val)
    {
        CHECK_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(
        dell_col_ind, hell_col_ind, sizeof(rocsparse_int) * ell_nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dell_val, hell_val, sizeof(T) * ell_nnz, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        // Obtain CSR nnz
        rocsparse_int csr_nnz;
        CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz(
            handle, M, N, descrA, ell_width, dell_col_ind, descrB, dcsr_row_ptr, &csr_nnz));

        // Allocate device memory
        device_vector<rocsparse_int> dcsr_col_ind(csr_nnz);
        device_vector<T>             dcsr_val(csr_nnz);

        if(!dcsr_col_ind || !dcsr_val)
        {
            CHECK_HIP_ERROR(hipErrorOutOfMemory);
            return;
        }

        // Perform CSR conversion
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_ell2csr<T>(handle,
                                                            M,
                                                            N,
                                                            descrA,
                                                            ell_width,
                                                            dell_val,
                                                            dell_col_ind,
                                                            descrB,
                                                            dcsr_val,
                                                            dcsr_row_ptr,
                                                            dcsr_col_ind));

        // Copy output to host
        hcsr_row_ptr.resize(M + 1);
        hcsr_col_ind.resize(csr_nnz);
        hcsr_val.resize(csr_nnz);

        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr, dcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_col_ind, dcsr_col_ind, sizeof(rocsparse_int) * csr_nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hcsr_val, dcsr_val, sizeof(T) * csr_nnz, hipMemcpyDeviceToHost));

        // CPU ell2csr
        rocsparse_int csr_nnz_gold;
        host_ell_to_csr<T>(M,
                           N,
                           hell_col_ind,
                           hell_val,
                           ell_width,
                           hcsr_row_ptr_gold,
                           hcsr_col_ind_gold,
                           hcsr_val_gold,
                           csr_nnz_gold,
                           baseA,
                           baseB);

        unit_check_scalar(csr_nnz_gold, csr_nnz);
        hcsr_row_ptr_gold.unit_check(hcsr_row_ptr);
        hcsr_col_ind_gold.unit_check(hcsr_col_ind);
        hcsr_val_gold.unit_check(hcsr_val);
    }

    if(arg.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = arg.iters;

        rocsparse_int csr_nnz;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz(
                handle, M, N, descrA, ell_width, dell_col_ind, descrB, dcsr_row_ptr, &csr_nnz));

            device_vector<rocsparse_int> dcsr_col_ind(csr_nnz);
            device_vector<T>             dcsr_val(csr_nnz);

            if(!dcsr_col_ind || !dcsr_val)
            {
                CHECK_HIP_ERROR(hipErrorOutOfMemory);
                return;
            }

            CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr<T>(handle,
                                                       M,
                                                       N,
                                                       descrA,
                                                       ell_width,
                                                       dell_val,
                                                       dell_col_ind,
                                                       descrB,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr_nnz(
                handle, M, N, descrA, ell_width, dell_col_ind, descrB, dcsr_row_ptr, &csr_nnz));

            device_vector<rocsparse_int> dcsr_col_ind(csr_nnz);
            device_vector<T>             dcsr_val(csr_nnz);

            if(!dcsr_col_ind || !dcsr_val)
            {
                CHECK_HIP_ERROR(hipErrorOutOfMemory);
                return;
            }

            CHECK_ROCSPARSE_ERROR(rocsparse_ell2csr<T>(handle,
                                                       M,
                                                       N,
                                                       descrA,
                                                       ell_width,
                                                       dell_val,
                                                       dell_col_ind,
                                                       descrB,
                                                       dcsr_val,
                                                       dcsr_row_ptr,
                                                       dcsr_col_ind));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = ell2csr_gbyte_count<T>(M, csr_nnz, ell_nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::csr_nnz,
                            csr_nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_ell2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_ell2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_ell2csr_extra(const Arguments& arg) {}
