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
void testing_coo2csr_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    const rocsparse_int* coo_row_ind = (const rocsparse_int*)0x4;
    rocsparse_int        nnz         = safe_size;
    rocsparse_int        m           = safe_size;
    rocsparse_int*       csr_row_ptr = (rocsparse_int*)0x4;
    rocsparse_index_base idx_base    = rocsparse_index_base_zero;

#define PARAMS handle, coo_row_ind, nnz, m, csr_row_ptr, idx_base
    bad_arg_analysis(rocsparse_coo2csr, PARAMS);
#undef PARAMS
}

template <typename T>
void testing_coo2csr(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);

    rocsparse_int        M    = arg.M;
    rocsparse_int        N    = arg.N;
    rocsparse_index_base base = arg.baseA;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Allocate host memory for COO matrix
    host_vector<rocsparse_int> hcoo_row_ind;
    host_vector<rocsparse_int> hcoo_col_ind;
    host_vector<T>             hcoo_val;

    // Sample matrix
    int64_t coo_nnz;
    matrix_factory.init_coo(hcoo_row_ind, hcoo_col_ind, hcoo_val, M, N, coo_nnz, base);

    rocsparse_int nnz = rocsparse_convert_to_int(coo_nnz);

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr(M + 1);
    host_vector<rocsparse_int> hcsr_row_ptr_gold(M + 1);

    // Allocate device memory
    device_vector<rocsparse_int> dcoo_row_ind(nnz);
    device_vector<rocsparse_int> dcsr_row_ptr(M + 1);

    // Copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcoo_row_ind, hcoo_row_ind, sizeof(rocsparse_int) * nnz, hipMemcpyHostToDevice));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(
            testing::rocsparse_coo2csr(handle, dcoo_row_ind, nnz, M, dcsr_row_ptr, base));

        // Copy output to host
        CHECK_HIP_ERROR(hipMemcpy(
            hcsr_row_ptr, dcsr_row_ptr, sizeof(rocsparse_int) * (M + 1), hipMemcpyDeviceToHost));

        // CPU coo2csr
        host_coo_to_csr(M, nnz, hcoo_row_ind.data(), hcsr_row_ptr_gold.data(), base);
        hcsr_row_ptr_gold.unit_check(hcsr_row_ptr);

        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("dcsr_row_ptr", dcsr_row_ptr);
        }
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(
            arg, rocsparse_coo2csr, handle, dcoo_row_ind, nnz, M, dcsr_row_ptr, base);

        double gbyte_count = coo2csr_gbyte_count<T>(M, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

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
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_coo2csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_coo2csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_coo2csr_extra(const Arguments& arg) {}
