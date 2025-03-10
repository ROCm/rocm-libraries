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
#include "rocsparse_enum.hpp"
#include "testing.hpp"

template <typename T>
void testing_csr2ell_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    // Create matrix descriptor for CSR matrix
    rocsparse_local_mat_descr local_csr_descr;

    // Create matrix descriptor for ELL matrix
    rocsparse_local_mat_descr local_ell_descr;

    rocsparse_handle          handle      = local_handle;
    rocsparse_int             m           = safe_size;
    const rocsparse_mat_descr csr_descr   = local_csr_descr;
    const T*                  csr_val     = (const T*)0x4;
    const rocsparse_int*      csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*      csr_col_ind = (const rocsparse_int*)0x4;
    const rocsparse_mat_descr ell_descr   = local_ell_descr;
    T*                        ell_val     = (T*)0x4;
    rocsparse_int*            ell_col_ind = (rocsparse_int*)0x4;

#define PARAMS_WIDTH handle, m, csr_descr, csr_row_ptr, ell_descr, ell_width

#define PARAMS                                                                              \
    handle, m, csr_descr, csr_val, csr_row_ptr, csr_col_ind, ell_descr, ell_width, ell_val, \
        ell_col_ind

    {
        rocsparse_int* ell_width = (rocsparse_int*)0x4;
        bad_arg_analysis(rocsparse_csr2ell_width, PARAMS_WIDTH);
    }

    {
        rocsparse_int ell_width = safe_size;
        bad_arg_analysis(rocsparse_csr2ell<T>, PARAMS);
    }

    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(csr_descr, rocsparse_storage_mode_unsorted));
    CHECK_ROCSPARSE_ERROR(
        rocsparse_set_mat_storage_mode(ell_descr, rocsparse_storage_mode_unsorted));
    {
        rocsparse_int* ell_width = (rocsparse_int*)0x4;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2ell_width(PARAMS_WIDTH),
                                rocsparse_status_requires_sorted_storage);
    }
    {
        rocsparse_int ell_width = safe_size;
        EXPECT_ROCSPARSE_STATUS(rocsparse_csr2ell<T>(PARAMS),
                                rocsparse_status_requires_sorted_storage);
    }

#undef PARAMS
#undef PARAMS_WIDTH
}

template <typename T>
void testing_csr2ell(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M     = arg.M;
    rocsparse_int               N     = arg.N;
    rocsparse_index_base        baseA = arg.baseA;
    rocsparse_index_base        baseB = arg.baseB;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Grab stream used by handle
    hipStream_t stream = handle.get_stream();

    // Create matrix descriptor for CSR matrix
    rocsparse_local_mat_descr descrA;

    // Create matrix descriptor for ELL matrix
    rocsparse_local_mat_descr descrB;

    // Set matrix index base
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrA, baseA));
    CHECK_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(descrB, baseB));

    // Sample matrix
    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA, M, N);
    rocsparse_int nnz = hA.nnz;

    device_csr_matrix<T> dA(hA);

    // Obtain ELL width
    rocsparse_int ell_width;
    CHECK_ROCSPARSE_ERROR(
        testing::rocsparse_csr2ell_width(handle, M, descrA, dA.ptr, descrB, &ell_width));

    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    // Allocate device memory
    rocsparse_int ell_nnz = ell_width * M;

    host_ell_matrix<T>   hB(M, N, ell_width, baseB);
    device_ell_matrix<T> dB(hB);

    if(arg.unit_check)
    {
        // Perform ELL conversion
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csr2ell<T>(
            handle, M, descrA, dA.val, dA.ptr, dA.ind, descrB, ell_width, dB.val, dB.ind));

        // Copy output to host
        hB.transfer_from(dB);
        if(ROCSPARSE_REPRODUCIBILITY)
        {
            rocsparse_reproducibility::save("B", hB);
        }

        // CPU csr2ell
        rocsparse_int      ell_width_gold;
        host_ell_matrix<T> hB_gold;
        host_csr_to_ell(
            M, hA.ptr, hA.ind, hA.val, hB_gold.ind, hB_gold.val, ell_width_gold, baseA, baseB);

        unit_check_scalar(ell_width_gold, ell_width);

        hB_gold.ind.unit_check(hB.ind);
        hB_gold.val.unit_check(hB.val);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csr2ell<T>,
                                                                      handle,
                                                                      M,
                                                                      descrA,
                                                                      dA.val,
                                                                      dA.ptr,
                                                                      dA.ind,
                                                                      descrB,
                                                                      ell_width,
                                                                      dB.val,
                                                                      dB.ind);

        double gbyte_count = csr2ell_gbyte_count<T>(M, nnz, ell_nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::ell_width,
                            ell_width,
                            display_key_t::ell_nnz,
                            ell_nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
}

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2ell_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2ell<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csr2ell_extra(const Arguments& arg) {}
