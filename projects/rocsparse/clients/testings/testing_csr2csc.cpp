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
void testing_csr2csc_bad_arg(const Arguments& arg)
{
    static const size_t safe_size = 100;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;

    rocsparse_handle     handle      = local_handle;
    rocsparse_int        m           = safe_size;
    rocsparse_int        n           = safe_size;
    rocsparse_int        nnz         = safe_size;
    const T*             csr_val     = (const T*)0x4;
    const rocsparse_int* csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int* csr_col_ind = (const rocsparse_int*)0x4;
    T*                   csc_val     = (T*)0x4;
    rocsparse_int*       csc_col_ptr = (rocsparse_int*)0x4;
    rocsparse_int*       csc_row_ind = (rocsparse_int*)0x4;
    rocsparse_action     copy_values = rocsparse_action_numeric;
    rocsparse_index_base idx_base    = rocsparse_index_base_zero;
    size_t*              buffer_size = (size_t*)0x4;
    void*                temp_buffer = (void*)0x4;

#define PARAMS_BUFFER_SIZE handle, m, n, nnz, csr_row_ptr, csr_col_ind, copy_values, buffer_size
#define PARAMS                                                                               \
    handle, m, n, nnz, csr_val, csr_row_ptr, csr_col_ind, csc_val, csc_row_ind, csc_col_ptr, \
        copy_values, idx_base, temp_buffer
    bad_arg_analysis(rocsparse_csr2csc_buffer_size, PARAMS_BUFFER_SIZE);
    bad_arg_analysis(rocsparse_csr2csc<T>, PARAMS);
#undef PARAMS
#undef PARAMS_BUFFER_SIZE
}

template <typename T>
void testing_csr2csc(const Arguments& arg)
{
    rocsparse_matrix_factory<T> matrix_factory(arg);
    rocsparse_int               M      = arg.M;
    rocsparse_int               N      = arg.N;
    rocsparse_index_base        base   = arg.baseA;
    rocsparse_action            action = arg.action;

    // Create rocsparse handle
    rocsparse_local_handle handle(arg);

    // Sample matrix
    host_csr_matrix<T> hA;
    matrix_factory.init_csr(hA, M, N);
    rocsparse_int nnz = hA.nnz;

    host_csc_matrix<T> hC(M, N, nnz, base);
    host_csc_matrix<T> hC_gold(M, N, nnz, base);

    device_csr_matrix<T> dA(hA);
    device_csc_matrix<T> dC(hC);

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(testing::rocsparse_csr2csc_buffer_size(
        handle, M, N, nnz, dA.ptr, dA.ind, action, &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    if(arg.unit_check)
    {
        CHECK_ROCSPARSE_ERROR(testing::rocsparse_csr2csc<T>(handle,
                                                            M,
                                                            N,
                                                            nnz,
                                                            dA.val,
                                                            dA.ptr,
                                                            dA.ind,
                                                            dC.val,
                                                            dC.ind,
                                                            dC.ptr,
                                                            action,
                                                            base,
                                                            dbuffer));
        hC.transfer_from(dC);

        host_csc_matrix<T> hC_gold(M, N, nnz, base);
        host_csr_to_csc(M,
                        N,
                        nnz,
                        hA.ptr.data(),
                        hA.ind.data(),
                        hA.val.data(),
                        hC_gold.ind,
                        hC_gold.ptr,
                        hC_gold.val,
                        action,
                        base);

        hC_gold.ptr.unit_check(hC.ptr);
        hC_gold.ind.unit_check(hC.ind);

        if(action == rocsparse_action_numeric)
        {
            hC_gold.val.unit_check(hC.val);
        }
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_csr2csc<T>,
                                                                      handle,
                                                                      M,
                                                                      N,
                                                                      nnz,
                                                                      dA.val,
                                                                      dA.ptr,
                                                                      dA.ind,
                                                                      dC.val,
                                                                      dC.ind,
                                                                      dC.ptr,
                                                                      action,
                                                                      base,
                                                                      dbuffer);

        double gbyte_count = csr2csc_gbyte_count<T>(M, N, nnz, action);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            M,
                            display_key_t::N,
                            N,
                            display_key_t::nnz,
                            nnz,
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

#define INSTANTIATE(TYPE)                                              \
    template void testing_csr2csc_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_csr2csc<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_csr2csc_extra(const Arguments& arg) {}
