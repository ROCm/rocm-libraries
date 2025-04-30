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
void testing_check_matrix_ell_bad_arg(const Arguments& arg)
{
    rocsparse_int          m           = 100;
    rocsparse_int          n           = 100;
    rocsparse_int          ell_width   = 100;
    rocsparse_index_base   idx_base    = rocsparse_index_base_zero;
    rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general;
    rocsparse_fill_mode    uplo        = rocsparse_fill_mode_lower;
    rocsparse_storage_mode storage     = rocsparse_storage_mode_sorted;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    const rocsparse_int*   ell_col_ind = (const rocsparse_int*)0x4;
    const T*               ell_val     = (const T*)0x4;
    void*                  temp_buffer = (void*)0x4;
    size_t*                buffer_size = (size_t*)0x4;
    rocsparse_data_status* data_status = (rocsparse_data_status*)0x4;

    int       nargs_to_exclude   = 2;
    const int args_to_exclude[2] = {4, 5};
#define PARAMS_BUFFER_SIZE \
    handle, m, n, ell_width, ell_val, ell_col_ind, idx_base, matrix_type, uplo, storage, buffer_size
#define PARAMS                                                                           \
    handle, m, n, ell_width, ell_val, ell_col_ind, idx_base, matrix_type, uplo, storage, \
        data_status, temp_buffer
    select_bad_arg_analysis(rocsparse_check_matrix_ell_buffer_size<T>,
                            nargs_to_exclude,
                            args_to_exclude,
                            PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(
        rocsparse_check_matrix_ell<T>, nargs_to_exclude, args_to_exclude, PARAMS);
#undef PARAMS_BUFFER_SIZE
#undef PARAMS
}

template <typename T>
void testing_check_matrix_ell(const Arguments& arg)
{
    rocsparse_int          m           = arg.M;
    rocsparse_int          n           = arg.N;
    rocsparse_index_base   base        = arg.baseA;
    rocsparse_matrix_type  matrix_type = arg.matrix_type;
    rocsparse_fill_mode    uplo        = arg.uplo;
    rocsparse_storage_mode storage     = arg.storage;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_matrix_factory<T> matrix_factory(arg);

    // Generate (or load from file) ELL matrix
    host_ell_matrix<T> hA;
    matrix_factory.init_ell(hA, m, n, base);

    device_ell_matrix<T> dA(hA);

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_ell_buffer_size<T>(
        handle, m, n, dA.width, dA.val, dA.ind, base, matrix_type, uplo, storage, &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    rocsparse_data_status data_status;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_ell<T>(handle,
                                                        m,
                                                        n,
                                                        dA.width,
                                                        dA.val,
                                                        dA.ind,
                                                        base,
                                                        matrix_type,
                                                        uplo,
                                                        storage,
                                                        &data_status,
                                                        dbuffer));

    CHECK_ROCSPARSE_DATA_ERROR(data_status);

    if(m > 1 && n > 1 && hA.width > 2)
    {
        rocsparse_int temp1;
        rocsparse_int temp2;
        T             temp_val;
        rocsparse_int row;

        rocsparse_seedrand();

        row      = random_generator_exact<rocsparse_int>(0, m - 1);
        temp1    = hA.ind[row];
        temp_val = hA.val[row];

        // Check matrix with column index not -1 but val being inf
        hA.ind[row] = random_generator_exact<rocsparse_int>(0, n - 1) + base;
        hA.val[row] = rocsparse_inf<T>();
        dA.transfer_from(hA);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_ell<T>(handle,
                                                            m,
                                                            n,
                                                            dA.width,
                                                            dA.val,
                                                            dA.ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_inf);

        // Check matrix with column index not -1 but val being nan
        hA.ind[row] = random_generator_exact<rocsparse_int>(0, n - 1) + base;
        hA.val[row] = rocsparse_nan<T>();
        dA.transfer_from(hA);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_ell<T>(handle,
                                                            m,
                                                            n,
                                                            dA.width,
                                                            dA.val,
                                                            dA.ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_nan);

        // Restore indices
        hA.ind[row] = temp1;
        hA.val[row] = temp_val;
        dA.transfer_from(hA);

        if(storage == rocsparse_storage_mode_sorted)
        {
            row   = random_generator_exact<rocsparse_int>(0, m - 1);
            temp1 = hA.ind[row];
            temp2 = hA.ind[m + row];

            // Check invalid sorting in column indices
            hA.ind[row]     = 1 + base;
            hA.ind[m + row] = 0 + base;
            dA.transfer_from(hA);

            CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_ell<T>(handle,
                                                                m,
                                                                n,
                                                                dA.width,
                                                                dA.val,
                                                                dA.ind,
                                                                base,
                                                                matrix_type,
                                                                uplo,
                                                                storage,
                                                                &data_status,
                                                                dbuffer));
            EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_sorting);

            // Restore indices
            hA.ind[row]     = temp1;
            hA.ind[m + row] = temp2;
            dA.transfer_from(hA);
        }
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_check_matrix_ell<T>,
                                                                      handle,
                                                                      m,
                                                                      n,
                                                                      dA.width,
                                                                      dA.val,
                                                                      dA.ind,
                                                                      base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      &data_status,
                                                                      dbuffer);

        double gbyte_count = check_matrix_ell_gbyte_count<T>(hA.nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            hA.nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                                       \
    template void testing_check_matrix_ell_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_check_matrix_ell<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_check_matrix_ell_extra(const Arguments& arg) {}
