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
void testing_check_matrix_csr_bad_arg(const Arguments& arg)
{
    rocsparse_int          m           = 100;
    rocsparse_int          n           = 100;
    rocsparse_int          nnz         = 100;
    rocsparse_index_base   idx_base    = rocsparse_index_base_zero;
    rocsparse_matrix_type  matrix_type = rocsparse_matrix_type_general;
    rocsparse_fill_mode    uplo        = rocsparse_fill_mode_lower;
    rocsparse_storage_mode storage     = rocsparse_storage_mode_sorted;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    const rocsparse_int*   csr_row_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*   csr_col_ind = (const rocsparse_int*)0x4;
    const T*               csr_val     = (const T*)0x4;
    void*                  temp_buffer = (void*)0x4;
    size_t*                buffer_size = (size_t*)0x4;
    rocsparse_data_status* data_status = (rocsparse_data_status*)0x4;

    int       nargs_to_exclude   = 3;
    const int args_to_exclude[3] = {4, 5, 6};
#define PARAMS_BUFFER_SIZE                                                                      \
    handle, m, n, nnz, csr_val, csr_row_ptr, csr_col_ind, idx_base, matrix_type, uplo, storage, \
        buffer_size
#define PARAMS                                                                                  \
    handle, m, n, nnz, csr_val, csr_row_ptr, csr_col_ind, idx_base, matrix_type, uplo, storage, \
        data_status, temp_buffer
    select_bad_arg_analysis(rocsparse_check_matrix_csr_buffer_size<T>,
                            nargs_to_exclude,
                            args_to_exclude,
                            PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(
        rocsparse_check_matrix_csr<T>, nargs_to_exclude, args_to_exclude, PARAMS);
#undef PARAMS_BUFFER_SIZE
#undef PARAMS
}

template <typename T>
void testing_check_matrix_csr(const Arguments& arg)
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

    // Allocate host memory for CSR matrix
    host_vector<rocsparse_int> hcsr_row_ptr;
    host_vector<rocsparse_int> hcsr_col_ind;
    host_vector<T>             hcsr_val;

    // Generate (or load from file) CSR matrix
    rocsparse_int nnz;
    matrix_factory.init_csr(hcsr_row_ptr, hcsr_col_ind, hcsr_val, m, n, nnz, base);

    // CSR matrix on device
    device_vector<rocsparse_int> dcsr_row_ptr(hcsr_row_ptr);
    device_vector<rocsparse_int> dcsr_col_ind(hcsr_col_ind);
    device_vector<T>             dcsr_val(hcsr_val);

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr_buffer_size<T>(handle,
                                                                    m,
                                                                    n,
                                                                    nnz,
                                                                    dcsr_val,
                                                                    dcsr_row_ptr,
                                                                    dcsr_col_ind,
                                                                    base,
                                                                    matrix_type,
                                                                    uplo,
                                                                    storage,
                                                                    &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    rocsparse_data_status data_status;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        dcsr_val,
                                                        dcsr_row_ptr,
                                                        dcsr_col_ind,
                                                        base,
                                                        matrix_type,
                                                        uplo,
                                                        storage,
                                                        &data_status,
                                                        dbuffer));

    CHECK_ROCSPARSE_DATA_ERROR(data_status);

    // Check passing shifting ptr array by large number
    host_vector<rocsparse_int> hcsr_row_ptr_shifted(hcsr_row_ptr);
    for(size_t i = 0; i < hcsr_row_ptr_shifted.size(); i++)
    {
        hcsr_row_ptr_shifted[i] += 10000;
    }
    device_vector<rocsparse_int> dcsr_row_ptr_shifted(hcsr_row_ptr_shifted);

    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        dcsr_val,
                                                        dcsr_row_ptr_shifted,
                                                        dcsr_col_ind,
                                                        base,
                                                        matrix_type,
                                                        uplo,
                                                        storage,
                                                        &data_status,
                                                        dbuffer));

    CHECK_ROCSPARSE_DATA_ERROR(data_status);

    if(nnz > 1 && m > 1)
    {
        rocsparse_int temp;
        T             temp_val;
        rocsparse_int rng;

        rocsparse_seedrand();

        rng  = random_generator_exact<rocsparse_int>(1, m - 1);
        temp = hcsr_row_ptr[rng];

        // Check passing matrix with invalid row ptr offset set to number less than zero
        hcsr_row_ptr[rng] = -1;
        dcsr_row_ptr.transfer_from(hcsr_row_ptr);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsr_val,
                                                            dcsr_row_ptr,
                                                            dcsr_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_offset_ptr);

        // Restore offset pointer
        hcsr_row_ptr[rng] = temp;
        dcsr_row_ptr.transfer_from(hcsr_row_ptr);

        rng  = random_generator_exact<rocsparse_int>(0, nnz - 1);
        temp = hcsr_col_ind[rng];

        // Check passing matrix with column index set to number less than zero
        hcsr_col_ind[rng] = -1;
        dcsr_col_ind.transfer_from(hcsr_col_ind);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsr_val,
                                                            dcsr_row_ptr,
                                                            dcsr_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_index);

        // Check passing matrix with column index set to number greater than n - 1 + base
        hcsr_col_ind[rng] = (n - 1 + base) + 10;
        dcsr_col_ind.transfer_from(hcsr_col_ind);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsr_val,
                                                            dcsr_row_ptr,
                                                            dcsr_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_index);

        // Restore indices
        hcsr_col_ind[rng] = temp;
        dcsr_col_ind.transfer_from(hcsr_col_ind);

        // Check passing matrix with duplicate columns
        {
            // Find first row with non-zeros in it
            rocsparse_int row = -1;
            for(size_t i = 1; i < m; i++)
            {
                if(hcsr_row_ptr[i + 1] - hcsr_row_ptr[i] >= 2)
                {
                    row = i;
                    break;
                }
            }

            if(row != -1)
            {
                rocsparse_int index = hcsr_row_ptr[row] - base + 1;
                temp                = hcsr_col_ind[index];
                hcsr_col_ind[index] = hcsr_col_ind[hcsr_row_ptr[row] - base];
                dcsr_col_ind.transfer_from(hcsr_col_ind);

                CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                                    m,
                                                                    n,
                                                                    nnz,
                                                                    dcsr_val,
                                                                    dcsr_row_ptr,
                                                                    dcsr_col_ind,
                                                                    base,
                                                                    matrix_type,
                                                                    uplo,
                                                                    storage,
                                                                    &data_status,
                                                                    dbuffer));
                EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_duplicate_entry);

                // Restore indices
                hcsr_col_ind[index] = temp;
                dcsr_col_ind.transfer_from(hcsr_col_ind);
            }
        }

        // Check passing matrix with invalid fill
        if(matrix_type != rocsparse_matrix_type_general)
        {
            // Find first row with non-zeros in it
            rocsparse_int row = -1;
            for(size_t i = 1; i < m; i++)
            {
                if(hcsr_row_ptr[i + 1] - hcsr_row_ptr[i] > 0)
                {
                    row = i;
                    break;
                }
            }

            if(row != -1)
            {
                rocsparse_int index = (uplo == rocsparse_fill_mode_lower)
                                          ? hcsr_row_ptr[row + 1] - base - 1
                                          : hcsr_row_ptr[row] - base;
                temp                = hcsr_col_ind[index];

                hcsr_col_ind[index] = (uplo == rocsparse_fill_mode_lower) ? (n - 1 + base) : base;
                dcsr_col_ind.transfer_from(hcsr_col_ind);

                CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                                    m,
                                                                    n,
                                                                    nnz,
                                                                    dcsr_val,
                                                                    dcsr_row_ptr,
                                                                    dcsr_col_ind,
                                                                    base,
                                                                    matrix_type,
                                                                    uplo,
                                                                    storage,
                                                                    &data_status,
                                                                    dbuffer));
                EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_fill);

                // Restore indices
                hcsr_col_ind[index] = temp;
                dcsr_col_ind.transfer_from(hcsr_col_ind);
            }
        }

        rng      = random_generator_exact<rocsparse_int>(0, nnz - 1);
        temp_val = hcsr_val[rng];

        // Check passing matrix with inf value
        hcsr_val[rng] = rocsparse_inf<T>();
        dcsr_val.transfer_from(hcsr_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsr_val,
                                                            dcsr_row_ptr,
                                                            dcsr_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_inf);

        // Check passing matrix with nan value
        hcsr_val[rng] = rocsparse_nan<T>();
        dcsr_val.transfer_from(hcsr_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csr<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsr_val,
                                                            dcsr_row_ptr,
                                                            dcsr_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_nan);

        // Restore indices
        hcsr_val[rng] = temp_val;
        dcsr_val.transfer_from(hcsr_val);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_check_matrix_csr<T>,
                                                                      handle,
                                                                      m,
                                                                      n,
                                                                      nnz,
                                                                      dcsr_val,
                                                                      dcsr_row_ptr,
                                                                      dcsr_col_ind,
                                                                      base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      &data_status,
                                                                      dbuffer);

        double gbyte_count = check_matrix_csr_gbyte_count<T>(m, nnz);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::nnz,
                            nnz,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                                       \
    template void testing_check_matrix_csr_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_check_matrix_csr<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_check_matrix_csr_extra(const Arguments& arg) {}
