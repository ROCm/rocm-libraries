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
void testing_check_matrix_csc_bad_arg(const Arguments& arg)
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

    const rocsparse_int*   csc_col_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*   csc_row_ind = (const rocsparse_int*)0x4;
    const T*               csc_val     = (const T*)0x4;
    void*                  temp_buffer = (void*)0x4;
    size_t*                buffer_size = (size_t*)0x4;
    rocsparse_data_status* data_status = (rocsparse_data_status*)0x4;

    int       nargs_to_exclude   = 3;
    const int args_to_exclude[3] = {4, 5, 6};
#define PARAMS_BUFFER_SIZE                                                                      \
    handle, m, n, nnz, csc_val, csc_col_ptr, csc_row_ind, idx_base, matrix_type, uplo, storage, \
        buffer_size
#define PARAMS                                                                                  \
    handle, m, n, nnz, csc_val, csc_col_ptr, csc_row_ind, idx_base, matrix_type, uplo, storage, \
        data_status, temp_buffer
    select_bad_arg_analysis(rocsparse_check_matrix_csc_buffer_size<T>,
                            nargs_to_exclude,
                            args_to_exclude,
                            PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(
        rocsparse_check_matrix_csc<T>, nargs_to_exclude, args_to_exclude, PARAMS);
#undef PARAMS_BUFFER_SIZE
#undef PARAMS
}

template <typename T>
void testing_check_matrix_csc(const Arguments& arg)
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

    // Allocate host memory for CSC matrix
    host_vector<rocsparse_int> hcsc_col_ptr;
    host_vector<rocsparse_int> hcsc_row_ind;
    host_vector<T>             hcsc_val;

    // Generate (or load from file) CSC matrix
    rocsparse_int nnz;
    matrix_factory.init_csc(hcsc_col_ptr, hcsc_row_ind, hcsc_val, m, n, nnz, base);

    // CSC matrix on device
    device_vector<rocsparse_int> dcsc_col_ptr(hcsc_col_ptr);
    device_vector<rocsparse_int> dcsc_row_ind(hcsc_row_ind);
    device_vector<T>             dcsc_val(hcsc_val);

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc_buffer_size<T>(handle,
                                                                    m,
                                                                    n,
                                                                    nnz,
                                                                    dcsc_val,
                                                                    dcsc_col_ptr,
                                                                    dcsc_row_ind,
                                                                    base,
                                                                    matrix_type,
                                                                    uplo,
                                                                    storage,
                                                                    &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    rocsparse_data_status data_status;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        dcsc_val,
                                                        dcsc_col_ptr,
                                                        dcsc_row_ind,
                                                        base,
                                                        matrix_type,
                                                        uplo,
                                                        storage,
                                                        &data_status,
                                                        dbuffer));
    CHECK_ROCSPARSE_DATA_ERROR(data_status);

    // Check passing shifting ptr array by large number
    host_vector<rocsparse_int> hcsc_col_ptr_shifted(hcsc_col_ptr);
    for(size_t i = 0; i < hcsc_col_ptr_shifted.size(); i++)
    {
        hcsc_col_ptr_shifted[i] += 10000;
    }
    device_vector<rocsparse_int> dcsc_col_ptr_shifted(hcsc_col_ptr_shifted);

    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        dcsc_val,
                                                        dcsc_col_ptr_shifted,
                                                        dcsc_row_ind,
                                                        base,
                                                        matrix_type,
                                                        uplo,
                                                        storage,
                                                        &data_status,
                                                        dbuffer));
    CHECK_ROCSPARSE_DATA_ERROR(data_status);

    if(nnz > 1 && n > 1)
    {
        rocsparse_int temp;
        T             temp_val;
        rocsparse_int rng;

        rocsparse_seedrand();

        rng  = random_generator_exact<rocsparse_int>(1, n - 1);
        temp = hcsc_col_ptr[rng];

        // Check passing matrix with invalid column ptr offset set number less than zero
        hcsc_col_ptr[rng] = -1;
        dcsc_col_ptr.transfer_from(hcsc_col_ptr);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsc_val,
                                                            dcsc_col_ptr,
                                                            dcsc_row_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_offset_ptr);

        // Restore offset pointer
        hcsc_col_ptr[rng] = temp;
        dcsc_col_ptr.transfer_from(hcsc_col_ptr);

        rng  = random_generator_exact<rocsparse_int>(0, nnz - 1);
        temp = hcsc_row_ind[rng];

        // Check passing matrix with row index set to number less than zero
        hcsc_row_ind[rng] = -1;
        dcsc_row_ind.transfer_from(hcsc_row_ind);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsc_val,
                                                            dcsc_col_ptr,
                                                            dcsc_row_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_index);

        // Check passing matrix with row index set to number greater than m - 1 + base
        hcsc_row_ind[rng] = (m - 1 + base) + 10;
        dcsc_row_ind.transfer_from(hcsc_row_ind);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsc_val,
                                                            dcsc_col_ptr,
                                                            dcsc_row_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_index);

        // Restore indices
        hcsc_row_ind[rng] = temp;
        dcsc_row_ind.transfer_from(hcsc_row_ind);

        // Check passing matrix with duplicate rows
        {
            // Find first column with non-zeros in it
            rocsparse_int col = -1;
            for(size_t i = 1; i < n; i++)
            {
                if(hcsc_col_ptr[i + 1] - hcsc_col_ptr[i] >= 2)
                {
                    col = i;
                    break;
                }
            }

            if(col != -1)
            {
                rocsparse_int index = hcsc_col_ptr[col] - base + 1;
                temp                = hcsc_row_ind[index];
                hcsc_row_ind[index] = hcsc_row_ind[hcsc_col_ptr[col] - base];
                dcsc_row_ind.transfer_from(hcsc_row_ind);

                CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                                    m,
                                                                    n,
                                                                    nnz,
                                                                    dcsc_val,
                                                                    dcsc_col_ptr,
                                                                    dcsc_row_ind,
                                                                    base,
                                                                    matrix_type,
                                                                    uplo,
                                                                    storage,
                                                                    &data_status,
                                                                    dbuffer));
                EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_duplicate_entry);

                // Restore indices
                hcsc_row_ind[index] = temp;
                dcsc_row_ind.transfer_from(hcsc_row_ind);
            }
        }

        // Check passing matrix with invalid fill
        if(matrix_type != rocsparse_matrix_type_general)
        {
            // Find first column with non-zeros in it
            rocsparse_int col = -1;
            for(size_t i = 1; i < n; i++)
            {
                if(hcsc_col_ptr[i + 1] - hcsc_col_ptr[i] > 0)
                {
                    col = i;
                    break;
                }
            }

            if(col != -1)
            {
                rocsparse_int index = (uplo == rocsparse_fill_mode_lower)
                                          ? hcsc_col_ptr[col + 1] - base - 1
                                          : hcsc_col_ptr[col] - base;
                temp                = hcsc_row_ind[index];

                hcsc_row_ind[index] = (uplo == rocsparse_fill_mode_lower) ? (m - 1 + base) : base;
                dcsc_row_ind.transfer_from(hcsc_row_ind);

                CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                                    m,
                                                                    n,
                                                                    nnz,
                                                                    dcsc_val,
                                                                    dcsc_col_ptr,
                                                                    dcsc_row_ind,
                                                                    base,
                                                                    matrix_type,
                                                                    uplo,
                                                                    storage,
                                                                    &data_status,
                                                                    dbuffer));
                EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_fill);

                // Restore indices
                hcsc_row_ind[index] = temp;
                dcsc_row_ind.transfer_from(hcsc_row_ind);
            }
        }

        rng      = random_generator_exact<rocsparse_int>(0, nnz - 1);
        temp_val = hcsc_val[rng];

        // Check passing matrix with inf value
        hcsc_val[rng] = rocsparse_inf<T>();
        dcsc_val.transfer_from(hcsc_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsc_val,
                                                            dcsc_col_ptr,
                                                            dcsc_row_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_inf);

        // Check passing matrix with nan value
        hcsc_val[rng] = rocsparse_nan<T>();
        dcsc_val.transfer_from(hcsc_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_csc<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcsc_val,
                                                            dcsc_col_ptr,
                                                            dcsc_row_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_nan);

        // Restore indices
        hcsc_val[rng] = temp_val;
        dcsc_val.transfer_from(hcsc_val);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_check_matrix_csc<T>,
                                                                      handle,
                                                                      m,
                                                                      n,
                                                                      nnz,
                                                                      dcsc_val,
                                                                      dcsc_col_ptr,
                                                                      dcsc_row_ind,
                                                                      base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      &data_status,
                                                                      dbuffer);

        double gbyte_count = check_matrix_csc_gbyte_count<T>(n, nnz);
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
    template void testing_check_matrix_csc_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_check_matrix_csc<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_check_matrix_csc_extra(const Arguments& arg) {}
