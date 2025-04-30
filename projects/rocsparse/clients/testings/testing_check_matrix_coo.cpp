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
void testing_check_matrix_coo_bad_arg(const Arguments& arg)
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

    // Allocate memory on device
    const rocsparse_int*   coo_row_ind = (const rocsparse_int*)0x4;
    const rocsparse_int*   coo_col_ind = (const rocsparse_int*)0x4;
    const T*               coo_val     = (const T*)0x4;
    void*                  temp_buffer = (void*)0x4;
    size_t*                buffer_size = (size_t*)0x4;
    rocsparse_data_status* data_status = (rocsparse_data_status*)0x4;

    int       nargs_to_exclude   = 3;
    const int args_to_exclude[3] = {4, 5, 6};
#define PARAMS_BUFFER_SIZE                                                                      \
    handle, m, n, nnz, coo_val, coo_row_ind, coo_col_ind, idx_base, matrix_type, uplo, storage, \
        buffer_size
#define PARAMS                                                                                  \
    handle, m, n, nnz, coo_val, coo_row_ind, coo_col_ind, idx_base, matrix_type, uplo, storage, \
        data_status, temp_buffer
    select_bad_arg_analysis(rocsparse_check_matrix_coo_buffer_size<T>,
                            nargs_to_exclude,
                            args_to_exclude,
                            PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(
        rocsparse_check_matrix_coo<T>, nargs_to_exclude, args_to_exclude, PARAMS);
#undef PARAMS_BUFFER_SIZE
#undef PARAMS
}

template <typename T>
void testing_check_matrix_coo(const Arguments& arg)
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

    // Allocate host memory for COO matrix
    host_vector<rocsparse_int> hcoo_row_ind;
    host_vector<rocsparse_int> hcoo_col_ind;
    host_vector<T>             hcoo_val;

    // Generate (or load from file) COO matrix
    int64_t coo_nnz;
    matrix_factory.init_coo(hcoo_row_ind, hcoo_col_ind, hcoo_val, m, n, coo_nnz, base);

    rocsparse_int nnz = rocsparse_convert_to_int(coo_nnz);

    // COO matrix on device
    device_vector<rocsparse_int> dcoo_row_ind(hcoo_row_ind);
    device_vector<rocsparse_int> dcoo_col_ind(hcoo_col_ind);
    device_vector<T>             dcoo_val(hcoo_val);

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_coo_buffer_size<T>(handle,
                                                                    m,
                                                                    n,
                                                                    nnz,
                                                                    dcoo_val,
                                                                    dcoo_row_ind,
                                                                    dcoo_col_ind,
                                                                    base,
                                                                    matrix_type,
                                                                    uplo,
                                                                    storage,
                                                                    &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    rocsparse_data_status data_status;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_coo<T>(handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        dcoo_val,
                                                        dcoo_row_ind,
                                                        dcoo_col_ind,
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

        rng  = random_generator_exact<rocsparse_int>(1, nnz - 1);
        temp = hcoo_row_ind[rng];

        // Check passing matrix with invalid row index set to number less than zero
        hcoo_row_ind[rng] = -1;
        dcoo_row_ind.transfer_from(hcoo_row_ind);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_coo<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcoo_val,
                                                            dcoo_row_ind,
                                                            dcoo_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_index);

        // Restore row indices
        hcoo_row_ind[rng] = temp;
        dcoo_row_ind.transfer_from(hcoo_row_ind);

        rng  = random_generator_exact<rocsparse_int>(0, nnz - 1);
        temp = hcoo_col_ind[rng];

        // Check passing matrix with column index set to number less than zero
        hcoo_col_ind[rng] = -1;
        dcoo_col_ind.transfer_from(hcoo_col_ind);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_coo<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcoo_val,
                                                            dcoo_row_ind,
                                                            dcoo_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_index);

        // Restore row indices
        hcoo_col_ind[rng] = temp;
        dcoo_col_ind.transfer_from(hcoo_col_ind);

        // Check passing matrix with invalid fill
        if(matrix_type != rocsparse_matrix_type_general)
        {
            rocsparse_int index = random_generator_exact<rocsparse_int>(1, nnz - 1);
            rocsparse_int row   = hcoo_row_ind[index] - base;

            if(row > 0 && row < n - 1)
            {
                if(uplo == rocsparse_fill_mode_lower)
                {
                    // Find index of last column in row
                    while((index + 1) < nnz && row == (hcoo_row_ind[index + 1] - base))
                    {
                        index++;
                    }
                }
                else
                {
                    // Find index of first column in row
                    while((index - 1) >= 0 && row == (hcoo_row_ind[index - 1] - base))
                    {
                        index--;
                    }
                }

                temp = hcoo_col_ind[index];

                hcoo_col_ind[index] = (uplo == rocsparse_fill_mode_lower) ? (n - 1 + base) : base;
                dcoo_col_ind.transfer_from(hcoo_col_ind);

                CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_coo<T>(handle,
                                                                    m,
                                                                    n,
                                                                    nnz,
                                                                    dcoo_val,
                                                                    dcoo_row_ind,
                                                                    dcoo_col_ind,
                                                                    base,
                                                                    matrix_type,
                                                                    uplo,
                                                                    storage,
                                                                    &data_status,
                                                                    dbuffer));
                EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_fill);

                // Restore indices
                hcoo_col_ind[index] = temp;
                dcoo_col_ind.transfer_from(hcoo_col_ind);
            }
        }

        rng      = random_generator_exact<rocsparse_int>(0, nnz - 1);
        temp_val = hcoo_val[rng];

        // Check passing matrix with inf value
        hcoo_val[rng] = rocsparse_inf<T>();
        dcoo_val.transfer_from(hcoo_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_coo<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcoo_val,
                                                            dcoo_row_ind,
                                                            dcoo_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_inf);

        // Check passing matrix with nan value
        hcoo_val[rng] = rocsparse_nan<T>();
        dcoo_val.transfer_from(hcoo_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_coo<T>(handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            dcoo_val,
                                                            dcoo_row_ind,
                                                            dcoo_col_ind,
                                                            base,
                                                            matrix_type,
                                                            uplo,
                                                            storage,
                                                            &data_status,
                                                            dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_nan);

        // Restore indices
        hcoo_val[rng] = temp_val;
        dcoo_val.transfer_from(hcoo_val);
    }

    if(arg.timing)
    {

        const double gpu_time_used = rocsparse_clients::run_benchmark(arg,
                                                                      rocsparse_check_matrix_coo<T>,
                                                                      handle,
                                                                      m,
                                                                      n,
                                                                      nnz,
                                                                      dcoo_val,
                                                                      dcoo_row_ind,
                                                                      dcoo_col_ind,
                                                                      base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      &data_status,
                                                                      dbuffer);

        double gbyte_count = check_matrix_coo_gbyte_count<T>(nnz);
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
    template void testing_check_matrix_coo_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_check_matrix_coo<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_check_matrix_coo_extra(const Arguments& arg) {}
