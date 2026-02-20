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
void testing_check_matrix_gebsc_bad_arg(const Arguments& arg)
{
    rocsparse_int          mb            = 100;
    rocsparse_int          nb            = 100;
    rocsparse_int          nnzb          = 100;
    rocsparse_int          row_block_dim = 2;
    rocsparse_int          col_block_dim = 2;
    rocsparse_direction    dir           = rocsparse_direction_row;
    rocsparse_index_base   idx_base      = rocsparse_index_base_zero;
    rocsparse_matrix_type  matrix_type   = rocsparse_matrix_type_general;
    rocsparse_fill_mode    uplo          = rocsparse_fill_mode_lower;
    rocsparse_storage_mode storage       = rocsparse_storage_mode_sorted;

    // Create rocsparse handle
    rocsparse_local_handle local_handle;
    rocsparse_handle       handle = local_handle;

    const rocsparse_int*   bsc_col_ptr = (const rocsparse_int*)0x4;
    const rocsparse_int*   bsc_row_ind = (const rocsparse_int*)0x4;
    const T*               bsc_val     = (const T*)0x4;
    void*                  temp_buffer = (void*)0x4;
    size_t*                buffer_size = (size_t*)0x4;
    rocsparse_data_status* data_status = (rocsparse_data_status*)0x4;

    int       nargs_to_exclude   = 3;
    const int args_to_exclude[3] = {7, 8, 9};
#define PARAMS_BUFFER_SIZE                                                                      \
    handle, dir, mb, nb, nnzb, row_block_dim, col_block_dim, bsc_val, bsc_col_ptr, bsc_row_ind, \
        idx_base, matrix_type, uplo, storage, buffer_size
#define PARAMS                                                                                  \
    handle, dir, mb, nb, nnzb, row_block_dim, col_block_dim, bsc_val, bsc_col_ptr, bsc_row_ind, \
        idx_base, matrix_type, uplo, storage, data_status, temp_buffer
    select_bad_arg_analysis(rocsparse_check_matrix_gebsc_buffer_size<T>,
                            nargs_to_exclude,
                            args_to_exclude,
                            PARAMS_BUFFER_SIZE);
    select_bad_arg_analysis(
        rocsparse_check_matrix_gebsc<T>, nargs_to_exclude, args_to_exclude, PARAMS);

    // row_block_dim == 0
    row_block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_check_matrix_gebsc_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_check_matrix_gebsc<T>(PARAMS), rocsparse_status_invalid_size);
    row_block_dim = 2;

    // col_block_dim == 0
    col_block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_check_matrix_gebsc_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_check_matrix_gebsc<T>(PARAMS), rocsparse_status_invalid_size);
    col_block_dim = 2;

    // row_block_dim == 0 && col_block_dim == 0
    row_block_dim = 0;
    col_block_dim = 0;
    EXPECT_ROCSPARSE_STATUS(rocsparse_check_matrix_gebsc_buffer_size<T>(PARAMS_BUFFER_SIZE),
                            rocsparse_status_invalid_size);
    EXPECT_ROCSPARSE_STATUS(rocsparse_check_matrix_gebsc<T>(PARAMS), rocsparse_status_invalid_size);
    row_block_dim = 2;
    col_block_dim = 2;
#undef PARAMS_BUFFER_SIZE
#undef PARAMS
}

template <typename T>
void testing_check_matrix_gebsc(const Arguments& arg)
{
    rocsparse_direction    direction     = arg.direction;
    rocsparse_int          m             = arg.M;
    rocsparse_int          n             = arg.N;
    rocsparse_int          row_block_dim = arg.row_block_dimA;
    rocsparse_int          col_block_dim = arg.col_block_dimA;
    rocsparse_index_base   base          = arg.baseA;
    rocsparse_matrix_type  matrix_type   = arg.matrix_type;
    rocsparse_fill_mode    uplo          = arg.uplo;
    rocsparse_storage_mode storage       = arg.storage;

    rocsparse_int mb = (row_block_dim > 0) ? (m + row_block_dim - 1) / row_block_dim : 0;
    rocsparse_int nb = (col_block_dim > 0) ? (n + col_block_dim - 1) / col_block_dim : 0;

    // Create rocsparse handle
    rocsparse_local_handle handle;

    rocsparse_matrix_factory<T> matrix_factory(arg);

    // Allocate host memory for BSC matrix
    host_vector<rocsparse_int> hbsc_col_ptr;
    host_vector<rocsparse_int> hbsc_row_ind;
    host_vector<T>             hbsc_val;

    // Generate (or load from file) BSC matrix
    rocsparse_int nnzb;
    matrix_factory.init_gebsc(hbsc_col_ptr,
                              hbsc_row_ind,
                              hbsc_val,
                              direction,
                              mb,
                              nb,
                              nnzb,
                              row_block_dim,
                              col_block_dim,
                              base);

    // BSC matrix on device
    device_vector<rocsparse_int> dbsc_col_ptr(hbsc_col_ptr);
    device_vector<rocsparse_int> dbsc_row_ind(hbsc_row_ind);
    device_vector<T>             dbsc_val(hbsc_val);

    size_t buffer_size;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc_buffer_size<T>(handle,
                                                                      direction,
                                                                      mb,
                                                                      nb,
                                                                      nnzb,
                                                                      row_block_dim,
                                                                      col_block_dim,
                                                                      dbsc_val,
                                                                      dbsc_col_ptr,
                                                                      dbsc_row_ind,
                                                                      base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      &buffer_size));

    void* dbuffer;
    CHECK_HIP_ERROR(rocsparse_hipMalloc(&dbuffer, buffer_size));

    rocsparse_data_status data_status;
    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                          direction,
                                                          mb,
                                                          nb,
                                                          nnzb,
                                                          row_block_dim,
                                                          col_block_dim,
                                                          dbsc_val,
                                                          dbsc_col_ptr,
                                                          dbsc_row_ind,
                                                          base,
                                                          matrix_type,
                                                          uplo,
                                                          storage,
                                                          &data_status,
                                                          dbuffer));
    CHECK_ROCSPARSE_DATA_ERROR(data_status);

    // Check passing shifting ptr array by large number
    host_vector<rocsparse_int> hbsc_col_ptr_shifted(hbsc_col_ptr);
    for(size_t i = 0; i < hbsc_col_ptr_shifted.size(); i++)
    {
        hbsc_col_ptr_shifted[i] += 10000;
    }
    device_vector<rocsparse_int> dbsc_col_ptr_shifted(hbsc_col_ptr_shifted);

    CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                          direction,
                                                          mb,
                                                          nb,
                                                          nnzb,
                                                          row_block_dim,
                                                          col_block_dim,
                                                          dbsc_val,
                                                          dbsc_col_ptr,
                                                          dbsc_row_ind,
                                                          base,
                                                          matrix_type,
                                                          uplo,
                                                          storage,
                                                          &data_status,
                                                          dbuffer));
    CHECK_ROCSPARSE_DATA_ERROR(data_status);

    if(nnzb > 1 && nb > 1)
    {
        rocsparse_int temp;
        T             temp_val;
        rocsparse_int rng;

        rocsparse_seedrand();

        rng  = random_generator_exact<rocsparse_int>(1, nb - 1);
        temp = hbsc_col_ptr[rng];

        // Check passing matrix with invalid column ptr offset set number less than zero
        hbsc_col_ptr[rng] = -1;
        dbsc_col_ptr.transfer_from(hbsc_col_ptr);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                              direction,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              row_block_dim,
                                                              col_block_dim,
                                                              dbsc_val,
                                                              dbsc_col_ptr,
                                                              dbsc_row_ind,
                                                              base,
                                                              matrix_type,
                                                              uplo,
                                                              storage,
                                                              &data_status,
                                                              dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_offset_ptr);

        // Restore offset pointer
        hbsc_col_ptr[rng] = temp;
        dbsc_col_ptr.transfer_from(hbsc_col_ptr);

        rng  = random_generator_exact<rocsparse_int>(0, nnzb - 1);
        temp = hbsc_row_ind[rng];

        // Check passing matrix with row index set to number less than zero
        hbsc_row_ind[rng] = -1;
        dbsc_row_ind.transfer_from(hbsc_row_ind);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                              direction,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              row_block_dim,
                                                              col_block_dim,
                                                              dbsc_val,
                                                              dbsc_col_ptr,
                                                              dbsc_row_ind,
                                                              base,
                                                              matrix_type,
                                                              uplo,
                                                              storage,
                                                              &data_status,
                                                              dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_index);

        // Check passing matrix with row index set to number greater than mb - 1 + base
        hbsc_row_ind[rng] = (mb - 1 + base) + 10;
        dbsc_row_ind.transfer_from(hbsc_row_ind);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                              direction,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              row_block_dim,
                                                              col_block_dim,
                                                              dbsc_val,
                                                              dbsc_col_ptr,
                                                              dbsc_row_ind,
                                                              base,
                                                              matrix_type,
                                                              uplo,
                                                              storage,
                                                              &data_status,
                                                              dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_index);

        // Restore indices
        hbsc_row_ind[rng] = temp;
        dbsc_row_ind.transfer_from(hbsc_row_ind);

        // Check passing matrix with duplicate rows
        {
            // Find first column with non-zeros in it
            rocsparse_int col = -1;
            for(size_t i = 1; i < nb; i++)
            {
                if(hbsc_col_ptr[i + 1] - hbsc_col_ptr[i] >= 2)
                {
                    col = i;
                    break;
                }
            }

            if(col != -1)
            {
                rocsparse_int index = hbsc_col_ptr[col] - base + 1;
                temp                = hbsc_row_ind[index];
                hbsc_row_ind[index] = hbsc_row_ind[hbsc_col_ptr[col] - base];
                dbsc_row_ind.transfer_from(hbsc_row_ind);

                CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                                      direction,
                                                                      mb,
                                                                      nb,
                                                                      nnzb,
                                                                      row_block_dim,
                                                                      col_block_dim,
                                                                      dbsc_val,
                                                                      dbsc_col_ptr,
                                                                      dbsc_row_ind,
                                                                      base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      &data_status,
                                                                      dbuffer));
                EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_duplicate_entry);

                // Restore indices
                hbsc_row_ind[index] = temp;
                dbsc_row_ind.transfer_from(hbsc_row_ind);
            }
        }

        // Check passing matrix with invalid fill
        if(matrix_type != rocsparse_matrix_type_general)
        {
            // Find first column with non-zeros in it
            rocsparse_int col = -1;
            for(size_t i = 1; i < nb; i++)
            {
                if(hbsc_col_ptr[i + 1] - hbsc_col_ptr[i] > 0)
                {
                    col = i;
                    break;
                }
            }

            if(col != -1)
            {
                rocsparse_int index = (uplo == rocsparse_fill_mode_lower)
                                          ? hbsc_col_ptr[col + 1] - base - 1
                                          : hbsc_col_ptr[col] - base;
                temp                = hbsc_row_ind[index];

                hbsc_row_ind[index] = (uplo == rocsparse_fill_mode_lower) ? (mb - 1 + base) : base;
                dbsc_row_ind.transfer_from(hbsc_row_ind);

                CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                                      direction,
                                                                      mb,
                                                                      nb,
                                                                      nnzb,
                                                                      row_block_dim,
                                                                      col_block_dim,
                                                                      dbsc_val,
                                                                      dbsc_col_ptr,
                                                                      dbsc_row_ind,
                                                                      base,
                                                                      matrix_type,
                                                                      uplo,
                                                                      storage,
                                                                      &data_status,
                                                                      dbuffer));
                EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_invalid_fill);

                // Restore indices
                hbsc_row_ind[index] = temp;
                dbsc_row_ind.transfer_from(hbsc_row_ind);
            }
        }

        rng      = random_generator_exact<rocsparse_int>(0, nnzb - 1);
        temp_val = hbsc_val[rng];

        // Check passing matrix with inf value
        hbsc_val[rng] = rocsparse_inf<T>();
        dbsc_val.transfer_from(hbsc_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                              direction,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              row_block_dim,
                                                              col_block_dim,
                                                              dbsc_val,
                                                              dbsc_col_ptr,
                                                              dbsc_row_ind,
                                                              base,
                                                              matrix_type,
                                                              uplo,
                                                              storage,
                                                              &data_status,
                                                              dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_inf);

        // Check passing matrix with nan value
        hbsc_val[rng] = rocsparse_nan<T>();
        dbsc_val.transfer_from(hbsc_val);

        CHECK_ROCSPARSE_ERROR(rocsparse_check_matrix_gebsc<T>(handle,
                                                              direction,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              row_block_dim,
                                                              col_block_dim,
                                                              dbsc_val,
                                                              dbsc_col_ptr,
                                                              dbsc_row_ind,
                                                              base,
                                                              matrix_type,
                                                              uplo,
                                                              storage,
                                                              &data_status,
                                                              dbuffer));
        EXPECT_ROCSPARSE_DATA_STATUS(data_status, rocsparse_data_status_nan);

        // Restore indices
        hbsc_val[rng] = temp_val;
        dbsc_val.transfer_from(hbsc_val);
    }

    if(arg.timing)
    {

        const double gpu_time_used
            = rocsparse_clients::run_benchmark(arg,
                                               rocsparse_check_matrix_gebsc<T>,
                                               handle,
                                               direction,
                                               mb,
                                               nb,
                                               nnzb,
                                               row_block_dim,
                                               col_block_dim,
                                               dbsc_val,
                                               dbsc_col_ptr,
                                               dbsc_row_ind,
                                               base,
                                               matrix_type,
                                               uplo,
                                               storage,
                                               &data_status,
                                               dbuffer);

        double gbyte_count
            = check_matrix_gebsc_gbyte_count<T>(nb, nnzb, row_block_dim, col_block_dim);
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::Mb,
                            mb,
                            display_key_t::Nb,
                            nb,
                            display_key_t::nnzb,
                            nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    CHECK_HIP_ERROR(rocsparse_hipFree(dbuffer));
}

#define INSTANTIATE(TYPE)                                                         \
    template void testing_check_matrix_gebsc_bad_arg<TYPE>(const Arguments& arg); \
    template void testing_check_matrix_gebsc<TYPE>(const Arguments& arg)
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
void testing_check_matrix_gebsc_extra(const Arguments& arg) {}
