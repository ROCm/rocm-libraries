/* ************************************************************************
 * Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once
#ifndef TESTING_GEBSR2GEBSC_HPP
#define TESTING_GEBSR2GEBSC_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_gebsr2gebsc_bad_arg(void)
{
#if(!defined(CUDART_VERSION))

    hipsparseStatus_t              status;
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    static const size_t safe_size = 1;

    auto bsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto bsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
    int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
    T*   bsr_val     = (T*)bsr_val_managed.get();

    auto bsc_row_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsc_col_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto bsc_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    auto  buffer_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    void* buffer         = buffer_managed.get();

    int* bsc_row_ind = (int*)bsc_row_ind_managed.get();
    int* bsc_col_ptr = (int*)bsc_col_ptr_managed.get();
    T*   bsc_val     = (T*)bsc_val_managed.get();

    int local_ptr[2] = {0, 1};
    CHECK_HIP_ERROR(
        hipMemcpy(bsr_row_ptr, local_ptr, sizeof(int) * (safe_size + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(bsc_col_ptr, local_ptr, sizeof(int) * (safe_size + 1), hipMemcpyHostToDevice));

    size_t buffer_size;
    status = hipsparseXgebsr2gebsc_bufferSize<T>(nullptr,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 -1,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 -1,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 -1,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val is nullptr");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 nullptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 nullptr,
                                                 safe_size,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind is nullptr");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 -1,
                                                 safe_size,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 -1,
                                                 &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim is invalid");

    status = hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                 safe_size,
                                                 safe_size,
                                                 safe_size,
                                                 bsr_val,
                                                 bsr_row_ptr,
                                                 bsr_col_ind,
                                                 safe_size,
                                                 safe_size,
                                                 nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");

    // Test hipsparseXgebsr2gebsc()
    status = hipsparseXgebsr2gebsc<T>(nullptr,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      -1,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      -1,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      -1,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      nullptr,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: bsr_val is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      nullptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      nullptr,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      -1,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      -1,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim is invalid");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      nullptr,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsc_val is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      nullptr,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsc_row_ind is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      nullptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsc_col_ptr is nullptr");

    status = hipsparseXgebsr2gebsc<T>(handle,
                                      safe_size,
                                      safe_size,
                                      safe_size,
                                      bsr_val,
                                      bsr_row_ptr,
                                      bsr_col_ind,
                                      safe_size,
                                      safe_size,
                                      bsc_val,
                                      bsc_row_ind,
                                      bsc_col_ptr,
                                      HIPSPARSE_ACTION_NUMERIC,
                                      HIPSPARSE_INDEX_BASE_ZERO,
                                      nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer is invalid");
#endif
}

#define DEVICE_ALLOC(TYPE, NAME, SIZE)                                                            \
    auto  NAME##_managed = hipsparse_unique_ptr{device_malloc(sizeof(TYPE) * SIZE), device_free}; \
    TYPE* NAME           = (TYPE*)NAME##_managed.get()

template <typename T>
hipsparseStatus_t testing_gebsr2gebsc(Arguments argus)
{
    int                  m             = argus.M;
    int                  n             = argus.N;
    int                  row_block_dim = argus.row_block_dimA;
    int                  col_block_dim = argus.col_block_dimA;
    hipsparseAction_t    action        = argus.action;
    hipsparseIndexBase_t base          = argus.baseA;
    std::string          filename      = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    if(m == 0 || n == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // cusparse does not support m == 0 for csr2bsr
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    int mb = m * row_block_dim;
    int nb = n * col_block_dim;

    srand(12345ULL);

    // Host structures
    std::vector<int> hbsr_row_ptr;
    std::vector<int> hbsr_col_ind;
    std::vector<T>   hbsr_val;

    // Read or construct CSR matrix
    int nnzb = 0;
    if(!generate_csr_matrix(filename, mb, nb, nnzb, hbsr_row_ptr, hbsr_col_ind, hbsr_val, base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    m          = mb * row_block_dim;
    n          = nb * col_block_dim;
    size_t nnz = nnzb * row_block_dim * col_block_dim;

    // Now use the csr matrix as the symbolic for the gebsr matrix.
    hbsr_val.resize(nnz);
    for(size_t i = 0; i < nnz; ++i)
    {
        hbsr_val[i] = random_generator<T>();
    }

    DEVICE_ALLOC(int, dbsr_row_ptr, (mb + 1));
    DEVICE_ALLOC(int, dbsr_col_ind, nnzb);
    DEVICE_ALLOC(T, dbsr_val, (nnzb * row_block_dim * col_block_dim));

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dbsr_val,
                              hbsr_val.data(),
                              sizeof(T) * nnzb * row_block_dim * col_block_dim,
                              hipMemcpyHostToDevice));

    // Obtain required buffer size (from host)
    size_t buffer_size;
    CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsc_bufferSize<T>(handle,
                                                              mb,
                                                              nb,
                                                              nnzb,
                                                              dbsr_val,
                                                              dbsr_row_ptr,
                                                              dbsr_col_ind,
                                                              row_block_dim,
                                                              col_block_dim,
                                                              &buffer_size));

    // Allocate the buffer size.
    auto  dbuffer_managed = hipsparse_unique_ptr{device_malloc(buffer_size), device_free};
    void* dbuffer         = dbuffer_managed.get();

    DEVICE_ALLOC(int, dbsc_row_ind, nnzb);
    DEVICE_ALLOC(int, dbsc_col_ptr, (nb + 1));
    DEVICE_ALLOC(T, dbsc_val, (nnzb * row_block_dim * col_block_dim));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsc<T>(handle,
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       dbsr_val,
                                                       dbsr_row_ptr,
                                                       dbsr_col_ind,
                                                       row_block_dim,
                                                       col_block_dim,
                                                       dbsc_val,
                                                       dbsc_row_ind,
                                                       dbsc_col_ptr,
                                                       action,
                                                       base,
                                                       dbuffer));
        // Transfer to host.
        std::vector<int> hbsc_from_device_row_ind(nnzb);
        std::vector<int> hbsc_from_device_col_ptr(nb + 1);
        std::vector<T>   hbsc_from_device_val(nnzb * row_block_dim * col_block_dim);
        CHECK_HIP_ERROR(hipMemcpy(hbsc_from_device_col_ptr.data(),
                                  dbsc_col_ptr,
                                  sizeof(int) * (nb + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsc_from_device_row_ind.data(),
                                  dbsc_row_ind,
                                  sizeof(int) * nnzb,
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsc_from_device_val.data(),
                                  dbsc_val,
                                  sizeof(T) * nnzb * row_block_dim * col_block_dim,
                                  hipMemcpyDeviceToHost));

        // Allocate host bsc matrix.
        std::vector<int> hbsc_row_ind(nnzb);
        std::vector<int> hbsc_col_ptr(nb + 1);
        std::vector<T>   hbsc_val(nnzb * row_block_dim * col_block_dim);
        host_gebsr_to_gebsc<T>(mb,
                               nb,
                               nnzb,
                               hbsr_row_ptr,
                               hbsr_col_ind,
                               hbsr_val,
                               row_block_dim,
                               col_block_dim,
                               hbsc_row_ind,
                               hbsc_col_ptr,
                               hbsc_val,
                               action,
                               base);

        unit_check_general(1, nb + 1, 1, hbsc_from_device_col_ptr.data(), hbsc_col_ptr.data());
        unit_check_general(1, nnzb, 1, hbsc_from_device_row_ind.data(), hbsc_row_ind.data());
        if(action == HIPSPARSE_ACTION_NUMERIC)
        {
            unit_check_general(1,
                               nnzb * row_block_dim * col_block_dim,
                               1,
                               hbsc_from_device_val.data(),
                               hbsc_val.data());
        }
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsc<T>(handle,
                                                           mb,
                                                           nb,
                                                           nnzb,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           row_block_dim,
                                                           col_block_dim,
                                                           dbsc_val,
                                                           dbsc_row_ind,
                                                           dbsc_col_ptr,
                                                           action,
                                                           base,
                                                           dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsc<T>(handle,
                                                           mb,
                                                           nb,
                                                           nnzb,
                                                           dbsr_val,
                                                           dbsr_row_ptr,
                                                           dbsr_col_ind,
                                                           row_block_dim,
                                                           col_block_dim,
                                                           dbsc_val,
                                                           dbsc_row_ind,
                                                           dbsc_col_ptr,
                                                           action,
                                                           base,
                                                           dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count
            = gebsr2gebsc_gbyte_count<T>(mb, nb, nnzb, row_block_dim, col_block_dim, action);
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::Mb,
                            mb,
                            display_key_t::Nb,
                            nb,
                            display_key_t::nnzb,
                            nnzb,
                            display_key_t::row_block_dim,
                            row_block_dim,
                            display_key_t::col_block_dim,
                            col_block_dim,
                            display_key_t::action,
                            hipsparse_action2string(action),
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GEBSR2GEBSC_HPP
