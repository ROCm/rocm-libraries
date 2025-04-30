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
#ifndef TESTING_CSR2GEBSR_HPP
#define TESTING_CSR2GEBSR_HPP

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
void testing_csr2gebsr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))

    hipsparseStatus_t              status;
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    hipsparseIndexBase_t csr_idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t bsr_idx_base = HIPSPARSE_INDEX_BASE_ZERO;

    static const size_t safe_size = 1;

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();

    auto bsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto bsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    auto  buffer_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    void* buffer         = buffer_managed.get();

    int* bsr_row_ptr = (int*)bsr_row_ptr_managed.get();
    int* bsr_col_ind = (int*)bsr_col_ind_managed.get();
    T*   bsr_val     = (T*)bsr_val_managed.get();

    { //
        int local_ptr[2] = {0, 1};
        CHECK_HIP_ERROR(
            hipMemcpy(csr_row_ptr, local_ptr, sizeof(int) * (1 + 1), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(
            hipMemcpy(bsr_row_ptr, local_ptr, sizeof(int) * (1 + 1), hipMemcpyHostToDevice));
    } //

    std::unique_ptr<descr_struct> unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t           csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct> unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t           bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    //
    // Declaration of arguments.
    //
    hipsparseDirection_t arg_direction;
    int                  arg_m;
    int                  arg_n;
    hipsparseMatDescr_t  arg_csr_descr;
    const T*             arg_csr_val;
    const int*           arg_csr_row_ptr;
    const int*           arg_csr_col_ind;
    hipsparseMatDescr_t  arg_bsr_descr;
    T*                   arg_bsr_val;
    int*                 arg_bsr_row_ptr;
    int*                 arg_bsr_col_ind;
    int                  arg_row_block_dim;
    int                  arg_col_block_dim;
    void*                arg_p_buffer;
    int*                 arg_bsr_nnz_devhost;
    size_t*              arg_p_buffer_size;

    int    hbsr_nnzb;
    size_t buffer_size;

    //
    // Macro to set arguments.
    //
#define ARGSET                                     \
    arg_direction       = HIPSPARSE_DIRECTION_ROW; \
    arg_m               = safe_size;               \
    arg_n               = safe_size;               \
    arg_csr_descr       = csr_descr;               \
    arg_csr_val         = (T*)csr_val;             \
    arg_csr_row_ptr     = csr_row_ptr;             \
    arg_csr_col_ind     = csr_col_ind;             \
    arg_bsr_descr       = bsr_descr;               \
    arg_bsr_val         = (T*)bsr_val;             \
    arg_bsr_row_ptr     = bsr_row_ptr;             \
    arg_bsr_col_ind     = bsr_col_ind;             \
    arg_row_block_dim   = safe_size;               \
    arg_col_block_dim   = safe_size;               \
    arg_p_buffer        = (void*)((T*)buffer);     \
    arg_bsr_nnz_devhost = &hbsr_nnzb;              \
    arg_p_buffer_size   = &buffer_size

    //
    // BUFFER_SIZE ############
    //
#define CALL_ARG_BUFFER_SIZE                                                                   \
    arg_direction, arg_m, arg_n, arg_csr_descr, arg_csr_val, arg_csr_row_ptr, arg_csr_col_ind, \
        arg_row_block_dim, arg_col_block_dim, arg_p_buffer_size

#define CALL_BUFFER_SIZE hipsparseXcsr2gebsr_bufferSize(handle, CALL_ARG_BUFFER_SIZE)

    {
        ARGSET;
        status = hipsparseXcsr2gebsr_bufferSize(nullptr, CALL_ARG_BUFFER_SIZE);
        verify_hipsparse_status_invalid_handle(status);
    }

    {
        ARGSET;
        arg_m  = -1;
        status = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_size(status, "Error: m is invalid");
    }

    {
        ARGSET;
        arg_n  = -1;
        status = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_size(status, "Error: n is invalid");
    }

    {
        ARGSET;
        arg_csr_descr = nullptr;
        status        = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    {
        ARGSET;
        arg_csr_val = nullptr;
        status      = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    {
        ARGSET;
        arg_csr_row_ptr = nullptr;
        status          = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    {
        ARGSET;
        arg_csr_col_ind = nullptr;
        status          = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    {
        ARGSET;
        arg_row_block_dim = -1;
        status            = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_size(status, "Error: row_block_dim is invalid");
    }

    {
        ARGSET;
        arg_col_block_dim = -1;
        status            = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_size(status, "Error: col_block_dim is invalid");
    }

    {
        ARGSET;
        arg_p_buffer_size = nullptr;
        status            = CALL_BUFFER_SIZE;
        verify_hipsparse_status_invalid_pointer(status, "Error: p_buffer_size is nullptr");
    }

#undef CALL_ARG_BUFFER_SIZE
#undef CALL_BUFFER_SIZE

    //
    // NNZ ############
    //
#define CALL_ARG_NNZ                                                                             \
    arg_direction, arg_m, arg_n, arg_csr_descr, arg_csr_row_ptr, arg_csr_col_ind, arg_bsr_descr, \
        arg_bsr_row_ptr, arg_row_block_dim, arg_col_block_dim, arg_bsr_nnz_devhost, arg_p_buffer

#define CALL_NNZ hipsparseXcsr2gebsrNnz(handle, CALL_ARG_NNZ)

    {
        ARGSET;
        status = hipsparseXcsr2gebsrNnz(nullptr, CALL_ARG_NNZ);
        verify_hipsparse_status_invalid_handle(status);
    }

    {
        ARGSET;
        arg_m  = -1;
        status = CALL_NNZ;
        verify_hipsparse_status_invalid_size(status, "Error: m is invalid");
    }

    {
        ARGSET;
        arg_n  = -1;
        status = CALL_NNZ;
        verify_hipsparse_status_invalid_size(status, "Error: n is invalid");
    }

    {
        ARGSET;
        arg_csr_descr = nullptr;
        status        = CALL_NNZ;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    {
        ARGSET;
        arg_csr_row_ptr = nullptr;
        status          = CALL_NNZ;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    {
        ARGSET;
        arg_bsr_descr = nullptr;
        status        = CALL_NNZ;
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_descr is nullptr");
    }

    {
        ARGSET;
        arg_bsr_row_ptr = nullptr;
        status          = CALL_NNZ;
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");
    }

    {
        ARGSET;
        arg_row_block_dim = -1;
        status            = CALL_NNZ;
        verify_hipsparse_status_invalid_size(status, "Error: row_block_dim is invalid");
    }

    {
        ARGSET;
        arg_col_block_dim = -1;
        status            = CALL_NNZ;
        verify_hipsparse_status_invalid_size(status, "Error: col_block_dim is invalid");
    }

    {
        ARGSET;
        arg_bsr_nnz_devhost = nullptr;
        status              = CALL_NNZ;
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_nnz_devhost is nullptr");
    }

#undef CALL_NNZ
#undef CALL_ARG_NNZ

#undef ARGSET
#define ARGSET                                   \
    arg_direction     = HIPSPARSE_DIRECTION_ROW; \
    arg_m             = safe_size;               \
    arg_n             = safe_size;               \
    arg_csr_descr     = csr_descr;               \
    arg_csr_val       = (T*)csr_val;             \
    arg_csr_row_ptr   = csr_row_ptr;             \
    arg_csr_col_ind   = csr_col_ind;             \
    arg_bsr_descr     = bsr_descr;               \
    arg_bsr_val       = (T*)bsr_val;             \
    arg_bsr_row_ptr   = bsr_row_ptr;             \
    arg_bsr_col_ind   = bsr_col_ind;             \
    arg_row_block_dim = safe_size;               \
    arg_col_block_dim = safe_size;               \
    arg_p_buffer      = (void*)((T*)buffer);     \
    arg_p_buffer_size = &buffer_size

#define CALL_ARG_FUNC                                                                          \
    arg_direction, arg_m, arg_n, arg_csr_descr, arg_csr_val, arg_csr_row_ptr, arg_csr_col_ind, \
        arg_bsr_descr, arg_bsr_val, arg_bsr_row_ptr, arg_bsr_col_ind, arg_row_block_dim,       \
        arg_col_block_dim, arg_p_buffer

#define CALL_FUNC hipsparseXcsr2gebsr<T>(handle, CALL_ARG_FUNC)

    {
        ARGSET;
        status = hipsparseXcsr2gebsr(nullptr, CALL_ARG_FUNC);
        verify_hipsparse_status_invalid_handle(status);
    }

    {
        ARGSET;
        arg_m  = -1;
        status = CALL_FUNC;
        verify_hipsparse_status_invalid_size(status, "Error: m is invalid");
    }

    {
        ARGSET;
        arg_n  = -1;
        status = CALL_FUNC;
        verify_hipsparse_status_invalid_size(status, "Error: n is invalid");
    }

    {
        ARGSET;
        arg_csr_descr = nullptr;
        status        = CALL_FUNC;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_descr is nullptr");
    }

    {
        ARGSET;
        arg_csr_val = nullptr;
        status      = CALL_FUNC;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");
    }

    {
        ARGSET;
        arg_csr_row_ptr = nullptr;
        status          = CALL_FUNC;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");
    }

    {
        ARGSET;
        arg_csr_col_ind = nullptr;
        status          = CALL_FUNC;
        verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");
    }

    {
        ARGSET;
        arg_bsr_descr = nullptr;
        status        = CALL_FUNC;
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_descr is nullptr");
    }

    {
        ARGSET;
        arg_bsr_val = nullptr;
        status      = CALL_FUNC;
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val is nullptr");
    }

    {
        ARGSET;
        arg_bsr_row_ptr = nullptr;
        status          = CALL_FUNC;
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr is nullptr");
    }
    {
        ARGSET;
        arg_bsr_col_ind = nullptr;
        status          = CALL_FUNC;
        verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind is nullptr");
    }

    {
        ARGSET;
        arg_row_block_dim = -1;
        status            = CALL_FUNC;
        verify_hipsparse_status_invalid_size(status, "Error: row_block_dim is invalid");
    }

    {
        ARGSET;
        arg_col_block_dim = -1;
        status            = CALL_FUNC;
        verify_hipsparse_status_invalid_size(status, "Error: col_block_dim is invalid");
    }

#undef CALL_FUNC
#undef CALL_ARG_FUNC

#undef ARGSET
#endif
}

template <typename T>
hipsparseStatus_t testing_csr2gebsr(Arguments argus)
{
    int                  m             = argus.M;
    int                  n             = argus.N;
    hipsparseIndexBase_t csr_idx_base  = argus.baseA;
    hipsparseIndexBase_t bsr_idx_base  = argus.baseB;
    hipsparseDirection_t dir           = argus.dirA;
    int                  row_block_dim = argus.row_block_dimA;
    int                  col_block_dim = argus.col_block_dimA;
    std::string          filename      = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_csr_descr(new descr_struct);
    hipsparseMatDescr_t            csr_descr = unique_ptr_csr_descr->descr;
    std::unique_ptr<descr_struct>  unique_ptr_bsr_descr(new descr_struct);
    hipsparseMatDescr_t            bsr_descr = unique_ptr_bsr_descr->descr;

    hipsparseSetMatIndexBase(csr_descr, csr_idx_base);
    hipsparseSetMatIndexBase(bsr_descr, bsr_idx_base);

    if(row_block_dim == 1 || m == 0 || n == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // Do not test cusparse with block dim 1
        return HIPSPARSE_STATUS_SUCCESS;
#endif
    }

    srand(12345ULL);

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Read or construct CSR matrix
    int nnz = 0;
    if(!generate_csr_matrix(
           filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, csr_idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    int mb = (m + row_block_dim - 1) / row_block_dim;
    int nb = (n + col_block_dim - 1) / col_block_dim;

    // Allocate memory on the device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();
    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();

    // Copy data from host to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    size_t buffer_size;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2gebsr_bufferSize(handle,
                                                         dir,
                                                         m,
                                                         n,
                                                         csr_descr,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         row_block_dim,
                                                         col_block_dim,
                                                         &buffer_size));

    auto  dbuffer_managed = hipsparse_unique_ptr{device_malloc(buffer_size), device_free};
    void* dbuffer         = dbuffer_managed.get();

    int hbsr_nnzb;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2gebsrNnz(handle,
                                                 dir,
                                                 m,
                                                 n,
                                                 csr_descr,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 bsr_descr,
                                                 dbsr_row_ptr,
                                                 row_block_dim,
                                                 col_block_dim,
                                                 &hbsr_nnzb,
                                                 dbuffer));

    // Allocate memory on the device
    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * hbsr_nnzb), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * hbsr_nnzb * row_block_dim * col_block_dim), device_free};

    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXcsr2gebsr(handle,
                                                  dir,
                                                  m,
                                                  n,
                                                  csr_descr,
                                                  dcsr_val,
                                                  dcsr_row_ptr,
                                                  dcsr_col_ind,
                                                  bsr_descr,
                                                  dbsr_val,
                                                  dbsr_row_ptr,
                                                  dbsr_col_ind,
                                                  row_block_dim,
                                                  col_block_dim,
                                                  dbuffer));

        // Copy output from device to host
        std::vector<int> hbsr_row_ptr(mb + 1);
        std::vector<int> hbsr_col_ind(hbsr_nnzb);
        std::vector<T>   hbsr_val(hbsr_nnzb * row_block_dim * col_block_dim);

        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr.data(), dbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind.data(), dbsr_col_ind, sizeof(int) * hbsr_nnzb, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val.data(),
                                  dbsr_val,
                                  sizeof(T) * hbsr_nnzb * row_block_dim * col_block_dim,
                                  hipMemcpyDeviceToHost));

        // Host csr2gebsr conversion
        std::vector<int> hbsr_row_ptr_gold(mb + 1);
        std::vector<int> hbsr_col_ind_gold(hbsr_nnzb, 0);
        std::vector<T>   hbsr_val_gold(hbsr_nnzb * row_block_dim * col_block_dim);

        // call host csr2gebsr here

        int bsr_nnzb_gold;

        host_csr_to_gebsr<T>(dir,
                             m,
                             n,
                             row_block_dim,
                             col_block_dim,
                             bsr_nnzb_gold,
                             csr_idx_base,
                             hcsr_row_ptr,
                             hcsr_col_ind,
                             hcsr_val,
                             bsr_idx_base,
                             hbsr_row_ptr_gold,
                             hbsr_col_ind_gold,
                             hbsr_val_gold);

        // Unit check
        unit_check_general(1, 1, 1, &bsr_nnzb_gold, &hbsr_nnzb);
        unit_check_general(1, mb + 1, 1, hbsr_row_ptr_gold.data(), hbsr_row_ptr.data());
        unit_check_general(1, hbsr_nnzb, 1, hbsr_col_ind_gold.data(), hbsr_col_ind.data());
        unit_check_general(
            1, hbsr_nnzb * row_block_dim * col_block_dim, 1, hbsr_val_gold.data(), hbsr_val.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsr2gebsr(handle,
                                                      dir,
                                                      m,
                                                      n,
                                                      csr_descr,
                                                      dcsr_val,
                                                      dcsr_row_ptr,
                                                      dcsr_col_ind,
                                                      bsr_descr,
                                                      dbsr_val,
                                                      dbsr_row_ptr,
                                                      dbsr_col_ind,
                                                      row_block_dim,
                                                      col_block_dim,
                                                      dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXcsr2gebsr(handle,
                                                      dir,
                                                      m,
                                                      n,
                                                      csr_descr,
                                                      dcsr_val,
                                                      dcsr_row_ptr,
                                                      dcsr_col_ind,
                                                      bsr_descr,
                                                      dbsr_val,
                                                      dbsr_row_ptr,
                                                      dbsr_col_ind,
                                                      row_block_dim,
                                                      col_block_dim,
                                                      dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count
            = csr2gebsr_gbyte_count<T>(m, mb, nnz, hbsr_nnzb, row_block_dim, col_block_dim);
        double gpu_gbyte = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::Mb,
                            mb,
                            display_key_t::Nb,
                            nb,
                            display_key_t::row_block_dim,
                            row_block_dim,
                            display_key_t::col_block_dim,
                            col_block_dim,
                            display_key_t::nnzb,
                            hbsr_nnzb,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSR2GEBSR_HPP
