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
#ifndef TESTING_GEBSR2GEBSR_HPP
#define TESTING_GEBSR2GEBSR_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_gebsr2gebsr_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int                  mb              = 1;
    int                  nb              = 1;
    int                  nnzb            = 1;
    int                  safe_size       = 1;
    int                  row_block_dim_A = 2;
    int                  col_block_dim_A = 2;
    int                  row_block_dim_C = 2;
    int                  col_block_dim_C = 2;
    hipsparseIndexBase_t idx_base_A      = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t idx_base_C      = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDirection_t dir             = HIPSPARSE_DIRECTION_ROW;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t            descr_A = unique_ptr_descr_A->descr;
    std::unique_ptr<descr_struct>  unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t            descr_C = unique_ptr_descr_C->descr;

    hipsparseSetMatIndexBase(descr_A, idx_base_A);
    hipsparseSetMatIndexBase(descr_C, idx_base_C);

    auto bsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto bsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto bsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (safe_size + 1)), device_free};
    auto bsr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto temp_buffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    int* bsr_row_ptr_A = (int*)bsr_row_ptr_A_managed.get();

    int* bsr_col_ind_A = (int*)bsr_col_ind_A_managed.get();
    T*   bsr_val_A     = (T*)bsr_val_A_managed.get();
    int* bsr_row_ptr_C = (int*)bsr_row_ptr_C_managed.get();
    int* bsr_col_ind_C = (int*)bsr_col_ind_C_managed.get();
    T*   bsr_val_C     = (T*)bsr_val_C_managed.get();
    T*   temp_buffer   = (T*)temp_buffer_managed.get();

    int local_ptr[2] = {0, 1};
    CHECK_HIP_ERROR(
        hipMemcpy(bsr_row_ptr_A, local_ptr, sizeof(int) * (safe_size + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(bsr_row_ptr_C, local_ptr, sizeof(int) * (safe_size + 1), hipMemcpyHostToDevice));

    // Testing hipsparseXgebsr2gebsr_bufferSize()

    int buffer_size;

    // Test invalid handle
    status = hipsparseXgebsr2gebsr_bufferSize(nullptr,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              nullptr,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              (const T*)nullptr,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val_A is nullptr");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              nullptr,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_A is nullptr");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              nullptr,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind_A is nullptr");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");

    // Test invalid sizes
    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              -1,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              -1,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              -1,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              -1,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              -1,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              -1,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_C is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              -1,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_C is invalid");

    // Testing hipsparseXgebsr2gebsrNnz()

    int nnz_total_dev_host_ptr;

    // Test invalid handle
    status = hipsparseXgebsr2gebsrNnz(nullptr,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      nullptr,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      nullptr,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_A is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      nullptr,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind_A is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      nullptr,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      nullptr,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_C is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      nullptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: nnz_total_dev_host_ptr is nullptr");

    // Test invalid sizes
    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      -1,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      -1,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      -1,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      -1,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      -1,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      -1,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_C is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      -1,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_C is invalid");

    // Test hipsparseXgebsr2gebsr()

    // Test invalid handle
    status = hipsparseXgebsr2gebsr(nullptr,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   nullptr,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   (const T*)nullptr,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val_A is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   nullptr,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_A is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   nullptr,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind_A is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   nullptr,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   (T*)nullptr,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);

    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val_C is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   nullptr,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_C is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   nullptr,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind_C is nullptr");

    // Test invalid sizes
    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   -1,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   -1,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   -1,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   -1,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   -1,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   -1,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_C is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   -1,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_C is invalid");
#endif
}

template <typename T>
hipsparseStatus_t testing_gebsr2gebsr(Arguments argus)
{
    int                  m               = argus.M;
    int                  n               = argus.N;
    int                  row_block_dim_A = argus.row_block_dimA;
    int                  col_block_dim_A = argus.col_block_dimA;
    int                  row_block_dim_C = argus.row_block_dimB;
    int                  col_block_dim_C = argus.col_block_dimB;
    hipsparseIndexBase_t idx_base_A      = argus.baseA;
    hipsparseIndexBase_t idx_base_C      = argus.baseB;
    hipsparseDirection_t dir             = argus.dirA;
    std::string          filename        = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t            descr_A = unique_ptr_descr_A->descr;
    std::unique_ptr<descr_struct>  unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t            descr_C = unique_ptr_descr_C->descr;

    hipsparseSetMatIndexBase(descr_A, idx_base_A);
    hipsparseSetMatIndexBase(descr_C, idx_base_C);

    if(m == 0 || n == 0)
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        // cusparse does not support m == 0 for csr2bsr
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
    if(!generate_csr_matrix(filename, m, n, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base_A))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    // mb and nb can be modified if reading from a file
    int mb   = (m + row_block_dim_A - 1) / row_block_dim_A;
    int nb   = (n + col_block_dim_A - 1) / col_block_dim_A;
    int mb_C = (mb * row_block_dim_A + row_block_dim_C - 1) / row_block_dim_C;
    int nb_C = (nb * col_block_dim_A + col_block_dim_C - 1) / col_block_dim_C;

    // allocate memory on device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};

    int* dcsr_row_ptr   = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind   = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val       = (T*)dcsr_val_managed.get();
    int* dbsr_row_ptr_A = (int*)dbsr_row_ptr_A_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    size_t buffer_size_conversion;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2gebsr_bufferSize(handle,
                                                         dir,
                                                         m,
                                                         n,
                                                         descr_A,
                                                         dcsr_val,
                                                         dcsr_row_ptr,
                                                         dcsr_col_ind,
                                                         row_block_dim_A,
                                                         col_block_dim_A,
                                                         &buffer_size_conversion));

    auto dbuffer_conversion_managed
        = hipsparse_unique_ptr{device_malloc(buffer_size_conversion), device_free};
    void* dbuffer_conversion = dbuffer_conversion_managed.get();

    // Obtain BSR nnzb first on the host and then using the device and ensure they give the same results
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    int nnzb;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2gebsrNnz(handle,
                                                 dir,
                                                 m,
                                                 n,
                                                 descr_A,
                                                 dcsr_row_ptr,
                                                 dcsr_col_ind,
                                                 descr_A,
                                                 dbsr_row_ptr_A,
                                                 row_block_dim_A,
                                                 col_block_dim_A,
                                                 &nnzb,
                                                 dbuffer_conversion));

    // Allocate memory on the device
    auto dbsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dbsr_val_A_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * row_block_dim_A * col_block_dim_A), device_free};
    auto dbsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb_C + 1)), device_free};

    int* dbsr_col_ind_A = (int*)dbsr_col_ind_A_managed.get();
    T*   dbsr_val_A     = (T*)dbsr_val_A_managed.get();
    int* dbsr_row_ptr_C = (int*)dbsr_row_ptr_C_managed.get();

    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2gebsr(handle,
                                              dir,
                                              m,
                                              n,
                                              descr_A,
                                              dcsr_val,
                                              dcsr_row_ptr,
                                              dcsr_col_ind,
                                              descr_A,
                                              dbsr_val_A,
                                              dbsr_row_ptr_A,
                                              dbsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              dbuffer_conversion));

    // Copy output from device to host
    std::vector<int> hbsr_row_ptr_A(mb + 1);
    std::vector<int> hbsr_col_ind_A(nnzb);
    std::vector<T>   hbsr_val_A(nnzb * row_block_dim_A * col_block_dim_A);

    CHECK_HIP_ERROR(hipMemcpy(
        hbsr_row_ptr_A.data(), dbsr_row_ptr_A, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(
        hbsr_col_ind_A.data(), dbsr_col_ind_A, sizeof(int) * nnzb, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hbsr_val_A.data(),
                              dbsr_val_A,
                              sizeof(T) * nnzb * row_block_dim_A * col_block_dim_A,
                              hipMemcpyDeviceToHost));

    int buffer_size = 0;
    CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsr_bufferSize(handle,
                                                           dir,
                                                           mb,
                                                           nb,
                                                           nnzb,
                                                           descr_A,
                                                           dbsr_val_A,
                                                           dbsr_row_ptr_A,
                                                           dbsr_col_ind_A,
                                                           row_block_dim_A,
                                                           col_block_dim_A,
                                                           row_block_dim_C,
                                                           col_block_dim_C,
                                                           &buffer_size));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    // Obtain BSR nnzb first on the host and then using the device and ensure they give the same results
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

    int hnnzb_C;
    CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsrNnz(handle,
                                                   dir,
                                                   mb,
                                                   nb,
                                                   nnzb,
                                                   descr_A,
                                                   dbsr_row_ptr_A,
                                                   dbsr_col_ind_A,
                                                   row_block_dim_A,
                                                   col_block_dim_A,
                                                   descr_C,
                                                   dbsr_row_ptr_C,
                                                   row_block_dim_C,
                                                   col_block_dim_C,
                                                   &hnnzb_C,
                                                   dbuffer));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));

    auto dnnzb_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    int* dnnzb_C         = (int*)dnnzb_C_managed.get();
    CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsrNnz(handle,
                                                   dir,
                                                   mb,
                                                   nb,
                                                   nnzb,
                                                   descr_A,
                                                   dbsr_row_ptr_A,
                                                   dbsr_col_ind_A,
                                                   row_block_dim_A,
                                                   col_block_dim_A,
                                                   descr_C,
                                                   dbsr_row_ptr_C,
                                                   row_block_dim_C,
                                                   col_block_dim_C,
                                                   dnnzb_C,
                                                   dbuffer));

    int hnnzb_C_copied_from_device;
    CHECK_HIP_ERROR(
        hipMemcpy(&hnnzb_C_copied_from_device, dnnzb_C, sizeof(int), hipMemcpyDeviceToHost));

    if(argus.unit_check)
    {
        // Check that using host and device pointer mode gives the same result
        unit_check_general(1, 1, 1, &hnnzb_C_copied_from_device, &hnnzb_C);
    }

    // Allocate memory on the device
    auto dbsr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnzb_C), device_free};
    auto dbsr_val_C_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * hnnzb_C * row_block_dim_C * col_block_dim_C), device_free};

    int* dbsr_col_ind_C = (int*)dbsr_col_ind_C_managed.get();
    T*   dbsr_val_C     = (T*)dbsr_val_C_managed.get();

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsr(handle,
                                                    dir,
                                                    mb,
                                                    nb,
                                                    nnzb,
                                                    descr_A,
                                                    dbsr_val_A,
                                                    dbsr_row_ptr_A,
                                                    dbsr_col_ind_A,
                                                    row_block_dim_A,
                                                    col_block_dim_A,
                                                    descr_C,
                                                    dbsr_val_C,
                                                    dbsr_row_ptr_C,
                                                    dbsr_col_ind_C,
                                                    row_block_dim_C,
                                                    col_block_dim_C,
                                                    dbuffer));

        // Copy output from device to host
        std::vector<int> hbsr_row_ptr_C(mb_C + 1);
        std::vector<int> hbsr_col_ind_C(hnnzb_C);
        std::vector<T>   hbsr_val_C(hnnzb_C * row_block_dim_C * col_block_dim_C);

        CHECK_HIP_ERROR(hipMemcpy(hbsr_row_ptr_C.data(),
                                  dbsr_row_ptr_C,
                                  sizeof(int) * (mb_C + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind_C.data(), dbsr_col_ind_C, sizeof(int) * hnnzb_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val_C.data(),
                                  dbsr_val_C,
                                  sizeof(T) * hnnzb_C * row_block_dim_C * col_block_dim_C,
                                  hipMemcpyDeviceToHost));

        // Host csr2bsr conversion
        std::vector<int> hbsr_row_ptr_C_gold;
        std::vector<int> hbsr_col_ind_C_gold;
        std::vector<T>   hbsr_val_C_gold;

        // call host gebsr2gebsr here
        host_gebsr_to_gebsr(dir,
                            mb,
                            nb,
                            nnzb,
                            hbsr_val_A,
                            hbsr_row_ptr_A,
                            hbsr_col_ind_A,
                            row_block_dim_A,
                            col_block_dim_A,
                            idx_base_A,
                            hbsr_val_C_gold,
                            hbsr_row_ptr_C_gold,
                            hbsr_col_ind_C_gold,
                            row_block_dim_C,
                            col_block_dim_C,
                            idx_base_C);

        int nnzb_C_gold = hbsr_row_ptr_C_gold[mb_C] - hbsr_row_ptr_C_gold[0];

        // Unit check
        unit_check_general(1, 1, 1, &nnzb_C_gold, &hnnzb_C);
        unit_check_general(1, mb_C + 1, 1, hbsr_row_ptr_C_gold.data(), hbsr_row_ptr_C.data());
        unit_check_general(1, hnnzb_C, 1, hbsr_col_ind_C_gold.data(), hbsr_col_ind_C.data());
        unit_check_general(1,
                           hnnzb_C * row_block_dim_C * col_block_dim_C,
                           1,
                           hbsr_val_C_gold.data(),
                           hbsr_val_C.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsr(handle,
                                                        dir,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        descr_A,
                                                        dbsr_val_A,
                                                        dbsr_row_ptr_A,
                                                        dbsr_col_ind_A,
                                                        row_block_dim_A,
                                                        col_block_dim_A,
                                                        descr_C,
                                                        dbsr_val_C,
                                                        dbsr_row_ptr_C,
                                                        dbsr_col_ind_C,
                                                        row_block_dim_C,
                                                        col_block_dim_C,
                                                        dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsr(handle,
                                                        dir,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        descr_A,
                                                        dbsr_val_A,
                                                        dbsr_row_ptr_A,
                                                        dbsr_col_ind_A,
                                                        row_block_dim_A,
                                                        col_block_dim_A,
                                                        descr_C,
                                                        dbsr_val_C,
                                                        dbsr_row_ptr_C,
                                                        dbsr_col_ind_C,
                                                        row_block_dim_C,
                                                        col_block_dim_C,
                                                        dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gbyte_count = gebsr2gebsr_gbyte_count<T>(mb,
                                                        mb_C,
                                                        row_block_dim_A,
                                                        col_block_dim_A,
                                                        row_block_dim_C,
                                                        col_block_dim_C,
                                                        nnzb,
                                                        hnnzb_C);
        double gpu_gbyte   = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::N,
                            n,
                            display_key_t::MbA,
                            mb,
                            display_key_t::NbA,
                            nb,
                            display_key_t::MbC,
                            mb_C,
                            display_key_t::NbC,
                            nb_C,
                            display_key_t::row_block_dimA,
                            row_block_dim_A,
                            display_key_t::col_block_dimA,
                            col_block_dim_A,
                            display_key_t::row_block_dimC,
                            row_block_dim_C,
                            display_key_t::col_block_dimC,
                            col_block_dim_C,
                            display_key_t::nnzbA,
                            nnzb,
                            display_key_t::nnzbC,
                            hnnzb_C,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GEBSR2GEBSR_HPP
