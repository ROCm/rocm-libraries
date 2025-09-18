/* ************************************************************************
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_BSRSM2_HPP
#define TESTING_BSRSM2_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_bsrsm2_bad_arg(void)
{
#if(!defined(CUDART_VERSION))
    int   mb        = 100;
    int   nrhs      = 100;
    int   nnzb      = 100;
    int   block_dim = 100;
    int   safe_size = 100;
    float alpha     = 0.6;
    int   ldb       = 100;
    int   ldx       = 100;

    hipsparseDirection_t   dirA   = HIPSPARSE_DIRECTION_ROW;
    hipsparseOperation_t   transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t   transX = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<bsrsm2_struct> unique_ptr_info(new bsrsm2_struct);
    bsrsm2Info_t                   info = unique_ptr_info->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dX_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   dptr = (int*)dptr_managed.get();
    int*   dcol = (int*)dcol_managed.get();
    float* dval = (float*)dval_managed.get();
    float* dB   = (float*)dB_managed.get();
    float* dX   = (float*)dX_managed.get();
    void*  dbuf = (void*)dbuf_managed.get();

    // testing hipsparseXbsrsm2_bufferSize
    int size;

    verify_hipsparse_status_invalid_handle(hipsparseXbsrsm2_bufferSize(nullptr,
                                                                       dirA,
                                                                       transA,
                                                                       transX,
                                                                       mb,
                                                                       nrhs,
                                                                       nnzb,
                                                                       descr,
                                                                       dval,
                                                                       dptr,
                                                                       dcol,
                                                                       block_dim,
                                                                       info,
                                                                       &size));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        nullptr,
                                                                        dval,
                                                                        dptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        info,
                                                                        &size),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        (float*)nullptr,
                                                                        dptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        info,
                                                                        &size),
                                            "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        dval,
                                                                        nullptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        info,
                                                                        &size),
                                            "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        dval,
                                                                        dptr,
                                                                        nullptr,
                                                                        block_dim,
                                                                        info,
                                                                        &size),
                                            "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        dval,
                                                                        dptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        nullptr,
                                                                        &size),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        dval,
                                                                        dptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        info,
                                                                        nullptr),
                                            "Error: size is nullptr");

    // testing hipsparseXbsrsm2_analysis
    verify_hipsparse_status_invalid_handle(hipsparseXbsrsm2_analysis(nullptr,
                                                                     dirA,
                                                                     transA,
                                                                     transX,
                                                                     mb,
                                                                     nrhs,
                                                                     nnzb,
                                                                     descr,
                                                                     dval,
                                                                     dptr,
                                                                     dcol,
                                                                     block_dim,
                                                                     info,
                                                                     policy,
                                                                     dbuf));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      nullptr,
                                                                      dval,
                                                                      dptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      dbuf),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      (const float*)nullptr,
                                                                      dptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      dbuf),
                                            "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      dval,
                                                                      nullptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      dbuf),
                                            "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      dval,
                                                                      dptr,
                                                                      nullptr,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      dbuf),
                                            "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      dval,
                                                                      dptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      nullptr,
                                                                      policy,
                                                                      dbuf),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      dval,
                                                                      dptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      nullptr),
                                            "Error: dbuf is nullptr");

    // testing hipsparseXbsrsm2_solve
    verify_hipsparse_status_invalid_handle(hipsparseXbsrsm2_solve(nullptr,
                                                                  dirA,
                                                                  transA,
                                                                  transX,
                                                                  mb,
                                                                  nrhs,
                                                                  nnzb,
                                                                  &alpha,
                                                                  descr,
                                                                  dval,
                                                                  dptr,
                                                                  dcol,
                                                                  block_dim,
                                                                  info,
                                                                  dB,
                                                                  ldb,
                                                                  dX,
                                                                  ldx,
                                                                  policy,
                                                                  dbuf));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   (const float*)nullptr,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   nullptr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   (const float*)nullptr,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   nullptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   nullptr,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   nullptr,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   (const float*)nullptr,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dB is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   (float*)nullptr,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dX is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   nullptr),
                                            "Error: dbuf is nullptr");
#endif
}

template <typename T>
hipsparseStatus_t testing_bsrsm2(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    int                  m         = argus.M;
    int                  nrhs      = argus.N;
    int                  block_dim = argus.block_dim;
    T                    h_alpha   = make_DataType<T>(argus.alpha);
    hipsparseDirection_t dir       = argus.dirA;
    hipsparseIndexBase_t idx_base  = argus.baseA;
    hipsparseOperation_t transA    = argus.transA;
    hipsparseOperation_t transX    = argus.transB;
    std::string          filename  = argus.filename;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<bsrsm2_struct> unique_ptr_bsrsm2_info(new bsrsm2_struct);
    bsrsm2Info_t                   info = unique_ptr_bsrsm2_info->info;

    // Set matrix index base
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    if(m == 0)
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
    if(!generate_csr_matrix(filename, m, m, nnz, hcsr_row_ptr, hcsr_col_ind, hcsr_val, idx_base))
    {
        fprintf(stderr, "Cannot open [read] %s\ncol", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    int mb = (m + block_dim - 1) / block_dim;

    int ldb = (transX == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? mb * block_dim : nrhs;
    int ldx = (transX == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? mb * block_dim : nrhs;

    int64_t nrowB = (transX == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? mb * block_dim : nrhs;
    int64_t ncolB = (transX == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? nrhs : mb * block_dim;

    int64_t nrowX = nrowB;
    int64_t ncolX = ncolB;

    std::vector<T> hB(nrowB * ncolB);
    std::vector<T> hX_1(nrowX * ncolX);
    std::vector<T> hX_2(nrowX * ncolX);
    std::vector<T> hX_gold(nrowX * ncolX);

    hipsparseInit<T>(hB, nrowB, ncolB);

    // Allocate memory on device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(T) * nrowB * ncolB), device_free};
    auto dX_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nrowX * ncolX), device_free};
    auto dX_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nrowX * ncolX), device_free};
    auto dalpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto dpos_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();
    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    T*   dB           = (T*)dB_managed.get();
    T*   dX_1         = (T*)dX_1_managed.get();
    T*   dX_2         = (T*)dX_2_managed.get();
    T*   dalpha       = (T*)dalpha_managed.get();
    int* dposition    = (int*)dpos_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * nrowB * ncolB, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Convert to BSR
    int nnzb;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                               dir,
                                               m,
                                               m,
                                               descr,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               block_dim,
                                               descr,
                                               dbsr_row_ptr,
                                               &nnzb));

    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};

    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();

    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                            dir,
                                            m,
                                            m,
                                            descr,
                                            dcsr_val,
                                            dcsr_row_ptr,
                                            dcsr_col_ind,
                                            block_dim,
                                            descr,
                                            dbsr_val,
                                            dbsr_row_ptr,
                                            dbsr_col_ind));

    // Obtain bsrsm2 buffer size
    int bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_bufferSize(handle,
                                                      dir,
                                                      transA,
                                                      transX,
                                                      mb,
                                                      nrhs,
                                                      nnzb,
                                                      descr,
                                                      dbsr_val,
                                                      dbsr_row_ptr,
                                                      dbsr_col_ind,
                                                      block_dim,
                                                      info,
                                                      &bufferSize));

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * bufferSize), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    // bsrsm2 analysis
    CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_analysis(handle,
                                                    dir,
                                                    transA,
                                                    transX,
                                                    mb,
                                                    nrhs,
                                                    nnzb,
                                                    descr,
                                                    dbsr_val,
                                                    dbsr_row_ptr,
                                                    dbsr_col_ind,
                                                    block_dim,
                                                    info,
                                                    HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                    dbuffer));

    int pos_analysis;
    hipsparseXbsrsm2_zeroPivot(handle, info, &pos_analysis);

    if(argus.unit_check)
    {
        // HIPSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_solve(handle,
                                                     dir,
                                                     transA,
                                                     transX,
                                                     mb,
                                                     nrhs,
                                                     nnzb,
                                                     &h_alpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     block_dim,
                                                     info,
                                                     dB,
                                                     ldb,
                                                     dX_1,
                                                     ldx,
                                                     HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                     dbuffer));

        int               hposition_1;
        hipsparseStatus_t pivot_status_1 = hipsparseXbsrsm2_zeroPivot(handle, info, &hposition_1);

        // HIPSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_solve(handle,
                                                     dir,
                                                     transA,
                                                     transX,
                                                     mb,
                                                     nrhs,
                                                     nnzb,
                                                     dalpha,
                                                     descr,
                                                     dbsr_val,
                                                     dbsr_row_ptr,
                                                     dbsr_col_ind,
                                                     block_dim,
                                                     info,
                                                     dB,
                                                     ldb,
                                                     dX_2,
                                                     ldx,
                                                     HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                     dbuffer));

        hipsparseStatus_t pivot_status_2 = hipsparseXbsrsm2_zeroPivot(handle, info, dposition);

        // Copy output from device to CPU
        int hposition_2;
        CHECK_HIP_ERROR(
            hipMemcpy(hX_1.data(), dX_1, sizeof(T) * nrowX * ncolX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hX_2.data(), dX_2, sizeof(T) * nrowX * ncolX, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(&hposition_2, dposition, sizeof(int), hipMemcpyDeviceToHost));

        // Host bsrsm2
        std::vector<int> hbsr_row_ptr(mb + 1);
        std::vector<int> hbsr_col_ind(nnzb);
        std::vector<T>   hbsr_val(nnzb * block_dim * block_dim);

        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_row_ptr.data(), dbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind.data(), dbsr_col_ind, sizeof(int) * nnzb, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val.data(),
                                  dbsr_val,
                                  sizeof(T) * nnzb * block_dim * block_dim,
                                  hipMemcpyDeviceToHost));

        int struct_position_gold;
        int numeric_position_gold;

        CHECK_HIP_ERROR(hipDeviceSynchronize());

        host_bsrsm(mb,
                   nrhs,
                   nnzb,
                   dir,
                   transA,
                   transX,
                   h_alpha,
                   hbsr_row_ptr.data(),
                   hbsr_col_ind.data(),
                   hbsr_val.data(),
                   block_dim,
                   hB.data(),
                   ldb,
                   hX_gold.data(),
                   ldx,
                   HIPSPARSE_DIAG_TYPE_NON_UNIT,
                   HIPSPARSE_FILL_MODE_LOWER,
                   idx_base,
                   &struct_position_gold,
                   &numeric_position_gold);

        unit_check_general(1, 1, 1, &struct_position_gold, &pos_analysis);
        unit_check_general(1, 1, 1, &numeric_position_gold, &hposition_1);
        unit_check_general(1, 1, 1, &numeric_position_gold, &hposition_2);

        if(hposition_1 != -1)
        {
            verify_hipsparse_status_zero_pivot(pivot_status_1,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
            return HIPSPARSE_STATUS_SUCCESS;
        }

        if(hposition_2 != -1)
        {
            verify_hipsparse_status_zero_pivot(pivot_status_2,
                                               "expected HIPSPARSE_STATUS_ZERO_PIVOT");
            return HIPSPARSE_STATUS_SUCCESS;
        }

        unit_check_near(nrowX, ncolX, ldx, hX_gold.data(), hX_1.data());
        unit_check_near(nrowX, ncolX, ldx, hX_gold.data(), hX_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        // Warm up
        for(int iter = 0; iter < number_cold_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_solve(handle,
                                                         dir,
                                                         transA,
                                                         transX,
                                                         mb,
                                                         nrhs,
                                                         nnzb,
                                                         &h_alpha,
                                                         descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         block_dim,
                                                         info,
                                                         dB,
                                                         ldb,
                                                         dX_1,
                                                         ldx,
                                                         HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                         dbuffer));
        }

        double gpu_time_used = get_time_us();

        // Performance run
        for(int iter = 0; iter < number_hot_calls; ++iter)
        {
            CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_solve(handle,
                                                         dir,
                                                         transA,
                                                         transX,
                                                         mb,
                                                         nrhs,
                                                         nnzb,
                                                         &h_alpha,
                                                         descr,
                                                         dbsr_val,
                                                         dbsr_row_ptr,
                                                         dbsr_col_ind,
                                                         block_dim,
                                                         info,
                                                         dB,
                                                         ldb,
                                                         dX_1,
                                                         ldx,
                                                         HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                         dbuffer));
        }

        gpu_time_used = (get_time_us() - gpu_time_used) / number_hot_calls;

        double gflop_count = csrsv_gflop_count(m,
                                               size_t(nnzb) * block_dim * block_dim,
                                               HIPSPARSE_DIAG_TYPE_NON_UNIT)
                             * nrhs;
        double gbyte_count = bsrsv_gbyte_count<T>(mb, nnzb, block_dim) * nrhs;

        double gpu_gflops = get_gpu_gflops(gpu_time_used, gflop_count);
        double gpu_gbyte  = get_gpu_gbyte(gpu_time_used, gbyte_count);

        display_timing_info(display_key_t::M,
                            m,
                            display_key_t::nnz,
                            nnzb * block_dim * block_dim,
                            display_key_t::nrhs,
                            nrhs,
                            display_key_t::block_dim,
                            block_dim,
                            display_key_t::alpha,
                            h_alpha,
                            display_key_t::transA,
                            hipsparse_operation2string(transA),
                            display_key_t::transX,
                            hipsparse_operation2string(transX),
                            display_key_t::diag_type,
                            hipsparse_diagtype2string(HIPSPARSE_DIAG_TYPE_NON_UNIT),
                            display_key_t::fill_mode,
                            hipsparse_fillmode2string(HIPSPARSE_FILL_MODE_LOWER),
                            display_key_t::solve_policy,
                            hipsparse_solvepolicy2string(HIPSPARSE_SOLVE_POLICY_USE_LEVEL),
                            display_key_t::gflops,
                            gpu_gflops,
                            display_key_t::bandwidth,
                            gpu_gbyte,
                            display_key_t::time_ms,
                            get_gpu_time_msec(gpu_time_used));
    }
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSRSV2_HPP
