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
#ifndef TESTING_CSRCOLOR_HPP
#define TESTING_CSRCOLOR_HPP

#include "hipsparse.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_csrcolor_bad_arg(void)
{
#if(!defined(CUDART_VERSION))

    static constexpr int M               = 10;
    static constexpr int NNZ             = 10;
    floating_data_t<T>   fractionToColor = make_DataType<floating_data_t<T>>(1.0);

    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    auto m_coloring    = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};
    auto m_reordering  = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};
    auto m_csr_val     = hipsparse_unique_ptr{device_malloc(sizeof(T) * 1), device_free};
    auto m_csr_row_ptr = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};
    auto m_csr_col_ind = hipsparse_unique_ptr{device_malloc(sizeof(int) * 1), device_free};
    T*   d_csr_val     = (T*)m_csr_val.get();
    int* d_coloring    = (int*)m_coloring.get();
    int* d_reordering  = (int*)m_reordering.get();
    int* d_csr_row_ptr = (int*)m_csr_row_ptr.get();
    int* d_csr_col_ind = (int*)m_csr_col_ind.get();
    int  ncolors;

    hipsparseColorInfo_t colorInfo = (hipsparseColorInfo_t)0x4;

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   NNZ,
                                   descr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.u");

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   NNZ,
                                   descr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   &ncolors,
                                   nullptr,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.u");

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   NNZ,
                                   descr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   nullptr,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.u");

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   NNZ,
                                   descr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   nullptr,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.u");

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   NNZ,
                                   descr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   nullptr,
                                   &fractionToColor,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.u");

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   NNZ,
                                   descr,
                                   d_csr_val,
                                   nullptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.u");

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   NNZ,
                                   descr,
                                   nullptr,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.u");

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   NNZ,
                                   nullptr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid pointer must be detected.u");

    status = hipsparseXcsrcolor<T>(nullptr,
                                   M,
                                   NNZ,
                                   descr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXcsrcolor<T>(handle,
                                   -1,
                                   NNZ,
                                   descr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_value(status, "Error: an invalid value must be detected.u");

    status = hipsparseXcsrcolor<T>(handle,
                                   M,
                                   -1,
                                   descr,
                                   d_csr_val,
                                   d_csr_row_ptr,
                                   d_csr_col_ind,
                                   &fractionToColor,
                                   &ncolors,
                                   d_coloring,
                                   d_reordering,
                                   colorInfo);
    verify_hipsparse_status_invalid_pointer(status, "Error: an invalid value must be detected.u");
#endif
}

template <typename T>
hipsparseStatus_t testing_csrcolor()
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION < 13000)
    // Determine absolute path of test matrix
    // Matrices are stored at the same path in matrices directory
    std::string filename = get_filename("nos3.bin");

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    // Host structures
    std::vector<int>     hrow_ptr;
    std::vector<int>     hcol_ind;
    std::vector<T>       hval;
    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;

    // Initial Data on CPU
    srand(12345ULL);
    floating_data_t<T> fractionToColor = make_DataType<floating_data_t<T>>(1.0);

    int m;
    int k;
    int nnz;

    if(read_bin_matrix(filename.c_str(), m, k, nnz, hrow_ptr, hcol_ind, hval, idx_base) != 0)
    {
        fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    hipsparseColorInfo_t colorInfo;
    hipsparseCreateColorInfo(&colorInfo);

    // allocate memory on device
    auto drow_ptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcol_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dval_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dcoloring_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * m), device_free};
    auto dreordering_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * m), device_free};

    int* drow_ptr    = (int*)drow_ptr_managed.get();
    int* dcol_ind    = (int*)dcol_ind_managed.get();
    T*   dval        = (T*)dval_managed.get();
    int* dcoloring   = (int*)dcoloring_managed.get();
    int* dreordering = (int*)dreordering_managed.get();

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(drow_ptr, hrow_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcol_ind, hcol_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dval, hval.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    int ncolors;

    CHECK_HIPSPARSE_ERROR(hipsparseXcsrcolor(handle,
                                             m,
                                             nnz,
                                             descr,
                                             dval,
                                             drow_ptr,
                                             dcol_ind,
                                             &fractionToColor,
                                             &ncolors,
                                             dcoloring,
                                             dreordering,
                                             colorInfo));

    hipsparseDestroyColorInfo(colorInfo);
#endif

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_CSRCOLOR_HPP
