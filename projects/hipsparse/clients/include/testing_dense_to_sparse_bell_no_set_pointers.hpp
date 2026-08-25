/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifndef TESTING_DENSE_TO_SPARSE_BELL_NO_SET_POINTERS_HPP
#define TESTING_DENSE_TO_SPARSE_BELL_NO_SET_POINTERS_HPP

#include "display.hpp"
#include "flops.hpp"
#include "gbyte.hpp"
#include "hipsparse_arguments.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <hipsparse.h>
#include <string>
#include <typeinfo>

using namespace hipsparse_test;

template <typename I, typename T>
static void host_dense_to_bell(I                     m,
                               I                     n,
                               hipsparseIndexBase_t  base,
                               const std::vector<T>& A,
                               int64_t               ld,
                               hipsparseOrder_t      order,
                               I                     ell_block_size,
                               I&                    ell_cols,
                               std::vector<T>&       bell_val,
                               std::vector<I>&       bell_col_ind)
{
    const I mb = (m + ell_block_size - 1) / ell_block_size;
    const I nb = (n + ell_block_size - 1) / ell_block_size;

    const I base_val = (base == HIPSPARSE_INDEX_BASE_ONE) ? 1 : 0;

    ell_cols = 0;

    // First pass: determine the number of block columns (ell_cols).
    for(I i = 0; i < mb; i++)
    {
        I blocks_in_row = 0;
        for(I j = 0; j < nb; j++)
        {
            bool block_col_found = false;
            for(I r = 0; r < ell_block_size && !block_col_found; r++)
            {
                for(I c = 0; c < ell_block_size; c++)
                {
                    const I gr = ell_block_size * i + r;
                    const I gc = ell_block_size * j + c;

                    T val_A = make_DataType<T>(0);
                    if(gr < m && gc < n)
                    {
                        val_A = (order == HIPSPARSE_ORDER_COL) ? A[ld * gc + gr] : A[ld * gr + gc];
                    }

                    if(val_A != make_DataType<T>(0))
                    {
                        block_col_found = true;
                        break;
                    }
                }
            }

            if(block_col_found)
            {
                blocks_in_row++;
            }
        }

        ell_cols = std::max(ell_cols, ell_block_size * blocks_in_row);
    }

    bell_col_ind.resize(mb * ell_cols / ell_block_size);
    bell_val.resize(m * ell_cols);

    const I ell_block_width = ell_cols / ell_block_size;

    std::fill(bell_val.begin(), bell_val.end(), make_DataType<T>(0));

    // Second pass: fill the blocked ELL column indices and values.
    for(I i = 0; i < mb; i++)
    {
        I slot = 0;
        for(I j = 0; j < nb; j++)
        {
            bool block_col_found = false;
            for(I r = 0; r < ell_block_size && !block_col_found; r++)
            {
                for(I c = 0; c < ell_block_size; c++)
                {
                    const I gr = ell_block_size * i + r;
                    const I gc = ell_block_size * j + c;

                    T val_A = make_DataType<T>(0);
                    if(gr < m && gc < n)
                    {
                        val_A = (order == HIPSPARSE_ORDER_COL) ? A[ld * gc + gr] : A[ld * gr + gc];
                    }

                    if(val_A != make_DataType<T>(0))
                    {
                        block_col_found = true;
                        break;
                    }
                }
            }

            if(block_col_found)
            {
                bell_col_ind[i * ell_block_width + slot] = j + base_val;

                for(I r = 0; r < ell_block_size; r++)
                {
                    const I gr = ell_block_size * i + r;
                    if(gr >= m)
                    {
                        continue;
                    }
                    for(I c = 0; c < ell_block_size; c++)
                    {
                        const I gc  = ell_block_size * j + c;
                        T       val = make_DataType<T>(0);
                        if(gc < n)
                        {
                            val = (order == HIPSPARSE_ORDER_COL) ? A[ld * gc + gr]
                                                                 : A[ld * gr + gc];
                        }
                        bell_val[gr * ell_cols + slot * ell_block_size + c] = val;
                    }
                }

                slot++;
            }
        }

        for(I s = slot; s < ell_block_width; s++)
        {
            bell_col_ind[i * ell_block_width + s] = base_val - 1;
        }
    }
}

template <typename I, typename T>
void testing_dense_to_sparse_bell_no_set_pointers_bad_arg(const Arguments& argus)
{
}

template <typename I, typename T>
void testing_dense_to_sparse_bell_no_set_pointers(Arguments argus)
{
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
    I                           m        = argus.M;
    I                           n        = argus.N;
    I                           blockDim = argus.block_dim;
    hipsparseIndexBase_t        idx_base = argus.baseA;
    hipsparseDenseToSparseAlg_t alg
        = static_cast<hipsparseDenseToSparseAlg_t>(argus.dense2sparse_alg);
    hipsparseOrder_t order = argus.orderA;

    I ellBlockSize = blockDim;

    I mb = (m + blockDim - 1) / blockDim;
    I nb = (n + blockDim - 1) / blockDim;

    // Index and data type
    hipsparseIndexType_t typeI = getIndexType<I>();
    hipDataType          typeT = getDataType<T>();

    // hipSPARSE handle
    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    int64_t ld = (order == HIPSPARSE_ORDER_COL) ? m : n;

    // Host dense matrix (leading dimension aware).
    int64_t        nrow = (order == HIPSPARSE_ORDER_COL) ? ld : m;
    int64_t        ncol = (order == HIPSPARSE_ORDER_COL) ? n : ld;
    std::vector<T> hdense_val(nrow * ncol, make_DataType<T>(0));

    // Randomly mark whole ELL blocks as non-zero so that the converted matrix
    // actually contains empty blocks. Roughly 40% of the blocks are kept.
    srand(0);
    std::vector<char> block_nonzero(mb * nb);
    for(I bi = 0; bi < mb; ++bi)
    {
        for(I bj = 0; bj < nb; ++bj)
        {
            block_nonzero[bi * nb + bj] = ((rand() % 10) < 4) ? 1 : 0;
        }
    }

    for(I i = 0; i < m; ++i)
    {
        for(I j = 0; j < n; ++j)
        {
            const I    brow = i / ellBlockSize;
            const I    bcol = j / ellBlockSize;
            const bool nz   = block_nonzero[brow * nb + bcol] != 0;

            const int64_t idx
                = (order == HIPSPARSE_ORDER_COL) ? (int64_t(j) * ld + i) : (int64_t(i) * ld + j);
            hdense_val[idx] = nz ? random_generator<T>() : make_DataType<T>(0);
        }
    }

    I              ellCols = 0;
    std::vector<I> hbell_col_ind_cpu;
    std::vector<T> hbell_val_cpu;
    host_dense_to_bell(m,
                       n,
                       idx_base,
                       hdense_val,
                       ld,
                       order,
                       ellBlockSize,
                       ellCols,
                       hbell_val_cpu,
                       hbell_col_ind_cpu);

    // Allocate and copy the dense matrix to the device.
    auto ddense_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nrow * ncol), device_free};
    T*   ddense         = (T*)ddense_managed.get();
    CHECK_HIP_ERROR(
        hipMemcpy(ddense, hdense_val.data(), sizeof(T) * nrow * ncol, hipMemcpyHostToDevice));

    // Create dense matrix descriptor.
    hipsparseDnMatDescr_t matA;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateDnMat(&matA, m, n, ld, ddense, typeT, order));

    const int64_t bell_col_ind_size = mb * ellCols / ellBlockSize;
    const int64_t bell_val_size     = int64_t(m) * ellCols;

    auto dcol_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(I) * std::max(int64_t(1), bell_col_ind_size)), device_free};
    auto dval_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * std::max(int64_t(1), bell_val_size)), device_free};

    I* dcol = (I*)dcol_managed.get();
    T* dval = (T*)dval_managed.get();

    CHECK_HIP_ERROR(hipMemcpy(
        dcol, hbell_col_ind_cpu.data(), sizeof(I) * bell_col_ind_size, hipMemcpyHostToDevice));

    // Create blocked ELL descriptor with null pointers; ellCols is discovered during analysis.
    hipsparseSpMatDescr_t matB;
    CHECK_HIPSPARSE_ERROR(hipsparseCreateBlockedEll(&matB,
                                                    mb * blockDim,
                                                    nb * blockDim,
                                                    ellBlockSize,
                                                    ellCols,
                                                    dcol,
                                                    dval,
                                                    typeI,
                                                    idx_base,
                                                    typeT));

    // Query DenseToSparse buffer size.
    size_t bufferSize;
    CHECK_HIPSPARSE_ERROR(hipsparseDenseToSparse_bufferSize(handle, matA, matB, alg, &bufferSize));

    void* buffer;
    CHECK_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    // Analysis populates the ellCols field of the descriptor.
    CHECK_HIPSPARSE_ERROR(hipsparseDenseToSparse_analysis(handle, matA, matB, alg, buffer));

    if(argus.unit_check)
    {
        CHECK_HIPSPARSE_ERROR(hipsparseDenseToSparse_convert(handle, matA, matB, alg, buffer));

        std::vector<I> hbell_col_ind(bell_col_ind_size);
        std::vector<T> hbell_val(bell_val_size);

        CHECK_HIP_ERROR(hipMemcpy(
            hbell_col_ind.data(), dcol, sizeof(I) * bell_col_ind_size, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hbell_val.data(), dval, sizeof(T) * bell_val_size, hipMemcpyDeviceToHost));

        unit_check_general(1, bell_col_ind_size, 1, hbell_col_ind_cpu.data(), hbell_col_ind.data());
        unit_check_general(1, bell_val_size, 1, hbell_val_cpu.data(), hbell_val.data());
    }

    CHECK_HIP_ERROR(hipFree(buffer));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroyDnMat(matA));
    CHECK_HIPSPARSE_ERROR(hipsparseDestroySpMat(matB));
#endif
}

#endif // TESTING_DENSE_TO_SPARSE_BELL_NO_SET_POINTERS_HPP
