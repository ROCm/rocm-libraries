/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once
#include "handle.h"
#ifndef LEGACY_HIPBLAS_DIRECT
#include <hipblas-common/hipblas-common.h>
#else
#include <hipblas/hipblas.h>
#endif
#include <rocblaslt.h>
#include <Debug.hpp>

hipblasStatus_t RocBlasLtStatusToHIPStatus(rocblaslt_status_ status);

// Matmul reads every layout as column-major or swizzled; only hipblasLtMatrixTransform reads
// HIPBLASLT_ORDER_ROW. Returns HIPBLAS_STATUS_NOT_SUPPORTED when any order is ROW. Callers check
// order only after the arguments their own validation checks first are present, so a missing
// handle, descriptor, or pointer keeps its usual status. The layout form returns SUCCESS when any
// layout is null or grouped; the grouped path has its own order check.
hipblasStatus_t refuse_row_order(const char*      caller,
                                 hipblasLtOrder_t orderA,
                                 hipblasLtOrder_t orderB,
                                 hipblasLtOrder_t orderC = HIPBLASLT_ORDER_COL,
                                 hipblasLtOrder_t orderD = HIPBLASLT_ORDER_COL);
hipblasStatus_t refuse_row_order(const char*             caller,
                                 rocblaslt_matrix_layout matA,
                                 rocblaslt_matrix_layout matB,
                                 rocblaslt_matrix_layout matC,
                                 rocblaslt_matrix_layout matD);

// True when validateMatmulArgs would report a null scalar or data pointer for this GEMM. The
// extension API checks it before refusing a row-major layout, so those calls keep their
// invalid-pointer status.
bool gemm_pointers_missing(int64_t                m,
                           int64_t                n,
                           int64_t                k,
                           int64_t                batchCount,
                           rocblaslt_compute_type computeType,
                           hipDataType            scaleType,
                           hipDataType            typeA,
                           rocblaslt_pointer_mode pointerMode,
                           const void*            alpha,
                           const void*            beta,
                           const void*            A,
                           const void*            B,
                           const void*            C,
                           const void*            D);
// The same check with the sizes and modes read from a descriptor and its layouts. Also true when
// the descriptor or a layout is null.
bool gemm_pointers_missing(rocblaslt_matmul_desc   desc,
                           rocblaslt_matrix_layout matA,
                           rocblaslt_matrix_layout matB,
                           rocblaslt_matrix_layout matC,
                           rocblaslt_matrix_layout matD,
                           const void*             alpha,
                           const void*             beta,
                           const void*             A,
                           const void*             B,
                           const void*             C,
                           const void*             D);
