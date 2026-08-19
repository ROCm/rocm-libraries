// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private hipBLASLt reference GEMM adapter.

#include "datatype_interface.hpp"

#include <hipblaslt/hipblaslt.h>

template <typename Compute>
void hipblaslt_reference_gemm(hipblasOperation_t transA,
                              hipblasOperation_t transB,
                              int64_t            m,
                              int64_t            n,
                              int64_t            k,
                              Compute            alpha,
                              const void*        A,
                              int64_t            lda,
                              const void*        B,
                              int64_t            ldb,
                              Compute            beta,
                              const void*        C,
                              int64_t            ldc,
                              void*              D,
                              int64_t            ldd,
                              const void*        alphaVector,
                              const void*        scaleA,
                              const void*        scaleB,
                              Compute            scaleD,
                              bool               isScaleAVector,
                              bool               isScaleBVector,
                              hipDataType        typeA,
                              hipDataType        typeB,
                              hipDataType        typeC,
                              hipDataType        typeD,
                              hipDataType        computeInputTypeA,
                              hipDataType        computeInputTypeB,
                              bool               isScaleAMxFormat = false,
                              bool               isScaleBMxFormat = false);

void hipblaslt_reference_gemm(hipblasOperation_t   transA,
                              hipblasOperation_t   transB,
                              int64_t              m,
                              int64_t              n,
                              int64_t              k,
                              computeTypeInterface alpha,
                              const void*          A,
                              int64_t              lda,
                              const void*          B,
                              int64_t              ldb,
                              computeTypeInterface beta,
                              const void*          C,
                              int64_t              ldc,
                              void*                D,
                              int64_t              ldd,
                              const void*          alphaVector,
                              const void*          scaleA,
                              const void*          scaleB,
                              const void*          scaleD,
                              bool                 isScaleAVector,
                              bool                 isScaleBVector,
                              hipDataType          typeA,
                              hipDataType          typeB,
                              hipDataType          typeC,
                              hipDataType          typeD,
                              hipDataType          computeType,
                              hipDataType          computeInputTypeA,
                              hipDataType          computeInputTypeB,
                              bool                 isScaleAMxFormat = false,
                              bool                 isScaleBMxFormat = false);
