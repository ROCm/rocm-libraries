/* ************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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
#ifndef ROCBLASLT_UTILS_HPP
#define ROCBLASLT_UTILS_HPP
#include "auxiliary.hpp"
#include "handle.h"
#include "utility.hpp"

inline bool isValidOrderForDatatype(hipDataType datatype, hipblasLtOrder_t order)
{
    if((datatype == HIP_R_16F && order != HIPBLASLT_ORDER_COL16_4R8)
       || (datatype == HIP_R_16BF && order != HIPBLASLT_ORDER_COL16_4R8)
       || (datatype == HIP_R_8F_E4M3 && order != HIPBLASLT_ORDER_COL16_4R16)
       || (datatype == HIP_R_8F_E4M3_FNUZ && order != HIPBLASLT_ORDER_COL16_4R16))
    {
        return false;
    }
    return true;
}

/*******************************************************************************
 * Validate Matmul Swizzle Arguments
 ******************************************************************************/
inline rocblaslt_status validateMatmulSwizzleArgs(const rocblaslt_matmul_desc   matmul_descr,
                                                  const rocblaslt_matrix_layout matA,
                                                  const rocblaslt_matrix_layout matB,
                                                  hipDataType                   a_type,
                                                  hipDataType                   b_type,
                                                  bool                          swizzleA,
                                                  bool                          swizzleB)
{
    //support TN for swizzle
    if(swizzleA && matmul_descr->op_A != HIPBLAS_OP_T)
        return rocblaslt_status_invalid_value;

    if(swizzleB && matmul_descr->op_B != HIPBLAS_OP_N)
        return rocblaslt_status_invalid_value;

    if(swizzleA && !isValidOrderForDatatype(a_type, matA->order))
    {
        log_error(__func__, "Error: Invalid Order for hipDataType A.");
        return rocblaslt_status_invalid_value;
    }

    if(swizzleB && !isValidOrderForDatatype(b_type, matB->order))
    {
        log_error(__func__, "Error: Invalid Order for hipDataType B.");
        return rocblaslt_status_invalid_value;
    }

    if(swizzleA && matA->ld != matA->m)
        log_hints(__func__,
                  "Warning: The lda parameter is ignored and disabled when swizzle_a is true. "
                  "There's no need to set lda.");

    if(swizzleB && matB->ld != matB->m)
        log_hints(__func__,
                  "Warning: The ldb parameter is ignored and disabled when swizzle_b is true. "
                  "There's no need to set ldb.");

    return rocblaslt_status_continue;
}

/*******************************************************************************
 * Validate Matmul Arguments
 ******************************************************************************/
inline rocblaslt_status validateMatmulArgs(int64_t                       m,
                                           int64_t                       n,
                                           int64_t                       k,
                                           const void*                   alpha,
                                           const void*                   a,
                                           const void*                   b,
                                           const void*                   beta,
                                           const void*                   c,
                                           const void*                   d,
                                           hipDataType                   type_a,
                                           hipDataType                   type_b,
                                           hipDataType                   type_c,
                                           hipDataType                   type_d,
                                           rocblaslt_compute_type        compute_type,
                                           hipblasOperation_t            opA,
                                           hipblasOperation_t            opB,
                                           int                           num_batches_a  = 1,
                                           int                           num_batches_b  = 1,
                                           int                           num_batches_c  = 1,
                                           int                           num_batches_d  = 1,
                                           int64_t                       batch_stride_a = 0,
                                           int64_t                       batch_stride_b = 0,
                                           int64_t                       batch_stride_c = 0,
                                           int64_t                       batch_stride_d = 0,
                                           const rocblaslt_pointer_mode& pointermode
                                           = rocblaslt_pointer_mode_host)
{
    rocblaslt_status status = rocblaslt_status_continue;

    if(!(type_a == HIP_R_32F && type_b == HIP_R_32F && type_c == HIP_R_32F && type_d == HIP_R_32F)
       && compute_type == rocblaslt_compute_f32_fast_xf32)
        status = rocblaslt_status_not_implemented;
    if(!((type_a == HIP_R_8I && type_b == HIP_R_8I && type_c == HIP_R_32I && type_d == HIP_R_32I)
         || (type_a == HIP_R_8I && type_b == HIP_R_8I && type_c == HIP_R_8I && type_d == HIP_R_8I))
       && compute_type == rocblaslt_compute_i32)
        status = rocblaslt_status_not_implemented;
    if(!strcmp(hip_datatype_to_string(type_a), hip_datatype_to_string(HIPBLASLT_DATATYPE_INVALID))
       || !strcmp(hip_datatype_to_string(type_b),
                  hip_datatype_to_string(HIPBLASLT_DATATYPE_INVALID))
       || !strcmp(hip_datatype_to_string(type_c),
                  hip_datatype_to_string(HIPBLASLT_DATATYPE_INVALID))
       || !strcmp(hip_datatype_to_string(type_d),
                  hip_datatype_to_string(HIPBLASLT_DATATYPE_INVALID))
       || !strcmp(rocblaslt_compute_type_string(compute_type),
                  rocblaslt_compute_type_string(ROCBLASLT_COMPUTE_TYPE_INVALID)))
        status = rocblaslt_status_not_implemented;

    if(status != rocblaslt_status_continue)
    {
        log_error(__func__,
                  "invalid args",
                  "datatype",
                  "matA",
                  hipDataType_to_string(type_a),
                  "matB",
                  hipDataType_to_string(type_b),
                  "matC",
                  hipDataType_to_string(type_c),
                  "matD",
                  hipDataType_to_string(type_d),
                  "computeType",
                  rocblaslt_compute_type_string(compute_type));
        return status;
    }

    if(opA == HIPBLASLT_OPERATION_INVALID || opB == HIPBLASLT_OPERATION_INVALID)
        status = rocblaslt_status_not_implemented;

    if(status != rocblaslt_status_continue)
    {
        log_error(__func__,
                  "invalid args",
                  "op",
                  "opA",
                  hipblasOperation_to_string(opA),
                  "opB",
                  hipblasOperation_to_string(opB));
        return status;
    }

    // sizes must not be negative
    if(batch_stride_a < 0 || batch_stride_b < 0 || batch_stride_c < 0 || batch_stride_d < 0)
    {
#ifndef CODE_COVERAGE
        std::cerr << "matrix and stride size must be positive" << std::endl;
#endif
        return rocblaslt_status_invalid_size;
    }

    // number of batches of matrics A,B,C,D must be the same and negative
    if(num_batches_a != num_batches_b || num_batches_a != num_batches_c
       || num_batches_a != num_batches_d || num_batches_a < 1)
    {
#ifndef CODE_COVERAGE
        std::cerr << " number of batches of matrics A,B,C,D must be the same and negative"
                  << std::endl;
#endif
        return rocblaslt_status_invalid_size;
    }

    // quick return 0 is valid in BLAS
    // Note: k==0 is not a quick return, because C must still be multiplied by
    // beta
    // Note: we don't check n here since grouped gemm accept some n == 0
    if(!m || !num_batches_a)
        return rocblaslt_status_success;

    if(!beta)
        return rocblaslt_status_invalid_pointer;

    // Update for the valid case: ((alpha_in_host && alpha=0) && (A=NULL || B=NULL))
    bool alpha_A_B_violation = (!alpha || ((pointermode || (*((float*)alpha))) && (!a || !b)));
    // pointers must be valid
    if(n && ((k && alpha_A_B_violation) || !c || !d))
        return rocblaslt_status_invalid_pointer;

    return rocblaslt_status_continue;
}

inline rocblaslt_status
    rocblaslt_epilogue_valid_args(const rocblaslt_epilogue& epilogue,
                                  const int64_t&            num_rows_e,
                                  const int64_t&            num_cols_e,
                                  const hipDataType&        d_type,
                                  const hipDataType&        original_bias_type,
                                  const void*               e_ptr,
                                  const hipDataType&        original_aux_type,
                                  const int64_t&            original_lde,
                                  const int64_t&            original_stride_e,
                                  const void*               original_bias,
                                  const void*               original_scaleAlphaVec,
                                  const void*               alpha,
                                  const RocblasltContractionProblem::ScalingFormat scaleAType,
                                  const RocblasltContractionProblem::ScalingFormat scaleBType,
                                  const uint32_t scaleABlockRowSize,
                                  const uint32_t scaleABlockColSize,
                                  const uint32_t scaleBBlockRowSize,
                                  const uint32_t scaleBBlockColSize,
                                  void*&         E,
                                  hipDataType&   aux_type,
                                  int64_t&       lde,
                                  int64_t&       batch_stride_e,
                                  void*&         bias,
                                  hipDataType&   bias_type,
                                  void*&         scaleAlphaVec,
                                  bool&          gradient)
{
    // Set status
    rocblaslt_status status = rocblaslt_status_continue;
    // External update args
    bias_type = original_bias_type; // == HIPBLASLT_DATATYPE_INVALID ? d_type : original_bias_type;
    gradient  = is_grad_enabled(epilogue);

    bias = nullptr;
    if(is_bias_enabled(epilogue))
    {
        if(original_bias == nullptr)
            status = rocblaslt_status_invalid_pointer;
        bias = (void*)original_bias;
    }

    if(original_scaleAlphaVec)
        scaleAlphaVec = (void*)original_scaleAlphaVec; //pointer mode alpha vector pass by alpha
    else
        scaleAlphaVec = nullptr;

    // matrix E
    E = nullptr;
    if(is_e_enabled(epilogue))
    {
        if(e_ptr == nullptr)
            status = rocblaslt_status_invalid_pointer;
        E = (void*)e_ptr;
    }
    aux_type       = original_aux_type;
    lde            = original_lde > 0 ? original_lde : num_rows_e;
    batch_stride_e = original_stride_e > 0 ? original_stride_e : original_lde * num_cols_e;
    if(E != nullptr && ((lde < num_rows_e) || (batch_stride_e < (num_cols_e * num_rows_e))))
        status = rocblaslt_status_invalid_value;
    if(scaleAType != RocblasltContractionProblem::ScalingFormat::None
       && scaleBType != RocblasltContractionProblem::ScalingFormat::None
       && scaleAType != scaleBType)
    {
        log_error(__func__, "Scale A and Scale B must be both scalar, vector or block.");
        status = rocblaslt_status_invalid_value;
    }
    if(scaleAType == RocblasltContractionProblem::ScalingFormat::Block)
    {
        if(scaleABlockRowSize != 32 || scaleABlockColSize != 1)
        {
            log_error(__func__, "ScaleA block row and column sizes currently only support 32x1");
            status = rocblaslt_status_invalid_value;
        }
    }
    if(scaleBType == RocblasltContractionProblem::ScalingFormat::Block)
    {
        if(scaleBBlockRowSize != 1 || scaleBBlockColSize != 32)
        {
            log_error(__func__, "ScaleB block row and column sizes currently only support 1x32");
            status = rocblaslt_status_invalid_value;
        }
    }
    return status;
}

inline rocblaslt_status rocblaslt_matmul_valid_args(const rocblaslt_matmul_desc matmul_descr,
                                                    const void*                 A,
                                                    const void*                 B,
                                                    const void*                 C,
                                                    const void*                 D,
                                                    rocblaslt_matrix_layout     matA,
                                                    rocblaslt_matrix_layout     matB,
                                                    rocblaslt_matrix_layout     matC,
                                                    rocblaslt_matrix_layout     matD,
                                                    const void*                 alpha,
                                                    const void*                 beta,
                                                    int64_t&                    m,
                                                    int64_t&                    n,
                                                    int64_t&                    k,
                                                    hipDataType&                a_type,
                                                    int64_t&                    lda,
                                                    int64_t&                    batch_stride_a,
                                                    hipDataType&                b_type,
                                                    int64_t&                    ldb,
                                                    int64_t&                    batch_stride_b,
                                                    hipDataType&                c_type,
                                                    int64_t&                    ldc,
                                                    int64_t&                    batch_stride_c,
                                                    hipDataType&                d_type,
                                                    int64_t&                    ldd,
                                                    int64_t&                    batch_stride_d,
                                                    int64_t&                    lde,
                                                    int64_t&                    batch_stride_e,
                                                    void*&                      bias,
                                                    hipDataType&                bias_type,
                                                    void*&                      scaleAlphaVec,
                                                    void*&                      E,
                                                    hipDataType&                aux_type,
                                                    bool&                       gradient,
                                                    rocblaslt_compute_type&     compute_type,
                                                    bool                        swizzleA,
                                                    bool                        swizzleB)
{
    // Internal assign
    hipblasOperation_t opA = matmul_descr->op_A;
    hipblasOperation_t opB = matmul_descr->op_B;

    auto matmul_swizzle_status
        = validateMatmulSwizzleArgs(matmul_descr, matA, matB, a_type, b_type, swizzleA, swizzleB);

    if(matmul_swizzle_status != rocblaslt_status_continue)
        return matmul_swizzle_status;

    // matrix A
    int64_t num_rows_a    = matA->m;
    int64_t num_cols_a    = matA->n;
    int     num_batches_a = matA->batch_count;
    a_type                = matA->type;
    lda                   = matA->ld;
    batch_stride_a        = matA->batch_stride;

    // matrix B
    int num_batches_b = matB->batch_count;
    b_type            = matB->type;
    ldb               = matB->ld;
    batch_stride_b    = matB->batch_stride;

    // matrix C
    int num_batches_c = matC->batch_count;
    c_type            = matC->type;
    ldc               = matC->ld;
    batch_stride_c    = matC->batch_stride;

    // matrix D
    int64_t num_rows_d    = matD->m;
    int64_t num_cols_d    = matD->n;
    int     num_batches_d = matD->batch_count;
    d_type                = matD->type;
    ldd                   = matD->ld;
    batch_stride_d        = matD->batch_stride;

    compute_type = matmul_descr->compute_type;

    m = num_rows_d;
    n = num_cols_d;
    k = (opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a;

    auto matmul_status = validateMatmulArgs(m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            B,
                                            beta,
                                            C,
                                            D,
                                            matA->type,
                                            matB->type,
                                            matC->type,
                                            matD->type,
                                            compute_type,
                                            matmul_descr->op_A,
                                            matmul_descr->op_B,
                                            num_batches_a,
                                            num_batches_b,
                                            num_batches_c,
                                            num_batches_d,
                                            batch_stride_a,
                                            batch_stride_b,
                                            batch_stride_c,
                                            batch_stride_d,
                                            matmul_descr->pointermode);

    const void* alphaVecPtr     = matmul_descr->pointermode ? alpha : nullptr;
    auto        epilogue_status = rocblaslt_epilogue_valid_args(matmul_descr->epilogue,
                                                         num_rows_d,
                                                         num_cols_d,
                                                         matD->type,
                                                         matmul_descr->bias_type,
                                                         matmul_descr->e,
                                                         matmul_descr->aux_type,
                                                         matmul_descr->lde,
                                                         matmul_descr->stride_e,
                                                         matmul_descr->bias,
                                                         alphaVecPtr,
                                                         alpha,
                                                         matmul_descr->scaleAType,
                                                         matmul_descr->scaleBType,
                                                         matmul_descr->scaleABlockRowSize,
                                                         matmul_descr->scaleABlockColSize,
                                                         matmul_descr->scaleBBlockRowSize,
                                                         matmul_descr->scaleBBlockColSize,
                                                         E,
                                                         aux_type,
                                                         lde,
                                                         batch_stride_e,
                                                         bias,
                                                         bias_type,
                                                         scaleAlphaVec,
                                                         gradient);

    // rocblaslt_epilogue_valid_args must to be called otherwise bias_type will be garbage value
    if(matmul_status != rocblaslt_status_continue)
        return matmul_status;

    if(epilogue_status != rocblaslt_status_continue)
        return epilogue_status;

    return rocblaslt_status_continue;
}

// Assign 1 to onePtr then set set the address to dst.
inline void setTo1(const rocblaslt_compute_type& compute_type, const void* onePtr, const void** dst)
{
    if(compute_type == rocblaslt_compute_f64)
    {
        *((double*)onePtr) = 1.f;
        *dst               = onePtr;
    }
    else if(compute_type == rocblaslt_compute_i32)
    {
        *((int32_t*)onePtr) = 1.f;
        *dst                = onePtr;
    }
    else
    {
        *((float*)onePtr) = 1.f;
        *dst              = onePtr;
    }
}
#endif
