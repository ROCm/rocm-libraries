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

#include "definitions.h"
#include "handle.h"
#include "rocblaslt_mat_utils.hpp"
#include "tensile_host.hpp"
#include <array>

#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief
 *******************************************************************************/

rocblaslt_status rocblaslt_matmul_impl(const rocblaslt_handle       handle,
                                       const rocblaslt_matmul_desc  matmul_descr,
                                       const void*                  A,
                                       const void*                  B,
                                       const void*                  C,
                                       void*                        D,
                                       rocblaslt_matrix_layout      matA,
                                       rocblaslt_matrix_layout      matB,
                                       rocblaslt_matrix_layout      matC,
                                       rocblaslt_matrix_layout      matD,
                                       const void*                  alpha,
                                       const void*                  beta,
                                       const rocblaslt_matmul_algo* algo,
                                       void*                        workspace,
                                       size_t                       workspaceSizeInBytes,
                                       hipStream_t                  stream)
{
    int64_t m, n, k, lda, ldb, ldc, ldd, lde, batch_stride_a, batch_stride_b, batch_stride_c,
        batch_stride_d, batch_stride_e;
    hipDataType            bias_type;
    hipDataType            aux_type;
    hipDataType            type_a, type_b, type_c, type_d;
    rocblaslt_compute_type compute_type;
    void *                 bias = nullptr, *scaleAlphaVec = nullptr, *E = nullptr;
    bool                   gradient = false;
    bool swizzleA = matA->order != HIPBLASLT_ORDER_COL && matA->order != HIPBLASLT_ORDER_ROW;
    bool swizzleB = matB->order != HIPBLASLT_ORDER_COL && matB->order != HIPBLASLT_ORDER_ROW;
    // no need to do batched swizzle stride check here, it was done in set-problem stage

    rocblaslt_status isValid = rocblaslt_matmul_valid_args(matmul_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           D,
                                                           matA,
                                                           matB,
                                                           matC,
                                                           matD,
                                                           alpha,
                                                           beta,
                                                           m,
                                                           n,
                                                           k,
                                                           type_a,
                                                           lda,
                                                           batch_stride_a,
                                                           type_b,
                                                           ldb,
                                                           batch_stride_b,
                                                           type_c,
                                                           ldc,
                                                           batch_stride_c,
                                                           type_d,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleAlphaVec,
                                                           E,
                                                           aux_type,
                                                           gradient,
                                                           compute_type,
                                                           swizzleA,
                                                           swizzleB);
    if(isValid != rocblaslt_status_continue)
        return isValid;

    // Internal assign
    hipblasOperation_t opA           = matmul_descr->op_A;
    hipblasOperation_t opB           = matmul_descr->op_B;
    int                num_batches_a = matA->batch_count;
    rocblaslt_epilogue epilogue      = matmul_descr->epilogue;
    void*              scaleA        = matmul_descr->scaleA;
    void*              scaleB        = matmul_descr->scaleB;
    void*              scaleC        = matmul_descr->scaleC;
    void*              scaleD        = matmul_descr->scaleD;
    void*              scaleE        = matmul_descr->scaleE;
    void*              amaxD         = matmul_descr->amaxD;
    hipDataType        scale_type    = matmul_descr->scale_type;

    // Others
    bool strided_batch = true;
    bool grouped_gemm  = false;

    auto& gemmData = matmul_descr->m_data;

    int8_t alpha_1[16] = {0}; // use dScaleAlphaVec instead, original alpha => 1.0
    if(scaleAlphaVec)
    {
        if(matmul_descr->compute_type == rocblaslt_compute_f64)
        {
            *((double*)alpha_1) = 1.f;
            alpha               = alpha_1;
        }
        else if(matmul_descr->compute_type == rocblaslt_compute_i32)
        {
            *((int32_t*)alpha_1) = 1.f;
            alpha                = alpha_1;
        }
        else
        {
            *((float*)alpha_1) = 1.f;
            alpha              = alpha_1;
        }
    }

    // FIXME: Is this still needed?
    // // check alignment of pointers before casting
    // if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
    //    || !isAligned(d, sizeof(To)))
    // {
    //     std::cerr << "memmory is not aligned" << std::endl;
    //     return rocblaslt_status_invalid_size;
    // }

    if(algo)
        workspaceSizeInBytes = min(workspaceSizeInBytes, algo->max_workspace_bytes);
    RocblasltContractionProblem problem{opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        type_a,
                                        A,
                                        nullptr,
                                        lda,
                                        batch_stride_a,
                                        type_b,
                                        B,
                                        nullptr,
                                        ldb,
                                        batch_stride_b,
                                        beta,
                                        type_c,
                                        C,
                                        nullptr,
                                        ldc,
                                        batch_stride_c,
                                        type_d,
                                        D,
                                        nullptr,
                                        ldd,
                                        batch_stride_d,
                                        E,
                                        nullptr,
                                        lde,
                                        batch_stride_e,
                                        num_batches_a,
                                        strided_batch,
                                        grouped_gemm,
                                        gradient,
                                        compute_type,
                                        matmul_descr->scale_type,
                                        bias,
                                        scaleA,
                                        scaleB,
                                        scaleC,
                                        scaleD,
                                        scaleE,
                                        scaleAlphaVec,
                                        matmul_descr->scaleAType,
                                        matmul_descr->scaleBType,
                                        matmul_descr->scaleABlockRowSize,
                                        matmul_descr->scaleABlockColSize,
                                        matmul_descr->scaleBBlockRowSize,
                                        matmul_descr->scaleBBlockColSize,
                                        bias_type,
                                        aux_type,
                                        epilogue,
                                        amaxD,
                                        workspace,
                                        workspaceSizeInBytes,
                                        matmul_descr->act0,
                                        matmul_descr->act1,
                                        stream,
                                        handle->Synchronizer,
                                        swizzleA,
                                        swizzleB};

    return runContractionProblem(handle, algo, problem, gemmData);
}

rocblaslt_status rocblaslt_gemm_create_cpp_impl(const rocblaslt_handle           handle,
                                                rocblaslt_matmul_desc            matmul_descr,
                                                const void*                      A,
                                                const void*                      B,
                                                const void*                      C,
                                                void*                            D,
                                                rocblaslt_matrix_layout          matA,
                                                rocblaslt_matrix_layout          matB,
                                                rocblaslt_matrix_layout          matC,
                                                rocblaslt_matrix_layout          matD,
                                                const void*                      alpha,
                                                const void*                      beta,
                                                rocblaslt::RocGemmProblemTypeV2& problemtype,
                                                std::shared_ptr<void>&           gemmData,
                                                size_t&                          gemmCount)
{
    int64_t m, n, k, lda, ldb, ldc, ldd, lde, batch_stride_a, batch_stride_b, batch_stride_c,
        batch_stride_d, batch_stride_e;
    hipDataType            bias_type;
    hipDataType            aux_type;
    hipDataType            type_a, type_b, type_c, type_d;
    rocblaslt_compute_type compute_type;
    void *                 bias = nullptr, *scaleAlphaVec = nullptr, *E = nullptr;
    bool                   gradient = false;
    bool swizzleA = matA->order != HIPBLASLT_ORDER_COL && matA->order != HIPBLASLT_ORDER_ROW;
    bool swizzleB = matB->order != HIPBLASLT_ORDER_COL && matB->order != HIPBLASLT_ORDER_ROW;

    rocblaslt_status isValid = rocblaslt_matmul_valid_args(matmul_descr,
                                                           A,
                                                           B,
                                                           C,
                                                           D,
                                                           matA,
                                                           matB,
                                                           matC,
                                                           matD,
                                                           alpha,
                                                           beta,
                                                           m,
                                                           n,
                                                           k,
                                                           type_a,
                                                           lda,
                                                           batch_stride_a,
                                                           type_b,
                                                           ldb,
                                                           batch_stride_b,
                                                           type_c,
                                                           ldc,
                                                           batch_stride_c,
                                                           type_d,
                                                           ldd,
                                                           batch_stride_d,
                                                           lde,
                                                           batch_stride_e,
                                                           bias,
                                                           bias_type,
                                                           scaleAlphaVec,
                                                           E,
                                                           aux_type,
                                                           gradient,
                                                           compute_type,
                                                           swizzleA,
                                                           swizzleB);
    if(isValid != rocblaslt_status_continue)
        return isValid;

    // Internal assign
    hipblasOperation_t opA           = matmul_descr->op_A;
    hipblasOperation_t opB           = matmul_descr->op_B;
    int                num_batches_a = matA->batch_count;
    rocblaslt_epilogue epilogue      = matmul_descr->epilogue;
    void*              scaleA        = matmul_descr->scaleA;
    void*              scaleB        = matmul_descr->scaleB;
    void*              scaleC        = matmul_descr->scaleC;
    void*              scaleD        = matmul_descr->scaleD;
    void*              scaleE        = matmul_descr->scaleE;
    void*              amaxD         = matmul_descr->amaxD;

    // Others
    bool strided_batch = true;
    bool grouped_gemm  = false;

    int8_t alpha_1[16] = {0}; // use dScaleAlphaVec instead, original alpha => 1.0
    if(scaleAlphaVec)
    {
        setTo1(matmul_descr->compute_type, (void*)alpha_1, &alpha);
    }

    // // check alignment of pointers before casting
    // if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
    //    || !isAligned(d, sizeof(To)))
    // {
    //     std::cerr << "memmory is not aligned" << std::endl;
    //     return rocblaslt_status_invalid_size;
    // }

    RocblasltContractionProblem problem{opA,
                                        opB,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        type_a,
                                        A,
                                        nullptr,
                                        lda,
                                        batch_stride_a,
                                        type_b,
                                        B,
                                        nullptr,
                                        ldb,
                                        batch_stride_b,
                                        beta,
                                        type_c,
                                        C,
                                        nullptr,
                                        ldc,
                                        batch_stride_c,
                                        type_d,
                                        D,
                                        nullptr,
                                        ldd,
                                        batch_stride_d,
                                        E,
                                        nullptr,
                                        lde,
                                        batch_stride_e,
                                        num_batches_a,
                                        strided_batch,
                                        grouped_gemm,
                                        gradient,
                                        compute_type,
                                        matmul_descr->scale_type,
                                        bias,
                                        scaleA,
                                        scaleB,
                                        scaleC,
                                        scaleD,
                                        scaleE,
                                        scaleAlphaVec,
                                        matmul_descr->scaleAType,
                                        matmul_descr->scaleBType,
                                        matmul_descr->scaleABlockRowSize,
                                        matmul_descr->scaleABlockColSize,
                                        matmul_descr->scaleBBlockRowSize,
                                        matmul_descr->scaleBBlockColSize,
                                        bias_type,
                                        aux_type,
                                        epilogue,
                                        amaxD,
                                        nullptr,
                                        0,
                                        matmul_descr->act0,
                                        matmul_descr->act1,
                                        0,
                                        handle->Synchronizer,
                                        swizzleA,
                                        swizzleB};
    return gemmCreate(problem, gemmData, gemmCount);
}

rocblaslt_status
    rocblaslt_groupedgemm_create_cpp_impl(const rocblaslt_handle                handle,
                                          std::vector<rocblaslt_matmul_desc>&   matmul_descr,
                                          std::vector<const void*>&             A,
                                          std::vector<const void*>&             B,
                                          std::vector<const void*>&             C,
                                          std::vector<void*>&                   D,
                                          std::vector<rocblaslt_matrix_layout>& matA,
                                          std::vector<rocblaslt_matrix_layout>& matB,
                                          std::vector<rocblaslt_matrix_layout>& matC,
                                          std::vector<rocblaslt_matrix_layout>& matD,
                                          std::vector<const void*>&             alpha,
                                          std::vector<const void*>&             beta,
                                          std::vector<rocblaslt::RocGemmProblemTypeV2>& problemtype,
                                          std::shared_ptr<void>&                        gemmData,
                                          size_t&                                       gemmCount)
{
    hipblasOperation_t     opA          = matmul_descr[0]->op_A;
    hipblasOperation_t     opB          = matmul_descr[0]->op_B;
    rocblaslt_compute_type compute_type = matmul_descr[0]->compute_type;
    hipDataType            type_a       = matA[0]->type;
    hipDataType            type_b       = matB[0]->type;
    hipDataType            type_c       = matC[0]->type;
    hipDataType            type_d       = matD[0]->type;

    std::vector<const void*>            A_vec, B_vec, C_vec, alpha_vec, beta_vec;
    std::vector<void*>                  D_vec, E_vec, amaxD_vec;
    std::vector<const void*>            bias_vec;
    std::vector<const void*>            scaleA_vec;
    std::vector<const void*>            scaleB_vec;
    std::vector<const void*>            scaleC_vec;
    std::vector<const void*>            scaleD_vec;
    std::vector<const void*>            scaleE_vec;
    std::vector<const void*>            scaleAlpha_vec;
    std::vector<hipDataType>            bias_type_vec;
    std::vector<hipDataType>            aux_type_vec;
    std::vector<rocblaslt_epilogue>     epilogue_vec;
    std::vector<int64_t>                m_vec, n_vec, k_vec;
    std::vector<int64_t>                lda_vec, batch_stride_a_vec, num_batches_a_vec;
    std::vector<int64_t>                ldb_vec, batch_stride_b_vec, num_batches_b_vec;
    std::vector<int64_t>                ldc_vec, batch_stride_c_vec, num_batches_c_vec;
    std::vector<int64_t>                ldd_vec, batch_stride_d_vec, num_batches_d_vec;
    std::vector<int64_t>                lde_vec, batch_stride_e_vec, num_batches_e_vec;
    std::vector<std::array<int8_t, 16>> alpha_1(matmul_descr.size());

    std::vector<bool> gradient_vec;

    for(int i = 0; i < matmul_descr.size(); i++)
    {
        // matrix A
        int64_t num_rows_a     = matA[i]->m;
        int64_t num_cols_a     = matA[i]->n;
        int64_t lda            = matA[i]->ld;
        int64_t batch_stride_a = matA[i]->batch_stride;
        int     num_batches_a  = matA[i]->batch_count;

        // matrix B
        // int64_t num_rows_b = matB[i]->m;
        // int64_t num_cols_b = matB[i]->n;
        int64_t ldb            = matB[i]->ld;
        int64_t batch_stride_b = matB[i]->batch_stride;
        int     num_batches_b  = matB[i]->batch_count;

        // matrix C
        // int64_t num_rows_c = matC[i]->m;
        // int64_t num_cols_c = matC[i]->n;
        int64_t ldc            = matC[i]->ld;
        int64_t batch_stride_c = matC[i]->batch_stride;
        int     num_batches_c  = matC[i]->batch_count;

        // matrix D
        int64_t num_rows_d     = matD[i]->m;
        int64_t num_cols_d     = matD[i]->n;
        int64_t ldd            = matD[i]->ld;
        int64_t batch_stride_d = matD[i]->batch_stride;
        int     num_batches_d  = matD[i]->batch_count;

        int64_t m = num_rows_d;
        int64_t n = num_cols_d;
        int64_t k = (opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a;

        auto validArgs = validateMatmulArgs(m,
                                            n,
                                            k,
                                            alpha[i],
                                            A[i],
                                            B[i],
                                            beta[i],
                                            C[i],
                                            D[i],
                                            matA[i]->type,
                                            matB[i]->type,
                                            matC[i]->type,
                                            matD[i]->type,
                                            matmul_descr[i]->compute_type,
                                            matmul_descr[i]->op_A,
                                            matmul_descr[i]->op_B,
                                            num_batches_a,
                                            num_batches_b,
                                            num_batches_c,
                                            num_batches_d,
                                            batch_stride_a,
                                            batch_stride_b,
                                            batch_stride_c,
                                            batch_stride_d,
                                            matmul_descr[i]->pointermode);
        if(validArgs == rocblaslt_status_success)
            continue;

        void*              bias = nullptr;
        hipDataType        bias_type;
        hipDataType        aux_type;
        void*              scaleAlphaVec = nullptr;
        void*              E             = nullptr;
        int64_t            lde, batch_stride_e;
        bool               gradient;
        rocblaslt_epilogue epilogue    = matmul_descr[i]->epilogue;
        const void*        alphaVecPtr = matmul_descr[i]->pointermode ? alpha[i] : nullptr;
        if(validArgs == rocblaslt_status_continue)
            validArgs = rocblaslt_epilogue_valid_args(epilogue, // add alpha
                                                      num_rows_d,
                                                      num_cols_d,
                                                      matD[i]->type,
                                                      matmul_descr[i]->bias_type,
                                                      matmul_descr[i]->e,
                                                      matmul_descr[i]->aux_type,
                                                      matmul_descr[i]->lde,
                                                      matmul_descr[i]->stride_e,
                                                      matmul_descr[i]->bias,
                                                      alphaVecPtr,
                                                      alpha[i],
                                                      matmul_descr[i]->scaleAType,
                                                      matmul_descr[i]->scaleBType,
                                                      matmul_descr[i]->scaleABlockRowSize,
                                                      matmul_descr[i]->scaleABlockColSize,
                                                      matmul_descr[i]->scaleBBlockRowSize,
                                                      matmul_descr[i]->scaleBBlockColSize,
                                                      E,
                                                      aux_type,
                                                      lde,
                                                      batch_stride_e,
                                                      bias,
                                                      bias_type,
                                                      scaleAlphaVec,
                                                      gradient);
        if(validArgs != rocblaslt_status_continue)
            return validArgs;

        const void* alphaTmp = nullptr;
        memset(alpha_1[i].data(), 0, sizeof(int8_t) * 16);
        if(scaleAlphaVec)
        {
            setTo1(compute_type, (void*)alpha_1[i].data(), &alphaTmp);
        }
        else
        {
            alphaTmp = alpha[i];
        }

        problemtype.push_back({matmul_descr[i]->op_A,
                               matmul_descr[i]->op_B,
                               matA[i]->type,
                               matB[i]->type,
                               matC[i]->type,
                               matD[i]->type,
                               compute_type});

        bias_type_vec.push_back(bias_type);
        aux_type_vec.push_back(aux_type);
        epilogue_vec.push_back(epilogue);
        bias_vec.push_back(bias);
        scaleA_vec.push_back(matmul_descr[i]->scaleA);
        scaleB_vec.push_back(matmul_descr[i]->scaleB);
        scaleC_vec.push_back(matmul_descr[i]->scaleC);
        scaleD_vec.push_back(matmul_descr[i]->scaleD);
        scaleE_vec.push_back(matmul_descr[i]->scaleE);
        scaleAlpha_vec.push_back(scaleAlphaVec);
        amaxD_vec.push_back(matmul_descr[i]->amaxD);

        // matrix A
        // int64_t           num_rows_a     = matA[i]->m;
        // int64_t           num_cols_a     = matA[i]->n;
        lda_vec.push_back(lda);
        batch_stride_a_vec.push_back(batch_stride_a);
        num_batches_a_vec.push_back(num_batches_a);

        // matrix B
        ldb_vec.push_back(ldb);
        batch_stride_b_vec.push_back(batch_stride_b);
        num_batches_b_vec.push_back(num_batches_b);

        // matrix C
        ldc_vec.push_back(ldc);
        batch_stride_c_vec.push_back(batch_stride_c);
        num_batches_c_vec.push_back(num_batches_c);

        // matrix D
        // int64_t           num_rows_d     = matD[i]->m;
        // int64_t           num_cols_d     = matD[i]->n;
        ldd_vec.push_back(ldd);
        batch_stride_d_vec.push_back(batch_stride_d);
        num_batches_d_vec.push_back(num_batches_d);

        lde_vec.push_back(lde);
        batch_stride_e_vec.push_back(batch_stride_e);
        num_batches_e_vec.push_back(num_batches_d);

        m_vec.push_back(num_rows_d);
        n_vec.push_back(num_cols_d);
        k_vec.push_back((opA == HIPBLAS_OP_N) ? num_cols_a : num_rows_a);

        A_vec.push_back(A[i]);
        B_vec.push_back(B[i]);
        C_vec.push_back(C[i]);
        D_vec.push_back(D[i]);
        E_vec.push_back(E);
        alpha_vec.push_back(alphaTmp);
        beta_vec.push_back(beta[i]);

        gradient_vec.push_back(gradient);
    }

    bool strided_batch = true;
    bool grouped_gemm  = true;

    std::vector<RocblasltContractionProblem> problems;
    for(int i = 0; i < m_vec.size(); i++)
    {
        bool swizzleA
            = matA[i]->order != HIPBLASLT_ORDER_COL && matA[i]->order != HIPBLASLT_ORDER_ROW;
        bool swizzleB
            = matB[i]->order != HIPBLASLT_ORDER_COL && matB[i]->order != HIPBLASLT_ORDER_ROW;
        problems.push_back(RocblasltContractionProblem{opA,
                                                       opB,
                                                       m_vec[i],
                                                       n_vec[i],
                                                       k_vec[i],
                                                       alpha_vec[i],
                                                       type_a,
                                                       A_vec[i],
                                                       nullptr,
                                                       lda_vec[i],
                                                       batch_stride_a_vec[i],
                                                       type_b,
                                                       B_vec[i],
                                                       nullptr,
                                                       ldb_vec[i],
                                                       batch_stride_b_vec[i],
                                                       beta_vec[i],
                                                       type_c,
                                                       C_vec[i],
                                                       nullptr,
                                                       ldc_vec[i],
                                                       batch_stride_c_vec[i],
                                                       type_d,
                                                       D_vec[i],
                                                       nullptr,
                                                       ldd_vec[i],
                                                       batch_stride_d_vec[i],
                                                       E_vec[i],
                                                       nullptr,
                                                       lde_vec[i],
                                                       batch_stride_e_vec[i],
                                                       num_batches_a_vec[i],
                                                       strided_batch,
                                                       grouped_gemm,
                                                       gradient_vec[i],
                                                       compute_type,
                                                       matmul_descr[i]->scale_type,
                                                       bias_vec[i],
                                                       scaleA_vec[i],
                                                       scaleB_vec[i],
                                                       scaleC_vec[i],
                                                       scaleD_vec[i],
                                                       scaleE_vec[i],
                                                       scaleAlpha_vec[i],
                                                       matmul_descr[i]->scaleAType,
                                                       matmul_descr[i]->scaleBType,
                                                       matmul_descr[i]->scaleABlockRowSize,
                                                       matmul_descr[i]->scaleABlockColSize,
                                                       matmul_descr[i]->scaleBBlockRowSize,
                                                       matmul_descr[i]->scaleBBlockColSize,
                                                       bias_type_vec[i],
                                                       aux_type_vec[i],
                                                       epilogue_vec[i],
                                                       amaxD_vec[i],
                                                       nullptr,
                                                       0,
                                                       matmul_descr[i]->act0,
                                                       matmul_descr[i]->act1,
                                                       0,
                                                       (char*)handle->Synchronizer+(409600*i*sizeof(int)),
                                                       swizzleA,
                                                       swizzleB});
    }
    return groupedGemmCreate(problems, gemmData, gemmCount);
}

/********************************************************************************
 * \brief
 *******************************************************************************/
rocblaslt_status rocblaslt_matmul(rocblaslt_handle             handle,
                                  rocblaslt_matmul_desc        matmul_descr,
                                  const void*                  alpha,
                                  const void*                  A,
                                  rocblaslt_matrix_layout      matA,
                                  const void*                  B,
                                  rocblaslt_matrix_layout      matB,
                                  const void*                  beta,
                                  const void*                  C,
                                  rocblaslt_matrix_layout      matC,
                                  void*                        D,
                                  rocblaslt_matrix_layout      matD,
                                  const rocblaslt_matmul_algo* algo,
                                  void*                        workspace,
                                  size_t                       workspaceSizeInBytes,
                                  hipStream_t                  stream)

{
    // Check if handle is valid
    if(handle == nullptr || matmul_descr == nullptr || matA == nullptr || matB == nullptr
       || matC == nullptr || matD == nullptr)
    {
        log_error(__func__, "invalid handle pointer");
        return rocblaslt_status_invalid_handle;
    }

    // Update for the valid case: ((alpha_in_host && alpha=0) && (A=NULL || B=NULL))
    bool alpha_A_B_violation
        = (!alpha || ((matmul_descr->pointermode || (*((float*)alpha))) && (!A || !B)));
    // Check if pointer is valid
    if(alpha == nullptr || beta == nullptr || C == nullptr || D == nullptr || alpha_A_B_violation)
    {
        log_error(__func__, "invalid data pointer");
        return rocblaslt_status_invalid_pointer;
    }
    if(workspace == nullptr && workspaceSizeInBytes > 0)
    {
        log_error(__func__, "invalid workspace pointer");
        return rocblaslt_status_invalid_pointer;
    }

    if(matC->type != matD->type)
    {
        log_error(__func__, "invalid  matrix datatype");
        return rocblaslt_status_type_mismatch;
    }

    if(get_logger_layer_mode() != rocblaslt_layer_mode_none)
    {
        log_api(__func__,
                "A",
                A,
                "Adesc",
                matA,
                "B",
                B,
                "Bdesc",
                matB,
                "C",
                C,
                "Cdesc",
                matC,
                "D",
                D,
                "Ddesc",
                matD,
                "computeDesc",
                matmul_descr,
                "algo",
                algo,
                "workSpace",
                workspace,
                "workSpaceSizeInBytes",
                workspaceSizeInBytes,
                "stream",
                stream);
    }

    if(get_logger_layer_mode() != rocblaslt_layer_mode_none)
    {
        log_trace(__func__,
                  "A",
                  A,
                  "Adesc",
                  rocblaslt_matrix_layout_to_string(matA),
                  "B",
                  B,
                  "Bdesc",
                  rocblaslt_matrix_layout_to_string(matB),
                  "C",
                  C,
                  "Cdesc",
                  rocblaslt_matrix_layout_to_string(matC),
                  "D",
                  D,
                  "Ddesc",
                  rocblaslt_matrix_layout_to_string(matD),
                  "computeDesc",
                  rocblaslt_matmul_desc_to_string(matmul_descr),
                  "workSpace",
                  workspace,
                  "workSpaceSizeInBytes",
                  workspaceSizeInBytes,
                  (matmul_descr->pointermode) ? "alphaVector" : "alpha",
                  *(reinterpret_cast<const float*>(
                      alpha)), // TODO: Add casts for f16 and int types of alpha.
                  "beta",
                  *(reinterpret_cast<const float*>(beta)),
                  "stream",
                  stream);
    }
    return rocblaslt_matmul_impl(handle,
                                 matmul_descr,
                                 A,
                                 B,
                                 C,
                                 D,
                                 matA,
                                 matB,
                                 matC,
                                 matD,
                                 alpha,
                                 beta,
                                 algo,
                                 workspace,
                                 workspaceSizeInBytes,
                                 stream);
}
#ifdef __cplusplus
}
#endif

template <typename ProblemType, typename Epilogue, typename Inputs>
rocblaslt_status rocblaslt_gemm_create_cpp_impl_2(const rocblaslt_handle handle,
                                                  int64_t                m,
                                                  int64_t                n,
                                                  int64_t                b,
                                                  int64_t                k,
                                                  int64_t                lda,
                                                  int64_t                ldb,
                                                  int64_t                ldc,
                                                  int64_t                ldd,
                                                  int64_t                batch_stride_a,
                                                  int64_t                batch_stride_b,
                                                  int64_t                batch_stride_c,
                                                  int64_t                batch_stride_d,
                                                  Epilogue&              rocEpilogue,
                                                  Inputs&                inputs,
                                                  ProblemType&           problemtype,
                                                  std::shared_ptr<void>& gemmData,
                                                  size_t&                gemmCount)
{
    // Internal assign
    void*                   A             = inputs.a;
    void*                   B             = inputs.b;
    void*                   C             = inputs.c;
    void*                   D             = inputs.d;
    void*                   alpha         = inputs.alpha;
    void*                   beta          = inputs.beta;
    void*                   scaleA        = inputs.scaleA;
    void*                   scaleB        = inputs.scaleB;
    void*                   scaleC        = inputs.scaleC;
    void*                   scaleD        = inputs.scaleD;
    void*                   scaleE        = inputs.scaleE;
    void*                   amaxD         = nullptr;
    hipblasOperation_t&     opA           = problemtype.op_a;
    hipblasOperation_t&     opB           = problemtype.op_b;
    hipDataType&            type_a        = problemtype.type_a;
    hipDataType&            type_b        = problemtype.type_b;
    hipDataType&            type_c        = problemtype.type_c;
    hipDataType&            type_d        = problemtype.type_d;
    int                     num_batches_a = b;
    rocblaslt_compute_type& compute_type  = problemtype.type_compute;
    rocblaslt_epilogue&     epilogue      = rocEpilogue.mode;

    // Others
    bool strided_batch = true;
    bool grouped_gemm  = false;

    bool swizzleA
        = problemtype.order_a != HIPBLASLT_ORDER_COL && problemtype.order_a != HIPBLASLT_ORDER_ROW;
    bool swizzleB
        = problemtype.order_b != HIPBLASLT_ORDER_COL && problemtype.order_b != HIPBLASLT_ORDER_ROW;

    auto status = validateMatmulArgs(m,
                                     n,
                                     k,
                                     inputs.alpha,
                                     inputs.a,
                                     inputs.b,
                                     inputs.beta,
                                     inputs.c,
                                     inputs.d,
                                     type_a,
                                     type_b,
                                     type_c,
                                     type_d,
                                     compute_type,
                                     opA,
                                     opB,
                                     b,
                                     b,
                                     b,
                                     b,
                                     batch_stride_a,
                                     batch_stride_b,
                                     batch_stride_c,
                                     batch_stride_d);

    void *      bias = nullptr, *scaleAlphaVec = nullptr, *E = nullptr;
    int64_t     lde = 0, batch_stride_e = 0;
    hipDataType bias_type = HIPBLASLT_DATATYPE_INVALID;
    hipDataType aux_type  = HIPBLASLT_DATATYPE_INVALID;
    bool        gradient  = false;

    if(status == rocblaslt_status_continue)
    {
        status = rocblaslt_epilogue_valid_args(rocEpilogue.mode,
                                               m,
                                               n,
                                               problemtype.type_d,
                                               rocEpilogue.bias_data_type,
                                               inputs.aux,
                                               rocEpilogue.aux_data_type,
                                               rocEpilogue.aux_ld,
                                               rocEpilogue.aux_stride,
                                               inputs.bias,
                                               inputs.scaleAlphaVec,
                                               inputs.alpha,
                                               rocEpilogue.scaling_a_type,
                                               rocEpilogue.scaling_b_type,
                                               // TODO: these scale block sizes might need to be set
                                               0, /* scaleABlockRowSize */
                                               0, /* scaleABlockColSize */
                                               0, /* scaleBBlockRowSize */
                                               0, /* scaleBBlockColSize */
                                               E,
                                               aux_type,
                                               lde,
                                               batch_stride_e,
                                               bias,
                                               bias_type,
                                               scaleAlphaVec,
                                               gradient);
    }
    if(status != rocblaslt_status_continue)
        return status;

    int8_t alpha_1[16] = {0}; // use dScaleAlphaVec instead, original alpha => 1.0
    if(scaleAlphaVec)
    {
        setTo1(compute_type, (void*)alpha_1, (const void**)&alpha);
    }

    if constexpr(std::is_same<Inputs, rocblaslt::RocGemmInputsV2>::value)
    {
        amaxD = inputs.amaxD;
    }

    // // check alignment of pointers before casting
    // if(!isAligned(a, sizeof(TiA)) || !isAligned(b, sizeof(TiB)) || !isAligned(c, sizeof(To))
    //    || !isAligned(d, sizeof(To)))
    // {
    //     std::cerr << "memmory is not aligned" << std::endl;
    //     return rocblaslt_status_invalid_size;
    // }

    RocblasltContractionProblem problem{
        opA,
        opB,
        m,
        n,
        k,
        alpha,
        type_a,
        A,
        nullptr,
        lda,
        batch_stride_a,
        type_b,
        B,
        nullptr,
        ldb,
        batch_stride_b,
        beta,
        type_c,
        C,
        nullptr,
        ldc,
        batch_stride_c,
        type_d,
        D,
        nullptr,
        ldd,
        batch_stride_d,
        E,
        nullptr,
        lde,
        batch_stride_e,
        num_batches_a,
        strided_batch,
        grouped_gemm,
        gradient,
        compute_type,
        HIPBLASLT_DATATYPE_INVALID,
        bias,
        scaleA,
        scaleB,
        scaleC,
        scaleD,
        scaleE,
        scaleAlphaVec,
        static_cast<RocblasltContractionProblem::ScalingFormat>(rocEpilogue.scaling_a_type),
        static_cast<RocblasltContractionProblem::ScalingFormat>(rocEpilogue.scaling_b_type),
        // TODO: these scale block sizes might need to be set
        0, /* scaleABlockRowSize */
        0, /* scaleABlockColSize */
        0, /* scaleBBlockRowSize */
        0, /* scaleBBlockColSize */
        bias_type,
        aux_type,
        epilogue,
        amaxD,
        nullptr,
        0,
        rocEpilogue.act0,
        rocEpilogue.act1,
        0,
        handle->Synchronizer,
        swizzleA,
        swizzleB};
    return gemmCreate(problem, gemmData, gemmCount);
}

rocblaslt_status rocblaslt_gemm_create_cpp(const rocblaslt_handle           handle,
                                           int64_t                          m,
                                           int64_t                          n,
                                           int64_t                          b,
                                           int64_t                          k,
                                           int64_t                          lda,
                                           int64_t                          ldb,
                                           int64_t                          ldc,
                                           int64_t                          ldd,
                                           int64_t                          strideA,
                                           int64_t                          strideB,
                                           int64_t                          strideC,
                                           int64_t                          strideD,
                                           rocblaslt::RocGemmEpilogueV2&    epilogue,
                                           rocblaslt::RocGemmInputsV2&      inputs,
                                           rocblaslt::RocGemmProblemTypeV2& problemtype,
                                           std::shared_ptr<void>&           gemmData,
                                           size_t&                          gemmCount)
{
    return rocblaslt_gemm_create_cpp_impl_2(handle,
                                            m,
                                            n,
                                            b,
                                            k,
                                            lda,
                                            ldb,
                                            ldc,
                                            ldd,
                                            strideA,
                                            strideB,
                                            strideC,
                                            strideD,
                                            epilogue,
                                            inputs,
                                            problemtype,
                                            gemmData,
                                            gemmCount);
}

rocblaslt_status rocblaslt_gemm_create_cpp(const rocblaslt_handle           handle,
                                           rocblaslt_matmul_desc            matmul_descr,
                                           const void*                      alpha,
                                           const void*                      A,
                                           rocblaslt_matrix_layout          matA,
                                           const void*                      B,
                                           rocblaslt_matrix_layout          matB,
                                           const void*                      beta,
                                           const void*                      C,
                                           rocblaslt_matrix_layout          matC,
                                           void*                            D,
                                           rocblaslt_matrix_layout          matD,
                                           rocblaslt::RocGemmProblemTypeV2& problemtype,
                                           std::shared_ptr<void>&           gemmData,
                                           size_t&                          gemmCount)
{
    // Check if handle is valid
    if(matmul_descr == nullptr || matA == nullptr || matB == nullptr || matC == nullptr
       || matD == nullptr)
    {
        log_error(__func__, "invalid handle pointer");
        return rocblaslt_status_invalid_handle;
    }

    if(matC->type != matD->type)
    {
        log_error(__func__, "invalid matrix datatype");
        return rocblaslt_status_type_mismatch;
    }

    return rocblaslt_gemm_create_cpp_impl(handle,
                                          matmul_descr,
                                          A,
                                          B,
                                          C,
                                          D,
                                          matA,
                                          matB,
                                          matC,
                                          matD,
                                          alpha,
                                          beta,
                                          problemtype,
                                          gemmData,
                                          gemmCount);
}

template <typename ProblemType, typename Epilogue, typename Inputs>
rocblaslt_status rocblaslt_groupedgemm_create_cpp_impl_2(const rocblaslt_handle    handle,
                                                         std::vector<int64_t>&     m,
                                                         std::vector<int64_t>&     n,
                                                         std::vector<int64_t>&     b,
                                                         std::vector<int64_t>&     k,
                                                         std::vector<int64_t>&     lda,
                                                         std::vector<int64_t>&     ldb,
                                                         std::vector<int64_t>&     ldc,
                                                         std::vector<int64_t>&     ldd,
                                                         std::vector<int64_t>&     strideA,
                                                         std::vector<int64_t>&     strideB,
                                                         std::vector<int64_t>&     strideC,
                                                         std::vector<int64_t>&     strideD,
                                                         std::vector<Epilogue>&    rocEpilogue,
                                                         std::vector<Inputs>&      inputs,
                                                         std::vector<ProblemType>& problemtype,
                                                         std::shared_ptr<void>&    gemmData,
                                                         size_t&                   gemmCount)
{
    hipblasOperation_t     opA          = problemtype[0].op_a;
    hipblasOperation_t     opB          = problemtype[0].op_b;
    rocblaslt_compute_type compute_type = problemtype[0].type_compute;
    hipDataType            type_a       = problemtype[0].type_a;
    hipDataType            type_b       = problemtype[0].type_b;
    hipDataType            type_c       = problemtype[0].type_c;
    hipDataType            type_d       = problemtype[0].type_d;
    hipblasLtOrder_t       orderA       = problemtype[0].order_a;
    hipblasLtOrder_t       orderB       = problemtype[0].order_b;

    std::vector<const void*>        A_vec, B_vec, C_vec, alpha_vec, beta_vec;
    std::vector<void*>              D_vec, E_vec, amaxD_vec;
    std::vector<const void*>        bias_vec;
    std::vector<const void*>        scaleA_vec;
    std::vector<const void*>        scaleB_vec;
    std::vector<const void*>        scaleC_vec;
    std::vector<const void*>        scaleD_vec;
    std::vector<const void*>        scaleE_vec;
    std::vector<const void*>        scaleAlpha_vec;
    std::vector<hipDataType>        bias_type_vec;
    std::vector<hipDataType>        aux_type_vec;
    std::vector<rocblaslt_epilogue> epilogue_vec;

    std::vector<int64_t> lde_vec, batch_stride_e_vec, num_batches_e_vec;
    std::vector<bool>    gradient_vec;

    std::vector<std::array<int8_t, 16>> alpha_1(m.size());

    for(int i = 0; i < m.size(); i++)
    {
        auto validArgs = validateMatmulArgs(m[i],
                                            n[i],
                                            k[i],
                                            inputs[i].alpha,
                                            inputs[i].a,
                                            inputs[i].b,
                                            inputs[i].beta,
                                            inputs[i].c,
                                            inputs[i].d,
                                            type_a,
                                            type_b,
                                            type_c,
                                            type_d,
                                            compute_type,
                                            opA,
                                            opB,
                                            b[i],
                                            b[i],
                                            b[i],
                                            b[i],
                                            strideA[i],
                                            strideB[i],
                                            strideC[i],
                                            strideD[i]);
        if(validArgs == rocblaslt_status_success)
            continue;

        void*       bias = nullptr;
        hipDataType bias_type;
        hipDataType aux_type;
        void*       scaleAlphaVec = nullptr;
        void*       E             = nullptr;
        int64_t     lde, batch_stride_e;
        bool        gradient;

        int                iIdx     = (rocEpilogue.size() <= i) ? rocEpilogue.size() - 1 : i;
        int                iIdx2    = (problemtype.size() <= i) ? problemtype.size() - 1 : i;
        rocblaslt_epilogue epilogue = rocEpilogue[iIdx].mode;
        if(validArgs == rocblaslt_status_continue)
        {
            validArgs = rocblaslt_epilogue_valid_args(
                epilogue,
                m[i],
                n[i],
                problemtype[iIdx2].type_d,
                rocEpilogue[iIdx].bias_data_type,
                inputs[i].aux,
                rocEpilogue[iIdx].aux_data_type,
                rocEpilogue[iIdx].aux_ld,
                rocEpilogue[iIdx].aux_stride,
                inputs[i].bias,
                inputs[i].scaleAlphaVec,
                inputs[i].alpha,
                static_cast<RocblasltContractionProblem::ScalingFormat>(
                    rocEpilogue[iIdx].scaling_a_type),
                static_cast<RocblasltContractionProblem::ScalingFormat>(
                    rocEpilogue[iIdx].scaling_b_type),
                // TODO: these scale block sizes might need to be set
                0, /* scaleABlockRowSize */
                0, /* scaleABlockColSize */
                0, /* scaleBBlockRowSize */
                0, /* scaleBBlockColSize */
                E,
                aux_type,
                lde,
                batch_stride_e,
                bias,
                bias_type,
                scaleAlphaVec,
                gradient);
        }
        if(validArgs != rocblaslt_status_continue)
            return validArgs;

        const void* alphaTmp = nullptr;
        memset(alpha_1[i].data(), 0, sizeof(int8_t) * 16);
        if(scaleAlphaVec)
        {
            setTo1(compute_type, (void*)alpha_1[i].data(), &alphaTmp);
        }
        else
        {
            alphaTmp = inputs[i].alpha;
        }

        bias_type_vec.push_back(bias_type);
        aux_type_vec.push_back(aux_type);
        epilogue_vec.push_back(epilogue);
        bias_vec.push_back(bias);
        scaleA_vec.push_back(inputs[i].scaleA);
        scaleB_vec.push_back(inputs[i].scaleB);
        scaleC_vec.push_back(inputs[i].scaleC);
        scaleD_vec.push_back(inputs[i].scaleD);
        scaleE_vec.push_back(inputs[i].scaleE);
        scaleAlpha_vec.push_back(scaleAlphaVec);

        if constexpr(std::is_same<Inputs, rocblaslt::RocGemmInputsV2>::value)
            amaxD_vec.push_back(inputs[i].amaxD);
        else
            amaxD_vec.push_back(nullptr);

        A_vec.push_back(inputs[i].a);
        B_vec.push_back(inputs[i].b);
        C_vec.push_back(inputs[i].c);
        D_vec.push_back(inputs[i].d);
        E_vec.push_back(E);
        alpha_vec.push_back(alphaTmp);
        beta_vec.push_back(inputs[i].beta);

        lde_vec.push_back(lde);
        batch_stride_e_vec.push_back(batch_stride_e);
        gradient_vec.push_back(gradient);
    }

    bool strided_batch = true;
    bool grouped_gemm  = true;

    bool swizzleA = orderA != HIPBLASLT_ORDER_COL && orderA != HIPBLASLT_ORDER_ROW;
    bool swizzleB = orderB != HIPBLASLT_ORDER_COL && orderB != HIPBLASLT_ORDER_ROW;

    std::vector<RocblasltContractionProblem> problems;
    for(int i = 0; i < m.size(); i++)
    {
        int iIdx = (rocEpilogue.size() <= i) ? rocEpilogue.size() - 1 : i;
        problems.push_back(
            RocblasltContractionProblem{opA,
                                        opB,
                                        m[i],
                                        n[i],
                                        k[i],
                                        alpha_vec[i],
                                        type_a,
                                        A_vec[i],
                                        nullptr,
                                        lda[i],
                                        strideA[i],
                                        type_b,
                                        B_vec[i],
                                        nullptr,
                                        ldb[i],
                                        strideB[i],
                                        beta_vec[i],
                                        type_c,
                                        C_vec[i],
                                        nullptr,
                                        ldc[i],
                                        strideC[i],
                                        type_d,
                                        D_vec[i],
                                        nullptr,
                                        ldd[i],
                                        strideD[i],
                                        E_vec[i],
                                        nullptr,
                                        lde_vec[i],
                                        batch_stride_e_vec[i],
                                        b[i],
                                        strided_batch,
                                        grouped_gemm,
                                        gradient_vec[i],
                                        compute_type,
                                        HIPBLASLT_DATATYPE_INVALID,
                                        bias_vec[i],
                                        scaleA_vec[i],
                                        scaleB_vec[i],
                                        scaleC_vec[i],
                                        scaleD_vec[i],
                                        scaleE_vec[i],
                                        scaleAlpha_vec[i],
                                        static_cast<RocblasltContractionProblem::ScalingFormat>(
                                            rocEpilogue[iIdx].scaling_a_type),
                                        static_cast<RocblasltContractionProblem::ScalingFormat>(
                                            rocEpilogue[iIdx].scaling_b_type),
                                        // TODO: these scale block sizes might need to be set
                                        0, /* scaleABlockRowSize */
                                        0, /* scaleABlockColSize */
                                        0, /* scaleBBlockRowSize */
                                        0, /* scaleBBlockColSize */
                                        bias_type_vec[i],
                                        aux_type_vec[i],
                                        epilogue_vec[i],
                                        amaxD_vec[i],
                                        nullptr,
                                        0,
                                        rocEpilogue[iIdx].act0,
                                        rocEpilogue[iIdx].act1,
                                        0,
                                        (char*)handle->Synchronizer+(409600*i*sizeof(int)),
                                        swizzleA,
                                        swizzleB});
    }
    return groupedGemmCreate(problems, gemmData, gemmCount);
}

rocblaslt_status
    rocblaslt_groupedgemm_create_cpp(const rocblaslt_handle                        handle,
                                     std::vector<int64_t>&                         m,
                                     std::vector<int64_t>&                         n,
                                     std::vector<int64_t>&                         b,
                                     std::vector<int64_t>&                         k,
                                     std::vector<int64_t>&                         lda,
                                     std::vector<int64_t>&                         ldb,
                                     std::vector<int64_t>&                         ldc,
                                     std::vector<int64_t>&                         ldd,
                                     std::vector<int64_t>&                         strideA,
                                     std::vector<int64_t>&                         strideB,
                                     std::vector<int64_t>&                         strideC,
                                     std::vector<int64_t>&                         strideD,
                                     std::vector<rocblaslt::RocGemmEpilogueV2>&    epilogue,
                                     std::vector<rocblaslt::RocGemmInputsV2>&      inputs,
                                     std::vector<rocblaslt::RocGemmProblemTypeV2>& problemtype,
                                     std::shared_ptr<void>&                        gemmData,
                                     size_t&                                       gemmCount)
{
    if(problemtype.size() != 1)
    {
        log_error(__func__, "Currently only supports same problem type for grouped gemm.");
        return rocblaslt_status_invalid_value;
    }
    return rocblaslt_groupedgemm_create_cpp_impl_2(handle,
                                                   m,
                                                   n,
                                                   b,
                                                   k,
                                                   lda,
                                                   ldb,
                                                   ldc,
                                                   ldd,
                                                   strideA,
                                                   strideB,
                                                   strideC,
                                                   strideD,
                                                   epilogue,
                                                   inputs,
                                                   problemtype,
                                                   gemmData,
                                                   gemmCount);
}

rocblaslt_status
    rocblaslt_groupedgemm_create_cpp(const rocblaslt_handle                        handle,
                                     std::vector<rocblaslt_matmul_desc>&           matmul_descr,
                                     std::vector<const void*>&                     alpha,
                                     std::vector<const void*>&                     A,
                                     std::vector<rocblaslt_matrix_layout>&         matA,
                                     std::vector<const void*>&                     B,
                                     std::vector<rocblaslt_matrix_layout>&         matB,
                                     std::vector<const void*>&                     beta,
                                     std::vector<const void*>&                     C,
                                     std::vector<rocblaslt_matrix_layout>&         matC,
                                     std::vector<void*>&                           D,
                                     std::vector<rocblaslt_matrix_layout>&         matD,
                                     std::vector<rocblaslt::RocGemmProblemTypeV2>& problemtype,
                                     std::shared_ptr<void>&                        gemmData,
                                     size_t&                                       gemmCount)
{
    for(int i = 0; i < matmul_descr.size(); i++)
    {
        // Check if handle is valid
        if(matmul_descr[i] == nullptr || matA[i] == nullptr || matB[i] == nullptr
           || matC[i] == nullptr || matD[i] == nullptr)
        {
            log_error(__func__, "invalid handle pointer");
            return rocblaslt_status_invalid_handle;
        }

        if(matA[i]->type != matB[i]->type || matC[i]->type != matD[i]->type
           || matA[0]->type != matA[i]->type || matC[0]->type != matC[i]->type)
        {
            log_error(__func__, "invalid  matrix datatype");
            return rocblaslt_status_type_mismatch;
        }
    }

    return rocblaslt_groupedgemm_create_cpp_impl(handle,
                                                 matmul_descr,
                                                 A,
                                                 B,
                                                 C,
                                                 D,
                                                 matA,
                                                 matB,
                                                 matC,
                                                 matD,
                                                 alpha,
                                                 beta,
                                                 problemtype,
                                                 gemmData,
                                                 gemmCount);
}

rocblaslt_status rocblaslt_run_cpp(rocblaslt_handle       handle,
                                   rocblaslt::RocGemmType gemmType,
                                   std::shared_ptr<void>  gemmData,
                                   hipStream_t            stream,
                                   hipEvent_t             start,
                                   hipEvent_t             stop)
{
    return runKernelFromInvocation(handle, gemmType, gemmData, stream, start, stop);
}

rocblaslt_status rocblaslt_run_user_args_cpp(rocblaslt_handle       handle,
                                             rocblaslt::RocGemmType gemmType,
                                             std::shared_ptr<void>  gemmData,
                                             void*                  deviceUserArgs,
                                             hipStream_t            stream)
{
    return runKernelFromNewDeviceUserArguments(handle, gemmType, gemmData, deviceUserArgs, stream);
}

rocblaslt_status rocblaslt_run_user_args_cpp(rocblaslt_handle             handle,
                                             rocblaslt::RocGemmType       gemmType,
                                             size_t                       gemmCount,
                                             std::shared_ptr<void>        gemmData,
                                             const rocblaslt_matmul_algo& algo,
                                             void*                        deviceUserArgs,
                                             void*                        workspace,
                                             hipStream_t                  stream)
{
    return runKernelFromDeviceUserArguments(
        handle, gemmType, gemmCount, gemmData, algo, deviceUserArgs, workspace, stream);
}

rocblaslt_status rocblaslt_get_default_user_args(rocblaslt_handle       handle,
                                                 rocblaslt::RocGemmType gemmType,
                                                 std::shared_ptr<void>  gemmData,
                                                 void*                  hostDeviceUserArgs)
{
    return getDeviceUserArgumentsValuesFromContractionProblem(
        handle, gemmType, gemmData, hostDeviceUserArgs);
}

rocblaslt_status rocblaslt_makeArgument_cpp(rocblaslt_handle              handle,
                                            const rocblaslt::RocGemmType  gemmType,
                                            const rocblaslt_matmul_algo&  algo,
                                            const rocblaslt::RocTuningV2* tuning,
                                            void*                         workspace,
                                            size_t                        workspaceSizeInBytes,
                                            bool                          useUserArgs,
                                            hipStream_t                   stream,
                                            std::shared_ptr<void>         gemmData)
{
    return makeArgument(handle, gemmType, algo, tuning, workspace, workspaceSizeInBytes, useUserArgs, stream, gemmData);
}

std::string rocblaslt_get_kernel_name_from_data_cpp(rocblaslt_handle             handle,
                                                    const rocblaslt::RocGemmType gemmType,
                                                    std::shared_ptr<void>        gemmData)
{
    return getKernelNameFromData(handle, gemmType, gemmData);
}

std::string rocblaslt_get_solution_name_from_data_cpp(rocblaslt_handle             handle,
                                                      const rocblaslt::RocGemmType gemmType,
                                                      std::shared_ptr<void>        gemmData)
{
    return getSolutionNameFromData(handle, gemmType, gemmData);
}

std::string rocblaslt_get_kernel_name_from_algo(rocblaslt_handle             handle,
                                                const rocblaslt_matmul_algo& algo)
{
    return getKernelNameFromAlgoIndex(handle, algo);
}

std::string rocblaslt_get_solution_name_from_algo(rocblaslt_handle             handle,
                                                  const rocblaslt_matmul_algo& algo)
{
    return getSolutionNameFromAlgoIndex(handle, algo);
}
