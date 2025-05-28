/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../utility.h"

struct hipsparseSpSMDescr
{
    void* externalBuffer{};
};

hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr)
{
    *descr = new hipsparseSpSMDescr;
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr)
{
    if(descr != nullptr)
    {
        descr->externalBuffer = nullptr;
        delete descr;
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnMatDescr_t  matB,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpSMAlg_t          alg,
                                           hipsparseSpSMDescr_t        spsmDescr,
                                           size_t*                     pBufferSizeInBytes)
{
    if(spsmDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spsm((rocsparse_handle)handle,
                                             hipsparse::hipOperationToHCCOperation(opA),
                                             hipsparse::hipOperationToHCCOperation(opB),
                                             alpha,
                                             to_rocsparse_const_spmat_descr(matA),
                                             (rocsparse_const_dnmat_descr)matB,
                                             (const rocsparse_dnmat_descr)matC,
                                             hipsparse::hipDataTypeToHCCDataType(computeType),
                                             hipsparse::hipSpSMAlgToHCCSpSMAlg(alg),
                                             rocsparse_spsm_stage_buffer_size,
                                             pBufferSizeInBytes,
                                             nullptr));

    pBufferSizeInBytes[0] = std::max(pBufferSizeInBytes[0], sizeof(int32_t));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         hipsparseOperation_t        opB,
                                         const void*                 alpha,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseConstDnMatDescr_t  matB,
                                         const hipsparseDnMatDescr_t matC,
                                         hipDataType                 computeType,
                                         hipsparseSpSMAlg_t          alg,
                                         hipsparseSpSMDescr_t        spsmDescr,
                                         void*                       externalBuffer)
{
    if(spsmDescr == nullptr || externalBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spsm((rocsparse_handle)handle,
                                             hipsparse::hipOperationToHCCOperation(opA),
                                             hipsparse::hipOperationToHCCOperation(opB),
                                             alpha,
                                             to_rocsparse_const_spmat_descr(matA),
                                             (rocsparse_const_dnmat_descr)matB,
                                             (const rocsparse_dnmat_descr)matC,
                                             hipsparse::hipDataTypeToHCCDataType(computeType),
                                             hipsparse::hipSpSMAlgToHCCSpSMAlg(alg),
                                             rocsparse_spsm_stage_preprocess,
                                             nullptr,
                                             externalBuffer));

    spsmDescr->externalBuffer = externalBuffer;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      hipsparseOperation_t        opB,
                                      const void*                 alpha,
                                      hipsparseConstSpMatDescr_t  matA,
                                      hipsparseConstDnMatDescr_t  matB,
                                      const hipsparseDnMatDescr_t matC,
                                      hipDataType                 computeType,
                                      hipsparseSpSMAlg_t          alg,
                                      hipsparseSpSMDescr_t        spsmDescr,
                                      void*                       externalBuffer)
{
    if(spsmDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spsm((rocsparse_handle)handle,
                       hipsparse::hipOperationToHCCOperation(opA),
                       hipsparse::hipOperationToHCCOperation(opB),
                       alpha,
                       to_rocsparse_const_spmat_descr(matA),
                       (rocsparse_const_dnmat_descr)matB,
                       (const rocsparse_dnmat_descr)matC,
                       hipsparse::hipDataTypeToHCCDataType(computeType),
                       hipsparse::hipSpSMAlgToHCCSpSMAlg(alg),
                       rocsparse_spsm_stage_compute,
                       nullptr,
                       spsmDescr->externalBuffer));
}
