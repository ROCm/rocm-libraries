/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <algorithm>

#include "../utility.h"

// hipSPARSE SpSV descriptor. The analysis lives in the owned rocSPARSE SpTRSV
// descriptor rather than in the sparse matrix descriptor, so that the solve
// does not depend on the matrix descriptor that was passed to the analysis
// (matching cuSPARSE). The operation and compute type are fixed by the analysis.
struct hipsparseSpSVDescr
{
    rocsparse_sptrsv_descr descr{};
    // rocsparse_sptrsv_descr_destroy only checks its handle for null, so the handle
    // used by the analysis is kept for hipsparseSpSV_destroyDescr, which receives none.
    rocsparse_handle     handle{};
    hipsparseOperation_t opA{};
    hipDataType          computeType{};
    void*                externalBuffer{};
    size_t               bufferSize{};
};

namespace hipsparse
{
    static hipsparseStatus_t setSpTRSVInputs(rocsparse_handle       handle,
                                             rocsparse_sptrsv_descr descr,
                                             hipsparseOperation_t   opA,
                                             hipDataType            computeType,
                                             hipsparseSpSVAlg_t     alg)
    {
        const rocsparse_sptrsv_alg      rocAlg  = hipsparse::hipSpSVAlgToHCCSpTRSVAlg(alg);
        const rocsparse_operation       rocOp   = hipsparse::hipOperationToHCCOperation(opA);
        const rocsparse_datatype        rocType = hipsparse::hipDataTypeToHCCDataType(computeType);
        const rocsparse_analysis_policy policy  = rocsparse_analysis_policy_force;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_alg, &rocAlg, sizeof(rocAlg), nullptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(
            handle, descr, rocsparse_sptrsv_input_operation, &rocOp, sizeof(rocOp), nullptr));
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sptrsv_set_input(handle,
                                       descr,
                                       rocsparse_sptrsv_input_compute_datatype,
                                       &rocType,
                                       sizeof(rocType),
                                       nullptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                             descr,
                                                             rocsparse_sptrsv_input_scalar_datatype,
                                                             &rocType,
                                                             sizeof(rocType),
                                                             nullptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(handle,
                                                             descr,
                                                             rocsparse_sptrsv_input_analysis_policy,
                                                             &policy,
                                                             sizeof(policy),
                                                             nullptr));
        return HIPSPARSE_STATUS_SUCCESS;
    }

    static hipsparseStatus_t createSpTRSVDescr(rocsparse_handle        handle,
                                               hipsparseOperation_t    opA,
                                               hipDataType             computeType,
                                               hipsparseSpSVAlg_t      alg,
                                               rocsparse_sptrsv_descr* p_descr)
    {
        rocsparse_sptrsv_descr descr{};
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_descr_create(handle, &descr, nullptr));

        const hipsparseStatus_t status = setSpTRSVInputs(handle, descr, opA, computeType, alg);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            rocsparse_sptrsv_descr_destroy(handle, descr, nullptr);
            return status;
        }

        *p_descr = descr;
        return HIPSPARSE_STATUS_SUCCESS;
    }

    // A single external buffer is used for both the analysis and the solve.
    static hipsparseStatus_t spTRSVBufferSize(rocsparse_handle           handle,
                                              rocsparse_sptrsv_descr     descr,
                                              hipsparseConstSpMatDescr_t matA,
                                              hipsparseConstDnVecDescr_t x,
                                              hipsparseConstDnVecDescr_t y,
                                              size_t*                    pBufferSizeInBytes)
    {
        size_t analysisSize = 0;
        size_t computeSize  = 0;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_buffer_size(handle,
                                                               descr,
                                                               to_rocsparse_const_spmat_descr(matA),
                                                               (rocsparse_const_dnvec_descr)x,
                                                               (rocsparse_const_dnvec_descr)y,
                                                               rocsparse_sptrsv_stage_analysis,
                                                               &analysisSize,
                                                               nullptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_buffer_size(handle,
                                                               descr,
                                                               to_rocsparse_const_spmat_descr(matA),
                                                               (rocsparse_const_dnvec_descr)x,
                                                               (rocsparse_const_dnvec_descr)y,
                                                               rocsparse_sptrsv_stage_compute,
                                                               &computeSize,
                                                               nullptr));
        *pBufferSizeInBytes = std::max({analysisSize, computeSize, sizeof(int32_t)});
        return HIPSPARSE_STATUS_SUCCESS;
    }
}

hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr)
{
    *descr = new hipsparseSpSVDescr;
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr)
{
    if(descr != nullptr)
    {
        rocsparse_status status = rocsparse_status_success;
        if(descr->descr != nullptr)
        {
            status = rocsparse_sptrsv_descr_destroy(descr->handle, descr->descr, nullptr);
        }
        delete descr;
        return hipsparse::rocSPARSEStatusToHIPStatus(status);
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     pBufferSizeInBytes)
{
    if(alpha == nullptr || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    rocsparse_handle       rocHandle = (rocsparse_handle)handle;
    rocsparse_sptrsv_descr descr{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::createSpTRSVDescr(rocHandle, opA, computeType, alg, &descr));

    const hipsparseStatus_t status
        = hipsparse::spTRSVBufferSize(rocHandle, descr, matA, x, y, pBufferSizeInBytes);
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_descr_destroy(rocHandle, descr, nullptr));
    return status;
}

hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         hipsparseConstSpMatDescr_t  matA,
                                         hipsparseConstDnVecDescr_t  x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer)
{
    if(alpha == nullptr || spsvDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // rocsparse_sptrsv runs the analysis only once per descriptor, but SpSV may be
    // analysed repeatedly. Analyse into a fresh descriptor and replace the previous
    // one only on success.
    rocsparse_handle       rocHandle = (rocsparse_handle)handle;
    rocsparse_sptrsv_descr descr{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::createSpTRSVDescr(rocHandle, opA, computeType, alg, &descr));

    size_t            bufferSize = 0;
    rocsparse_status  status     = rocsparse_status_success;
    hipsparseStatus_t hipStatus
        = hipsparse::spTRSVBufferSize(rocHandle, descr, matA, x, y, &bufferSize);
    if(hipStatus == HIPSPARSE_STATUS_SUCCESS)
    {
        status    = rocsparse_sptrsv(rocHandle,
                                  descr,
                                  to_rocsparse_const_spmat_descr(matA),
                                  (rocsparse_const_dnvec_descr)x,
                                  (const rocsparse_dnvec_descr)y,
                                  rocsparse_sptrsv_stage_analysis,
                                  bufferSize,
                                  externalBuffer,
                                  nullptr);
        hipStatus = hipsparse::rocSPARSEStatusToHIPStatus(status);
    }

    if(hipStatus != HIPSPARSE_STATUS_SUCCESS)
    {
        rocsparse_sptrsv_descr_destroy(rocHandle, descr, nullptr);
        return hipStatus;
    }

    if(spsvDescr->descr != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sptrsv_descr_destroy(rocHandle, spsvDescr->descr, nullptr));
    }

    spsvDescr->descr          = descr;
    spsvDescr->handle         = rocHandle;
    spsvDescr->opA            = opA;
    spsvDescr->computeType    = computeType;
    spsvDescr->externalBuffer = externalBuffer;
    spsvDescr->bufferSize     = bufferSize;
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      hipsparseConstSpMatDescr_t  matA,
                                      hipsparseConstDnVecDescr_t  x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr)
{
    if(alpha == nullptr || spsvDescr == nullptr || spsvDescr->descr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(opA != spsvDescr->opA || computeType != spsvDescr->computeType)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    rocsparse_handle rocHandle = (rocsparse_handle)handle;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sptrsv_set_input(rocHandle,
                                                         spsvDescr->descr,
                                                         rocsparse_sptrsv_input_scalar_alpha,
                                                         alpha,
                                                         sizeof(void*),
                                                         nullptr));

    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sptrsv(rocHandle,
                         spsvDescr->descr,
                         to_rocsparse_const_spmat_descr(matA),
                         (rocsparse_const_dnvec_descr)x,
                         (const rocsparse_dnvec_descr)y,
                         rocsparse_sptrsv_stage_compute,
                         spsvDescr->bufferSize,
                         spsvDescr->externalBuffer,
                         nullptr));
}
