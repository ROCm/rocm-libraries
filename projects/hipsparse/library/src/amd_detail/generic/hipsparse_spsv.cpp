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

struct hipsparseSpSVDescr
{
    void* externalBuffer{};
};

hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr)
{
    *descr = new hipsparseSpSVDescr;
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr)
{
    if(descr != nullptr)
    {
        descr->externalBuffer = nullptr;
        delete descr;
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
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spsv((rocsparse_handle)handle,
                       hipsparse::hipOperationToHCCOperation(opA),
                       alpha,
                       to_rocsparse_const_spmat_descr(matA),
                       (rocsparse_const_dnvec_descr)x,
                       (const rocsparse_dnvec_descr)y,
                       hipsparse::hipDataTypeToHCCDataType(computeType),
                       hipsparse::hipSpSVAlgToHCCSpSVAlg(alg),
                       rocsparse_spsv_stage_buffer_size,
                       pBufferSizeInBytes,
                       nullptr));
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

    if(spsvDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spsv((rocsparse_handle)handle,
                                             hipsparse::hipOperationToHCCOperation(opA),
                                             alpha,
                                             to_rocsparse_const_spmat_descr(matA),
                                             (rocsparse_const_dnvec_descr)x,
                                             (const rocsparse_dnvec_descr)y,
                                             hipsparse::hipDataTypeToHCCDataType(computeType),
                                             hipsparse::hipSpSVAlgToHCCSpSVAlg(alg),
                                             rocsparse_spsv_stage_preprocess,
                                             nullptr,
                                             externalBuffer));
    spsvDescr->externalBuffer = externalBuffer;
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
    if(spsvDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }
    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_spsv((rocsparse_handle)handle,
                       hipsparse::hipOperationToHCCOperation(opA),
                       alpha,
                       to_rocsparse_const_spmat_descr(matA),
                       (rocsparse_const_dnvec_descr)x,
                       (const rocsparse_dnvec_descr)y,
                       hipsparse::hipDataTypeToHCCDataType(computeType),
                       hipsparse::hipSpSVAlgToHCCSpSVAlg(alg),
                       rocsparse_spsv_stage_compute,
                       nullptr,
                       spsvDescr->externalBuffer));
}
