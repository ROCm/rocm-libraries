/*! \file */
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

#include "hipsparse.h"

#include <hip/hip_runtime_api.h>
#include <rocsparse/rocsparse.h>

#include "../utility.h"

// hipSPARSE SpGEAM descriptor. Wraps the staged rocSPARSE SpGEAM descriptor and
// caches the analysis buffer size so that it can be shared by the nnz and compute
// steps (matching the single-buffer cuSPARSE SpGEAM interface).
struct hipsparseSpGEAMDescr
{
    rocsparse_spgeam_descr descr{};
    size_t                 bufferSize{};
};

hipsparseStatus_t hipsparseSpGEAM_createDescr(hipsparseSpGEAMDescr_t* descr)
{
    if(descr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    hipsparseSpGEAMDescr_t spgeamDescr = new hipsparseSpGEAMDescr;

    rocsparse_status status = rocsparse_create_spgeam_descr(&spgeamDescr->descr);
    if(status != rocsparse_status_success)
    {
        delete spgeamDescr;
        return hipsparse::rocSPARSEStatusToHIPStatus(status);
    }

    *descr = spgeamDescr;
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpGEAM_destroyDescr(hipsparseSpGEAMDescr_t descr)
{
    if(descr != nullptr)
    {
        rocsparse_status status = rocsparse_status_success;
        if(descr->descr != nullptr)
        {
            status = rocsparse_destroy_spgeam_descr(descr->descr);
        }
        delete descr;
        return hipsparse::rocSPARSEStatusToHIPStatus(status);
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

namespace hipsparse
{
    static hipsparseStatus_t setSpGEAMInputs(hipsparseHandle_t      handle,
                                             hipsparseSpGEAMDescr_t descr,
                                             hipsparseOperation_t   opA,
                                             hipsparseOperation_t   opB,
                                             hipDataType            computeType,
                                             hipsparseSpGEAMAlg_t   alg,
                                             const void*            alpha,
                                             const void*            beta)
    {
        rocsparse_handle     rocHandle = (rocsparse_handle)handle;
        rocsparse_spgeam_alg rocAlg    = hipsparse::hipSpGEAMAlgToHCCSpGEAMAlg(alg);
        rocsparse_operation  rocOpA    = hipsparse::hipOperationToHCCOperation(opA);
        rocsparse_operation  rocOpB    = hipsparse::hipOperationToHCCOperation(opB);
        rocsparse_datatype   rocType   = hipsparse::hipDataTypeToHCCDataType(computeType);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(
            rocHandle, descr->descr, rocsparse_spgeam_input_alg, &rocAlg, sizeof(rocAlg), nullptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(rocHandle,
                                                             descr->descr,
                                                             rocsparse_spgeam_input_operation_A,
                                                             &rocOpA,
                                                             sizeof(rocOpA),
                                                             nullptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(rocHandle,
                                                             descr->descr,
                                                             rocsparse_spgeam_input_operation_B,
                                                             &rocOpB,
                                                             sizeof(rocOpB),
                                                             nullptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(rocHandle,
                                                             descr->descr,
                                                             rocsparse_spgeam_input_scalar_datatype,
                                                             &rocType,
                                                             sizeof(rocType),
                                                             nullptr));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(rocHandle,
                                                             descr->descr,
                                                             rocsparse_spgeam_input_compute_datatype,
                                                             &rocType,
                                                             sizeof(rocType),
                                                             nullptr));

        if(alpha != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(rocHandle,
                                                                 descr->descr,
                                                                 rocsparse_spgeam_input_scalar_alpha,
                                                                 alpha,
                                                                 sizeof(void*),
                                                                 nullptr));
        }

        if(beta != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_set_input(rocHandle,
                                                                 descr->descr,
                                                                 rocsparse_spgeam_input_scalar_beta,
                                                                 beta,
                                                                 sizeof(void*),
                                                                 nullptr));
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }
}

hipsparseStatus_t hipsparseSpGEAM_bufferSize(hipsparseHandle_t          handle,
                                             hipsparseOperation_t       opA,
                                             hipsparseOperation_t       opB,
                                             const void*                alpha,
                                             hipsparseConstSpMatDescr_t matA,
                                             const void*                beta,
                                             hipsparseConstSpMatDescr_t matB,
                                             hipsparseSpMatDescr_t      matC,
                                             hipDataType                computeType,
                                             hipsparseSpGEAMAlg_t       alg,
                                             hipsparseSpGEAMDescr_t     spgeamDescr,
                                             size_t*                    bufferSize)
{
    // Match cusparse error handling
    if(handle == nullptr || alpha == nullptr || beta == nullptr || matA == nullptr
       || matB == nullptr || matC == nullptr || spgeamDescr == nullptr || bufferSize == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::setSpGEAMInputs(handle, spgeamDescr, opA, opB, computeType, alg, alpha, beta));

    // The rocSPARSE compute stage does not require any additional workspace, therefore the buffer
    // returned to the user is the buffer required by the analysis (nnz) stage.
    size_t analysisBufferSize = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_spgeam_buffer_size((rocsparse_handle)handle,
                                     spgeamDescr->descr,
                                     to_rocsparse_const_spmat_descr(matA),
                                     to_rocsparse_const_spmat_descr(matB),
                                     to_rocsparse_const_spmat_descr(matC),
                                     rocsparse_spgeam_stage_analysis,
                                     &analysisBufferSize,
                                     nullptr));

    spgeamDescr->bufferSize = analysisBufferSize;

    // Ensure the user always allocates a valid (non-null) buffer pointer.
    *bufferSize = (analysisBufferSize > 0) ? analysisBufferSize : 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpGEAM_nnz(hipsparseHandle_t          handle,
                                      hipsparseOperation_t       opA,
                                      hipsparseOperation_t       opB,
                                      const void*                alpha,
                                      hipsparseConstSpMatDescr_t matA,
                                      const void*                beta,
                                      hipsparseConstSpMatDescr_t matB,
                                      hipsparseSpMatDescr_t      matC,
                                      hipDataType                computeType,
                                      hipsparseSpGEAMAlg_t       alg,
                                      hipsparseSpGEAMDescr_t     spgeamDescr,
                                      void*                      externalBuffer)
{
    if(handle == nullptr || alpha == nullptr || beta == nullptr || matA == nullptr
       || matB == nullptr || matC == nullptr || spgeamDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::setSpGEAMInputs(handle, spgeamDescr, opA, opB, computeType, alg, alpha, beta));

    // The analysis stage computes the number of non-zeros of C (stored in matC so that it can be
    // retrieved through hipsparseSpMatGetSize) as well as the internal C row offsets array (copied
    // into matC during the compute step).
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam((rocsparse_handle)handle,
                                               spgeamDescr->descr,
                                               to_rocsparse_const_spmat_descr(matA),
                                               to_rocsparse_const_spmat_descr(matB),
                                               to_rocsparse_spmat_descr(matC),
                                               rocsparse_spgeam_stage_analysis,
                                               spgeamDescr->bufferSize,
                                               externalBuffer,
                                               nullptr));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpGEAM(hipsparseHandle_t          handle,
                                  hipsparseOperation_t       opA,
                                  hipsparseOperation_t       opB,
                                  const void*                alpha,
                                  hipsparseConstSpMatDescr_t matA,
                                  const void*                beta,
                                  hipsparseConstSpMatDescr_t matB,
                                  hipsparseSpMatDescr_t      matC,
                                  hipDataType                computeType,
                                  hipsparseSpGEAMAlg_t       alg,
                                  hipsparseSpGEAMDescr_t     spgeamDescr,
                                  void*                      externalBuffer)
{
    if(handle == nullptr || alpha == nullptr || beta == nullptr || matA == nullptr
       || matB == nullptr || matC == nullptr || spgeamDescr == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::setSpGEAMInputs(handle, spgeamDescr, opA, opB, computeType, alg, alpha, beta));

    // The compute stage copies the C row offsets computed during the analysis stage into matC and
    // fills the C column indices and values arrays. It does not require any external workspace.
    size_t computeBufferSize = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_spgeam_buffer_size((rocsparse_handle)handle,
                                     spgeamDescr->descr,
                                     to_rocsparse_const_spmat_descr(matA),
                                     to_rocsparse_const_spmat_descr(matB),
                                     to_rocsparse_const_spmat_descr(matC),
                                     rocsparse_spgeam_stage_compute,
                                     &computeBufferSize,
                                     nullptr));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam((rocsparse_handle)handle,
                                               spgeamDescr->descr,
                                               to_rocsparse_const_spmat_descr(matA),
                                               to_rocsparse_const_spmat_descr(matB),
                                               to_rocsparse_spmat_descr(matC),
                                               rocsparse_spgeam_stage_compute,
                                               computeBufferSize,
                                               externalBuffer,
                                               nullptr));

    return HIPSPARSE_STATUS_SUCCESS;
}
