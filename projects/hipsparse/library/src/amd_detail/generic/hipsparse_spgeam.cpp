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

#include <vector>

#ifdef HIPSPARSE_WITH_SPGEAM

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
    static rocsparse_spgeam_alg hipSpGEAMAlgToHCCSpGEAMAlg(hipsparseSpGEAMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEAM_ALG1:
            return rocsparse_spgeam_alg_default;
        default:
            throw "Non existent hipsparseSpGEAMAlg_t";
        }
    }

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
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_spgeam_set_input(rocHandle,
                                       descr->descr,
                                       rocsparse_spgeam_input_compute_datatype,
                                       &rocType,
                                       sizeof(rocType),
                                       nullptr));

        if(alpha != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_spgeam_set_input(rocHandle,
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

    // Only C = alpha * op(A) + beta * op(B) with non-transpose operations on CSR matrices is
    // supported. Reject any other request with a well-defined error rather than silently
    // computing an incorrect result.
    static hipsparseStatus_t checkSpGEAMSupport(hipsparseOperation_t       opA,
                                                hipsparseOperation_t       opB,
                                                hipsparseConstSpMatDescr_t matA,
                                                hipsparseConstSpMatDescr_t matB,
                                                hipsparseSpMatDescr_t      matC)
    {
        if(opA != HIPSPARSE_OPERATION_NON_TRANSPOSE || opB != HIPSPARSE_OPERATION_NON_TRANSPOSE)
        {
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        }

        hipsparseFormat_t formatA, formatB, formatC;
        RETURN_IF_HIPSPARSE_ERROR(hipsparseSpMatGetFormat(matA, &formatA));
        RETURN_IF_HIPSPARSE_ERROR(hipsparseSpMatGetFormat(matB, &formatB));
        RETURN_IF_HIPSPARSE_ERROR(hipsparseSpMatGetFormat(matC, &formatC));

        if(formatA != HIPSPARSE_FORMAT_CSR || formatB != HIPSPARSE_FORMAT_CSR
           || formatC != HIPSPARSE_FORMAT_CSR)
        {
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // rocSPARSE SpGEAM computes the union sparsity pattern of A and B independently of the scalar
    // values, whereas cuSPARSE drops the operand scaled by a zero scalar: alpha == 0 -> C = beta*B
    // (pattern of B), beta == 0 -> C = alpha*A (pattern of A), and both zero -> empty C. These
    // degenerate cases are handled here on the ROCm backend using rocsparse_spmat_scale instead of
    // rocsparse_spgeam so the behavior matches cuSPARSE.
    enum class spgeam_zero_mode
    {
        normal, // both scalars nonzero: use rocsparse_spgeam
        collapse_to_a, // beta  == 0: C = alpha * A
        collapse_to_b, // alpha == 0: C = beta  * B
        empty // both zero: C is empty
    };

    static size_t spgeamIndextypeSize(rocsparse_indextype type)
    {
        return (type == rocsparse_indextype_i64) ? sizeof(int64_t) : sizeof(int32_t);
    }

    // Read a host/device scalar (per the handle pointer mode) and test whether it is zero.
    static bool spgeamScalarIsZero(rocsparse_handle   rocHandle,
                                   const void*        scalar,
                                   rocsparse_datatype dataType)
    {
        rocsparse_pointer_mode pmode = rocsparse_pointer_mode_host;
        rocsparse_get_pointer_mode(rocHandle, &pmode);

        size_t sz = 0;
        switch(dataType)
        {
        case rocsparse_datatype_f32_r:
            sz = sizeof(float);
            break;
        case rocsparse_datatype_f64_r:
            sz = sizeof(double);
            break;
        case rocsparse_datatype_f32_c:
            sz = 2 * sizeof(float);
            break;
        case rocsparse_datatype_f64_c:
            sz = 2 * sizeof(double);
            break;
        default:
            return false;
        }

        unsigned char buffer[2 * sizeof(double)];
        const void*   host_scalar = scalar;
        if(pmode == rocsparse_pointer_mode_device)
        {
            if(hipMemcpy(buffer, scalar, sz, hipMemcpyDeviceToHost) != hipSuccess)
            {
                return false;
            }
            host_scalar = buffer;
        }

        switch(dataType)
        {
        case rocsparse_datatype_f32_r:
            return *reinterpret_cast<const float*>(host_scalar) == 0.0f;
        case rocsparse_datatype_f64_r:
            return *reinterpret_cast<const double*>(host_scalar) == 0.0;
        case rocsparse_datatype_f32_c:
        {
            const float* p = reinterpret_cast<const float*>(host_scalar);
            return p[0] == 0.0f && p[1] == 0.0f;
        }
        case rocsparse_datatype_f64_c:
        {
            const double* p = reinterpret_cast<const double*>(host_scalar);
            return p[0] == 0.0 && p[1] == 0.0;
        }
        default:
            return false;
        }
    }

    static spgeam_zero_mode spgeamZeroMode(rocsparse_handle   rocHandle,
                                           const void*        alpha,
                                           const void*        beta,
                                           rocsparse_datatype dataType)
    {
        const bool alpha_zero = spgeamScalarIsZero(rocHandle, alpha, dataType);
        const bool beta_zero  = spgeamScalarIsZero(rocHandle, beta, dataType);

        if(alpha_zero && beta_zero)
        {
            return spgeam_zero_mode::empty;
        }
        if(alpha_zero)
        {
            return spgeam_zero_mode::collapse_to_b;
        }
        if(beta_zero)
        {
            return spgeam_zero_mode::collapse_to_a;
        }
        return spgeam_zero_mode::normal;
    }

    // Analysis (nnz) step for the degenerate zero-scalar cases: set C's nonzero count and copy the
    // surviving operand's row offsets into C (or fill an empty pattern when both scalars are zero).
    static hipsparseStatus_t spgeamCollapseNnz(rocsparse_handle            rocHandle,
                                               spgeam_zero_mode            mode,
                                               rocsparse_const_spmat_descr matA,
                                               rocsparse_const_spmat_descr matB,
                                               rocsparse_spmat_descr       matC)
    {
        int64_t              rowsC, colsC, nnzC;
        void *               c_row_ptr, *c_col_ind, *c_val;
        rocsparse_indextype  c_row_type, c_col_type;
        rocsparse_index_base c_base;
        rocsparse_datatype   c_data_type;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr_get(matC,
                                                    &rowsC,
                                                    &colsC,
                                                    &nnzC,
                                                    &c_row_ptr,
                                                    &c_col_ind,
                                                    &c_val,
                                                    &c_row_type,
                                                    &c_col_type,
                                                    &c_base,
                                                    &c_data_type));

        if(mode == spgeam_zero_mode::empty)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_set_nnz(matC, 0));

            // An empty CSR matrix still needs its row offsets initialized to the index base.
            const int base_value = (c_base == rocsparse_index_base_one) ? 1 : 0;
            if(c_row_type == rocsparse_indextype_i64)
            {
                std::vector<int64_t> host_row_ptr(rowsC + 1, base_value);
                RETURN_IF_HIP_ERROR(hipMemcpy(c_row_ptr,
                                              host_row_ptr.data(),
                                              sizeof(int64_t) * (rowsC + 1),
                                              hipMemcpyHostToDevice));
            }
            else
            {
                std::vector<int32_t> host_row_ptr(rowsC + 1, base_value);
                RETURN_IF_HIP_ERROR(hipMemcpy(c_row_ptr,
                                              host_row_ptr.data(),
                                              sizeof(int32_t) * (rowsC + 1),
                                              hipMemcpyHostToDevice));
            }
            return HIPSPARSE_STATUS_SUCCESS;
        }

        rocsparse_const_spmat_descr source
            = (mode == spgeam_zero_mode::collapse_to_a) ? matA : matB;

        int64_t              rowsS, colsS, nnzS;
        const void *         s_row_ptr, *s_col_ind, *s_val;
        rocsparse_indextype  s_row_type, s_col_type;
        rocsparse_index_base s_base;
        rocsparse_datatype   s_data_type;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_csr_get(source,
                                                          &rowsS,
                                                          &colsS,
                                                          &nnzS,
                                                          &s_row_ptr,
                                                          &s_col_ind,
                                                          &s_val,
                                                          &s_row_type,
                                                          &s_col_type,
                                                          &s_base,
                                                          &s_data_type));

        // The pattern is copied verbatim, so the surviving operand and C must share the row-offset
        // index type and index base (base conversion in the collapse path is not supported).
        if(s_row_type != c_row_type || s_base != c_base)
        {
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_set_nnz(matC, nnzS));
        RETURN_IF_HIP_ERROR(hipMemcpy(c_row_ptr,
                                      s_row_ptr,
                                      spgeamIndextypeSize(c_row_type) * (rowsC + 1),
                                      hipMemcpyDeviceToDevice));

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Compute step for the degenerate zero-scalar cases: copy the surviving operand's column
    // indices into C and write C's values as the surviving scalar times the operand's values.
    static hipsparseStatus_t spgeamCollapseCompute(rocsparse_handle            rocHandle,
                                                   spgeam_zero_mode            mode,
                                                   const void*                 alpha,
                                                   const void*                 beta,
                                                   rocsparse_const_spmat_descr matA,
                                                   rocsparse_const_spmat_descr matB,
                                                   rocsparse_spmat_descr       matC)
    {
        if(mode == spgeam_zero_mode::empty)
        {
            // Nothing to fill: C's row offsets were initialized during the nnz step and nnz == 0.
            return HIPSPARSE_STATUS_SUCCESS;
        }

        rocsparse_const_spmat_descr source
            = (mode == spgeam_zero_mode::collapse_to_a) ? matA : matB;
        const void* scalar = (mode == spgeam_zero_mode::collapse_to_a) ? alpha : beta;

        int64_t              rowsC, colsC, nnzC;
        void *               c_row_ptr, *c_col_ind, *c_val;
        rocsparse_indextype  c_row_type, c_col_type;
        rocsparse_index_base c_base;
        rocsparse_datatype   c_data_type;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr_get(matC,
                                                    &rowsC,
                                                    &colsC,
                                                    &nnzC,
                                                    &c_row_ptr,
                                                    &c_col_ind,
                                                    &c_val,
                                                    &c_row_type,
                                                    &c_col_type,
                                                    &c_base,
                                                    &c_data_type));

        int64_t              rowsS, colsS, nnzS;
        const void *         s_row_ptr, *s_col_ind, *s_val;
        rocsparse_indextype  s_row_type, s_col_type;
        rocsparse_index_base s_base;
        rocsparse_datatype   s_data_type;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_const_csr_get(source,
                                                          &rowsS,
                                                          &colsS,
                                                          &nnzS,
                                                          &s_row_ptr,
                                                          &s_col_ind,
                                                          &s_val,
                                                          &s_row_type,
                                                          &s_col_type,
                                                          &s_base,
                                                          &s_data_type));

        if(s_col_type != c_col_type || s_base != c_base)
        {
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        }

        // Copy the surviving operand's column indices into C (spmat_scale writes values only).
        if(nnzS > 0)
        {
            RETURN_IF_HIP_ERROR(hipMemcpy(c_col_ind,
                                          s_col_ind,
                                          spgeamIndextypeSize(c_col_type) * nnzS,
                                          hipMemcpyDeviceToDevice));
        }

        // C.values = scalar * source.values, with the scalar's memory space taken from the handle
        // pointer mode and described on the dense vector descriptor itself.
        rocsparse_pointer_mode pmode = rocsparse_pointer_mode_host;
        rocsparse_get_pointer_mode(rocHandle, &pmode);

        rocsparse_dnvec_descr scalar_descr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnvec_descr_create_scalar(
            rocHandle, &scalar_descr, pmode, s_data_type, scalar, nullptr, nullptr));

        rocsparse_status status
            = rocsparse_spmat_scale(rocHandle, scalar_descr, source, matC, nullptr);

        rocsparse_destroy_dnvec_descr(scalar_descr);
        RETURN_IF_ROCSPARSE_ERROR(status);

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

    RETURN_IF_HIPSPARSE_ERROR(hipsparse::checkSpGEAMSupport(opA, opB, matA, matB, matC));

    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::setSpGEAMInputs(handle, spgeamDescr, opA, opB, computeType, alg, alpha, beta));

    // The rocSPARSE compute stage does not require any additional workspace, therefore the buffer
    // returned to the user is the buffer required by the analysis (nnz) stage.
    size_t analysisBufferSize = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_buffer_size((rocsparse_handle)handle,
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

    RETURN_IF_HIPSPARSE_ERROR(hipsparse::checkSpGEAMSupport(opA, opB, matA, matB, matC));

    // Degenerate zero-scalar cases collapse to a scaled copy of a single operand (matching
    // cuSPARSE) and are handled without rocsparse_spgeam.
    const rocsparse_datatype          rocType = hipsparse::hipDataTypeToHCCDataType(computeType);
    const hipsparse::spgeam_zero_mode zeroMode
        = hipsparse::spgeamZeroMode((rocsparse_handle)handle, alpha, beta, rocType);
    if(zeroMode != hipsparse::spgeam_zero_mode::normal)
    {
        return hipsparse::spgeamCollapseNnz((rocsparse_handle)handle,
                                            zeroMode,
                                            to_rocsparse_const_spmat_descr(matA),
                                            to_rocsparse_const_spmat_descr(matB),
                                            to_rocsparse_spmat_descr(matC));
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

    RETURN_IF_HIPSPARSE_ERROR(hipsparse::checkSpGEAMSupport(opA, opB, matA, matB, matC));

    // Degenerate zero-scalar cases collapse to a scaled copy of a single operand (matching
    // cuSPARSE) and are handled with rocsparse_spmat_scale instead of rocsparse_spgeam.
    const rocsparse_datatype          rocType = hipsparse::hipDataTypeToHCCDataType(computeType);
    const hipsparse::spgeam_zero_mode zeroMode
        = hipsparse::spgeamZeroMode((rocsparse_handle)handle, alpha, beta, rocType);
    if(zeroMode != hipsparse::spgeam_zero_mode::normal)
    {
        return hipsparse::spgeamCollapseCompute((rocsparse_handle)handle,
                                                zeroMode,
                                                alpha,
                                                beta,
                                                to_rocsparse_const_spmat_descr(matA),
                                                to_rocsparse_const_spmat_descr(matB),
                                                to_rocsparse_spmat_descr(matC));
    }

    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::setSpGEAMInputs(handle, spgeamDescr, opA, opB, computeType, alg, alpha, beta));

    // The compute stage copies the C row offsets computed during the analysis stage into matC and
    // fills the C column indices and values arrays. It does not require any external workspace.
    size_t computeBufferSize = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgeam_buffer_size((rocsparse_handle)handle,
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

#endif /* HIPSPARSE_WITH_SPGEAM */
