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

hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     pBufferSizeInBytes)
{

    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(alpha == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(matA == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecX == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(beta == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecY == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    const rocsparse_datatype  datatype  = hipsparse::hipDataTypeToHCCDataType(computeType);
    const rocsparse_operation operation = hipsparse::hipOperationToHCCOperation(opA);
    rocsparse_spmv_alg        spmv_alg  = hipsparse::hipSpMVAlgToHCCSpMVAlg(alg);

    //
    // Fallback algorithm?
    //
    if(spmv_alg == rocsparse_spmv_alg_csr_lrb)
    {
        rocsparse_matrix_type matrix_type;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_spmat_get_attribute(to_rocsparse_const_spmat_descr(matA),
                                          rocsparse_spmat_matrix_type,
                                          &matrix_type,
                                          sizeof(matrix_type)));

        if((matrix_type == rocsparse_matrix_type_symmetric)
           || (operation != rocsparse_operation_none))
        {
            spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
        }
    }
    else if((spmv_alg == rocsparse_spmv_alg_csr_adaptive)
            && (operation != rocsparse_operation_none))
    {
        spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
    }

    hipsparseSpMVDescr_st* hip_spmv_descr = matA->get_hip_spmv_descr();

    //
    // If spmv_descr alreay exists, then destroy it.
    //
    rocsparse_spmv_descr spmv_descr = hip_spmv_descr->get_spmv_descr();
    if(spmv_descr != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_spmv_descr(spmv_descr));
        hip_spmv_descr->set_spmv_descr(nullptr);
    }

    //
    // Create spmv_descr.
    //
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_spmv_descr(&spmv_descr));

    hip_spmv_descr->set_spmv_descr(spmv_descr);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                       spmv_descr,
                                                       rocsparse_spmv_input_alg,
                                                       &spmv_alg,
                                                       sizeof(spmv_alg),
                                                       nullptr));

    //
    // Set operation.
    //
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                       spmv_descr,
                                                       rocsparse_spmv_input_operation,
                                                       &operation,
                                                       sizeof(operation),
                                                       nullptr));

    //
    // Set datatypes.
    //
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                       spmv_descr,
                                                       rocsparse_spmv_input_scalar_datatype,
                                                       &datatype,
                                                       sizeof(datatype),
                                                       nullptr));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                       spmv_descr,
                                                       rocsparse_spmv_input_compute_datatype,
                                                       &datatype,
                                                       sizeof(datatype),
                                                       nullptr));

    //
    // Buffer size for the analysis phase.
    //
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size((rocsparse_handle)handle,
                                                            spmv_descr,
                                                            to_rocsparse_const_spmat_descr(matA),
                                                            (rocsparse_const_dnvec_descr)vecX,
                                                            (rocsparse_dnvec_descr)vecY,
                                                            rocsparse_v2_spmv_stage_analysis,
                                                            pBufferSizeInBytes,
                                                            nullptr));

    hip_spmv_descr->set_buffer_size_stage_analysis(pBufferSizeInBytes[0]);
    hip_spmv_descr->buffer_size_called();

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           hipsparseConstSpMatDescr_t  matA,
                                           hipsparseConstDnVecDescr_t  vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer)
{
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(alpha == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(matA == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecX == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(beta == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecY == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    const rocsparse_datatype  datatype  = hipsparse::hipDataTypeToHCCDataType(computeType);
    const rocsparse_operation operation = hipsparse::hipOperationToHCCOperation(opA);
    rocsparse_spmv_alg        spmv_alg  = hipsparse::hipSpMVAlgToHCCSpMVAlg(alg);

    //
    // Fallback algorithm?
    //
    if(spmv_alg == rocsparse_spmv_alg_csr_lrb)
    {
        rocsparse_matrix_type matrix_type;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_spmat_get_attribute(to_rocsparse_const_spmat_descr(matA),
                                          rocsparse_spmat_matrix_type,
                                          &matrix_type,
                                          sizeof(matrix_type)));

        if((matrix_type == rocsparse_matrix_type_symmetric)
           || (operation != rocsparse_operation_none))
        {
            spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
        }
    }
    else if((spmv_alg == rocsparse_spmv_alg_csr_adaptive)
            && (operation != rocsparse_operation_none))
    {
        spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
    }

    hipsparseSpMVDescr_st* hip_spmv_descr = matA->get_hip_spmv_descr();
    rocsparse_spmv_descr   spmv_descr     = hip_spmv_descr->get_spmv_descr();

    if(spmv_descr == nullptr)
    {
        //
        // Create spmv_descr.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_spmv_descr(&spmv_descr));

        hip_spmv_descr->set_spmv_descr(spmv_descr);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_alg,
                                                           &spmv_alg,
                                                           sizeof(spmv_alg),
                                                           nullptr));
        //
        // Set operation.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_operation,
                                                           &operation,
                                                           sizeof(operation),
                                                           nullptr));

        //
        // Set datatypes.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_scalar_datatype,
                                                           &datatype,
                                                           sizeof(datatype),
                                                           nullptr));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_compute_datatype,
                                                           &datatype,
                                                           sizeof(datatype),
                                                           nullptr));
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv((rocsparse_handle)handle,
                                                spmv_descr,
                                                alpha,
                                                to_rocsparse_const_spmat_descr(matA),
                                                (rocsparse_const_dnvec_descr)vecX,
                                                beta,
                                                (rocsparse_dnvec_descr)vecY,
                                                rocsparse_v2_spmv_stage_analysis,
                                                hip_spmv_descr->get_buffer_size_stage_analysis(),
                                                externalBuffer,
                                                nullptr));
    hip_spmv_descr->stage_analysis_called();
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                hipsparseConstSpMatDescr_t  matA,
                                hipsparseConstDnVecDescr_t  vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer)
{
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(alpha == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(matA == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecX == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(beta == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(vecY == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    const rocsparse_datatype  datatype  = hipsparse::hipDataTypeToHCCDataType(computeType);
    const rocsparse_operation operation = hipsparse::hipOperationToHCCOperation(opA);
    rocsparse_spmv_alg        spmv_alg  = hipsparse::hipSpMVAlgToHCCSpMVAlg(alg);

    hipsparseSpMVDescr_st* hip_spmv_descr = matA->get_hip_spmv_descr();
    rocsparse_spmv_descr   spmv_descr     = hip_spmv_descr->get_spmv_descr();

    if(spmv_descr == nullptr)
    {
        //
        // Create spmv_descr.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_spmv_descr(&spmv_descr));

        hip_spmv_descr->set_spmv_descr(spmv_descr);

        if(spmv_alg == rocsparse_spmv_alg_csr_lrb)
        {
            rocsparse_matrix_type matrix_type;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_spmat_get_attribute(to_rocsparse_const_spmat_descr(matA),
                                              rocsparse_spmat_matrix_type,
                                              &matrix_type,
                                              sizeof(matrix_type)));
            if((matrix_type == rocsparse_matrix_type_symmetric)
               || (operation != rocsparse_operation_none))
            {
                spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
            }
        }
        else if((spmv_alg == rocsparse_spmv_alg_csr_adaptive)
                && (operation != rocsparse_operation_none))
        {
            spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
        }
        else if(((spmv_alg == rocsparse_spmv_alg_csr_adaptive)
                 || (spmv_alg == rocsparse_spmv_alg_csr_lrb))
                && (hip_spmv_descr->is_stage_analysis_called() == false))
        {
            spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_alg,
                                                           &spmv_alg,
                                                           sizeof(spmv_alg),
                                                           nullptr));

        //
        // Set operation.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_operation,
                                                           &operation,
                                                           sizeof(operation),
                                                           nullptr));

        //
        // Set datatypes.
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_scalar_datatype,
                                                           &datatype,
                                                           sizeof(datatype),
                                                           nullptr));

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmv_set_input((rocsparse_handle)handle,
                                                           spmv_descr,
                                                           rocsparse_spmv_input_compute_datatype,
                                                           &datatype,
                                                           sizeof(datatype),
                                                           nullptr));
    }

    if(hip_spmv_descr->is_stage_analysis_called() == false)
    {
        //
        // No analysis has been performed, we have to call the analysis since this is a requirement for v2_spmv.
        //
        if(hip_spmv_descr->is_implicit_stage_analysis_called() == false)
        {

            if(hip_spmv_descr->is_buffer_size_called() == false)
            {
                size_t buffer_size_in_bytes;
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse_v2_spmv_buffer_size((rocsparse_handle)handle,
                                                  spmv_descr,
                                                  to_rocsparse_const_spmat_descr(matA),
                                                  (rocsparse_const_dnvec_descr)vecX,
                                                  (rocsparse_dnvec_descr)vecY,
                                                  rocsparse_v2_spmv_stage_analysis,
                                                  &buffer_size_in_bytes,
                                                  nullptr));

                hipStream_t stream{};
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_stream((rocsparse_handle)handle, &stream));

                RETURN_IF_HIP_ERROR(hipMallocAsync(
                    hip_spmv_descr->get_buffer_reference(), buffer_size_in_bytes, stream));

                RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv((rocsparse_handle)handle,
                                                            spmv_descr,
                                                            alpha,
                                                            to_rocsparse_const_spmat_descr(matA),
                                                            (rocsparse_const_dnvec_descr)vecX,
                                                            beta,
                                                            (const rocsparse_dnvec_descr)vecY,
                                                            rocsparse_v2_spmv_stage_analysis,
                                                            buffer_size_in_bytes,
                                                            hip_spmv_descr->get_buffer(),
                                                            nullptr));

                RETURN_IF_HIP_ERROR(hipFreeAsync(hip_spmv_descr->get_buffer(), stream));
                hip_spmv_descr->set_buffer(nullptr);
            }
            else
            {
                //
                // We can use the externalBuffer since the user is allocating a buffer for the analysis phase only, but the user
                // does not know it.
                //
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse_v2_spmv((rocsparse_handle)handle,
                                      spmv_descr,
                                      alpha,
                                      to_rocsparse_const_spmat_descr(matA),
                                      (rocsparse_const_dnvec_descr)vecX,
                                      beta,
                                      (const rocsparse_dnvec_descr)vecY,
                                      rocsparse_v2_spmv_stage_analysis,
                                      hip_spmv_descr->get_buffer_size_stage_analysis(),
                                      externalBuffer,
                                      nullptr));
            }

            hip_spmv_descr->implicit_stage_analysis_called();
        }

        //
        // We keep hip_spmv_descr->spmv_analysis_called = false;
        // because it hasn't been explicitly called.
        //
    }

    if(hip_spmv_descr->is_stage_compute_subsequent() == false)
    {
        //
        // Get the buffer size for the compute phase, the buffer size returned in hipsparseSpMV_bufferSize is the buffer size for the analysis phase.
        //
        size_t buffer_size_in_bytes;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_v2_spmv_buffer_size((rocsparse_handle)handle,
                                          spmv_descr,
                                          to_rocsparse_const_spmat_descr(matA),
                                          (rocsparse_const_dnvec_descr)vecX,
                                          (rocsparse_dnvec_descr)vecY,
                                          rocsparse_v2_spmv_stage_compute,
                                          &buffer_size_in_bytes,
                                          nullptr));

        hip_spmv_descr->set_buffer_size_stage_compute(buffer_size_in_bytes);

        hipStream_t stream{};
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_stream((rocsparse_handle)handle, &stream));

        RETURN_IF_HIP_ERROR(hipMallocAsync(hip_spmv_descr->get_buffer_reference(),
                                           hip_spmv_descr->get_buffer_size_stage_compute(),
                                           stream));

        hip_spmv_descr->stage_compute_subsequent();
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv((rocsparse_handle)handle,
                                                spmv_descr,
                                                alpha,
                                                to_rocsparse_const_spmat_descr(matA),
                                                (rocsparse_const_dnvec_descr)vecX,
                                                beta,
                                                (rocsparse_dnvec_descr)vecY,
                                                rocsparse_v2_spmv_stage_compute,
                                                hip_spmv_descr->get_buffer_size_stage_compute(),
                                                hip_spmv_descr->get_buffer(),
                                                nullptr));

    return HIPSPARSE_STATUS_SUCCESS;
}
