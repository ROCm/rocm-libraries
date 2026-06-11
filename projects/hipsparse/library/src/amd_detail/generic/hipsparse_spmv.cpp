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

#include "../utility.h"

// Look up the per-configuration entry for (operation, alg, datatype) on this
// sparse matrix descriptor's SpMV cache, creating it if necessary. Both
// hipsparseSpMV_bufferSize and hipsparseSpMV_preprocess share this lookup; the
// SpMV-only override for csr_adaptive + non_transpose (when no entry yet
// exists) is handled separately in hipsparseSpMV.
static hipsparseStatus_t hipsparseSpMV_get_or_add_entry(hipsparseHandle_t      handle,
                                                        hipsparseSpMVDescr_st* hip_spmv_descr,
                                                        rocsparse_operation    operation,
                                                        rocsparse_spmv_alg     spmv_alg,
                                                        rocsparse_datatype     datatype,
                                                        hipsparseSpMVDescr_st::Entry** out_entry)
{
    hipsparseSpMVDescr_st::Entry* entry
        = hip_spmv_descr->find_entry(operation, spmv_alg, datatype, datatype);

    if(entry == nullptr)
    {
        RETURN_IF_HIPSPARSE_ERROR(
            hip_spmv_descr->add_entry(handle, operation, spmv_alg, datatype, datatype, &entry));
    }

    *out_entry = entry;
    return HIPSPARSE_STATUS_SUCCESS;
}

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

    if((spmv_alg == rocsparse_spmv_alg_csr_adaptive) && (operation != rocsparse_operation_none))
    {
        spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
    }

    hipsparseSpMVDescr_st*        hip_spmv_descr = matA->get_hip_spmv_descr();
    hipsparseSpMVDescr_st::Entry* entry          = nullptr;

    RETURN_IF_HIPSPARSE_ERROR(hipsparseSpMV_get_or_add_entry(
        handle, hip_spmv_descr, operation, spmv_alg, datatype, &entry));

    if(entry->is_buffer_size_called)
    {
        // We have already queried rocsparse for this configuration and
        // cached the result. Return the cached size verbatim rather than
        // re-querying
        *pBufferSizeInBytes = entry->buffer_size_stage_analysis;
        return HIPSPARSE_STATUS_SUCCESS;
    }

    // First-ever bufferSize query for this configuration. Cache the result
    // on the entry so subsequent hipsparseSpMV / hipsparseSpMV_preprocess
    // calls know the size of the user-provided externalBuffer that was
    // sized for the analysis stage.
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv_buffer_size((rocsparse_handle)handle,
                                                            entry->spmv_descr,
                                                            to_rocsparse_const_spmat_descr(matA),
                                                            (rocsparse_const_dnvec_descr)vecX,
                                                            (rocsparse_dnvec_descr)vecY,
                                                            rocsparse_v2_spmv_stage_analysis,
                                                            pBufferSizeInBytes,
                                                            nullptr));

    entry->buffer_size_stage_analysis = pBufferSizeInBytes[0];
    entry->is_buffer_size_called      = true;

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

    if((spmv_alg == rocsparse_spmv_alg_csr_adaptive) && (operation != rocsparse_operation_none))
    {
        spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
    }

    hipsparseSpMVDescr_st*        hip_spmv_descr = matA->get_hip_spmv_descr();
    hipsparseSpMVDescr_st::Entry* entry          = nullptr;

    RETURN_IF_HIPSPARSE_ERROR(hipsparseSpMV_get_or_add_entry(
        handle, hip_spmv_descr, operation, spmv_alg, datatype, &entry));

    if(entry->is_stage_analysis_called || entry->is_implicit_stage_analysis_called)
    {
        // Analysis has already been performed for this configuration
        // (either by a previous hipsparseSpMV_preprocess call or by the
        // implicit-analysis branch in a previous hipsparseSpMV call). No
        // further analysis work is required.
        entry->is_stage_analysis_called = true;
        return HIPSPARSE_STATUS_SUCCESS;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv((rocsparse_handle)handle,
                                                entry->spmv_descr,
                                                alpha,
                                                to_rocsparse_const_spmat_descr(matA),
                                                (rocsparse_const_dnvec_descr)vecX,
                                                beta,
                                                (rocsparse_dnvec_descr)vecY,
                                                rocsparse_v2_spmv_stage_analysis,
                                                entry->buffer_size_stage_analysis,
                                                externalBuffer,
                                                nullptr));
    entry->is_stage_analysis_called = true;
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

    if((spmv_alg == rocsparse_spmv_alg_csr_adaptive) && (operation != rocsparse_operation_none))
    {
        spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
    }

    hipsparseSpMVDescr_st*        hip_spmv_descr = matA->get_hip_spmv_descr();
    hipsparseSpMVDescr_st::Entry* entry
        = hip_spmv_descr->find_entry(operation, spmv_alg, datatype, datatype);

    if(entry == nullptr)
    {
        // SpMV-only override: when the user asks for csr_adaptive on the
        // non-transpose path with no prior bufferSize / preprocess call
        // for this configuration, fall back to csr_rowsplit (csr_adaptive
        // requires the caller to have run analysis first via the explicit
        // bufferSize / preprocess flow).
        if((spmv_alg == rocsparse_spmv_alg_csr_adaptive) && (operation == rocsparse_operation_none))
        {
            spmv_alg = rocsparse_spmv_alg_csr_rowsplit;
        }

        // Look up under the (possibly-adjusted) key, reusing any matching
        // entry that already exists or creating a fresh one if not.
        RETURN_IF_HIPSPARSE_ERROR(hipsparseSpMV_get_or_add_entry(
            handle, hip_spmv_descr, operation, spmv_alg, datatype, &entry));
    }

    if(entry->is_stage_analysis_called == false)
    {
        // No analysis has been performed, we have to call the analysis since
        // this is a requirement for v2_spmv.
        if(entry->is_implicit_stage_analysis_called == false)
        {
            if(entry->is_buffer_size_called == false)
            {
                size_t buffer_size_in_bytes;
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse_v2_spmv_buffer_size((rocsparse_handle)handle,
                                                  entry->spmv_descr,
                                                  to_rocsparse_const_spmat_descr(matA),
                                                  (rocsparse_const_dnvec_descr)vecX,
                                                  (rocsparse_dnvec_descr)vecY,
                                                  rocsparse_v2_spmv_stage_analysis,
                                                  &buffer_size_in_bytes,
                                                  nullptr));

                hipStream_t stream{};
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_stream((rocsparse_handle)handle, &stream));

                void* temp_buffer = nullptr;
                if(buffer_size_in_bytes > 0)
                {
                    RETURN_IF_HIP_ERROR(hipMallocAsync(&temp_buffer, buffer_size_in_bytes, stream));
                }

                RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv((rocsparse_handle)handle,
                                                            entry->spmv_descr,
                                                            alpha,
                                                            to_rocsparse_const_spmat_descr(matA),
                                                            (rocsparse_const_dnvec_descr)vecX,
                                                            beta,
                                                            (const rocsparse_dnvec_descr)vecY,
                                                            rocsparse_v2_spmv_stage_analysis,
                                                            buffer_size_in_bytes,
                                                            temp_buffer,
                                                            nullptr));

                if(temp_buffer != nullptr)
                {
                    RETURN_IF_HIP_ERROR(hipFreeAsync(temp_buffer, stream));
                }
            }
            else
            {
                // We can use the externalBuffer since the user is allocating
                // a buffer for the analysis phase only, but the user does not
                // know it.
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv((rocsparse_handle)handle,
                                                            entry->spmv_descr,
                                                            alpha,
                                                            to_rocsparse_const_spmat_descr(matA),
                                                            (rocsparse_const_dnvec_descr)vecX,
                                                            beta,
                                                            (const rocsparse_dnvec_descr)vecY,
                                                            rocsparse_v2_spmv_stage_analysis,
                                                            entry->buffer_size_stage_analysis,
                                                            externalBuffer,
                                                            nullptr));
            }

            entry->is_implicit_stage_analysis_called = true;
        }
    }

    if(entry->is_stage_compute_subsequent == false)
    {
        // Query the compute-stage buffer size and allocate the per-entry
        // compute buffer. Each entry owns its own compute buffer so that
        // switching between configurations on the same sparse matrix
        // descriptor does not invalidate any previously allocated compute
        // buffer or re-trigger this query.
        size_t buffer_size_in_bytes;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_v2_spmv_buffer_size((rocsparse_handle)handle,
                                          entry->spmv_descr,
                                          to_rocsparse_const_spmat_descr(matA),
                                          (rocsparse_const_dnvec_descr)vecX,
                                          (rocsparse_dnvec_descr)vecY,
                                          rocsparse_v2_spmv_stage_compute,
                                          &buffer_size_in_bytes,
                                          nullptr));

        entry->buffer_size_stage_compute = buffer_size_in_bytes;

        hipStream_t stream{};
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_stream((rocsparse_handle)handle, &stream));

        RETURN_IF_HIP_ERROR(
            hipMallocAsync(&entry->compute_buffer, entry->buffer_size_stage_compute, stream));

        entry->is_stage_compute_subsequent = true;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_v2_spmv((rocsparse_handle)handle,
                                                entry->spmv_descr,
                                                alpha,
                                                to_rocsparse_const_spmat_descr(matA),
                                                (rocsparse_const_dnvec_descr)vecX,
                                                beta,
                                                (rocsparse_dnvec_descr)vecY,
                                                rocsparse_v2_spmv_stage_compute,
                                                entry->buffer_size_stage_compute,
                                                entry->compute_buffer,
                                                nullptr));

    return HIPSPARSE_STATUS_SUCCESS;
}
