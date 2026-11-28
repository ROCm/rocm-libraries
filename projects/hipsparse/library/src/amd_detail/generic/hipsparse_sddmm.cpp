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

hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t          handle,
                                 hipsparseOperation_t       opA,
                                 hipsparseOperation_t       opB,
                                 const void*                alpha,
                                 hipsparseConstDnMatDescr_t matA,
                                 hipsparseConstDnMatDescr_t matB,
                                 const void*                beta,
                                 hipsparseSpMatDescr_t      matC,
                                 hipDataType                computeType,
                                 hipsparseSDDMMAlg_t        alg,
                                 void*                      tempBuffer)
{
    rocsparse_operation_ backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToHCCOperation(opA, &backend_opA));

    rocsparse_operation_ backend_opB{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToHCCOperation(opB, &backend_opB));

    rocsparse_datatype_ backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipDataTypeToHCCDataType(computeType, &backend_data_type));

    rocsparse_sddmm_alg_ backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSDDMMAlgToHCCSDDMMAlg(alg, &backend_alg));

    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_sddmm((rocsparse_handle)handle,
                                                                 backend_opA,
                                                                 backend_opB,
                                                                 alpha,
                                                                 (rocsparse_const_dnmat_descr)matA,
                                                                 (rocsparse_const_dnmat_descr)matB,
                                                                 beta,
                                                                 to_rocsparse_spmat_descr(matC),
                                                                 backend_data_type,
                                                                 backend_alg,
                                                                 tempBuffer));
}

hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            const void*                alpha,
                                            hipsparseConstDnMatDescr_t matA,
                                            hipsparseConstDnMatDescr_t matB,
                                            const void*                beta,
                                            hipsparseSpMatDescr_t      matC,
                                            hipDataType                computeType,
                                            hipsparseSDDMMAlg_t        alg,
                                            size_t*                    pBufferSizeInBytes)
{
    rocsparse_operation_ backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToHCCOperation(opA, &backend_opA));

    rocsparse_operation_ backend_opB{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToHCCOperation(opB, &backend_opB));

    rocsparse_datatype_ backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipDataTypeToHCCDataType(computeType, &backend_data_type));

    rocsparse_sddmm_alg_ backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSDDMMAlgToHCCSDDMMAlg(alg, &backend_alg));

    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sddmm_buffer_size((rocsparse_handle)handle,
                                    backend_opA,
                                    backend_opB,
                                    alpha,
                                    (rocsparse_const_dnmat_descr)matA,
                                    (rocsparse_const_dnmat_descr)matB,
                                    beta,
                                    to_rocsparse_spmat_descr(matC),
                                    backend_data_type,
                                    backend_alg,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t          handle,
                                            hipsparseOperation_t       opA,
                                            hipsparseOperation_t       opB,
                                            const void*                alpha,
                                            hipsparseConstDnMatDescr_t matA,
                                            hipsparseConstDnMatDescr_t matB,
                                            const void*                beta,
                                            hipsparseSpMatDescr_t      matC,
                                            hipDataType                computeType,
                                            hipsparseSDDMMAlg_t        alg,
                                            void*                      tempBuffer)
{
    rocsparse_operation_ backend_opA{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToHCCOperation(opA, &backend_opA));

    rocsparse_operation_ backend_opB{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToHCCOperation(opB, &backend_opB));

    rocsparse_datatype_ backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipDataTypeToHCCDataType(computeType, &backend_data_type));

    rocsparse_sddmm_alg_ backend_alg{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipSDDMMAlgToHCCSDDMMAlg(alg, &backend_alg));

    return hipsparse::rocSPARSEStatusToHIPStatus(
        rocsparse_sddmm_preprocess((rocsparse_handle)handle,
                                   backend_opA,
                                   backend_opB,
                                   alpha,
                                   (rocsparse_const_dnmat_descr)matA,
                                   (rocsparse_const_dnmat_descr)matB,
                                   beta,
                                   to_rocsparse_spmat_descr(matC),
                                   backend_data_type,
                                   backend_alg,
                                   tempBuffer));
}
