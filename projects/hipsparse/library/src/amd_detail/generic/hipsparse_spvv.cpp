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

hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opX,
                                           hipsparseConstSpVecDescr_t vecX,
                                           hipsparseConstDnVecDescr_t vecY,
                                           void*                      result,
                                           hipDataType                computeType,
                                           size_t*                    pBufferSizeInBytes)
{
    rocsparse_operation_ backend_opX{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToHCCOperation(opX, &backend_opX));

    rocsparse_datatype_ backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipDataTypeToHCCDataType(computeType, &backend_data_type));

    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_spvv((rocsparse_handle)handle,
                                                                backend_opX,
                                                                (rocsparse_const_spvec_descr)vecX,
                                                                (rocsparse_const_dnvec_descr)vecY,
                                                                result,
                                                                backend_data_type,
                                                                pBufferSizeInBytes,
                                                                nullptr));
}

hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t          handle,
                                hipsparseOperation_t       opX,
                                hipsparseConstSpVecDescr_t vecX,
                                hipsparseConstDnVecDescr_t vecY,
                                void*                      result,
                                hipDataType                computeType,
                                void*                      externalBuffer)
{
    rocsparse_operation_ backend_opX{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToHCCOperation(opX, &backend_opX));

    rocsparse_datatype_ backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipDataTypeToHCCDataType(computeType, &backend_data_type));

    size_t bufferSize;

    // Check for buffer == nullptr as this is not done in rocsparse
    if(externalBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    return hipsparse::rocSPARSEStatusToHIPStatus(rocsparse_spvv((rocsparse_handle)handle,
                                                                backend_opX,
                                                                (rocsparse_const_spvec_descr)vecX,
                                                                (rocsparse_const_dnvec_descr)vecY,
                                                                result,
                                                                backend_data_type,
                                                                &bufferSize,
                                                                externalBuffer));
}
