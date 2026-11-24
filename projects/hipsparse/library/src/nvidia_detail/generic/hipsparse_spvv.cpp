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

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>

#include "../utility.h"

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t          handle,
                                           hipsparseOperation_t       opX,
                                           hipsparseConstSpVecDescr_t vecX,
                                           hipsparseConstDnVecDescr_t vecY,
                                           void*                      result,
                                           hipDataType                computeType,
                                           size_t*                    pBufferSizeInBytes)
{
    cusparseOperation_t backend_opX{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opX, &backend_opX));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpVV_bufferSize((cusparseHandle_t)handle,
                                backend_opX,
                                (cusparseConstSpVecDescr_t)vecX,
                                (cusparseConstDnVecDescr_t)vecY,
                                result,
                                backend_data_type,
                                pBufferSizeInBytes));
}
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                           hipsparseOperation_t  opX,
                                           hipsparseSpVecDescr_t vecX,
                                           hipsparseDnVecDescr_t vecY,
                                           void*                 result,
                                           hipDataType           computeType,
                                           size_t*               pBufferSizeInBytes)
{
    cusparseOperation_t backend_opX{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opX, &backend_opX));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(
        cusparseSpVV_bufferSize((cusparseHandle_t)handle,
                                backend_opX,
                                (cusparseSpVecDescr_t)vecX,
                                (cusparseDnVecDescr_t)vecY,
                                result,
                                backend_data_type,
                                pBufferSizeInBytes));
}
#endif

#if(CUDART_VERSION >= 12000)
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t          handle,
                                hipsparseOperation_t       opX,
                                hipsparseConstSpVecDescr_t vecX,
                                hipsparseConstDnVecDescr_t vecY,
                                void*                      result,
                                hipDataType                computeType,
                                void*                      externalBuffer)
{
    cusparseOperation_t backend_opX{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opX, &backend_opX));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSpVV((cusparseHandle_t)handle,
                                                                backend_opX,
                                                                (cusparseConstSpVecDescr_t)vecX,
                                                                (cusparseConstDnVecDescr_t)vecY,
                                                                result,
                                                                backend_data_type,
                                                                externalBuffer));
}
#elif(CUDART_VERSION > 10010 || (CUDART_VERSION == 10010 && CUDART_10_1_UPDATE_VERSION == 1))
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                                hipsparseOperation_t  opX,
                                hipsparseSpVecDescr_t vecX,
                                hipsparseDnVecDescr_t vecY,
                                void*                 result,
                                hipDataType           computeType,
                                void*                 externalBuffer)
{
    cusparseOperation_t backend_opX{};
    RETURN_IF_HIPSPARSE_ERROR(hipsparse::hipOperationToCudaOperation(opX, &backend_opX));

    cudaDataType backend_data_type{};
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparse::hipDataTypeToCudaDataType(computeType, &backend_data_type));

    return hipsparse::hipCUSPARSEStatusToHIPStatus(cusparseSpVV((cusparseHandle_t)handle,
                                                                backend_opX,
                                                                (cusparseSpVecDescr_t)vecX,
                                                                (cusparseDnVecDescr_t)vecY,
                                                                result,
                                                                backend_data_type,
                                                                externalBuffer));
}
#endif
