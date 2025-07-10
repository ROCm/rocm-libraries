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

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#define RETURN_IF_CUSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                          \
    {                                                                             \
        cusparseStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;           \
        if(TMP_STATUS_FOR_CHECK != CUSPARSE_STATUS_SUCCESS)                       \
        {                                                                         \
            return hipsparse::hipCUSPARSEStatusToHIPStatus(TMP_STATUS_FOR_CHECK); \
        }                                                                         \
    }

namespace hipsparse
{
    inline hipsparseStatus_t hipCUSPARSEStatusToHIPStatus(cusparseStatus_t cuStatus)
    {
#if(CUDART_VERSION >= 11003)
        switch(cuStatus)
        {
        case CUSPARSE_STATUS_SUCCESS:
            return HIPSPARSE_STATUS_SUCCESS;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return HIPSPARSE_STATUS_NOT_INITIALIZED;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        case CUSPARSE_STATUS_INVALID_VALUE:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return HIPSPARSE_STATUS_ARCH_MISMATCH;
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return HIPSPARSE_STATUS_MAPPING_ERROR;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return HIPSPARSE_STATUS_EXECUTION_FAILED;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
        case CUSPARSE_STATUS_ZERO_PIVOT:
            return HIPSPARSE_STATUS_ZERO_PIVOT;
        case CUSPARSE_STATUS_NOT_SUPPORTED:
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
        default:
            throw "Non existent cusparseStatus_t";
        }
#elif(CUDART_VERSION >= 10010)
        switch(cuStatus)
        {
        case CUSPARSE_STATUS_SUCCESS:
            return HIPSPARSE_STATUS_SUCCESS;
        case CUSPARSE_STATUS_NOT_INITIALIZED:
            return HIPSPARSE_STATUS_NOT_INITIALIZED;
        case CUSPARSE_STATUS_ALLOC_FAILED:
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        case CUSPARSE_STATUS_INVALID_VALUE:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        case CUSPARSE_STATUS_ARCH_MISMATCH:
            return HIPSPARSE_STATUS_ARCH_MISMATCH;
        case CUSPARSE_STATUS_MAPPING_ERROR:
            return HIPSPARSE_STATUS_MAPPING_ERROR;
        case CUSPARSE_STATUS_EXECUTION_FAILED:
            return HIPSPARSE_STATUS_EXECUTION_FAILED;
        case CUSPARSE_STATUS_INTERNAL_ERROR:
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
        case CUSPARSE_STATUS_ZERO_PIVOT:
            return HIPSPARSE_STATUS_ZERO_PIVOT;
        case CUSPARSE_STATUS_NOT_SUPPORTED:
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        default:
            throw "Non existent cusparseStatus_t";
        }
#else
#error "CUDART_VERSION is not supported"
#endif
    }

    inline cusparseStatus_t hipSPARSEStatusToCUSPARSEStatus(hipsparseStatus_t hipStatus)
    {
#if(CUDART_VERSION >= 11003)
        switch(hipStatus)
        {
        case HIPSPARSE_STATUS_SUCCESS:
            return CUSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_STATUS_NOT_INITIALIZED:
            return CUSPARSE_STATUS_NOT_INITIALIZED;
        case HIPSPARSE_STATUS_ALLOC_FAILED:
            return CUSPARSE_STATUS_ALLOC_FAILED;
        case HIPSPARSE_STATUS_INVALID_VALUE:
            return CUSPARSE_STATUS_INVALID_VALUE;
        case HIPSPARSE_STATUS_ARCH_MISMATCH:
            return CUSPARSE_STATUS_ARCH_MISMATCH;
        case HIPSPARSE_STATUS_MAPPING_ERROR:
            return CUSPARSE_STATUS_MAPPING_ERROR;
        case HIPSPARSE_STATUS_EXECUTION_FAILED:
            return CUSPARSE_STATUS_EXECUTION_FAILED;
        case HIPSPARSE_STATUS_INTERNAL_ERROR:
            return CUSPARSE_STATUS_INTERNAL_ERROR;
        case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
        case HIPSPARSE_STATUS_ZERO_PIVOT:
            return CUSPARSE_STATUS_ZERO_PIVOT;
        case HIPSPARSE_STATUS_NOT_SUPPORTED:
            return CUSPARSE_STATUS_NOT_SUPPORTED;
        case HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return CUSPARSE_STATUS_INSUFFICIENT_RESOURCES;
        default:
            throw "Non existent hipsparseStatus_t";
        }
#elif(CUDART_VERSION >= 10010)
        switch(hipStatus)
        {
        case HIPSPARSE_STATUS_SUCCESS:
            return CUSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_STATUS_NOT_INITIALIZED:
            return CUSPARSE_STATUS_NOT_INITIALIZED;
        case HIPSPARSE_STATUS_ALLOC_FAILED:
            return CUSPARSE_STATUS_ALLOC_FAILED;
        case HIPSPARSE_STATUS_INVALID_VALUE:
            return CUSPARSE_STATUS_INVALID_VALUE;
        case HIPSPARSE_STATUS_ARCH_MISMATCH:
            return CUSPARSE_STATUS_ARCH_MISMATCH;
        case HIPSPARSE_STATUS_MAPPING_ERROR:
            return CUSPARSE_STATUS_MAPPING_ERROR;
        case HIPSPARSE_STATUS_EXECUTION_FAILED:
            return CUSPARSE_STATUS_EXECUTION_FAILED;
        case HIPSPARSE_STATUS_INTERNAL_ERROR:
            return CUSPARSE_STATUS_INTERNAL_ERROR;
        case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
        case HIPSPARSE_STATUS_ZERO_PIVOT:
            return CUSPARSE_STATUS_ZERO_PIVOT;
        case HIPSPARSE_STATUS_NOT_SUPPORTED:
            return CUSPARSE_STATUS_NOT_SUPPORTED;
        default:
            throw "Non existent hipsparseStatus_t";
        }
#else
#error "CUDART_VERSION is not supported"
#endif
    }

    inline cusparsePointerMode_t hipPointerModeToCudaPointerMode(hipsparsePointerMode_t mode)
    {
        switch(mode)
        {
        case HIPSPARSE_POINTER_MODE_HOST:
            return CUSPARSE_POINTER_MODE_HOST;
        case HIPSPARSE_POINTER_MODE_DEVICE:
            return CUSPARSE_POINTER_MODE_DEVICE;
        default:
            throw "Non existent hipsparsePointerMode_t";
        }
    }

    inline hipsparsePointerMode_t CudaPointerModeToHIPPointerMode(cusparsePointerMode_t mode)
    {
        switch(mode)
        {
        case CUSPARSE_POINTER_MODE_HOST:
            return HIPSPARSE_POINTER_MODE_HOST;
        case CUSPARSE_POINTER_MODE_DEVICE:
            return HIPSPARSE_POINTER_MODE_DEVICE;
        default:
            throw "Non existent cusparsePointerMode_t";
        }
    }

    inline cusparseAction_t hipActionToCudaAction(hipsparseAction_t action)
    {
        switch(action)
        {
        case HIPSPARSE_ACTION_SYMBOLIC:
            return CUSPARSE_ACTION_SYMBOLIC;
        case HIPSPARSE_ACTION_NUMERIC:
            return CUSPARSE_ACTION_NUMERIC;
        default:
            throw "Non existent hipsparseAction_t";
        }
    }

    inline hipsparseAction_t CudaActionToHIPAction(cusparseAction_t action)
    {
        switch(action)
        {
        case CUSPARSE_ACTION_SYMBOLIC:
            return HIPSPARSE_ACTION_SYMBOLIC;
        case CUSPARSE_ACTION_NUMERIC:
            return HIPSPARSE_ACTION_NUMERIC;
        default:
            throw "Non existent cusparseAction_t";
        }
    }

    inline cusparseMatrixType_t hipMatrixTypeToCudaMatrixType(hipsparseMatrixType_t type)
    {
        switch(type)
        {
        case HIPSPARSE_MATRIX_TYPE_GENERAL:
            return CUSPARSE_MATRIX_TYPE_GENERAL;
        case HIPSPARSE_MATRIX_TYPE_SYMMETRIC:
            return CUSPARSE_MATRIX_TYPE_SYMMETRIC;
        case HIPSPARSE_MATRIX_TYPE_HERMITIAN:
            return CUSPARSE_MATRIX_TYPE_HERMITIAN;
        case HIPSPARSE_MATRIX_TYPE_TRIANGULAR:
            return CUSPARSE_MATRIX_TYPE_TRIANGULAR;
        default:
            throw "Non existent hipsparseMatrixType_t";
        }
    }

    inline hipsparseMatrixType_t CudaMatrixTypeToHIPMatrixType(cusparseMatrixType_t type)
    {
        switch(type)
        {
        case CUSPARSE_MATRIX_TYPE_GENERAL:
            return HIPSPARSE_MATRIX_TYPE_GENERAL;
        case CUSPARSE_MATRIX_TYPE_SYMMETRIC:
            return HIPSPARSE_MATRIX_TYPE_SYMMETRIC;
        case CUSPARSE_MATRIX_TYPE_HERMITIAN:
            return HIPSPARSE_MATRIX_TYPE_HERMITIAN;
        case CUSPARSE_MATRIX_TYPE_TRIANGULAR:
            return HIPSPARSE_MATRIX_TYPE_TRIANGULAR;
        default:
            throw "Non existent cusparseMatrixType_t";
        }
    }

    inline cusparseFillMode_t hipFillToCudaFill(hipsparseFillMode_t fill)
    {
        switch(fill)
        {
        case HIPSPARSE_FILL_MODE_LOWER:
            return CUSPARSE_FILL_MODE_LOWER;
        case HIPSPARSE_FILL_MODE_UPPER:
            return CUSPARSE_FILL_MODE_UPPER;
        default:
            throw "Non existent hipsparseFillMode_t";
        }
    }

    inline hipsparseFillMode_t CudaFillToHIPFill(cusparseFillMode_t fill)
    {
        switch(fill)
        {
        case CUSPARSE_FILL_MODE_LOWER:
            return HIPSPARSE_FILL_MODE_LOWER;
        case CUSPARSE_FILL_MODE_UPPER:
            return HIPSPARSE_FILL_MODE_UPPER;
        default:
            throw "Non existent cusparseFillMode_t";
        }
    }

    inline cusparseDiagType_t hipDiagonalToCudaDiagonal(hipsparseDiagType_t diagonal)
    {
        switch(diagonal)
        {
        case HIPSPARSE_DIAG_TYPE_NON_UNIT:
            return CUSPARSE_DIAG_TYPE_NON_UNIT;
        case HIPSPARSE_DIAG_TYPE_UNIT:
            return CUSPARSE_DIAG_TYPE_UNIT;
        default:
            throw "Non existent hipsparseDiagType_t";
        }
    }

    inline hipsparseDiagType_t CudaDiagonalToHIPDiagonal(cusparseDiagType_t diagonal)
    {
        switch(diagonal)
        {
        case CUSPARSE_DIAG_TYPE_NON_UNIT:
            return HIPSPARSE_DIAG_TYPE_NON_UNIT;
        case CUSPARSE_DIAG_TYPE_UNIT:
            return HIPSPARSE_DIAG_TYPE_UNIT;
        default:
            throw "Non existent cusparseDiagType_t";
        }
    }

    inline cusparseIndexBase_t hipIndexBaseToCudaIndexBase(hipsparseIndexBase_t base)
    {
        switch(base)
        {
        case HIPSPARSE_INDEX_BASE_ZERO:
            return CUSPARSE_INDEX_BASE_ZERO;
        case HIPSPARSE_INDEX_BASE_ONE:
            return CUSPARSE_INDEX_BASE_ONE;
        default:
            throw "Non existent hipsparseIndexBase_t";
        }
    }

    inline hipsparseIndexBase_t CudaIndexBaseToHIPIndexBase(cusparseIndexBase_t base)
    {
        switch(base)
        {
        case CUSPARSE_INDEX_BASE_ZERO:
            return HIPSPARSE_INDEX_BASE_ZERO;
        case CUSPARSE_INDEX_BASE_ONE:
            return HIPSPARSE_INDEX_BASE_ONE;
        default:
            throw "Non existent cusparseIndexBase_t";
        }
    }

    inline cusparseOperation_t hipOperationToCudaOperation(hipsparseOperation_t op)
    {
        switch(op)
        {
        case HIPSPARSE_OPERATION_NON_TRANSPOSE:
            return CUSPARSE_OPERATION_NON_TRANSPOSE;
        case HIPSPARSE_OPERATION_TRANSPOSE:
            return CUSPARSE_OPERATION_TRANSPOSE;
        case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        default:
            throw "Non existent hipsparseOperation_t";
        }
    }

    inline hipsparseOperation_t CudaOperationToHIPOperation(cusparseOperation_t op)
    {
        switch(op)
        {
        case CUSPARSE_OPERATION_NON_TRANSPOSE:
            return HIPSPARSE_OPERATION_NON_TRANSPOSE;
        case CUSPARSE_OPERATION_TRANSPOSE:
            return HIPSPARSE_OPERATION_TRANSPOSE;
        case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        default:
            throw "Non existent cusparseOperation_t";
        }
    }

    inline cusparseDirection_t hipDirectionToCudaDirection(hipsparseDirection_t op)
    {
        switch(op)
        {
        case HIPSPARSE_DIRECTION_ROW:
            return CUSPARSE_DIRECTION_ROW;
        case HIPSPARSE_DIRECTION_COLUMN:
            return CUSPARSE_DIRECTION_COLUMN;
        default:
            throw "Non existent hipsparseDirection_t";
        }
    }

    inline hipsparseDirection_t CudaDirectionToHIPDirection(cusparseDirection_t op)
    {
        switch(op)
        {
        case CUSPARSE_DIRECTION_ROW:
            return HIPSPARSE_DIRECTION_ROW;
        case CUSPARSE_DIRECTION_COLUMN:
            return HIPSPARSE_DIRECTION_COLUMN;
        default:
            throw "Non existent cusparseDirection_t";
        }
    }

#if CUDART_VERSION < 11000
    inline cusparseHybPartition_t hipHybPartitionToCudaHybPartition(hipsparseHybPartition_t part)
    {
        switch(part)
        {
        case HIPSPARSE_HYB_PARTITION_AUTO:
            return CUSPARSE_HYB_PARTITION_AUTO;
        case HIPSPARSE_HYB_PARTITION_USER:
            return CUSPARSE_HYB_PARTITION_USER;
        case HIPSPARSE_HYB_PARTITION_MAX:
            return CUSPARSE_HYB_PARTITION_MAX;
        default:
            throw "Non existent hipsparseHybPartition_t";
        }
    }
#endif

    inline cusparseSolvePolicy_t hipPolicyToCudaPolicy(hipsparseSolvePolicy_t policy)
    {
        switch(policy)
        {
        case HIPSPARSE_SOLVE_POLICY_NO_LEVEL:
            return CUSPARSE_SOLVE_POLICY_NO_LEVEL;
        case HIPSPARSE_SOLVE_POLICY_USE_LEVEL:
            return CUSPARSE_SOLVE_POLICY_USE_LEVEL;
        default:
            throw "Non existent hipsparseSolvePolicy_t";
        }
    }

#if CUDART_VERSION < 11050
    inline cusparseSideMode_t hipSideToCudaSide(hipsparseSideMode_t side)
    {
        switch(side)
        {
        case HIPSPARSE_SIDE_LEFT:
            return CUSPARSE_SIDE_LEFT;
        case HIPSPARSE_SIDE_RIGHT:
            return CUSPARSE_SIDE_RIGHT;
        default:
            throw "Non existent hipsparseSideMode_t";
        }
    }

    inline hipsparseSideMode_t CudaSideToHIPSide(cusparseSideMode_t side)
    {
        switch(side)
        {
        case CUSPARSE_SIDE_LEFT:
            return HIPSPARSE_SIDE_LEFT;
        case CUSPARSE_SIDE_RIGHT:
            return HIPSPARSE_SIDE_RIGHT;
        default:
            throw "Non existent cusparseSideMode_t";
        }
    }
#endif

#if CUDART_VERSION > 10000
    inline cudaDataType hipDataTypeToCudaDataType(hipDataType datatype)
    {
        switch(datatype)
        {
        case HIP_R_8I:
            return CUDA_R_8I;
        case HIP_R_32I:
            return CUDA_R_32I;
        case HIP_R_16F:
            return CUDA_R_16F;
        case HIP_R_16BF:
            return CUDA_R_16BF;
        case HIP_R_32F:
            return CUDA_R_32F;
        case HIP_R_64F:
            return CUDA_R_64F;
        case HIP_C_32F:
            return CUDA_C_32F;
        case HIP_C_64F:
            return CUDA_C_64F;
        default:
            throw "Non existent hipDataType";
        }
    }

    inline hipDataType CudaDataTypeToHIPDataType(cudaDataType datatype)
    {
        switch(datatype)
        {
        case CUDA_R_8I:
            return HIP_R_8I;
        case CUDA_R_32I:
            return HIP_R_32I;
        case CUDA_R_16F:
            return HIP_R_16F;
        case CUDA_R_16BF:
            return HIP_R_16BF;
        case CUDA_R_32F:
            return HIP_R_32F;
        case CUDA_R_64F:
            return HIP_R_64F;
        case CUDA_C_32F:
            return HIP_C_32F;
        case CUDA_C_64F:
            return HIP_C_64F;
        default:
            throw "Non existent cudaDataType";
        }
    }
#endif

#if(CUDART_VERSION >= 12000)
    inline cusparseCsr2CscAlg_t hipCsr2CscAlgToCudaCsr2CscAlg(hipsparseCsr2CscAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_CSR2CSC_ALG1:
            return CUSPARSE_CSR2CSC_ALG1;
        case HIPSPARSE_CSR2CSC_ALG_DEFAULT:
            return CUSPARSE_CSR2CSC_ALG_DEFAULT;
        default:
            throw "Non existant hipsparseCsr2CscAlg_t";
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
    inline cusparseCsr2CscAlg_t hipCsr2CscAlgToCudaCsr2CscAlg(hipsparseCsr2CscAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_CSR2CSC_ALG1:
            return CUSPARSE_CSR2CSC_ALG1;
        case HIPSPARSE_CSR2CSC_ALG2:
            return CUSPARSE_CSR2CSC_ALG2;
        default:
            throw "Non existant hipsparseCsr2CscAlg_t";
        }
    }
#endif

    /* Generic API */
#if(CUDART_VERSION >= 12000)
    inline cusparseFormat_t hipFormatToCudaFormat(hipsparseFormat_t format)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            return CUSPARSE_FORMAT_CSR;
        case HIPSPARSE_FORMAT_CSC:
            return CUSPARSE_FORMAT_CSC;
        case HIPSPARSE_FORMAT_COO:
            return CUSPARSE_FORMAT_COO;
        case HIPSPARSE_FORMAT_BLOCKED_ELL:
            return CUSPARSE_FORMAT_BLOCKED_ELL;
        default:
            throw "Non existent hipsparseFormat_t";
        }
    }
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
    inline cusparseFormat_t hipFormatToCudaFormat(hipsparseFormat_t format)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            return CUSPARSE_FORMAT_CSR;
        case HIPSPARSE_FORMAT_CSC:
            return CUSPARSE_FORMAT_CSC;
        case HIPSPARSE_FORMAT_COO:
            return CUSPARSE_FORMAT_COO;
        case HIPSPARSE_FORMAT_COO_AOS:
            return CUSPARSE_FORMAT_COO_AOS;
        case HIPSPARSE_FORMAT_BLOCKED_ELL:
            return CUSPARSE_FORMAT_BLOCKED_ELL;
        default:
            throw "Non existent hipsparseFormat_t";
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
    inline cusparseFormat_t hipFormatToCudaFormat(hipsparseFormat_t format)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            return CUSPARSE_FORMAT_CSR;
        case HIPSPARSE_FORMAT_COO:
            return CUSPARSE_FORMAT_COO;
        case HIPSPARSE_FORMAT_COO_AOS:
            return CUSPARSE_FORMAT_COO_AOS;
        default:
            throw "Non existent hipsparseFormat_t";
        }
    }
#endif

#if(CUDART_VERSION >= 12000)
    inline hipsparseFormat_t CudaFormatToHIPFormat(cusparseFormat_t format)
    {
        switch(format)
        {
        case CUSPARSE_FORMAT_CSR:
            return HIPSPARSE_FORMAT_CSR;
        case CUSPARSE_FORMAT_CSC:
            return HIPSPARSE_FORMAT_CSC;
        case CUSPARSE_FORMAT_COO:
            return HIPSPARSE_FORMAT_COO;
        case CUSPARSE_FORMAT_BLOCKED_ELL:
            return HIPSPARSE_FORMAT_BLOCKED_ELL;
        default:
            throw "Non existent cusparseFormat_t";
        }
    }
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
    inline hipsparseFormat_t CudaFormatToHIPFormat(cusparseFormat_t format)
    {
        switch(format)
        {
        case CUSPARSE_FORMAT_CSR:
            return HIPSPARSE_FORMAT_CSR;
        case CUSPARSE_FORMAT_CSC:
            return HIPSPARSE_FORMAT_CSC;
        case CUSPARSE_FORMAT_COO:
            return HIPSPARSE_FORMAT_COO;
        case CUSPARSE_FORMAT_COO_AOS:
            return HIPSPARSE_FORMAT_COO_AOS;
        case CUSPARSE_FORMAT_BLOCKED_ELL:
            return HIPSPARSE_FORMAT_BLOCKED_ELL;
        default:
            throw "Non existent cusparseFormat_t";
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
    inline hipsparseFormat_t CudaFormatToHIPFormat(cusparseFormat_t format)
    {
        switch(format)
        {
        case CUSPARSE_FORMAT_CSR:
            return HIPSPARSE_FORMAT_CSR;
        case CUSPARSE_FORMAT_COO:
            return HIPSPARSE_FORMAT_COO;
        case CUSPARSE_FORMAT_COO_AOS:
            return HIPSPARSE_FORMAT_COO_AOS;
        default:
            throw "Non existent cusparseFormat_t";
        }
    }
#endif

#if(CUDART_VERSION >= 11000)
    inline cusparseOrder_t hipOrderToCudaOrder(hipsparseOrder_t op)
    {
        switch(op)
        {
        case HIPSPARSE_ORDER_ROW:
            return CUSPARSE_ORDER_ROW;
        case HIPSPARSE_ORDER_COL:
            return CUSPARSE_ORDER_COL;
        default:
            throw "Non existent hipsparseOrder_t";
        }
    }

    inline hipsparseOrder_t CudaOrderToHIPOrder(cusparseOrder_t op)
    {
        switch(op)
        {
        case CUSPARSE_ORDER_ROW:
            return HIPSPARSE_ORDER_ROW;
        case CUSPARSE_ORDER_COL:
            return HIPSPARSE_ORDER_COL;
        default:
            throw "Non existent cusparseOrder_t";
        }
    }
#elif(CUDART_VERSION >= 10010)
    inline cusparseOrder_t hipOrderToCudaOrder(hipsparseOrder_t op)
    {
        switch(op)
        {
        case HIPSPARSE_ORDER_COL:
            return CUSPARSE_ORDER_COL;
        default:
            throw "Non existent hipsparseOrder_t";
        }
    }

    inline hipsparseOrder_t CudaOrderToHIPOrder(cusparseOrder_t op)
    {
        switch(op)
        {
        case CUSPARSE_ORDER_COL:
            return HIPSPARSE_ORDER_COL;
        default:
            throw "Non existent cusparseOrder_t";
        }
    }
#endif

#if(CUDART_VERSION >= 10010)
    inline cusparseIndexType_t hipIndexTypeToCudaIndexType(hipsparseIndexType_t type)
    {
        switch(type)
        {
        case HIPSPARSE_INDEX_16U:
            return CUSPARSE_INDEX_16U;
        case HIPSPARSE_INDEX_32I:
            return CUSPARSE_INDEX_32I;
        case HIPSPARSE_INDEX_64I:
            return CUSPARSE_INDEX_64I;
        default:
            throw "Non existant hipsparseIndexType_t";
        }
    }

    inline hipsparseIndexType_t CudaIndexTypeToHIPIndexType(cusparseIndexType_t type)
    {
        switch(type)
        {
        case CUSPARSE_INDEX_16U:
            return HIPSPARSE_INDEX_16U;
        case CUSPARSE_INDEX_32I:
            return HIPSPARSE_INDEX_32I;
        case CUSPARSE_INDEX_64I:
            return HIPSPARSE_INDEX_64I;
        default:
            throw "Non existant cusparseIndexType_t";
        }
    }
#endif

#if(CUDART_VERSION >= 12000)
    inline cusparseSpMVAlg_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPMV_ALG_DEFAULT:
            return CUSPARSE_SPMV_ALG_DEFAULT;
        case HIPSPARSE_SPMV_COO_ALG1:
            return CUSPARSE_SPMV_COO_ALG1;
        case HIPSPARSE_SPMV_COO_ALG2:
            return CUSPARSE_SPMV_COO_ALG2;
        case HIPSPARSE_SPMV_CSR_ALG1:
            return CUSPARSE_SPMV_CSR_ALG1;
        case HIPSPARSE_SPMV_CSR_ALG2:
            return CUSPARSE_SPMV_CSR_ALG2;
        default:
            throw "Non existant hipsparseSpMVAlg_t";
        }
    }
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
    inline cusparseSpMVAlg_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg)
    {
        switch(alg)
        {
        // case HIPSPARSE_MV_ALG_DEFAULT:
        case HIPSPARSE_SPMV_ALG_DEFAULT:
            return CUSPARSE_SPMV_ALG_DEFAULT;
        // case HIPSPARSE_COOMV_ALG:
        case HIPSPARSE_SPMV_COO_ALG1:
            return CUSPARSE_SPMV_COO_ALG1;
        case HIPSPARSE_SPMV_COO_ALG2:
            return CUSPARSE_SPMV_COO_ALG2;
        // case HIPSPARSE_CSRMV_ALG1:
        case HIPSPARSE_SPMV_CSR_ALG1:
            return CUSPARSE_SPMV_CSR_ALG1;
        // case HIPSPARSE_CSRMV_ALG2:
        case HIPSPARSE_SPMV_CSR_ALG2:
            return CUSPARSE_SPMV_CSR_ALG2;
        default:
            throw "Non existant hipsparseSpMVAlg_t";
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
    inline cusparseSpMVAlg_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_MV_ALG_DEFAULT:
            return CUSPARSE_MV_ALG_DEFAULT;
        case HIPSPARSE_COOMV_ALG:
            return CUSPARSE_COOMV_ALG;
        case HIPSPARSE_CSRMV_ALG1:
            return CUSPARSE_CSRMV_ALG1;
        case HIPSPARSE_CSRMV_ALG2:
            return CUSPARSE_CSRMV_ALG2;
        default:
            throw "Non existant hipsparseSpMVAlg_t";
        }
    }
#endif

#if(CUDART_VERSION >= 12000)
    inline cusparseSpMMAlg_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            return CUSPARSE_SPMM_ALG_DEFAULT;
        case HIPSPARSE_SPMM_COO_ALG1:
            return CUSPARSE_SPMM_COO_ALG1;
        case HIPSPARSE_SPMM_COO_ALG2:
            return CUSPARSE_SPMM_COO_ALG2;
        case HIPSPARSE_SPMM_COO_ALG3:
            return CUSPARSE_SPMM_COO_ALG3;
        case HIPSPARSE_SPMM_COO_ALG4:
            return CUSPARSE_SPMM_COO_ALG4;
        case HIPSPARSE_SPMM_CSR_ALG1:
            return CUSPARSE_SPMM_CSR_ALG1;
        case HIPSPARSE_SPMM_CSR_ALG2:
            return CUSPARSE_SPMM_CSR_ALG2;
        case HIPSPARSE_SPMM_CSR_ALG3:
            return CUSPARSE_SPMM_CSR_ALG3;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            return CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
        default:
            throw "Non existant hipsparseSpMMAlg_t";
        }
    }
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
    inline cusparseSpMMAlg_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg)
    {
        switch(alg)
        {
        // case HIPSPARSE_MM_ALG_DEFAULT:
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            return CUSPARSE_SPMM_ALG_DEFAULT;
        // case HIPSPARSE_COOMM_ALG1:
        case HIPSPARSE_SPMM_COO_ALG1:
            return CUSPARSE_SPMM_COO_ALG1;
        // case HIPSPARSE_COOMM_ALG2:
        case HIPSPARSE_SPMM_COO_ALG2:
            return CUSPARSE_SPMM_COO_ALG2;
        // case HIPSPARSE_COOMM_ALG3:
        case HIPSPARSE_SPMM_COO_ALG3:
            return CUSPARSE_SPMM_COO_ALG3;
        case HIPSPARSE_SPMM_COO_ALG4:
            return CUSPARSE_SPMM_COO_ALG4;
        // case HIPSPARSE_CSRMM_ALG1:
        case HIPSPARSE_SPMM_CSR_ALG1:
            return CUSPARSE_SPMM_CSR_ALG1;
        case HIPSPARSE_SPMM_CSR_ALG2:
            return CUSPARSE_SPMM_CSR_ALG2;
        case HIPSPARSE_SPMM_CSR_ALG3:
            return CUSPARSE_SPMM_CSR_ALG3;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            return CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
        default:
            throw "Non existant hipsparseSpMMAlg_t";
        }
    }
#elif(CUDART_VERSION >= 11003 && CUDART_VERSION < 11021)
    inline cusparseSpMMAlg_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg)
    {
        switch(alg)
        {
        // case HIPSPARSE_MM_ALG_DEFAULT:
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            return CUSPARSE_SPMM_ALG_DEFAULT;
        // case HIPSPARSE_COOMM_ALG1:
        case HIPSPARSE_SPMM_COO_ALG1:
            return CUSPARSE_SPMM_COO_ALG1;
        // case HIPSPARSE_COOMM_ALG2:
        case HIPSPARSE_SPMM_COO_ALG2:
            return CUSPARSE_SPMM_COO_ALG2;
        // case HIPSPARSE_COOMM_ALG3:
        case HIPSPARSE_SPMM_COO_ALG3:
            return CUSPARSE_SPMM_COO_ALG3;
        case HIPSPARSE_SPMM_COO_ALG4:
            return CUSPARSE_SPMM_COO_ALG4;
        // case HIPSPARSE_CSRMM_ALG1:
        case HIPSPARSE_SPMM_CSR_ALG1:
            return CUSPARSE_SPMM_CSR_ALG1;
        case HIPSPARSE_SPMM_CSR_ALG2:
            return CUSPARSE_SPMM_CSR_ALG2;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            return CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
        default:
            throw "Non existant hipsparseSpMMAlg_t";
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11003)
    inline cusparseSpMMAlg_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_MM_ALG_DEFAULT:
            return CUSPARSE_MM_ALG_DEFAULT;
        case HIPSPARSE_COOMM_ALG1:
            return CUSPARSE_COOMM_ALG1;
        case HIPSPARSE_COOMM_ALG2:
            return CUSPARSE_COOMM_ALG2;
        case HIPSPARSE_COOMM_ALG3:
            return CUSPARSE_COOMM_ALG3;
        case HIPSPARSE_CSRMM_ALG1:
            return CUSPARSE_CSRMM_ALG1;
        default:
            throw "Non existant hipsparseSpMMAlg_t";
        }
    }
#endif

#if(CUDART_VERSION >= 12000)
    inline cusparseSpGEMMAlg_t hipSpGEMMAlgToCudaSpGEMMAlg(hipsparseSpGEMMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEMM_DEFAULT:
            return CUSPARSE_SPGEMM_DEFAULT;
        case HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC:
            return CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;
        case HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC:
            return CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
        case HIPSPARSE_SPGEMM_ALG1:
            return CUSPARSE_SPGEMM_ALG1;
        case HIPSPARSE_SPGEMM_ALG2:
            return CUSPARSE_SPGEMM_ALG2;
        case HIPSPARSE_SPGEMM_ALG3:
            return CUSPARSE_SPGEMM_ALG3;
        default:
            throw "Non existant cusparseSpGEMMAlg_t";
        }
    }
#elif(CUDART_VERSION >= 11031)
    inline cusparseSpGEMMAlg_t hipSpGEMMAlgToCudaSpGEMMAlg(hipsparseSpGEMMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEMM_DEFAULT:
            return CUSPARSE_SPGEMM_DEFAULT;
        case HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC:
            return CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;
        case HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC:
            return CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
        default:
            throw "Non existant cusparseSpGEMMAlg_t";
        }
    }
#elif(CUDART_VERSION >= 11000)
    inline cusparseSpGEMMAlg_t hipSpGEMMAlgToCudaSpGEMMAlg(hipsparseSpGEMMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEMM_DEFAULT:
            return CUSPARSE_SPGEMM_DEFAULT;
        default:
            throw "Non existant cusparseSpGEMMAlg_t";
        }
    }
#endif

#if(CUDART_VERSION >= 11020)
    inline cusparseSparseToDenseAlg_t hipSpToDnAlgToCudaSpToDnAlg(hipsparseSparseToDenseAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPARSETODENSE_ALG_DEFAULT:
            return CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
        default:
            throw "Non existent hipsparseSparseToDenseAlg_t";
        }
    }

    inline hipsparseSparseToDenseAlg_t CudaSpToDnAlgToHipSpToDnAlg(cusparseSparseToDenseAlg_t alg)
    {
        switch(alg)
        {
        case CUSPARSE_SPARSETODENSE_ALG_DEFAULT:
            return HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
        default:
            throw "Non existent cusparseSparseToDenseAlg_t";
        }
    }
#endif

#if(CUDART_VERSION >= 11020)
    inline cusparseDenseToSparseAlg_t hipDnToSpAlgToCudaDnToSpAlg(hipsparseDenseToSparseAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT:
            return CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
        default:
            throw "Non existent hipsparseDenseToSparseAlg_t";
        }
    }

    inline hipsparseDenseToSparseAlg_t CudaDnToSpAlgToHipDnToSpAlg(cusparseDenseToSparseAlg_t alg)
    {
        switch(alg)
        {
        case CUSPARSE_DENSETOSPARSE_ALG_DEFAULT:
            return HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
        default:
            throw "Non existent cusparseDenseToSparseAlg_t";
        }
    }
#endif

#if(CUDART_VERSION >= 11022)
    inline cusparseSDDMMAlg_t hipSDDMMAlgToCudaSDDMMAlg(hipsparseSDDMMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SDDMM_ALG_DEFAULT:
            return CUSPARSE_SDDMM_ALG_DEFAULT;
        default:
            throw "Non existant cusparseSDDMMAlg_t";
        }
    }
#endif

#if(CUDART_VERSION >= 11030)
    inline cusparseSpSVAlg_t hipSpSVAlgToCudaSpSVAlg(hipsparseSpSVAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPSV_ALG_DEFAULT:
            return CUSPARSE_SPSV_ALG_DEFAULT;
        default:
            throw "Non existant cusparseSpSVAlg_t";
        }
    }
#endif

#if(CUDART_VERSION >= 11031)
    inline cusparseSpSMAlg_t hipSpSMAlgToCudaSpSMAlg(hipsparseSpSMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPSM_ALG_DEFAULT:
            return CUSPARSE_SPSM_ALG_DEFAULT;
        default:
            throw "Non existant cusparseSpSMAlg_t";
        }
    }
#endif
}
