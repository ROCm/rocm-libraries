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

#define RETURN_IF_HIPSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                                    \
        hipsparseStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != HIPSPARSE_STATUS_SUCCESS)             \
        {                                                                \
            return TMP_STATUS_FOR_CHECK;                                 \
        }                                                                \
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

    inline hipsparseStatus_t hipPointerModeToCudaPointerMode(hipsparsePointerMode_t mode,
                                                             cusparsePointerMode_t* out)
    {
        switch(mode)
        {
        case HIPSPARSE_POINTER_MODE_HOST:
            *out = CUSPARSE_POINTER_MODE_HOST;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_POINTER_MODE_DEVICE:
            *out = CUSPARSE_POINTER_MODE_DEVICE;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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

    inline hipsparseStatus_t hipActionToCudaAction(hipsparseAction_t action, cusparseAction_t* out)
    {
        switch(action)
        {
        case HIPSPARSE_ACTION_SYMBOLIC:
            *out = CUSPARSE_ACTION_SYMBOLIC;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_ACTION_NUMERIC:
            *out = CUSPARSE_ACTION_NUMERIC;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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

    inline hipsparseStatus_t hipMatrixTypeToCudaMatrixType(hipsparseMatrixType_t type,
                                                           cusparseMatrixType_t* out)
    {
        switch(type)
        {
        case HIPSPARSE_MATRIX_TYPE_GENERAL:
            *out = CUSPARSE_MATRIX_TYPE_GENERAL;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_MATRIX_TYPE_SYMMETRIC:
            *out = CUSPARSE_MATRIX_TYPE_SYMMETRIC;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_MATRIX_TYPE_HERMITIAN:
            *out = CUSPARSE_MATRIX_TYPE_HERMITIAN;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_MATRIX_TYPE_TRIANGULAR:
            *out = CUSPARSE_MATRIX_TYPE_TRIANGULAR;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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

    inline hipsparseStatus_t hipFillToCudaFill(hipsparseFillMode_t fill, cusparseFillMode_t* out)
    {
        switch(fill)
        {
        case HIPSPARSE_FILL_MODE_LOWER:
            *out = CUSPARSE_FILL_MODE_LOWER;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FILL_MODE_UPPER:
            *out = CUSPARSE_FILL_MODE_UPPER;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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

    inline hipsparseStatus_t hipDiagonalToCudaDiagonal(hipsparseDiagType_t diagonal,
                                                       cusparseDiagType_t* out)
    {
        switch(diagonal)
        {
        case HIPSPARSE_DIAG_TYPE_NON_UNIT:
            *out = CUSPARSE_DIAG_TYPE_NON_UNIT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_DIAG_TYPE_UNIT:
            *out = CUSPARSE_DIAG_TYPE_UNIT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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

    inline hipsparseStatus_t hipIndexBaseToCudaIndexBase(hipsparseIndexBase_t base,
                                                         cusparseIndexBase_t* out)
    {
        switch(base)
        {
        case HIPSPARSE_INDEX_BASE_ZERO:
            *out = CUSPARSE_INDEX_BASE_ZERO;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_INDEX_BASE_ONE:
            *out = CUSPARSE_INDEX_BASE_ONE;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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

    inline hipsparseStatus_t hipOperationToCudaOperation(hipsparseOperation_t op,
                                                         cusparseOperation_t* out)
    {
        switch(op)
        {
        case HIPSPARSE_OPERATION_NON_TRANSPOSE:
            *out = CUSPARSE_OPERATION_NON_TRANSPOSE;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_OPERATION_TRANSPOSE:
            *out = CUSPARSE_OPERATION_TRANSPOSE;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            *out = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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

    inline hipsparseStatus_t hipDirectionToCudaDirection(hipsparseDirection_t op,
                                                         cusparseDirection_t* out)
    {
        switch(op)
        {
        case HIPSPARSE_DIRECTION_ROW:
            *out = CUSPARSE_DIRECTION_ROW;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_DIRECTION_COLUMN:
            *out = CUSPARSE_DIRECTION_COLUMN;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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
    inline hipsparseStatus_t hipHybPartitionToCudaHybPartition(hipsparseHybPartition_t part,
                                                               cusparseHybPartition_t* out)
    {
        switch(part)
        {
        case HIPSPARSE_HYB_PARTITION_AUTO:
            *out = CUSPARSE_HYB_PARTITION_AUTO;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_HYB_PARTITION_USER:
            *out = CUSPARSE_HYB_PARTITION_USER;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_HYB_PARTITION_MAX:
            *out = CUSPARSE_HYB_PARTITION_MAX;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

    inline hipsparseStatus_t hipPolicyToCudaPolicy(hipsparseSolvePolicy_t policy,
                                                   cusparseSolvePolicy_t* out)
    {
        switch(policy)
        {
        case HIPSPARSE_SOLVE_POLICY_NO_LEVEL:
            *out = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SOLVE_POLICY_USE_LEVEL:
            *out = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

#if CUDART_VERSION < 11050
    inline hipsparseStatus_t hipSideToCudaSide(hipsparseSideMode_t side, cusparseSideMode_t* out)
    {
        switch(side)
        {
        case HIPSPARSE_SIDE_LEFT:
            *out = CUSPARSE_SIDE_LEFT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SIDE_RIGHT:
            *out = CUSPARSE_SIDE_RIGHT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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
    inline hipsparseStatus_t hipDataTypeToCudaDataType(hipDataType datatype, cudaDataType* out)
    {
        switch(datatype)
        {
        case HIP_R_8I:
            *out = CUDA_R_8I;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_32I:
            *out = CUDA_R_32I;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_16F:
            *out = CUDA_R_16F;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_16BF:
            *out = CUDA_R_16BF;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_32F:
            *out = CUDA_R_32F;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_64F:
            *out = CUDA_R_64F;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_C_32F:
            *out = CUDA_C_32F;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_C_64F:
            *out = CUDA_C_64F;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_C_16F:
        case HIP_C_8I:
        case HIP_C_8U:
        case HIP_R_8U:
        case HIP_C_32I:
        case HIP_R_32U:
        case HIP_C_32U:
        case HIP_C_16BF:
#if CUDART_VERSION >= 11000
        case HIP_R_4I:
        case HIP_C_4I:
        case HIP_R_4U:
        case HIP_C_4U:
        case HIP_R_16I:
        case HIP_C_16I:
        case HIP_R_16U:
        case HIP_C_16U:
        case HIP_R_64I:
        case HIP_C_64I:
        case HIP_R_64U:
        case HIP_C_64U:
#endif
#if CUDART_VERSION >= 11080
        case HIP_R_8F_E4M3:
        case HIP_R_8F_E5M2:
#endif
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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
    inline hipsparseStatus_t hipCsr2CscAlgToCudaCsr2CscAlg(hipsparseCsr2CscAlg_t alg,
                                                           cusparseCsr2CscAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_CSR2CSC_ALG1:
            *out = CUSPARSE_CSR2CSC_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_CSR2CSC_ALG_DEFAULT:
            *out = CUSPARSE_CSR2CSC_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
    inline hipsparseStatus_t hipCsr2CscAlgToCudaCsr2CscAlg(hipsparseCsr2CscAlg_t alg,
                                                           cusparseCsr2CscAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_CSR2CSC_ALG1:
            *out = CUSPARSE_CSR2CSC_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_CSR2CSC_ALG2:
            *out = CUSPARSE_CSR2CSC_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

    /* Generic API */
#if(CUDART_VERSION >= 12011)
    inline hipsparseStatus_t hipFormatToCudaFormat(hipsparseFormat_t format, cusparseFormat_t* out)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            *out = CUSPARSE_FORMAT_CSR;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_CSC:
            *out = CUSPARSE_FORMAT_CSC;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_COO:
            *out = CUSPARSE_FORMAT_COO;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_BLOCKED_ELL:
            *out = CUSPARSE_FORMAT_BLOCKED_ELL;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_SLICED_ELL:
            *out
                = CUSPARSE_FORMAT_SLICED_ELLPACK; // Called CUSPARSE_FORMAT_SLICED_ELL in the cusparse docs
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 12000 && CUDART_VERSION < 12011)
    inline hipsparseStatus_t hipFormatToCudaFormat(hipsparseFormat_t format, cusparseFormat_t* out)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            *out = CUSPARSE_FORMAT_CSR;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_CSC:
            *out = CUSPARSE_FORMAT_CSC;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_COO:
            *out = CUSPARSE_FORMAT_COO;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_BLOCKED_ELL:
            *out = CUSPARSE_FORMAT_BLOCKED_ELL;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
    inline hipsparseStatus_t hipFormatToCudaFormat(hipsparseFormat_t format, cusparseFormat_t* out)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            *out = CUSPARSE_FORMAT_CSR;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_CSC:
            *out = CUSPARSE_FORMAT_CSC;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_COO:
            *out = CUSPARSE_FORMAT_COO;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_COO_AOS:
            *out = CUSPARSE_FORMAT_COO_AOS;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_BLOCKED_ELL:
            *out = CUSPARSE_FORMAT_BLOCKED_ELL;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
    inline hipsparseStatus_t hipFormatToCudaFormat(hipsparseFormat_t format, cusparseFormat_t* out)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            *out = CUSPARSE_FORMAT_CSR;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_COO:
            *out = CUSPARSE_FORMAT_COO;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_COO_AOS:
            *out = CUSPARSE_FORMAT_COO_AOS;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

#if(CUDART_VERSION >= 12011)
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
        case CUSPARSE_FORMAT_SLICED_ELLPACK: // Called CUSPARSE_FORMAT_SLICED_ELL in the cusparse docs
            return HIPSPARSE_FORMAT_SLICED_ELL;
        default:
            throw "Non existent cusparseFormat_t";
        }
    }
#elif(CUDART_VERSION >= 12000 && CUDART_VERSION < 12011)
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
    inline hipsparseStatus_t hipOrderToCudaOrder(hipsparseOrder_t op, cusparseOrder_t* out)
    {
        switch(op)
        {
        case HIPSPARSE_ORDER_ROW:
            *out = CUSPARSE_ORDER_ROW;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_ORDER_COL:
            *out = CUSPARSE_ORDER_COL;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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
    inline hipsparseStatus_t hipOrderToCudaOrder(hipsparseOrder_t op, cusparseOrder_t* out)
    {
        switch(op)
        {
        case HIPSPARSE_ORDER_COL:
            *out = CUSPARSE_ORDER_COL;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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
    inline hipsparseStatus_t hipIndexTypeToCudaIndexType(hipsparseIndexType_t type,
                                                         cusparseIndexType_t* out)
    {
        switch(type)
        {
        case HIPSPARSE_INDEX_16U:
            *out = CUSPARSE_INDEX_16U;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_INDEX_32I:
            *out = CUSPARSE_INDEX_32I;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_INDEX_64I:
            *out = CUSPARSE_INDEX_64I;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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

#if(CUDART_VERSION >= 13001)
    inline hipsparseStatus_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg, cusparseSpMVAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPMV_ALG_DEFAULT:
            *out = CUSPARSE_SPMV_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_COO_ALG1:
            *out = CUSPARSE_SPMV_COO_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_COO_ALG2:
            *out = CUSPARSE_SPMV_COO_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_CSR_ALG1:
            *out = CUSPARSE_SPMV_CSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_CSR_ALG2:
            *out = CUSPARSE_SPMV_CSR_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_SELL_ALG1:
            *out = CUSPARSE_SPMV_SELL_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_BSR_ALG1:
            *out = CUSPARSE_SPMV_BSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 12011 && CUDART_VERSION < 13001)
    inline hipsparseStatus_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg, cusparseSpMVAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPMV_ALG_DEFAULT:
            *out = CUSPARSE_SPMV_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_COO_ALG1:
            *out = CUSPARSE_SPMV_COO_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_COO_ALG2:
            *out = CUSPARSE_SPMV_COO_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_CSR_ALG1:
            *out = CUSPARSE_SPMV_CSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_CSR_ALG2:
            *out = CUSPARSE_SPMV_CSR_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_SELL_ALG1:
            *out = CUSPARSE_SPMV_SELL_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 12000 && CUDART_VERSION < 12011)
    inline hipsparseStatus_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg, cusparseSpMVAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPMV_ALG_DEFAULT:
            *out = CUSPARSE_SPMV_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_COO_ALG1:
            *out = CUSPARSE_SPMV_COO_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_COO_ALG2:
            *out = CUSPARSE_SPMV_COO_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_CSR_ALG1:
            *out = CUSPARSE_SPMV_CSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_CSR_ALG2:
            *out = CUSPARSE_SPMV_CSR_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
    inline hipsparseStatus_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg, cusparseSpMVAlg_t* out)
    {
        switch(alg)
        {
        // case HIPSPARSE_MV_ALG_DEFAULT:
        case HIPSPARSE_SPMV_ALG_DEFAULT:
            *out = CUSPARSE_SPMV_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMV_ALG:
        case HIPSPARSE_SPMV_COO_ALG1:
            *out = CUSPARSE_SPMV_COO_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_COO_ALG2:
            *out = CUSPARSE_SPMV_COO_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_CSRMV_ALG1:
        case HIPSPARSE_SPMV_CSR_ALG1:
            *out = CUSPARSE_SPMV_CSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_CSRMV_ALG2:
        case HIPSPARSE_SPMV_CSR_ALG2:
            *out = CUSPARSE_SPMV_CSR_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
    inline hipsparseStatus_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg, cusparseSpMVAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_MV_ALG_DEFAULT:
            *out = CUSPARSE_MV_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_COOMV_ALG:
            *out = CUSPARSE_COOMV_ALG;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_CSRMV_ALG1:
            *out = CUSPARSE_CSRMV_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_CSRMV_ALG2:
            *out = CUSPARSE_CSRMV_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

#if(CUDART_VERSION >= 12051)
    inline hipsparseStatus_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg, cusparseSpMMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            *out = CUSPARSE_SPMM_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG1:
            *out = CUSPARSE_SPMM_COO_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG2:
            *out = CUSPARSE_SPMM_COO_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG3:
            *out = CUSPARSE_SPMM_COO_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG4:
            *out = CUSPARSE_SPMM_COO_ALG4;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG1:
            *out = CUSPARSE_SPMM_CSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG2:
            *out = CUSPARSE_SPMM_CSR_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG3:
            *out = CUSPARSE_SPMM_CSR_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            *out = CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_BSR_ALG1:
            *out = CUSPARSE_SPMM_BSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 12000 && CUDART_VERSION < 12051)
    inline hipsparseStatus_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg, cusparseSpMMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            *out = CUSPARSE_SPMM_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG1:
            *out = CUSPARSE_SPMM_COO_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG2:
            *out = CUSPARSE_SPMM_COO_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG3:
            *out = CUSPARSE_SPMM_COO_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG4:
            *out = CUSPARSE_SPMM_COO_ALG4;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG1:
            *out = CUSPARSE_SPMM_CSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG2:
            *out = CUSPARSE_SPMM_CSR_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG3:
            *out = CUSPARSE_SPMM_CSR_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            *out = CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
    inline hipsparseStatus_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg, cusparseSpMMAlg_t* out)
    {
        switch(alg)
        {
        // case HIPSPARSE_MM_ALG_DEFAULT:
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            *out = CUSPARSE_SPMM_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG1:
        case HIPSPARSE_SPMM_COO_ALG1:
            *out = CUSPARSE_SPMM_COO_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG2:
        case HIPSPARSE_SPMM_COO_ALG2:
            *out = CUSPARSE_SPMM_COO_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG3:
        case HIPSPARSE_SPMM_COO_ALG3:
            *out = CUSPARSE_SPMM_COO_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG4:
            *out = CUSPARSE_SPMM_COO_ALG4;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_CSRMM_ALG1:
        case HIPSPARSE_SPMM_CSR_ALG1:
            *out = CUSPARSE_SPMM_CSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG2:
            *out = CUSPARSE_SPMM_CSR_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG3:
            *out = CUSPARSE_SPMM_CSR_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            *out = CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 11003 && CUDART_VERSION < 11021)
    inline hipsparseStatus_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg, cusparseSpMMAlg_t* out)
    {
        switch(alg)
        {
        // case HIPSPARSE_MM_ALG_DEFAULT:
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            *out = CUSPARSE_SPMM_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG1:
        case HIPSPARSE_SPMM_COO_ALG1:
            *out = CUSPARSE_SPMM_COO_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG2:
        case HIPSPARSE_SPMM_COO_ALG2:
            *out = CUSPARSE_SPMM_COO_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG3:
        case HIPSPARSE_SPMM_COO_ALG3:
            *out = CUSPARSE_SPMM_COO_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_COO_ALG4:
            *out = CUSPARSE_SPMM_COO_ALG4;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_CSRMM_ALG1:
        case HIPSPARSE_SPMM_CSR_ALG1:
            *out = CUSPARSE_SPMM_CSR_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG2:
            *out = CUSPARSE_SPMM_CSR_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            *out = CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11003)
    inline hipsparseStatus_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg, cusparseSpMMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_MM_ALG_DEFAULT:
            *out = CUSPARSE_MM_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_COOMM_ALG1:
            *out = CUSPARSE_COOMM_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_COOMM_ALG2:
            *out = CUSPARSE_COOMM_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_COOMM_ALG3:
            *out = CUSPARSE_COOMM_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_CSRMM_ALG1:
            *out = CUSPARSE_CSRMM_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

#if(CUDART_VERSION >= 12000)
    inline hipsparseStatus_t hipSpGEMMAlgToCudaSpGEMMAlg(hipsparseSpGEMMAlg_t alg,
                                                         cusparseSpGEMMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEMM_DEFAULT:
            *out = CUSPARSE_SPGEMM_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC:
            *out = CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC:
            *out = CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPGEMM_ALG1:
            *out = CUSPARSE_SPGEMM_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPGEMM_ALG2:
            *out = CUSPARSE_SPGEMM_ALG2;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPGEMM_ALG3:
            *out = CUSPARSE_SPGEMM_ALG3;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 11031)
    inline hipsparseStatus_t hipSpGEMMAlgToCudaSpGEMMAlg(hipsparseSpGEMMAlg_t alg,
                                                         cusparseSpGEMMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEMM_DEFAULT:
            *out = CUSPARSE_SPGEMM_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC:
            *out = CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC:
            *out = CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#elif(CUDART_VERSION >= 11000)
    inline hipsparseStatus_t hipSpGEMMAlgToCudaSpGEMMAlg(hipsparseSpGEMMAlg_t alg,
                                                         cusparseSpGEMMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEMM_DEFAULT:
            *out = CUSPARSE_SPGEMM_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

    // clang-format 19 (math-ci) rewrites "#if(" to "#if (", which clang-format
    // 18 (the repo pre-commit hook) reverts; disable so both formatters agree.
    // clang-format off
#if(CUDART_VERSION >= 13030)
    // clang-format on
    inline hipsparseStatus_t hipSpGEAMAlgToCudaSpGEAMAlg(hipsparseSpGEAMAlg_t alg,
                                                         cusparseSpGEAMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEAM_ALG1:
            *out = CUSPARSE_SPGEAM_ALG1;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

#if(CUDART_VERSION >= 11020)
    inline hipsparseStatus_t hipSpToDnAlgToCudaSpToDnAlg(hipsparseSparseToDenseAlg_t alg,
                                                         cusparseSparseToDenseAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPARSETODENSE_ALG_DEFAULT:
            *out = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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
    inline hipsparseStatus_t hipDnToSpAlgToCudaDnToSpAlg(hipsparseDenseToSparseAlg_t alg,
                                                         cusparseDenseToSparseAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT:
            *out = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
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
    inline hipsparseStatus_t hipSDDMMAlgToCudaSDDMMAlg(hipsparseSDDMMAlg_t alg,
                                                       cusparseSDDMMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SDDMM_ALG_DEFAULT:
            *out = CUSPARSE_SDDMM_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

#if(CUDART_VERSION >= 11030)
    inline hipsparseStatus_t hipSpSVAlgToCudaSpSVAlg(hipsparseSpSVAlg_t alg, cusparseSpSVAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPSV_ALG_DEFAULT:
            *out = CUSPARSE_SPSV_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

#if(CUDART_VERSION >= 11031)
    inline hipsparseStatus_t hipSpSMAlgToCudaSpSMAlg(hipsparseSpSMAlg_t alg, cusparseSpSMAlg_t* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPSM_ALG_DEFAULT:
            *out = CUSPARSE_SPSM_ALG_DEFAULT;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif

#if(CUDART_VERSION >= 11030)
    inline hipsparseStatus_t
        hipSpMatAttributeToCudaSpMatAttribute(hipsparseSpMatAttribute_t attribute,
                                              cusparseSpMatAttribute_t* out)
    {
        switch(attribute)
        {
        case HIPSPARSE_SPMAT_FILL_MODE:
            *out = CUSPARSE_SPMAT_FILL_MODE;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMAT_DIAG_TYPE:
            *out = CUSPARSE_SPMAT_DIAG_TYPE;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }
#endif
}
