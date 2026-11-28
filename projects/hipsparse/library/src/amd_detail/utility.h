/* ************************************************************************
* Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "hipsparse-config.h"

#include <vector>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                            \
    {                                                                          \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;              \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                                 \
        {                                                                      \
            return hipsparse::hipErrorToHIPSPARSEStatus(TMP_STATUS_FOR_CHECK); \
        }                                                                      \
    }

#define RETURN_IF_HIPSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                                    \
        hipsparseStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != HIPSPARSE_STATUS_SUCCESS)             \
        {                                                                \
            return TMP_STATUS_FOR_CHECK;                                 \
        }                                                                \
    }

#define RETURN_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                       \
    {                                                                           \
        rocsparse_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;         \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)                    \
        {                                                                       \
            return hipsparse::rocSPARSEStatusToHIPStatus(TMP_STATUS_FOR_CHECK); \
        }                                                                       \
    }

namespace hipsparse
{
    inline hipsparseStatus_t hipErrorToHIPSPARSEStatus(hipError_t status)
    {
        switch(status)
        {
        case hipSuccess:
            return HIPSPARSE_STATUS_SUCCESS;
        case hipErrorMemoryAllocation:
        case hipErrorLaunchOutOfResources:
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        case hipErrorInvalidDevicePointer:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        case hipErrorInvalidDevice:
        case hipErrorInvalidResourceHandle:
            return HIPSPARSE_STATUS_NOT_INITIALIZED;
        case hipErrorInvalidValue:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        case hipErrorNoDevice:
        case hipErrorUnknown:
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        default:
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }

    inline hipsparseStatus_t rocSPARSEStatusToHIPStatus(rocsparse_status_ status)
    {
        switch(status)
        {
        case rocsparse_status_success:
            return HIPSPARSE_STATUS_SUCCESS;
        case rocsparse_status_invalid_handle:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        case rocsparse_status_not_implemented:
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        case rocsparse_status_invalid_pointer:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        case rocsparse_status_invalid_size:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        case rocsparse_status_memory_error:
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        case rocsparse_status_internal_error:
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        case rocsparse_status_invalid_value:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        case rocsparse_status_arch_mismatch:
            return HIPSPARSE_STATUS_ARCH_MISMATCH;
        case rocsparse_status_zero_pivot:
            return HIPSPARSE_STATUS_ZERO_PIVOT;
        case rocsparse_status_not_initialized:
            return HIPSPARSE_STATUS_NOT_INITIALIZED;
        case rocsparse_status_type_mismatch:
        case rocsparse_status_requires_sorted_storage:
        case rocsparse_status_thrown_exception:
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        case rocsparse_status_continue:
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            throw "Non existent rocsparse_status";
        }
    }

    inline rocsparse_status_ hipSPARSEStatusToRocSPARSEStatus(hipsparseStatus_t status)
    {
        switch(status)
        {
        case HIPSPARSE_STATUS_SUCCESS:
            return rocsparse_status_success;
        case HIPSPARSE_STATUS_NOT_INITIALIZED:
            return rocsparse_status_not_initialized;
        case HIPSPARSE_STATUS_ALLOC_FAILED:
            return rocsparse_status_memory_error;
        case HIPSPARSE_STATUS_INVALID_VALUE:
            return rocsparse_status_invalid_value;
        case HIPSPARSE_STATUS_ARCH_MISMATCH:
            return rocsparse_status_arch_mismatch;
        case HIPSPARSE_STATUS_MAPPING_ERROR:
        case HIPSPARSE_STATUS_EXECUTION_FAILED:
        case HIPSPARSE_STATUS_INTERNAL_ERROR:
        case HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return rocsparse_status_internal_error;
        case HIPSPARSE_STATUS_ZERO_PIVOT:
            return rocsparse_status_zero_pivot;
        case HIPSPARSE_STATUS_NOT_SUPPORTED:
            return rocsparse_status_not_implemented;
        case HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            return rocsparse_status_internal_error;
        default:
            throw "Non existent hipsparseStatus_t";
        }
    }

    inline hipsparseStatus_t hipPtrModeToHCCPtrMode(hipsparsePointerMode_t   mode,
                                                    rocsparse_pointer_mode_* out)
    {
        switch(mode)
        {
        case HIPSPARSE_POINTER_MODE_HOST:
            *out = rocsparse_pointer_mode_host;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_POINTER_MODE_DEVICE:
            *out = rocsparse_pointer_mode_device;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparsePointerMode_t HCCPtrModeToHIPPtrMode(rocsparse_pointer_mode_ mode)
    {
        switch(mode)
        {
        case rocsparse_pointer_mode_host:
            return HIPSPARSE_POINTER_MODE_HOST;
        case rocsparse_pointer_mode_device:
            return HIPSPARSE_POINTER_MODE_DEVICE;
        default:
            throw "Non existent rocsparse_pointer_mode";
        }
    }

    inline hipsparseStatus_t hipActionToHCCAction(hipsparseAction_t action, rocsparse_action_* out)
    {
        switch(action)
        {
        case HIPSPARSE_ACTION_SYMBOLIC:
            *out = rocsparse_action_symbolic;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_ACTION_NUMERIC:
            *out = rocsparse_action_numeric;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipMatTypeToHCCMatType(hipsparseMatrixType_t   type,
                                                    rocsparse_matrix_type_* out)
    {
        switch(type)
        {
        case HIPSPARSE_MATRIX_TYPE_GENERAL:
            *out = rocsparse_matrix_type_general;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_MATRIX_TYPE_SYMMETRIC:
            *out = rocsparse_matrix_type_symmetric;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_MATRIX_TYPE_HERMITIAN:
            *out = rocsparse_matrix_type_hermitian;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_MATRIX_TYPE_TRIANGULAR:
            *out = rocsparse_matrix_type_triangular;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseMatrixType_t HCCMatTypeToHIPMatType(rocsparse_matrix_type_ type)
    {
        switch(type)
        {
        case rocsparse_matrix_type_general:
            return HIPSPARSE_MATRIX_TYPE_GENERAL;
        case rocsparse_matrix_type_symmetric:
            return HIPSPARSE_MATRIX_TYPE_SYMMETRIC;
        case rocsparse_matrix_type_hermitian:
            return HIPSPARSE_MATRIX_TYPE_HERMITIAN;
        case rocsparse_matrix_type_triangular:
            return HIPSPARSE_MATRIX_TYPE_TRIANGULAR;
        default:
            throw "Non existent rocsparse_matrix_type";
        }
    }

    inline hipsparseStatus_t hipFillModeToHCCFillMode(hipsparseFillMode_t   fillMode,
                                                      rocsparse_fill_mode_* out)
    {
        switch(fillMode)
        {
        case HIPSPARSE_FILL_MODE_LOWER:
            *out = rocsparse_fill_mode_lower;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FILL_MODE_UPPER:
            *out = rocsparse_fill_mode_upper;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseFillMode_t HCCFillModeToHIPFillMode(rocsparse_fill_mode_ fillMode)
    {
        switch(fillMode)
        {
        case rocsparse_fill_mode_lower:
            return HIPSPARSE_FILL_MODE_LOWER;
        case rocsparse_fill_mode_upper:
            return HIPSPARSE_FILL_MODE_UPPER;
        default:
            throw "Non existent rocsparse_fill_mode";
        }
    }

    inline hipsparseStatus_t hipDiagTypeToHCCDiagType(hipsparseDiagType_t   diagType,
                                                      rocsparse_diag_type_* out)
    {
        switch(diagType)
        {
        case HIPSPARSE_DIAG_TYPE_UNIT:
            *out = rocsparse_diag_type_unit;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_DIAG_TYPE_NON_UNIT:
            *out = rocsparse_diag_type_non_unit;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseDiagType_t HCCDiagTypeToHIPDiagType(rocsparse_diag_type_ diagType)
    {
        switch(diagType)
        {
        case rocsparse_diag_type_unit:
            return HIPSPARSE_DIAG_TYPE_UNIT;
        case rocsparse_diag_type_non_unit:
            return HIPSPARSE_DIAG_TYPE_NON_UNIT;
        default:
            throw "Non existent rocsparse_diag_type";
        }
    }

    inline hipsparseStatus_t hipBaseToHCCBase(hipsparseIndexBase_t base, rocsparse_index_base_* out)
    {
        switch(base)
        {
        case HIPSPARSE_INDEX_BASE_ZERO:
            *out = rocsparse_index_base_zero;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_INDEX_BASE_ONE:
            *out = rocsparse_index_base_one;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseIndexBase_t HCCBaseToHIPBase(rocsparse_index_base_ base)
    {
        switch(base)
        {
        case rocsparse_index_base_zero:
            return HIPSPARSE_INDEX_BASE_ZERO;
        case rocsparse_index_base_one:
            return HIPSPARSE_INDEX_BASE_ONE;
        default:
            throw "Non existent rocsparse_index_base";
        }
    }

    inline hipsparseStatus_t hipOperationToHCCOperation(hipsparseOperation_t  op,
                                                        rocsparse_operation_* out)
    {
        switch(op)
        {
        case HIPSPARSE_OPERATION_NON_TRANSPOSE:
            *out = rocsparse_operation_none;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_OPERATION_TRANSPOSE:
            *out = rocsparse_operation_transpose;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            *out = rocsparse_operation_conjugate_transpose;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseOperation_t HCCOperationToHIPOperation(rocsparse_operation_ op)
    {
        switch(op)
        {
        case rocsparse_operation_none:
            return HIPSPARSE_OPERATION_NON_TRANSPOSE;
        case rocsparse_operation_transpose:
            return HIPSPARSE_OPERATION_TRANSPOSE;
        case rocsparse_operation_conjugate_transpose:
            return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        default:
            throw "Non existent rocsparse_operation";
        }
    }

    inline hipsparseStatus_t hipHybPartToHCCHybPart(hipsparseHybPartition_t   partition,
                                                    rocsparse_hyb_partition_* out)
    {
        switch(partition)
        {
        case HIPSPARSE_HYB_PARTITION_AUTO:
            *out = rocsparse_hyb_partition_auto;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_HYB_PARTITION_USER:
            *out = rocsparse_hyb_partition_user;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_HYB_PARTITION_MAX:
            *out = rocsparse_hyb_partition_max;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseHybPartition_t HCCHybPartToHIPHybPart(rocsparse_hyb_partition_ partition)
    {
        switch(partition)
        {
        case rocsparse_hyb_partition_auto:
            return HIPSPARSE_HYB_PARTITION_AUTO;
        case rocsparse_hyb_partition_user:
            return HIPSPARSE_HYB_PARTITION_USER;
        case rocsparse_hyb_partition_max:
            return HIPSPARSE_HYB_PARTITION_MAX;
        default:
            throw "Non existent rocsparse_hyb_partition";
        }
    }

    inline hipsparseStatus_t hipSpGEAMAlgToHCCSpGEAMAlg(hipsparseSpGEAMAlg_t   alg,
                                                        rocsparse_spgeam_alg_* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEAM_ALG1:
            *out = rocsparse_spgeam_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipPolicyToHCCPolicy(hipsparseSolvePolicy_t   policy,
                                                  rocsparse_solve_policy_* out)
    {
        switch(policy)
        {
        case HIPSPARSE_SOLVE_POLICY_NO_LEVEL:
        case HIPSPARSE_SOLVE_POLICY_USE_LEVEL:
            *out = rocsparse_solve_policy_auto;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipCsr2CscAlgCheck(hipsparseCsr2CscAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_CSR2CSC_ALG_DEFAULT:
        case HIPSPARSE_CSR2CSC_ALG1:
        case HIPSPARSE_CSR2CSC_ALG2:
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t
        hipSpMatAttributeToHCCSpMatAttribute(hipsparseSpMatAttribute_t   attribute,
                                             rocsparse_spmat_attribute_* out)
    {
        switch(attribute)
        {
        case HIPSPARSE_SPMAT_FILL_MODE:
            *out = rocsparse_spmat_fill_mode;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMAT_DIAG_TYPE:
            *out = rocsparse_spmat_diag_type;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipDirectionToHCCDirection(hipsparseDirection_t  op,
                                                        rocsparse_direction_* out)
    {
        switch(op)
        {
        case HIPSPARSE_DIRECTION_ROW:
            *out = rocsparse_direction_row;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_DIRECTION_COLUMN:
            *out = rocsparse_direction_column;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseDirection_t HCCDirectionToHIPDirection(rocsparse_direction_ op)
    {
        switch(op)
        {
        case rocsparse_direction_row:
            return HIPSPARSE_DIRECTION_ROW;
        case rocsparse_direction_column:
            return HIPSPARSE_DIRECTION_COLUMN;
        default:
            throw "Non existent rocsparse_direction";
        }
    }

    inline hipsparseStatus_t hipOrderToHCCOrder(hipsparseOrder_t op, rocsparse_order_* out)
    {
        switch(op)
        {
        case HIPSPARSE_ORDER_ROW:
            *out = rocsparse_order_row;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_ORDER_COL:
            *out = rocsparse_order_column;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseOrder_t HCCOrderToHIPOrder(rocsparse_order_ op)
    {
        switch(op)
        {
        case rocsparse_order_row:
            return HIPSPARSE_ORDER_ROW;
        case rocsparse_order_column:
            return HIPSPARSE_ORDER_COL;
        default:
            throw "Non existent rocsparse_order";
        }
    }

    inline hipsparseStatus_t hipIndexTypeToHCCIndexType(hipsparseIndexType_t  indextype,
                                                        rocsparse_indextype_* out)
    {
        switch(indextype)
        {
        case HIPSPARSE_INDEX_32I:
            *out = rocsparse_indextype_i32;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_INDEX_64I:
            *out = rocsparse_indextype_i64;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_INDEX_16U:
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseIndexType_t HCCIndexTypeToHIPIndexType(rocsparse_indextype_ indextype)
    {
        switch(indextype)
        {
        case rocsparse_indextype_i32:
            return HIPSPARSE_INDEX_32I;
        case rocsparse_indextype_i64:
            return HIPSPARSE_INDEX_64I;
        default:
            throw "Non existent rocsparse_indextype";
        }
    }

    inline hipsparseStatus_t hipDataTypeToHCCDataType(hipDataType          datatype,
                                                      rocsparse_datatype_* out)
    {
        switch(datatype)
        {
        case HIP_R_8I:
            *out = rocsparse_datatype_i8_r;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_32I:
            *out = rocsparse_datatype_i32_r;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_16F:
            *out = rocsparse_datatype_f16_r;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_16BF:
            *out = rocsparse_datatype_bf16_r;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_32F:
            *out = rocsparse_datatype_f32_r;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_R_64F:
            *out = rocsparse_datatype_f64_r;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_C_32F:
            *out = rocsparse_datatype_f32_c;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_C_64F:
            *out = rocsparse_datatype_f64_c;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIP_C_16F:
        case HIP_C_8I:
        case HIP_C_8U:
        case HIP_R_8U:
        case HIP_C_32I:
        case HIP_R_32U:
        case HIP_C_32U:
        case HIP_C_16BF:
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
        case HIP_R_8F_E4M3:
        case HIP_R_8F_E5M2:
        case HIP_R_8F_UE8M0:
        case HIP_R_6F_E2M3:
        case HIP_R_6F_E3M2:
        case HIP_R_4F_E2M1:
        case HIP_R_8F_E4M3_FNUZ:
        case HIP_R_8F_E5M2_FNUZ:
            return HIPSPARSE_STATUS_NOT_SUPPORTED;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipDataType HCCDataTypeToHIPDataType(rocsparse_datatype_ datatype)
    {
        switch(datatype)
        {
        case rocsparse_datatype_i8_r:
            return HIP_R_8I;
        case rocsparse_datatype_i32_r:
            return HIP_R_32I;
        case rocsparse_datatype_f16_r:
            return HIP_R_16F;
        case rocsparse_datatype_bf16_r:
            return HIP_R_16BF;
        case rocsparse_datatype_f32_r:
            return HIP_R_32F;
        case rocsparse_datatype_f64_r:
            return HIP_R_64F;
        case rocsparse_datatype_f32_c:
            return HIP_C_32F;
        case rocsparse_datatype_f64_c:
            return HIP_C_64F;
        default:
            throw "Non existent rocsparse_datatype";
        }
    }

    inline hipsparseStatus_t hipSpMVAlgToHCCSpMVAlg(hipsparseSpMVAlg_t   alg,
                                                    rocsparse_spmv_alg_* out)
    {
        switch(alg)
        {
        // case HIPSPARSE_MV_ALG_DEFAULT:
        case HIPSPARSE_SPMV_ALG_DEFAULT:
            *out = rocsparse_spmv_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMV_ALG:
        case HIPSPARSE_SPMV_COO_ALG1:
        case HIPSPARSE_SPMV_COO_ALG2:
            *out = rocsparse_spmv_alg_coo;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_CSRMV_ALG1:
        case HIPSPARSE_SPMV_CSR_ALG1:
            *out = rocsparse_spmv_alg_csr_adaptive;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_CSRMV_ALG2:
        case HIPSPARSE_SPMV_CSR_ALG2:
            *out = rocsparse_spmv_alg_csr_rowsplit;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_CSR_ALG3:
            *out = rocsparse_spmv_alg_csr_nnzsplit;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_SELL_ALG1:
            *out = rocsparse_spmv_alg_sell;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMV_BSR_ALG1:
            *out = rocsparse_spmv_alg_bsr;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipSpMMAlgToHCCSpMMAlg(hipsparseSpMMAlg_t   alg,
                                                    rocsparse_spmm_alg_* out)
    {
        switch(alg)
        {
        // case HIPSPARSE_MM_ALG_DEFAULT:
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            *out = rocsparse_spmm_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG1:
        case HIPSPARSE_SPMM_COO_ALG1:
            *out = rocsparse_spmm_alg_coo_atomic;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG2:
        case HIPSPARSE_SPMM_COO_ALG2:
            *out = rocsparse_spmm_alg_coo_segmented;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_COOMM_ALG3:
        case HIPSPARSE_SPMM_COO_ALG3:
        case HIPSPARSE_SPMM_COO_ALG4:
            *out = rocsparse_spmm_alg_coo_segmented_atomic;
            return HIPSPARSE_STATUS_SUCCESS;
        // case HIPSPARSE_CSRMM_ALG1:
        case HIPSPARSE_SPMM_CSR_ALG1:
            *out = rocsparse_spmm_alg_csr_row_split;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_CSR_ALG2:
        case HIPSPARSE_SPMM_CSR_ALG3:
            *out = rocsparse_spmm_alg_csr;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            *out = rocsparse_spmm_alg_bell;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_SPMM_BSR_ALG1:
            *out = rocsparse_spmm_alg_bsr;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipSpToDnAlgToHCCSpToDnAlg(hipsparseSparseToDenseAlg_t     alg,
                                                        rocsparse_sparse_to_dense_alg_* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPARSETODENSE_ALG_DEFAULT:
            *out = rocsparse_sparse_to_dense_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseSparseToDenseAlg_t
        HCCSpToDnAlgToHipSpToDnAlg(rocsparse_sparse_to_dense_alg_ alg)
    {
        switch(alg)
        {
        case rocsparse_sparse_to_dense_alg_default:
            return HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
        default:
            throw "Non existent rocsparse_sparse_to_dense_alg";
        }
    }

    inline hipsparseStatus_t hipDnToSpAlgToHCCDnToSpAlg(hipsparseDenseToSparseAlg_t     alg,
                                                        rocsparse_dense_to_sparse_alg_* out)
    {
        switch(alg)
        {
        case HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT:
            *out = rocsparse_dense_to_sparse_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseDenseToSparseAlg_t
        HCCDnToSpAlgToHipDnToSpAlg(rocsparse_dense_to_sparse_alg_ alg)
    {
        switch(alg)
        {
        case rocsparse_dense_to_sparse_alg_default:
            return HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
        default:
            throw "Non existent rocsparse_dense_to_sparse_alg";
        }
    }

    inline hipsparseStatus_t hipSpGEMMAlgToHCCSpGEMMAlg(hipsparseSpGEMMAlg_t   alg,
                                                        rocsparse_spgemm_alg_* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEMM_DEFAULT:
        case HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC:
        case HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC:
        case HIPSPARSE_SPGEMM_ALG1:
        case HIPSPARSE_SPGEMM_ALG2:
        case HIPSPARSE_SPGEMM_ALG3:
            *out = rocsparse_spgemm_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipSDDMMAlgToHCCSDDMMAlg(hipsparseSDDMMAlg_t   alg,
                                                      rocsparse_sddmm_alg_* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SDDMM_ALG_DEFAULT:
            *out = rocsparse_sddmm_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipSpSVAlgToHCCSpSVAlg(hipsparseSpSVAlg_t   alg,
                                                    rocsparse_spsv_alg_* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPSV_ALG_DEFAULT:
            *out = rocsparse_spsv_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipSpSMAlgToHCCSpSMAlg(hipsparseSpSMAlg_t   alg,
                                                    rocsparse_spsm_alg_* out)
    {
        switch(alg)
        {
        case HIPSPARSE_SPSM_ALG_DEFAULT:
            *out = rocsparse_spsm_alg_default;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseStatus_t hipFormatToHCCFormat(hipsparseFormat_t format, rocsparse_format_* out)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            *out = rocsparse_format_csr;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_CSC:
            *out = rocsparse_format_csc;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_COO:
            *out = rocsparse_format_coo;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_COO_AOS:
            *out = rocsparse_format_coo_aos;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_BLOCKED_ELL:
            *out = rocsparse_format_bell;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_SLICED_ELL:
            *out = rocsparse_format_sell;
            return HIPSPARSE_STATUS_SUCCESS;
        case HIPSPARSE_FORMAT_BSR:
            *out = rocsparse_format_bsr;
            return HIPSPARSE_STATUS_SUCCESS;
        default:
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }
    }

    inline hipsparseFormat_t HCCFormatToHIPFormat(rocsparse_format_ format)
    {
        switch(format)
        {
        case rocsparse_format_csr:
            return HIPSPARSE_FORMAT_CSR;
        case rocsparse_format_csc:
            return HIPSPARSE_FORMAT_CSC;
        case rocsparse_format_coo:
            return HIPSPARSE_FORMAT_COO;
        case rocsparse_format_coo_aos:
            return HIPSPARSE_FORMAT_COO_AOS;
        case rocsparse_format_bell:
            return HIPSPARSE_FORMAT_BLOCKED_ELL;
        case rocsparse_format_sell:
            return HIPSPARSE_FORMAT_SLICED_ELL;
        case rocsparse_format_bsr:
            return HIPSPARSE_FORMAT_BSR;
        default:
            throw "Non existent rocsparse_format";
        }
    }
}

struct hipsparseSpMVDescr_st
{
public:
    // Per-configuration cache entry. Each entry holds the rocsparse_spmv_descr,
    // analysis state, and compute-stage buffer for one unique combination of
    // (operation, algorithm, scalar datatype, compute datatype). Callers that
    // alternate among several configurations (e.g. NON_TRANSPOSE and TRANSPOSE
    // in an iterative solver) get a separate entry per configuration, so
    // switching back and forth does not invalidate any analysis or force any
    // buffer reallocation.
    struct Entry
    {
        rocsparse_operation  operation{};
        rocsparse_spmv_alg   alg{};
        rocsparse_datatype   scalar_datatype{};
        rocsparse_datatype   compute_datatype{};
        rocsparse_spmv_descr spmv_descr{};
        size_t               buffer_size_stage_analysis{};
        size_t               buffer_size_stage_compute{};
        void*                compute_buffer{};
        bool                 is_buffer_size_called{};
        bool                 is_stage_analysis_called{};
        bool                 is_implicit_stage_analysis_called{};
        bool                 is_stage_compute_subsequent{};

        Entry()             = default;
        Entry(const Entry&) = delete;
        Entry& operator=(const Entry&) = delete;
        Entry(Entry&& other) noexcept;
        Entry& operator=(Entry&&) = delete;
        ~Entry();
    };

protected:
    std::vector<Entry> m_entries;

public:
    // Look up an entry matching (operation, alg, scalar_datatype,
    // compute_datatype); returns nullptr if none exists.
    Entry* find_entry(rocsparse_operation operation,
                      rocsparse_spmv_alg  alg,
                      rocsparse_datatype  scalar_datatype,
                      rocsparse_datatype  compute_datatype);

    // Create a new entry for (operation, alg, scalar_datatype,
    // compute_datatype), allocate its rocsparse_spmv_descr, push the four
    // set_input values, and store it in m_entries. Returns the new entry via
    // *out_entry.
    hipsparseStatus_t add_entry(hipsparseHandle_t   handle,
                                rocsparse_operation operation,
                                rocsparse_spmv_alg  alg,
                                rocsparse_datatype  scalar_datatype,
                                rocsparse_datatype  compute_datatype,
                                Entry**             out_entry);

    hipsparseSpMVDescr_st()  = default;
    ~hipsparseSpMVDescr_st() = default;
};

struct hipsparseSpMatDescr_st
{
protected:
    rocsparse_spmat_descr         m_spmat_descr{};
    mutable hipsparseSpMVDescr_st m_hip_spmv_descr{};

public:
    hipsparseSpMatDescr_st()  = default;
    ~hipsparseSpMatDescr_st() = default;
    hipsparseSpMVDescr_st*       get_hip_spmv_descr();
    hipsparseSpMVDescr_st*       get_hip_spmv_descr() const;
    rocsparse_spmat_descr        get_spmat_descr();
    rocsparse_const_spmat_descr  get_const_spmat_descr() const;
    rocsparse_spmat_descr*       get_spmat_descr_reference();
    rocsparse_const_spmat_descr* get_const_spmat_descr_reference() const;
    void                         set_spmat_descr(rocsparse_spmat_descr value);
};

rocsparse_spmat_descr       to_rocsparse_spmat_descr(const hipsparseSpMatDescr_t source);
rocsparse_const_spmat_descr to_rocsparse_const_spmat_descr(const hipsparseConstSpMatDescr_t source);
