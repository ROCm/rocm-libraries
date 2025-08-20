/* ************************************************************************
* Copyright (C) 2018-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

    inline rocsparse_pointer_mode_ hipPtrModeToHCCPtrMode(hipsparsePointerMode_t mode)
    {
        switch(mode)
        {
        case HIPSPARSE_POINTER_MODE_HOST:
            return rocsparse_pointer_mode_host;
        case HIPSPARSE_POINTER_MODE_DEVICE:
            return rocsparse_pointer_mode_device;
        default:
            throw "Non existent hipsparsePointerMode_t";
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

    inline rocsparse_action_ hipActionToHCCAction(hipsparseAction_t action)
    {
        switch(action)
        {
        case HIPSPARSE_ACTION_SYMBOLIC:
            return rocsparse_action_symbolic;
        case HIPSPARSE_ACTION_NUMERIC:
            return rocsparse_action_numeric;
        default:
            throw "Non existent hipsparseAction_t";
        }
    }

    inline rocsparse_matrix_type_ hipMatTypeToHCCMatType(hipsparseMatrixType_t type)
    {
        switch(type)
        {
        case HIPSPARSE_MATRIX_TYPE_GENERAL:
            return rocsparse_matrix_type_general;
        case HIPSPARSE_MATRIX_TYPE_SYMMETRIC:
            return rocsparse_matrix_type_symmetric;
        case HIPSPARSE_MATRIX_TYPE_HERMITIAN:
            return rocsparse_matrix_type_hermitian;
        case HIPSPARSE_MATRIX_TYPE_TRIANGULAR:
            return rocsparse_matrix_type_triangular;
        default:
            throw "Non existent hipsparseMatrixType_t";
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

    inline rocsparse_fill_mode_ hipFillModeToHCCFillMode(hipsparseFillMode_t fillMode)
    {
        switch(fillMode)
        {
        case HIPSPARSE_FILL_MODE_LOWER:
            return rocsparse_fill_mode_lower;
        case HIPSPARSE_FILL_MODE_UPPER:
            return rocsparse_fill_mode_upper;
        default:
            throw "Non existent hipsparseFillMode_t";
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

    inline rocsparse_diag_type_ hipDiagTypeToHCCDiagType(hipsparseDiagType_t diagType)
    {
        switch(diagType)
        {
        case HIPSPARSE_DIAG_TYPE_UNIT:
            return rocsparse_diag_type_unit;
        case HIPSPARSE_DIAG_TYPE_NON_UNIT:
            return rocsparse_diag_type_non_unit;
        default:
            throw "Non existent hipsparseDiagType_t";
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

    inline rocsparse_index_base_ hipBaseToHCCBase(hipsparseIndexBase_t base)
    {
        switch(base)
        {
        case HIPSPARSE_INDEX_BASE_ZERO:
            return rocsparse_index_base_zero;
        case HIPSPARSE_INDEX_BASE_ONE:
            return rocsparse_index_base_one;
        default:
            throw "Non existent hipsparseIndexBase_t";
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

    inline rocsparse_operation_ hipOperationToHCCOperation(hipsparseOperation_t op)
    {
        switch(op)
        {
        case HIPSPARSE_OPERATION_NON_TRANSPOSE:
            return rocsparse_operation_none;
        case HIPSPARSE_OPERATION_TRANSPOSE:
            return rocsparse_operation_transpose;
        case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            return rocsparse_operation_conjugate_transpose;
        default:
            throw "Non existent hipsparseOperation_t";
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

    inline rocsparse_hyb_partition_ hipHybPartToHCCHybPart(hipsparseHybPartition_t partition)
    {
        switch(partition)
        {
        case HIPSPARSE_HYB_PARTITION_AUTO:
            return rocsparse_hyb_partition_auto;
        case HIPSPARSE_HYB_PARTITION_USER:
            return rocsparse_hyb_partition_user;
        case HIPSPARSE_HYB_PARTITION_MAX:
            return rocsparse_hyb_partition_max;
        default:
            throw "Non existent hipsparseHybPartition_t";
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

    inline rocsparse_direction_ hipDirectionToHCCDirection(hipsparseDirection_t op)
    {
        switch(op)
        {
        case HIPSPARSE_DIRECTION_ROW:
            return rocsparse_direction_row;
        case HIPSPARSE_DIRECTION_COLUMN:
            return rocsparse_direction_column;
        default:
            throw "Non existent hipsparseDirection_t";
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

    inline rocsparse_order_ hipOrderToHCCOrder(hipsparseOrder_t op)
    {
        switch(op)
        {
        case HIPSPARSE_ORDER_ROW:
            return rocsparse_order_row;
        case HIPSPARSE_ORDER_COL:
            return rocsparse_order_column;
        default:
            throw "Non existent hipsparseOrder_t";
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

    inline rocsparse_indextype_ hipIndexTypeToHCCIndexType(hipsparseIndexType_t indextype)
    {
        switch(indextype)
        {
        case HIPSPARSE_INDEX_32I:
            return rocsparse_indextype_i32;
        case HIPSPARSE_INDEX_64I:
            return rocsparse_indextype_i64;
        default:
            throw "Non existent hipsparseIndexType_t";
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

    inline rocsparse_datatype_ hipDataTypeToHCCDataType(hipDataType datatype)
    {
        switch(datatype)
        {
        case HIP_R_8I:
            return rocsparse_datatype_i8_r;
        case HIP_R_32I:
            return rocsparse_datatype_i32_r;
        case HIP_R_16F:
            return rocsparse_datatype_f16_r;
        case HIP_R_16BF:
            return rocsparse_datatype_bf16_r;
        case HIP_R_32F:
            return rocsparse_datatype_f32_r;
        case HIP_R_64F:
            return rocsparse_datatype_f64_r;
        case HIP_C_32F:
            return rocsparse_datatype_f32_c;
        case HIP_C_64F:
            return rocsparse_datatype_f64_c;
        default:
            throw "Non existent hipDataType";
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

    inline rocsparse_spmv_alg_ hipSpMVAlgToHCCSpMVAlg(hipsparseSpMVAlg_t alg)
    {
        switch(alg)
        {
        // case HIPSPARSE_MV_ALG_DEFAULT:
        case HIPSPARSE_SPMV_ALG_DEFAULT:
            return rocsparse_spmv_alg_default;
        // case HIPSPARSE_COOMV_ALG:
        case HIPSPARSE_SPMV_COO_ALG1:
        case HIPSPARSE_SPMV_COO_ALG2:
            return rocsparse_spmv_alg_coo;
        // case HIPSPARSE_CSRMV_ALG1:
        case HIPSPARSE_SPMV_CSR_ALG1:
            return rocsparse_spmv_alg_csr_adaptive;
        // case HIPSPARSE_CSRMV_ALG2:
        case HIPSPARSE_SPMV_CSR_ALG2:
            return rocsparse_spmv_alg_csr_stream;
        default:
            throw "Non existent hipsparseSpMVAlg_t";
        }
    }

    inline rocsparse_spmm_alg_ hipSpMMAlgToHCCSpMMAlg(hipsparseSpMMAlg_t alg)
    {
        switch(alg)
        {
        // case HIPSPARSE_MM_ALG_DEFAULT:
        case HIPSPARSE_SPMM_ALG_DEFAULT:
            return rocsparse_spmm_alg_default;
        // case HIPSPARSE_COOMM_ALG1:
        case HIPSPARSE_SPMM_COO_ALG1:
            return rocsparse_spmm_alg_coo_atomic;
        // case HIPSPARSE_COOMM_ALG2:
        case HIPSPARSE_SPMM_COO_ALG2:
            return rocsparse_spmm_alg_coo_segmented;
        // case HIPSPARSE_COOMM_ALG3:
        case HIPSPARSE_SPMM_COO_ALG3:
        case HIPSPARSE_SPMM_COO_ALG4:
            return rocsparse_spmm_alg_coo_segmented_atomic;
        // case HIPSPARSE_CSRMM_ALG1:
        case HIPSPARSE_SPMM_CSR_ALG1:
            return rocsparse_spmm_alg_csr_row_split;
        case HIPSPARSE_SPMM_CSR_ALG2:
        case HIPSPARSE_SPMM_CSR_ALG3:
            return rocsparse_spmm_alg_csr;
        case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
            return rocsparse_spmm_alg_bell;
        default:
            throw "Non existent hipsparseSpMMAlg_t";
        }
    }

    inline rocsparse_sparse_to_dense_alg_
        hipSpToDnAlgToHCCSpToDnAlg(hipsparseSparseToDenseAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPARSETODENSE_ALG_DEFAULT:
            return rocsparse_sparse_to_dense_alg_default;
        default:
            throw "Non existent hipsparseSparseToDenseAlg_t";
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

    inline rocsparse_dense_to_sparse_alg_
        hipDnToSpAlgToHCCDnToSpAlg(hipsparseDenseToSparseAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT:
            return rocsparse_dense_to_sparse_alg_default;
        default:
            throw "Non existent hipsparseDenseToSparseAlg_t";
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

    inline rocsparse_spgemm_alg_ hipSpGEMMAlgToHCCSpGEMMAlg(hipsparseSpGEMMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPGEMM_DEFAULT:
        case HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC:
        case HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC:
        case HIPSPARSE_SPGEMM_ALG1:
        case HIPSPARSE_SPGEMM_ALG2:
        case HIPSPARSE_SPGEMM_ALG3:
            return rocsparse_spgemm_alg_default;
        default:
            throw "Non existent hipSpGEMMAlg_t";
        }
    }

    inline rocsparse_sddmm_alg_ hipSDDMMAlgToHCCSDDMMAlg(hipsparseSDDMMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SDDMM_ALG_DEFAULT:
            return rocsparse_sddmm_alg_default;
        default:
            throw "Non existent hipSDDMMAlg_t";
        }
    }

    inline rocsparse_spsv_alg_ hipSpSVAlgToHCCSpSVAlg(hipsparseSpSVAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPSV_ALG_DEFAULT:
            return rocsparse_spsv_alg_default;
        default:
            throw "Non existent hipsparseSpSVAlg_t";
        }
    }

    inline rocsparse_spsm_alg_ hipSpSMAlgToHCCSpSMAlg(hipsparseSpSMAlg_t alg)
    {
        switch(alg)
        {
        case HIPSPARSE_SPSM_ALG_DEFAULT:
            return rocsparse_spsm_alg_default;
        default:
            throw "Non existent hipsparseSpSMAlg_t";
        }
    }

    inline rocsparse_format_ hipFormatToHCCFormat(hipsparseFormat_t format)
    {
        switch(format)
        {
        case HIPSPARSE_FORMAT_CSR:
            return rocsparse_format_csr;
        case HIPSPARSE_FORMAT_CSC:
            return rocsparse_format_csc;
        case HIPSPARSE_FORMAT_COO:
            return rocsparse_format_coo;
        case HIPSPARSE_FORMAT_COO_AOS:
            return rocsparse_format_coo_aos;
        case HIPSPARSE_FORMAT_BLOCKED_ELL:
            return rocsparse_format_bell;
        default:
            throw "Non existent hipsparseFormat_t";
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
        default:
            throw "Non existent rocsparse_format";
        }
    }
}

struct hipsparseSpMVDescr_st
{
protected:
    rocsparse_spmv_descr m_spmv_descr{};
    bool                 m_is_stage_analysis_called{};
    bool                 m_is_implicit_stage_analysis_called{};
    size_t               m_buffer_size_stage_analysis{};
    size_t               m_buffer_size_stage_compute{};
    void*                m_buffer{};
    bool                 m_is_stage_compute_subsequent{};
    bool                 m_is_buffer_size_called{};

public:
    rocsparse_spmv_descr get_spmv_descr();
    void                 set_spmv_descr(rocsparse_spmv_descr value);

    bool is_stage_analysis_called() const;
    void stage_analysis_called();

    bool is_implicit_stage_analysis_called() const;
    void implicit_stage_analysis_called();

    bool is_stage_compute_subsequent() const;
    void stage_compute_subsequent();

    bool is_buffer_size_called() const;
    void buffer_size_called();

    size_t get_buffer_size_stage_analysis() const;
    void   set_buffer_size_stage_analysis(size_t value);

    size_t get_buffer_size_stage_compute() const;
    void   set_buffer_size_stage_compute(size_t value);

    void* get_buffer();
    void  set_buffer(void* value);

    void** get_buffer_reference();

    hipsparseSpMVDescr_st() = default;
    ~hipsparseSpMVDescr_st();
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
