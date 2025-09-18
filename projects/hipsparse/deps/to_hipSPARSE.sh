# ########################################################################
# Copyright (C) 2018 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

#!/bin/bash

echo Replacing rocSPARSE function calls in \"$1\" ...

# Int
sed -i 's/rocsparse_int/int/g' $1

# Status verification
sed -i 's/verify_rocsparse_status_invalid_pointer/verify_hipsparse_status_invalid_pointer/g' $1
sed -i 's/verify_rocsparse_status_invalid_size/verify_hipsparse_status_invalid_size/g' $1
sed -i 's/verify_rocsparse_status_invalid_value/verify_hipsparse_status_invalid_value/g' $1
sed -i 's/verify_rocsparse_status_invalid_handle/verify_hipsparse_status_invalid_handle/g' $1
sed -i 's/verify_rocsparse_status_zero_pivot/verify_hipsparse_status_zero_pivot/g' $1
sed -i 's/verify_rocsparse_status_success/verify_hipsparse_status_success/g' $1

# Handle
sed -i 's/rocsparse_handle/hipsparseHandle_t/g' $1
sed -i 's/rocsparse_create_handle/hipsparseCreate/g' $1
sed -i 's/rocsparse_destroy_handle/hipsparseDestroy/g' $1
sed -i 's/rocsparse_get_version/hipsparseGetVersion/g' $1

# Stream
sed -i 's/rocsparse_set_stream/hipsparseSetStream/g' $1
sed -i 's/rocsparse_get_stream/hipsparseGetStream/g' $1

# Mat descr
sed -i 's/rocsparse_mat_descr/hipsparseMatDescr_t/g' $1
sed -i 's/rocsparse_create_mat_descr/hipsparseCreateMatDescr/g' $1
sed -i 's/rocsparse_destroy_mat_descr/hipsparseDestroyMatDescr/g' $1
sed -i 's/rocsparse_set_mat_index_base/hipsparseSetMatIndexBase/g' $1
sed -i 's/rocsparse_get_mat_index_base/hipsparseGetMatIndexBase/g' $1
sed -i 's/rocsparse_set_mat_fill_mode/hipsparseSetMatFillMode/g' $1
sed -i 's/rocsparse_get_mat_fill_mode/hipsparseGetMatFillMode/g' $1
sed -i 's/rocsparse_set_mat_diag_type/hipsparseSetMatDiagType/g' $1
sed -i 's/rocsparse_get_mat_diag_type/hipsparseGetMatDiagType/g' $1

# Operation
sed -i 's/rocsparse_operation_none/HIPSPARSE_OPERATION_NON_TRANSPOSE/g' $1
sed -i 's/rocsparse_operation_transpose/HIPSPARSE_OPERATION_TRANSPOSE/g' $1
sed -i 's/rocsparse_operation_conjugate_transpose/HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE/g' $1
sed -i 's/rocsparse_operation/hipsparseOperation_t/g' $1

# Index base
sed -i 's/rocsparse_index_base_zero/HIPSPARSE_INDEX_BASE_ZERO/g' $1
sed -i 's/rocsparse_index_base_one/HIPSPARSE_INDEX_BASE_ONE/g' $1
sed -i 's/rocsparse_index_base/hipsparseIndexBase_t/g' $1

# Matrix type
sed -i 's/rocsparse_matrix_type_general/HIPSPARSE_MATRIX_TYPE_GENERAL/g' $1
sed -i 's/rocsparse_matrix_type_symmetric/HIPSPARSE_MATRIX_TYPE_SYMMETRIC/g' $1
sed -i 's/rocsparse_matrix_type_hermitian/HIPSPARSE_MATRIX_TYPE_HERMITIAN/g' $1
sed -i 's/rocsparse_matrix_type_triangular/HIPSPARSE_MATRIX_TYPE_TRIANGULAR/g' $1
sed -i 's/rocsparse_matrix_type/hipsparseMatrixType_t/g' $1

# Fill mode
sed -i 's/rocsparse_fill_mode_lower/HIPSPARSE_FILL_MODE_LOWER/g' $1
sed -i 's/rocsparse_fill_mode_upper/HIPSPARSE_FILL_MODE_UPPER/g' $1
sed -i 's/rocsparse_fill_mode/hipsparseFillMode_t/g' $1

# Diag type
sed -i 's/rocsparse_diag_type_unit/HIPSPARSE_DIAG_TYPE_UNIT/g' $1
sed -i 's/rocsparse_diag_type_non_unit/HIPSPARSE_DIAG_TYPE_NON_UNIT/g' $1
sed -i 's/rocsparse_diag_type/hipsparseDiagType_t/g' $1

# Action
sed -i 's/rocsparse_action_symbolic/HIPSPARSE_ACTION_SYMBOLIC/g' $1
sed -i 's/rocsparse_action_numeric/HIPSPARSE_ACTION_NUMERIC/g' $1
sed -i 's/rocsparse_action/hipsparseAction_t/g' $1

# Hyb
sed -i 's/rocsparse_hyb_partition_auto/HIPSPARSE_HYB_PARTITION_AUTO/g' $1
sed -i 's/rocsparse_hyb_partition_user/HIPSPARSE_HYB_PARTITION_USER/g' $1
sed -i 's/rocsparse_hyb_partition_max/HIPSPARSE_HYB_PARTITION_MAX/g' $1
sed -i 's/rocsparse_hyb_partition/hipsparseHybPartition_t/g' $1
sed -i 's/rocsparse_hyb_mat/hipsparseHybMat_t/g' $1
sed -i 's/rocsparse_create_hyb_mat/hipsparseCreateHybMat/g' $1
sed -i 's/rocsparse_destroy_hyb_mat/hipsparseDestroyHybMat/g' $1

# Status
sed -i 's/rocsparse_status_success/HIPSPARSE_STATUS_SUCCESS/g' $1
sed -i 's/rocsparse_status_invalid_handle/HIPSPARSE_STATUS_NOT_INITIALIZED/g' $1
sed -i 's/rocsparse_status_not_implemented/HIPSPARSE_STATUS_NOT_SUPPORTED/g' $1
sed -i 's/rocsparse_status_invalid_pointer/HIPSPARSE_STATUS_INVALID_VALUE/g' $1
sed -i 's/rocsparse_status_invalid_size/HIPSPARSE_STATUS_INVALID_VALUE/g' $1
sed -i 's/rocsparse_status_memory_error/HIPSPARSE_STATUS_ALLOC_FAILED/g' $1
sed -i 's/rocsparse_status_internal_error/HIPSPARSE_STATUS_INTERNAL_ERROR/g' $1
sed -i 's/rocsparse_status_invalid_value/HIPSPARSE_STATUS_INVALID_VALUE/g' $1
sed -i 's/rocsparse_status_arch_mismatch/HIPSPARSE_STATUS_ARCH_MISMATCH/g' $1
sed -i 's/rocsparse_status_zero_pivot/HIPSPARSE_STATUS_ZERO_PIVOT/g' $1
sed -i 's/rocsparse_status/hipsparseStatus_t/g' $1

# Pointer mode
sed -i 's/rocsparse_pointer_mode_host/HIPSPARSE_POINTER_MODE_HOST/g' $1
sed -i 's/rocsparse_pointer_mode_device/HIPSPARSE_POINTER_MODE_DEVICE/g' $1
sed -i 's/rocsparse_pointer_mode/hipsparsePointerMode_t/g' $1
sed -i 's/rocsparse_set_pointer_mode/hipsparseSetPointerMode/g' $1
sed -i 's/rocsparse_get_pointer_mode/hipsparseGetPointerMode/g' $1

# Level 1
sed -i 's/rocsparse_saxpyi/hipsparseSaxpyi/g' $1
sed -i 's/rocsparse_daxpyi/hipsparseDaxpyi/g' $1
sed -i 's/rocsparse_axpyi/hipsparseXaxpyi/g' $1
sed -i 's/rocsparse_sdoti/hipsparseSdoti/g' $1
sed -i 's/rocsparse_ddoti/hipsparseDdoti/g' $1
sed -i 's/rocsparse_doti/hipsparseXdoti/g' $1
sed -i 's/rocsparse_sgthr/hipsparseSgthr/g' $1
sed -i 's/rocsparse_dgthr/hipsparseDgthr/g' $1
sed -i 's/rocsparse_gthr/hipsparseXgthr/g' $1
sed -i 's/rocsparse_sgthrz/hipsparseSgthrz/g' $1
sed -i 's/rocsparse_dgthrz/hipsparseDgthrz/g' $1
sed -i 's/rocsparse_gthrz/hipsparseXgthrz/g' $1
sed -i 's/rocsparse_sroti/hipsparseSroti/g' $1
sed -i 's/rocsparse_droti/hipsparseDroti/g' $1
sed -i 's/rocsparse_roti/hipsparseXroti/g' $1
sed -i 's/rocsparse_ssctr/hipsparseSsctr/g' $1
sed -i 's/rocsparse_dsctr/hipsparseDsctr/g' $1
sed -i 's/rocsparse_sctr/hipsparseXsctr/g' $1

# Level 2
sed -i 's/rocsparse_scoomv/hipsparseScoomv/g' $1
sed -i 's/rocsparse_dcoomv/hipsparseDcoomv/g' $1
sed -i 's/rocsparse_coomv/hipsparseXcoomv/g' $1
sed -i 's/rocsparse_scsrmv/hipsparseScsrmv/g' $1
sed -i 's/rocsparse_dcsrmv/hipsparseDcsrmv/g' $1
sed -i 's/rocsparse_csrmv/hipsparseXcsrmv/g' $1
sed -i 's/rocsparse_sellmv/hipsparseSellmv/g' $1
sed -i 's/rocsparse_dellmv/hipsparseDellmv/g' $1
sed -i 's/rocsparse_ellmv/hipsparseXellmv/g' $1
sed -i 's/rocsparse_shybmv/hipsparseShybmv/g' $1
sed -i 's/rocsparse_dhybmv/hipsparseDhybmv/g' $1
sed -i 's/rocsparse_hybmv/hipsparseXhybmv/g' $1

# Level 3
sed -i 's/rocsparse_sbsrmm/hipsparseSbsrmm/g' $1
sed -i 's/rocsparse_dbsrmm/hipsparseDbsrmm/g' $1
sed -i 's/rocsparse_bsrmm/hipsparseXbsrmm/g' $1
sed -i 's/rocsparse_scsrmm/hipsparseScsrmm2/g' $1
sed -i 's/rocsparse_dcsrmm/hipsparseDcsrmm2/g' $1
sed -i 's/rocsparse_csrmm/hipsparseXcsrmm2/g' $1

# Conversion
sed -i 's/rocsparse_csr2coo/hipsparseXcsr2coo/g' $1
sed -i 's/rocsparse_scsr2csc/hipsparseScsr2csc/g' $1
sed -i 's/rocsparse_dcsr2csc/hipsparseDcsr2csc/g' $1
sed -i 's/rocsparse_csr2csc/hipsparseXcsr2csc/g' $1
sed -i 's/rocsparse_csr2ell_width/hipsparseXcsr2ellWidth/g' $1
sed -i 's/rocsparse_scsr2ell/hipsparseScsr2ell/g' $1
sed -i 's/rocsparse_dcsr2ell/hipsparseDcsr2ell/g' $1
sed -i 's/rocsparse_csr2ell/hipsparseXcsr2ell/g' $1
sed -i 's/rocsparse_scsr2hyb/hipsparseScsr2hyb/g' $1
sed -i 's/rocsparse_dcsr2hyb/hipsparseDcsr2hyb/g' $1
sed -i 's/rocsparse_csr2hyb/hipsparseXcsr2hyb/g' $1
sed -i 's/rocsparse_coo2csr/hipsparseXcoo2csr/g' $1
sed -i 's/rocsparse_create_identity_permutation/hipsparseCreateIdentityPermutation/g' $1
sed -i 's/rocsparse_csrsort_buffer_size/hipsparseXcsrsort_bufferSizeExt/g' $1
sed -i 's/rocsparse_csrsort/hipsparseXcsrsort/g' $1
sed -i 's/rocsparse_coosort_buffer_size/hipsparseXcoosort_bufferSizeExt/g' $1
sed -i 's/rocsparse_coosort_by_row/hipsparseXcoosortByRow/g' $1
sed -i 's/rocsparse_coosort_by_column/hipsparseXcoosortByColumn/g' $1
sed -i 's/rocsparse_ell2csr_nnz/hipsparseXell2csrNnz/g' $1
sed -i 's/rocsparse_sell2csr/hipsparseSell2csr/g' $1
sed -i 's/rocsparse_dell2csr/hipsparseDell2csr/g' $1
sed -i 's/rocsparse_ell2csr/hipsparseXell2csr/g' $1

# Header
sed -i 's/rocsparse.h/hipsparse.h/g' $1
sed -i 's/rocsparse_test_unique_ptr.hpp/hipsparse_test_unique_ptr.hpp/g' $1

# Namespace
sed -i 's/namespace rocsparse/namespace hipsparse/g' $1
sed -i 's/namespace rocsparse_test/namespace hipsparse_test/g' $1

# Unique ptr
sed -i 's/rocsparse_unique_ptr/hipsparse_unique_ptr/g' $1

# Error macro
sed -i 's/CHECK_ROCSPARSE_ERROR/CHECK_HIPSPARSE_ERROR/g' $1

# Utilities
sed -i 's/rocsparse_init_index/hipsparseInitIndex/g' $1
sed -i 's/rocsparse_init_csr/hipsparseInitCSR/g' $1
sed -i 's/rocsparse_init/hipsparseInit/g' $1
