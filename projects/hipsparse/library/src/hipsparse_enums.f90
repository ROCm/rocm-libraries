!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020 Advanced Micro Devices, Inc. All rights Reserved.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module hipsparse_enums
    implicit none

! ===========================================================================
!   types SPARSE
! ===========================================================================

!   hipsparseStatus_t
    enum, bind(c)
        enumerator :: HIPSPARSE_STATUS_SUCCESS = 0
        enumerator :: HIPSPARSE_STATUS_NOT_INITIALIZED = 1
        enumerator :: HIPSPARSE_STATUS_ALLOC_FAILED = 2
        enumerator :: HIPSPARSE_STATUS_INVALID_VALUE = 3
        enumerator :: HIPSPARSE_STATUS_ARCH_MISMATCH = 4
        enumerator :: HIPSPARSE_STATUS_MAPPING_ERROR = 5
        enumerator :: HIPSPARSE_STATUS_EXECUTION_FAILED = 6
        enumerator :: HIPSPARSE_STATUS_INTERNAL_ERROR = 7
        enumerator :: HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
        enumerator :: HIPSPARSE_STATUS_ZERO_PIVOT = 9
    end enum

!   hipsparsePointerMode_t
    enum, bind(c)
        enumerator :: HIPSPARSE_POINTER_MODE_HOST = 0
        enumerator :: HIPSPARSE_POINTER_MODE_DEVICE = 1
    end enum

!   hipsparseAction_t
    enum, bind(c)
        enumerator :: HIPSPARSE_ACTION_SYMBOLIC = 0
        enumerator :: HIPSPARSE_ACTION_NUMERIC = 1
    end enum

!   hipsparseMatrixType_t
    enum, bind(c)
        enumerator :: HIPSPARSE_MATRIX_TYPE_GENERAL = 0
        enumerator :: HIPSPARSE_MATRIX_TYPE_SYMMETRIC = 1
        enumerator :: HIPSPARSE_MATRIX_TYPE_HERMITIAN = 2
        enumerator :: HIPSPARSE_MATRIX_TYPE_TRIANGULAR = 3
    end enum

!   hipsparseFillMode_t
    enum, bind(c)
        enumerator :: HIPSPARSE_FILL_MODE_LOWER = 0
        enumerator :: HIPSPARSE_FILL_MODE_UPPER = 1
    end enum

!   hipsparseDiagType_t
    enum, bind(c)
        enumerator :: HIPSPARSE_DIAG_TYPE_NON_UNIT = 0
        enumerator :: HIPSPARSE_DIAG_TYPE_UNIT = 1
    end enum

!   hipsparseIndexBase_t
    enum, bind(c)
        enumerator :: HIPSPARSE_INDEX_BASE_ZERO = 0
        enumerator :: HIPSPARSE_INDEX_BASE_ONE = 1
    end enum

!   hipsparseOperation_t
    enum, bind(c)
        enumerator :: HIPSPARSE_OPERATION_NON_TRANSPOSE = 0
        enumerator :: HIPSPARSE_OPERATION_TRANSPOSE = 1
        enumerator :: HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
    end enum

!   hipsparseHybPartition_t
    enum, bind(c)
        enumerator :: HIPSPARSE_HYB_PARTITION_AUTO = 0
        enumerator :: HIPSPARSE_HYB_PARTITION_USER = 1
        enumerator :: HIPSPARSE_HYB_PARTITION_MAX = 2
    end enum

!   hipsparseSolvePolicy_t
    enum, bind(c)
        enumerator :: HIPSPARSE_SOLVE_POLICY_NO_LEVEL = 0
        enumerator :: HIPSPARSE_SOLVE_POLICY_USE_LEVEL = 1
    end enum

!   hipsparseSideMode_t
    enum, bind(c)
        enumerator :: HIPSPARSE_SIDE_LEFT = 0
        enumerator :: HIPSPARSE_SIDE_RIGHT = 1
    end enum

!   hipsparseDirection_t
    enum, bind(c)
        enumerator :: HIPSPARSE_DIRECTION_ROW = 0
        enumerator :: HIPSPARSE_DIRECTION_COLUMN = 1
    end enum

end module hipsparse_enums
