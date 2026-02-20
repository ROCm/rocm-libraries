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

module rocsparse_enums
    implicit none

! ===========================================================================
!   types SPARSE
! ===========================================================================

!   rocsparse_operation
    enum, bind(c)
        enumerator :: rocsparse_operation_none = 111
        enumerator :: rocsparse_operation_transpose = 112
        enumerator :: rocsparse_operation_conjugate_transpose = 113
    end enum

!   rocsparse_index_base
    enum, bind(c)
        enumerator :: rocsparse_index_base_zero = 0
        enumerator :: rocsparse_index_base_one = 1
    end enum

!   rocsparse_matrix_type
    enum, bind(c)
        enumerator :: rocsparse_matrix_type_general = 0
        enumerator :: rocsparse_matrix_type_symmetric = 1
        enumerator :: rocsparse_matrix_type_hermitian = 2
        enumerator :: rocsparse_matrix_type_triangular = 3
    end enum

!   rocsparse_diag_type
    enum, bind(c)
        enumerator :: rocsparse_diag_type_non_unit = 0
        enumerator :: rocsparse_diag_type_unit = 1
    end enum

!   rocsparse_fill_mode
    enum, bind(c)
        enumerator :: rocsparse_fill_mode_lower = 0
        enumerator :: rocsparse_fill_mode_upper = 1
    end enum

!   rocsparse_storage_mode
    enum, bind(c)
        enumerator :: rocsparse_storage_mode_sorted = 0
        enumerator :: rocsparse_storage_mode_unsorted = 1
    end enum

!   rocsparse_action
    enum, bind(c)
        enumerator :: rocsparse_action_symbolic = 0
        enumerator :: rocsparse_action_numeric = 1
    end enum

!   rocsparse_direction
    enum, bind(c)
        enumerator :: rocsparse_direction_row = 0
        enumerator :: rocsparse_direction_column = 1
    end enum

!   rocsparse_hyb_partition
    enum, bind(c)
        enumerator :: rocsparse_hyb_partition_auto = 0
        enumerator :: rocsparse_hyb_partition_user = 1
        enumerator :: rocsparse_hyb_partition_max = 2
    end enum

!   rocsparse_analysis_policy
    enum, bind(c)
        enumerator :: rocsparse_analysis_policy_reuse = 0
        enumerator :: rocsparse_analysis_policy_force = 1
    end enum

!   rocsparse_solve_policy
    enum, bind(c)
        enumerator :: rocsparse_solve_policy_auto = 0
    end enum

!   rocsparse_pointer_mode
    enum, bind(c)
        enumerator :: rocsparse_pointer_mode_host = 0
        enumerator :: rocsparse_pointer_mode_device = 1
    end enum

!   rocsparse_layer_mode
    enum, bind(c)
        enumerator :: rocsparse_layer_mode_none = 0
        enumerator :: rocsparse_layer_mode_log_trace = 1
        enumerator :: rocsparse_layer_mode_log_bench = 2
        enumerator :: rocsparse_layer_mode_log_debug = 4
    end enum

!   rocsparse_status
    enum, bind(c)
        enumerator :: rocsparse_status_success = 0
        enumerator :: rocsparse_status_invalid_handle = 1
        enumerator :: rocsparse_status_not_implemented = 2
        enumerator :: rocsparse_status_invalid_pointer = 3
        enumerator :: rocsparse_status_invalid_size = 4
        enumerator :: rocsparse_status_memory_error = 5
        enumerator :: rocsparse_status_internal_error = 6
        enumerator :: rocsparse_status_invalid_value = 7
        enumerator :: rocsparse_status_arch_mismatch = 8
        enumerator :: rocsparse_status_zero_pivot = 9
        enumerator :: rocsparse_status_not_initialized = 10
        enumerator :: rocsparse_status_type_mismatch = 11
        enumerator :: rocsparse_status_requires_sorted_storage = 12
    end enum

end module rocsparse_enums
