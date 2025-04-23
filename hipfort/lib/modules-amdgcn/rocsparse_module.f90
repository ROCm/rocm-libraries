!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

module rocsparse
    use iso_c_binding

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
    end enum

! ===========================================================================
!   auxiliary SPARSE
! ===========================================================================

    interface

!       rocsparse_handle
        function rocsparse_create_handle(handle) &
                result(c_int) &
                bind(c, name = 'rocsparse_create_handle')
            use iso_c_binding
            implicit none
            type(c_ptr) :: handle
        end function rocsparse_create_handle

        function rocsparse_destroy_handle(handle) &
                result(c_int) &
                bind(c, name = 'rocsparse_destroy_handle')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function rocsparse_destroy_handle

!       rocsparse_stream
        function rocsparse_set_stream(handle, stream) &
                result(c_int) &
                bind(c, name = 'rocsparse_set_stream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function rocsparse_set_stream

        function rocsparse_get_stream(handle, stream) &
                result(c_int) &
                bind(c, name = 'rocsparse_get_stream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr) :: stream
        end function

!       rocsparse_pointer_mode
        function rocsparse_set_pointer_mode(handle, pointer_mode) &
                result(c_int) &
                bind(c, name = 'rocsparse_set_pointer_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: pointer_mode
        end function rocsparse_set_pointer_mode

        function rocsparse_get_pointer_mode(handle, pointer_mode) &
                result(c_int) &
                bind(c, name = 'rocsparse_get_pointer_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int) :: pointer_mode
        end function

!       rocsparse_version
        function rocsparse_get_version(handle, version) &
                result(c_int) &
                bind(c, name = 'rocsparse_get_version')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int) :: version
        end function rocsparse_get_version

        function rocsparse_get_git_rev(handle, rev) &
                result(c_int) &
                bind(c, name = 'rocsparse_get_git_rev')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            character(c_char) :: rev(*)
        end function rocsparse_get_git_rev

!       rocsparse_mat_descr
        function rocsparse_create_mat_descr(descr) &
                result(c_int) &
                bind(c, name = 'rocsparse_create_mat_descr')
            use iso_c_binding
            implicit none
            type(c_ptr) :: descr
        end function rocsparse_create_mat_descr

        function rocsparse_copy_mat_descr(dest, src) &
                result(c_int) &
                bind(c, name = 'rocsparse_copy_mat_descr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dest
            type(c_ptr), intent(in), value :: src
        end function rocsparse_copy_mat_descr

        function rocsparse_destroy_mat_descr(descr) &
                result(c_int) &
                bind(c, name = 'rocsparse_destroy_mat_descr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
        end function rocsparse_destroy_mat_descr

!       rocsparse_index_base
        function rocsparse_set_mat_index_base(descr, base) &
                result(c_int) &
                bind(c, name = 'rocsparse_set_mat_index_base')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
            integer(c_int), value :: base
        end function rocsparse_set_mat_index_base

        function rocsparse_get_mat_index_base(descr) &
                result(c_int) &
                bind(c, name = 'rocsparse_get_mat_index_base')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: descr
        end function rocsparse_get_mat_index_base

!       rocsparse_matrix_type
        function rocsparse_set_mat_type(descr, mat_type) &
                result(c_int) &
                bind(c, name = 'rocsparse_set_mat_type')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
            integer(c_int), value :: mat_type
        end function rocsparse_set_mat_type

        function rocsparse_get_mat_type(descr) &
                result(c_int) &
                bind(c, name = 'rocsparse_get_mat_type')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: descr
        end function rocsparse_get_mat_type

!       rocsparse_fill_mode
        function rocsparse_set_mat_fill_mode(descr, fill_mode) &
                result(c_int) &
                bind(c, name = 'rocsparse_set_mat_fill_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
            integer(c_int), value :: fill_mode
        end function rocsparse_set_mat_fill_mode

        function rocsparse_get_mat_fill_mode(descr) &
                result(c_int) &
                bind(c, name = 'rocsparse_get_mat_fill_mode')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: descr
        end function rocsparse_get_mat_fill_mode

!       rocsparse_diag_type
        function rocsparse_set_mat_diag_type(descr, diag_type) &
                result(c_int) &
                bind(c, name = 'rocsparse_set_mat_diag_type')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
            integer(c_int), value :: diag_type
        end function rocsparse_set_mat_diag_type

        function rocsparse_get_mat_diag_type(descr) &
                result(c_int) &
                bind(c, name = 'rocsparse_get_mat_diag_type')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: descr
        end function rocsparse_get_mat_diag_type

!       rocsparse_hyb_mat
        function rocsparse_create_hyb_mat(hyb) &
                result(c_int) &
                bind(c, name = 'rocsparse_create_hyb_mat')
            use iso_c_binding
            implicit none
            type(c_ptr) :: hyb
        end function rocsparse_create_hyb_mat

        function rocsparse_destroy_hyb_mat(hyb) &
                result(c_int) &
                bind(c, name = 'rocsparse_destroy_hyb_mat')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: hyb
        end function rocsparse_destroy_hyb_mat

!       rocsparse_mat_info
        function rocsparse_create_mat_info(info) &
                result(c_int) &
                bind(c, name = 'rocsparse_create_mat_info')
            use iso_c_binding
            implicit none
            type(c_ptr) :: info
        end function rocsparse_create_mat_info

        function rocsparse_destroy_mat_info(info) &
                result(c_int) &
                bind(c, name = 'rocsparse_destroy_mat_info')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: info
        end function rocsparse_destroy_mat_info

! ===========================================================================
!   level 1 SPARSE
! ===========================================================================

!       rocsparse_axpyi
        function rocsparse_saxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_saxpyi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_saxpyi

        function rocsparse_daxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_daxpyi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_daxpyi

        function rocsparse_caxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_caxpyi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_caxpyi

        function rocsparse_zaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_zaxpyi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_zaxpyi

!       rocsparse_doti
        function rocsparse_sdoti(handle, nnz, x_val, x_ind, y, result, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_sdoti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_sdoti

        function rocsparse_ddoti(handle, nnz, x_val, x_ind, y, result, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_ddoti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_ddoti

        function rocsparse_cdoti(handle, nnz, x_val, x_ind, y, result, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_cdoti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_cdoti

        function rocsparse_zdoti(handle, nnz, x_val, x_ind, y, result, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_zdoti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            complex(c_double_complex) :: result
            integer(c_int), value :: idx_base
        end function rocsparse_zdoti

!       rocsparse_dotci
        function rocsparse_cdotci(handle, nnz, x_val, x_ind, y, result, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_cdotci')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_cdotci

        function rocsparse_zdotci(handle, nnz, x_val, x_ind, y, result, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_zdotci')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_zdotci

!       rocsparse_gthr
        function rocsparse_sgthr(handle, nnz, y, x_val, x_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_sgthr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_sgthr

        function rocsparse_dgthr(handle, nnz, y, x_val, x_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_dgthr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_dgthr

        function rocsparse_cgthr(handle, nnz, y, x_val, x_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_cgthr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_cgthr

        function rocsparse_zgthr(handle, nnz, y, x_val, x_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_zgthr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_zgthr

!       rocsparse_gthrz
        function rocsparse_sgthrz(handle, nnz, y, x_val, x_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_sgthrz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_sgthrz

        function rocsparse_dgthrz(handle, nnz, y, x_val, x_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_dgthrz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_dgthrz

        function rocsparse_cgthrz(handle, nnz, y, x_val, x_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_cgthrz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_cgthrz

        function rocsparse_zgthrz(handle, nnz, y, x_val, x_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_zgthrz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_zgthrz

!       rocsparse_roti
        function rocsparse_sroti(handle, nnz, x_val, x_ind, y, c, s, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_sroti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            type(c_ptr), intent(in), value :: c
            type(c_ptr), intent(in), value :: s
            integer(c_int), value :: idx_base
        end function rocsparse_sroti

        function rocsparse_droti(handle, nnz, x_val, x_ind, y, c, s, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_droti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            type(c_ptr), intent(in), value :: c
            type(c_ptr), intent(in), value :: s
            integer(c_int), value :: idx_base
        end function rocsparse_droti

!       rocsparse_sctr
        function rocsparse_ssctr(handle, nnz, x_val, x_ind, y, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_ssctr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_ssctr

        function rocsparse_dsctr(handle, nnz, x_val, x_ind, y, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_dsctr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_dsctr

        function rocsparse_csctr(handle, nnz, x_val, x_ind, y, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_csctr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_csctr

        function rocsparse_zsctr(handle, nnz, x_val, x_ind, y, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_zsctr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_zsctr

! ===========================================================================
!   level 2 SPARSE
! ===========================================================================

!       rocsparse_bsrmv
        function rocsparse_sbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_sbsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_sbsrmv

        function rocsparse_dbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_dbsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_dbsrmv

        function rocsparse_cbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_cbsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_cbsrmv

        function rocsparse_zbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_zbsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_zbsrmv

!       rocsparse_coomv
        function rocsparse_scoomv(handle, trans, m, n, nnz, alpha, descr, coo_val, &
                coo_row_ind, coo_col_ind, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_scoomv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: coo_val
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_scoomv

        function rocsparse_dcoomv(handle, trans, m, n, nnz, alpha, descr, coo_val, &
                coo_row_ind, coo_col_ind, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcoomv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: coo_val
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_dcoomv

        function rocsparse_ccoomv(handle, trans, m, n, nnz, alpha, descr, coo_val, &
                coo_row_ind, coo_col_ind, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccoomv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: coo_val
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_ccoomv

        function rocsparse_zcoomv(handle, trans, m, n, nnz, alpha, descr, coo_val, &
                coo_row_ind, coo_col_ind, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcoomv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: coo_val
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_zcoomv

!       rocsparse_csrmv_analysis
        function rocsparse_scsrmv_analysis(handle, trans, m, n, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrmv_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
        end function rocsparse_scsrmv_analysis

        function rocsparse_dcsrmv_analysis(handle, trans, m, n, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrmv_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
        end function rocsparse_dcsrmv_analysis

        function rocsparse_ccsrmv_analysis(handle, trans, m, n, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrmv_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: handle
            integer(c_int), intent(in), value :: trans
            integer(c_int), intent(in), value :: m
            integer(c_int), intent(in), value :: n
            integer(c_int), intent(in), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
        end function rocsparse_ccsrmv_analysis

        function rocsparse_zcsrmv_analysis(handle, trans, m, n, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrmv_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
        end function rocsparse_zcsrmv_analysis

!       rocsparse_csrmv_clear
        function rocsparse_csrmv_clear(handle, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrmv_clear')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_csrmv_clear

!       rocsparse_csrmv
        function rocsparse_scsrmv(handle, trans, m, n, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_scsrmv

        function rocsparse_dcsrmv(handle, trans, m, n, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_dcsrmv

        function rocsparse_ccsrmv(handle, trans, m, n, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_ccsrmv

        function rocsparse_zcsrmv(handle, trans, m, n, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_zcsrmv

!       rocsparse_csrsv_zero_pivot
        function rocsparse_csrsv_zero_pivot(handle, descr, info, position) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrsv_zero_pivot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_csrsv_zero_pivot

!       rocsparse_csrsv_buffer_size
        function rocsparse_scsrsv_buffer_size(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrsv_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_scsrsv_buffer_size

        function rocsparse_dcsrsv_buffer_size(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrsv_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_dcsrsv_buffer_size

        function rocsparse_ccsrsv_buffer_size(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrsv_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_ccsrsv_buffer_size

        function rocsparse_zcsrsv_buffer_size(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrsv_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_zcsrsv_buffer_size

!       rocsparse_csrsv_analysis
        function rocsparse_scsrsv_analysis(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrsv_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsrsv_analysis

        function rocsparse_dcsrsv_analysis(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrsv_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsrsv_analysis

        function rocsparse_ccsrsv_analysis(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrsv_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsrsv_analysis

        function rocsparse_zcsrsv_analysis(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrsv_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsrsv_analysis

!       rocsparse_csrsv_clear
        function rocsparse_csrsv_clear(handle, descr, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrsv_clear')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: info
        end function rocsparse_csrsv_clear

!       rocsparse_csrsv_solve
        function rocsparse_scsrsv_solve(handle, trans, m, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, y, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrsv_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsrsv_solve

        function rocsparse_dcsrsv_solve(handle, trans, m, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, y, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrsv_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsrsv_solve

        function rocsparse_ccsrsv_solve(handle, trans, m, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, y, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrsv_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsrsv_solve

        function rocsparse_zcsrsv_solve(handle, trans, m, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, y, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrsv_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            compleX(c_double_complex), intent(in) :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsrsv_solve

!       rocsparse_ellmv
        function rocsparse_sellmv(handle, trans, m, n, alpha, descr, ell_val, &
                ell_col_ind, ell_width, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_sellmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: ell_val
            type(c_ptr), intent(in), value :: ell_col_ind
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_sellmv

        function rocsparse_dellmv(handle, trans, m, n, alpha, descr, ell_val, &
                ell_col_ind, ell_width, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_dellmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: ell_val
            type(c_ptr), intent(in), value :: ell_col_ind
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_dellmv

        function rocsparse_cellmv(handle, trans, m, n, alpha, descr, ell_val, &
                ell_col_ind, ell_width, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_cellmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: ell_val
            type(c_ptr), intent(in), value :: ell_col_ind
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_cellmv

        function rocsparse_zellmv(handle, trans, m, n, alpha, descr, ell_val, &
                ell_col_ind, ell_width, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_zellmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: ell_val
            type(c_ptr), intent(in), value :: ell_col_ind
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_zellmv

!       rocsparse_hybmv
        function rocsparse_shybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_shybmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_shybmv

        function rocsparse_dhybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_dhybmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_dhybmv

        function rocsparse_chybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_chybmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_chybmv

        function rocsparse_zhybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                result(c_int) &
                bind(c, name = 'rocsparse_zhybmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_zhybmv

! ===========================================================================
!   level 3 SPARSE
! ===========================================================================

!       rocsparse_csrmm
        function rocsparse_scsrmm(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, &
                csr_val, csr_row_ptr, csr_col_ind, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrmm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_scsrmm

        function rocsparse_dcsrmm(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, &
                csr_val, csr_row_ptr, csr_col_ind, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrmm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_dcsrmm

        function rocsparse_ccsrmm(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, &
                csr_val, csr_row_ptr, csr_col_ind, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrmm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_ccsrmm

        function rocsparse_zcsrmm(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, &
                csr_val, csr_row_ptr, csr_col_ind, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrmm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_zcsrmm

!       rocsparse_csrsm_zero_pivot
        function rocsparse_csrsm_zero_pivot(handle, info, position) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrsm_zero_pivot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_csrsm_zero_pivot

!       rocsparse_csrsm_buffer_size
        function rocsparse_scsrsm_buffer_size(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrsm_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer_size
        end function rocsparse_scsrsm_buffer_size

        function rocsparse_dcsrsm_buffer_size(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrsm_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer_size
        end function rocsparse_dcsrsm_buffer_size

        function rocsparse_ccsrsm_buffer_size(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrsm_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer_size
        end function rocsparse_ccsrsm_buffer_size

        function rocsparse_zcsrsm_buffer_size(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrsm_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer_size
        end function rocsparse_zcsrsm_buffer_size

!       rocsparse_csrsm_analysis
        function rocsparse_scsrsm_analysis(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, &
                analysis, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrsm_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsrsm_analysis

        function rocsparse_dcsrsm_analysis(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, &
                analysis, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrsm_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsrsm_analysis

        function rocsparse_ccsrsm_analysis(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, &
                analysis, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrsm_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsrsm_analysis

        function rocsparse_zcsrsm_analysis(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, &
                analysis, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrsm_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsrsm_analysis

!       rocsparse_csrsm_clear
        function rocsparse_csrsm_clear(handle, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrsm_clear')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_csrsm_clear

!       rocsparse_csrsm_solve
        function rocsparse_scsrsm_solve(handle, trans_A, trans_B, m, nrhs, nnz, alpha, &
                descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrsm_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsrsm_solve

        function rocsparse_dcsrsm_solve(handle, trans_A, trans_B, m, nrhs, nnz, alpha, &
                descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrsm_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsrsm_solve

        function rocsparse_ccsrsm_solve(handle, trans_A, trans_B, m, nrhs, nnz, alpha, &
                descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrsm_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsrsm_solve

        function rocsparse_zcsrsm_solve(handle, trans_A, trans_B, m, nrhs, nnz, alpha, &
                descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrsm_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsrsm_solve

! ===========================================================================
!   extra SPARSE
! ===========================================================================

!       rocsparse_csrgeam_nnz
        function rocsparse_csrgeam_nnz(handle, m, n, descr_A, nnz_A, csr_row_ptr_A, &
                csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, csr_col_ind_B, descr_C, &
                csr_row_ptr_C, nnz_C) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrgeam_nnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_row_ptr_C
            type(c_ptr), value :: nnz_C
        end function rocsparse_csrgeam_nnz

!       rocsparse_csrgeam
        function rocsparse_scsrgeam(handle, m, n, alpha, descr_A, nnz_A, csr_val_A, &
                csr_row_ptr_A, csr_col_ind_A, beta, descr_B, nnz_B, csr_val_B, &
                csr_row_ptr_B, csr_col_ind_B, descr_C, csr_val_C, csr_row_ptr_C, &
                csr_col_ind_C) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrgeam')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_val_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
        end function rocsparse_scsrgeam

        function rocsparse_dcsrgeam(handle, m, n, alpha, descr_A, nnz_A, csr_val_A, &
                csr_row_ptr_A, csr_col_ind_A, beta, descr_B, nnz_B, csr_val_B, &
                csr_row_ptr_B, csr_col_ind_B, descr_C, csr_val_C, csr_row_ptr_C, &
                csr_col_ind_C) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrgeam')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_val_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
        end function rocsparse_dcsrgeam

        function rocsparse_ccsrgeam(handle, m, n, alpha, descr_A, nnz_A, csr_val_A, &
                csr_row_ptr_A, csr_col_ind_A, beta, descr_B, nnz_B, csr_val_B, &
                csr_row_ptr_B, csr_col_ind_B, descr_C, csr_val_C, csr_row_ptr_C, &
                csr_col_ind_C) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrgeam')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_val_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
        end function rocsparse_ccsrgeam

        function rocsparse_zcsrgeam(handle, m, n, alpha, descr_A, nnz_A, csr_val_A, &
                csr_row_ptr_A, csr_col_ind_A, beta, descr_B, nnz_B, csr_val_B, &
                csr_row_ptr_B, csr_col_ind_B, descr_C, csr_val_C, csr_row_ptr_C, &
                csr_col_ind_C) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrgeam')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_val_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
        end function rocsparse_zcsrgeam

!       rocsparse_csrgemm_buffer_size
        function rocsparse_scsrgemm_buffer_size(handle, trans_A, trans_B, m, n, k, alpha, &
                descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, &
                csr_col_ind_B, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrgemm_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), value :: info_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_scsrgemm_buffer_size

        function rocsparse_dcsrgemm_buffer_size(handle, trans_A, trans_B, m, n, k, alpha, &
                descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, &
                csr_col_ind_B, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrgemm_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), value :: info_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_dcsrgemm_buffer_size

        function rocsparse_ccsrgemm_buffer_size(handle, trans_A, trans_B, m, n, k, alpha, &
                descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, &
                csr_col_ind_B, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrgemm_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), value :: info_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_ccsrgemm_buffer_size

        function rocsparse_zcsrgemm_buffer_size(handle, trans_A, trans_B, m, n, k, alpha, &
                descr_A, nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, &
                csr_col_ind_B, beta, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrgemm_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), value :: info_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_zcsrgemm_buffer_size

!       rocsparse_csrgemm_nnz
        function rocsparse_csrgemm_nnz(handle, trans_A, trans_B, m, n, k, descr_A, &
                nnz_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, &
                csr_col_ind_B, descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, descr_C, &
                csr_row_ptr_C, nnz_C, info_C, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrgemm_nnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_row_ptr_C
            type(c_ptr), value :: nnz_C
            type(c_ptr), intent(in), value :: info_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_csrgemm_nnz

!       rocsparse_csrgemm
        function rocsparse_scsrgemm(handle, trans_A, trans_B, m, n, k, alpha, descr_A, &
                nnz_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_val_B, &
                csr_row_ptr_B, csr_col_ind_B, beta, descr_D, nnz_D, csr_val_D, csr_row_ptr_D, &
                csr_col_ind_D, descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info_C, &
                temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrgemm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_val_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_val_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), intent(in), value :: info_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsrgemm

        function rocsparse_dcsrgemm(handle, trans_A, trans_B, m, n, k, alpha, descr_A, &
                nnz_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_val_B, &
                csr_row_ptr_B, csr_col_ind_B, beta, descr_D, nnz_D, csr_val_D, csr_row_ptr_D, &
                csr_col_ind_D, descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info_C, &
                temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrgemm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_val_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_val_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), intent(in), value :: info_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsrgemm

        function rocsparse_ccsrgemm(handle, trans_A, trans_B, m, n, k, alpha, descr_A, &
                nnz_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_val_B, &
                csr_row_ptr_B, csr_col_ind_B, beta, descr_D, nnz_D, csr_val_D, csr_row_ptr_D, &
                csr_col_ind_D, descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info_C, &
                temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrgemm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_val_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_val_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), intent(in), value :: info_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsrgemm

        function rocsparse_zcsrgemm(handle, trans_A, trans_B, m, n, k, alpha, descr_A, &
                nnz_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_val_B, &
                csr_row_ptr_B, csr_col_ind_B, beta, descr_D, nnz_D, csr_val_D, csr_row_ptr_D, &
                csr_col_ind_D, descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info_C, &
                temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrgemm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: descr_B
            integer(c_int), value :: nnz_B
            type(c_ptr), intent(in), value :: csr_val_B
            type(c_ptr), intent(in), value :: csr_row_ptr_B
            type(c_ptr), intent(in), value :: csr_col_ind_B
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descr_D
            integer(c_int), value :: nnz_D
            type(c_ptr), intent(in), value :: csr_val_D
            type(c_ptr), intent(in), value :: csr_row_ptr_D
            type(c_ptr), intent(in), value :: csr_col_ind_D
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), intent(in), value :: info_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsrgemm

! ===========================================================================
!   preconditioner SPARSE
! ===========================================================================

!       rocsparse_csric0_zero_pivot
        function rocsparse_csric0_zero_pivot(handle, info, position) &
                result(c_int) &
                bind(c, name = 'rocsparse_csric0_zero_pivot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_csric0_zero_pivot

!       rocsparse_csric0_buffer_size
        function rocsparse_scsric0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsric0_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_scsric0_buffer_size

        function rocsparse_dcsric0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsric0_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_dcsric0_buffer_size

        function rocsparse_ccsric0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsric0_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_ccsric0_buffer_size

        function rocsparse_zcsric0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsric0_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_zcsric0_buffer_size

!       rocsparse_csric0_analysis
        function rocsparse_scsric0_analysis(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsric0_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsric0_analysis

        function rocsparse_dcsric0_analysis(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsric0_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsric0_analysis

        function rocsparse_ccsric0_analysis(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsric0_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsric0_analysis

        function rocsparse_zcsric0_analysis(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsric0_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsric0_analysis

!       rocsparse_csric0_clear
        function rocsparse_csric0_clear(handle, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_csric0_clear')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_csric0_clear

!       rocsparse_csric0
        function rocsparse_scsric0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsric0')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsric0

        function rocsparse_dcsric0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsric0')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsric0

        function rocsparse_ccsric0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsric0')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsric0

        function rocsparse_zcsric0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsric0')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsric0

!       rocsparse_csrilu0_zero_pivot
        function rocsparse_csrilu0_zero_pivot(handle, info, position) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrilu0_zero_pivot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_csrilu0_zero_pivot

!       rocsparse_csrilu0_buffer_size
        function rocsparse_scsrilu0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrilu0_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_scsrilu0_buffer_size

        function rocsparse_dcsrilu0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrilu0_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_dcsrilu0_buffer_size

        function rocsparse_ccsrilu0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrilu0_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_ccsrilu0_buffer_size

        function rocsparse_zcsrilu0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrilu0_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_zcsrilu0_buffer_size

!       rocsparse_csrilu0_analysis
        function rocsparse_scsrilu0_analysis(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrilu0_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsrilu0_analysis

        function rocsparse_dcsrilu0_analysis(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrilu0_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsrilu0_analysis

        function rocsparse_ccsrilu0_analysis(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrilu0_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsrilu0_analysis

        function rocsparse_zcsrilu0_analysis(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, analysis, solve, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrilu0_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsrilu0_analysis

!       rocsparse_csrilu0_clear
        function rocsparse_csrilu0_clear(handle, info) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrilu0_clear')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_csrilu0_clear

!       rocsparse_csrilu0
        function rocsparse_scsrilu0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsrilu0')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsrilu0

        function rocsparse_dcsrilu0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsrilu0')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsrilu0

        function rocsparse_ccsrilu0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsrilu0')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsrilu0

        function rocsparse_zcsrilu0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsrilu0')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsrilu0

! ===========================================================================
!   conversion SPARSE
! ===========================================================================

!       rocsparse_nnz
        function rocsparse_snnz(handle, dir, m, n, descr, A, ld, nnz_per_row_columns, &
                nnz_total_dev_host_ptr) &
                result(c_int) &
                bind(c, name = 'rocsparse_snnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), value :: nnz_per_row_columns
            type(c_ptr), value :: nnz_total_dev_host_ptr
        end function rocsparse_snnz

        function rocsparse_dnnz(handle, dir, m, n, descr, A, ld, nnz_per_row_columns, &
                nnz_total_dev_host_ptr) &
                result(c_int) &
                bind(c, name = 'rocsparse_dnnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), value :: nnz_per_row_columns
            type(c_ptr), value :: nnz_total_dev_host_ptr
        end function rocsparse_dnnz

        function rocsparse_cnnz(handle, dir, m, n, descr, A, ld, nnz_per_row_columns, &
                nnz_total_dev_host_ptr) &
                result(c_int) &
                bind(c, name = 'rocsparse_cnnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), value :: nnz_per_row_columns
            type(c_ptr), value :: nnz_total_dev_host_ptr
        end function rocsparse_cnnz

        function rocsparse_znnz(handle, dir, m, n, descr, A, ld, nnz_per_row_columns, &
                nnz_total_dev_host_ptr) &
                result(c_int) &
                bind(c, name = 'rocsparse_znnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), value :: nnz_per_row_columns
            type(c_ptr), value :: nnz_total_dev_host_ptr
        end function rocsparse_znnz

!       rocsparse_dense2csr
        function rocsparse_sdense2csr(handle, m, n, descr, A, ld, nnz_per_rows, csr_val, &
                csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_sdense2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_rows
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_sdense2csr

        function rocsparse_ddense2csr(handle, m, n, descr, A, ld, nnz_per_rows, csr_val, &
                csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_ddense2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_rows
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_ddense2csr

        function rocsparse_cdense2csr(handle, m, n, descr, A, ld, nnz_per_rows, csr_val, &
                csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_cdense2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_rows
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_cdense2csr

        function rocsparse_zdense2csr(handle, m, n, descr, A, ld, nnz_per_rows, csr_val, &
                csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_zdense2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_rows
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_zdense2csr

!       rocsparse_dense2csc
        function rocsparse_sdense2csc(handle, m, n, descr, A, ld, nnz_per_columns, &
                csc_val, csc_col_ptr, csc_row_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_sdense2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_columns
            type(c_ptr), value :: csc_val
            type(c_ptr), value :: csc_col_ptr
            type(c_ptr), value :: csc_row_ind
        end function rocsparse_sdense2csc

        function rocsparse_ddense2csc(handle, m, n, descr, A, ld, nnz_per_columns, &
                csc_val, csc_col_ptr, csc_row_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_ddense2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_columns
            type(c_ptr), value :: csc_val
            type(c_ptr), value :: csc_col_ptr
            type(c_ptr), value :: csc_row_ind
        end function rocsparse_ddense2csc

        function rocsparse_cdense2csc(handle, m, n, descr, A, ld, nnz_per_columns, &
                csc_val, csc_col_ptr, csc_row_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_cdense2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_columns
            type(c_ptr), value :: csc_val
            type(c_ptr), value :: csc_col_ptr
            type(c_ptr), value :: csc_row_ind
        end function rocsparse_cdense2csc

        function rocsparse_zdense2csc(handle, m, n, descr, A, ld, nnz_per_columns, &
                csc_val, csc_col_ptr, csc_row_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_zdense2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_columns
            type(c_ptr), value :: csc_val
            type(c_ptr), value :: csc_col_ptr
            type(c_ptr), value :: csc_row_ind
        end function rocsparse_zdense2csc

!       rocsparse_csr2dense
        function rocsparse_scsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, A, ld) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsr2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_scsr2dense

        function rocsparse_dcsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, A, ld) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsr2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_dcsr2dense

        function rocsparse_ccsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, A, ld) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsr2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_ccsr2dense

        function rocsparse_zcsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, A, ld) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsr2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_zcsr2dense

!       rocsparse_csc2dense
        function rocsparse_scsc2dense(handle, m, n, descr, csc_val, csc_col_ptr, &
                csc_row_ind, A, ld) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsc2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csc_val
            type(c_ptr), intent(in), value :: csc_col_ptr
            type(c_ptr), intent(in), value :: csc_row_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_scsc2dense

        function rocsparse_dcsc2dense(handle, m, n, descr, csc_val, csc_col_ptr, &
                csc_row_ind, A, ld) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsc2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csc_val
            type(c_ptr), intent(in), value :: csc_col_ptr
            type(c_ptr), intent(in), value :: csc_row_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_dcsc2dense

        function rocsparse_ccsc2dense(handle, m, n, descr, csc_val, csc_col_ptr, &
                csc_row_ind, A, ld) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsc2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csc_val
            type(c_ptr), intent(in), value :: csc_col_ptr
            type(c_ptr), intent(in), value :: csc_row_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_ccsc2dense

        function rocsparse_zcsc2dense(handle, m, n, descr, csc_val, csc_col_ptr, &
                csc_row_ind, A, ld) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsc2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csc_val
            type(c_ptr), intent(in), value :: csc_col_ptr
            type(c_ptr), intent(in), value :: csc_row_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_zcsc2dense

!       rocsparse_nnz_compress
        function rocsparse_snnz_compress(handle, m, descr_A, csr_val_A, csr_row_ptr_A, &
                nnz_per_row, nnz_C, tol) &
                result(c_int) &
                bind(c, name = 'rocsparse_snnz_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), value :: nnz_per_row
            type(c_ptr), value :: nnz_C
            real(c_float), value :: tol
        end function rocsparse_snnz_compress

        function rocsparse_dnnz_compress(handle, m, descr_A, csr_val_A, csr_row_ptr_A, &
                nnz_per_row, nnz_C, tol) &
                result(c_int) &
                bind(c, name = 'rocsparse_dnnz_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), value :: nnz_per_row
            type(c_ptr), value :: nnz_C
            real(c_double), value :: tol
        end function rocsparse_dnnz_compress

        function rocsparse_cnnz_compress(handle, m, descr_A, csr_val_A, csr_row_ptr_A, &
                nnz_per_row, nnz_C, tol) &
                result(c_int) &
                bind(c, name = 'rocsparse_cnnz_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), value :: nnz_per_row
            type(c_ptr), value :: nnz_C
            complex(c_float_complex), value :: tol
        end function rocsparse_cnnz_compress

        function rocsparse_znnz_compress(handle, m, descr_A, csr_val_A, csr_row_ptr_A, &
                nnz_per_row, nnz_C, tol) &
                result(c_int) &
                bind(c, name = 'rocsparse_znnz_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), value :: nnz_per_row
            type(c_ptr), value :: nnz_C
            complex(c_double_complex), value :: tol
        end function rocsparse_znnz_compress

!       rocsparse_csr2coo
        function rocsparse_csr2coo(handle, csr_row_ptr, nnz, m, coo_row_ind, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_csr2coo')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: csr_row_ptr
            integer(c_int), value :: nnz
            integer(c_int), value :: m
            type(c_ptr), value :: coo_row_ind
            integer(c_int), value :: idx_base
        end function rocsparse_csr2coo

!       rocsparse_csr2csc_buffer_size
        function rocsparse_csr2csc_buffer_size(handle, m, n, nnz, csr_row_ptr, &
                csr_col_ind, copy_values, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_csr2csc_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: copy_values
            type(c_ptr), value :: buffer_size
        end function rocsparse_csr2csc_buffer_size

!       rocsparse_csr2csc
        function rocsparse_scsr2csc(handle, m, n, nnz, csr_val, csr_row_ptr, &
                csr_col_ind, csc_val, csc_row_ind, csc_col_ptr, copy_values, &
                idx_base, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsr2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: csc_val
            type(c_ptr), value :: csc_row_ind
            type(c_ptr), value :: csc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsr2csc

        function rocsparse_dcsr2csc(handle, m, n, nnz, csr_val, csr_row_ptr, &
                csr_col_ind, csc_val, csc_row_ind, csc_col_ptr, copy_values, &
                idx_base, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsr2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: csc_val
            type(c_ptr), value :: csc_row_ind
            type(c_ptr), value :: csc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsr2csc

        function rocsparse_ccsr2csc(handle, m, n, nnz, csr_val, csr_row_ptr, &
                csr_col_ind, csc_val, csc_row_ind, csc_col_ptr, copy_values, &
                idx_base, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsr2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: csc_val
            type(c_ptr), value :: csc_row_ind
            type(c_ptr), value :: csc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsr2csc

        function rocsparse_zcsr2csc(handle, m, n, nnz, csr_val, csr_row_ptr, &
                csr_col_ind, csc_val, csc_row_ind, csc_col_ptr, copy_values, &
                idx_base, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsr2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: csc_val
            type(c_ptr), value :: csc_row_ind
            type(c_ptr), value :: csc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsr2csc

!       rocsparse_csr2ell_width
        function rocsparse_csr2ell_width(handle, m, csr_descr, csr_row_ptr, &
                ell_descr, ell_width) &
                result(c_int) &
                bind(c, name = 'rocsparse_csr2ell_width')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: ell_descr
            type(c_ptr), value :: ell_width
        end function rocsparse_csr2ell_width

!       rocsparse_csr2ell
        function rocsparse_scsr2ell(handle, m, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, ell_descr, ell_width, ell_val, ell_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsr2ell')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), value :: ell_val
            type(c_ptr), value :: ell_col_ind
        end function rocsparse_scsr2ell

        function rocsparse_dcsr2ell(handle, m, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, ell_descr, ell_width, ell_val, ell_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsr2ell')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), value :: ell_val
            type(c_ptr), value :: ell_col_ind
        end function rocsparse_dcsr2ell

        function rocsparse_ccsr2ell(handle, m, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, ell_descr, ell_width, ell_val, ell_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsr2ell')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), value :: ell_val
            type(c_ptr), value :: ell_col_ind
        end function rocsparse_ccsr2ell

        function rocsparse_zcsr2ell(handle, m, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, ell_descr, ell_width, ell_val, ell_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsr2ell')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), value :: ell_val
            type(c_ptr), value :: ell_col_ind
        end function rocsparse_zcsr2ell

!       rocsparse_csr2hyb
        function rocsparse_scsr2hyb(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, hyb, user_ell_width, partition_type) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsr2hyb')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: hyb
            integer(c_int), value :: user_ell_width
            integer(c_int), value :: partition_type
        end function rocsparse_scsr2hyb

        function rocsparse_dcsr2hyb(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, hyb, user_ell_width, partition_type) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsr2hyb')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: hyb
            integer(c_int), value :: user_ell_width
            integer(c_int), value :: partition_type
        end function rocsparse_dcsr2hyb

        function rocsparse_ccsr2hyb(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, hyb, user_ell_width, partition_type) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsr2hyb')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: hyb
            integer(c_int), value :: user_ell_width
            integer(c_int), value :: partition_type
        end function rocsparse_ccsr2hyb

        function rocsparse_zcsr2hyb(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, hyb, user_ell_width, partition_type) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsr2hyb')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: hyb
            integer(c_int), value :: user_ell_width
            integer(c_int), value :: partition_type
        end function rocsparse_zcsr2hyb

!       rocsparse_csr2bsr_nnz
        function rocsparse_csr2bsr_nnz(handle, dir, m, n, csr_descr, csr_row_ptr, &
                csr_col_ind, block_dim, bsr_descr, bsr_row_ptr, bsr_nnz) &
                result(c_int) &
                bind(c, name = 'rocsparse_csr2bsr_nnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_nnz
        end function rocsparse_csr2bsr_nnz

!       rocsparse_csr2bsr
        function rocsparse_scsr2bsr(handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsr2bsr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_val
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_col_ind
        end function rocsparse_scsr2bsr

        function rocsparse_dcsr2bsr(handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsr2bsr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_val
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_col_ind
        end function rocsparse_dcsr2bsr

        function rocsparse_ccsr2bsr(handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsr2bsr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_val
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_col_ind
        end function rocsparse_ccsr2bsr

        function rocsparse_zcsr2bsr(handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, block_dim, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsr2bsr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_val
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_col_ind
        end function rocsparse_zcsr2bsr

!       rocsparse_csr2csr_compress
        function rocsparse_scsr2csr_compress(handle, m, n, descr_A, csr_val_A, &
                csr_col_ind_A, csr_row_ptr_A, nnz_A, nnz_per_row, csr_val_C, &
                csr_col_ind_C, csr_row_ptr_C, tol) &
                result(c_int) &
                bind(c, name = 'rocsparse_scsr2csr_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: nnz_per_row
            type(c_ptr), value :: csr_val_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), value :: csr_row_ptr_C
            real(c_float), value :: tol
        end function rocsparse_scsr2csr_compress

        function rocsparse_dcsr2csr_compress(handle, m, n, descr_A, csr_val_A, &
                csr_col_ind_A, csr_row_ptr_A, nnz_A, nnz_per_row, csr_val_C, &
                csr_col_ind_C, csr_row_ptr_C, tol) &
                result(c_int) &
                bind(c, name = 'rocsparse_dcsr2csr_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: nnz_per_row
            type(c_ptr), value :: csr_val_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), value :: csr_row_ptr_C
            real(c_float), value :: tol
        end function rocsparse_dcsr2csr_compress

        function rocsparse_ccsr2csr_compress(handle, m, n, descr_A, csr_val_A, &
                csr_col_ind_A, csr_row_ptr_A, nnz_A, nnz_per_row, csr_val_C, &
                csr_col_ind_C, csr_row_ptr_C, tol) &
                result(c_int) &
                bind(c, name = 'rocsparse_ccsr2csr_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: nnz_per_row
            type(c_ptr), value :: csr_val_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), value :: csr_row_ptr_C
            real(c_float), value :: tol
        end function rocsparse_ccsr2csr_compress

        function rocsparse_zcsr2csr_compress(handle, m, n, descr_A, csr_val_A, &
                csr_col_ind_A, csr_row_ptr_A, nnz_A, nnz_per_row, csr_val_C, &
                csr_col_ind_C, csr_row_ptr_C, tol) &
                result(c_int) &
                bind(c, name = 'rocsparse_zcsr2csr_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: nnz_per_row
            type(c_ptr), value :: csr_val_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), value :: csr_row_ptr_C
            real(c_float), value :: tol
        end function rocsparse_zcsr2csr_compress

!       rocsparse_coo2csr
        function rocsparse_coo2csr(handle, coo_row_ind, nnz, m, csr_row_ptr, idx_base) &
                result(c_int) &
                bind(c, name = 'rocsparse_coo2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: coo_row_ind
            integer(c_int), value :: nnz
            integer(c_int), value :: m
            type(c_ptr), value :: csr_row_ptr
            integer(c_int), value :: idx_base
        end function rocsparse_coo2csr

!       rocsparse_ell2csr_nnz
        function rocsparse_ell2csr_nnz(handle, m, n, ell_descr, ell_width, ell_col_ind, &
                csr_descr, csr_row_ptr, csr_nnz) &
                result(c_int) &
                bind(c, name = 'rocsparse_ell2csr_nnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: ell_col_ind
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_nnz
        end function rocsparse_ell2csr_nnz

!       rocsparse_ell2csr
        function rocsparse_sell2csr(handle, m, n, ell_descr, ell_width, ell_val, &
                ell_col_ind, csr_descr, csr_val, csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_sell2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: ell_val
            type(c_ptr), intent(in), value :: ell_col_ind
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_sell2csr

        function rocsparse_dell2csr(handle, m, n, ell_descr, ell_width, ell_val, &
                ell_col_ind, csr_descr, csr_val, csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_dell2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: ell_val
            type(c_ptr), intent(in), value :: ell_col_ind
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_dell2csr

        function rocsparse_cell2csr(handle, m, n, ell_descr, ell_width, ell_val, &
                ell_col_ind, csr_descr, csr_val, csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_cell2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: ell_val
            type(c_ptr), intent(in), value :: ell_col_ind
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_cell2csr

        function rocsparse_zell2csr(handle, m, n, ell_descr, ell_width, ell_val, &
                ell_col_ind, csr_descr, csr_val, csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_zell2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: ell_descr
            integer(c_int), value :: ell_width
            type(c_ptr), intent(in), value :: ell_val
            type(c_ptr), intent(in), value :: ell_col_ind
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_zell2csr

!       rocsparse_hyb2csr_buffer_size
        function rocsparse_hyb2csr_buffer_size(handle, descr, hyb, csr_row_ptr, &
                buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_hyb2csr_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: buffer_size
        end function rocsparse_hyb2csr_buffer_size

!       rocsparse_hyb2csr
        function rocsparse_shyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, &
                csr_col_ind, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_shyb2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: temp_buffer
        end function rocsparse_shyb2csr

        function rocsparse_dhyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, &
                csr_col_ind, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_dhyb2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dhyb2csr

        function rocsparse_chyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, &
                csr_col_ind, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_chyb2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: temp_buffer
        end function rocsparse_chyb2csr

        function rocsparse_zhyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, &
                csr_col_ind, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_zhyb2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zhyb2csr

!       rocsparse_create_identity_permutation
        function rocsparse_create_identity_permutation(handle, n, p) &
                result(c_int) &
                bind(c, name = 'rocsparse_create_identity_permutation')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: p
        end function rocsparse_create_identity_permutation

!       rocsparse_csrsort_buffer_size
        function rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr, &
                csr_col_ind, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrsort_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: buffer_size
        end function rocsparse_csrsort_buffer_size

!       rocsparse_csrsort
        function rocsparse_csrsort(handle, m, n, nnz, csr_row_ptr, &
                csr_col_ind, perm, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_csrsort')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: perm
            type(c_ptr), value :: temp_buffer
        end function rocsparse_csrsort

!       rocsparse_cscsort_buffer_size
        function rocsparse_cscsort_buffer_size(handle, m, n, nnz, csc_col_ptr, &
                csc_row_ind, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_cscsort_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csc_col_ptr
            type(c_ptr), intent(in), value :: csc_row_ind
            type(c_ptr), value :: buffer_size
        end function rocsparse_cscsort_buffer_size

!       rocsparse_cscsort
        function rocsparse_cscsort(handle, m, n, nnz, csc_col_ptr, &
                csc_row_ind, perm, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_cscsort')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csc_col_ptr
            type(c_ptr), value :: csc_row_ind
            type(c_ptr), value :: perm
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cscsort

!       rocsparse_coosort_buffer_size
        function rocsparse_coosort_buffer_size(handle, m, n, nnz, coo_row_ind, &
                coo_col_ind, buffer_size) &
                result(c_int) &
                bind(c, name = 'rocsparse_coosort_buffer_size')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), value :: buffer_size
        end function rocsparse_coosort_buffer_size

!       rocsparse_coosort_by_row
        function rocsparse_coosort_by_row(handle, m, n, nnz, coo_row_ind, &
                coo_col_ind, perm, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_coosort_by_row')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: coo_row_ind
            type(c_ptr), value :: coo_col_ind
            type(c_ptr), value :: perm
            type(c_ptr), value :: temp_buffer
        end function rocsparse_coosort_by_row

!       rocsparse_coosort_by_column
        function rocsparse_coosort_by_column(handle, m, n, nnz, coo_row_ind, &
                coo_col_ind, perm, temp_buffer) &
                result(c_int) &
                bind(c, name = 'rocsparse_coosort_by_column')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: coo_row_ind
            type(c_ptr), value :: coo_col_ind
            type(c_ptr), value :: perm
            type(c_ptr), value :: temp_buffer
        end function rocsparse_coosort_by_column

!       rocsparse_bsr2csr
        function rocsparse_sbsr2csr(handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, csr_descr, csr_val, csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_sbsr2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_sbsr2csr

        function rocsparse_dbsr2csr(handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, csr_descr, csr_val, csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_dbsr2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_dbsr2csr

        function rocsparse_cbsr2csr(handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, csr_descr, csr_val, csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_cbsr2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_cbsr2csr

        function rocsparse_zbsr2csr(handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, csr_descr, csr_val, csr_row_ptr, csr_col_ind) &
                result(c_int) &
                bind(c, name = 'rocsparse_zbsr2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_zbsr2csr

    end interface

    contains
        subroutine rocsparseCheck(rocsparseError_t)
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparseError_t
            if (rocsparseError_t /= rocsparse_status_success) then
                write(*,*) "ROCSPARSE ERROR: Error code = ", rocsparseError_t
                call exit(rocsparseError_t)
            end if
        end subroutine rocsparseCheck

end module rocsparse