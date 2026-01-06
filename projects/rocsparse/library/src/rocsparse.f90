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

module rocsparse
    use rocsparse_enums
    use iso_c_binding
    implicit none

! ===========================================================================
!   auxiliary SPARSE
! ===========================================================================

    interface

!       rocsparse_handle
        function rocsparse_create_handle(handle) &
                bind(c, name = 'rocsparse_create_handle')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_create_handle
            type(c_ptr) :: handle
        end function rocsparse_create_handle

        function rocsparse_destroy_handle(handle) &
                bind(c, name = 'rocsparse_destroy_handle')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_destroy_handle
            type(c_ptr), value :: handle
        end function rocsparse_destroy_handle

!       rocsparse_stream
        function rocsparse_set_stream(handle, stream) &
                bind(c, name = 'rocsparse_set_stream')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_set_stream
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function rocsparse_set_stream

        function rocsparse_get_stream(handle, stream) &
                bind(c, name = 'rocsparse_get_stream')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_get_stream
            type(c_ptr), value :: handle
            type(c_ptr) :: stream
        end function rocsparse_get_stream

!       rocsparse_pointer_mode
        function rocsparse_set_pointer_mode(handle, pointer_mode) &
                bind(c, name = 'rocsparse_set_pointer_mode')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_set_pointer_mode
            type(c_ptr), value :: handle
            integer(c_int), value :: pointer_mode
        end function rocsparse_set_pointer_mode

        function rocsparse_get_pointer_mode(handle, pointer_mode) &
                bind(c, name = 'rocsparse_get_pointer_mode')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_get_pointer_mode
            type(c_ptr), value :: handle
            integer(c_int) :: pointer_mode
        end function rocsparse_get_pointer_mode

!       rocsparse_version
        function rocsparse_get_version(handle, version) &
                bind(c, name = 'rocsparse_get_version')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_get_version
            type(c_ptr), value :: handle
            integer(c_int) :: version
        end function rocsparse_get_version

        function rocsparse_get_git_rev(handle, rev) &
                bind(c, name = 'rocsparse_get_git_rev')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_get_git_rev
            type(c_ptr), value :: handle
            character(c_char) :: rev(*)
        end function rocsparse_get_git_rev

!       rocsparse_mat_descr
        function rocsparse_create_mat_descr(descr) &
                bind(c, name = 'rocsparse_create_mat_descr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_create_mat_descr
            type(c_ptr) :: descr
        end function rocsparse_create_mat_descr

        function rocsparse_copy_mat_descr(dest, src) &
                bind(c, name = 'rocsparse_copy_mat_descr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_copy_mat_descr
            type(c_ptr), value :: dest
            type(c_ptr), intent(in), value :: src
        end function rocsparse_copy_mat_descr

        function rocsparse_destroy_mat_descr(descr) &
                bind(c, name = 'rocsparse_destroy_mat_descr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_destroy_mat_descr
            type(c_ptr), value :: descr
        end function rocsparse_destroy_mat_descr

!       rocsparse_index_base
        function rocsparse_set_mat_index_base(descr, base) &
                bind(c, name = 'rocsparse_set_mat_index_base')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_set_mat_index_base
            type(c_ptr), value :: descr
            integer(c_int), value :: base
        end function rocsparse_set_mat_index_base

        function rocsparse_get_mat_index_base(descr) &
                bind(c, name = 'rocsparse_get_mat_index_base')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_get_mat_index_base
            type(c_ptr), intent(in), value :: descr
        end function rocsparse_get_mat_index_base

!       rocsparse_matrix_type
        function rocsparse_set_mat_type(descr, mat_type) &
                bind(c, name = 'rocsparse_set_mat_type')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_set_mat_type
            type(c_ptr), value :: descr
            integer(c_int), value :: mat_type
        end function rocsparse_set_mat_type

        function rocsparse_get_mat_type(descr) &
                bind(c, name = 'rocsparse_get_mat_type')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_get_mat_type
            type(c_ptr), intent(in), value :: descr
        end function rocsparse_get_mat_type

!       rocsparse_fill_mode
        function rocsparse_set_mat_fill_mode(descr, fill_mode) &
                bind(c, name = 'rocsparse_set_mat_fill_mode')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_set_mat_fill_mode
            type(c_ptr), value :: descr
            integer(c_int), value :: fill_mode
        end function rocsparse_set_mat_fill_mode

        function rocsparse_get_mat_fill_mode(descr) &
                bind(c, name = 'rocsparse_get_mat_fill_mode')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_get_mat_fill_mode
            type(c_ptr), intent(in), value :: descr
        end function rocsparse_get_mat_fill_mode

!       rocsparse_diag_type
        function rocsparse_set_mat_diag_type(descr, diag_type) &
                bind(c, name = 'rocsparse_set_mat_diag_type')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_set_mat_diag_type
            type(c_ptr), value :: descr
            integer(c_int), value :: diag_type
        end function rocsparse_set_mat_diag_type

        function rocsparse_get_mat_diag_type(descr) &
                bind(c, name = 'rocsparse_get_mat_diag_type')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_get_mat_diag_type
            type(c_ptr), intent(in), value :: descr
        end function rocsparse_get_mat_diag_type

!       rocsparse_hyb_mat
        function rocsparse_create_hyb_mat(hyb) &
                bind(c, name = 'rocsparse_create_hyb_mat')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_create_hyb_mat
            type(c_ptr) :: hyb
        end function rocsparse_create_hyb_mat

        function rocsparse_destroy_hyb_mat(hyb) &
                bind(c, name = 'rocsparse_destroy_hyb_mat')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_destroy_hyb_mat
            type(c_ptr), value :: hyb
        end function rocsparse_destroy_hyb_mat

!       rocsparse_mat_info
        function rocsparse_create_mat_info(info) &
                bind(c, name = 'rocsparse_create_mat_info')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_create_mat_info
            type(c_ptr) :: info
        end function rocsparse_create_mat_info

        function rocsparse_destroy_mat_info(info) &
                bind(c, name = 'rocsparse_destroy_mat_info')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_destroy_mat_info
            type(c_ptr), value :: info
        end function rocsparse_destroy_mat_info

! ===========================================================================
!   level 1 SPARSE
! ===========================================================================

!       rocsparse_axpyi
        function rocsparse_saxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base) &
                bind(c, name = 'rocsparse_saxpyi')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_saxpyi
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_saxpyi

        function rocsparse_daxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base) &
                bind(c, name = 'rocsparse_daxpyi')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_daxpyi
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_daxpyi

        function rocsparse_caxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base) &
                bind(c, name = 'rocsparse_caxpyi')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_caxpyi
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_caxpyi

        function rocsparse_zaxpyi(handle, nnz, alpha, x_val, x_ind, y, idx_base) &
                bind(c, name = 'rocsparse_zaxpyi')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zaxpyi
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
                bind(c, name = 'rocsparse_sdoti')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sdoti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_sdoti

        function rocsparse_ddoti(handle, nnz, x_val, x_ind, y, result, idx_base) &
                bind(c, name = 'rocsparse_ddoti')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ddoti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_ddoti

        function rocsparse_cdoti(handle, nnz, x_val, x_ind, y, result, idx_base) &
                bind(c, name = 'rocsparse_cdoti')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cdoti
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_cdoti

        function rocsparse_zdoti(handle, nnz, x_val, x_ind, y, result, idx_base) &
                bind(c, name = 'rocsparse_zdoti')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zdoti
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
                bind(c, name = 'rocsparse_cdotci')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cdotci
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idx_base
        end function rocsparse_cdotci

        function rocsparse_zdotci(handle, nnz, x_val, x_ind, y, result, idx_base) &
                bind(c, name = 'rocsparse_zdotci')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zdotci
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
                bind(c, name = 'rocsparse_sgthr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgthr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_sgthr

        function rocsparse_dgthr(handle, nnz, y, x_val, x_ind, idx_base) &
                bind(c, name = 'rocsparse_dgthr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgthr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_dgthr

        function rocsparse_cgthr(handle, nnz, y, x_val, x_ind, idx_base) &
                bind(c, name = 'rocsparse_cgthr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgthr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_cgthr

        function rocsparse_zgthr(handle, nnz, y, x_val, x_ind, idx_base) &
                bind(c, name = 'rocsparse_zgthr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgthr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_zgthr

!       rocsparse_gthrz
        function rocsparse_sgthrz(handle, nnz, y, x_val, x_ind, idx_base) &
                bind(c, name = 'rocsparse_sgthrz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgthrz
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_sgthrz

        function rocsparse_dgthrz(handle, nnz, y, x_val, x_ind, idx_base) &
                bind(c, name = 'rocsparse_dgthrz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgthrz
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_dgthrz

        function rocsparse_cgthrz(handle, nnz, y, x_val, x_ind, idx_base) &
                bind(c, name = 'rocsparse_cgthrz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgthrz
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_cgthrz

        function rocsparse_zgthrz(handle, nnz, y, x_val, x_ind, idx_base) &
                bind(c, name = 'rocsparse_zgthrz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgthrz
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            integer(c_int), value :: idx_base
        end function rocsparse_zgthrz

!       rocsparse_roti
        function rocsparse_sroti(handle, nnz, x_val, x_ind, y, c, s, idx_base) &
                bind(c, name = 'rocsparse_sroti')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sroti
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
                bind(c, name = 'rocsparse_droti')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_droti
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
                bind(c, name = 'rocsparse_ssctr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ssctr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_ssctr

        function rocsparse_dsctr(handle, nnz, x_val, x_ind, y, idx_base) &
                bind(c, name = 'rocsparse_dsctr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dsctr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_dsctr

        function rocsparse_csctr(handle, nnz, x_val, x_ind, y, idx_base) &
                bind(c, name = 'rocsparse_csctr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csctr
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: x_val
            type(c_ptr), intent(in), value :: x_ind
            type(c_ptr), value :: y
            integer(c_int), value :: idx_base
        end function rocsparse_csctr

        function rocsparse_zsctr(handle, nnz, x_val, x_ind, y, idx_base) &
                bind(c, name = 'rocsparse_zsctr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zsctr
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
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, beta, y) &
                bind(c, name = 'rocsparse_sbsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsrmv
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
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_sbsrmv

        function rocsparse_dbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, beta, y) &
                bind(c, name = 'rocsparse_dbsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsrmv
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
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_dbsrmv

        function rocsparse_cbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, beta, y) &
                bind(c, name = 'rocsparse_cbsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsrmv
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
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_cbsrmv

        function rocsparse_zbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, beta, y) &
                bind(c, name = 'rocsparse_zbsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsrmv
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
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_zbsrmv

!       rocsparse_bsrsv_zero_pivot
        function rocsparse_bsrsv_zero_pivot(handle, info, position) &
                bind(c, name = 'rocsparse_bsrsv_zero_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_bsrsv_zero_pivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_bsrsv_zero_pivot

!       rocsparse_bsrsv_buffer_size
        function rocsparse_sbsrsv_buffer_size(handle, dir, trans, mb, nnzb, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_sbsrsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsrsv_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_sbsrsv_buffer_size

        function rocsparse_dbsrsv_buffer_size(handle, dir, trans, mb, nnzb, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_dbsrsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsrsv_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_dbsrsv_buffer_size

        function rocsparse_cbsrsv_buffer_size(handle, dir, trans, mb, nnzb, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_cbsrsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsrsv_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_cbsrsv_buffer_size

        function rocsparse_zbsrsv_buffer_size(handle, dir, trans, mb, nnzb, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_zbsrsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsrsv_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_zbsrsv_buffer_size

!       rocsparse_bsrsv_analysis
        function rocsparse_sbsrsv_analysis(handle, dir, trans, mb, nnzb, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, analysis, solve, &
                temp_buffer) &
                bind(c, name = 'rocsparse_sbsrsv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsrsv_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sbsrsv_analysis

        function rocsparse_dbsrsv_analysis(handle, dir, trans, mb, nnzb, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, analysis, solve, &
                temp_buffer) &
                bind(c, name = 'rocsparse_dbsrsv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsrsv_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dbsrsv_analysis

        function rocsparse_cbsrsv_analysis(handle, dir, trans, mb, nnzb, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, analysis, solve, &
                temp_buffer) &
                bind(c, name = 'rocsparse_cbsrsv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsrsv_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cbsrsv_analysis

        function rocsparse_zbsrsv_analysis(handle, dir, trans, mb, nnzb, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, analysis, solve, &
                temp_buffer) &
                bind(c, name = 'rocsparse_zbsrsv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsrsv_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zbsrsv_analysis

!       rocsparse_bsrsv_clear
        function rocsparse_bsrsv_clear(handle, info) &
                bind(c, name = 'rocsparse_bsrsv_clear')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_bsrsv_clear
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_bsrsv_clear

!       rocsparse_bsrsv_solve
        function rocsparse_sbsrsv_solve(handle, dir, trans, mb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, y, policy, &
                temp_buffer) &
                bind(c, name = 'rocsparse_sbsrsv_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsrsv_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sbsrsv_solve

        function rocsparse_dbsrsv_solve(handle, dir, trans, mb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, y, policy, &
                temp_buffer) &
                bind(c, name = 'rocsparse_dbsrsv_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsrsv_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dbsrsv_solve

        function rocsparse_cbsrsv_solve(handle, dir, trans, mb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, y, policy, &
                temp_buffer) &
                bind(c, name = 'rocsparse_cbsrsv_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsrsv_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cbsrsv_solve

        function rocsparse_zbsrsv_solve(handle, dir, trans, mb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, bsr_dim, info, x, y, policy, &
                temp_buffer) &
                bind(c, name = 'rocsparse_zbsrsv_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsrsv_solve
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: bsr_dim
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zbsrsv_solve

!       rocsparse_coomv
        function rocsparse_scoomv(handle, trans, m, n, nnz, alpha, descr, coo_val, &
                coo_row_ind, coo_col_ind, x, beta, y) &
                bind(c, name = 'rocsparse_scoomv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scoomv
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
                bind(c, name = 'rocsparse_dcoomv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcoomv
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
                bind(c, name = 'rocsparse_ccoomv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccoomv
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
                bind(c, name = 'rocsparse_zcoomv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcoomv
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
                bind(c, name = 'rocsparse_scsrmv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrmv_analysis
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
                bind(c, name = 'rocsparse_dcsrmv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrmv_analysis
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
                bind(c, name = 'rocsparse_ccsrmv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrmv_analysis
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
                bind(c, name = 'rocsparse_zcsrmv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrmv_analysis
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
                bind(c, name = 'rocsparse_csrmv_clear')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrmv_clear
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_csrmv_clear

!       rocsparse_csrmv
        function rocsparse_scsrmv(handle, trans, m, n, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, beta, y) &
                bind(c, name = 'rocsparse_scsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrmv
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
                bind(c, name = 'rocsparse_dcsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrmv
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
                bind(c, name = 'rocsparse_ccsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrmv
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
                bind(c, name = 'rocsparse_zcsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrmv
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
                bind(c, name = 'rocsparse_csrsv_zero_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrsv_zero_pivot
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_csrsv_zero_pivot

!       rocsparse_csrsv_buffer_size
        function rocsparse_scsrsv_buffer_size(handle, trans, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                bind(c, name = 'rocsparse_scsrsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrsv_buffer_size
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
                bind(c, name = 'rocsparse_dcsrsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrsv_buffer_size
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
                bind(c, name = 'rocsparse_ccsrsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrsv_buffer_size
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
                bind(c, name = 'rocsparse_zcsrsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrsv_buffer_size
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
                bind(c, name = 'rocsparse_scsrsv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrsv_analysis
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
                bind(c, name = 'rocsparse_dcsrsv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrsv_analysis
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
                bind(c, name = 'rocsparse_ccsrsv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrsv_analysis
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
                bind(c, name = 'rocsparse_zcsrsv_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrsv_analysis
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
                bind(c, name = 'rocsparse_csrsv_clear')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrsv_clear
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: info
        end function rocsparse_csrsv_clear

!       rocsparse_csrsv_solve
        function rocsparse_scsrsv_solve(handle, trans, m, nnz, alpha, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, x, y, policy, temp_buffer) &
                bind(c, name = 'rocsparse_scsrsv_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrsv_solve
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
                bind(c, name = 'rocsparse_dcsrsv_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrsv_solve
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
                bind(c, name = 'rocsparse_ccsrsv_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrsv_solve
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
                bind(c, name = 'rocsparse_zcsrsv_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrsv_solve
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
                bind(c, name = 'rocsparse_sellmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sellmv
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
                bind(c, name = 'rocsparse_dellmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dellmv
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
                bind(c, name = 'rocsparse_cellmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cellmv
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
                bind(c, name = 'rocsparse_zellmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zellmv
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
                bind(c, name = 'rocsparse_shybmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_shybmv
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
                bind(c, name = 'rocsparse_dhybmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dhybmv
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
                bind(c, name = 'rocsparse_chybmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_chybmv
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
                bind(c, name = 'rocsparse_zhybmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zhybmv
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_zhybmv

!       rocsparse_gebsrmv
        function rocsparse_sgebsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, x, beta, y) &
                bind(c, name = 'rocsparse_sgebsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgebsrmv
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
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_sgebsrmv

        function rocsparse_dgebsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, x, beta, y) &
                bind(c, name = 'rocsparse_dgebsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgebsrmv
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
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_dgebsrmv

        function rocsparse_cgebsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, x, beta, y) &
                bind(c, name = 'rocsparse_cgebsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgebsrmv
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
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_cgebsrmv

        function rocsparse_zgebsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, row_block_dim, col_block_dim, x, beta, y) &
                bind(c, name = 'rocsparse_zgebsrmv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgebsrmv
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
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function rocsparse_zgebsrmv

! ===========================================================================
!   level 3 SPARSE
! ===========================================================================
!       rocsparse_bsrmm
        function rocsparse_sbsrmm(handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, B, ldb, beta, C, ldc) &
                bind(c, name = 'rocsparse_sbsrmm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_sbsrmm

        function rocsparse_dbsrmm(handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, B, ldb, beta, C, ldc) &
                bind(c, name = 'rocsparse_dbsrmm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_dbsrmm

        function rocsparse_cbsrmm(handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, B, ldb, beta, C, ldc) &
                bind(c, name = 'rocsparse_cbsrmm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_cbsrmm

        function rocsparse_zbsrmm(handle, dir, trans_A, trans_B, mb, n, kb, nnzb, alpha, descr, &
                bsr_val, bsr_row_ptr, bsr_col_ind, block_dim, B, ldb, beta, C, ldc) &
                bind(c, name = 'rocsparse_zbsrmm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsrmm
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: mb
            integer(c_int), value :: n
            integer(c_int), value :: kb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_zbsrmm

!       rocsparse_csrmm
        function rocsparse_scsrmm(handle, trans_A, trans_B, m, n, k, nnz, alpha, descr, &
                csr_val, csr_row_ptr, csr_col_ind, B, ldb, beta, C, ldc) &
                bind(c, name = 'rocsparse_scsrmm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrmm
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
                bind(c, name = 'rocsparse_dcsrmm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrmm
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
                bind(c, name = 'rocsparse_ccsrmm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrmm
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
                bind(c, name = 'rocsparse_zcsrmm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrmm
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
                bind(c, name = 'rocsparse_csrsm_zero_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrsm_zero_pivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_csrsm_zero_pivot

!       rocsparse_csrsm_buffer_size
        function rocsparse_scsrsm_buffer_size(handle, trans_A, trans_B, m, nrhs, nnz, &
                alpha, descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                buffer_size) &
                bind(c, name = 'rocsparse_scsrsm_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrsm_buffer_size
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
                bind(c, name = 'rocsparse_dcsrsm_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrsm_buffer_size
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
                bind(c, name = 'rocsparse_ccsrsm_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrsm_buffer_size
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
                bind(c, name = 'rocsparse_zcsrsm_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrsm_buffer_size
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
                bind(c, name = 'rocsparse_scsrsm_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrsm_analysis
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
                bind(c, name = 'rocsparse_dcsrsm_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrsm_analysis
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
                bind(c, name = 'rocsparse_ccsrsm_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrsm_analysis
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
                bind(c, name = 'rocsparse_zcsrsm_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrsm_analysis
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
                bind(c, name = 'rocsparse_csrsm_clear')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrsm_clear
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_csrsm_clear

!       rocsparse_csrsm_solve
        function rocsparse_scsrsm_solve(handle, trans_A, trans_B, m, nrhs, nnz, alpha, &
                descr, csr_val, csr_row_ptr, csr_col_ind, B, ldb, info, policy, &
                temp_buffer) &
                bind(c, name = 'rocsparse_scsrsm_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrsm_solve
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
                bind(c, name = 'rocsparse_dcsrsm_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrsm_solve
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
                bind(c, name = 'rocsparse_ccsrsm_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrsm_solve
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
                bind(c, name = 'rocsparse_zcsrsm_solve')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrsm_solve
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

!       rocsparse_gemmi
        function rocsparse_sgemmi(handle, trans_A, trans_B, m, n, k, nnz, alpha, A, &
                lda, descr, csr_val, csr_row_ptr, csr_col_ind, beta, C, ldc) &
                bind(c, name = 'rocsparse_sgemmi')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgemmi
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_sgemmi

        function rocsparse_dgemmi(handle, trans_A, trans_B, m, n, k, nnz, alpha, A, &
                lda, descr, csr_val, csr_row_ptr, csr_col_ind, beta, C, ldc) &
                bind(c, name = 'rocsparse_dgemmi')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgemmi
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_dgemmi

        function rocsparse_cgemmi(handle, trans_A, trans_B, m, n, k, nnz, alpha, A, &
                lda, descr, csr_val, csr_row_ptr, csr_col_ind, beta, C, ldc) &
                bind(c, name = 'rocsparse_cgemmi')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgemmi
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_cgemmi

        function rocsparse_zgemmi(handle, trans_A, trans_B, m, n, k, nnz, alpha, A, &
                lda, descr, csr_val, csr_row_ptr, csr_col_ind, beta, C, ldc) &
                bind(c, name = 'rocsparse_zgemmi')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgemmi
            type(c_ptr), value :: handle
            integer(c_int), value :: trans_A
            integer(c_int), value :: trans_B
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function rocsparse_zgemmi

! ===========================================================================
!   extra SPARSE
! ===========================================================================

!       rocsparse_csrgeam_nnz
        function rocsparse_csrgeam_nnz(handle, m, n, descr_A, nnz_A, csr_row_ptr_A, &
                csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, csr_col_ind_B, descr_C, &
                csr_row_ptr_C, nnz_C) &
                bind(c, name = 'rocsparse_csrgeam_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrgeam_nnz
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
                bind(c, name = 'rocsparse_scsrgeam')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrgeam
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
                bind(c, name = 'rocsparse_dcsrgeam')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrgeam
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
                bind(c, name = 'rocsparse_ccsrgeam')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrgeam
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
                bind(c, name = 'rocsparse_zcsrgeam')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrgeam
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
                bind(c, name = 'rocsparse_scsrgemm_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrgemm_buffer_size
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
                bind(c, name = 'rocsparse_dcsrgemm_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrgemm_buffer_size
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
                bind(c, name = 'rocsparse_ccsrgemm_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrgemm_buffer_size
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
                bind(c, name = 'rocsparse_zcsrgemm_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrgemm_buffer_size
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
                bind(c, name = 'rocsparse_csrgemm_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrgemm_nnz
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
                bind(c, name = 'rocsparse_scsrgemm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrgemm
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
                bind(c, name = 'rocsparse_dcsrgemm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrgemm
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
                bind(c, name = 'rocsparse_ccsrgemm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrgemm
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
                bind(c, name = 'rocsparse_zcsrgemm')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrgemm
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

!       rocsparse_bsric0_zero_pivot
        function rocsparse_bsric0_zero_pivot(handle, info, position) &
                bind(c, name = 'rocsparse_bsric0_zero_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_bsric0_zero_pivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_bsric0_zero_pivot

!       rocsparse_bsric0_buffer_size
        function rocsparse_sbsric0_buffer_size(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_sbsric0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsric0_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_sbsric0_buffer_size

        function rocsparse_dbsric0_buffer_size(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_dbsric0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsric0_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_dbsric0_buffer_size

        function rocsparse_cbsric0_buffer_size(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_cbsric0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsric0_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_cbsric0_buffer_size

        function rocsparse_zbsric0_buffer_size(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_zbsric0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsric0_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_zbsric0_buffer_size

!       rocsparse_bsric0_analysis
        function rocsparse_sbsric0_analysis(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, solve, temp_buffer) &
                bind(c, name = 'rocsparse_sbsric0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsric0_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sbsric0_analysis

        function rocsparse_dbsric0_analysis(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, solve, temp_buffer) &
                bind(c, name = 'rocsparse_dbsric0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsric0_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dbsric0_analysis

        function rocsparse_cbsric0_analysis(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, solve, temp_buffer) &
                bind(c, name = 'rocsparse_cbsric0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsric0_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cbsric0_analysis

        function rocsparse_zbsric0_analysis(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, solve, temp_buffer) &
                bind(c, name = 'rocsparse_zbsric0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsric0_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zbsric0_analysis

!       rocsparse_bsric0_clear
        function rocsparse_bsric0_clear(handle, info) &
                bind(c, name = 'rocsparse_bsric0_clear')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_bsric0_clear
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_bsric0_clear

!       rocsparse_bsric0
        function rocsparse_sbsric0(handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_sbsric0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsric0
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sbsric0

        function rocsparse_dbsric0(handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_dbsric0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsric0
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dbsric0

        function rocsparse_cbsric0(handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_cbsric0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsric0
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cbsric0

        function rocsparse_zbsric0(handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_zbsric0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsric0
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zbsric0

!       rocsparse_bsrilu0_zero_pivot
        function rocsparse_bsrilu0_zero_pivot(handle, info, position) &
                bind(c, name = 'rocsparse_bsrilu0_zero_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_bsrilu0_zero_pivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_bsrilu0_zero_pivot

!       rocsparse_bsrilu0_buffer_size
        function rocsparse_sbsrilu0_buffer_size(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_sbsrilu0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsrilu0_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_sbsrilu0_buffer_size

        function rocsparse_dbsrilu0_buffer_size(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_dbsrilu0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsrilu0_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_dbsrilu0_buffer_size

        function rocsparse_cbsrilu0_buffer_size(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_cbsrilu0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsrilu0_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_cbsrilu0_buffer_size

        function rocsparse_zbsrilu0_buffer_size(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, buffer_size) &
                bind(c, name = 'rocsparse_zbsrilu0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsrilu0_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_zbsrilu0_buffer_size

!       rocsparse_bsrilu0_analysis
        function rocsparse_sbsrilu0_analysis(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, solve, temp_buffer) &
                bind(c, name = 'rocsparse_sbsrilu0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsrilu0_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sbsrilu0_analysis

        function rocsparse_dbsrilu0_analysis(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, solve, temp_buffer) &
                bind(c, name = 'rocsparse_dbsrilu0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsrilu0_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dbsrilu0_analysis

        function rocsparse_cbsrilu0_analysis(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, solve, temp_buffer) &
                bind(c, name = 'rocsparse_cbsrilu0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsrilu0_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cbsrilu0_analysis

        function rocsparse_zbsrilu0_analysis(handle, dir, mb, nnzb, descr, bsr_val, &
                bsr_row_ptr, bsr_col_ind, block_dim, info, analysis, solve, temp_buffer) &
                bind(c, name = 'rocsparse_zbsrilu0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsrilu0_analysis
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: analysis
            integer(c_int), value :: solve
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zbsrilu0_analysis

!       rocsparse_bsrilu0_clear
        function rocsparse_bsrilu0_clear(handle, info) &
                bind(c, name = 'rocsparse_bsrilu0_clear')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_bsrilu0_clear
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_bsrilu0_clear

!       rocsparse_bsrilu0
        function rocsparse_sbsrilu0(handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_sbsrilu0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsrilu0
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sbsrilu0

        function rocsparse_dbsrilu0(handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_dbsrilu0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsrilu0
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dbsrilu0

        function rocsparse_cbsrilu0(handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_cbsrilu0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsrilu0
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cbsrilu0

        function rocsparse_zbsrilu0(handle, dir, mb, nnzb, descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, block_dim, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_zbsrilu0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsrilu0
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: block_dim
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zbsrilu0

!       rocsparse_csric0_zero_pivot
        function rocsparse_csric0_zero_pivot(handle, info, position) &
                bind(c, name = 'rocsparse_csric0_zero_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csric0_zero_pivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_csric0_zero_pivot

!       rocsparse_csric0_buffer_size
        function rocsparse_scsric0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                bind(c, name = 'rocsparse_scsric0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsric0_buffer_size
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
                bind(c, name = 'rocsparse_dcsric0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsric0_buffer_size
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
                bind(c, name = 'rocsparse_ccsric0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsric0_buffer_size
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
                bind(c, name = 'rocsparse_zcsric0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsric0_buffer_size
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
                bind(c, name = 'rocsparse_scsric0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsric0_analysis
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
                bind(c, name = 'rocsparse_dcsric0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsric0_analysis
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
                bind(c, name = 'rocsparse_ccsric0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsric0_analysis
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
                bind(c, name = 'rocsparse_zcsric0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsric0_analysis
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
                bind(c, name = 'rocsparse_csric0_clear')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csric0_clear
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_csric0_clear

!       rocsparse_csric0
        function rocsparse_scsric0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_scsric0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsric0
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
                bind(c, name = 'rocsparse_dcsric0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsric0
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
                bind(c, name = 'rocsparse_ccsric0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsric0
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
                bind(c, name = 'rocsparse_zcsric0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsric0
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
                bind(c, name = 'rocsparse_csrilu0_zero_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrilu0_zero_pivot
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function rocsparse_csrilu0_zero_pivot

!       rocsparse_csrilu0_buffer_size
        function rocsparse_scsrilu0_buffer_size(handle, m, nnz, descr, csr_val, &
                csr_row_ptr, csr_col_ind, info, buffer_size) &
                bind(c, name = 'rocsparse_scsrilu0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrilu0_buffer_size
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
                bind(c, name = 'rocsparse_dcsrilu0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrilu0_buffer_size
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
                bind(c, name = 'rocsparse_ccsrilu0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrilu0_buffer_size
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
                bind(c, name = 'rocsparse_zcsrilu0_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrilu0_buffer_size
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
                bind(c, name = 'rocsparse_scsrilu0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrilu0_analysis
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
                bind(c, name = 'rocsparse_dcsrilu0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrilu0_analysis
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
                bind(c, name = 'rocsparse_ccsrilu0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrilu0_analysis
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
                bind(c, name = 'rocsparse_zcsrilu0_analysis')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrilu0_analysis
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
                bind(c, name = 'rocsparse_csrilu0_clear')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrilu0_clear
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
        end function rocsparse_csrilu0_clear

!       rocsparse_csrilu0
        function rocsparse_scsrilu0(handle, m, nnz, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, policy, temp_buffer) &
                bind(c, name = 'rocsparse_scsrilu0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsrilu0
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
                bind(c, name = 'rocsparse_dcsrilu0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsrilu0
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
                bind(c, name = 'rocsparse_ccsrilu0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsrilu0
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
                bind(c, name = 'rocsparse_zcsrilu0')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsrilu0
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

!       rocsparse_gtsv_buffer_size
        function rocsparse_sgtsv_buffer_size(handle, m, n, dl, d, du, &
                B, ldb, buffer_size) &
                bind(c, name = 'rocsparse_sgtsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgtsv_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: buffer_size
        end function rocsparse_sgtsv_buffer_size

        function rocsparse_dgtsv_buffer_size(handle, m, n, dl, d, du, &
                B, ldb, buffer_size) &
                bind(c, name = 'rocsparse_dgtsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgtsv_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: buffer_size
        end function rocsparse_dgtsv_buffer_size

        function rocsparse_cgtsv_buffer_size(handle, m, n, dl, d, du, &
                B, ldb, buffer_size) &
                bind(c, name = 'rocsparse_cgtsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgtsv_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: buffer_size
        end function rocsparse_cgtsv_buffer_size

        function rocsparse_zgtsv_buffer_size(handle, m, n, dl, d, du, &
                B, ldb, buffer_size) &
                bind(c, name = 'rocsparse_zgtsv_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgtsv_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: buffer_size
        end function rocsparse_zgtsv_buffer_size

!       rocsparse_gtsv
        function rocsparse_sgtsv(handle, m, n, dl, d, du, &
                B, ldb, temp_buffer) &
                bind(c, name = 'rocsparse_sgtsv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgtsv
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sgtsv

        function rocsparse_dgtsv(handle, m, n, dl, d, du, &
                B, ldb, temp_buffer) &
                bind(c, name = 'rocsparse_dgtsv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgtsv
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dgtsv

        function rocsparse_cgtsv(handle, m, n, dl, d, du, &
                B, ldb, temp_buffer) &
                bind(c, name = 'rocsparse_cgtsv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgtsv
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cgtsv

        function rocsparse_zgtsv(handle, m, n, dl, d, du, &
                B, ldb, temp_buffer) &
                bind(c, name = 'rocsparse_zgtsv')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgtsv
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zgtsv

!       rocsparse_gtsv_no_pivot_buffer_size
        function rocsparse_sgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, &
                B, ldb, buffer_size) &
                bind(c, name = 'rocsparse_sgtsv_no_pivot_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgtsv_no_pivot_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: buffer_size
        end function rocsparse_sgtsv_no_pivot_buffer_size

        function rocsparse_dgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, &
                B, ldb, buffer_size) &
                bind(c, name = 'rocsparse_dgtsv_no_pivot_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgtsv_no_pivot_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: buffer_size
        end function rocsparse_dgtsv_no_pivot_buffer_size

        function rocsparse_cgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, &
                B, ldb, buffer_size) &
                bind(c, name = 'rocsparse_cgtsv_no_pivot_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgtsv_no_pivot_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: buffer_size
        end function rocsparse_cgtsv_no_pivot_buffer_size

        function rocsparse_zgtsv_no_pivot_buffer_size(handle, m, n, dl, d, du, &
                B, ldb, buffer_size) &
                bind(c, name = 'rocsparse_zgtsv_no_pivot_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgtsv_no_pivot_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: buffer_size
        end function rocsparse_zgtsv_no_pivot_buffer_size

!       rocsparse_gtsv_no_pivot
        function rocsparse_sgtsv_no_pivot(handle, m, n, dl, d, du, &
                B, ldb, temp_buffer) &
                bind(c, name = 'rocsparse_sgtsv_no_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgtsv_no_pivot
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sgtsv_no_pivot

        function rocsparse_dgtsv_no_pivot(handle, m, n, dl, d, du, &
                B, ldb, temp_buffer) &
                bind(c, name = 'rocsparse_dgtsv_no_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgtsv_no_pivot
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dgtsv_no_pivot

        function rocsparse_cgtsv_no_pivot(handle, m, n, dl, d, du, &
                B, ldb, temp_buffer) &
                bind(c, name = 'rocsparse_cgtsv_no_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgtsv_no_pivot
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cgtsv_no_pivot

        function rocsparse_zgtsv_no_pivot(handle, m, n, dl, d, du, &
                B, ldb, temp_buffer) &
                bind(c, name = 'rocsparse_zgtsv_no_pivot')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgtsv_no_pivot
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zgtsv_no_pivot

!       rocsparse_gtsv_no_pivot_strided_batch_buffer_size
        function rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(handle, m, dl, d, du, &
                x, batch_count, batch_stride, buffer_size) &
                bind(c, name = 'rocsparse_sgtsv_no_pivot_strided_batch_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgtsv_no_pivot_strided_batch_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: x
            integer(c_int), value :: batch_count
            integer(c_int), value :: batch_stride
            type(c_ptr), value :: buffer_size
        end function rocsparse_sgtsv_no_pivot_strided_batch_buffer_size

        function rocsparse_dgtsv_no_pivot_strided_batch_buffer_size(handle, m, dl, d, du, &
                x, batch_count, batch_stride, buffer_size) &
                bind(c, name = 'rocsparse_dgtsv_no_pivot_strided_batch_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgtsv_no_pivot_strided_batch_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: x
            integer(c_int), value :: batch_count
            integer(c_int), value :: batch_stride
            type(c_ptr), value :: buffer_size
        end function rocsparse_dgtsv_no_pivot_strided_batch_buffer_size

        function rocsparse_cgtsv_no_pivot_strided_batch_buffer_size(handle, m, dl, d, du, &
                x, batch_count, batch_stride, buffer_size) &
                bind(c, name = 'rocsparse_cgtsv_no_pivot_strided_batch_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgtsv_no_pivot_strided_batch_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: x
            integer(c_int), value :: batch_count
            integer(c_int), value :: batch_stride
            type(c_ptr), value :: buffer_size
        end function rocsparse_cgtsv_no_pivot_strided_batch_buffer_size

        function rocsparse_zgtsv_no_pivot_strided_batch_buffer_size(handle, m, dl, d, du, &
                x, batch_count, batch_stride, buffer_size) &
                bind(c, name = 'rocsparse_zgtsv_no_pivot_strided_batch_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgtsv_no_pivot_strided_batch_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), intent(in), value :: x
            integer(c_int), value :: batch_count
            integer(c_int), value :: batch_stride
            type(c_ptr), value :: buffer_size
        end function rocsparse_zgtsv_no_pivot_strided_batch_buffer_size

!       rocsparse_gtsv_no_pivot_strided_batch
        function rocsparse_sgtsv_no_pivot_strided_batch(handle, m, dl, d, du, &
                x, batch_count, batch_stride, temp_buffer) &
                bind(c, name = 'rocsparse_sgtsv_no_pivot_strided_batch')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgtsv_no_pivot_strided_batch
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: x
            integer(c_int), value :: batch_count
            integer(c_int), value :: batch_stride
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sgtsv_no_pivot_strided_batch

        function rocsparse_dgtsv_no_pivot_strided_batch(handle, m, dl, d, du, &
                x, batch_count, batch_stride, temp_buffer) &
                bind(c, name = 'rocsparse_dgtsv_no_pivot_strided_batch')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgtsv_no_pivot_strided_batch
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: x
            integer(c_int), value :: batch_count
            integer(c_int), value :: batch_stride
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dgtsv_no_pivot_strided_batch

        function rocsparse_cgtsv_no_pivot_strided_batch(handle, m, dl, d, du, &
                x, batch_count, batch_stride, temp_buffer) &
                bind(c, name = 'rocsparse_cgtsv_no_pivot_strided_batch')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgtsv_no_pivot_strided_batch
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: x
            integer(c_int), value :: batch_count
            integer(c_int), value :: batch_stride
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cgtsv_no_pivot_strided_batch

        function rocsparse_zgtsv_no_pivot_strided_batch(handle, m, dl, d, du, &
                x, batch_count, batch_stride, temp_buffer) &
                bind(c, name = 'rocsparse_zgtsv_no_pivot_strided_batch')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgtsv_no_pivot_strided_batch
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: dl
            type(c_ptr), intent(in), value :: d
            type(c_ptr), intent(in), value :: du
            type(c_ptr), value :: x
            integer(c_int), value :: batch_count
            integer(c_int), value :: batch_stride
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zgtsv_no_pivot_strided_batch

! ===========================================================================
!   conversion SPARSE
! ===========================================================================

!       rocsparse_nnz
        function rocsparse_snnz(handle, dir, m, n, descr, A, ld, nnz_per_row_columns, &
                nnz_total_dev_host_ptr) &
                bind(c, name = 'rocsparse_snnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_snnz
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
                bind(c, name = 'rocsparse_dnnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dnnz
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
                bind(c, name = 'rocsparse_cnnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cnnz
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
                bind(c, name = 'rocsparse_znnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_znnz
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
                bind(c, name = 'rocsparse_sdense2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sdense2csr
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
                bind(c, name = 'rocsparse_ddense2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ddense2csr
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
                bind(c, name = 'rocsparse_cdense2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cdense2csr
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
                bind(c, name = 'rocsparse_zdense2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zdense2csr
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
                bind(c, name = 'rocsparse_sdense2csc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sdense2csc
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
                bind(c, name = 'rocsparse_ddense2csc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ddense2csc
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
                bind(c, name = 'rocsparse_cdense2csc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cdense2csc
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
                bind(c, name = 'rocsparse_zdense2csc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zdense2csc
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

!       rocsparse_dense2coo
        function rocsparse_sdense2coo(handle, m, n, descr, A, ld, nnz_per_rows, coo_val, &
                coo_row_ind, coo_col_ind) &
                bind(c, name = 'rocsparse_sdense2coo')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sdense2coo
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_rows
            type(c_ptr), value :: coo_val
            type(c_ptr), value :: coo_row_ind
            type(c_ptr), value :: coo_col_ind
        end function rocsparse_sdense2coo

        function rocsparse_ddense2coo(handle, m, n, descr, A, ld, nnz_per_rows, coo_val, &
                coo_row_ind, coo_col_ind) &
                bind(c, name = 'rocsparse_ddense2coo')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ddense2coo
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_rows
            type(c_ptr), value :: coo_val
            type(c_ptr), value :: coo_row_ind
            type(c_ptr), value :: coo_col_ind
        end function rocsparse_ddense2coo

        function rocsparse_cdense2coo(handle, m, n, descr, A, ld, nnz_per_rows, coo_val, &
                coo_row_ind, coo_col_ind) &
                bind(c, name = 'rocsparse_cdense2coo')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cdense2coo
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_rows
            type(c_ptr), value :: coo_val
            type(c_ptr), value :: coo_row_ind
            type(c_ptr), value :: coo_col_ind
        end function rocsparse_cdense2coo

        function rocsparse_zdense2coo(handle, m, n, descr, A, ld, nnz_per_rows, coo_val, &
                coo_row_ind, coo_col_ind) &
                bind(c, name = 'rocsparse_zdense2coo')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zdense2coo
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: ld
            type(c_ptr), intent(in), value :: nnz_per_rows
            type(c_ptr), value :: coo_val
            type(c_ptr), value :: coo_row_ind
            type(c_ptr), value :: coo_col_ind
        end function rocsparse_zdense2coo

!       rocsparse_csr2dense
        function rocsparse_scsr2dense(handle, m, n, descr, csr_val, csr_row_ptr, &
                csr_col_ind, A, ld) &
                bind(c, name = 'rocsparse_scsr2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsr2dense
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
                bind(c, name = 'rocsparse_dcsr2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsr2dense
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
                bind(c, name = 'rocsparse_ccsr2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsr2dense
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
                bind(c, name = 'rocsparse_zcsr2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsr2dense
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
                bind(c, name = 'rocsparse_scsc2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsc2dense
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
                bind(c, name = 'rocsparse_dcsc2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsc2dense
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
                bind(c, name = 'rocsparse_ccsc2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsc2dense
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
                bind(c, name = 'rocsparse_zcsc2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsc2dense
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

!       rocsparse_coo2dense
        function rocsparse_scoo2dense(handle, m, n, nnz, descr, coo_val, coo_row_ind, &
                coo_col_ind, A, ld) &
                bind(c, name = 'rocsparse_scoo2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scoo2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: coo_val
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_scoo2dense

        function rocsparse_dcoo2dense(handle, m, n, nnz, descr, coo_val, coo_row_ind, &
                coo_col_ind, A, ld) &
                bind(c, name = 'rocsparse_dcoo2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcoo2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: coo_val
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_dcoo2dense

        function rocsparse_ccoo2dense(handle, m, n, nnz, descr, coo_val, coo_row_ind, &
                coo_col_ind, A, ld) &
                bind(c, name = 'rocsparse_ccoo2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccoo2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: coo_val
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_ccoo2dense

        function rocsparse_zcoo2dense(handle, m, n, nnz, descr, coo_val, coo_row_ind, &
                coo_col_ind, A, ld) &
                bind(c, name = 'rocsparse_zcoo2dense')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcoo2dense
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: coo_val
            type(c_ptr), intent(in), value :: coo_row_ind
            type(c_ptr), intent(in), value :: coo_col_ind
            type(c_ptr), value :: A
            integer(c_int), value :: ld
        end function rocsparse_zcoo2dense

!       rocsparse_nnz_compress
        function rocsparse_snnz_compress(handle, m, descr_A, csr_val_A, csr_row_ptr_A, &
                nnz_per_row, nnz_C, tol) &
                bind(c, name = 'rocsparse_snnz_compress')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_snnz_compress
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
                bind(c, name = 'rocsparse_dnnz_compress')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dnnz_compress
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
                bind(c, name = 'rocsparse_cnnz_compress')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cnnz_compress
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
                bind(c, name = 'rocsparse_znnz_compress')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_znnz_compress
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
                bind(c, name = 'rocsparse_csr2coo')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csr2coo
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: csr_row_ptr
            integer(c_int), value :: nnz
            integer(c_int), value :: m
            type(c_ptr), value :: coo_row_ind
            integer(c_int), value :: idx_base
        end function rocsparse_csr2coo

!       rocsparse_gebsr2csr
        function rocsparse_sgebsr2csr(handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind) &
                bind(c, name = 'rocsparse_sgebsr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgebsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_sgebsr2csr

        function rocsparse_dgebsr2csr(handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind) &
                bind(c, name = 'rocsparse_dgebsr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgebsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_dgebsr2csr

        function rocsparse_cgebsr2csr(handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind) &
                bind(c, name = 'rocsparse_cgebsr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgebsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_cgebsr2csr

        function rocsparse_zgebsr2csr(handle, dir, mb, nb, bsr_descr, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind) &
                bind(c, name = 'rocsparse_zgebsr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgebsr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), value :: csr_val
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
        end function rocsparse_zgebsr2csr

!       rocsparse_gebsr2gebsc_buffer_size
        function rocsparse_sgebsr2gebsc_buffer_size(handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, buffer_size) &
                bind(c, name = 'rocsparse_sgebsr2gebsc_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgebsr2gebsc_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: buffer_size
        end function rocsparse_sgebsr2gebsc_buffer_size

      function rocsparse_dgebsr2gebsc_buffer_size(handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, buffer_size) &
                bind(c, name = 'rocsparse_dgebsr2gebsc_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgebsr2gebsc_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: buffer_size
      end function rocsparse_dgebsr2gebsc_buffer_size

      function rocsparse_cgebsr2gebsc_buffer_size(handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, buffer_size) &
                bind(c, name = 'rocsparse_cgebsr2gebsc_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgebsr2gebsc_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: buffer_size
      end function rocsparse_cgebsr2gebsc_buffer_size

      function rocsparse_zgebsr2gebsc_buffer_size(handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, buffer_size) &
                bind(c, name = 'rocsparse_zgebsr2gebsc_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgebsr2gebsc_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: buffer_size
        end function rocsparse_zgebsr2gebsc_buffer_size


!       rocsparse_gebsr2gebsc
        function rocsparse_sgebsr2gebsc(handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, bsc_val, bsc_row_ind, bsc_col_ptr, copy_values, &
                idx_base, temp_buffer) &
                bind(c, name = 'rocsparse_sgebsr2gebsc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgebsr2gebsc
            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: bsc_val
            type(c_ptr), value :: bsc_row_ind
            type(c_ptr), value :: bsc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sgebsr2gebsc


       function rocsparse_dgebsr2gebsc(handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, bsc_val, bsc_row_ind, bsc_col_ptr, copy_values, &
                idx_base, temp_buffer) &
                bind(c, name = 'rocsparse_dgebsr2gebsc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgebsr2gebsc
            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: bsc_val
            type(c_ptr), value :: bsc_row_ind
            type(c_ptr), value :: bsc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dgebsr2gebsc

      function rocsparse_cgebsr2gebsc(handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, bsc_val, bsc_row_ind, bsc_col_ptr, copy_values, &
                idx_base, temp_buffer) &
                bind(c, name = 'rocsparse_cgebsr2gebsc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgebsr2gebsc
            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: bsc_val
            type(c_ptr), value :: bsc_row_ind
            type(c_ptr), value :: bsc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cgebsr2gebsc

      function rocsparse_zgebsr2gebsc(handle, mb, nb, nnzb, bsr_val, bsr_row_ptr, &
                bsr_col_ind, row_block_dim, col_block_dim, bsc_val, bsc_row_ind, bsc_col_ptr, copy_values, &
                idx_base, temp_buffer) &
                bind(c, name = 'rocsparse_zgebsr2gebsc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgebsr2gebsc
            type(c_ptr), value :: handle
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: bsr_val
            type(c_ptr), intent(in), value :: bsr_row_ptr
            type(c_ptr), intent(in), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: bsc_val
            type(c_ptr), value :: bsc_row_ind
            type(c_ptr), value :: bsc_col_ptr
            integer(c_int), value :: copy_values
            integer(c_int), value :: idx_base
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zgebsr2gebsc


!       rocsparse_gebsr2gebsr_buffer_size
        function rocsparse_sgebsr2gebsr_buffer_size(handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, &
                bsr_col_ind_A, row_block_dim_A, col_block_dim_A, row_block_dim_C, col_block_dim_C, buffer_size) &
                bind(c, name = 'rocsparse_sgebsr2gebsr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgebsr2gebsr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_val_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_sgebsr2gebsr_buffer_size

        function rocsparse_dgebsr2gebsr_buffer_size(handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, &
                bsr_col_ind_A, row_block_dim_A, col_block_dim_A, row_block_dim_C, col_block_dim_C, buffer_size) &
                bind(c, name = 'rocsparse_dgebsr2gebsr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgebsr2gebsr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_val_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_dgebsr2gebsr_buffer_size

      function rocsparse_cgebsr2gebsr_buffer_size(handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, &
                bsr_col_ind_A, row_block_dim_A, col_block_dim_A, row_block_dim_C, col_block_dim_C, buffer_size) &
                bind(c, name = 'rocsparse_cgebsr2gebsr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgebsr2gebsr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_val_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_cgebsr2gebsr_buffer_size

      function rocsparse_zgebsr2gebsr_buffer_size(handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, &
                bsr_col_ind_A, row_block_dim_A, col_block_dim_A, row_block_dim_C, col_block_dim_C, buffer_size) &
                bind(c, name = 'rocsparse_zgebsr2gebsr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgebsr2gebsr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_val_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_zgebsr2gebsr_buffer_size

!       rocsparse_gebsr2gebsr_nnz
        function rocsparse_gebsr2gebsr_nnz(handle, dir, mb, nb, nnzb, descr_A, bsr_row_ptr_A, bsr_col_ind_A, &
                  row_block_dim_A, col_block_dim_A, descr_C, bsr_row_ptr_C, row_block_dim_C, col_block_dim_C, &
                  nnz_total_dev_host_ptr, temp_buffer) &
                bind(c, name = 'rocsparse_gebsr2gebsr_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_gebsr2gebsr_nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: bsr_row_ptr_C
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: temp_buffer
        end function rocsparse_gebsr2gebsr_nnz

!       rocsparse_gebsr2gebsr
        function rocsparse_sgebsr2gebsr(handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, &
                bsr_col_ind_A, row_block_dim_A, col_block_dim_A, descr_C, bsr_val_C, bsr_row_ptr_C, &
                bsr_col_ind_C, row_block_dim_C, col_block_dim_C, temp_buffer) &
                bind(c, name = 'rocsparse_sgebsr2gebsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sgebsr2gebsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_val_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: bsr_val_C
            type(c_ptr), value :: bsr_row_ptr_C
            type(c_ptr), value :: bsr_col_ind_C
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sgebsr2gebsr


       function rocsparse_dgebsr2gebsr(handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, &
                bsr_col_ind_A, row_block_dim_A, col_block_dim_A, descr_C, bsr_val_C, bsr_row_ptr_C, &
                bsr_col_ind_C, row_block_dim_C, col_block_dim_C, temp_buffer) &
                bind(c, name = 'rocsparse_dgebsr2gebsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dgebsr2gebsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_val_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: bsr_val_C
            type(c_ptr), value :: bsr_row_ptr_C
            type(c_ptr), value :: bsr_col_ind_C
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dgebsr2gebsr

      function rocsparse_cgebsr2gebsr(handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, &
                bsr_col_ind_A, row_block_dim_A, col_block_dim_A, descr_C, bsr_val_C, bsr_row_ptr_C, &
                bsr_col_ind_C, row_block_dim_C, col_block_dim_C, temp_buffer) &
                bind(c, name = 'rocsparse_cgebsr2gebsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cgebsr2gebsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_val_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: bsr_val_C
            type(c_ptr), value :: bsr_row_ptr_C
            type(c_ptr), value :: bsr_col_ind_C
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_cgebsr2gebsr

      function rocsparse_zgebsr2gebsr(handle, dir, mb, nb, nnzb, descr_A, bsr_val_A, bsr_row_ptr_A, &
                bsr_col_ind_A, row_block_dim_A, col_block_dim_A, descr_C, bsr_val_C, bsr_row_ptr_C, &
                bsr_col_ind_C, row_block_dim_C, col_block_dim_C, temp_buffer) &
                bind(c, name = 'rocsparse_zgebsr2gebsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zgebsr2gebsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: descr_A
            type(c_ptr), intent(in), value :: bsr_val_A
            type(c_ptr), intent(in), value :: bsr_row_ptr_A
            type(c_ptr), intent(in), value :: bsr_col_ind_A
            integer(c_int), value :: row_block_dim_A
            integer(c_int), value :: col_block_dim_A
            type(c_ptr), intent(in), value :: descr_C
            type(c_ptr), value :: bsr_val_C
            type(c_ptr), value :: bsr_row_ptr_C
            type(c_ptr), value :: bsr_col_ind_C
            integer(c_int), value :: row_block_dim_C
            integer(c_int), value :: col_block_dim_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zgebsr2gebsr


!       rocsparse_csr2csc_buffer_size
        function rocsparse_csr2csc_buffer_size(handle, m, n, nnz, csr_row_ptr, &
                csr_col_ind, copy_values, buffer_size) &
                bind(c, name = 'rocsparse_csr2csc_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csr2csc_buffer_size
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
                bind(c, name = 'rocsparse_scsr2csc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsr2csc
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
                bind(c, name = 'rocsparse_dcsr2csc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsr2csc
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
                bind(c, name = 'rocsparse_ccsr2csc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsr2csc
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
                bind(c, name = 'rocsparse_zcsr2csc')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsr2csc
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
                bind(c, name = 'rocsparse_csr2ell_width')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csr2ell_width
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
                bind(c, name = 'rocsparse_scsr2ell')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsr2ell
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
                bind(c, name = 'rocsparse_dcsr2ell')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsr2ell
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
                bind(c, name = 'rocsparse_ccsr2ell')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsr2ell
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
                bind(c, name = 'rocsparse_zcsr2ell')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsr2ell
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
                bind(c, name = 'rocsparse_scsr2hyb')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsr2hyb
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
                bind(c, name = 'rocsparse_dcsr2hyb')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsr2hyb
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
                bind(c, name = 'rocsparse_ccsr2hyb')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsr2hyb
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
                bind(c, name = 'rocsparse_zcsr2hyb')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsr2hyb
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
                bind(c, name = 'rocsparse_csr2bsr_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csr2bsr_nnz
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
                bind(c, name = 'rocsparse_scsr2bsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsr2bsr
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
                bind(c, name = 'rocsparse_dcsr2bsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsr2bsr
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
                bind(c, name = 'rocsparse_ccsr2bsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsr2bsr
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
                bind(c, name = 'rocsparse_zcsr2bsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsr2bsr
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

!       rocsparse_csr2gebsr_nnz
        function rocsparse_csr2gebsr_nnz(handle, dir, m, n, csr_descr, csr_row_ptr, &
                csr_col_ind, bsr_descr, bsr_row_ptr, row_block_dim, &
                col_block_dim, bsr_nnz, temp_buffer) &
                bind(c, name = 'rocsparse_csr2gebsr_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csr2gebsr_nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_row_ptr
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: bsr_nnz
            type(c_ptr), value :: temp_buffer
        end function rocsparse_csr2gebsr_nnz

!       rocsparse_csr2gebsr_buffer_size
      function rocsparse_scsr2gebsr_buffer_size(handle, &
                dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, row_block_dim, col_block_dim, &
                buffer_size) &
                bind(c, name = 'rocsparse_scsr2gebsr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsr2gebsr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: buffer_size
        end function rocsparse_scsr2gebsr_buffer_size

      function rocsparse_dcsr2gebsr_buffer_size(handle, &
                dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, row_block_dim, col_block_dim, &
                buffer_size) &
                bind(c, name = 'rocsparse_dcsr2gebsr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsr2gebsr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: buffer_size
        end function rocsparse_dcsr2gebsr_buffer_size


      function rocsparse_ccsr2gebsr_buffer_size(handle, &
                dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, row_block_dim, col_block_dim, &
                buffer_size) &
                bind(c, name = 'rocsparse_ccsr2gebsr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsr2gebsr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: buffer_size
        end function rocsparse_ccsr2gebsr_buffer_size


      function rocsparse_zcsr2gebsr_buffer_size(handle, &
                dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, row_block_dim, col_block_dim, &
                buffer_size) &
                bind(c, name = 'rocsparse_zcsr2gebsr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsr2gebsr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: buffer_size
        end function rocsparse_zcsr2gebsr_buffer_size


!     rocsparse_csr2gebsr
        function rocsparse_scsr2gebsr(handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind, &
                row_block_dim, col_block_dim, temp_buffer) &
                bind(c, name = 'rocsparse_scsr2gebsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsr2gebsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_val
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: temp_buffer
        end function rocsparse_scsr2gebsr


        function rocsparse_dcsr2gebsr(handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind, &
                row_block_dim, col_block_dim, temp_buffer) &
                bind(c, name = 'rocsparse_dcsr2gebsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsr2gebsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_val
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dcsr2gebsr


      function rocsparse_ccsr2gebsr(handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind, &
                row_block_dim, col_block_dim, temp_buffer) &
                bind(c, name = 'rocsparse_ccsr2gebsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsr2gebsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_val
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: temp_buffer
        end function rocsparse_ccsr2gebsr


      function rocsparse_zcsr2gebsr(handle, dir, m, n, csr_descr, csr_val, csr_row_ptr, &
                csr_col_ind, bsr_descr, bsr_val, bsr_row_ptr, bsr_col_ind, &
                row_block_dim, col_block_dim, temp_buffer) &
                bind(c, name = 'rocsparse_zcsr2gebsr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsr2gebsr
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: csr_descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), intent(in), value :: bsr_descr
            type(c_ptr), value :: bsr_val
            type(c_ptr), value :: bsr_row_ptr
            type(c_ptr), value :: bsr_col_ind
            integer(c_int), value :: row_block_dim
            integer(c_int), value :: col_block_dim
            type(c_ptr), value :: temp_buffer
        end function rocsparse_zcsr2gebsr


!       rocsparse_csr2csr_compress
        function rocsparse_scsr2csr_compress(handle, m, n, descr_A, csr_val_A, &
                csr_col_ind_A, csr_row_ptr_A, nnz_A, nnz_per_row, csr_val_C, &
                csr_col_ind_C, csr_row_ptr_C, tol) &
                bind(c, name = 'rocsparse_scsr2csr_compress')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_scsr2csr_compress
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
                bind(c, name = 'rocsparse_dcsr2csr_compress')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dcsr2csr_compress
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
            real(c_double), value :: tol
        end function rocsparse_dcsr2csr_compress

        function rocsparse_ccsr2csr_compress(handle, m, n, descr_A, csr_val_A, &
                csr_col_ind_A, csr_row_ptr_A, nnz_A, nnz_per_row, csr_val_C, &
                csr_col_ind_C, csr_row_ptr_C, tol) &
                bind(c, name = 'rocsparse_ccsr2csr_compress')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ccsr2csr_compress
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
            complex(c_float_complex), value :: tol
        end function rocsparse_ccsr2csr_compress

        function rocsparse_zcsr2csr_compress(handle, m, n, descr_A, csr_val_A, &
                csr_col_ind_A, csr_row_ptr_A, nnz_A, nnz_per_row, csr_val_C, &
                csr_col_ind_C, csr_row_ptr_C, tol) &
                bind(c, name = 'rocsparse_zcsr2csr_compress')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zcsr2csr_compress
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
            complex(c_double_complex), value :: tol
        end function rocsparse_zcsr2csr_compress

!       rocsparse_coo2csr
        function rocsparse_coo2csr(handle, coo_row_ind, nnz, m, csr_row_ptr, idx_base) &
                bind(c, name = 'rocsparse_coo2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_coo2csr
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
                bind(c, name = 'rocsparse_ell2csr_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_ell2csr_nnz
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
                bind(c, name = 'rocsparse_sell2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sell2csr
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
                bind(c, name = 'rocsparse_dell2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dell2csr
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
                bind(c, name = 'rocsparse_cell2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cell2csr
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
                bind(c, name = 'rocsparse_zell2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zell2csr
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
                bind(c, name = 'rocsparse_hyb2csr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_hyb2csr_buffer_size
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: buffer_size
        end function rocsparse_hyb2csr_buffer_size

!       rocsparse_hyb2csr
        function rocsparse_shyb2csr(handle, descr, hyb, csr_val, csr_row_ptr, &
                csr_col_ind, temp_buffer) &
                bind(c, name = 'rocsparse_shyb2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_shyb2csr
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
                bind(c, name = 'rocsparse_dhyb2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dhyb2csr
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
                bind(c, name = 'rocsparse_chyb2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_chyb2csr
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
                bind(c, name = 'rocsparse_zhyb2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zhyb2csr
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
                bind(c, name = 'rocsparse_create_identity_permutation')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_create_identity_permutation
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: p
        end function rocsparse_create_identity_permutation

!       rocsparse_csrsort_buffer_size
        function rocsparse_csrsort_buffer_size(handle, m, n, nnz, csr_row_ptr, &
                csr_col_ind, buffer_size) &
                bind(c, name = 'rocsparse_csrsort_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrsort_buffer_size
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
                bind(c, name = 'rocsparse_csrsort')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_csrsort
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
                bind(c, name = 'rocsparse_cscsort_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cscsort_buffer_size
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
                bind(c, name = 'rocsparse_cscsort')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cscsort
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
                bind(c, name = 'rocsparse_coosort_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_coosort_buffer_size
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
                bind(c, name = 'rocsparse_coosort_by_row')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_coosort_by_row
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
                bind(c, name = 'rocsparse_coosort_by_column')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_coosort_by_column
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
                bind(c, name = 'rocsparse_sbsr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sbsr2csr
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
                bind(c, name = 'rocsparse_dbsr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dbsr2csr
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
                bind(c, name = 'rocsparse_cbsr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_cbsr2csr
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
                bind(c, name = 'rocsparse_zbsr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_zbsr2csr
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

!       rocsparse_prune_dense2csr_buffer_size
        function rocsparse_sprune_dense2csr_buffer_size(handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, &
                csr_col_ind, buffer_size) &
                bind(c, name = 'rocsparse_sprune_dense2csr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_dense2csr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: buffer_size
        end function rocsparse_sprune_dense2csr_buffer_size

        function rocsparse_dprune_dense2csr_buffer_size(handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, &
                csr_col_ind, buffer_size) &
                bind(c, name = 'rocsparse_dprune_dense2csr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_dense2csr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: buffer_size
        end function rocsparse_dprune_dense2csr_buffer_size

!       rocsparse_prune_dense2csr_nnz
        function rocsparse_sprune_dense2csr_nnz(handle, m, n, A, lda, threshold, descr, csr_row_ptr, &
                nnz_total_dev_host_ptr, temp_buffer) &
                bind(c, name = 'rocsparse_sprune_dense2csr_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_dense2csr_nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sprune_dense2csr_nnz

        function rocsparse_dprune_dense2csr_nnz(handle, m, n, A, lda, threshold, descr, csr_row_ptr, &
                nnz_total_dev_host_ptr, temp_buffer) &
                bind(c, name = 'rocsparse_dprune_dense2csr_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_dense2csr_nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dprune_dense2csr_nnz

!       rocsparse_prune_dense2csr
        function rocsparse_sprune_dense2csr(handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, &
                csr_col_ind, temp_buffer) &
                bind(c, name = 'rocsparse_sprune_dense2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_dense2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sprune_dense2csr

        function rocsparse_dprune_dense2csr(handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, &
                csr_col_ind, temp_buffer) &
                bind(c, name = 'rocsparse_dprune_dense2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_dense2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dprune_dense2csr

!       rocsparse_prune_dense2csr_by_percentage_buffer_size
        function rocsparse_sprune_dense2csr_by_percentage_buffer_size(handle, m, n, A, lda, percentage, descr, &
                csr_val, csr_row_ptr, csr_col_ind, info, buffer_size) &
                bind(c, name = 'rocsparse_sprune_dense2csr_by_percentage_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_dense2csr_by_percentage_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            real(c_float), value :: percentage
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_sprune_dense2csr_by_percentage_buffer_size

        function rocsparse_dprune_dense2csr_by_percentage_buffer_size(handle, m, n, A, lda, percentage, descr, &
                csr_val, csr_row_ptr, csr_col_ind, info, buffer_size) &
                bind(c, name = 'rocsparse_dprune_dense2csr_by_percentage_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_dense2csr_by_percentage_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            real(c_double), value :: percentage
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), intent(in), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_dprune_dense2csr_by_percentage_buffer_size

!       rocsparse_prune_dense2csr_nnz_by_percentage
        function rocsparse_sprune_dense2csr_nnz_by_percentage(handle, m, n, A, lda, percentage, descr, csr_row_ptr, &
                nnz_total_dev_host_ptr, info, temp_buffer) &
                bind(c, name = 'rocsparse_sprune_dense2csr_nnz_by_percentage')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_dense2csr_nnz_by_percentage
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            real(c_float), value :: percentage
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: info
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sprune_dense2csr_nnz_by_percentage

        function rocsparse_dprune_dense2csr_nnz_by_percentage(handle, m, n, A, lda, percentage, descr, csr_row_ptr, &
                nnz_total_dev_host_ptr, info, temp_buffer) &
                bind(c, name = 'rocsparse_dprune_dense2csr_nnz_by_percentage')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_dense2csr_nnz_by_percentage
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            real(c_double), value :: percentage
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csr_row_ptr
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: info
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dprune_dense2csr_nnz_by_percentage

!       rocsparse_prune_dense2csr_by_percentage
        function rocsparse_sprune_dense2csr_by_percentage(handle, m, n, A, lda, percentage, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, temp_buffer) &
                bind(c, name = 'rocsparse_sprune_dense2csr_by_percentage')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_dense2csr_by_percentage
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            real(c_float), value :: percentage
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sprune_dense2csr_by_percentage

        function rocsparse_dprune_dense2csr_by_percentage(handle, m, n, A, lda, percentage, descr, csr_val, csr_row_ptr, &
                csr_col_ind, info, temp_buffer) &
                bind(c, name = 'rocsparse_dprune_dense2csr_by_percentage')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_dense2csr_by_percentage
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            real(c_double), value :: percentage
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csr_val
            type(c_ptr), intent(in), value :: csr_row_ptr
            type(c_ptr), value :: csr_col_ind
            type(c_ptr), value :: info
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dprune_dense2csr_by_percentage


!       rocsparse_prune_csr2csr_buffer_size
        function rocsparse_sprune_csr2csr_buffer_size(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, &
                csr_col_ind_A, threshold, csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, buffer_size) &
                bind(c, name = 'rocsparse_sprune_csr2csr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_csr2csr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), intent(in), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), intent(in), value :: csr_col_ind_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_sprune_csr2csr_buffer_size

        function rocsparse_dprune_csr2csr_buffer_size(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, &
                csr_col_ind_A, threshold, csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, buffer_size) &
                bind(c, name = 'rocsparse_dprune_csr2csr_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_csr2csr_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), intent(in), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), intent(in), value :: csr_col_ind_C
            type(c_ptr), value :: buffer_size
        end function rocsparse_dprune_csr2csr_buffer_size

!       rocsparse_prune_csr2csr_nnz
        function rocsparse_sprune_csr2csr_nnz(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, &
                csr_col_ind_A, threshold, csr_descr_C, csr_row_ptr_C, nnz_total_dev_host_ptr, temp_buffer) &
                bind(c, name = 'rocsparse_sprune_csr2csr_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_csr2csr_nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), value :: csr_row_ptr_C
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sprune_csr2csr_nnz

        function rocsparse_dprune_csr2csr_nnz(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, &
                csr_col_ind_A, threshold, csr_descr_C, csr_row_ptr_C, nnz_total_dev_host_ptr, temp_buffer) &
                bind(c, name = 'rocsparse_dprune_csr2csr_nnz')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_csr2csr_nnz
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), value :: csr_row_ptr_C
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dprune_csr2csr_nnz

!       rocsparse_prune_csr2csr
        function rocsparse_sprune_csr2csr(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, &
                threshold, csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, temp_buffer) &
                bind(c, name = 'rocsparse_sprune_csr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_csr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sprune_csr2csr

        function rocsparse_dprune_csr2csr(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, &
                threshold, csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, temp_buffer) &
                bind(c, name = 'rocsparse_dprune_csr2csr')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_csr2csr
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            type(c_ptr), intent(in), value :: threshold
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dprune_csr2csr


!       rocsparse_prune_csr2csr_by_percentage_buffer_size
        function rocsparse_sprune_csr2csr_by_percentage_buffer_size(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, &
                csr_col_ind_A, percentage, csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info, buffer_size) &
                bind(c, name = 'rocsparse_sprune_csr2csr_by_percentage_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_csr2csr_by_percentage_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            real(c_float), value :: percentage
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), intent(in), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), intent(in), value :: csr_col_ind_C
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_sprune_csr2csr_by_percentage_buffer_size

        function rocsparse_dprune_csr2csr_by_percentage_buffer_size(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, &
                csr_col_ind_A, percentage, csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info, buffer_size) &
                bind(c, name = 'rocsparse_dprune_csr2csr_by_percentage_buffer_size')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_csr2csr_by_percentage_buffer_size
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            real(c_double), value :: percentage
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), intent(in), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), intent(in), value :: csr_col_ind_C
            type(c_ptr), value :: info
            type(c_ptr), value :: buffer_size
        end function rocsparse_dprune_csr2csr_by_percentage_buffer_size

!       rocsparse_prune_csr2csr_nnz_by_percentage
        function rocsparse_sprune_csr2csr_nnz_by_percentage(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, &
                csr_col_ind_A, percentage, csr_descr_C, csr_row_ptr_C, nnz_total_dev_host_ptr, info, temp_buffer) &
                bind(c, name = 'rocsparse_sprune_csr2csr_nnz_by_percentage')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_csr2csr_nnz_by_percentage
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            real(c_float), value :: percentage
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), value :: csr_row_ptr_C
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: info
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sprune_csr2csr_nnz_by_percentage

        function rocsparse_dprune_csr2csr_nnz_by_percentage(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, &
                csr_col_ind_A, percentage, csr_descr_C, csr_row_ptr_C, nnz_total_dev_host_ptr, info, temp_buffer) &
                bind(c, name = 'rocsparse_dprune_csr2csr_nnz_by_percentage')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_csr2csr_nnz_by_percentage
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            real(c_double), value :: percentage
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), value :: csr_row_ptr_C
            type(c_ptr), value :: nnz_total_dev_host_ptr
            type(c_ptr), value :: info
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dprune_csr2csr_nnz_by_percentage

!       rocsparse_prune_csr2csr_by_percentage
        function rocsparse_sprune_csr2csr_by_percentage(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, &
                percentage, csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info, temp_buffer) &
                bind(c, name = 'rocsparse_sprune_csr2csr_by_percentage')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_sprune_csr2csr_by_percentage
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            real(c_float), value :: percentage
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), value :: info
            type(c_ptr), value :: temp_buffer
        end function rocsparse_sprune_csr2csr_by_percentage

        function rocsparse_dprune_csr2csr_by_percentage(handle, m, n, nnz_A, csr_descr_A, csr_val_A, csr_row_ptr_A, csr_col_ind_A, &
                percentage, csr_descr_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C, info, temp_buffer) &
                bind(c, name = 'rocsparse_dprune_csr2csr_by_percentage')
            use rocsparse_enums
            use iso_c_binding
            implicit none
            integer(kind(rocsparse_status_success)) :: rocsparse_dprune_csr2csr_by_percentage
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz_A
            type(c_ptr), intent(in), value :: csr_descr_A
            type(c_ptr), intent(in), value :: csr_val_A
            type(c_ptr), intent(in), value :: csr_row_ptr_A
            type(c_ptr), intent(in), value :: csr_col_ind_A
            real(c_float), value :: percentage
            type(c_ptr), intent(in), value :: csr_descr_C
            type(c_ptr), value :: csr_val_C
            type(c_ptr), intent(in), value :: csr_row_ptr_C
            type(c_ptr), value :: csr_col_ind_C
            type(c_ptr), value :: info
            type(c_ptr), value :: temp_buffer
        end function rocsparse_dprune_csr2csr_by_percentage

    end interface

end module rocsparse
