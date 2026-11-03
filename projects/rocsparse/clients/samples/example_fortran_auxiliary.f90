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

subroutine ROCSPARSE_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: rocsparse error'
        stop
    end if
end subroutine ROCSPARSE_CHECK

subroutine COMPARE_EQUAL(a, b)
    use iso_c_binding

    implicit none

    integer(c_int) :: a
    integer(c_int) :: b

    if(a /= b) then
        write(*,*) 'Error: rocsparse_error'
        stop
    end if
end subroutine COMPARE_EQUAL

program example_fortran_auxiliary
    use iso_c_binding
    use rocsparse

    implicit none

    type(c_ptr) :: handle
    type(c_ptr) :: descr_A
    type(c_ptr) :: descr_B

    integer :: version
    integer :: pointer_mode
    integer :: index_base
    integer :: mat_type
    integer :: fill_mode
    integer :: diag_type

    character(len=12) :: rev

!   Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

!   Get rocSPARSE version
    call ROCSPARSE_CHECK(rocsparse_get_version(handle, version))
    call ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev))

!   Print version on screen
    write(*,fmt='(A,I0,A,I0,A,I0,A,A)') 'rocSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100), '-', rev

!   Pointer mode
    call ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host))
    call ROCSPARSE_CHECK(rocsparse_get_pointer_mode(handle, pointer_mode))
    call COMPARE_EQUAL(pointer_mode, rocsparse_pointer_mode_host);

    call ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device))
    call ROCSPARSE_CHECK(rocsparse_get_pointer_mode(handle, pointer_mode))
    call COMPARE_EQUAL(pointer_mode, rocsparse_pointer_mode_device);

!   Matrix descriptor

!   Create matrix descriptors
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr_B))

!   Index base
    call ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descr_A, rocsparse_index_base_zero))
    index_base = rocsparse_get_mat_index_base(descr_A)
    call COMPARE_EQUAL(index_base, rocsparse_index_base_zero);

    call ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descr_A, rocsparse_index_base_one))
    index_base = rocsparse_get_mat_index_base(descr_A)
    call COMPARE_EQUAL(index_base, rocsparse_index_base_one);

!   Matrix type
    call ROCSPARSE_CHECK(rocsparse_set_mat_type(descr_A, rocsparse_matrix_type_general))
    mat_type = rocsparse_get_mat_type(descr_A)
    call COMPARE_EQUAL(mat_type, rocsparse_matrix_type_general);

    call ROCSPARSE_CHECK(rocsparse_set_mat_type(descr_A, rocsparse_matrix_type_symmetric))
    mat_type = rocsparse_get_mat_type(descr_A)
    call COMPARE_EQUAL(mat_type, rocsparse_matrix_type_symmetric);

    call ROCSPARSE_CHECK(rocsparse_set_mat_type(descr_A, rocsparse_matrix_type_hermitian))
    mat_type = rocsparse_get_mat_type(descr_A)
    call COMPARE_EQUAL(mat_type, rocsparse_matrix_type_hermitian);

    call ROCSPARSE_CHECK(rocsparse_set_mat_type(descr_A, rocsparse_matrix_type_triangular))
    mat_type = rocsparse_get_mat_type(descr_A)
    call COMPARE_EQUAL(mat_type, rocsparse_matrix_type_triangular);

!   Fill mode
    call ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_A, rocsparse_fill_mode_lower))
    fill_mode = rocsparse_get_mat_fill_mode(descr_A)
    call COMPARE_EQUAL(fill_mode, rocsparse_fill_mode_lower);

    call ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr_A, rocsparse_fill_mode_upper))
    fill_mode = rocsparse_get_mat_fill_mode(descr_A)
    call COMPARE_EQUAL(fill_mode, rocsparse_fill_mode_upper);

!   Diag type
    call ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_A, rocsparse_diag_type_non_unit))
    diag_type = rocsparse_get_mat_diag_type(descr_A)
    call COMPARE_EQUAL(diag_type, rocsparse_diag_type_non_unit);

    call ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr_A, rocsparse_diag_type_unit))
    diag_type = rocsparse_get_mat_diag_type(descr_A)
    call COMPARE_EQUAL(diag_type, rocsparse_diag_type_unit);

!   Copy matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_copy_mat_descr(descr_B, descr_A))
    index_base = rocsparse_get_mat_index_base(descr_B)
    mat_type = rocsparse_get_mat_type(descr_B)
    fill_mode = rocsparse_get_mat_fill_mode(descr_B)
    diag_type = rocsparse_get_mat_diag_type(descr_B)
    call COMPARE_EQUAL(index_base, rocsparse_index_base_one);
    call COMPARE_EQUAL(mat_type, rocsparse_matrix_type_triangular);
    call COMPARE_EQUAL(fill_mode, rocsparse_fill_mode_upper);
    call COMPARE_EQUAL(diag_type, rocsparse_diag_type_unit);

!   Clear rocSPARSE
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_A))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr_B))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

    write(*,fmt='(A)') 'All tests passed.'

end program example_fortran_auxiliary
