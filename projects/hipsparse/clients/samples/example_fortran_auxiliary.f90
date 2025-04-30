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

subroutine HIPSPARSE_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hipsparse error'
        stop
    end if
end subroutine HIPSPARSE_CHECK

subroutine COMPARE_EQUAL(a, b)
    use iso_c_binding

    implicit none

    integer(c_int) :: a
    integer(c_int) :: b

    if(a /= b) then
        write(*,*) 'Error: hipsparse_error'
        stop
    end if
end subroutine COMPARE_EQUAL

program example_fortran_auxiliary
    use iso_c_binding
    use hipsparse

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

!   Create hipSPARSE handle
    call HIPSPARSE_CHECK(hipsparseCreate(handle))

!   Get hipSPARSE version
    call HIPSPARSE_CHECK(hipsparseGetVersion(handle, version))

!   Print version on screen
    write(*,fmt='(A,I0,A,I0,A,I0)') 'hipSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100)

!   Pointer mode
    call HIPSPARSE_CHECK(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST))
    call HIPSPARSE_CHECK(hipsparseGetPointerMode(handle, pointer_mode))
    call COMPARE_EQUAL(pointer_mode, HIPSPARSE_POINTER_MODE_HOST);

    call HIPSPARSE_CHECK(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE))
    call HIPSPARSE_CHECK(hipsparseGetPointerMode(handle, pointer_mode))
    call COMPARE_EQUAL(pointer_mode, HIPSPARSE_POINTER_MODE_DEVICE);

!   Matrix descriptor

!   Create matrix descriptors
    call HIPSPARSE_CHECK(hipsparseCreateMatDescr(descr_A))
    call HIPSPARSE_CHECK(hipsparseCreateMatDescr(descr_B))

!   Index base
    call HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr_A, HIPSPARSE_INDEX_BASE_ZERO))
    index_base = hipsparseGetMatIndexBase(descr_A)
    call COMPARE_EQUAL(index_base, HIPSPARSE_INDEX_BASE_ZERO);

    call HIPSPARSE_CHECK(hipsparseSetMatIndexBase(descr_A, HIPSPARSE_INDEX_BASE_ONE))
    index_base = hipsparseGetMatIndexBase(descr_A)
    call COMPARE_EQUAL(index_base, HIPSPARSE_INDEX_BASE_ONE);

!   Matrix type
    call HIPSPARSE_CHECK(hipsparseSetMatType(descr_A, HIPSPARSE_MATRIX_TYPE_GENERAL))
    mat_type = hipsparseGetMatType(descr_A)
    call COMPARE_EQUAL(mat_type, HIPSPARSE_MATRIX_TYPE_GENERAL);

    call HIPSPARSE_CHECK(hipsparseSetMatType(descr_A, HIPSPARSE_MATRIX_TYPE_SYMMETRIC))
    mat_type = hipsparseGetMatType(descr_A)
    call COMPARE_EQUAL(mat_type, HIPSPARSE_MATRIX_TYPE_SYMMETRIC);

    call HIPSPARSE_CHECK(hipsparseSetMatType(descr_A, HIPSPARSE_MATRIX_TYPE_HERMITIAN))
    mat_type = hipsparseGetMatType(descr_A)
    call COMPARE_EQUAL(mat_type, HIPSPARSE_MATRIX_TYPE_HERMITIAN);

    call HIPSPARSE_CHECK(hipsparseSetMatType(descr_A, HIPSPARSE_MATRIX_TYPE_TRIANGULAR))
    mat_type = hipsparseGetMatType(descr_A)
    call COMPARE_EQUAL(mat_type, HIPSPARSE_MATRIX_TYPE_TRIANGULAR);

!   Fill mode
    call HIPSPARSE_CHECK(hipsparseSetMatFillMode(descr_A, HIPSPARSE_FILL_MODE_LOWER))
    fill_mode = hipsparseGetMatFillMode(descr_A)
    call COMPARE_EQUAL(fill_mode, HIPSPARSE_FILL_MODE_LOWER);

    call HIPSPARSE_CHECK(hipsparseSetMatFillMode(descr_A, HIPSPARSE_FILL_MODE_UPPER))
    fill_mode = hipsparseGetMatFillMode(descr_A)
    call COMPARE_EQUAL(fill_mode, HIPSPARSE_FILL_MODE_UPPER);

!   Diag type
    call HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr_A, HIPSPARSE_DIAG_TYPE_NON_UNIT))
    diag_type = hipsparseGetMatDiagType(descr_A)
    call COMPARE_EQUAL(diag_type, HIPSPARSE_DIAG_TYPE_NON_UNIT);

    call HIPSPARSE_CHECK(hipsparseSetMatDiagType(descr_A, HIPSPARSE_DIAG_TYPE_UNIT))
    diag_type = hipsparseGetMatDiagType(descr_A)
    call COMPARE_EQUAL(diag_type, HIPSPARSE_DIAG_TYPE_UNIT);

!   Copy matrix descriptor
    call HIPSPARSE_CHECK(hipsparseCopyMatDescr(descr_B, descr_A))
    index_base = hipsparseGetMatIndexBase(descr_B)
    mat_type = hipsparseGetMatType(descr_B)
    fill_mode = hipsparseGetMatFillMode(descr_B)
    diag_type = hipsparseGetMatDiagType(descr_B)
    call COMPARE_EQUAL(index_base, HIPSPARSE_INDEX_BASE_ONE);
    call COMPARE_EQUAL(mat_type, HIPSPARSE_MATRIX_TYPE_TRIANGULAR);
    call COMPARE_EQUAL(fill_mode, HIPSPARSE_FILL_MODE_UPPER);
    call COMPARE_EQUAL(diag_type, HIPSPARSE_DIAG_TYPE_UNIT);

!   Clear hipSPARSE
    call HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr_A))
    call HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr_B))
    call HIPSPARSE_CHECK(hipsparseDestroy(handle))

    write(*,fmt='(A)') 'All tests passed.'

end program example_fortran_auxiliary
