!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

subroutine HIP_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hip error'
        stop
    end if

end subroutine HIP_CHECK

subroutine HIPSPARSE_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hipsparse error'
        stop
    end if

end subroutine HIPSPARSE_CHECK

! [doc example start]
program example_fortran_roti
    use iso_c_binding
    use hipsparse

    implicit none

    interface
        function hipMalloc(ptr, size) &
                bind(c, name = 'hipMalloc')
            use iso_c_binding
            implicit none
            integer :: hipMalloc
            type(c_ptr) :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc

        function hipFree(ptr) &
                bind(c, name = 'hipFree')
            use iso_c_binding
            implicit none
            integer :: hipFree
            type(c_ptr), value :: ptr
        end function hipFree

        function hipMemcpy(dst, src, size, kind) &
                bind(c, name = 'hipMemcpy')
            use iso_c_binding
            implicit none
            integer :: hipMemcpy
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy
    end interface

    integer, target :: h_xind(3)
    real(4), target :: h_xval(3), h_y(9)

    type(c_ptr) :: d_xind
    type(c_ptr) :: d_xval
    type(c_ptr) :: d_y

    integer :: i
    integer(c_int) :: size, nnz

    real(c_float), target :: c
    real(c_float), target :: s

    type(c_ptr) :: handle

!   Input data

!   Number of entries in the dense vector
    size = 9

!   Number of non-zero entries
    nnz = 3

!   Fill structures
    h_xind = (/0, 3, 5/)
    h_xval = (/1.0, 2.0, 3.0/)
    h_y    = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0/)

!   c and s
    c = 3.7
    s = 1.3

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_xind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_xval, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_y, int(size, c_size_t) * 4))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_xind, c_loc(h_xind), int(nnz, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_xval, c_loc(h_xval), int(nnz, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_y, c_loc(h_y), int(size, c_size_t) * 4, 1))

!   Create hipSPARSE handle
    call HIPSPARSE_CHECK(hipsparseCreate(handle))

!   Call sroti
    call HIPSPARSE_CHECK(hipsparseSroti(handle, &
                                        nnz, &
                                        d_xval, &
                                        d_xind, &
                                        d_y, &
                                        c_loc(c), &
                                        c_loc(s), &
                                        HIPSPARSE_INDEX_BASE_ZERO))

!   Copy results back to host
    call HIP_CHECK(hipMemcpy(c_loc(h_xval), d_xval, int(nnz, c_size_t) * 4, 2))
    call HIP_CHECK(hipMemcpy(c_loc(h_y), d_y, int(size, c_size_t) * 4, 2))

    do i = 1, nnz
        write(*,fmt='(A,I0,A,F0.2)') 'x(', h_xind(i), ') = ', h_xval(i)
    end do

    write(*,fmt='(A)',advance='no') 'y:'
    do i = 1, size
        write(*,fmt='(A,F0.2)',advance='no') ' ', h_y(i)
    end do
    write(*,*)

!   Clear hipSPARSE
    call HIPSPARSE_CHECK(hipsparseDestroy(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_xind))
    call HIP_CHECK(hipFree(d_xval))
    call HIP_CHECK(hipFree(d_y))

end program example_fortran_roti
! [doc example end]

