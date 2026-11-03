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
program example_fortran_bsrmv
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

    integer, target :: h_bsr_row_ptr(3), h_bsr_col_ind(4)
    real(8), target :: h_bsr_val(16), h_x(4), h_y(4)

    type(c_ptr) :: d_bsr_row_ptr, d_bsr_col_ind, d_bsr_val, d_x, d_y

    integer :: i
    integer(c_int) :: bsr_dim, mb, nb, nnzb, m, n, dir, trans

    real(c_double), target :: alpha, beta

    type(c_ptr) :: handle, descr

    integer :: version

!   BSR block dimension
    bsr_dim = 2

!   Number of block rows and columns
    mb = 2
    nb = 2

!   Number of non-zero blocks
    nnzb = 4

!   Number of rows and columns
    m = mb * bsr_dim
    n = nb * bsr_dim

!   Fill BSR structure
    h_bsr_row_ptr = (/0, 2, 4/)
    h_bsr_col_ind = (/0, 1, 0, 1/)
    h_bsr_val     = (/1.0d0, 3.0d0, 0.0d0, 0.0d0, 2.0d0, 4.0d0, 0.0d0, 0.0d0, &
                      5.0d0, 7.0d0, 6.0d0, 0.0d0, 0.0d0, 8.0d0, 0.0d0, 0.0d0/)

!   Block storage direction and transposition
    dir = HIPSPARSE_DIRECTION_COLUMN
    trans = HIPSPARSE_OPERATION_NON_TRANSPOSE

!   Scalar alpha and beta
    alpha = 3.7d0
    beta  = 1.3d0

!   x and y vectors
    h_x = (/1.0d0, 2.0d0, 3.0d0, 0.0d0/)
    h_y = (/4.0d0, 5.0d0, 6.0d0, 7.0d0/)

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_bsr_row_ptr, int(mb + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_col_ind, int(nnzb, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_val, int(nnzb * bsr_dim * bsr_dim, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_x, int(n, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_y, int(m, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_bsr_row_ptr, c_loc(h_bsr_row_ptr), int(mb + 1, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_bsr_col_ind, c_loc(h_bsr_col_ind), int(nnzb, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_bsr_val, c_loc(h_bsr_val), int(nnzb * bsr_dim * bsr_dim, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_x, c_loc(h_x), int(n, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_y, c_loc(h_y), int(m, c_size_t) * 8, 1))

!   Create hipSPARSE handle
    call HIPSPARSE_CHECK(hipsparseCreate(handle))

!   Create matrix descriptor
    call HIPSPARSE_CHECK(hipsparseCreateMatDescr(descr))

!   Get hipSPARSE version
    call HIPSPARSE_CHECK(hipsparseGetVersion(handle, version))
    write(*,fmt='(A,I0,A,I0,A,I0)') 'hipSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100)

!   Call dbsrmv to perform y = alpha * A * x + beta * y
    call HIPSPARSE_CHECK(hipsparseDbsrmv(handle, &
                                        dir, &
                                        trans, &
                                        mb, &
                                        nb, &
                                        nnzb, &
                                        c_loc(alpha), &
                                        descr, &
                                        d_bsr_val, &
                                        d_bsr_row_ptr, &
                                        d_bsr_col_ind, &
                                        bsr_dim, &
                                        d_x, &
                                        c_loc(beta), &
                                        d_y))

!   Copy result back to host
    call HIP_CHECK(hipMemcpy(c_loc(h_y), d_y, int(m, c_size_t) * 8, 2))

!   Print result
    write(*,fmt='(A)',advance='no') 'hy:'
    do i = 1, m
        write(*,fmt='(A,F0.2)',advance='no') ' ', h_y(i)
    end do
    write(*,*)

!   Clear hipSPARSE
    call HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr))
    call HIPSPARSE_CHECK(hipsparseDestroy(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_bsr_row_ptr))
    call HIP_CHECK(hipFree(d_bsr_col_ind))
    call HIP_CHECK(hipFree(d_bsr_val))
    call HIP_CHECK(hipFree(d_x))
    call HIP_CHECK(hipFree(d_y))

end program example_fortran_bsrmv
! [doc example end]
