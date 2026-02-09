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
program example_fortran_hybmv
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

    integer, target :: h_a_ptr(5), h_a_col(8)
    real(8), target :: h_a_val(8), h_x(4), h_y(4)

    type(c_ptr) :: d_a_ptr, d_a_col, d_a_val, d_x, d_y

    integer :: i
    integer(c_int) :: m, n, nnz

    real(c_double), target :: alpha, beta

    type(c_ptr) :: handle, descr_a, hyb_a

    integer :: version

!   Input data
    m = 4
    n = 4
    nnz = 8

!   Fill CSR structure (to be converted to HYB)
    h_a_ptr = (/0, 3, 5, 6, 8/)
    h_a_col = (/0, 2, 3, 2, 3, 1, 0, 3/)
    h_a_val = (/1.0d0, 3.0d0, 4.0d0, 5.0d0, 1.0d0, 2.0d0, 4.0d0, 8.0d0/)

!   Scalar alpha and beta
    alpha = 1.0d0
    beta  = 0.0d0

!   x and y vectors
    h_x = (/1.0d0, 2.0d0, 3.0d0, 4.0d0/)
    h_y = (/4.0d0, 5.0d0, 6.0d0, 7.0d0/)

!   Allocate device memory for CSR matrix
    call HIP_CHECK(hipMalloc(d_a_ptr, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_a_col, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_a_val, int(nnz, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_x, int(n, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_y, int(m, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_a_ptr, c_loc(h_a_ptr), int(m + 1, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_a_col, c_loc(h_a_col), int(nnz, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_a_val, c_loc(h_a_val), int(nnz, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_x, c_loc(h_x), int(n, c_size_t) * 8, 1))

!   Create hipSPARSE handle
    call HIPSPARSE_CHECK(hipsparseCreate(handle))

!   Create matrix descriptor
    call HIPSPARSE_CHECK(hipsparseCreateMatDescr(descr_a))

!   Get hipSPARSE version
    call HIPSPARSE_CHECK(hipsparseGetVersion(handle, version))
    write(*,fmt='(A,I0,A,I0,A,I0)') 'hipSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100)

!   Create HYB matrix
    call HIPSPARSE_CHECK(hipsparseCreateHybMat(hyb_a))

!   Convert CSR to HYB format
    call HIPSPARSE_CHECK(hipsparseDcsr2hyb(handle, &
                                           m, &
                                           n, &
                                           descr_a, &
                                           d_a_val, &
                                           d_a_ptr, &
                                           d_a_col, &
                                           hyb_a, &
                                           0, &
                                           HIPSPARSE_HYB_PARTITION_AUTO))

!   Free CSR structures (no longer needed after conversion)
    call HIP_CHECK(hipFree(d_a_ptr))
    call HIP_CHECK(hipFree(d_a_col))
    call HIP_CHECK(hipFree(d_a_val))

!   Call dhybmv to perform y = alpha * A * x + beta * y
    call HIPSPARSE_CHECK(hipsparseDhybmv(handle, &
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                                        c_loc(alpha), &
                                        descr_a, &
                                        hyb_a, &
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
    call HIPSPARSE_CHECK(hipsparseDestroyHybMat(hyb_a))
    call HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr_a))
    call HIPSPARSE_CHECK(hipsparseDestroy(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_x))
    call HIP_CHECK(hipFree(d_y))

end program example_fortran_hybmv
! [doc example end]
