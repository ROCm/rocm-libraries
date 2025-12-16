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
program example_fortran_csrgeam2
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

    integer, target :: h_csr_row_ptr_a(5), h_csr_col_ind_a(9)
    integer, target :: h_csr_row_ptr_b(5), h_csr_col_ind_b(6)
    integer, target :: h_csr_row_ptr_c(5)
    integer, allocatable, target :: h_csr_col_ind_c(:)
    real(4), target :: h_csr_val_a(9), h_csr_val_b(6)
    real(4), allocatable, target :: h_csr_val_c(:)
    real(4) :: temp(4)

    type(c_ptr) :: d_csr_row_ptr_a, d_csr_col_ind_a, d_csr_val_a
    type(c_ptr) :: d_csr_row_ptr_b, d_csr_col_ind_b, d_csr_val_b
    type(c_ptr) :: d_csr_row_ptr_c, d_csr_col_ind_c, d_csr_val_c
    type(c_ptr) :: d_buffer

    integer :: i, j, row_start, row_end
    integer(c_int) :: m, n, nnz_a, nnz_b
    integer(c_int), target :: nnz_c
    integer(c_size_t), target :: buffer_size

    real(c_float), target :: alpha, beta

    type(c_ptr) :: handle
    type(c_ptr) :: descr_a, descr_b, descr_c

!   Input data
    m = 4
    n = 4
    nnz_a = 9
    nnz_b = 6

!   Matrix A (4x4)
    h_csr_row_ptr_a = (/0, 2, 4, 8, 9/)
    h_csr_col_ind_a = (/0, 3, 0, 1, 0, 1, 2, 3, 2/)
    h_csr_val_a     = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0/)

!   Matrix B (4x4)
    h_csr_row_ptr_b = (/0, 1, 3, 5, 6/)
    h_csr_col_ind_b = (/1, 0, 2, 1, 3, 2/)
    h_csr_val_b     = (/1.0, 1.0, 1.0, 1.0, 1.0, 1.0/)

!   Scalar alpha and beta
    alpha = 1.0
    beta  = 1.0

!   Allocate device memory for A and B
    call HIP_CHECK(hipMalloc(d_csr_row_ptr_a, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind_a, int(nnz_a, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_a, int(nnz_a, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_row_ptr_b, int(m + 1, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind_b, int(nnz_b, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_b, int(nnz_b, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_row_ptr_c, int(m + 1, c_size_t) * 4))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr_a, c_loc(h_csr_row_ptr_a), int(m + 1, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind_a, c_loc(h_csr_col_ind_a), int(nnz_a, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_val_a, c_loc(h_csr_val_a), int(nnz_a, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr_b, c_loc(h_csr_row_ptr_b), int(m + 1, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind_b, c_loc(h_csr_col_ind_b), int(nnz_b, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_val_b, c_loc(h_csr_val_b), int(nnz_b, c_size_t) * 4, 1))

!   Create hipSPARSE handle
    call HIPSPARSE_CHECK(hipsparseCreate(handle))

!   Create matrix descriptors
    call HIPSPARSE_CHECK(hipsparseCreateMatDescr(descr_a))
    call HIPSPARSE_CHECK(hipsparseCreateMatDescr(descr_b))
    call HIPSPARSE_CHECK(hipsparseCreateMatDescr(descr_c))

!   Get buffer size
    call HIPSPARSE_CHECK(hipsparseScsrgeam2_bufferSizeExt(handle, &
                                                          m, &
                                                          n, &
                                                          c_loc(alpha), &
                                                          descr_a, &
                                                          nnz_a, &
                                                          d_csr_val_a, &
                                                          d_csr_row_ptr_a, &
                                                          d_csr_col_ind_a, &
                                                          c_loc(beta), &
                                                          descr_b, &
                                                          nnz_b, &
                                                          d_csr_val_b, &
                                                          d_csr_row_ptr_b, &
                                                          d_csr_col_ind_b, &
                                                          descr_c, &
                                                          c_null_ptr, &
                                                          d_csr_row_ptr_c, &
                                                          c_null_ptr, &
                                                          c_loc(buffer_size)))

!   Allocate buffer
    call HIP_CHECK(hipMalloc(d_buffer, buffer_size))

!   Get nnz of C
    call HIPSPARSE_CHECK(hipsparseXcsrgeam2Nnz(handle, &
                                               m, &
                                               n, &
                                               descr_a, &
                                               nnz_a, &
                                               d_csr_row_ptr_a, &
                                               d_csr_col_ind_a, &
                                               descr_b, &
                                               nnz_b, &
                                               d_csr_row_ptr_b, &
                                               d_csr_col_ind_b, &
                                               descr_c, &
                                               d_csr_row_ptr_c, &
                                               c_loc(nnz_c), &
                                               d_buffer))

!   Allocate device memory for C
    call HIP_CHECK(hipMalloc(d_csr_col_ind_c, int(nnz_c, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val_c, int(nnz_c, c_size_t) * 4))

!   Call scsrgeam2 to compute C = alpha * A + beta * B
    call HIPSPARSE_CHECK(hipsparseScsrgeam2(handle, &
                                           m, &
                                           n, &
                                           c_loc(alpha), &
                                           descr_a, &
                                           nnz_a, &
                                           d_csr_val_a, &
                                           d_csr_row_ptr_a, &
                                           d_csr_col_ind_a, &
                                           c_loc(beta), &
                                           descr_b, &
                                           nnz_b, &
                                           d_csr_val_b, &
                                           d_csr_row_ptr_b, &
                                           d_csr_col_ind_b, &
                                           descr_c, &
                                           d_csr_val_c, &
                                           d_csr_row_ptr_c, &
                                           d_csr_col_ind_c, &
                                           d_buffer))

!   Allocate host memory for C
    allocate(h_csr_col_ind_c(nnz_c))
    allocate(h_csr_val_c(nnz_c))

!   Copy result back to host
    call HIP_CHECK(hipMemcpy(c_loc(h_csr_row_ptr_c), d_csr_row_ptr_c, int(m + 1, c_size_t) * 4, 2))
    call HIP_CHECK(hipMemcpy(c_loc(h_csr_col_ind_c), d_csr_col_ind_c, int(nnz_c, c_size_t) * 4, 2))
    call HIP_CHECK(hipMemcpy(c_loc(h_csr_val_c), d_csr_val_c, int(nnz_c, c_size_t) * 4, 2))

!   Print result in dense format
    write(*,'(A)') 'C'
    do i = 1, m
        row_start = h_csr_row_ptr_c(i) + 1
        row_end = h_csr_row_ptr_c(i + 1)
        
        ! Initialize temp array to zeros
        temp = 0.0
        
        ! Fill in non-zero values
        do j = row_start, row_end
            temp(h_csr_col_ind_c(j) + 1) = h_csr_val_c(j)
        end do
        
        ! Print the row
        do j = 1, n
            write(*,'(I0,1X)',advance='no') int(temp(j))
        end do
        write(*,*)
    end do
    write(*,*)

!   Deallocate host arrays
    deallocate(h_csr_col_ind_c)
    deallocate(h_csr_val_c)

!   Clear hipSPARSE
    call HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr_a))
    call HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr_b))
    call HIPSPARSE_CHECK(hipsparseDestroyMatDescr(descr_c))
    call HIPSPARSE_CHECK(hipsparseDestroy(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_csr_row_ptr_a))
    call HIP_CHECK(hipFree(d_csr_col_ind_a))
    call HIP_CHECK(hipFree(d_csr_val_a))
    call HIP_CHECK(hipFree(d_csr_row_ptr_b))
    call HIP_CHECK(hipFree(d_csr_col_ind_b))
    call HIP_CHECK(hipFree(d_csr_val_b))
    call HIP_CHECK(hipFree(d_csr_row_ptr_c))
    call HIP_CHECK(hipFree(d_csr_col_ind_c))
    call HIP_CHECK(hipFree(d_csr_val_c))
    call HIP_CHECK(hipFree(d_buffer))

end program example_fortran_csrgeam2
! [doc example end]
