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

subroutine HIP_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hip error'
        stop
    end if

end subroutine HIP_CHECK

subroutine ROCSPARSE_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: rocsparse error'
        stop
    end if

end subroutine ROCSPARSE_CHECK

program example_fortran_gemmi
    use iso_c_binding
    use rocsparse

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

        function hipMemset(dst, val, size) &
                bind(c, name = 'hipMemset')
            use iso_c_binding
            implicit none
            integer :: hipMemset
            type(c_ptr), value :: dst
            integer(c_int), value :: val
            integer(c_size_t), value :: size
        end function hipMemset

        function hipDeviceSynchronize() &
                bind(c, name = 'hipDeviceSynchronize')
            use iso_c_binding
            implicit none
            integer :: hipDeviceSynchronize
        end function hipDeviceSynchronize

        function hipDeviceReset() &
                bind(c, name = 'hipDeviceReset')
            use iso_c_binding
            implicit none
            integer :: hipDeviceReset
        end function hipDeviceReset
    end interface

    integer, target :: h_csr_row_ptr(6), h_csr_col_ind(8)
    real(8), target :: h_csr_val(8), h_A(6), h_C(10)

    type(c_ptr) :: d_csr_row_ptr
    type(c_ptr) :: d_csr_col_ind
    type(c_ptr) :: d_csr_val
    type(c_ptr) :: d_A
    type(c_ptr) :: d_C

    integer :: i, j
    integer(c_int) :: M, N, K, nnz
    integer(c_int) :: lda, ldc

    real(c_double), target :: alpha
    real(c_double), target :: beta

    type(c_ptr) :: handle
    type(c_ptr) :: descr

    integer :: version

    character(len=12) :: rev

!   This example is going to compute C = alpha * A * B^T + beta * C

!   Input data

!   Number of rows and columns
    M = 2
    N = 5
    K = 3

    lda = M
    ldc = M

!   Number of non-zero entries
    nnz = 8

!   A = (  9 10 11 )
!       ( 12 13 14 )

!       ( 1 0 6 )
!       ( 2 4 0 )
!   B = ( 0 5 0 )
!       ( 3 0 7 )
!       ( 0 0 8 )

!   C = ( 15 16 17 18 19 )
!       ( 20 21 22 23 24 )

!   Fill A
    h_A = (/9, 12, 10, 13, 11, 14/)

!   Fill CSR structure of B
    h_csr_row_ptr = (/0, 2, 4, 5, 7, 8/)
    h_csr_col_ind = (/0, 2, 0, 1, 1, 0, 2, 2/)
    h_csr_val     = (/1, 6, 2, 4, 5, 3, 7, 8/)

!   Fill C
    h_C = (/15, 20, 16, 21, 17, 22, 18, 23, 19, 24/)

!   Scalar alpha and beta
    alpha = 3.7
    beta  = 1.3

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_A, int(M * K, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_csr_row_ptr, (int(N, c_size_t) + 1) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val, int(nnz, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_C, int(M * N, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_A, c_loc(h_A), int(M * K, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr), (int(N, c_size_t) + 1) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind), int(nnz, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_val, c_loc(h_csr_val), int(nnz, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_C, c_loc(h_C), int(M * N, c_size_t) * 8, 1))

!   Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

!   Get rocSPARSE version
    call ROCSPARSE_CHECK(rocsparse_get_version(handle, version))
    call ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev))

!   Print version on screen
    write(*,fmt='(A,I0,A,I0,A,I0,A,A)') 'rocSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100), '-', rev

!   Create matrix descriptor
    call ROCSPARSE_CHECK(rocsparse_create_mat_descr(descr))

!   Call dgemmi to perform matrix multiplication
    call ROCSPARSE_CHECK(rocsparse_dgemmi(handle, &
                                          rocsparse_operation_none, &
                                          rocsparse_operation_transpose, &
                                          M, &
                                          N, &
                                          K, &
                                          nnz, &
                                          c_loc(alpha), &
                                          d_A, &
                                          lda, &
                                          descr, &
                                          d_csr_val, &
                                          d_csr_row_ptr, &
                                          d_csr_col_ind, &
                                          c_loc(beta), &
                                          d_C, &
                                          ldc))

!   Print result
    call HIP_CHECK(hipMemcpy(c_loc(h_C), d_C, int(M * N, c_size_t) * 8, 2))

    write(*,*) 'C:'
    do i = 1, M
        do j = 0, N - 1
            write(*,fmt='(A,F0.5)', advance='no') ' ', h_C(i + j * ldc)
        end do
        write(*,*)
    end do

!   Clear rocSPARSE
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_A))
    call HIP_CHECK(hipFree(d_csr_row_ptr))
    call HIP_CHECK(hipFree(d_csr_col_ind))
    call HIP_CHECK(hipFree(d_csr_val))
    call HIP_CHECK(hipFree(d_C))

end program example_fortran_gemmi
