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

program example_fortran_bsrmm
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

    integer, target :: h_bsr_row_ptr(3), h_bsr_col_ind(4)
    real(8), target :: h_bsr_val(16), h_B(6 * 10), h_C(4 * 10)

    type(c_ptr) :: d_bsr_row_ptr
    type(c_ptr) :: d_bsr_col_ind
    type(c_ptr) :: d_bsr_val
    type(c_ptr) :: d_B
    type(c_ptr) :: d_C

    integer :: i, j
    integer(c_int) :: M, Mb, N, K, Kb, nnzb, block_dim

    real(c_double), target :: alpha, beta

    type(c_ptr) :: handle
    type(c_ptr) :: descr

    integer :: version

    character(len=12) :: rev

!   Input data
!       ( 1 2 0 3 0 0 )
!   A = ( 0 4 5 0 0 0 )
!       ( 0 0 0 7 8 0 )
!       ( 0 0 1 2 4 1 )

!       ( 9  11 13 15 17 10 12 14 16 18 )
!       ( 8  10 1  10 6  11 7  3  12 17 )
!   B = ( 11 11 0  4  6  12 2  9  13 2  )
!       ( 15 3  2  3  8  1  2  4  6  6  )
!       ( 2  5  7  0  1  15 9  4  10 1  )
!       ( 7  12 12 1  12 5  1  11 1  14 )

!   Number of rows and columns
    block_dim = 2
    Mb = 2
    Kb = 3
    N = 10
    M = Mb * block_dim
    K = Kb * block_dim

!   Number of non-zero blocks
    nnzb = 4

!   Fill BSR structure
    h_bsr_row_ptr = (/0, 2, 4/)
    h_bsr_col_ind = (/0, 1, 1, 2/)
    h_bsr_val     = (/1, 2, 0, 4, 0, 3, 5, 0, 0, 7, 1, 2, 8, 0, 4, 1/)

!   Scalar alpha and beta
    alpha = 1.0
    beta = 0.0

!   Fill B in column-major
    h_B = (/9,  8,  11, 15, 2,  7,  &
            11, 10, 11, 3,  5,  12, &
            13, 1,  0,  2,  7,  12, &
            15, 10, 4,  3,  0,  1,  &
            17, 6,  6,  8,  1,  12, &
            10, 11, 12, 1,  15, 5,  &
            12, 7,  2,  2,  9,  1,  &
            14, 3,  9,  4,  4,  11, &
            16, 12, 13, 6,  10, 1,  &
            18, 17, 2,  6,  1,  14/)

!   Fill C in column-major
    h_C = (/0, 0, 0, 0, &
            0, 0, 0, 0, &
            0, 0, 0, 0, &
            0, 0, 0, 0, &
            0, 0, 0, 0, &
            0, 0, 0, 0, &
            0, 0, 0, 0, &
            0, 0, 0, 0, &
            0, 0, 0, 0, &
            0, 0, 0, 0/)

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_bsr_row_ptr, (int(Mb, c_size_t) + 1) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_col_ind, int(nnzb, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_val, int(nnzb * block_dim * block_dim, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_B, int(K * N, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_C, int(M * N, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_bsr_row_ptr, c_loc(h_bsr_row_ptr), (int(Mb, c_size_t) + 1) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_bsr_col_ind, c_loc(h_bsr_col_ind), int(nnzb, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_bsr_val, c_loc(h_bsr_val), int(nnzb * block_dim * block_dim, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_B, c_loc(h_B), int(K * N, c_size_t) * 8, 1))
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

!   Perform the matrix multiplication
    call ROCSPARSE_CHECK(rocsparse_dbsrmm(handle, &
                                          rocsparse_direction_row, &
                                          rocsparse_operation_none, &
                                          rocsparse_operation_none, &
                                          Mb, &
                                          N, &
                                          Kb, &
                                          nnzb, &
                                          c_loc(alpha), &
                                          descr, &
                                          d_bsr_val, &
                                          d_bsr_row_ptr, &
                                          d_bsr_col_ind, &
                                          block_dim, &
                                          d_B, &
                                          K, &
                                          c_loc(beta), &
                                          d_C, &
                                          M))

!   Print result
    call HIP_CHECK(hipMemcpy(c_loc(h_C), d_C, int(M * N, c_size_t) * 8, 2))

!   Note: C in column major ordering
    do i = 1, M
        do j = 1, N
            write(*,fmt='(A,F6.2)',advance='no') ' ', h_C(M * (j - 1) + i)
        end do
        write(*,*)
    end do

!   Clear rocSPARSE
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_bsr_row_ptr))
    call HIP_CHECK(hipFree(d_bsr_col_ind))
    call HIP_CHECK(hipFree(d_bsr_val))
    call HIP_CHECK(hipFree(d_B))
    call HIP_CHECK(hipFree(d_C))

end program example_fortran_bsrmm
