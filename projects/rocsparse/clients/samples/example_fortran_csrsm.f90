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

program example_fortran_csrsm
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

    integer, target :: h_csr_row_ptr(4), h_csr_col_ind(6)
    real(8), target :: h_csr_val(6), h_B(12)

    type(c_ptr) :: d_csr_row_ptr
    type(c_ptr) :: d_csr_col_ind
    type(c_ptr) :: d_csr_val
    type(c_ptr) :: d_B
    type(c_ptr) :: temp_buffer

    integer :: i, j
    integer(c_int) :: M, N, nnz
    integer(c_int) :: ldb, nrhs
    integer(c_int) :: stat
    integer(c_int), target :: pivot
    integer(c_size_t), target :: buffer_size

    real(c_double), target :: alpha

    type(c_ptr) :: handle
    type(c_ptr) :: descr
    type(c_ptr) :: info

    integer :: version

    character(len=12) :: rev

!   Input data

!       ( 1 0 0 )
!   A = ( 2 3 0 )
!       ( 4 5 6 )

!   Number of rows and columns
    M = 3
    N = 3

!   Number of non-zero entries
    nnz = 6

!   Fill CSR structure
    h_csr_row_ptr = (/0, 1, 3, 6/)
    h_csr_col_ind = (/0, 0, 1, 0, 1, 2/)
    h_csr_val     = (/1, 2, 3, 4, 5, 6/)

!   Scalar alpha
    alpha = 1.0

!   B (nrhs rhs vectors)
    ldb  = m
    nrhs = 4
    h_B  = (/1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12/)

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_csr_row_ptr, (int(M, c_size_t) + 1) * 4))
    call HIP_CHECK(hipMalloc(d_csr_col_ind, int(nnz, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_csr_val, int(nnz, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_B, int(M * nrhs, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr), (int(M, c_size_t) + 1) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind), int(nnz, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_csr_val, c_loc(h_csr_val), int(nnz, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_B, c_loc(h_B), int(M * nrhs, c_size_t) * 8, 1))

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

!   Matrix fill mode
    call ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descr, rocsparse_fill_mode_lower))

!   Matrix diagonal type
    call ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descr, rocsparse_diag_type_non_unit))

!   Create matrix info structure
    call ROCSPARSE_CHECK(rocsparse_create_mat_info(info))

!   Obtain required buffer size
    call ROCSPARSE_CHECK(rocsparse_dcsrsm_buffer_size(handle, &
                                                      rocsparse_operation_none, &
                                                      rocsparse_operation_none, &
                                                      M, &
                                                      nrhs, &
                                                      nnz, &
                                                      c_loc(alpha), &
                                                      descr, &
                                                      d_csr_val, &
                                                      d_csr_row_ptr, &
                                                      d_csr_col_ind, &
                                                      d_B, &
                                                      ldb, &
                                                      info, &
                                                      rocsparse_solve_policy_auto, &
                                                      c_loc(buffer_size)))

!   Allocate temporary buffer
    write(*,fmt='(A,I0,A)') 'Allocating ', buffer_size / 1024, 'kB temporary storage buffer'

    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

!   Perform analysis step
    call ROCSPARSE_CHECK(rocsparse_dcsrsm_analysis(handle, &
                                                   rocsparse_operation_none, &
                                                   rocsparse_operation_none, &
                                                   M, &
                                                   nrhs, &
                                                   nnz, &
                                                   c_loc(alpha), &
                                                   descr, &
                                                   d_csr_val, &
                                                   d_csr_row_ptr, &
                                                   d_csr_col_ind, &
                                                   d_B, &
                                                   ldb, &
                                                   info, &
                                                   rocsparse_analysis_policy_reuse, &
                                                   rocsparse_solve_policy_auto, &
                                                   temp_buffer))

!   Call dcsrsm to perform lower triangular solve Ly = x
    call ROCSPARSE_CHECK(rocsparse_dcsrsm_solve(handle, &
                                                rocsparse_operation_none, &
                                                rocsparse_operation_none, &
                                                M, &
                                                nrhs, &
                                                nnz, &
                                                c_loc(alpha), &
                                                descr, &
                                                d_csr_val, &
                                                d_csr_row_ptr, &
                                                d_csr_col_ind, &
                                                d_B, &
                                                ldb, &
                                                info, &
                                                rocsparse_solve_policy_auto, &
                                                temp_buffer))

!   Check for zero pivots
    stat = rocsparse_csrsm_zero_pivot(handle, info, c_loc(pivot))

    if(stat .eq. rocsparse_status_zero_pivot) then
        write(*,fmt='(A,I0)') 'Found zero pivot in matrix row ', pivot
    end if

!   Print result
    call HIP_CHECK(hipMemcpy(c_loc(h_B), d_B, int(m * nrhs, c_size_t) * 8, 2))

    do i = 0, nrhs - 1
        write(*,fmt='(A,I0,A)', advance='no') 'B(', i + 1, '):'
        do j = 1, M
            write(*,fmt='(A,F0.5)', advance='no') ' ', h_B(i * ldb + j)
        end do
        write(*,*)
    end do

!   Clear rocSPARSE
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_csr_row_ptr))
    call HIP_CHECK(hipFree(d_csr_col_ind))
    call HIP_CHECK(hipFree(d_csr_val))
    call HIP_CHECK(hipFree(d_B))
    call HIP_CHECK(hipFree(temp_buffer))

end program example_fortran_csrsm
