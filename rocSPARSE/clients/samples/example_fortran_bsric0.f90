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

program example_fortran_bsric0
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

    integer, target :: h_bsr_row_ptr(4), h_bsr_col_ind(9)
    real(8), target :: h_bsr_val(36)

    type(c_ptr) :: d_bsr_row_ptr
    type(c_ptr) :: d_bsr_col_ind
    type(c_ptr) :: d_bsr_val
    type(c_ptr) :: temp_buffer

    integer :: i, j, k, s, t
    integer(c_int) :: Mb, Nb, nnzb, block_dim
    integer(c_int) :: dir
    integer(c_int) :: stat
    integer(c_int), target :: pivot
    integer(c_size_t), target :: buffer_size

    type(c_ptr) :: handle
    type(c_ptr) :: descr
    type(c_ptr) :: info

    integer :: version

    character(len=12) :: rev

!   Input data

!       (  3  0 -1 -1  0 -1 )
!       (  0  2  0 -1  0  0 )
!       ( -1  0  3  0 -1  0 )
!   A = ( -1 -1  0  2  0 -1 )
!       (  0  0 -1  0  3 -1 )
!       ( -1  0  0 -1 -1  4 )

!   Number of rows and columns
    Mb = 3
    Nb = 3

!   Number of non-zero entries
    nnzb = 9

!   BSR block dimension
    block_dim = 2

!   BSR block direction
    dir = 0

!   Fill BSR structure
    h_bsr_row_ptr = (/0, 3, 6, 9/)
    h_bsr_col_ind = (/0, 1, 2, 0, 1, 2, 0, 1, 2/)
    h_bsr_val     = (/3, 0, 0, 2, -1, -1, 0, -1, 0, -1, 0, 0, &
                     -1, 0, -1, -1, 3, 0, 0, 2, -1, 0, 0, -1, &
                      0, 0, -1, 0, -1, 0, 0, -1, 3, -1, -1, 4/)

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_bsr_row_ptr, (int(Mb, c_size_t) + 1) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_col_ind, int(nnzb, c_size_t) * 4))
    call HIP_CHECK(hipMalloc(d_bsr_val, int(nnzb * block_dim * block_dim, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_bsr_row_ptr, c_loc(h_bsr_row_ptr), (int(Mb, c_size_t) + 1) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_bsr_col_ind, c_loc(h_bsr_col_ind), int(nnzb, c_size_t) * 4, 1))
    call HIP_CHECK(hipMemcpy(d_bsr_val, c_loc(h_bsr_val), int(nnzb * block_dim * block_dim, c_size_t) * 8, 1))

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

!   Create matrix info structure
    call ROCSPARSE_CHECK(rocsparse_create_mat_info(info))

!   Obtain required buffer size
    call ROCSPARSE_CHECK(rocsparse_dbsric0_buffer_size(handle, &
                                                       dir, &
                                                       Mb, &
                                                       nnzb, &
                                                       descr, &
                                                       d_bsr_val, &
                                                       d_bsr_row_ptr, &
                                                       d_bsr_col_ind, &
                                                       block_dim, &
                                                       info, &
                                                       c_loc(buffer_size)))

!   Allocate temporary buffer
    write(*,fmt='(A,I0,A)') 'Allocating ', buffer_size / 1024, 'kB temporary storage buffer'

    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

!   Perform analysis step
    call ROCSPARSE_CHECK(rocsparse_dbsric0_analysis(handle, &
                                                    dir, &
                                                    Mb, &
                                                    nnzb, &
                                                    descr, &
                                                    d_bsr_val, &
                                                    d_bsr_row_ptr, &
                                                    d_bsr_col_ind, &
                                                    block_dim, &
                                                    info, &
                                                    rocsparse_analysis_policy_reuse, &
                                                    rocsparse_solve_policy_auto, &
                                                    temp_buffer))

!   Call dbsric0 to perform incomplete Cholesky factorization
    call ROCSPARSE_CHECK(rocsparse_dbsric0(handle, &
                                           dir, &
                                           Mb, &
                                           nnzb, &
                                           descr, &
                                           d_bsr_val, &
                                           d_bsr_row_ptr, &
                                           d_bsr_col_ind, &
                                           block_dim, &
                                           info, &
                                           rocsparse_solve_policy_auto, &
                                           temp_buffer))

!   Check for zero pivots
    stat = rocsparse_bsric0_zero_pivot(handle, info, c_loc(pivot))

    if(stat .eq. rocsparse_status_zero_pivot) then
        write(*,fmt='(A,I0)') 'Found zero pivot in matrix row ', pivot
    end if

!   Print result
    call HIP_CHECK(hipMemcpy(c_loc(h_bsr_val), d_bsr_val, int(nnzb * block_dim * block_dim, c_size_t) * 8, 2))

    write(*,fmt='(A)') "Incomplete Cholesky factorization:"
    do i = 1, Mb
        do s = 1, block_dim
            k = h_bsr_row_ptr(i) + 1
            do j = 1, i
                if(j .eq. h_bsr_col_ind(k) + 1)  then
                    do t = 1, block_dim
                        if(dir .eq. 0) then
                            write(*,fmt='(A,F6.2)',advance='no') ' ', h_bsr_val(block_dim * block_dim * (k - 1) + & 
                                                                                block_dim * (s - 1) + t)
                        else
                            write(*,fmt='(A,F6.2)',advance='no') ' ', h_bsr_val(block_dim * block_dim * (k - 1) + &
                                                                                block_dim * (t - 1) + s)
                        end if
                    end do
                    k = k + 1
                else
                    do t = 1, block_dim
                        write(*,fmt='(A,F6.2)',advance='no') ' ', 0.0
                    end do
                end if
            end do
            write(*,*)
        end do
    end do

!   Clear rocSPARSE
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_info(info))
    call ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descr))
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_bsr_row_ptr))
    call HIP_CHECK(hipFree(d_bsr_col_ind))
    call HIP_CHECK(hipFree(d_bsr_val))
    call HIP_CHECK(hipFree(temp_buffer))

end program example_fortran_bsric0
