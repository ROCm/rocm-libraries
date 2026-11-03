!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2021 Advanced Micro Devices, Inc. All rights Reserved.
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

program example_fortran_gtsv
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

    real(8), target :: h_dl(10), h_d(10), h_du(10), h_B(20)

    type(c_ptr) :: d_dl
    type(c_ptr) :: d_d
    type(c_ptr) :: d_du
    type(c_ptr) :: d_B
    type(c_ptr) :: temp_buffer

    integer :: i, j, k
    integer(c_int) :: M, N, ldb
    integer(c_int) :: stat
    integer(c_size_t), target :: buffer_size

    type(c_ptr) :: handle

    integer :: version

    character(len=12) :: rev

!   Input data

!       (  1  1  0  0  0  0  0  0  0  0  )
!       (  1  1  1  0  0  0  0  0  0  0  )
!       (  0  1  1  1  0  0  0  0  0  0  )
!       (  0  0  1  1  1  0  0  0  0  0  )
!   A = (  0  0  0  1  1  1  0  0  0  0  )
!       (  0  0  0  0  1  1  1  0  0  0  )
!       (  0  0  0  0  0  1  1  1  0  0  )
!       (  0  0  0  0  0  0  1  1  1  0  )
!       (  0  0  0  0  0  0  0  1  1  1  )
!       (  0  0  0  0  0  0  0  0  1  1  )
!
!       (  1  1  )
!       (  2  1  )
!       (  4  1  )
!       (  4  0  )
!   B = (  4  1  )
!       (  6  0  )
!       (  4  5  )
!       (  2  3  )
!       (  1  2  )
!       (  1  4  )

!   Number of rows and columns
    M = 10
    N = 2
    ldb = M

!   Fill tri-diagonal arrays
    h_dl = (/0, 1, 1, 1, 1, 1, 1, 1, 1, 1/)
    h_d  = (/1, 1, 1, 1, 1, 1, 1, 1, 1, 1/)
    h_du = (/1, 1, 1, 1, 1, 1, 1, 1, 1, 0/)
    h_B  = (/1, 2, 4, 4, 4, 6, 4, 2, 1, 1, 1, 1, 1, 0, 1, 0, 5, 3, 2, 4/)

!   Allocate device memory
    call HIP_CHECK(hipMalloc(d_dl, int(M, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_d, int(M, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_du, int(M, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(d_B, int(M, c_size_t) * int(N, c_size_t) * 8))

!   Copy host data to device
    call HIP_CHECK(hipMemcpy(d_dl, c_loc(h_dl), int(M, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_d, c_loc(h_d), int(M, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_du, c_loc(h_du), int(M, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(d_B, c_loc(h_B), int(M, c_size_t) * int(N, c_size_t) * 8, 1))

!   Create rocSPARSE handle
    call ROCSPARSE_CHECK(rocsparse_create_handle(handle))

!   Get rocSPARSE version
    call ROCSPARSE_CHECK(rocsparse_get_version(handle, version))
    call ROCSPARSE_CHECK(rocsparse_get_git_rev(handle, rev))

!   Print version on screen
    write(*,fmt='(A,I0,A,I0,A,I0,A,A)') 'rocSPARSE version: ', version / 100000, '.', &
        mod(version / 100, 1000), '.', mod(version, 100), '-', rev

!   Obtain required buffer size
    call ROCSPARSE_CHECK(rocsparse_dgtsv_buffer_size(handle, &
                                                       M, &
                                                       N, &
                                                       d_dl, &
                                                       d_d, &
                                                       d_du, &
                                                       d_B, &
                                                       ldb, &
                                                       c_loc(buffer_size)))

!   Allocate temporary buffer
    write(*,fmt='(A,I0,A)') 'Allocating ', buffer_size / 1024, 'kB temporary storage buffer'

    call HIP_CHECK(hipMalloc(temp_buffer, buffer_size))

!   Call dgtsv to perform tri-diagonal solve
    call ROCSPARSE_CHECK(rocsparse_dgtsv(handle, &
                                                M, &
                                                N, &
                                                d_dl, &
                                                d_d, &
                                                d_du, &
                                                d_B, &
                                                ldb, &
                                                temp_buffer))

!   Print result
    call HIP_CHECK(hipMemcpy(c_loc(h_B), d_B, int(M, c_size_t) * int(N, c_size_t) * 8, 2))

!    write(*,fmt='(A)') "Tri-diagonal solve:"
    do i = 1, N
        do j = 1, M
            write(*,fmt='(A,I0,A,F0.2)') 'B(', j, ') = ', h_B(ldb * (i - 1) + j)
        end do
        write(*,*)
    end do

!   Clear rocSPARSE
    call ROCSPARSE_CHECK(rocsparse_destroy_handle(handle))

!   Clear device memory
    call HIP_CHECK(hipFree(d_dl))
    call HIP_CHECK(hipFree(d_d))
    call HIP_CHECK(hipFree(d_du))
    call HIP_CHECK(hipFree(d_B))
    call HIP_CHECK(hipFree(temp_buffer))

end program example_fortran_gtsv
