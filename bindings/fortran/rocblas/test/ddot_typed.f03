!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
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

! The same check as ddot.f03, through the _typed form: the result is a Fortran
! variable rather than c_loc() of one, so the generic resolves to the _typed
! module procedure, which hands the C API its address. The handle is in host
! pointer mode, as that form requires.
program rocblas_ddot_test_typed

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer, parameter :: N = 10240
    integer, parameter :: bytes_per_element = 8 ! double precision
    integer(c_size_t), parameter :: Nbytes = N * bytes_per_element

    type(c_ptr) :: dx = c_null_ptr
    type(c_ptr) :: dy = c_null_ptr

    double precision,allocatable,target,dimension(:) :: hx
    double precision,allocatable,target,dimension(:) :: hy
    double precision :: res

    double precision :: res_exact
    double precision :: error
    double precision, parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    write(*,"(a)",advance="no") "-- Running test 'ddot typed' (Fortran 2003 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, rocblas_pointer_mode_host)) ! host pointer mode

    ! Allocate host-side memory
    allocate(hx(N))
    allocate(hy(N))

    ! Initialize host memory
    hx = 1.d0                                         ! x = 1
    hy = 2.d0                                         ! y = 2
    res_exact = 2.d0 * N                             ! sum(x*y) = 2n

    ! Allocate device-side memory
    call hipCheck(hipMalloc(dx, Nbytes))
    call hipCheck(hipMalloc(dy, Nbytes))

    ! Transfer data from host to device memory
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(dy, c_loc(hy(1)), Nbytes, hipMemcpyHostToDevice))

    ! Call rocblas function
    res = 0.d0
    call rocblasCheck(rocblas_ddot(rocblas_handle, N, dx, 1, dy, 1, res))
    call hipCheck(hipDeviceSynchronize()) ! res now valid host-side

    ! Verification
    error = abs((res_exact - res) / res_exact)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " result = ", res
        call exit(1)
    end if

    ! Cleanup
    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    deallocate(hx, hy)
    call rocblasCheck(rocblas_destroy_handle(rocblas_handle))

    write(*,*) "PASSED!"

end program rocblas_ddot_test_typed
