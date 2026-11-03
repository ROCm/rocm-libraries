!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

program rocblas_sdot_test

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer, parameter :: N = 10240

    real(c_float),allocatable,target,dimension(:) :: hx
    real(c_float),allocatable,target,dimension(:) :: hy
    real(c_float),target :: res

    real(c_float),pointer,dimension(:) :: dx
    real(c_float),pointer,dimension(:) :: dy

    real(c_float) :: res_exact
    real(c_float) :: error
    real(c_float), parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    write(*,"(a)",advance="no") "-- Running test 'sdot' (Fortran 2008 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, rocblas_pointer_mode_host)) ! host pointer mode

    ! Allocate host-side memory
    allocate(hx(N))
    allocate(hy(N))

    ! Initialize host memory
    hx = 1.0                                          ! x = 1
    hy = 2.0                                          ! y = 2
    res_exact = 2.0 * N                              ! sum(x*y) = 2n

    ! Allocate device-side memory
    ! Transfer data from host to device memory
    call hipCheck(hipMalloc(dx, source=hx))
    call hipCheck(hipMalloc(dy, source=hy))

    ! Call rocblas function
    res = 0.0
    call rocblasCheck(rocblas_sdot(rocblas_handle, N, dx, 1, dy, 1, c_loc(res)))
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

end program rocblas_sdot_test
