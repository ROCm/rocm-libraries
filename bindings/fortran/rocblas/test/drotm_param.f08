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

! Checks rocblas_drotm's 5-value `param` = [flag, h11, h21, h12, h22], read
! per the pointer mode, with x and y as device arrays:
!   1. host pointer mode, param(5) a host array      -> X_rank_1
!   2. device pointer mode, param a type(c_ptr) to
!      device memory                                 -> X_rank_1_devptr
! for flag = -1 (full H), 0 (unit diagonal), 1 (h12 = 1, h21 = -1) and
! -2 (identity), each against the host formula
!   x' = h11*x + h12*y,  y' = h21*x + h22*y.
program rocblas_drotm_param_test

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer(c_int), parameter :: n = 64
    real(c_double), parameter :: error_max = 10 * epsilon(1.d0)
    real(c_double), parameter :: flags(4) = [-1.d0, 0.d0, 1.d0, -2.d0]

    real(c_double), target :: hx(n), hy(n), rx(n), ry(n), param(5)
    real(c_double), pointer :: dx(:) => null(), dy(:) => null(), dparam(:) => null()
    type(c_ptr) :: handle
    integer :: i, f, mode

    write(*,"(a)",advance="no") "-- Running test 'drotm_param' (Fortran 2008 interfaces) - "

    call rocblasCheck(rocblas_create_handle(handle))
    call hipCheck(hipMalloc(dx, n))
    call hipCheck(hipMalloc(dy, n))
    call hipCheck(hipMalloc(dparam, 5))

    do mode = 1, 2
        do f = 1, size(flags)
            do i = 1, n
                hx(i) = dble(i) / 3.d0
                hy(i) = 1.d0 - dble(i) / 5.d0
            end do
            ! the entries the flag ignores are set to garbage, to catch their use
            param = [flags(f), 2.d0, -0.5d0, 0.25d0, 3.d0]
            call reference(param, hx, hy, rx, ry)
            call hipCheck(hipMemcpy(dx, hx, hipMemcpyHostToDevice))
            call hipCheck(hipMemcpy(dy, hy, hipMemcpyHostToDevice))

            if (mode == 1) then
                call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))
                call rocblasCheck(rocblas_drotm(handle, n, dx, 1, dy, 1, param))
            else
                call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device))
                call hipCheck(hipMemcpy(dparam, param, hipMemcpyHostToDevice))
                call rocblasCheck(rocblas_drotm(handle, n, dx, 1, dy, 1, c_loc(dparam)))
            end if
            call hipCheck(hipDeviceSynchronize())
            call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))
            call hipCheck(hipMemcpy(hy, dy, hipMemcpyDeviceToHost))

            if (maxval(abs(hx - rx)) > error_max * maxval(abs(rx)) .or. &
                maxval(abs(hy - ry)) > error_max * maxval(abs(ry))) then
                write(*,*) "FAILED! drotm with flag ", flags(f), " in ", &
                    merge("host  ", "device", mode == 1), " pointer mode: x(1:2) = ", hx(1:2), &
                    " expected ", rx(1:2), " y(1:2) = ", hy(1:2), " expected ", ry(1:2)
                call exit(1)
            end if
        end do
    end do

    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    call hipCheck(hipFree(dparam))
    call rocblasCheck(rocblas_destroy_handle(handle))
    call hipCheck(hipDeviceReset())

    write(*,*) "PASSED!"

contains

    ! The modified Givens rotation of the BLAS reference drotm
    subroutine reference(p, x, y, xo, yo)
        real(c_double), intent(in) :: p(5), x(n), y(n)
        real(c_double), intent(out) :: xo(n), yo(n)
        real(c_double) :: h11, h21, h12, h22
        select case (nint(p(1)))
        case (-1)
            h11 = p(2); h21 = p(3); h12 = p(4); h22 = p(5)
        case (0)
            h11 = 1.d0; h21 = p(3); h12 = p(4); h22 = 1.d0
        case (1)
            h11 = p(2); h21 = -1.d0; h12 = 1.d0; h22 = p(5)
        case default
            h11 = 1.d0; h21 = 0.d0; h12 = 0.d0; h22 = 1.d0
        end select
        xo = h11 * x + h12 * y
        yo = h21 * x + h22 * y
    end subroutine

end program rocblas_drotm_param_test
