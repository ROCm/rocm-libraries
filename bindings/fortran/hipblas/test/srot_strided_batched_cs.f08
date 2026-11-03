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

! Checks hipblasSrotStridedBatched's `c` and `s`: one cosine and one sine shared
! by all batch entries, read per the pointer mode (as in rocBLAS), with x and y
! as device arrays holding one vector per column (x(n, batch)):
!   1. host pointer mode, c and s real variables      -> X_strided_rank
!   2. device pointer mode, c and s type(c_ptr) to
!      device scalars                                 -> X_strided_rank_devptr
! against x' = c*x + s*y, y' = c*y - s*x for every batch entry.
program hipblas_srot_strided_batched_cs_test

    use iso_c_binding
    use hip
    use hipblas

    implicit none

    integer(c_int), parameter :: n = 50, batch = 3
    integer(c_int64_t), parameter :: stride = n
    real(c_float), parameter :: error_max = 10 * epsilon(1.0)

    real(c_float), target :: hx(n,batch), hy(n,batch), rx(n,batch), ry(n,batch)
    real(c_float), target :: c, s
    real(c_float), pointer :: dx(:,:) => null(), dy(:,:) => null()
    real(c_float), pointer :: dc => null(), ds => null()
    type(c_ptr) :: handle
    integer :: i, b, mode

    write(*,"(a)",advance="no") "-- Running test 'srot_strided_batched_cs' (Fortran 2008 interfaces) - "

    call hipblasCheck(hipblasCreate(handle))
    call hipCheck(hipMalloc(dx, n, batch))
    call hipCheck(hipMalloc(dy, n, batch))

    do mode = 1, 2
        ! a different angle in each mode, so a stale result cannot pass
        c = cos(0.3 * mode)
        s = sin(0.3 * mode)
        do b = 1, batch
            do i = 1, n
                hx(i,b) = real(i + b, c_float) / 10.0
                hy(i,b) = real(b - i, c_float) / 7.0
            end do
        end do
        rx = c * hx + s * hy
        ry = c * hy - s * hx
        call hipCheck(hipMemcpy(dx, hx, hipMemcpyHostToDevice))
        call hipCheck(hipMemcpy(dy, hy, hipMemcpyHostToDevice))

        if (mode == 1) then
            call hipblasCheck(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST))
            call hipblasCheck(hipblasSrotStridedBatched(handle, n, dx, 1, stride, dy, 1, stride, c, s, batch))
        else
            call hipCheck(hipMalloc(dc, source=c))
            call hipCheck(hipMalloc(ds, source=s))
            call hipblasCheck(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE))
            call hipblasCheck(hipblasSrotStridedBatched(handle, n, dx, 1, stride, dy, 1, stride, &
                c_loc(dc), c_loc(ds), batch))
        end if
        call hipCheck(hipDeviceSynchronize())
        call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))
        call hipCheck(hipMemcpy(hy, dy, hipMemcpyDeviceToHost))

        if (maxval(abs(hx - rx)) > error_max * maxval(abs(rx)) .or. &
            maxval(abs(hy - ry)) > error_max * maxval(abs(ry))) then
            write(*,*) "FAILED! rotated vectors differ in ", merge("host  ", "device", mode == 1), &
                " pointer mode: x(1,:) = ", hx(1,:), " expected ", rx(1,:)
            call exit(1)
        end if
    end do

    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    call hipCheck(hipFree(dc))
    call hipCheck(hipFree(ds))
    call hipblasCheck(hipblasDestroy(handle))
    call hipCheck(hipDeviceReset())

    write(*,*) "PASSED!"

end program hipblas_srot_strided_batched_cs_test
