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

! Checks the per-batch `result` of the strided-batched reductions, an array of
! batch_count values read per the pointer mode:
!   rocblas_ddot_strided_batched
!     1. host pointer mode, result(batch) a host array   -> X_strided_rank
!     2. device pointer mode, result a device array
!        (real(c_double), pointer :: dres(:))            -> X_strided_rank
!     3. device pointer mode, result a type(c_ptr)       -> X_strided_rank_devptr
!   rocblas_isamax_strided_batched (integer results), the same three ways.
! Each batch entry has its own expected value, so a result written to the wrong
! place or only once is caught.
program rocblas_ddot_strided_batched_result_test

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer(c_int), parameter :: n = 100, batch = 4
    integer(c_int64_t), parameter :: stride = n
    real(c_double), parameter :: error_max = 100 * epsilon(1.d0)

    real(c_double), allocatable, target :: hx(:,:), hy(:,:)
    real(c_double), target :: res(batch), ref(batch)
    real(c_double), pointer :: dx(:,:) => null(), dy(:,:) => null(), dres(:) => null()
    real(c_float), allocatable, target :: hs(:,:)
    real(c_float), pointer :: ds(:,:) => null()
    integer(c_int), target :: ires(batch), iref(batch)
    integer(c_int), pointer :: dires(:) => null()
    type(c_ptr) :: handle
    integer :: i, b

    write(*,"(a)",advance="no") "-- Running test 'ddot_strided_batched_result' (Fortran 2008 interfaces) - "

    call rocblasCheck(rocblas_create_handle(handle))

    ! x(:,b) = 1, y(:,b) = b * i  =>  dot_b = b * n(n+1)/2
    allocate(hx(n,batch), hy(n,batch), hs(n,batch))
    do b = 1, batch
        do i = 1, n
            hx(i,b) = 1.d0
            hy(i,b) = dble(b * i)
            hs(i,b) = real(mod(i, 7), c_float)
        end do
        ref(b) = dble(b) * n * (n + 1) / 2.d0
        ! the largest |s| of batch b is at index 10*b (1-based), first occurrence
        hs(10*b, b) = -50.0 - b
        iref(b) = 10*b
    end do
    call hipCheck(hipMalloc(dx, source=hx))
    call hipCheck(hipMalloc(dy, source=hy))
    call hipCheck(hipMalloc(ds, source=hs))
    call hipCheck(hipMalloc(dres, batch))
    call hipCheck(hipMalloc(dires, batch))

    ! ---- ddot ----
    ! 1. host pointer mode, host array
    call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))
    res = 0.d0
    call rocblasCheck(rocblas_ddot_strided_batched(handle, n, dx, 1, stride, dy, 1, stride, batch, res))
    call check_real(res, "host mode, host array")

    ! 2. device pointer mode, device array through the array overload
    call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device))
    res = 0.d0
    call hipCheck(hipMemcpy(dres, res, hipMemcpyHostToDevice))
    call rocblasCheck(rocblas_ddot_strided_batched(handle, n, dx, 1, stride, dy, 1, stride, batch, dres))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(res, dres, hipMemcpyDeviceToHost))
    call check_real(res, "device mode, device array")

    ! 3. device pointer mode, type(c_ptr) (the _devptr twin)
    res = 0.d0
    call hipCheck(hipMemcpy(dres, res, hipMemcpyHostToDevice))
    call rocblasCheck(rocblas_ddot_strided_batched(handle, n, dx, 1, stride, dy, 1, stride, batch, c_loc(dres)))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(res, dres, hipMemcpyDeviceToHost))
    call check_real(res, "device mode, type(c_ptr)")

    ! ---- isamax ----
    ! 1. host pointer mode, host integer array
    call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))
    ires = 0
    call rocblasCheck(rocblas_isamax_strided_batched(handle, n, ds, 1, stride, batch, ires))
    call check_int(ires, "host mode, host array")

    ! 2. device pointer mode, device integer array through the array overload
    call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device))
    ires = 0
    call hipCheck(hipMemcpy(dires, ires, hipMemcpyHostToDevice))
    call rocblasCheck(rocblas_isamax_strided_batched(handle, n, ds, 1, stride, batch, dires))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(ires, dires, hipMemcpyDeviceToHost))
    call check_int(ires, "device mode, device array")

    ! 3. device pointer mode, type(c_ptr)
    ires = 0
    call hipCheck(hipMemcpy(dires, ires, hipMemcpyHostToDevice))
    call rocblasCheck(rocblas_isamax_strided_batched(handle, n, ds, 1, stride, batch, c_loc(dires)))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(ires, dires, hipMemcpyDeviceToHost))
    call check_int(ires, "device mode, type(c_ptr)")

    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    call hipCheck(hipFree(ds))
    call hipCheck(hipFree(dres))
    call hipCheck(hipFree(dires))
    deallocate(hx, hy, hs)
    call rocblasCheck(rocblas_destroy_handle(handle))
    call hipCheck(hipDeviceReset())

    write(*,*) "PASSED!"

contains

    subroutine check_real(r, what)
        real(c_double), intent(in) :: r(batch)
        character(*), intent(in) :: what
        if (maxval(abs(r - ref) / abs(ref)) > error_max) then
            write(*,*) "FAILED! ddot_strided_batched, ", what, ": ", r, " expected ", ref
            call exit(1)
        end if
    end subroutine

    subroutine check_int(r, what)
        integer(c_int), intent(in) :: r(batch)
        character(*), intent(in) :: what
        if (any(r /= iref)) then
            write(*,*) "FAILED! isamax_strided_batched, ", what, ": ", r, " expected ", iref
            call exit(1)
        end if
    end subroutine

end program rocblas_ddot_strided_batched_result_test
