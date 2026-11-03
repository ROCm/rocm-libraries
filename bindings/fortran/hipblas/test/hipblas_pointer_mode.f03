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

! Exercises hipblasGetPointerMode / hipblasSetPointerMode on a handle created
! with hipblasCreate, checking both that the mode round-trips through the
! getter and that hipBLAS keeps computing the right answer across the switch:
!
!   1. The freshly created handle must report HIPBLAS_POINTER_MODE_HOST.
!   2. In HOST mode, hipblasSdot writes its scalar result into host memory;
!      the value must match the exact dot product 408.0.
!   3. hipblasSetPointerMode(HIPBLAS_POINTER_MODE_DEVICE) must round-trip
!      through hipblasGetPointerMode.
!   4. In DEVICE mode, the very same hipblasSdot call writes into a device
!      scalar; copying it back must yield the same 408.0.
!   5. Switching back to HOST must again round-trip through the getter.
!
! The test fails (prints "FAILED! ..." and stops with status 1) if any query
! reports the wrong mode or either dot product deviates from the exact value.
program hipblas_pointer_mode
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(c_int), parameter :: n = 8
  integer(c_size_t), parameter :: nbytes = int(n, c_size_t) * 4_c_size_t
  integer(c_size_t), parameter :: sbytes = 4_c_size_t
  real(c_float), parameter :: res_exact = 408.0_c_float
  real(c_float), parameter :: tol = 1.0e-5_c_float

  real(c_float), target :: hx(n), hy(n)
  real(c_float), target :: res_host, res_dev
  integer(kind(HIPBLAS_POINTER_MODE_HOST)) :: mode
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr, dres = c_null_ptr
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hipBLAS pointer mode' &
                              &(Fortran 2003 interfaces) - "

  ! hx(i) = i, hy(i) = 2i, so dot(hx,hy) = 2 * sum(i**2) = 408 for n = 8.
  do i = 1, n
     hx(i) = real(i, c_float)
     hy(i) = 2.0_c_float * real(i, c_float)
  end do

  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx, nbytes))
  call hipCheck(hipMalloc(dy, nbytes))
  call hipCheck(hipMalloc(dres, sbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(hy(1)), nbytes, hipMemcpyHostToDevice))

  ! 1. A new handle defaults to host pointer mode. The 'mode' dummy argument is
  !    the enum itself, passed by reference, so the variable is handed over
  !    directly and hipBLAS writes the mode into it.
  mode = -1
  call hipblasCheck(hipblasGetPointerMode(handle, mode))
  if (mode /= HIPBLAS_POINTER_MODE_HOST) then
     write(*,*) "FAILED! fresh handle reported pointer mode ", mode, &
                " instead of HIPBLAS_POINTER_MODE_HOST"
     STOP 1
  end if

  ! 2. In host mode the result scalar lives in host memory.
  res_host = 0.0_c_float
  call hipblasCheck(hipblasSdot(handle, n, dx, 1, dy, 1, c_loc(res_host)))
  call hipCheck(hipDeviceSynchronize())
  if (abs(res_host - res_exact) > tol * res_exact) then
     write(*,*) "FAILED! host-mode sdot returned ", res_host, " expected ", res_exact
     STOP 1
  end if

  ! 3. Flip to device pointer mode and make sure the getter agrees.
  call hipblasCheck(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE))
  mode = -1
  call hipblasCheck(hipblasGetPointerMode(handle, mode))
  if (mode /= HIPBLAS_POINTER_MODE_DEVICE) then
     write(*,*) "FAILED! after hipblasSetPointerMode(DEVICE) the getter reported ", mode
     STOP 1
  end if

  ! 4. The identical call now has to write the scalar into device memory.
  res_dev = 0.0_c_float
  call hipblasCheck(hipblasSdot(handle, n, dx, 1, dy, 1, dres))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(res_dev), dres, sbytes, hipMemcpyDeviceToHost))
  if (abs(res_dev - res_exact) > tol * res_exact) then
     write(*,*) "FAILED! device-mode sdot returned ", res_dev, " expected ", res_exact
     STOP 1
  end if
  if (res_dev /= res_host) then
     write(*,*) "FAILED! device-mode result ", res_dev, " differs from host-mode result ", res_host
     STOP 1
  end if

  ! 5. Close the round-trip by restoring the host pointer mode.
  call hipblasCheck(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST))
  mode = -1
  call hipblasCheck(hipblasGetPointerMode(handle, mode))
  if (mode /= HIPBLAS_POINTER_MODE_HOST) then
     write(*,*) "FAILED! after restoring HIPBLAS_POINTER_MODE_HOST the getter reported ", mode
     STOP 1
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipCheck(hipFree(dres))
  call hipblasCheck(hipblasDestroy(handle))

  write(*,*) "PASSED! pointer mode round-trips HOST/DEVICE, sdot = ", res_dev

end program hipblas_pointer_mode
