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

program hip_snrm2
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! nrm2(x) with x = 1 over n elements, so the result is sqrt(n).
  integer, parameter :: n = 1024
  real(c_float), allocatable, dimension(:) :: hx
  real(c_float), target :: res
  real(c_float) :: res_exact, error
  real(c_float), parameter :: error_max = 10*epsilon(error)
  real(c_float), pointer, dimension(:) :: dx
  type(c_ptr) :: handle = c_null_ptr

  write(*,"(a)",advance="no") "-- Running test 'snrm2' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(hx(n))
  hx = 1.0
  res_exact = sqrt(real(n, kind=kind(res_exact)))

  call hipCheck(hipMalloc(dx, shape(hx)))
  call hipCheck(hipMemcpy(dx, hx, hipMemcpyHostToDevice))

  res = 0.0
  call hipblasCheck(hipblasSnrm2(handle, n, dx, 1, c_loc(res)))
  call hipCheck(hipDeviceSynchronize())

  error = abs((res_exact - res) / res_exact)
  if (error > error_max) then
    write(*,*) "FAILED! error = ", error, " result = ", res
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipblasCheck(hipblasDestroy(handle))

  deallocate(hx)

  write(*,*) "PASSED!"

end program hip_snrm2
