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

program hip_idamax
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! The largest |x| sits at index 5, and hipBLAS returns a 1-based index.
  integer, parameter :: n = 8
  real(c_double), allocatable, dimension(:) :: hx
  integer(c_int), target :: res
  integer(c_int), parameter :: res_exact = 5
  real(c_double), pointer, dimension(:) :: dx
  type(c_ptr) :: handle = c_null_ptr

  write(*,"(a)",advance="no") "-- Running test 'idamax' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(hx(n))
  hx = 1.0
  hx(5) = 10.0

  call hipCheck(hipMalloc(dx, shape(hx)))
  call hipCheck(hipMemcpy(dx, hx, hipMemcpyHostToDevice))

  res = 0
  call hipblasCheck(hipblasIdamax(handle, n, dx, 1, c_loc(res)))
  call hipCheck(hipDeviceSynchronize())

  if (res /= res_exact) then
    write(*,*) "FAILED! wrong index: result = ", res, " expected ", res_exact
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipblasCheck(hipblasDestroy(handle))

  deallocate(hx)

  write(*,*) "PASSED!"

end program hip_idamax
