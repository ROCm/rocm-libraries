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

program hip_sswap
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer :: n = 6
  integer :: i
  real(kind=4), allocatable, target, dimension(:) :: x, y, x_exact, y_exact
  real(kind=4), pointer, dimension(:) :: dx, dy
  type(c_ptr) :: handle = c_null_ptr

  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  allocate(x(n))
  allocate(y(n))
  allocate(x_exact(n))
  allocate(y_exact(n))

  do i = 1,n
    x(i) = i
    x_exact(i) = x(i)
    y(i) = 2 * i
    y_exact(i) = y(i)
  end do

  write(*,"(a)",advance="no") "-- Running test 'SSWAP' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx,shape(x)))
  call hipCheck(hipMalloc(dy,shape(y)))

  call hipCheck(hipMemcpy(dx, x, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, y, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasSswap(handle,n,dx,1,dy,1))

  call hipCheck(hipMemcpy(x, dx, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(y, dy, hipMemcpyDeviceToHost))

  do i = 1,n
    error = MAX(abs((y_exact(i) - x(i))/y_exact(i)), abs((x_exact(i) - y(i))/x_exact(i)))
      if( error > error_max )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error
        call exit(1)
      end if
  end do

  call hipblasCheck(hipblasDestroy(handle))

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  deallocate(x,y,x_exact,y_exact)

  write(*,*) "PASSED!"

end program hip_sswap
