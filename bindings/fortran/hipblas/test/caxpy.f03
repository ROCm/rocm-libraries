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

program hip_caxpy
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer :: n = 6
  type(c_ptr) :: handle = c_null_ptr
  integer :: j
  complex(kind=4), allocatable, target, dimension(:) :: x, y, y_exact

  complex(kind=4), target :: alpha = (2.0, 0.)
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr

  integer, parameter :: bytes_per_element = 8

  integer(c_size_t) :: Nxbytes
  integer(c_size_t) :: Nybytes

  real(kind=4) :: error
  real(kind=4), parameter :: error_max = 10*epsilon(error)

  Nxbytes = n * bytes_per_element
  Nybytes = n * bytes_per_element
  allocate(x(n))
  allocate(y(n))
  allocate(y_exact(n))

  do j = 1, n
    x(j) = cmplx(j, 0)
    y(j) = cmplx(j, 0)
  end do

  do j = 1, n
    y_exact(j) = alpha*x(j) + y(j)
  end do

  write(*,"(a)",advance="no") "-- Running test 'CAXPY' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx, Nxbytes))
  call hipCheck(hipMalloc(dy, Nybytes))

  call hipCheck(hipMemcpy(dx, c_loc(x(1)), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y(1)), Nybytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasCaxpy(handle, n, c_loc(alpha), dx, 1, dy, 1))

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(y(1)), dy, Nybytes, hipMemcpyDeviceToHost))

  do j = 1, n
    error = abs((y_exact(j) - y(j))/y_exact(j))
    if (error > error_max) then
      write(*,*) "FAILED! Error bigger than max! Error = ", error
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(x)
  deallocate(y)
  deallocate(y_exact)

  write(*,*) "PASSED!"

end program hip_caxpy
