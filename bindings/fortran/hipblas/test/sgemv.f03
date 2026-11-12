!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

program hip_sgemv
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer :: m = 6
  integer :: n = 5
  integer :: i, j

  real, target :: alpha = 1.0
  real, target :: beta = 0.0

  type(c_ptr) :: handle = c_null_ptr

  real(kind=4), allocatable, target, dimension(:) :: a, x, y

  type(c_ptr) :: da = c_null_ptr, dx = c_null_ptr, dy = c_null_ptr

  integer(c_size_t) :: Nabytes, Nxbytes, Nybytes
  integer, parameter :: bytes_per_element = 4 !float precision

  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  Nabytes = m * n * bytes_per_element
  Nxbytes = n * bytes_per_element
  Nybytes = m * bytes_per_element

  allocate(x(n))
  allocate(y(m))
  allocate(a(m*n))

    a(:) = 1.0
    x(:) = 1.0
    y(:) = 1.0

  write(*,"(a)",advance="no") "-- Running test 'SGEMV' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx,Nxbytes))
  call hipCheck(hipMalloc(dy,Nybytes))
  call hipCheck(hipMalloc(da,Nabytes))

  call hipCheck(hipMemcpy(da, c_loc(a(1)), Nabytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx, c_loc(x(1)), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y(1)), Nybytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasSgemv(handle,HIPBLAS_OP_N,m,n,c_loc(alpha),da,m,dx,1,c_loc(beta),dy,1))

  call hipCheck(hipMemcpy(c_loc(y(1)), dy, Nybytes, hipMemcpyDeviceToHost))

  do i = 1,m
    error = abs(5.0 - y(i))
      if( error > 10*epsilon(error) )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, "y(i) = ", y(i)
        call exit(1)
     end if
  end do

  call hipblasCheck(hipblasDestroy(handle))

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  deallocate(a)
  deallocate(x)
  deallocate(y)

  write(*,*) "PASSED!"

end program hip_sgemv
