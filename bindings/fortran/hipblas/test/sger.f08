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

! HIPBLAS, i.e. cuBLAS and rocBLAS, assumes column-major matrix memory layouts.
! Hence, no matrix must be transposed when interfacing with Fortran.
program hip_sger
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer :: m = 6
  integer :: n = 5
  real, target :: alpha = 2.0
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j

  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  real(kind=4), allocatable, target, dimension(:,:) :: a
  real(kind=4), allocatable, target, dimension(:) :: x, y

  type(c_ptr) :: da = c_null_ptr, dx = c_null_ptr, dy = c_null_ptr

  integer(c_size_t) :: Nabytes, Nxbytes, Nybytes
  integer, parameter :: bytes_per_element = 4 !float precision

  Nxbytes = m * bytes_per_element
  Nybytes = n * bytes_per_element
  Nabytes = m * n * bytes_per_element

  allocate(x(m))
  allocate(y(n))
  allocate(a(m,n))

  do i = 1,m
    do j = 1,n
      a(i,j) = 1.0
    end do
  end do

  do i = 1,m
    x(i) = 1.0
  end do

  do i = 1,n
    y(i) = 1.0
  end do


  write(*,"(a)",advance="no") "-- Running test 'SGER' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx,Nxbytes))
  call hipCheck(hipMalloc(dy,Nybytes))
  call hipCheck(hipMalloc(da,Nabytes))

  !call hipCheck(hipblasSetMatrix(m,n,bytes_per_element,a,m,da,m))
  !call hipCheck(hipblasSetVector(m,bytes_per_element,x,1,dx,1))
  !call hipCheck(hipblasSetVector(n,bytes_per_element,y,1,dy,1))

  call hipCheck(hipMemcpy(da, c_loc(a), Nabytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx, c_loc(x), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y), Nybytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasSger(handle,m,n,c_loc(alpha),dx,1,dy,1,da,m))

  !call hipCheck(hipblasGetMatrix(m,n,bytes_per_element,da,m,a,m));
  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(a), da, Nabytes, hipMemcpyDeviceToHost))

  !do i=1,m
  !  do j = 1,n
  !    write(*,*) a(i,j)
  !  end do
  !end do

  do i = 1,m
  do j = 1,n
    error = abs(3.0 - a(i,j))
      if( error > 10*epsilon(error) )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, "a(i,j) = ", a(i,j)
        call exit(1)
     end if
  end do
  end do

  call hipblasCheck(hipblasDestroy(handle))

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  deallocate(a)
  deallocate(x)
  deallocate(y)

  write(*,*) "PASSED!"

end program hip_sger
