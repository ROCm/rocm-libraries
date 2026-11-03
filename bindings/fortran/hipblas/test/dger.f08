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

program hip_dger

  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer, parameter ::  m = 100, n = 100
  double precision, target :: alpha = 1.1d0

  double precision, allocatable, target, dimension(:)   :: hx, hy
  double precision, allocatable, target, dimension(:,:) :: hA

  type(c_ptr) :: handle = c_null_ptr

  double precision, pointer, dimension(:)   :: dx, dy
  double precision, pointer, dimension(:,:) :: dA

  integer :: i,j
  double precision :: error

  write(*,"(a)",advance="no") "-- Running test 'dger' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(hx(m))
  allocate(hy(n))
  allocate(ha(m,n))

  hx(:)   = 1.d0
  hy(:)   = 1.d0
  hA(:,:) = 1.d0

  ! Allocate device memory
  call hipCheck(hipMalloc(dx,m))
  call hipCheck(hipMalloc(dy,n))
  call hipCheck(hipMalloc(dA,m,n))

  !Transfer from host to device
  call hipCheck(hipMemcpy(dx, hx, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, hy, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dA, hA, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasDger(handle,m,n,alpha,dx,1,dy,1,dA,m))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(hA, dA, hipMemcpyDeviceToHost))

  do j = 1,n
    do i = 1,m
     error = abs(2.1d0 - hA(i,j))
     if( error > 10*epsilon(error) )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, "hA(i,j) = ", hA(i,j)
        call exit(1)
     end if
   end do
  end do

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipCheck(hipFree(dA))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(hx,hy,hA)

  write(*,*) "PASSED!"

end program hip_dger
