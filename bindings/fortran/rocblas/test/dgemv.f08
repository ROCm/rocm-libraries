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

program rocblas_dgemv_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_operation_none)), parameter :: trans = rocblas_operation_none
  double precision, target :: alpha = 1.1d0, beta = 0.9d0

  integer, parameter :: m = 1024, n = 1024

  double precision, allocatable, dimension(:,:) :: hA
  double precision, allocatable, dimension(:) :: hx, hy
  double precision :: y_exact

  double precision, pointer, dimension(:,:) :: dA
  double precision, pointer, dimension(:) :: dx, dy
  type(c_ptr) :: handle = c_null_ptr

  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'DGEMV' (Fortran 2008 interfaces) - "

  ! Create rocblas handle and set host pointer mode for host alpha/beta
  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))

  allocate(hA(m,n))
  allocate(hx(n))
  allocate(hy(m))

  ! Use these constant matrix/vectors so the exact answer is also a
  ! constant vector and therefore easy to check
  hA(:,:) = 1.d0
  hx(:) = 1.d0
  hy(:) = 1.d0
  y_exact = alpha * n + beta   ! = 1.1*1024 + 0.9 = 1127.3

  ! Allocate device memory
  call hipCheck(hipMalloc(dA,source=hA)) ! implies (blocking) memcpy
  call hipCheck(hipMalloc(dx,source=hx))
  call hipCheck(hipMalloc(dy,source=hy))

  call rocblasCheck(rocblas_dgemv(handle, trans, m, n, alpha, dA, size(dA,1), dx, 1, beta, dy, 1))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(hy, dy, hipMemcpyDeviceToHost))

  do i = 1,m
     error = abs((y_exact - hy(i))/y_exact)
     if( error > error_max )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  call rocblasCheck(rocblas_destroy_handle(handle))

  deallocate(hA, hx, hy)

  write(*,*) "PASSED!"

end program rocblas_dgemv_test
