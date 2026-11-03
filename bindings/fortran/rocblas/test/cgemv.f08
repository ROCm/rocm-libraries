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

program rocblas_cgemv_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_operation_none)), parameter :: trans = rocblas_operation_none
  complex(c_float_complex), target :: alpha = (1.1, 0.), beta = (0.9, 0.)

  integer, parameter :: m = 512, n = 512

  complex(kind=4), allocatable, dimension(:,:) :: hA
  complex(kind=4), allocatable, dimension(:) :: hx, hy
  complex(kind=4) :: y_exact

  complex(kind=4), pointer, dimension(:,:) :: dA
  complex(kind=4), pointer, dimension(:) :: dx, dy
  type(c_ptr) :: handle = c_null_ptr

  integer :: i
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(0.0_c_float)

  write(*,"(a)",advance="no") "-- Running test 'CGEMV' (Fortran 2008 interfaces) - "

  ! Create rocblas handle and set host pointer mode for host alpha/beta
  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))

  allocate(hA(m,n))
  allocate(hx(n))
  allocate(hy(m))

  ! Use these constant matrix/vectors so the exact answer is also a
  ! constant vector and therefore easy to check
  hA(:,:) = (1., 0.)
  hx(:) = (1., 0.)
  hy(:) = (1., 0.)
  y_exact = alpha * n + beta   ! = (1.1*512 + 0.9, 0.0) = (564.1, 0.0)

  ! Allocate device memory
  call hipCheck(hipMalloc(dA,source=hA)) ! implies (blocking) memcpy
  call hipCheck(hipMalloc(dx,source=hx))
  call hipCheck(hipMalloc(dy,source=hy))

  call rocblasCheck(rocblas_cgemv(handle, trans, m, n, alpha, dA, size(dA,1), dx, 1, beta, dy, 1))

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

end program rocblas_cgemv_test
