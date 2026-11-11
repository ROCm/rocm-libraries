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

program hipblas_dgemv_test
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  double precision, target :: alpha = 1.1d0, beta = 0.9d0
  integer, parameter :: m = 1024, n = 1024

  double precision, allocatable, target, dimension(:,:) :: hA
  double precision, allocatable, target, dimension(:) :: hx, hy
  double precision, pointer, dimension(:,:) :: dA
  double precision, pointer, dimension(:)   :: dx, dy
  type(c_ptr) :: handle = c_null_ptr

  double precision :: y_exact, error
  double precision, parameter :: error_max = 10*epsilon(error)
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'DGEMV' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(hA(m,n), hx(n), hy(m))
  hA = 1.d0;  hx = 1.d0;  hy = 1.d0
  y_exact = alpha * n + beta

  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dx, source=hx))
  call hipCheck(hipMalloc(dy, source=hy))

  call hipblasCheck(hipblasDgemv(handle, HIPBLAS_OP_N, m, n, alpha, dA, size(dA,1), dx, 1, beta, dy, 1))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hy, dy, hipMemcpyDeviceToHost))

  do i = 1, m
    error = abs((y_exact - hy(i)) / y_exact)
    if (error > error_max) then
      write(*,*) "FAILED! Error bigger than max! Error = ", error
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipblasCheck(hipblasDestroy(handle))
  deallocate(hA, hx, hy)
  write(*,*) "PASSED!"

end program hipblas_dgemv_test
