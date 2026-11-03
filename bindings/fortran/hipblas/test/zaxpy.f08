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

program hip_zaxpy
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer :: n = 6
  type(c_ptr) :: handle = c_null_ptr
  integer :: j
  complex(kind=8), allocatable, dimension(:) :: x, y, y_exact

  complex(kind=8), allocatable, target :: alpha
  complex(kind=8), pointer, dimension(:) :: dx, dy

  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)
  ! On the heap, so 16-byte aligned: the CUDA backend accesses a
  ! cuDoubleComplex scalar with an aligned load or store.
  allocate(alpha)
  alpha = (2.0d0, 0.d0)

  allocate(x(n))
  allocate(y(n))
  allocate(y_exact(n))

  do j = 1, n
    x(j) = cmplx(j, 0, kind=8)
    y(j) = cmplx(j, 0, kind=8)
  end do

  do j = 1, n
    y_exact(j) = alpha*x(j) + y(j)
  end do

  write(*,"(a)",advance="no") "-- Running test 'ZAXPY' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx, shape(x)))
  call hipCheck(hipMalloc(dy, shape(y)))

  call hipCheck(hipMemcpy(dx, x, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, y, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasZaxpy(handle, n, alpha, dx, 1, dy, 1))

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(y, dy, hipMemcpyDeviceToHost))

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

  deallocate(x, y, y_exact, alpha)

  write(*,*) "PASSED!"

end program hip_zaxpy
