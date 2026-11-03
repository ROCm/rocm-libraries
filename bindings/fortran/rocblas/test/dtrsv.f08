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

program rocblas_dtrsv_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_fill_lower)),        parameter :: uplo   = rocblas_fill_lower
  integer(kind(rocblas_operation_none)),    parameter :: transA = rocblas_operation_none
  integer(kind(rocblas_diagonal_non_unit)), parameter :: diag   = rocblas_diagonal_non_unit

  integer, parameter :: m = 1024

  double precision, allocatable, dimension(:,:) :: hA
  double precision, allocatable, dimension(:) :: hx
  double precision, parameter :: x_exact = 1.d0

  double precision, pointer, dimension(:,:) :: dA
  double precision, pointer, dimension(:) :: dx
  type(c_ptr) :: handle = c_null_ptr

  integer :: i, j
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'DTRSV' (Fortran 2008 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))

  allocate(hA(m,m), hx(m))

  ! Lower-triangular A with all ones on/below the diagonal; upper part zeroed
  hA(:,:) = 0.d0
  do j = 1, m
    do i = j, m
      hA(i,j) = 1.d0
    end do
  end do
  ! Right-hand side b(i) = i  ->  exact solution x(i) = 1
  do i = 1, m
    hx(i) = dble(i)
  end do

  ! Allocate device memory
  call hipCheck(hipMalloc(dA, source=hA)) ! implies (blocking) memcpy
  call hipCheck(hipMalloc(dx, source=hx))

  ! Solve A * x = b, in place: dx holds b on entry, x on exit
  call rocblasCheck(rocblas_dtrsv(handle, uplo, transA, diag, m, dA, size(dA,1), dx, 1))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))

  do i = 1, m
    error = abs((x_exact - hx(i))/x_exact)
    if( error > error_max )then
      write(*,*) "FAILED! Error bigger than max! Error = ", error, " hx(", i, ") = ", hx(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dx))

  call rocblasCheck(rocblas_destroy_handle(handle))

  deallocate(hA, hx)

  write(*,*) "PASSED!"

end program rocblas_dtrsv_test
