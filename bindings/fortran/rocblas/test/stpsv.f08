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


program rocblas_stpsv_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_fill_lower)),        parameter :: uplo   = rocblas_fill_lower
  integer(kind(rocblas_operation_none)),    parameter :: transA = rocblas_operation_none
  integer(kind(rocblas_diagonal_non_unit)), parameter :: diag   = rocblas_diagonal_non_unit

  integer, parameter :: n = 64

  ! Packed lower-triangular matrix (column-by-column). Every stored element of an
  ! all-ones lower-triangular matrix is 1, so the packing order is irrelevant here.
  real(c_float), allocatable, dimension(:) :: hAP, hx
  real(c_float), parameter :: x_exact = 1.0

  real(c_float), pointer, dimension(:) :: dAP => null(), dx => null()
  type(c_ptr) :: handle = c_null_ptr

  integer :: i
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'STPSV' (Fortran 2008 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))

  allocate(hAP(n*(n+1)/2), hx(n))

  hAP(:) = 1.0            ! all-ones lower-triangular L, packed
  do i = 1, n
    hx(i) = real(i)       ! b(i) = i  ->  exact solution of L x = b is x(i) = 1
  end do

  ! Allocate device memory (source= implies a blocking memcpy)
  call hipCheck(hipMalloc(dAP, source=hAP))
  call hipCheck(hipMalloc(dx,  source=hx))

  ! Packed triangular solve L x = b, in place. dAP is passed as a Fortran array
  ! (the generic array form): this is the case that used to fail to compile
  ! because the packed matrix dummy was declared type(c_ptr) (SWDEV-485451).
  call rocblasCheck(rocblas_stpsv(handle, uplo, transA, diag, n, dAP, dx, 1))

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))

  do i = 1, n
    error = abs((x_exact - hx(i))/x_exact)
    if( error > error_max )then
      write(*,*) "FAILED! Error bigger than max! Error = ", error, " hx(", i, ") = ", hx(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dAP))
  call hipCheck(hipFree(dx))

  call rocblasCheck(rocblas_destroy_handle(handle))

  deallocate(hAP, hx)

  write(*,*) "PASSED!"

end program rocblas_stpsv_test
