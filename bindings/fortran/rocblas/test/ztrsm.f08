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

program rocblas_ztrsm_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_side_left)),         parameter :: side   = rocblas_side_left
  integer(kind(rocblas_fill_lower)),        parameter :: uplo   = rocblas_fill_lower
  integer(kind(rocblas_operation_none)),    parameter :: transA = rocblas_operation_none
  integer(kind(rocblas_diagonal_non_unit)), parameter :: diag   = rocblas_diagonal_non_unit

  integer, parameter :: m = 512, n = 512

  complex(kind=8), allocatable, dimension(:,:) :: hA, hB
  complex(c_double_complex), target :: alpha = (2.d0, 0.d0)
  complex(kind=8), parameter :: x_exact = (1.d0, 0.d0)

  complex(kind=8), pointer, dimension(:,:) :: dA, dB
  type(c_ptr) :: handle = c_null_ptr

  integer :: i, j
  real(c_double) :: error
  real(c_double), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'ZTRSM' (Fortran 2008 interfaces) - "

  ! Create rocblas handle and set host pointer mode for host alpha
  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))

  allocate(hA(m,m), hB(m,n))

  ! Lower-triangular A with all ones on/below the diagonal; upper part zeroed
  hA(:,:) = (0.d0, 0.d0)
  do j = 1, m
    do i = j, m
      hA(i,j) = (1.d0, 0.d0)
    end do
  end do
  ! RHS B(i,j) = i/2 = alpha^-1 * i  ->  exact solution X(i,j) = 1
  do j = 1, n
    do i = 1, m
      hB(i,j) = cmplx(dble(i) / 2.d0, 0., kind=8)
    end do
  end do

  ! Allocate device memory
  call hipCheck(hipMalloc(dA, source=hA)) ! implies (blocking) memcpy
  call hipCheck(hipMalloc(dB, source=hB))

  ! Solve A * X = alpha * B, in place: dB holds B on entry, X on exit
  call rocblasCheck(rocblas_ztrsm(handle, side, uplo, transA, diag, m, n, alpha, dA, size(dA,1), dB, size(dB,1)))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(hB, dB, hipMemcpyDeviceToHost))

  do j = 1, n
    do i = 1, m
      error = abs((x_exact - hB(i,j))/x_exact)
      if( error > error_max )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " hB(", i, ",", j, ") = ", hB(i,j)
        call exit(1)
      end if
    end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))

  call rocblasCheck(rocblas_destroy_handle(handle))

  deallocate(hA, hB)

  write(*,*) "PASSED!"

end program rocblas_ztrsm_test
