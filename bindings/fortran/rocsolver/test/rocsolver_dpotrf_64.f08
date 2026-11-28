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

!!!!!!!!!!!!!!
! dpotrf_64 example (Cholesky with the 64-bit integer API)
! Same result as dpotrf; n/lda/info are 64-bit. Device buffers are passed as
! type(c_ptr) (the _64 routines have no native-array overloads). info lives on
! the DEVICE. With rocblas_fill_upper the factor is written to the upper
! triangle; the lower triangle keeps the input values.
!!!!!!!!!!!!!!
!
program dpotrf_64
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer(c_int64_t), parameter :: N = 3, lda = 3
  real(c_double), target :: hA(3,3) = reshape((/4, 12, -16, 12, 37, -43, -16, -43, 98/), (/3, 3/))
  real(c_double), target :: hResult(3,3) = reshape((/&
     2.0000000000000000d0, 12.0000000000000000d0, -16.0000000000000000d0, &
     6.0000000000000000d0,  1.0000000000000000d0, -43.0000000000000000d0, &
    -8.0000000000000000d0,  5.0000000000000000d0,   3.0000000000000000d0/), (/3, 3/))
  integer(c_size_t) :: size_A = 9
  type(c_ptr) :: dA, dInfo, handle
  integer(c_int64_t), target :: hInfo(1)
  integer :: i, j
  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dpotrf_64' (Fortran 2008 interfaces) - "
  call hipCheck(hipMalloc(dA,    size_A * 8))
  call hipCheck(hipMalloc(dInfo, 8_c_size_t))   ! int64 info
  call rocblasCheck(rocblas_create_handle(handle))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call rocsolverCheck(rocsolver_dpotrf_64(handle, rocblas_fill_upper, N, dA, lda, dInfo))
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, 8_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))
  do j = 1, 3
     do i = 1, 3
        error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0_c_double)
        if (.not. (error <= error_max)) then
           write(*,*) "FAILED! hA(", i, ",", j, ") = ", hA(i,j)
           call exit(1)
        end if
     end do
  end do
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program dpotrf_64
