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
! ssytrf example (Bunch-Kaufman symmetric factorization, single, Fortran 2003)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/
!
! Factorizes a symmetric matrix and checks info == 0 (read back from
! device memory, seeded with -1), the pivots, and A = U*D*U**T.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program ssytrf
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer(c_int), parameter :: N = 4, lda = 4
  real(c_float), target :: hA(N,N) = reshape((/ &
      10.0,  2.0,  3.0,  6.0, &
       2.0, 11.0,  1.0,  0.0, &
       3.0,  1.0, 12.0,  2.0, &
       6.0,  0.0,  2.0, 13.0/), (/N,N/))
  integer(c_int), target :: hInfo(1)
  type(c_ptr) :: dA, dIpiv, dInfo
  type(c_ptr) :: handle
  real(c_float) :: hA0(N,N), hU(N,N), hD(N,N)
  integer(c_int), target :: hIpiv(N)
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: rtol = 1.0e-5
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_ssytrf' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(dA, int(N*N,c_size_t) * 4))
  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(N*N,c_size_t) * 4, hipMemcpyHostToDevice))
  call rocblasCheck(rocblas_create_handle(handle))
  hA0 = hA
  hInfo(1) = -1
  call hipCheck(hipMemcpy(dInfo, c_loc(hInfo(1)), 4_c_size_t, hipMemcpyHostToDevice))
  call rocsolverCheck(rocsolver_ssytrf(handle, rocblas_fill_upper, N, dA, lda, dIpiv, dInfo))
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
     write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, int(N*N,c_size_t) * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hIpiv(1)), dIpiv, int(N,c_size_t) * 4, hipMemcpyDeviceToHost))
  ! A is diagonally dominant, so Bunch-Kaufman takes 1x1 pivots with no
  ! interchange: ipiv(k) = k, U is unit upper and D is diagonal.
  do i = 1, N
    if (hIpiv(i) /= i) then
      write(*,*) "FAILED! ipiv(", i, ") = ", hIpiv(i), " expected ", i; call exit(1)
    end if
  end do
  hU = 0; hD = 0
  do j = 1, N
    hU(1:j-1,j) = hA(1:j-1,j)
    hU(j,j) = 1
    hD(j,j) = hA(j,j)
  end do
  error = sqrt(sum(abs(hA0 - matmul(hU, matmul(hD, transpose(hU))))**2)) / sqrt(sum(abs(hA0)**2))
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! ||A - U*D*U**T||_F / ||A||_F = ", error; call exit(1)
  end if
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dIpiv)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program ssytrf
