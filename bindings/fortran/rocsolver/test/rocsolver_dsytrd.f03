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
! dsytrd example (rocSOLVER)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Reduces a real symmetric matrix A to symmetric tridiagonal form
! (A = Q T Q**T). The device buffers, tau included, are passed as type(c_ptr).
!!!!!!!!!!!!!!
!
program dsytrd
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: n = 4, lda = 4

  ! Symmetric 4x4 matrix (column-major); its trace is 10+11+12+13 = 46.
  real(c_double), target :: hA(n,n) = reshape([ &
      10.0d0,  2.0d0,  3.0d0,  6.0d0, &
       2.0d0, 11.0d0,  1.0d0,  0.0d0, &
       3.0d0,  1.0d0, 12.0d0,  2.0d0, &
       6.0d0,  0.0d0,  2.0d0, 13.0d0], [n,n])
  real(c_double), target :: hD(n) = 0.0d0
  real(c_double), target :: hE(n-1) = 0.0d0

  integer(c_size_t) :: size_A   = n*n
  integer(c_size_t) :: size_D   = n
  integer(c_size_t) :: size_E   = n-1
  integer(c_size_t) :: size_tau = n-1

  type(c_ptr) :: dA, dD, dE, dtau
  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: trace_A, normA2, error
  real(c_double), parameter :: rtol = 1.0d-9

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsytrd' (Fortran 2003 interfaces) - "

  trace_A = hA(1,1) + hA(2,2) + hA(3,3) + hA(4,4)   ! = 46
  normA2 = sum(hA**2)

  call hipCheck(hipMalloc(dA,   size_A   * 8))
  call hipCheck(hipMalloc(dD,   size_D   * 8))
  call hipCheck(hipMalloc(dE,   size_E   * 8))
  call hipCheck(hipMalloc(dtau, size_tau * 8))

  call rocblasCheck(rocblas_create_handle(handle))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  call rocsolverCheck(rocsolver_dsytrd(handle, rocblas_fill_lower, n, dA, lda, dD, dE, dtau))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD, size_D * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hE(1)), dE, size_E * 8, hipMemcpyDeviceToHost))

  ! The reduction is an orthogonal similarity, so trace(T) = trace(A), i.e.
  ! sum(D) = trace(A). This is convention-independent.
  error = abs(sum(hD) - trace_A) / abs(trace_A)
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(D) = ", sum(hD), " expected trace(A) = ", trace_A
     call exit(1)
  end if

  ! It also preserves the Frobenius norm: ||T||_F**2 = sum(D**2) + 2*sum(E**2).
  error = abs(sum(hD**2) + 2 * sum(hE**2) - normA2) / normA2
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! ||T||_F**2 = ", sum(hD**2) + 2 * sum(hE**2), " expected ", normA2
     call exit(1)
  end if

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dD))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dtau))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dsytrd
