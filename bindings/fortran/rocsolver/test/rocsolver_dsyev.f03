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
! dsyev example (rocSOLVER)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Computes the eigenvalues of a real symmetric matrix. rocSOLVER writes `info`
! to DEVICE memory, so it is backed by a device allocation and passed as a
! device pointer.
!!!!!!!!!!!!!!
!
program dsyev
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
  real(c_double), target :: hD(n) = 0.0d0   ! eigenvalues

  integer(c_size_t) :: size_A = n*n
  integer(c_size_t) :: size_D = n
  integer(c_size_t) :: size_E = n

  type(c_ptr) :: dA, dD, dE, dInfo
  type(c_ptr) :: handle ! rocblas_handle
  integer(c_int), target :: hInfo

  real(c_double) :: trace_A, normA2, error
  real(c_double), parameter :: rtol = 1.0d-9

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsyev' (Fortran 2003 interfaces) - "

  trace_A = hA(1,1) + hA(2,2) + hA(3,3) + hA(4,4)   ! = 46
  normA2 = sum(hA**2)

  call hipCheck(hipMalloc(dA,    size_A * 8))
  call hipCheck(hipMalloc(dD,    size_D * 8))
  call hipCheck(hipMalloc(dE,    size_E * 8))
  call hipCheck(hipMalloc(dInfo, int(4, c_size_t)))

  call rocblasCheck(rocblas_create_handle(handle))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  hInfo = -1
  call hipCheck(hipMemcpy(dInfo, c_loc(hInfo), 4_c_size_t, hipMemcpyHostToDevice))
  call rocsolverCheck(rocsolver_dsyev(handle, rocblas_evect_none, rocblas_fill_lower, n, dA, lda, &
                                      dD, dE, dInfo))
  call hipCheck(hipMemcpy(c_loc(hInfo), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo /= 0) then
     write(*,*) "FAILED! info = ", hInfo, " (expected 0)"; call exit(1)
  end if

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD, size_D * 8, hipMemcpyDeviceToHost))

  ! An orthogonal diagonalization preserves the trace: sum of eigenvalues =
  ! trace(A). This is convention-independent (order of eigenvalues irrelevant).
  error = abs(sum(hD) - trace_A) / abs(trace_A)
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(eigenvalues) = ", sum(hD), " expected trace(A) = ", trace_A
     call exit(1)
  end if

  ! ... and the Frobenius norm: sum of squared eigenvalues = ||A||_F**2.
  error = abs(sum(hD**2) - normA2) / normA2
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(eigenvalues**2) = ", sum(hD**2), " expected ", normA2
     call exit(1)
  end if

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dD))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dsyev
