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
! to DEVICE memory, so it is backed by a device allocation and passed as
! c_loc(dInfo).
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
  real(c_double) :: hA(n,n) = reshape([ &
      10.0d0,  2.0d0,  3.0d0,  6.0d0, &
       2.0d0, 11.0d0,  1.0d0,  0.0d0, &
       3.0d0,  1.0d0, 12.0d0,  2.0d0, &
       6.0d0,  0.0d0,  2.0d0, 13.0d0], [n,n])
  real(c_double) :: hD(n) = 0.0d0   ! eigenvalues
  real(c_double) :: hE(n) = 0.0d0   ! workspace

  real(c_double), pointer :: dA(:,:)   ! GPU buffer for A
  real(c_double), pointer :: dD(:)     ! GPU buffer for the eigenvalues
  real(c_double), pointer :: dE(:)     ! GPU workspace
  integer(c_int), pointer :: dInfo(:)  ! GPU buffer for info
  integer(c_int) :: hInfo(1)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: trace_A, frob2_A, error
  real(c_double), parameter :: rtol = 1.0d-9

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsyev' (Fortran 2008 interfaces) - "

  trace_A = hA(1,1) + hA(2,2) + hA(3,3) + hA(4,4)   ! = 46
  frob2_A = sum(hA**2)   ! ||A||_F**2, from the original A

  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dD,    source=hD))
  call hipCheck(hipMalloc(dE,    source=hE))
  call hipCheck(hipMalloc(dInfo, 1))

  call rocblasCheck(rocblas_create_handle(handle))

  ! Eigenvalues only (rocblas_evect_none); A/D/E as native arrays, info device.
  call rocsolverCheck(rocsolver_dsyev(handle, rocblas_evect_none, rocblas_fill_lower, n, dA, lda, &
                                      dD, dE, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hD, dD, hipMemcpyDeviceToHost))

  ! An orthogonal diagonalization preserves the trace: sum of eigenvalues =
  ! trace(A). This is convention-independent (order of eigenvalues irrelevant).
  error = abs(sum(hD) - trace_A) / abs(trace_A)
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(eigenvalues) = ", sum(hD), " expected trace(A) = ", trace_A
     call exit(1)
  end if

  ! The squared Frobenius norm is also invariant: sum of squared eigenvalues = ||A||_F**2.
  error = abs(sum(hD**2) - frob2_A) / abs(frob2_A)
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(eigenvalues**2) = ", sum(hD**2), " expected ||A||_F**2 = ", frob2_A
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
