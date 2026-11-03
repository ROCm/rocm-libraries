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

!!!!!!!!!!!!!/
! sgesvdj example (single-precision Jacobi SVD)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: compute the singular value decomposition A = U*S*V**T with the
! Jacobi method and confirm the factors reconstruct the original matrix. Using
! the reconstruction avoids sign/order ambiguity in the singular vectors.
!
! NOTE: gesvdj writes residual, n_sweeps and info to DEVICE memory, so those are
! backed by device allocations and passed as c_loc(...); host scalars fault.
! gesvdj has no generated array-pointer overload, so the matrix/vector arguments
! are passed to the raw interface via c_loc of their device pointers.
!!!!!!!!!!!!!!/
!
program sgesvdj
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldu = 3
  integer(c_int), parameter :: ldv = 3
  integer(c_int), parameter :: max_sweeps = 100
  real(c_float), parameter :: abstol = 0.0_c_float

  ! Nonsingular input matrix (column-major)
  real(c_float) :: hA(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3, 3/))
  real(c_float) :: hA0(3,3)          ! original kept for verification
  real(c_float) :: hS(3)             ! singular values
  real(c_float) :: hU(3,3)           ! left singular vectors
  real(c_float) :: hV(3,3)           ! right singular vectors
  real(c_float) :: recon(3,3)

  real(c_float), pointer :: dA(:,:), dU(:,:), dV(:,:)
  real(c_float), pointer :: dS(:)
  real(c_float), pointer :: dResidual(:)
  integer(c_int), pointer :: dNsweeps(:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_sgesvdj' (Fortran 2008 interfaces) - "

  hA0 = hA ! keep original for the reconstruction check

  ! Allocate device-side memory & copy memory from host to device
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dS, mold=hS))
  call hipCheck(hipMalloc(dU, mold=hU))
  call hipCheck(hipMalloc(dV, mold=hV))
  call hipCheck(hipMalloc(dResidual, 1))
  call hipCheck(hipMalloc(dNsweeps, 1))
  call hipCheck(hipMalloc(dInfo, 1))

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Compute the singular value decomposition
  call rocsolverCheck(rocsolver_sgesvdj(handle, rocblas_svect_singular, rocblas_svect_singular, &
       M, N, c_loc(dA(1,1)), lda, abstol, c_loc(dResidual(1)), max_sweeps, c_loc(dNsweeps(1)), &
       c_loc(dS(1)), c_loc(dU(1,1)), ldu, c_loc(dV(1,1)), ldv, c_loc(dInfo(1))))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if

  ! Copy results back to host
  call hipCheck(hipMemcpy(hS, dS, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hU, dU, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hV, dV, hipMemcpyDeviceToHost))

  ! Reconstruct A = U * diag(S) * V**T and compare with the original
  do j = 1,N
    do i = 1,M
      recon(i,j) = sum(hU(i,:) * hS(:) * hV(:,j))
    end do
  end do

  do j = 1,N
    do i = 1,M
      error = abs(recon(i,j) - hA0(i,j))
      if(.not. (error .le. error_max)) then
          write(*,*) "FAILED! Error bigger than max! Error = ", error, " (", i, ",", j, ")"
          call exit(1)
      end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dS))
  call hipCheck(hipFree(dU))
  call hipCheck(hipFree(dV))
  call hipCheck(hipFree(dResidual))
  call hipCheck(hipFree(dNsweeps))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program sgesvdj
