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
! ssyevj example (single-precision symmetric Jacobi eigensolver)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: with evect=original, A is overwritten with the eigenvectors
! (as columns) and W holds the eigenvalues. We confirm A0*v_k = lambda_k*v_k.
!
! NOTE: syevj writes residual, n_sweeps and info to DEVICE memory, so those are
! backed by device allocations and passed as c_loc(...); host scalars fault.
! syevj has no generated array-pointer overload, so A and W are passed to the
! raw interface via c_loc of their device pointers.
!!!!!!!!!!!!!!/
!
program ssyevj
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, k ! indices for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: max_sweeps = 100
  real(c_float), parameter :: abstol = 0.0_c_float

  ! Symmetric input (column-major); eigenvalues are 2-sqrt(2), 2, 2+sqrt(2)
  real(c_float) :: hA(3,3) = reshape((/2, -1, 0, -1, 2, -1, 0, -1, 2/), (/3, 3/))
  real(c_float) :: hA0(3,3)   ! original kept for verification
  real(c_float) :: hW(3)      ! eigenvalues
  real(c_float) :: lhs(3), rhs(3)

  real(c_float), pointer :: dA(:,:)
  real(c_float), pointer :: dW(:)
  real(c_float), pointer :: dResidual(:)
  integer(c_int), pointer :: dNsweeps(:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_ssyevj' (Fortran 2008 interfaces) - "

  hA0 = hA ! keep original for the A*v = lambda*v check

  ! Allocate device-side memory & copy memory from host to device
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dW, mold=hW))
  call hipCheck(hipMalloc(dResidual, 1))
  call hipCheck(hipMalloc(dNsweeps, 1))
  call hipCheck(hipMalloc(dInfo, 1))

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Compute eigenvalues and eigenvectors (A overwritten with eigenvectors)
  call rocsolverCheck(rocsolver_ssyevj(handle, rocblas_esort_ascending, rocblas_evect_original, &
       rocblas_fill_upper, N, c_loc(dA(1,1)), lda, abstol, c_loc(dResidual(1)), max_sweeps, &
       c_loc(dNsweeps(1)), c_loc(dW(1)), c_loc(dInfo(1))))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if

  ! Copy results back to host
  call hipCheck(hipMemcpy(hA, dA, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hW, dW, hipMemcpyDeviceToHost))

  ! Verify A0 * v_k = lambda_k * v_k for each eigenpair
  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hW(k) * hA(:,k)
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(.not. (error .le. error_max)) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k
            call exit(1)
        end if
    end do
    ! Eigenvectors are normalised: ||v_k|| = 1
    error = abs(sqrt(sum(abs(hA(:,k))**2)) - 1)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! ||v|| - 1 = ", error, " eigenvector ", k; call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dResidual))
  call hipCheck(hipFree(dNsweeps))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program ssyevj
