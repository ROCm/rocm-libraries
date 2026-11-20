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
! cheevj example (single-complex Hermitian Jacobi eigensolver, Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: with evect=original, A is overwritten with the (complex)
! eigenvectors and the real vector W holds the eigenvalues. We confirm
! A0*v_k = lambda_k*v_k.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. heevj writes residual, n_sweeps and
! info to device memory, so those are device allocations passed directly.
!!!!!!!!!!!!!!/
!
program cheevj
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer(c_int), target :: hInfo
  integer :: i, k ! indices for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: max_sweeps = 100
  real(c_float), parameter :: abstol = 0.0_c_float

  ! Hermitian input (column-major)
  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (2.0,0.0), (1.0,-1.0), (0.0,0.0), &
    (1.0,1.0), (3.0,0.0), (0.0,-1.0), &
    (0.0,0.0), (0.0,1.0), (4.0,0.0)/), (/3, 3/))
  complex(c_float_complex) :: hA0(3,3)   ! original kept for verification
  real(c_float), target :: hW(3)         ! eigenvalues (real)
  complex(c_float_complex) :: lhs(3), rhs(3)

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_W = size(hW)

  type(c_ptr) :: dA
  type(c_ptr) :: dW
  type(c_ptr) :: dResidual
  type(c_ptr) :: dNsweeps
  type(c_ptr) :: dInfo

  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_cheevj' (Fortran 2003 interfaces) - "

  hA0 = hA ! keep original for the A*v = lambda*v check

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dW, size_W * 4))
  call hipCheck(hipMalloc(dResidual, 4_c_size_t))
  call hipCheck(hipMalloc(dNsweeps, 4_c_size_t))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Compute eigenvalues and eigenvectors (A overwritten with eigenvectors)
  hInfo = -1
  call hipCheck(hipMemcpy(dInfo, c_loc(hInfo), 4_c_size_t, hipMemcpyHostToDevice))
  call rocsolverCheck(rocsolver_cheevj(handle, rocblas_esort_ascending, rocblas_evect_original, &
       rocblas_fill_upper, N, dA, lda, abstol, dResidual, max_sweeps, dNsweeps, dW, dInfo))
  call hipCheck(hipMemcpy(c_loc(hInfo), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo /= 0) then
    write(*,*) "FAILED! info = ", hInfo, " (expected 0)"; call exit(1)
  end if

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, size_W * 4, hipMemcpyDeviceToHost))

  ! Verify A0 * v_k = lambda_k * v_k for each eigenpair
  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hW(k) * hA(:,k)
    error = abs(sqrt(sum(abs(hA(:,k))**2)) - 1)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! eigenvector ", k, " norm error = ", error; call exit(1)
    end if
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(.not. (error .le. error_max)) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k
            call exit(1)
        end if
    end do
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

end program cheevj
