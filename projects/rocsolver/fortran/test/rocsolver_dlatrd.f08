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
! dlatrd example (rocSOLVER)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/auxiliary.html
!
! Reduces the first k columns of a symmetric matrix A to symmetric tridiagonal
! form (uplo = lower). This test passes A, E, tau and W as native Fortran device
! arrays: W is a 2-D matrix (dimension(:,:)) and tau a 1-D vector (dimension(:)).
! Before the binding fix, W was a bare type(c_ptr) and tau a scalar, so this
! call could not be expressed with typed arrays at all.
!!!!!!!!!!!!!!
!
program dlatrd
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: n = 4, k = 2, lda = 4, ldw = 4

  ! Symmetric 4x4 matrix (column-major). Its first sub-column A(2:n,1) = [2,3,6]
  ! has 2-norm 7, which is the quantity we check below.
  real(c_double) :: hA(n,n) = reshape([ &
      10.0d0,  2.0d0,  3.0d0,  6.0d0, &
       2.0d0, 11.0d0,  1.0d0,  0.0d0, &
       3.0d0,  1.0d0, 12.0d0,  2.0d0, &
       6.0d0,  0.0d0,  2.0d0, 13.0d0], [n,n])
  real(c_double) :: hE(n-1)   = 0.0d0
  real(c_double) :: htau(n-1) = 0.0d0
  real(c_double) :: hW(n,k)   = 0.0d0

  real(c_double), pointer :: dA(:,:)   ! GPU buffer for A
  real(c_double), pointer :: dE(:)     ! GPU buffer for the off-diagonal E
  real(c_double), pointer :: dtau(:)   ! GPU buffer for the Householder scalars
  real(c_double), pointer :: dW(:,:)   ! GPU buffer for the update matrix W

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: expected, error
  real(c_double), parameter :: rtol = 1.0d-10

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dlatrd' (Fortran 2008 interfaces) - "

  ! Allocate device-side memory & copy the inputs from host to device
  call hipCheck(hipMalloc(dA,   source=hA))
  call hipCheck(hipMalloc(dE,   source=hE))
  call hipCheck(hipMalloc(dtau, source=htau))
  call hipCheck(hipMalloc(dW,   source=hW))

  call rocblasCheck(rocblas_create_handle(handle))

  ! Reduce the first k columns to tridiagonal form. A/E/tau/W are all passed as
  ! native Fortran device arrays (resolves to the _full_rank generic wrapper).
  call rocsolverCheck(rocsolver_dlatrd(handle, rocblas_fill_lower, n, k, dA, lda, dE, dtau, dW, ldw))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hE, dE, hipMemcpyDeviceToHost))

  ! The first Householder reflector maps the original sub-column A(2:n,1) onto
  ! [E(1), 0, ..., 0], so |E(1)| == ||A(2:n,1)||_2, independent of the sign
  ! convention rocSOLVER uses for the reflector.
  expected = sqrt(2.0d0**2 + 3.0d0**2 + 6.0d0**2)   ! = 7
  error = abs(abs(hE(1)) - expected) / expected
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! |E(1)| = ", abs(hE(1)), " expected ", expected
     call exit(1)
  end if

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dtau))
  call hipCheck(hipFree(dW))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dlatrd
