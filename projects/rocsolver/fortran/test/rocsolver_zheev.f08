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
! zheev example (double-precision complex Hermitian eigenvalues)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Computes the eigenvalues of a Hermitian matrix. rocSOLVER writes `info` to
! DEVICE memory (passed as c_loc(dInfo)). A Hermitian matrix has a real trace
! equal to the sum of its (real) eigenvalues, which is what we check.
!!!!!!!!!!!!!!
!
program zheev
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: N = 3, lda = 3

  ! Hermitian 3x3 (column-major); diagonal is real, trace = 4+3+5 = 12.
  complex(c_double_complex) :: hA(N,N) = reshape((/ &
    (4.0d0, 0.0d0), (1.0d0,-1.0d0), (0.0d0, 0.0d0), &
    (1.0d0, 1.0d0), (3.0d0, 0.0d0), (0.0d0,-2.0d0), &
    (0.0d0, 0.0d0), (0.0d0, 2.0d0), (5.0d0, 0.0d0) /), (/N, N/))
  real(c_double) :: hD(N) = 0.0d0   ! eigenvalues
  real(c_double) :: hE(N) = 0.0d0   ! workspace

  complex(c_double_complex), pointer :: dA(:,:)
  real(c_double), pointer :: dD(:)
  real(c_double), pointer :: dE(:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)

  type(c_ptr) :: handle

  real(c_double) :: trace_A, frob2_A, error
  real(c_double), parameter :: rtol = 1.0d-9

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_zheev' (Fortran 2008 interfaces) - "

  trace_A = real(hA(1,1),c_double) + real(hA(2,2),c_double) + real(hA(3,3),c_double)   ! = 12
  frob2_A = sum(abs(hA)**2)   ! ||A||_F**2, from the original A

  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dD,    source=hD))
  call hipCheck(hipMalloc(dE,    source=hE))
  call hipCheck(hipMalloc(dInfo, 1))

  call rocblasCheck(rocblas_create_handle(handle))

  call rocsolverCheck(rocsolver_zheev(handle, rocblas_evect_none, rocblas_fill_upper, N, dA, lda, &
                                dD, dE, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if

  call hipCheck(hipMemcpy(hD, dD, hipMemcpyDeviceToHost))

  ! Sum of eigenvalues equals the (real) trace, independent of their order.
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

end program zheev
