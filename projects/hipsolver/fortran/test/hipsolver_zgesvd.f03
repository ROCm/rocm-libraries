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
! zgesvd example (complex singular value decomposition, Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Computes the singular values of A (jobu = jobv = 'N') and checks the
! convention-independent invariant sum(sigma_i^2) == ||A||_F^2.
!!!!!!!!!!!!!!
!
program hipsolver_zgesvd
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer(c_int), parameter :: M = 2, N = 2, lda = 2, ldu = 2, ldv = 2
  integer(c_int), parameter :: mn = 2
  complex(c_double_complex), target :: hA(M,N) = reshape((/ (1.,0.),(3.,0.),(2.,0.),(4.,0.) /), (/M,N/))
  real(c_double), target :: hS(mn)
  type(c_ptr) :: dA, dS, dU, dV, dWork, dRwork, dInfo, handle = c_null_ptr
  integer(c_int) :: lwork
  integer(c_size_t) :: szA = M*N, szS = mn, szU = M*M, szV = N*N, szR = mn
  real(c_double) :: frob, ssum, error
  real(c_double), parameter :: rtol = 1.0d-10
  integer :: i
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_zgesvd' (Fortran 2003 interfaces) - "
  call hipsolverCheck(hipsolverCreate(handle))
  call hipCheck(hipMalloc(dA,    szA * 16))
  call hipCheck(hipMalloc(dS,    szS * 8))
  call hipCheck(hipMalloc(dU,    szU * 16))
  call hipCheck(hipMalloc(dV,    szV * 16))
  call hipCheck(hipMalloc(dRwork, szR * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), szA * 16, hipMemcpyHostToDevice))
  call hipsolverCheck(hipsolverZgesvd_bufferSize(handle, 'N', 'N', M, N, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 16, 1_c_size_t)))
  call hipsolverCheck(hipsolverZgesvd(handle, 'N', 'N', M, N, dA, lda, dS, &
                                      dU, ldu, dV, ldv, dWork, lwork, dRwork, dInfo))
  call hipCheck(hipMemcpy(c_loc(hS(1)), dS, szS * 8, hipMemcpyDeviceToHost))
  frob = 1.0 + 9.0 + 4.0 + 16.0
  ssum = 0.0
  do i = 1, mn
     ssum = ssum + hS(i)**2
  end do
  error = abs(ssum - frob) / frob
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(sigma^2) = ", ssum, " expected ||A||_F^2 = ", frob
     call exit(1)
  end if
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dS)); call hipCheck(hipFree(dU))
  call hipCheck(hipFree(dV)); call hipCheck(hipFree(dRwork)); call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  write(*,*) "PASSED!"
end program hipsolver_zgesvd
