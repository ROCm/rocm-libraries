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
! hipsolver dgetrs example (solve A*X=B after LU, Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Factorizes A with getrf, then solves A*X=B with getrs and checks X.
! Native-array f2008 form; workspaces via *_bufferSize; devInfo device-backed.
!!!!!!!!!!!!!!
!
program hipsolver_dgetrs
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer :: i
  integer(c_int), parameter :: N = 3, nrhs = 1, lda = 3, ldb = 3
  real(c_double) :: hA(N,N)     = reshape((/2, 1, 0,  1, 2, 1,  0, 1, 2/), (/N,N/))
  real(c_double) :: hB(N,nrhs)  = reshape((/3, 4, 3/), (/N,nrhs/))
  real(c_double) :: hX_ref(N)   = (/1, 1, 1/)
  real(c_double), pointer :: dA(:,:), dB(:,:)
  integer(c_int), pointer :: dIpiv(:)
  integer(c_int), pointer :: dInfo
  integer(c_int) :: hIpiv(N), hInfo
  type(c_ptr) :: dWork1, dWork2, handle = c_null_ptr
  integer(c_int) :: lwork1, lwork2
  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_dgetrs' (Fortran 2008 interfaces) - "
  call hipsolverCheck(hipsolverCreate(handle))
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dB, source=hB))
  call hipCheck(hipMalloc(dIpiv, mold=hIpiv))
  call hipCheck(hipMalloc(dInfo))
  ! LU factorization
  call hipsolverCheck(hipsolverDgetrf_bufferSize(handle, N, N, dA, lda, lwork1))
  call hipCheck(hipMalloc(dWork1, max(int(lwork1,c_size_t) * 8, 1_c_size_t)))
  call hipsolverCheck(hipsolverDgetrf(handle, N, N, dA, lda, dWork1, lwork1, dIpiv, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo /= 0) then
     write(*,*) "FAILED! getrf info = ", hInfo, " (expected 0)"
     call exit(1)
  end if
  ! Solve
  call hipsolverCheck(hipsolverDgetrs_bufferSize(handle, HIPSOLVER_OP_N, N, nrhs, dA, lda, dIpiv, dB, ldb, lwork2))
  call hipCheck(hipMalloc(dWork2, max(int(lwork2,c_size_t) * 8, 1_c_size_t)))
  call hipsolverCheck(hipsolverDgetrs(handle, HIPSOLVER_OP_N, N, nrhs, dA, lda, dIpiv, dB, ldb, dWork2, lwork2, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo /= 0) then
     write(*,*) "FAILED! getrs info = ", hInfo, " (expected 0)"
     call exit(1)
  end if
  call hipCheck(hipMemcpy(hB, dB, hipMemcpyDeviceToHost))
  do i = 1, N
     error = abs(hB(i,1) - hX_ref(i)) / max(abs(hX_ref(i)), 1.0d0)
     if (.not. (error <= error_max)) then
        write(*,*) "FAILED! X(", i, ") = ", hB(i,1), " expected ", hX_ref(i)
        call exit(1)
     end if
  end do
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dB)); call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo)); call hipCheck(hipFree(dWork1)); call hipCheck(hipFree(dWork2))
  call hipsolverCheck(hipsolverDestroy(handle))
  write(*,*) "PASSED!"
end program hipsolver_dgetrs
