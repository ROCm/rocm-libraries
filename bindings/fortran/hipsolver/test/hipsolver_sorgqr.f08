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
! hipsolverSorgqr example (single-precision generation of Q from a QR
! factorization)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: factorize A with geqrf, generate the orthogonal factor Q with
! orgqr, and confirm Q**T * Q = I. Orthogonality is sign-convention independent,
! so no reference matrix is needed.
!!!!!!!!!!!!!!/
!
program sorgqr
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: K = 3
  integer(c_int), parameter :: lda = 3

  ! Input matrix (column-major)
  real(c_float) :: hA(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3, 3/))
  real(c_float) :: hTau(3)
  real(c_float) :: gram

  type(c_ptr) :: handle = c_null_ptr
  real(c_float), pointer :: dA(:,:)
  real(c_float), pointer :: dTau(:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)
  type(c_ptr) :: dWork
  integer(c_int) :: lwork_qr, lwork_or, lwork

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverSorgqr' (Fortran 2008 interfaces) - "

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device-side memory & copy memory from host to device
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dTau, mold=hTau))
  call hipCheck(hipMalloc(dInfo, 1))

  ! Workspace big enough for both geqrf and orgqr
  call hipsolverCheck(hipsolverSgeqrf_bufferSize(handle, M, N, dA, lda, lwork_qr))
  call hipsolverCheck(hipsolverSorgqr_bufferSize(handle, M, N, K, dA, lda, dTau, lwork_or))
  lwork = max(lwork_qr, lwork_or)
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 4))

  ! Factorize A = Q*R, then form the explicit Q in place
  call hipsolverCheck(hipsolverSgeqrf(handle, M, N, dA, lda, dTau, dWork, lwork, c_loc(dInfo(1))))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if(hInfo(1) /= 0) then
    write(*,*) "FAILED! geqrf info = ", hInfo(1), " (expected 0)"
    call exit(1)
  end if
  call hipsolverCheck(hipsolverSorgqr(handle, M, N, K, dA, lda, dTau, dWork, lwork, c_loc(dInfo(1))))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if(hInfo(1) /= 0) then
    write(*,*) "FAILED! orgqr info = ", hInfo(1), " (expected 0)"
    call exit(1)
  end if

  ! Copy Q back to host
  call hipCheck(hipMemcpy(hA, dA, hipMemcpyDeviceToHost))

  ! Verify Q**T * Q = I
  do j = 1,N
    do i = 1,N
      gram = sum(hA(:,i) * hA(:,j))
      if(i .eq. j) then
        error = abs(gram - 1.0_c_float)
      else
        error = abs(gram)
      end if
      if(.not. (error .le. error_max)) then
          write(*,*) "FAILED! Q not orthogonal! Error = ", error, " (", i, ",", j, ")"
          call exit(1)
      end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dTau))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program sorgqr
