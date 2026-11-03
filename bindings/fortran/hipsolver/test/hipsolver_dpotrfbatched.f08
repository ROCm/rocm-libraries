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
! hipsolver dpotrfBatched example (batched Cholesky factorization)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Factorizes a batch of SPD matrices. The batched API takes A as an array of
! device pointers that itself lives in DEVICE memory: each matrix is allocated on
! the device, their device addresses are collected in a host array, and that
! array is copied to a device buffer whose address (by value) is passed as A.
! Checks info == 0 for every batch entry and the Cholesky diagonal.
!!!!!!!!!!!!!!
!
program dpotrfbatched
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer :: b
  integer(c_int), parameter :: N = 3, lda = 3, batch = 2
  ! SPD A = diag(4, 9, 16) -> upper Cholesky diag = 2, 3, 4
  real(c_double), target :: hA(N,N) = reshape((/ &
      4.0d0, 0.0d0, 0.0d0, &
      0.0d0, 9.0d0, 0.0d0, &
      0.0d0, 0.0d0,16.0d0/), (/N,N/))
  real(c_double), target :: hOut(N,N)
  integer(c_int) :: hInfo(batch)
  type(c_ptr), target :: hostPtrs(batch)
  type(c_ptr) :: dA1, dA2, dPtrArray, dWork
  integer(c_int), pointer :: dInfo(:)
  type(c_ptr) :: handle
  integer(c_int) :: lwork
  integer(c_size_t) :: ptrbytes
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_dpotrfbatched' (Fortran 2008 interfaces) - "

  call hipsolverCheck(hipsolverCreate(handle))
  call hipCheck(hipMalloc(dA1, int(N*N,c_size_t) * 8))
  call hipCheck(hipMalloc(dA2, int(N*N,c_size_t) * 8))
  call hipCheck(hipMemcpy(dA1, c_loc(hA(1,1)), int(N*N,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dA2, c_loc(hA(1,1)), int(N*N,c_size_t) * 8, hipMemcpyHostToDevice))

  ! Host array of device addresses, copied to a device-resident pointer array.
  hostPtrs(1) = dA1
  hostPtrs(2) = dA2
  ptrbytes = int(batch,c_size_t) * c_sizeof(c_null_ptr)
  call hipCheck(hipMalloc(dPtrArray, ptrbytes))
  call hipCheck(hipMemcpy(dPtrArray, c_loc(hostPtrs), ptrbytes, hipMemcpyHostToDevice))

  call hipCheck(hipMalloc(dInfo, batch))

  call hipsolverCheck(hipsolverDpotrfBatched_bufferSize(handle, HIPSOLVER_FILL_MODE_UPPER, &
       N, dPtrArray, lda, lwork, batch))
  call hipCheck(hipMalloc(dWork, int(max(lwork,1),c_size_t) * 8))
  call hipsolverCheck(hipsolverDpotrfBatched(handle, HIPSOLVER_FILL_MODE_UPPER, &
       N, dPtrArray, lda, dWork, lwork, c_loc(dInfo(1)), batch))
  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))

  do b = 1, batch
     if (hInfo(b) /= 0) then
        write(*,*) "FAILED! info(", b, ") = ", hInfo(b), " (expected 0)"; call exit(1)
     end if
  end do

  ! Check the Cholesky diagonal of every batch entry
  do b = 1, batch
     call hipCheck(hipMemcpy(c_loc(hOut(1,1)), hostPtrs(b), int(N*N,c_size_t) * 8, hipMemcpyDeviceToHost))
     if (.not. (abs(hOut(1,1)-2.0d0) <= 1.0d-9 .and. abs(hOut(2,2)-3.0d0) <= 1.0d-9 .and. &
                abs(hOut(3,3)-4.0d0) <= 1.0d-9)) then
        write(*,*) "FAILED! batch ", b, " chol diag = ", hOut(1,1), hOut(2,2), hOut(3,3), " expected 2 3 4"; call exit(1)
     end if
  end do

  call hipCheck(hipFree(dA1)); call hipCheck(hipFree(dA2)); call hipCheck(hipFree(dPtrArray))
  call hipCheck(hipFree(dWork)); call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program dpotrfbatched
