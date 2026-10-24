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
! hipsolver dpotrfBatched example (batched Cholesky factorization, double, Fortran 2003)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Factorizes a batch of SPD matrices. The batched API takes A as an array
! of device pointers that itself lives in DEVICE memory: each matrix is allocated
! on the device, their device addresses are collected in a host array, and that
! array is copied to a device buffer whose address (by value) is passed as A.
! Batch b holds b**2 * U**H*U, so its upper factor is b*U. Checks info == 0
! (seeded with -1) and the upper triangle of every factor.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program dpotrfbatched
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer :: b, i, j
  integer(c_int), parameter :: N = 3, lda = 3, batch = 2
  ! Upper Cholesky factor with a positive diagonal; A = U**T * U.
  real(c_double), parameter :: hU(N,N) = reshape((/ &
      2, 0, 0, &
      1, 3, 0, &
      1, 1, 4/), (/N,N/))
  real(c_double), target :: hA(N,N,batch), hOut(N,N)
  integer(c_int), target :: hInfo(batch)
  type(c_ptr), target :: hostPtrs(batch)
  type(c_ptr) :: dPtrArray, dWork, dInfo
  type(c_ptr) :: handle
  integer(c_int) :: lwork
  integer(c_size_t) :: ptrbytes, matbytes
  real(c_double) :: error
  real(c_double), parameter :: rtol = 1.0d-12
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_dpotrfbatched' (Fortran 2003 interfaces) - "

  matbytes = int(N*N,c_size_t) * 8
  call hipsolverCheck(hipsolverCreate(handle))
  do b = 1, batch
     hA(:,:,b) = b**2 * matmul(transpose(hU), hU)
     call hipCheck(hipMalloc(hostPtrs(b), matbytes))
     call hipCheck(hipMemcpy(hostPtrs(b), c_loc(hA(1,1,b)), matbytes, hipMemcpyHostToDevice))
  end do

  ! Host array of device addresses, copied to a device-resident pointer array.
  ptrbytes = int(batch,c_size_t) * c_sizeof(c_null_ptr)
  call hipCheck(hipMalloc(dPtrArray, ptrbytes))
  call hipCheck(hipMemcpy(dPtrArray, c_loc(hostPtrs), ptrbytes, hipMemcpyHostToDevice))

  hInfo = -1
  call hipCheck(hipMalloc(dInfo, int(batch,c_size_t) * 4))
  call hipCheck(hipMemcpy(dInfo, c_loc(hInfo(1)), int(batch,c_size_t) * 4, hipMemcpyHostToDevice))

  call hipsolverCheck(hipsolverDpotrfBatched_bufferSize(handle, HIPSOLVER_FILL_MODE_UPPER, &
       N, dPtrArray, lda, lwork, batch))
  call hipCheck(hipMalloc(dWork, int(max(lwork,1),c_size_t) * 8))
  call hipsolverCheck(hipsolverDpotrfBatched(handle, HIPSOLVER_FILL_MODE_UPPER, &
       N, dPtrArray, lda, dWork, lwork, dInfo, batch))
  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, int(batch,c_size_t) * 4, hipMemcpyDeviceToHost))
  do b = 1, batch
     if (hInfo(b) /= 0) then
        write(*,*) "FAILED! info(", b, ") = ", hInfo(b), " (expected 0)"; call exit(1)
     end if
  end do

  do b = 1, batch
     call hipCheck(hipMemcpy(c_loc(hOut(1,1)), hostPtrs(b), matbytes, hipMemcpyDeviceToHost))
     do j = 1, N
        do i = 1, j
           error = abs(hOut(i,j) - b * hU(i,j)) / max(abs(b * hU(i,j)), 1.0_c_double)
           if (.not. (error <= rtol)) then
              write(*,*) "FAILED! batch ", b, " U(", i, ",", j, ") = ", hOut(i,j), &
                         " expected ", b * hU(i,j)
              call exit(1)
           end if
        end do
     end do
  end do

  do b = 1, batch
     call hipCheck(hipFree(hostPtrs(b)))
  end do
  call hipCheck(hipFree(dPtrArray)); call hipCheck(hipFree(dWork)); call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program dpotrfbatched
