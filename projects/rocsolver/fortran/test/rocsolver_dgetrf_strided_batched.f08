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

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! rocsolver_dgetrf_strided_batched example / unit test
!
! Computes the LU factorization of a batch of matrices on the GPU.
!
! This test guards the fix for the batched LU "info" argument (myInfo):
! in rocSOLVER the batched routines write an array of `batch_count` integers
! to a *device* pointer, so the Fortran binding must declare `myInfo` as
! type(c_ptr). Previously it was `integer(c_int)`, which can only represent a
! single scalar passed by reference and cannot point at a device array.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
program dgetrf_strided_batched
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: M = 3, N = 3, lda = 3
  integer(c_int), parameter :: batch_count = 2
  integer(c_int), parameter :: sz_piv = min(M, N)
  integer(c_int64_t), parameter :: strideA = int(lda, c_int64_t) * int(N, c_int64_t)
  integer(c_int64_t), parameter :: strideP = int(sz_piv, c_int64_t)

  ! Two invertible 3x3 matrices stored back-to-back (column-major).
  ! Columns 1..N     -> batch entry 0
  ! Columns N+1..2*N -> batch entry 1
  real(c_double) :: hA(lda, N*batch_count)
  integer(c_int) :: hInfo(batch_count)
  integer(c_int) :: hIpiv(sz_piv*batch_count)

  type(c_ptr)             :: handle = c_null_ptr
  real(c_double), pointer :: dA(:,:)  => null()  ! GPU buffer for the strided matrices
  integer(c_int), pointer :: dIpiv(:) => null()  ! GPU buffer for the pivot indices
  integer(c_int), pointer :: dInfo(:) => null()  ! GPU buffer for the info array

  integer :: b

  write(*,"(a)",advance="no") &
    "-- Running test 'rocsolver_dgetrf_strided_batched' (Fortran 2008 interfaces) - "

  hA(:, 1:N)       = reshape((/ 12.d0, -51.d0,   4.d0, &
                                 6.d0, 167.d0, -68.d0, &
                                -4.d0,  24.d0, -41.d0/), (/lda, N/))
  hA(:, N+1:2*N)   = reshape((/  2.d0,   0.d0,   0.d0, &
                                 0.d0,   3.d0,   0.d0, &
                                 0.d0,   0.d0,   5.d0/), (/lda, N/))

  ! Create rocBLAS handle
  call rocblasCheck(rocblas_create_handle(handle))

  ! Allocate device-side memory & copy the input matrices to the device
  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dIpiv, mold=hIpiv))
  call hipCheck(hipMalloc(dInfo, mold=hInfo))

  ! Compute the batched LU factorization on the device.
  ! `dInfo` is passed as a device pointer to an array of `batch_count` ints,
  ! which is exactly what the myInfo -> c_ptr binding fix enables.
  call rocsolverCheck(rocsolver_dgetrf_strided_batched(handle, M, N, dA, lda, strideA, &
                                                 dIpiv, strideP, c_loc(dInfo), batch_count))

  ! Copy the info array back to the host
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))

  ! For invertible inputs rocSOLVER sets info(b) == 0 for every batch entry.
  do b = 1, batch_count
    if (hInfo(b) /= 0) then
      write(*,*) "FAILED! info(", b, ") = ", hInfo(b), " (expected 0)"
      call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dgetrf_strided_batched
