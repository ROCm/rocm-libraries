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
! dposv example (rocSOLVER)
! see: https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Solves the symmetric positive-definite system A*X = B (X overwrites B).
! rocSOLVER writes `info` to DEVICE memory, so it is backed by a device
! allocation and passed as c_loc(dInfo).
!!!!!!!!!!!!!!
!
program dposv
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: n = 3, nrhs = 1, lda = 3, ldb = 3

  ! SPD matrix A (column-major) and RHS B chosen so the solution is X = [1,1,1].
  real(c_double) :: hA(n,n) = reshape([ &
      2.0d0, 1.0d0, 0.0d0, &
      1.0d0, 2.0d0, 1.0d0, &
      0.0d0, 1.0d0, 2.0d0], [n,n])
  real(c_double) :: hB(n,nrhs) = reshape([3.0d0, 4.0d0, 3.0d0], [n,nrhs])
  real(c_double) :: hX_ref(n) = [1.0d0, 1.0d0, 1.0d0]

  real(c_double), pointer :: dA(:,:)   ! GPU buffer for A
  real(c_double), pointer :: dB(:,:)   ! GPU buffer for B (holds X on output)
  integer(c_int), pointer :: dInfo(:)  ! GPU buffer for info
  integer(c_int) :: hInfo(1)

  type(c_ptr) :: handle ! rocblas_handle

  integer :: i
  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dposv' (Fortran 2008 interfaces) - "

  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dB,    source=hB))
  call hipCheck(hipMalloc(dInfo, 1))

  call rocblasCheck(rocblas_create_handle(handle))

  ! A/B passed as native Fortran device arrays; info as a device pointer.
  call rocsolverCheck(rocsolver_dposv(handle, rocblas_fill_upper, n, nrhs, dA, lda, dB, ldb, &
                                      c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
    write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hB, dB, hipMemcpyDeviceToHost))

  do i = 1, n
     error = abs(hB(i,1) - hX_ref(i)) / max(abs(hX_ref(i)), 1.0d0)
     if (.not. (error <= error_max)) then
        write(*,*) "FAILED! X(", i, ") = ", hB(i,1), " expected ", hX_ref(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dposv
