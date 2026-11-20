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
! hipsolver dpotrf example (Cholesky factorization, Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Native-array f2008 form: A is a Fortran array pointer; the workspace is a
! type(c_ptr) sized by hipsolverDpotrf_bufferSize; devInfo is device-backed.
!!!!!!!!!!!!!!
!
program hipsolver_dpotrf
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer :: i, j
  integer(c_int), parameter :: N = 3, lda = 3
  real(c_double) :: hA(3,3) = reshape((/4, 12, -16, 12, 37, -43, -16, -43, 98/), (/3, 3/))
  real(c_double) :: hResult(3,3) = reshape((/&
     2.0000000000000000d0, 12.0000000000000000d0, -16.0000000000000000d0, &
     6.0000000000000000d0,  1.0000000000000000d0, -43.0000000000000000d0, &
    -8.0000000000000000d0,  5.0000000000000000d0,   3.0000000000000000d0/), (/3, 3/))
  real(c_double), pointer :: dA(:,:)
  integer(c_int), pointer :: dInfo
  integer(c_int) :: hInfo
  type(c_ptr) :: dWork, handle = c_null_ptr
  integer(c_int) :: lwork
  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_dpotrf' (Fortran 2008 interfaces) - "
  call hipsolverCheck(hipsolverCreate(handle))
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dInfo))
  call hipsolverCheck(hipsolverDpotrf_bufferSize(handle, HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 8, 1_c_size_t)))
  call hipsolverCheck(hipsolverDpotrf(handle, HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dWork, lwork, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo /= 0) then
    write(*,*) "FAILED! info = ", hInfo, " (expected 0)"
    call exit(1)
  end if
  call hipCheck(hipMemcpy(hA, dA, hipMemcpyDeviceToHost))
  do j = 1,3
    do i = 1,3
      error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0_c_double)
      if (.not. (error <= error_max)) then
        write(*,*) "FAILED! hA(", i, ",", j, ") = ", hA(i,j)
        call exit(1)
      end if
    end do
  end do
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dInfo)); call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  write(*,*) "PASSED!"
end program hipsolver_dpotrf
