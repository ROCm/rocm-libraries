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
! dsytf2 example (unblocked Bunch-Kaufman factorization)
! Factorizes a symmetric matrix and checks the algorithm succeeded (info == 0).
! rocSOLVER writes info to DEVICE memory, so it is read back from a device
! allocation (this is what the test primarily exercises).
!!!!!!!!!!!!!!
!
program dsytf2
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer(c_int), parameter :: N = 4, lda = 4
  real(c_double) :: hA(N,N) = reshape((/ &
      10.0d0,  2.0d0,  3.0d0,  6.0d0, &
       2.0d0, 11.0d0,  1.0d0,  0.0d0, &
       3.0d0,  1.0d0, 12.0d0,  2.0d0, &
       6.0d0,  0.0d0,  2.0d0, 13.0d0/), (/N,N/))
  integer(c_int) :: hIpiv(N), hInfo(1)
  real(c_double), pointer :: dA(:,:)
  integer(c_int), pointer :: dIpiv(:)
  integer(c_int), pointer :: dInfo(:)
  type(c_ptr) :: handle
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsytf2' (Fortran 2008 interfaces) - "
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dIpiv, mold=hIpiv))
  call hipCheck(hipMalloc(dInfo, 1))
  call rocblasCheck(rocblas_create_handle(handle))
  call rocsolverCheck(rocsolver_dsytf2(handle, rocblas_fill_upper, N, dA, lda, dIpiv, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hIpiv, dIpiv, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
     write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"
     call exit(1)
  end if
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dIpiv)); call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program dsytf2
