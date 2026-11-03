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
! ssyevd example (symmetric eigenvalues, Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Checks sum(eigenvalues) == trace(A) and sum(eigenvalues**2) == ||A||_F**2.
! Native-array f2008 form; workspace via
! hipsolverSsyevd_bufferSize; devInfo device-backed.
!!!!!!!!!!!!!!
!
program hipsolver_ssyevd
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer(c_int), parameter :: N = 4, lda = 4
  real(c_float) :: hA(N,N) = reshape((/ &
      10.0, 2.0, 3.0, 6.0, 2.0, 11.0, 1.0, 0.0, &
      3.0, 1.0, 12.0, 2.0, 6.0, 0.0, 2.0, 13.0 /), (/N,N/))
  real(c_float) :: hD(N) = 0.0
  real(c_float), pointer :: dA(:,:)
  real(c_float), pointer :: dD(:)
  integer(c_int), pointer :: dInfo
  integer(c_int) :: hInfo
  type(c_ptr) :: dWork, handle = c_null_ptr
  integer(c_int) :: lwork
  real(c_float) :: trace_A, fro2_A, error
  real(c_float), parameter :: rtol = 1.0e-5
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_ssyevd' (Fortran 2008 interfaces) - "
  trace_A = real(hA(1,1)) + real(hA(2,2)) + real(hA(3,3)) + real(hA(4,4))
  fro2_A = sum(hA**2)
  call hipsolverCheck(hipsolverCreate(handle))
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dD, source=hD))
  call hipCheck(hipMalloc(dInfo))
  call hipsolverCheck(hipsolverSsyevd_bufferSize(handle, HIPSOLVER_EIG_MODE_NOVECTOR, &
                                                 HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dD, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 4, 1_c_size_t)))
  call hipsolverCheck(hipsolverSsyevd(handle, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                                      N, dA, lda, dD, dWork, lwork, c_loc(dInfo)))
  call hipCheck(hipMemcpy(hD, dD, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo /= 0) then
     write(*,*) "FAILED! info = ", hInfo, " (expected 0)"
     call exit(1)
  end if
  error = abs(sum(hD) - trace_A) / abs(trace_A)
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(eigenvalues) = ", sum(hD), " expected trace = ", trace_A
     call exit(1)
  end if
  error = abs(sum(hD**2) - fro2_A) / fro2_A
  if (.not. (error <= rtol)) then
     write(*,*) "FAILED! sum(eigenvalues**2) = ", sum(hD**2), " expected ||A||_F**2 = ", fro2_A
     call exit(1)
  end if
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dD)); call hipCheck(hipFree(dInfo)); call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  write(*,*) "PASSED!"
end program hipsolver_ssyevd
