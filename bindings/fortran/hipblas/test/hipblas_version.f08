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

! Exercises the hipBLAS version/property query API:
!   hipblasCreate / hipblasGetVersion / hipblasGetProperty / hipblasDestroy.
!
! hipblasGetVersion now takes an `integer(c_int) :: version` output dummy by
! reference, so the variable is passed DIRECTLY (no c_loc).
! The variable is likewise passed directly to hipblasGetProperty, through the generic's _typed form.
!
! The test passes when:
!   1. hipblasGetVersion overwrites the sentinel with a positive version,
!   2. that packed version decomposes into a plausible major/minor/patch,
!   3. the decomposition agrees with the three hipblasGetProperty queries
!      (two independent APIs must report the same version), and
!   4. hipblasGetVersion with a NULL handle (explicitly allowed) agrees too.
! Any mismatch prints "FAILED! ..." and stops with exit code 1.
program hipblas_version
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  type(c_ptr) :: handle = c_null_ptr
  integer(c_int) :: version, version2
  integer(c_int) :: major, minor, patch
  integer(c_int), target :: pmajor, pminor, ppatch

  write(*,"(a)",advance="no") "-- Running test 'hipblas_version' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  ! Sentinel: if the binding were still `type(c_ptr),value`, the integer would
  ! be reinterpreted as an address and `version` would stay at -1 (or crash).
  version = -1
  call hipblasCheck(hipblasGetVersion(handle, version))

  if (version <= 0) then
     write(*,*) "FAILED! hipblasGetVersion did not fill in the version, got ", version
     STOP 1
  end if

  ! The packed encoding is major*10000 + minor*100 + patch.
  major = version / 10000
  minor = mod(version, 10000) / 100
  patch = mod(version, 100)

  if (major < 1 .or. minor < 0 .or. patch < 0) then
     write(*,*) "FAILED! implausible hipBLAS version decomposition ", major, minor, patch
     STOP 1
  end if

  ! hipblasGetProperty needs no handle and writes through a pointer.
  pmajor = -1
  pminor = -1
  ppatch = -1
  call hipblasCheck(hipblasGetProperty(HIPBLAS_MAJOR_VERSION, pmajor))
  call hipblasCheck(hipblasGetProperty(HIPBLAS_MINOR_VERSION, pminor))
  call hipblasCheck(hipblasGetProperty(HIPBLAS_PATCH_LEVEL, ppatch))

  if (major /= pmajor .or. minor /= pminor .or. patch /= ppatch) then
     write(*,*) "FAILED! hipblasGetVersion and hipblasGetProperty disagree: ", &
                major, minor, patch, " vs ", pmajor, pminor, ppatch
     STOP 1
  end if

  ! The handle argument of hipblasGetVersion is documented as optional (NULL).
  version2 = -1
  call hipblasCheck(hipblasGetVersion(c_null_ptr, version2))

  if (version2 /= version) then
     write(*,*) "FAILED! hipblasGetVersion with a NULL handle returned ", version2, &
                " instead of ", version
     STOP 1
  end if

  call hipblasCheck(hipblasDestroy(handle))

  write(*,"(a,i0,a,i0,a,i0,a,i0,a)") " PASSED! hipBLAS version: ", major, ".", minor, ".", &
                                     patch, " (raw ", version, ")"

end program hipblas_version
