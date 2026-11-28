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
! hipSPARSE library version / git revision query (Fortran 2003)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Exercises the pure "ask the library about itself" entry points:
!   hipsparseCreate -> hipsparseGetVersion -> hipsparseGetGitRevision -> hipsparseDestroy
!
! hipsparseGetVersion takes its 'version' output as a BARE integer(c_int) dummy,
! so the variable is passed directly, NOT as c_loc(version). A non-zero decoded
! version therefore proves the scalar really was written through the binding.
! hipsparseGetGitRevision takes a type(c_ptr),value buffer, so that one needs c_loc.
!
! Passes when version > 0, the decoded major/minor/patch are in range
! (1 <= major < 100, 0 <= minor < 1000, 0 <= patch < 100), and the git revision
! string is non-empty. Fails (STOP 1) otherwise.
!
! This test touches no device memory.
!!!!!!!!!!!!!!
!
program hipsparse_version
  use iso_c_binding
  use hip
  use hipsparse
  implicit none

  integer, parameter :: revlen = 128

  type(c_ptr) :: handle = c_null_ptr
  integer(c_int) :: version
  integer(c_int) :: major, minor, patch
  character(kind=c_char,len=1), target :: rev(revlen)
  character(len=revlen) :: revstr
  integer :: i, n

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_version' (Fortran 2003 interfaces) - "

  call hipsparseCheck(hipsparseCreate(handle))

  ! ---- version -------------------------------------------------------------
  ! 'version' is a bare integer dummy: pass it directly, no c_loc.
  version = -1
  call hipsparseCheck(hipsparseGetVersion(handle, version))

  if (version <= 0) then
     write(*,*) "FAILED! hipsparseGetVersion did not write a positive version, got ", version
     STOP 1
  end if

  major = version / 100000
  minor = mod(version / 100, 1000)
  patch = mod(version, 100)

  if (major < 1 .or. major >= 100) then
     write(*,*) "FAILED! implausible hipSPARSE major version ", major
     STOP 1
  end if
  if (minor < 0 .or. minor >= 1000) then
     write(*,*) "FAILED! implausible hipSPARSE minor version ", minor
     STOP 1
  end if
  if (patch < 0 .or. patch >= 100) then
     write(*,*) "FAILED! implausible hipSPARSE patch version ", patch
     STOP 1
  end if

  ! ---- git revision --------------------------------------------------------
  ! 'rev' is a type(c_ptr),value dummy: the C side writes a NUL-terminated
  ! string into the caller-provided buffer, so pass its address.
  rev(:) = c_null_char
  call hipsparseCheck(hipsparseGetGitRevision(handle, c_loc(rev)))

  revstr = ''
  n = 0
  do i = 1, revlen
     if (rev(i) == c_null_char) exit
     revstr(i:i) = rev(i)
     n = n + 1
  end do

  if (n == 0 .or. len_trim(revstr) == 0) then
     write(*,*) "FAILED! hipsparseGetGitRevision returned an empty revision string"
     STOP 1
  end if

  call hipsparseCheck(hipsparseDestroy(handle))

  write(*,"(a,i0,a,i0,a,i0,a,a)") " PASSED! hipSPARSE ", major, ".", minor, ".", patch, &
       ", git revision ", trim(revstr)

end program hipsparse_version
