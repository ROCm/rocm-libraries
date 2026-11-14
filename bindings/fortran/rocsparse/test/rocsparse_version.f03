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

! Demonstrates the rocSPARSE library version queries, which need a handle:
! rocsparse_create_handle / rocsparse_get_version / rocsparse_get_git_rev /
! rocsparse_destroy_handle. No device memory is touched.
!
! 'version' is an output scalar, so it is declared as a plain
! integer(c_int) and passed directly (NOT wrapped in c_loc). 'rev' is a
! caller-supplied character buffer that the C API fills in, so it is still
! declared type(c_ptr),value and c_loc() IS required for it.
!
! The test passes if the encoded version decomposes into a plausible
! major.minor.patch triple and the git revision string is not empty.
program rocsparse_version
  use iso_c_binding
  use hip
  use rocsparse

  implicit none

  ! rocsparse_get_git_rev writes a NUL-terminated SHA-1 into the buffer.
  integer, parameter :: buflen = 256

  type(c_ptr) :: handle
  integer(c_int) :: version
  character(kind=c_char), target :: buf(buflen)
  character(kind=c_char, len=buflen) :: rev
  integer :: major, minor, patch, i

  write(*,"(a)",advance="no") "-- Running test 'rocSPARSE version' &
                              &(Fortran 2003 interfaces) - "

  call rocsparseCheck(rocsparse_create_handle(handle))

  ! Output scalar: passed by reference, no c_loc.
  version = -1
  call rocsparseCheck(rocsparse_get_version(handle, version))

  ! Caller-supplied char buffer: pass the address of the first element.
  buf(:) = c_null_char
  call rocsparseCheck(rocsparse_get_git_rev(handle, c_loc(buf(1))))

  call rocsparseCheck(rocsparse_destroy_handle(handle))

  ! Copy the C string into a Fortran character variable, stopping at the NUL.
  rev = c_char_''
  do i = 1, buflen
     if (buf(i) == c_null_char) exit
     rev(i:i) = buf(i)
  end do

  if (version <= 0) then
     write(*,*) "FAILED! rocsparse_get_version returned ", version
     STOP 1
  end if

  ! rocSPARSE encodes the version as major * 100000 + minor * 100 + patch.
  patch = mod(version, 100)
  minor = mod(version / 100, 1000)
  major = version / 100000

  if (major < 1 .or. major > 99) then
     write(*,*) "FAILED! implausible rocSPARSE major version ", major
     STOP 1
  end if

  if (minor < 0 .or. minor > 999) then
     write(*,*) "FAILED! implausible rocSPARSE minor version ", minor
     STOP 1
  end if

  if (len_trim(rev) == 0) then
     write(*,*) "FAILED! rocsparse_get_git_rev returned an empty string"
     STOP 1
  end if

  write(*,"(a,i0,a,i0,a,i0,a,a,a)") "PASSED! rocSPARSE version ", major, ".", &
        minor, ".", patch, " (git rev ", trim(rev), ")"

end program rocsparse_version
