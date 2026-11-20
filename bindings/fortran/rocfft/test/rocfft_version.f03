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

! Demonstrates rocfft_get_version_string, the rocFFT library version query.
program rocfft_version
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  ! rocfft_get_version_string requires a buffer of at least 30 characters.
  integer(c_size_t), parameter :: buflen = 64

  character(kind=c_char), target :: buf(buflen)
  character(kind=c_char, len=buflen) :: version
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'rocFFT version string' &
                              &(Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_setup())

  ! The C API writes a NUL-terminated string into the buffer, so pass the
  ! address of the first element and the buffer capacity.
  buf(:) = c_null_char
  call rocfftCheck(rocfft_get_version_string(c_loc(buf(1)), buflen))

  call rocfftCheck(rocfft_cleanup())

  ! Copy the C string into a Fortran character variable, stopping at the NUL.
  version = c_char_''
  do i = 1, buflen
     if (buf(i) == c_null_char) exit
     version(i:i) = buf(i)
  end do

  if (len_trim(version) == 0) then
     write(*,*) "FAILED! rocfft_get_version_string returned an empty string"
     STOP 1
  end if

  write(*,*) "PASSED! rocFFT version: ", trim(version)

end program rocfft_version
