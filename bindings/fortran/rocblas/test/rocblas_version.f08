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

! Exercises the rocBLAS version query pair:
!
!   * rocblas_get_version_string_size(len) - 'len' is a bare (by reference)
!     integer(c_size_t) output dummy, so the variable is passed directly and
!     NOT wrapped in c_loc().
!   * rocblas_get_version_string(buf,len) - 'buf' is type(c_ptr),value and
!     'len' is integer(c_size_t),value, so a c_loc() and a by-value capacity.
!
! The test passes when the two routines agree with each other: the number of
! characters rocBLAS writes before the NUL terminator must be exactly
! len-1 as reported by rocblas_get_version_string_size, the string must look
! like a version (leading digit, at least two '.'), and an undersized buffer
! must be rejected with rocblas_status_invalid_size. Any violation prints
! "FAILED! <why>" and stops with exit code 1.
program rocblas_version
  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(c_size_t), parameter :: buflen = 128

  character(kind=c_char), target :: buf(buflen)
  character(kind=c_char), target :: small(4)
  character(kind=c_char, len=buflen) :: version
  integer(c_size_t) :: vlen
  integer(kind(rocblas_status_success)) :: st
  integer :: i, nchars, ndots

  write(*,"(a)",advance="no") "-- Running test 'rocBLAS version string' &
                              &(Fortran 2008 interfaces) - "

  ! Query the minimum buffer size first. 'len' is an output scalar passed by
  ! reference, so hand over the variable itself.
  vlen = 0
  call rocblasCheck(rocblas_get_version_string_size(vlen))

  if (vlen <= 0 .or. vlen >= buflen) then
     write(*,*) "FAILED! rocblas_get_version_string_size returned an implausible length"
     STOP 1
  end if

  ! The C API writes a NUL-terminated string into the buffer, so pass the
  ! address of the first element and the buffer capacity by value.
  buf(:) = c_null_char
  call rocblasCheck(rocblas_get_version_string(c_loc(buf(1)), buflen))

  ! Copy the C string into a Fortran character variable, stopping at the NUL.
  version = c_char_''
  nchars = 0
  do i = 1, int(buflen)
     if (buf(i) == c_null_char) exit
     version(i:i) = buf(i)
     nchars = nchars + 1
  end do

  if (nchars == 0) then
     write(*,*) "FAILED! rocblas_get_version_string returned an empty string"
     STOP 1
  end if

  ! Cross-check the two routines against each other: the reported size must
  ! account for exactly the written characters plus the NUL terminator.
  if (int(nchars,c_size_t) + 1_c_size_t /= vlen) then
     write(*,*) "FAILED! version string length disagrees with rocblas_get_version_string_size"
     STOP 1
  end if

  ! Sanity-check that the payload really looks like a version number.
  ndots = 0
  do i = 1, nchars
     if (version(i:i) == '.') ndots = ndots + 1
  end do

  if (version(1:1) < '0' .or. version(1:1) > '9' .or. ndots < 2) then
     write(*,*) "FAILED! version string is not of the form X.Y.Z"
     STOP 1
  end if

  ! Negative case: an undersized buffer must be rejected. Use the raw status
  ! here, rocblasCheck would stop on any non-success code.
  small(:) = c_null_char
  st = rocblas_get_version_string(c_loc(small(1)), 4_c_size_t)

  if (st /= rocblas_status_invalid_size) then
     write(*,*) "FAILED! undersized buffer did not return rocblas_status_invalid_size"
     STOP 1
  end if

  write(*,*) "PASSED! rocBLAS version: ", trim(version)

end program rocblas_version
