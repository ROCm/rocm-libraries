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
! rocSOLVER library version query (Fortran 2008 interfaces)
!
! Exercises the pair rocsolver_get_version_string_size / rocsolver_get_version_string.
!
! rocsolver_get_version_string_size declares its `len` dummy as a bare
! `integer(c_size_t)` (by reference), so the Fortran variable is passed
! DIRECTLY -- not wrapped in c_loc. The test seeds that variable with the
! sentinel -1 and checks it really was written through, which is exactly the
! regression an incorrect out-argument classification would cause.
!
! rocsolver_get_version_string takes `type(c_ptr),value :: buf` and
! `integer(c_size_t),value :: len`, so the buffer address and the capacity are
! passed by value.
!
! Pass/fail: the reported minimum buffer size must be strictly positive, must
! fit in our fixed buffer, and must be large enough for the returned string
! plus its NUL terminator; the string itself must be non-empty and look like a
! dotted version number. Any violation prints "FAILED! ..." and stops with 1.
!
! No handle, no rocsolver_setup and no device memory are needed for this query,
! so the f2003 and f2008 variants are identical apart from the banner.
!!!!!!!!!!!!!!
program rocsolver_version
  use iso_c_binding
  use hip
  use rocsolver

  implicit none

  ! Comfortably larger than the size rocSOLVER is expected to report.
  integer(c_size_t), parameter :: buflen = 128

  integer(c_size_t) :: needed
  character(kind=c_char), target :: buf(buflen)
  character(kind=c_char, len=buflen) :: version
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'rocSOLVER version string' &
                              &(Fortran 2008 interfaces) - "

  ! Seed with a sentinel so we can prove the library wrote the out-argument.
  needed = -1
  call rocsolverCheck(rocsolver_get_version_string_size(needed))

  if (needed <= 0) then
     write(*,*) "FAILED! rocsolver_get_version_string_size did not write a positive size (got ", needed, ")"
     STOP 1
  end if

  if (needed > buflen) then
     write(*,*) "FAILED! rocsolver_get_version_string_size reports ", needed, &
                " bytes, which exceeds our buffer of ", buflen
     STOP 1
  end if

  ! The C API writes a NUL-terminated string into the buffer, so pass the
  ! address of the first element and the buffer capacity.
  buf(:) = c_null_char
  call rocsolverCheck(rocsolver_get_version_string(c_loc(buf(1)), buflen))

  ! Copy the C string into a Fortran character variable, stopping at the NUL.
  version = c_char_''
  do i = 1, buflen
     if (buf(i) == c_null_char) exit
     version(i:i) = buf(i)
  end do

  if (len_trim(version) == 0) then
     write(*,*) "FAILED! rocsolver_get_version_string returned an empty string"
     STOP 1
  end if

  ! Cross-check the two routines against each other: the advertised minimum
  ! buffer size must accommodate the string plus its NUL terminator.
  if (needed < len_trim(version) + 1) then
     write(*,*) "FAILED! reported minimum buffer size ", needed, &
                " is too small for the version string of length ", len_trim(version)
     STOP 1
  end if

  if (index(trim(version), '.') == 0) then
     write(*,*) "FAILED! rocsolver_get_version_string returned '", trim(version), &
                "', which is not a dotted version number"
     STOP 1
  end if

  write(*,*) "PASSED! rocSOLVER version: ", trim(version), " (minimum buffer size: ", needed, ")"

end program rocsolver_version
