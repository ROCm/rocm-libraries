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
! hipsparseGetPointerMode / hipsparseSetPointerMode round trip
! (Fortran 2003 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! hipsparseSetPointerMode takes the mode by value, while hipsparseGetPointerMode
! receives it as a plain scalar output argument (not a type(c_ptr)), so the
! variable is passed directly rather than wrapped in c_loc.
!
! The test asserts that the handle starts in HOST pointer mode, that switching
! to DEVICE mode is observed by the getter, and that switching back to HOST is
! observed as well. Before every query the local variable is reset to a -1
! sentinel, so a getter that silently writes nothing is detected instead of
! being hidden by a value left over from the preceding step.
!
! No device memory is involved.
!!!!!!!!!!!!!!
!
program hipsparse_pointer_mode
  use iso_c_binding
  use hip
  use hipsparse

  implicit none

  type(c_ptr) :: handle = c_null_ptr
  integer(kind(HIPSPARSE_POINTER_MODE_HOST)) :: mode

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_pointer_mode' &
                              &(Fortran 2003 interfaces) - "

  call hipsparseCheck(hipsparseCreate(handle))

  ! 1. A freshly created handle must default to host pointer mode.
  mode = -1
  call hipsparseCheck(hipsparseGetPointerMode(handle, mode))
  if (mode /= HIPSPARSE_POINTER_MODE_HOST) then
     write(*,"(a,i0,a,i0)") "FAILED! expected the default pointer mode HOST (", &
                            HIPSPARSE_POINTER_MODE_HOST, "), got ", mode
     call hipsparseCheck(hipsparseDestroy(handle))
     STOP 1
  end if

  ! 2. Switch to device pointer mode and read it back.
  call hipsparseCheck(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE))
  mode = -1
  call hipsparseCheck(hipsparseGetPointerMode(handle, mode))
  if (mode /= HIPSPARSE_POINTER_MODE_DEVICE) then
     write(*,"(a,i0,a,i0)") "FAILED! expected pointer mode DEVICE (", &
                            HIPSPARSE_POINTER_MODE_DEVICE, "), got ", mode
     call hipsparseCheck(hipsparseDestroy(handle))
     STOP 1
  end if

  ! 3. Switch back to host pointer mode and read it back.
  call hipsparseCheck(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST))
  mode = -1
  call hipsparseCheck(hipsparseGetPointerMode(handle, mode))
  if (mode /= HIPSPARSE_POINTER_MODE_HOST) then
     write(*,"(a,i0,a,i0)") "FAILED! expected pointer mode HOST (", &
                            HIPSPARSE_POINTER_MODE_HOST, "), got ", mode
     call hipsparseCheck(hipsparseDestroy(handle))
     STOP 1
  end if

  call hipsparseCheck(hipsparseDestroy(handle))

  write(*,"(a,i0)") "PASSED! pointer mode round-trips HOST -> DEVICE -> HOST, final mode = ", mode

end program hipsparse_pointer_mode
