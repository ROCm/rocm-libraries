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

! Exercises rocsparse_get_pointer_mode, whose 'pointer_mode' argument is a
! plain output scalar (passed directly, not through c_loc).
!
! The test performs three round trips on a freshly created handle:
!   1. the default pointer mode must be rocsparse_pointer_mode_host,
!   2. after rocsparse_set_pointer_mode(..., device) the getter must report device,
!   3. after switching back to host the getter must report host again, which proves
!      the getter genuinely re-reads the handle instead of latching a single value.
! Before every query 'pmode' is poisoned with -1 so that a binding which never
! writes the argument cannot accidentally pass. Any mismatch prints FAILED! and
! stops with a non-zero exit code. No device memory is involved.
program rocsparse_pointer_mode_test

    use iso_c_binding
    use hip
    use rocsparse

    implicit none

    type(c_ptr) :: handle
    integer(kind(rocsparse_pointer_mode_host)) :: pmode

    write(*,"(a)",advance="no") "-- Running test 'rocsparse_pointer_mode' (Fortran 2003 interfaces) - "

    call rocsparseCheck(rocsparse_create_handle(handle))

    ! 1. Freshly created handles default to host pointer mode.
    pmode = -1
    call rocsparseCheck(rocsparse_get_pointer_mode(handle, pmode))
    if (pmode /= rocsparse_pointer_mode_host) then
        write(*,*) "FAILED! expected default pointer mode ", rocsparse_pointer_mode_host, ", got ", pmode
        STOP 1
    end if

    ! 2. Switch to device pointer mode and read it back.
    pmode = -1
    call rocsparseCheck(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device))
    call rocsparseCheck(rocsparse_get_pointer_mode(handle, pmode))
    if (pmode /= rocsparse_pointer_mode_device) then
        write(*,*) "FAILED! expected pointer mode ", rocsparse_pointer_mode_device, ", got ", pmode
        STOP 1
    end if

    ! 3. Switch back to host pointer mode and read it back.
    pmode = -1
    call rocsparseCheck(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_host))
    call rocsparseCheck(rocsparse_get_pointer_mode(handle, pmode))
    if (pmode /= rocsparse_pointer_mode_host) then
        write(*,*) "FAILED! expected pointer mode ", rocsparse_pointer_mode_host, ", got ", pmode
        STOP 1
    end if

    call rocsparseCheck(rocsparse_destroy_handle(handle))

    write(*,"(a)") "PASSED! pointer mode round-trips host -> device -> host"

end program rocsparse_pointer_mode_test
