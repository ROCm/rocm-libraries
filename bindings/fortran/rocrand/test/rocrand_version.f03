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

! ROCRAND_FORTRAN_EXACT_VERSION is set only in-tree, where the module and the
! library come from the same headers; standalone a mismatch is only reported.
program rocrand_version_test

    use iso_c_binding
    use rocrand

    implicit none

    integer(c_int), parameter :: poison = -12345

    integer(c_int) :: version, version2
    integer(c_int) :: major, minor, patch
    integer :: exact_status

    write(*,"(a)",advance="no") "-- Running test 'rocRAND version' &
                                &(Fortran 2003 interfaces) - "

    version = poison
    version2 = poison

    call rocrandCheck(rocrand_get_version(version))
    call rocrandCheck(rocrand_get_version(version2))

    if (version == poison) then
       write(*,*) "FAILED! rocrand_get_version did not write the output argument"
       STOP 1
    end if

    if (version <= 0) then
       write(*,*) "FAILED! rocrand_get_version returned a non-positive code: ", version
       STOP 1
    end if

    major = version / 100000
    minor = mod(version / 100, 1000)
    patch = mod(version, 100)

    if (major < 1) then
       write(*,*) "FAILED! implausible rocRAND major version: ", major, " (code ", version, ")"
       STOP 1
    end if

    if (minor < 0 .or. minor > 999) then
       write(*,*) "FAILED! implausible rocRAND minor version: ", minor, " (code ", version, ")"
       STOP 1
    end if

    if (patch < 0 .or. patch > 99) then
       write(*,*) "FAILED! implausible rocRAND patch version: ", patch, " (code ", version, ")"
       STOP 1
    end if

    if (version2 /= version) then
       write(*,*) "FAILED! rocrand_get_version is not stable: ", version, " then ", version2
       STOP 1
    end if

    if (version /= ROCRAND_VERSION) then
       call get_environment_variable("ROCRAND_FORTRAN_EXACT_VERSION", status=exact_status)
       if (exact_status == 0) then
          write(*,*) "FAILED! rocrand_get_version returned ", version, &
                     " but the module's ROCRAND_VERSION is ", ROCRAND_VERSION
          STOP 1
       end if
       write(*,*) "note: the library reports ", version, &
                  ", the module was generated for ", ROCRAND_VERSION
    end if

    write(*,"(a,i0,a,i0,a,i0,a,i0,a)") " PASSED! rocRAND version: ", &
         major, ".", minor, ".", patch, " (code ", version, ")"

end program rocrand_version_test
