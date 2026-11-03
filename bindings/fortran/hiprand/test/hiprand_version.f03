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

! hipRAND on ROCm reports rocRAND's version, so the module's HIPRAND_VERSION is
! checked against HIPRAND_FORTRAN_EXPECTED_VERSION (set in-tree) instead.
program hiprand_version_test

    use iso_c_binding
    use hiprand

    implicit none

    integer(c_int), parameter :: poison = -12345

    integer(c_int) :: version, version2
    integer(c_int) :: major, minor, patch
    character(len=32) :: expected_str
    integer :: expected, env_status, io_status

    write(*,"(a)",advance="no") "-- Running test 'hipRAND version' &
                                &(Fortran 2003 interfaces) - "

    version  = poison
    version2 = poison

    call hiprandCheck(hiprandGetVersion(version))
    call hiprandCheck(hiprandGetVersion(version2))

    if (version == poison) then
       write(*,*) "FAILED! hiprandGetVersion did not write the version argument"
       STOP 1
    end if

    if (version <= 0) then
       write(*,*) "FAILED! hiprandGetVersion returned a non-positive version code: ", version
       STOP 1
    end if

    if (version2 /= version) then
       write(*,*) "FAILED! hiprandGetVersion is not stable: ", version, " then ", version2
       STOP 1
    end if

    ! HIPRAND_VERSION / 100000 is major, / 100 % 1000 is minor, % 100 is patch.
    major = version / 100000
    minor = mod(version / 100, 1000)
    patch = mod(version, 100)

    if (major < 1 .or. major > 99) then
       write(*,*) "FAILED! implausible hipRAND major version: ", major
       STOP 1
    end if

    if (minor < 0 .or. minor > 999 .or. patch < 0 .or. patch > 99) then
       write(*,*) "FAILED! implausible hipRAND minor/patch version: ", minor, patch
       STOP 1
    end if

    call get_environment_variable("HIPRAND_FORTRAN_EXPECTED_VERSION", expected_str, &
                                  status=env_status)
    if (env_status == 0) then
       read(expected_str, *, iostat=io_status) expected
       if (io_status /= 0 .or. expected /= HIPRAND_VERSION) then
          write(*,*) "FAILED! the module's HIPRAND_VERSION ", HIPRAND_VERSION, &
                     " is not this tree's version ", trim(expected_str)
          STOP 1
       end if
    end if

    write(*,"(a,i0,a,i0,a,i0,a,i0,a)") " PASSED! hipRAND version: ", &
        major, ".", minor, ".", patch, " (code ", version, ")"

end program hiprand_version_test
