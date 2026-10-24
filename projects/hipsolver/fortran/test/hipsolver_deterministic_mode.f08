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
! hipSOLVER handle attribute round-trip (Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Exercises the pure "query the handle" entry points, which have no test
! coverage otherwise:
!   hipsolverSetDeterministicMode / hipsolverGetDeterministicMode
!   hipsolverDnSetDeterministicMode / hipsolverDnGetDeterministicMode
!   hipsolverSetStream / hipsolverGetStream
!   hipsolverDnSetStream / hipsolverDnGetStream
!
! This test touches no device memory, so the Fortran 2003 and Fortran 2008
! sources are identical apart from the banner (same as the rocfft_version
! precedent).
!
! The getters take their output argument BY REFERENCE (a bare
! integer(kind(HIPSOLVER_DETERMINISTIC_RESULTS)) and a bare type(c_ptr)),
! so the variable is passed directly, never wrapped in c_loc.
!
! Pass/fail: the mode variable is poisoned with -99 before every get, so a
! getter that writes nothing is caught. Each set/get pair must round-trip the
! exact enum that was stored, the Dn aliases must observe the same handle
! state as the non-Dn entry points, and the stream pointer must come back
! both as the stream that was set and as NULL after the stream is cleared.
! Any mismatch prints "FAILED! ..." and stops with exit code 1.
!!!!!!!!!!!!!!
!
program hipsolver_deterministic_mode
  use iso_c_binding
  use hip
  use hipsolver

  implicit none

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: stream = c_null_ptr
  type(c_ptr) :: got_stream = c_null_ptr
  integer(kind(HIPSOLVER_DETERMINISTIC_RESULTS)) :: mode

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_deterministic_mode' &
                              &(Fortran 2008 interfaces) - "

  ! Sanity check: the two enumerators must be distinct, otherwise every
  ! round-trip assertion below would be vacuous.
  if (HIPSOLVER_DETERMINISTIC_RESULTS == HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS) then
     write(*,"(a)") "FAILED! deterministic-mode enum values are not distinct"
     STOP 1
  end if

  call hipsolverCheck(hipsolverCreate(handle))

  ! The freshly created handle must report one of the two legal modes, and in
  ! doing so must overwrite the poison value.
  mode = -99
  call hipsolverCheck(hipsolverGetDeterministicMode(handle, mode))
  if (mode /= HIPSOLVER_DETERMINISTIC_RESULTS .and. &
      mode /= HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS) then
     write(*,"(a,i0)") "FAILED! hipsolverGetDeterministicMode returned an invalid enum value: ", mode
     STOP 1
  end if

  ! Round-trip A: switch to the non-deterministic mode and read it back.
  call hipsolverCheck(hipsolverSetDeterministicMode(handle, HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS))
  mode = -99
  call hipsolverCheck(hipsolverGetDeterministicMode(handle, mode))
  if (mode /= HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS) then
     write(*,"(a,i0)") "FAILED! expected ALLOW_NON_DETERMINISTIC_RESULTS, got ", mode
     STOP 1
  end if

  ! Round-trip B: switch back, proving the getter tracks the handle state and
  ! is not simply returning a constant.
  call hipsolverCheck(hipsolverSetDeterministicMode(handle, HIPSOLVER_DETERMINISTIC_RESULTS))
  mode = -99
  call hipsolverCheck(hipsolverGetDeterministicMode(handle, mode))
  if (mode /= HIPSOLVER_DETERMINISTIC_RESULTS) then
     write(*,"(a,i0)") "FAILED! expected DETERMINISTIC_RESULTS, got ", mode
     STOP 1
  end if

  ! Cross-alias check: hipsolverDn* must address the same handle state.
  call hipsolverCheck(hipsolverDnSetDeterministicMode(handle, HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS))
  mode = -99
  call hipsolverCheck(hipsolverGetDeterministicMode(handle, mode))
  if (mode /= HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS) then
     write(*,"(a,i0)") "FAILED! hipsolverDnSetDeterministicMode was not seen by hipsolverGetDeterministicMode, got ", mode
     STOP 1
  end if

  call hipsolverCheck(hipsolverSetDeterministicMode(handle, HIPSOLVER_DETERMINISTIC_RESULTS))
  mode = -99
  call hipsolverCheck(hipsolverDnGetDeterministicMode(handle, mode))
  if (mode /= HIPSOLVER_DETERMINISTIC_RESULTS) then
     write(*,"(a,i0)") "FAILED! hipsolverDnGetDeterministicMode disagrees with hipsolverSetDeterministicMode, got ", mode
     STOP 1
  end if

  ! Stream round-trip. A fresh handle uses the NULL (default) stream.
  got_stream = c_null_ptr
  call hipsolverCheck(hipsolverGetStream(handle, got_stream))
  if (c_associated(got_stream)) then
     write(*,"(a)") "FAILED! a fresh hipSOLVER handle should report the NULL default stream"
     STOP 1
  end if

  call hipCheck(hipStreamCreate(stream))
  if (.not. c_associated(stream)) then
     write(*,"(a)") "FAILED! hipStreamCreate returned a NULL stream"
     STOP 1
  end if

  ! Set through the Dn alias, read through the non-Dn getter.
  call hipsolverCheck(hipsolverDnSetStream(handle, stream))
  got_stream = c_null_ptr
  call hipsolverCheck(hipsolverGetStream(handle, got_stream))
  if (.not. c_associated(got_stream, stream)) then
     write(*,"(a)") "FAILED! hipsolverGetStream did not return the stream set by hipsolverDnSetStream"
     STOP 1
  end if

  ! Clear the stream and read it back through the Dn alias, seeding the output
  ! variable with the old stream so that an untouched out-argument is caught.
  call hipsolverCheck(hipsolverSetStream(handle, c_null_ptr))
  got_stream = stream
  call hipsolverCheck(hipsolverDnGetStream(handle, got_stream))
  if (c_associated(got_stream)) then
     write(*,"(a)") "FAILED! hipsolverDnGetStream did not report the NULL stream after hipsolverSetStream"
     STOP 1
  end if

  call hipCheck(hipStreamDestroy(stream))
  call hipsolverCheck(hipsolverDestroy(handle))

  write(*,"(a,i0,a,i0,a)") "PASSED! deterministic mode round-trips (", HIPSOLVER_DETERMINISTIC_RESULTS, "/", &
       HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS, "), Dn aliases agree, stream round-trips"

end program hipsolver_deterministic_mode
