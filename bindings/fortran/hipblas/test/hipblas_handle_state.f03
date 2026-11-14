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

! Exercises the three state getters of a hipBLAS handle. All take their output
! argument by reference -- hipblasGetAtomicsMode and hipblasGetMathMode an enum
! ('integer(kind(HIPBLAS_ATOMICS_NOT_ALLOWED)) :: atomics_mode' and
! 'integer(kind(HIPBLAS_DEFAULT_MATH)) :: mode'), hipblasGetStream a bare
! 'type(c_ptr) :: streamId' -- so in every case the variable itself is passed
! and the library writes into it.
!
! The test passes only if the getters actually report back what the matching
! setter installed: the atomics mode is driven in both directions (ALLOWED and
! NOT_ALLOWED) so a setter that silently ignores its argument cannot pass, and
! the stream is first set to a freshly created stream (the getter must hand
! back the very same object) and then reset to the null/default stream with a
! pre-poisoned output variable (so the getter must overwrite it, not leave it).
!
! The math mode is checked more loosely on purpose. On ROCm HIPBLAS_XF32_XDL_MATH
! is accepted on every architecture but only honoured on hardware with XF32 XDL
! support, so the handle may legitimately stay on HIPBLAS_DEFAULT_MATH; a hard
! round-trip would be a false failure elsewhere. On the CUDA backend it has no
! cuBLAS counterpart and the setter returns HIPBLAS_STATUS_INVALID_VALUE.
! What is asserted is that the getter overwrote the sentinel with a member of
! the enum, and that setting HIPBLAS_DEFAULT_MATH back does round-trip exactly.
program hipblas_handle_state
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: stream = c_null_ptr
  type(c_ptr) :: got = c_null_ptr
  integer(kind(HIPBLAS_ATOMICS_NOT_ALLOWED)) :: amode
  integer(kind(HIPBLAS_DEFAULT_MATH)) :: mmode
  ! Not a hipblasMath_t member on either backend (XF32_XDL is -1 under CUDA).
  integer(kind(HIPBLAS_DEFAULT_MATH)), parameter :: sentinel = -12345
  integer(kind(HIPBLAS_STATUS_SUCCESS)) :: stat

  write(*,"(a)",advance="no") "-- Running test 'hipblas_handle_state' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  ! 1. The default atomics mode must be one of the two legal enum values.
  amode = -1
  call hipblasCheck(hipblasGetAtomicsMode(handle, amode))
  if (amode /= HIPBLAS_ATOMICS_NOT_ALLOWED .and. amode /= HIPBLAS_ATOMICS_ALLOWED) then
     write(*,*) "FAILED! hipblasGetAtomicsMode did not return a legal mode, got ", amode
     STOP 1
  end if

  ! 2. Switch atomics on and check that the getter observes the change.
  call hipblasCheck(hipblasSetAtomicsMode(handle, HIPBLAS_ATOMICS_ALLOWED))
  amode = -1
  call hipblasCheck(hipblasGetAtomicsMode(handle, amode))
  if (amode /= HIPBLAS_ATOMICS_ALLOWED) then
     write(*,*) "FAILED! expected HIPBLAS_ATOMICS_ALLOWED, got ", amode
     STOP 1
  end if

  ! 3. Switch atomics back off, so an ignored setter cannot sneak through.
  call hipblasCheck(hipblasSetAtomicsMode(handle, HIPBLAS_ATOMICS_NOT_ALLOWED))
  amode = -1
  call hipblasCheck(hipblasGetAtomicsMode(handle, amode))
  if (amode /= HIPBLAS_ATOMICS_NOT_ALLOWED) then
     write(*,*) "FAILED! expected HIPBLAS_ATOMICS_NOT_ALLOWED, got ", amode
     STOP 1
  end if

  ! 4. Create a real stream to bind to the handle.
  call hipCheck(hipStreamCreate(stream))
  if (.not. c_associated(stream)) then
     write(*,*) "FAILED! hipStreamCreate returned a null stream"
     STOP 1
  end if

  ! 5. The handle must hand back exactly the stream it was given.
  call hipblasCheck(hipblasSetStream(handle, stream))
  got = c_null_ptr
  call hipblasCheck(hipblasGetStream(handle, got))
  if (.not. c_associated(got, stream)) then
     write(*,*) "FAILED! hipblasGetStream did not return the stream set by hipblasSetStream"
     STOP 1
  end if

  ! 6. Back to the null (default) stream, with 'got' pre-poisoned so that the
  !    getter has to overwrite it rather than leave it untouched.
  call hipblasCheck(hipblasSetStream(handle, c_null_ptr))
  got = stream
  call hipblasCheck(hipblasGetStream(handle, got))
  if (c_associated(got)) then
     write(*,*) "FAILED! hipblasGetStream did not report the default (null) stream"
     STOP 1
  end if

  ! 7. Math mode: the default must be reported exactly, a non-default request
  !    must at least leave a legal enum member behind, and restoring the
  !    default must round-trip.
  mmode = sentinel
  call hipblasCheck(hipblasGetMathMode(handle, mmode))
  if (mmode /= HIPBLAS_DEFAULT_MATH) then
     write(*,*) "FAILED! fresh handle math mode is ", mmode, &
                " instead of HIPBLAS_DEFAULT_MATH"
     STOP 1
  end if

  ! XF32_XDL is an AMD mode: on the CUDA backend cuBLAS rejects it.
  stat = hipblasSetMathMode(handle, HIPBLAS_XF32_XDL_MATH)
  if (stat /= HIPBLAS_STATUS_SUCCESS .and. stat /= HIPBLAS_STATUS_INVALID_VALUE) then
     write(*,*) "FAILED! hipblasSetMathMode(HIPBLAS_XF32_XDL_MATH) returned ", stat
     STOP 1
  end if
  mmode = sentinel
  call hipblasCheck(hipblasGetMathMode(handle, mmode))
  if (mmode /= HIPBLAS_DEFAULT_MATH .and. mmode /= HIPBLAS_XF32_XDL_MATH .and. &
      mmode /= HIPBLAS_PEDANTIC_MATH .and. mmode /= HIPBLAS_TF32_TENSOR_OP_MATH .and. &
      mmode /= HIPBLAS_TENSOR_OP_MATH) then
     write(*,*) "FAILED! hipblasGetMathMode returned ", mmode, &
                " which is not a member of hipblasMath_t"
     STOP 1
  end if

  call hipblasCheck(hipblasSetMathMode(handle, HIPBLAS_DEFAULT_MATH))
  mmode = sentinel
  call hipblasCheck(hipblasGetMathMode(handle, mmode))
  if (mmode /= HIPBLAS_DEFAULT_MATH) then
     write(*,*) "FAILED! hipblasGetMathMode did not restore HIPBLAS_DEFAULT_MATH, got ", mmode
     STOP 1
  end if

  ! 8. The stream was unbound in step 6, so it can be destroyed.
  call hipCheck(hipStreamDestroy(stream))
  call hipblasCheck(hipblasDestroy(handle))

  write(*,*) "PASSED! atomics mode and stream round-trip through the hipBLAS handle"

end program hipblas_handle_state
