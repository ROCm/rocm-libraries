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

program hipfft_setstream_z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N     = 16
  ! Two different harmonics so a swapped-stream bug lands in the wrong bin.
  integer(c_int), parameter :: k0    = 3
  integer(c_int), parameter :: k1    = 5
  integer(c_int), parameter :: one_i = 1
  integer(c_size_t), parameter :: Nbytes = int(N, c_size_t) * 16

  double precision, parameter :: pi  = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-8

  complex(c_double_complex), allocatable, target, dimension(:) :: hx0, hx1
  type(c_ptr) :: dx0     = c_null_ptr
  type(c_ptr) :: dx1     = c_null_ptr
  type(c_ptr) :: plan0   = c_null_ptr
  type(c_ptr) :: plan1   = c_null_ptr
  ! hipStreamCreate fills stream0/stream1 by reference (type(c_ptr), no VALUE).
  type(c_ptr) :: stream0 = c_null_ptr
  type(c_ptr) :: stream1 = c_null_ptr

  double precision         :: max_error, error
  complex(c_double_complex) :: expected
  integer :: j

  write(*,"(a)",advance="no") &
    "-- Running test 'hipFFT SetStream Z2Z (z)' (Fortran 2003 interfaces) - "

  allocate(hx0(N), hx1(N))

  do j = 0, N - 1
    hx0(j+1) = exp(cmplx(0.0d0, 2.0d0*pi*dble(k0*j)/dble(N), kind=c_double_complex))
    hx1(j+1) = exp(cmplx(0.0d0, 2.0d0*pi*dble(k1*j)/dble(N), kind=c_double_complex))
  end do

  call hipCheck(hipMalloc(dx0, Nbytes))
  call hipCheck(hipMalloc(dx1, Nbytes))
  call hipCheck(hipMemcpy(dx0, c_loc(hx0(1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx1, c_loc(hx1(1)), Nbytes, hipMemcpyHostToDevice))

  call hipCheck(hipStreamCreate(stream0))
  call hipCheck(hipStreamCreate(stream1))

  call hipfftCheck(hipfftPlan1d(plan0, N, HIPFFT_Z2Z, one_i))
  call hipfftCheck(hipfftPlan1d(plan1, N, HIPFFT_Z2Z, one_i))

  ! hipfftSetStream takes stream by value (type(c_ptr),value).
  call hipfftCheck(hipfftSetStream(plan0, stream0))
  call hipfftCheck(hipfftSetStream(plan1, stream1))

  call hipfftCheck(hipfftExecZ2Z(plan0, dx0, dx0, HIPFFT_FORWARD))
  call hipfftCheck(hipfftExecZ2Z(plan1, dx1, dx1, HIPFFT_FORWARD))

  ! Synchronise each stream individually — this is what is under test.
  call hipCheck(hipStreamSynchronize(stream0))
  call hipCheck(hipStreamSynchronize(stream1))

  call hipCheck(hipMemcpy(c_loc(hx0(1)), dx0, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hx1(1)), dx1, Nbytes, hipMemcpyDeviceToHost))

  call hipfftCheck(hipfftDestroy(plan0))
  call hipfftCheck(hipfftDestroy(plan1))
  call hipCheck(hipStreamDestroy(stream0))
  call hipCheck(hipStreamDestroy(stream1))
  call hipCheck(hipFree(dx0))
  call hipCheck(hipFree(dx1))

  max_error = 0.0d0
  do j = 0, N - 1
    if (j == k0) then
      expected = cmplx(dble(N), 0.0d0, kind=c_double_complex)
    else
      expected = (0.0d0, 0.0d0)
    end if
    error = abs(hx0(j+1) - expected)
    if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do
  do j = 0, N - 1
    if (j == k1) then
      expected = cmplx(dble(N), 0.0d0, kind=c_double_complex)
    else
      expected = (0.0d0, 0.0d0)
    end if
    error = abs(hx1(j+1) - expected)
    if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do

  if (.not. (max_error <= tol * N)) then
    write(*,*) "FAILED! max error = ", max_error
    STOP 1
  end if

  deallocate(hx0, hx1)

  write(*,*) "PASSED!"

end program hipfft_setstream_z
