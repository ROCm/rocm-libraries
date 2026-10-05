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

  ! Two 1-D Z2Z plans bound to different HIP streams, different input harmonics.
  ! A swapped stream would place energy in the wrong bin and fail verification.
  ! (Fortran 2008 interfaces)
  integer(c_int), parameter :: N = 16

  ! Distinct harmonics so a stream swap is observable in both buffers.
  integer, parameter :: k1 = 3, k2 = 7

  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-8

  complex(8), allocatable, target, dimension(:) :: hx1, hx2
  complex(8), pointer, dimension(:) :: dx1 => null(), dx2 => null()
  type(c_ptr) :: plan1 = c_null_ptr, plan2 = c_null_ptr
  type(c_ptr) :: stream1 = c_null_ptr, stream2 = c_null_ptr
  integer(c_int) :: direction
  integer :: j
  double precision :: error, max_error

  write(*,"(a)",advance="no") &
    "-- Running test 'hipFFT SetStream Z2Z 1D (z)' (Fortran 2008 interfaces) - "

  allocate(hx1(N))
  allocate(hx2(N))

  ! Buffer 1: pure harmonic at bin k1 (0-indexed).
  do j = 0, N - 1
     hx1(j + 1) = exp(cmplx(0.0d0, 2.0d0 * pi * dble(k1 * j) / dble(N), kind=8))
  end do

  ! Buffer 2: pure harmonic at bin k2 (0-indexed).
  do j = 0, N - 1
     hx2(j + 1) = exp(cmplx(0.0d0, 2.0d0 * pi * dble(k2 * j) / dble(N), kind=8))
  end do

  call hipCheck(hipMalloc(dx1, source=hx1))
  call hipCheck(hipMalloc(dx2, source=hx2))

  call hipCheck(hipStreamCreate(stream1))
  call hipCheck(hipStreamCreate(stream2))

  call hipfftCheck(hipfftPlan1d(plan1, N, HIPFFT_Z2Z, 1))
  call hipfftCheck(hipfftPlan1d(plan2, N, HIPFFT_Z2Z, 1))

  call hipfftCheck(hipfftSetStream(plan1, stream1))
  call hipfftCheck(hipfftSetStream(plan2, stream2))

  direction = HIPFFT_FORWARD
  call hipfftCheck(hipfftExecZ2Z(plan1, dx1, dx1, direction))
  call hipfftCheck(hipfftExecZ2Z(plan2, dx2, dx2, direction))

  ! Synchronize each stream independently (not hipDeviceSynchronize).
  call hipCheck(hipStreamSynchronize(stream1))
  call hipCheck(hipStreamSynchronize(stream2))

  call hipfftCheck(hipfftDestroy(plan1))
  call hipfftCheck(hipfftDestroy(plan2))

  call hipCheck(hipMemcpy(hx1, dx1, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hx2, dx2, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx1))
  call hipCheck(hipFree(dx2))

  call hipCheck(hipStreamDestroy(stream1))
  call hipCheck(hipStreamDestroy(stream2))

  ! Verify buffer 1: energy only at Fortran index k1+1 (0-indexed bin k1).
  max_error = 0.0d0
  do j = 1, N
     if (j == k1 + 1) then
        error = abs(hx1(j) - cmplx(dble(N), 0.0d0, kind=8))
     else
        error = abs(hx1(j))
     end if
     if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do

  if (.not. (max_error <= tol * N)) then
     write(*,*) "FAILED! buffer1 max error = ", max_error
     STOP 1
  end if

  ! Verify buffer 2: energy only at Fortran index k2+1 (0-indexed bin k2).
  max_error = 0.0d0
  do j = 1, N
     if (j == k2 + 1) then
        error = abs(hx2(j) - cmplx(dble(N), 0.0d0, kind=8))
     else
        error = abs(hx2(j))
     end if
     if (max_error == max_error .and. .not. (error <= max_error)) max_error = error  ! keeps a NaN
  end do

  if (.not. (max_error <= tol * N)) then
     write(*,*) "FAILED! buffer2 max error = ", max_error
     STOP 1
  end if

  deallocate(hx1)
  deallocate(hx2)

  write(*,*) "PASSED!"

end program hipfft_setstream_z
