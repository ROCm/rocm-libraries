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

program hipfft_c2c_1d_c
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N = 16
  integer(c_size_t), parameter :: Nbytes = int(N,c_size_t)*4*2

  type float2
     real :: x
     real :: y
  end type float2

  type(float2), allocatable, target, dimension(:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer :: i
  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'hipFFT C2C 1D single (c)' (Fortran 2003 interfaces) - "

  allocate(hx(N))
  allocate(hx_input(N))
  do i = 1, N
     hx(i)%x = real(i)
     hx(i)%y = real(N - i)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! A single hipFFT plan covers both directions: the transform direction is
  ! passed to hipfftExecC2C rather than baked into the plan.
  call hipfftCheck(hipfftPlan1d(plan, N, HIPFFT_C2C, 1))

  call hipfftCheck(hipfftExecC2C(plan, dx, dx, HIPFFT_FORWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftExecC2C(plan, dx, dx, HIPFFT_BACKWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! hipFFT is unnormalized, so forward+inverse yields N times the original input.
  do i = 1, N
     error = abs(hx(i)%x - N * hx_input(i)%x) + abs(hx(i)%y - N * hx_input(i)%y)
     if (.not. (error <= error_max * N)) then
        write(*,*) "FAILED! i=", i, " error=", error
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)

  write(*,*) "PASSED!"

end program hipfft_c2c_1d_c
