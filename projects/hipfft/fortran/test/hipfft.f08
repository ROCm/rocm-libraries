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

program hipfft_example
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N = 16

  complex(8), allocatable, dimension(:) :: hx, x_ref
  integer(c_int) :: direction = HIPFFT_FORWARD
  complex(8), pointer, dimension(:) :: dx
  type(c_ptr) :: plan = c_null_ptr
  integer(c_size_t)            :: lengths(3)
  integer(c_size_t), parameter :: one = 1
  integer :: i, j
  double precision :: error, x_scale
  double precision, parameter :: error_max = 100*epsilon(error)
  double precision, parameter :: two_pi = 2*acos(-1.0d0)

  write(*,"(a)",advance="no") "-- Running test 'hipFFT' (Fortran 2008 interfaces) - "

  lengths(1) = N

  ! A non-constant input, so that a transform that leaves the data untouched
  ! cannot pass. The reference is a direct O(N^2) DFT on the host.
  allocate(hx(N), x_ref(N))
  do i = 1, N
     hx(i) = cmplx(i, mod(3*i, 5) - 2, kind=8)
  end do
  do i = 1, N
     x_ref(i) = (0.d0, 0.d0)
     do j = 1, N
        x_ref(i) = x_ref(i) + hx(j) * exp(cmplx(0.d0, -two_pi*mod((i-1)*(j-1), N)/N, kind=8))
     end do
  end do
  x_scale = sum(abs(hx))

  call hipCheck(hipMalloc(dx, source=hx))

  call hipfftCheck(hipfftPlan1d(plan, N, HIPFFT_Z2Z, 1))

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  
  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(hx,dx,hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  do i = 1, N
     error = abs(hx(i) - x_ref(i))
     if (.not. (error <= error_max * x_scale)) then
        write(*,*) "FAILED! i = ", i, " error = ", error, " hx(i) = ", hx(i)
        call exit(1)
     end if
  end do

  deallocate(hx, x_ref)

  call hipfftcheck( hipfftDestroy(plan))

  write(*,*) "PASSED!"
end program hipfft_example
