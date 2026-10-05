!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

program rocfft_example
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: N=16
  integer(c_size_t), parameter :: Nbytes = N*8*2

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:) :: hx
  complex(8), allocatable, dimension(:) :: x_in, x_ref
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer :: i, j
  double precision :: error
  double precision, parameter :: error_max = 100*epsilon(error)
  double precision, parameter :: two_pi = 2*acos(-1.0d0)

  write(*,"(a)",advance="no") "-- Running test 'rocFFT' (Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_setup())

  allocate(lengths(3))
  lengths(1) = N

  ! A non-constant input, so that a transform that leaves the data untouched
  ! cannot pass. The reference is a direct O(N^2) DFT on the host; the
  ! twiddle exponent is reduced modulo N so the angle stays in [0, 2*pi).
  allocate(hx(N), x_in(N), x_ref(N))
  do i = 1, int(N)
     x_in(i) = cmplx(i, mod(3*i, 5) - 2, kind=8)
     hx(i)%x = real(x_in(i), 8)
     hx(i)%y = aimag(x_in(i))
  end do
  do i = 1, int(N)
     x_ref(i) = (0.d0, 0.d0)
     do j = 1, int(N)
        x_ref(i) = x_ref(i) + x_in(j) * exp(cmplx(0.d0, -two_pi*mod((i-1)*(j-1), int(N))/N, kind=8))
     end do
  end do

  call hipCheck(hipMalloc(dx,Nbytes))
  call hipCheck(hipMemcpy(dx,c_loc(hx(1)),Nbytes,hipMemcpyHostToDevice))

  call rocfftCheck(rocfft_plan_create(plan,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))

  call rocfftCheck(rocfft_execute(plan,dx,c_null_ptr,c_null_ptr))

  call hipCheck(hipDeviceSynchronize())

  call rocfftCheck(rocfft_plan_destroy(plan))

  call hipCheck(hipMemcpy(c_loc(hx(1)),dx,Nbytes,hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  do i = 1, int(N)
     error = abs(cmplx(hx(i)%x, hx(i)%y, kind=8) - x_ref(i))
     if (.not. (error <= error_max * sum(abs(x_in)))) then
        write(*,*) "FAILED! i = ", i, " error = ", error, " hx(i) = ", hx(i)%x, hx(i)%y
        call exit(1)
     end if
  end do

  deallocate(hx, x_in, x_ref)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED!"

end program rocfft_example