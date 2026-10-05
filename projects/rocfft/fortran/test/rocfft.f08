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

program rocfft_example
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: N=16

  complex(8), allocatable, target, dimension(:) :: hx
  complex(8), pointer, dimension(:) :: dx => null()
  type(c_ptr) :: plan = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  complex(8), allocatable, dimension(:) :: x_in, x_ref
  integer :: i, j
  double precision :: error
  double precision, parameter :: error_max = 100*epsilon(error)
  double precision, parameter :: two_pi = 2*acos(-1.0d0)


  allocate(lengths(3))
  lengths(1) = N
  ! A non-constant input, so that a transform that leaves the data untouched
  ! cannot pass. The reference is a direct O(N^2) DFT on the host.
  allocate(hx(N), x_in(N), x_ref(N))
  do i = 1, int(N)
     x_in(i) = cmplx(i, mod(3*i, 5) - 2, kind=8)
  end do
  hx(:) = x_in(:)
  do i = 1, int(N)
     x_ref(i) = (0.d0, 0.d0)
     do j = 1, int(N)
        x_ref(i) = x_ref(i) + x_in(j) * exp(cmplx(0.d0, -two_pi*mod((i-1)*(j-1), int(N))/N, kind=8))
     end do
  end do
  
  call hipCheck(hipMalloc(dx,source=hx))
  
  write(*,"(a)",advance="no") "-- Running test 'rocFFT' (Fortran 2008 interfaces) - "

  call rocfftCheck(rocfft_setup())
  call rocfftCheck(rocfft_plan_create(plan,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths),&
                                      one,&
                                      c_null_ptr))

  call rocfftCheck(rocfft_execute(plan,c_loc(dx),c_null_ptr,c_null_ptr))

  call hipCheck(hipDeviceSynchronize())

  call rocfftCheck(rocfft_plan_destroy(plan))

  call hipCheck(hipMemcpy(hx,dx,hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  do i = 1, int(N)
     error = abs(hx(i) - x_ref(i))
     if (.not. (error <= error_max * sum(abs(x_in)))) then
        write(*,*) "FAILED! Error = ", error, "hx(i)=",hx(i)
        STOP 1
     end if
  end do

  deallocate(hx, x_in, x_ref)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED!"

end program rocfft_example
