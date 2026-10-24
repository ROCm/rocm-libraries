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

program rocfft_r2c_c2r_1d_d
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: N = 16
  integer(c_size_t), parameter :: Ncomplex = N/2 + 1

  real(8), allocatable, target, dimension(:) :: hr, hr_input
  complex(8), pointer, dimension(:) :: dc => null()
  real(8), pointer, dimension(:) :: dr => null()
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  allocate(lengths(1))
  lengths(1) = N

  allocate(hr(N))
  allocate(hr_input(N))
  do i = 1, N
     hr(i) = dble(i) + dble(mod(i,3)) - dble(mod(i,7))
  end do
  hr_input(:) = hr(:)

  ! Device buffers: real input of length N, complex output of length N/2+1.
  call hipCheck(hipMalloc(dr, source=hr))
  call hipCheck(hipMalloc(dc, Ncomplex))

  write(*,"(a)",advance="no") "-- Running test 'rocFFT R2C/C2R 1D double (d)' (Fortran 2008 interfaces) - "

  call rocfftCheck(rocfft_setup())

  ! Forward real-to-complex (out-of-place): dr -> dc.
  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_notinplace,&
                                      rocfft_transform_type_real_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_execute(plan_fwd, c_loc(dr), c_loc(dc), c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_fwd))

  ! Inverse complex-to-real (out-of-place): dc -> dr. rocFFT unnormalized -> N*input.
  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_notinplace,&
                                      rocfft_transform_type_real_inverse,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_execute(plan_bwd, c_loc(dc), c_loc(dr), c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_bwd))

  call hipCheck(hipMemcpy(hr, dr, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dr))
  call hipCheck(hipFree(dc))

  ! After forward+inverse the real data should equal N times the original input.
  do i = 1, N
     error = abs(hr(i) - N * hr_input(i))
     if (.not. (error <= error_max * N)) then
        write(*,*) "FAILED! i=", i, " error=", error, " got=", hr(i), " expected=", N * hr_input(i)
        call rocfftCheck(rocfft_cleanup())
        STOP 1
     end if
  end do

  deallocate(hr)
  deallocate(hr_input)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED!"

end program rocfft_r2c_c2r_1d_d
