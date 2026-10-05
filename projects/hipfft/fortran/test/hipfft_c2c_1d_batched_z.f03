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

program hipfft_c2c_1d_batched_z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N = 16
  integer(c_int), parameter :: Nbatch = 4
  integer(c_int), parameter :: Ntot = N * Nbatch
  integer(c_size_t), parameter :: Nbytes = int(Ntot,c_size_t)*8*2

  ! The raw hipfftPlanMany_ interface takes integer(c_int) by value.
  integer(c_int), parameter :: one_i = 1

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer(c_int), target :: nlen(1)
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'hipFFT C2C 1D batched double (z)' (Fortran 2003 interfaces) - "

  nlen(1) = N

  allocate(hx(Ntot))
  allocate(hx_input(Ntot))
  do i = 1, Ntot
     hx(i)%x = dble(i)
     hx(i)%y = dble(Ntot - i)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! c_loc/c_null_ptr select the raw hipfftPlanMany_ interface; NULL
  ! inembed/onembed means contiguous batches (idist = odist = N).
  call hipfftCheck(hipfftPlanMany(plan, one_i, c_loc(nlen), &
                                  c_null_ptr, one_i, N, &
                                  c_null_ptr, one_i, N, &
                                  HIPFFT_Z2Z, Nbatch))

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, HIPFFT_FORWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, HIPFFT_BACKWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! hipFFT is unnormalized: forward+inverse yields N (per-transform length,
  ! not Ntot) times the original input for each element.
  do i = 1, Ntot
     error = abs(hx(i)%x - N * hx_input(i)%x) + abs(hx(i)%y - N * hx_input(i)%y)
     if (.not. (error <= error_max * N)) then
        write(*,*) "FAILED! i=", i, " error=", error
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)

  write(*,*) "PASSED!"

end program hipfft_c2c_1d_batched_z
