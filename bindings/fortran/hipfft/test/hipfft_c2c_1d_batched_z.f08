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

  ! one_i avoids ambiguity: the raw hipfftPlanMany_ interface requires integer(c_int)
  ! VALUE arguments, and a bare literal '1' would have the default integer kind.
  integer(c_int), parameter :: one_i = 1

  complex(8), allocatable, target, dimension(:) :: hx, hx_input
  complex(8), pointer, dimension(:) :: dx => null()
  type(c_ptr) :: plan = c_null_ptr
  integer(c_int) :: direction
  ! nlen is the per-dimension transform length array passed by address to
  ! hipfftPlanMany (it cannot be called "n": Fortran is case-insensitive and N
  ! is already the transform length parameter).
  integer(c_int), target :: nlen(1)
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'hipFFT C2C 1D batched double (z)' (Fortran 2008 interfaces) - "

  nlen(1) = N

  allocate(hx(Ntot))
  allocate(hx_input(Ntot))
  do i = 1, Ntot
     hx(i) = cmplx(dble(i), dble(Ntot - i), kind=8)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, source=hx))

  ! NULL inembed/onembed selects the simple contiguous batched layout: each
  ! transform spans N consecutive complex doubles (istride=1, idist=N,
  ! ostride=1, odist=N). Passing c_loc(nlen) and c_null_ptr routes to the raw
  ! hipfftPlanMany_ C interface rather than the array
  ! overloads (which expect integer(c_int) arrays for n, inembed, onembed).
  ! Unlike rocFFT, a single plan handle serves both directions; the direction
  ! is supplied at exec time via hipfftExecZ2Z.
  call hipfftCheck(hipfftPlanMany(plan, one_i, c_loc(nlen), &
                                  c_null_ptr, one_i, N, &
                                  c_null_ptr, one_i, N, &
                                  HIPFFT_Z2Z, Nbatch))

  direction = HIPFFT_FORWARD
  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  call hipCheck(hipDeviceSynchronize())

  direction = HIPFFT_BACKWARD
  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! hipFFT is unnormalized: forward+inverse yields N (per-transform length,
  ! not Ntot) times the original input for each element.
  do i = 1, Ntot
     error = abs(hx(i) - N * hx_input(i))
     if (.not. (error <= error_max * N)) then
        write(*,*) "FAILED! i=", i, " error=", error, " got=", hx(i), " expected=", N * hx_input(i)
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)

  write(*,*) "PASSED!"

end program hipfft_c2c_1d_batched_z
