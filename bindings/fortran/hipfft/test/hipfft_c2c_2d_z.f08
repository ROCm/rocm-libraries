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

program hipfft_c2c_2d_z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: Nx = 4, Ny = 8
  integer(c_int), parameter :: Ntot = Nx * Ny

  complex(8), allocatable, target, dimension(:) :: hx, hx_input
  complex(8), pointer, dimension(:) :: dx => null()
  type(c_ptr) :: plan = c_null_ptr
  integer(c_int) :: direction
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'hipFFT C2C 2D double (z)' (Fortran 2008 interfaces) - "

  allocate(hx(Ntot))
  allocate(hx_input(Ntot))
  do i = 1, Ntot
     hx(i) = cmplx(dble(i), dble(Ntot - i), kind=8)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, source=hx))

  ! In-place 2D C2C plan over an Nx*Ny complex double grid; the direction is
  ! supplied at exec time rather than at plan creation.
  call hipfftCheck(hipfftPlan2d(plan, Nx, Ny, HIPFFT_Z2Z))

  direction = HIPFFT_FORWARD
  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  call hipCheck(hipDeviceSynchronize())

  direction = HIPFFT_BACKWARD
  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(hx, dx, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! hipFFT is unnormalized, so forward+inverse yields Ntot times the original input.
  do i = 1, Ntot
     error = abs(hx(i) - Ntot * hx_input(i))
     if (.not. (error <= error_max * Ntot)) then
        write(*,*) "FAILED! i=", i, " error=", error, " got=", hx(i), " expected=", Ntot * hx_input(i)
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)

  write(*,*) "PASSED!"

end program hipfft_c2c_2d_z
