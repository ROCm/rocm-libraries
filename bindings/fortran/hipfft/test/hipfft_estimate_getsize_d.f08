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

program hipfft_estimate_getsize_d
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  ! hipfftEstimate1d/2d/3d vs hipfftGetSize1d/2d/3d vs hipfftGetSize.
  ! Estimate is a pre-plan heuristic upper bound; GetSize* is accurate.
  ! (Fortran 2008 interfaces)

  ! Sizes for the three dimension tests.
  integer(c_int), parameter :: N1 = 32, Nx2 = 8, Ny2 = 4
  integer(c_int), parameter :: Nx3 = 4, Ny3 = 4, Nz3 = 2

  ! D2Z real input / complex output for the functional transform check.
  integer(c_int), parameter :: Ncomplex = N1 / 2 + 1

  double precision, parameter :: tol = 1.0d-8

  real(8), allocatable, target, dimension(:) :: hr
  complex(8), allocatable, target, dimension(:) :: hc
  real(8), pointer, dimension(:) :: dr => null()
  complex(8), pointer, dimension(:) :: dc => null()
  type(c_ptr) :: plan = c_null_ptr
  ! Every size output is poisoned first, so a binding that never writes it fails.
  integer(c_size_t), parameter :: poison = -12345_c_size_t
  integer(c_size_t) :: estSz, mkSz, gsSz
  integer :: i
  double precision :: dc_sum, dc_expected

  write(*,"(a)",advance="no") &
    "-- Running test 'hipFFT Estimate/GetSize double (d)' (Fortran 2008 interfaces) - "

  ! -----------------------------------------------------------------------
  ! 1-D: HIPFFT_D2Z, N=32, batch=1
  ! -----------------------------------------------------------------------
  estSz = poison
  mkSz  = poison
  call hipfftCheck(hipfftEstimate1d(N1, HIPFFT_D2Z, 1, estSz))

  call hipfftCheck(hipfftCreate(plan))
  ! hipfftGetSize needs a made plan; hipfftGetSize1d does not configure the handle.
  call hipfftCheck(hipfftMakePlan1d(plan, N1, HIPFFT_D2Z, 1, mkSz))
  call check_sizes("1d", estSz, mkSz)
  gsSz = poison
  call hipfftCheck(hipfftGetSize1d(plan, N1, HIPFFT_D2Z, 1, gsSz))
  call check_equal("GetSize1d", gsSz, mkSz)
  gsSz = poison
  call hipfftCheck(hipfftGetSize(plan, gsSz))
  call check_equal("GetSize", gsSz, mkSz)
  call hipfftCheck(hipfftDestroy(plan))

  ! -----------------------------------------------------------------------
  ! 2-D: HIPFFT_Z2Z, Nx=8, Ny=4
  ! -----------------------------------------------------------------------
  estSz = poison
  mkSz  = poison
  call hipfftCheck(hipfftEstimate2d(Nx2, Ny2, HIPFFT_Z2Z, estSz))

  call hipfftCheck(hipfftCreate(plan))
  call hipfftCheck(hipfftMakePlan2d(plan, Nx2, Ny2, HIPFFT_Z2Z, mkSz))
  call check_sizes("2d", estSz, mkSz)
  gsSz = poison
  call hipfftCheck(hipfftGetSize2d(plan, Nx2, Ny2, HIPFFT_Z2Z, gsSz))
  call check_equal("GetSize2d", gsSz, mkSz)
  gsSz = poison
  call hipfftCheck(hipfftGetSize(plan, gsSz))
  call check_equal("GetSize", gsSz, mkSz)
  call hipfftCheck(hipfftDestroy(plan))

  ! -----------------------------------------------------------------------
  ! 3-D: HIPFFT_Z2Z, Nx=4, Ny=4, Nz=2
  ! -----------------------------------------------------------------------
  estSz = poison
  mkSz  = poison
  call hipfftCheck(hipfftEstimate3d(Nx3, Ny3, Nz3, HIPFFT_Z2Z, estSz))

  call hipfftCheck(hipfftCreate(plan))
  call hipfftCheck(hipfftMakePlan3d(plan, Nx3, Ny3, Nz3, HIPFFT_Z2Z, mkSz))
  call check_sizes("3d", estSz, mkSz)
  gsSz = poison
  call hipfftCheck(hipfftGetSize3d(plan, Nx3, Ny3, Nz3, HIPFFT_Z2Z, gsSz))
  call check_equal("GetSize3d", gsSz, mkSz)
  gsSz = poison
  call hipfftCheck(hipfftGetSize(plan, gsSz))
  call check_equal("GetSize", gsSz, mkSz)
  call hipfftCheck(hipfftDestroy(plan))

  ! -----------------------------------------------------------------------
  ! Functional check: 1-D D2Z, N=32. DC bin must equal sum(input).
  ! -----------------------------------------------------------------------
  allocate(hr(N1))
  allocate(hc(Ncomplex))

  do i = 1, N1
     hr(i) = dble(i) + dble(mod(i, 5)) * 0.5d0
  end do
  ! Poisoned, and copied to the device, so a transform that never writes fails.
  hc(:) = cmplx(1.0d30, 1.0d30, kind=8)

  dc_expected = sum(hr)

  call hipCheck(hipMalloc(dr, source=hr))
  call hipCheck(hipMalloc(dc, source=hc))

  call hipfftCheck(hipfftPlan1d(plan, N1, HIPFFT_D2Z, 1))
  call hipfftCheck(hipfftExecD2Z(plan, dr, dc))
  call hipCheck(hipDeviceSynchronize())
  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(hc, dc, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dr))
  call hipCheck(hipFree(dc))

  ! DC bin (Fortran index 1) = sum of all input values.
  dc_sum = dble(real(hc(1), kind=8))
  if (.not. (abs(dc_sum - dc_expected) <= tol * N1 * abs(dc_expected))) then
     write(*,*) "FAILED! DC bin =", dc_sum, " expected =", dc_expected
     STOP 1
  end if
  if (.not. (abs(aimag(hc(1))) <= tol * N1 * abs(dc_expected))) then
     write(*,*) "FAILED! DC imaginary part non-zero:", aimag(hc(1))
     STOP 1
  end if

  deallocate(hr)
  deallocate(hc)

  write(*,*) "PASSED!"

contains

  ! hipfftEstimate* is an upper bound on the planned work size.
  subroutine check_sizes(what, est, gs)
    character(*), intent(in) :: what
    integer(c_size_t), intent(in) :: est, gs
    if (gs == poison .or. gs < 0) then
      write(*,*) "FAILED! MakePlan", what, " work size: ", gs
      STOP 1
    end if
    if (est == poison .or. est < gs) then
      write(*,*) "FAILED! Estimate", what, " = ", est, " < MakePlan", what, " = ", gs
      STOP 1
    end if
  end subroutine check_sizes

  subroutine check_equal(what, got, ref)
    character(*), intent(in) :: what
    integer(c_size_t), intent(in) :: got, ref
    if (got /= ref) then
      write(*,*) "FAILED! hipfft", what, " = ", got, ", MakePlan reported ", ref
      STOP 1
    end if
  end subroutine check_equal

end program hipfft_estimate_getsize_d
