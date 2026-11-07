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

  integer(c_int), parameter :: N1d   = 32
  integer(c_int), parameter :: Nout  = N1d / 2 + 1    ! D2Z output: N/2+1 complex elements
  integer(c_int), parameter :: Nx2   = 8,  Ny2 = 4
  integer(c_int), parameter :: Nx3   = 4,  Ny3 = 4,  Nz3 = 2
  integer(c_int), parameter :: one_i = 1

  integer(c_size_t), parameter :: Nbytes_r = int(N1d, c_size_t) * 8
  integer(c_size_t), parameter :: Nbytes_c = int(Nout, c_size_t) * 16

  ! Every size output is poisoned first, so a binding that never writes it fails.
  integer(c_size_t), parameter :: poison = -12345_c_size_t

  integer(c_size_t) :: workEst, workGS, workTmp

  type(c_ptr) :: plan1d = c_null_ptr
  type(c_ptr) :: plan2d = c_null_ptr
  type(c_ptr) :: plan3d = c_null_ptr
  type(c_ptr) :: dx_r   = c_null_ptr
  type(c_ptr) :: dx_c   = c_null_ptr

  real(c_double), allocatable, target, dimension(:)            :: hrx
  complex(c_double_complex), allocatable, target, dimension(:) :: hcx
  complex(c_double_complex) :: expected

  double precision, parameter :: tol = 1.0d-8
  integer :: i

  write(*,"(a)",advance="no") &
    "-- Running test 'hipFFT Estimate/GetSize D2Z (d)' (Fortran 2003 interfaces) - "

  ! --- 1D D2Z ---
  workEst = poison
  workGS  = poison
  call hipfftCheck(hipfftEstimate1d(N1d, HIPFFT_D2Z, one_i, workEst))
  call hipfftCheck(hipfftCreate(plan1d))
  call hipfftCheck(hipfftMakePlan1d(plan1d, N1d, HIPFFT_D2Z, one_i, workGS))
  call check_sizes("1d", workEst, workGS)
  workTmp = poison
  call hipfftCheck(hipfftGetSize1d(plan1d, N1d, HIPFFT_D2Z, one_i, workTmp))
  call check_equal("GetSize1d", workTmp, workGS)
  workTmp = poison
  call hipfftCheck(hipfftGetSize(plan1d, workTmp))
  call check_equal("GetSize", workTmp, workGS)

  ! --- 2D Z2Z ---
  workEst = poison
  workGS  = poison
  call hipfftCheck(hipfftEstimate2d(Nx2, Ny2, HIPFFT_Z2Z, workEst))
  call hipfftCheck(hipfftCreate(plan2d))
  call hipfftCheck(hipfftMakePlan2d(plan2d, Nx2, Ny2, HIPFFT_Z2Z, workGS))
  call check_sizes("2d", workEst, workGS)
  workTmp = poison
  call hipfftCheck(hipfftGetSize2d(plan2d, Nx2, Ny2, HIPFFT_Z2Z, workTmp))
  call check_equal("GetSize2d", workTmp, workGS)

  ! --- 3D Z2Z ---
  workEst = poison
  workGS  = poison
  call hipfftCheck(hipfftEstimate3d(Nx3, Ny3, Nz3, HIPFFT_Z2Z, workEst))
  call hipfftCheck(hipfftCreate(plan3d))
  call hipfftCheck(hipfftMakePlan3d(plan3d, Nx3, Ny3, Nz3, HIPFFT_Z2Z, workGS))
  call check_sizes("3d", workEst, workGS)
  workTmp = poison
  call hipfftCheck(hipfftGetSize3d(plan3d, Nx3, Ny3, Nz3, HIPFFT_Z2Z, workTmp))
  call check_equal("GetSize3d", workTmp, workGS)

  ! --- 1D D2Z transform of all ones: N1d in the DC bin, 0 elsewhere ---
  allocate(hrx(N1d), hcx(Nout))
  hrx(:) = 1.0d0
  hcx(:) = cmplx(1.0d30, 1.0d30, kind=c_double_complex)

  call hipCheck(hipMalloc(dx_r, Nbytes_r))
  call hipCheck(hipMalloc(dx_c, Nbytes_c))
  call hipCheck(hipMemcpy(dx_r, c_loc(hrx(1)), Nbytes_r, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx_c, c_loc(hcx(1)), Nbytes_c, hipMemcpyHostToDevice))

  call hipfftCheck(hipfftExecD2Z(plan1d, dx_r, dx_c))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan1d))
  call hipfftCheck(hipfftDestroy(plan2d))
  call hipfftCheck(hipfftDestroy(plan3d))

  call hipCheck(hipMemcpy(c_loc(hcx(1)), dx_c, Nbytes_c, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx_r))
  call hipCheck(hipFree(dx_c))

  do i = 1, Nout
    expected = (0.0d0, 0.0d0)
    if (i == 1) expected = cmplx(dble(N1d), 0.0d0, kind=c_double_complex)
    if (.not. (abs(hcx(i) - expected) <= tol * N1d)) then
      write(*,*) "FAILED! bin ", i - 1, " = ", hcx(i)
      STOP 1
    end if
  end do

  deallocate(hrx, hcx)

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
