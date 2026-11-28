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

!!!!!!!!!!!!!!
! hipsolverXsyevjGetResidual/GetSweeps (and Dn aliases) after two hipsolverDsyevj
! runs with max sweeps 1 and 100: sweeps must grow, the residual must shrink.
!!!!!!!!!!!!!!
!
program hipsolver_syevj_info
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  ! Symmetric input (column-major); eigenvalues are 2-sqrt(2), 2, 2+sqrt(2)
  real(c_double), target :: hA(3,3) = reshape((/2, -1, 0, -1, 2, -1, 0, -1, 2/), (/3, 3/))
  real(c_double), target :: hW(3)      ! eigenvalues
  real(c_double) :: expected(3)

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_W = size(hW)

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: params = c_null_ptr
  type(c_ptr) :: dA, dW, dInfo, dWork
  integer(c_int), target :: hInfo
  integer(c_int) :: lwork

  ! Outputs of the getters under test, poisoned before every read
  real(c_double) :: res1, res2, resDn
  integer(c_int) :: sweeps1, sweeps2, sweepsDn

  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_syevj_info' (Fortran 2003 interfaces) - "

  expected = (/2d0 - sqrt(2d0), 2d0, 2d0 + sqrt(2d0)/)

  call hipsolverCheck(hipsolverCreate(handle))
  call hipsolverCheck(hipsolverCreateSyevjInfo(params))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dW, size_W * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! A tight tolerance so that convergence is governed by the sweep cap
  call hipsolverCheck(hipsolverXsyevjSetTolerance(params, 1.0d-14))

  ! Query workspace size once and allocate it
  call hipsolverCheck(hipsolverDsyevj_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dW, lwork, params))
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  ! ---- Run 1: at most a single Jacobi sweep ---------------------------------
  call hipsolverCheck(hipsolverXsyevjSetMaxSweeps(params, 1))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipsolverCheck(hipsolverDsyevj(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dW, dWork, lwork, dInfo, params))
  call hipCheck(hipDeviceSynchronize())

  res1 = -1.0d0
  sweeps1 = -1
  call hipsolverCheck(hipsolverXsyevjGetResidual(handle, params, res1))
  call hipsolverCheck(hipsolverXsyevjGetSweeps(handle, params, sweeps1))

  ! ---- Run 2: the same solve, now allowed to converge -----------------------
  call hipsolverCheck(hipsolverXsyevjSetMaxSweeps(params, 100))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  hInfo = -1
  call hipCheck(hipMemcpy(dInfo, c_loc(hInfo), 4_c_size_t, hipMemcpyHostToDevice))
  call hipsolverCheck(hipsolverDsyevj(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dW, dWork, lwork, dInfo, params))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hInfo), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo /= 0) then
     write(*,*) "FAILED! converged run info = ", hInfo, " (expected 0)"
     STOP 1
  end if

  res2 = -1.0d0
  sweeps2 = -1
  call hipsolverCheck(hipsolverXsyevjGetResidual(handle, params, res2))
  call hipsolverCheck(hipsolverXsyevjGetSweeps(handle, params, sweeps2))

  ! The hipsolverDn* aliases must observe exactly the same state
  resDn = -1.0d0
  sweepsDn = -1
  call hipsolverCheck(hipsolverDnXsyevjGetResidual(handle, params, resDn))
  call hipsolverCheck(hipsolverDnXsyevjGetSweeps(handle, params, sweepsDn))

  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, size_W * 8, hipMemcpyDeviceToHost))

  ! ---- Assertions -----------------------------------------------------------
  if (sweeps1 /= 1) then
     write(*,*) "FAILED! max_sweeps=1 was not honoured, GetSweeps returned ", sweeps1
     STOP 1
  end if

  if (.not. (sweeps2 > sweeps1 .and. sweeps2 <= 100)) then
     write(*,*) "FAILED! second run sweeps out of range, got ", sweeps2, " vs ", sweeps1
     STOP 1
  end if

  if (.not. (res1 > 1.0d-3)) then
     write(*,*) "FAILED! single-sweep run should be far from converged, residual = ", res1
     STOP 1
  end if

  if (.not. (res2 >= 0.0d0 .and. res2 < 1.0d-6)) then
     write(*,*) "FAILED! converged run residual is not essentially zero, residual = ", res2
     STOP 1
  end if

  if (.not. (res2 < res1)) then
     write(*,*) "FAILED! more sweeps did not reduce the residual, ", res1, " -> ", res2
     STOP 1
  end if

  if (resDn /= res2 .or. sweepsDn /= sweeps2) then
     write(*,*) "FAILED! hipsolverDnXsyevjGet* disagree: ", resDn, sweepsDn, " vs ", res2, sweeps2
     STOP 1
  end if

  do i = 1,N
     if (.not. (abs(hW(i) - expected(i)) <= error_max)) then
        write(*,*) "FAILED! converged eigenvalue ", i, " is ", hW(i), " expected ", expected(i)
        STOP 1
     end if
  end do

  ! Clean up
  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroySyevjInfo(params))
  call hipsolverCheck(hipsolverDestroy(handle))

  write(*,*) "PASSED! syevj sweeps ", sweeps1, "->", sweeps2, " residual ", res1, "->", res2

end program hipsolver_syevj_info
