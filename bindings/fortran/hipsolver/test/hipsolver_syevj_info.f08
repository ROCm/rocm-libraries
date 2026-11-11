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
! hipsolver syevjInfo query test (Fortran 2008 interfaces)
! see: https://rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Exercises the Jacobi-convergence getters hipsolverXsyevjGetResidual and
! hipsolverXsyevjGetSweeps (plus their hipsolverDnXsyevj* aliases), whose
! output arguments are now plain Fortran scalars passed directly, not
! c_loc(...) wrappers.
!
! The same 3x3 symmetric system is solved twice with hipsolverDsyevj, the two
! runs differing only in hipsolverXsyevjSetMaxSweeps. That makes the getters'
! outputs provably data-dependent rather than constant:
!   run 1 caps the solver at 1 sweep  -> few sweeps, a large residual
!   run 2 allows up to 100 sweeps     -> more sweeps, a residual near zero
! Both out-arguments are poisoned (-1) before every read so a binding that
! failed to write them would be caught. We then assert sweeps/residual order,
! that the Dn aliases report the identical values, and that run 2's
! eigenvalues really are 2-sqrt(2), 2, 2+sqrt(2).
!
! f2008 style: device buffers are Fortran pointers filled by hipMalloc with
! source=/mold=; syevj has no array-pointer overload, so they are handed to
! the C-binding routines through c_loc. The getters themselves take plain
! scalars in both dialects.
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
  real(c_double) :: hA(3,3) = reshape((/2, -1, 0, -1, 2, -1, 0, -1, 2/), (/3, 3/))
  real(c_double) :: hW(3)      ! eigenvalues
  real(c_double) :: expected(3)

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: params = c_null_ptr
  real(c_double), pointer :: dA(:,:)
  real(c_double), pointer :: dW(:)
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: hInfo(1)
  type(c_ptr) :: dWork
  integer(c_int) :: lwork

  ! Outputs of the getters under test, poisoned before every read
  real(c_double) :: res1, res2, resDn
  integer(c_int) :: sweeps1, sweeps2, sweepsDn

  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_syevj_info' (Fortran 2008 interfaces) - "

  expected = (/2d0 - sqrt(2d0), 2d0, 2d0 + sqrt(2d0)/)

  call hipsolverCheck(hipsolverCreate(handle))
  call hipsolverCheck(hipsolverCreateSyevjInfo(params))

  ! Allocate device-side memory & copy memory from host to device
  call hipCheck(hipMalloc(dA, source=hA))
  call hipCheck(hipMalloc(dW, mold=hW))
  call hipCheck(hipMalloc(dInfo, 1))

  ! A tight tolerance so that convergence is governed by the sweep cap
  call hipsolverCheck(hipsolverXsyevjSetTolerance(params, 1.0d-14))

  ! Query workspace size once and allocate it
  call hipsolverCheck(hipsolverDsyevj_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, c_loc(dA(1,1)), lda, c_loc(dW(1)), lwork, params))
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  ! ---- Run 1: at most a single Jacobi sweep ---------------------------------
  call hipsolverCheck(hipsolverXsyevjSetMaxSweeps(params, 1))
  call hipCheck(hipMemcpy(dA, hA, hipMemcpyHostToDevice))
  call hipsolverCheck(hipsolverDsyevj(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, c_loc(dA(1,1)), lda, c_loc(dW(1)), dWork, lwork, c_loc(dInfo(1)), params))
  call hipCheck(hipDeviceSynchronize())

  res1 = -1.0d0
  sweeps1 = -1
  call hipsolverCheck(hipsolverXsyevjGetResidual(handle, params, res1))
  call hipsolverCheck(hipsolverXsyevjGetSweeps(handle, params, sweeps1))

  ! ---- Run 2: the same solve, now allowed to converge -----------------------
  call hipsolverCheck(hipsolverXsyevjSetMaxSweeps(params, 100))
  call hipCheck(hipMemcpy(dA, hA, hipMemcpyHostToDevice))
  call hipsolverCheck(hipsolverDsyevj(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, c_loc(dA(1,1)), lda, c_loc(dW(1)), dWork, lwork, c_loc(dInfo(1)), params))
  call hipCheck(hipDeviceSynchronize())

  res2 = -1.0d0
  sweeps2 = -1
  call hipsolverCheck(hipsolverXsyevjGetResidual(handle, params, res2))
  call hipsolverCheck(hipsolverXsyevjGetSweeps(handle, params, sweeps2))

  ! The hipsolverDn* aliases must observe exactly the same state
  resDn = -1.0d0
  sweepsDn = -1
  call hipsolverCheck(hipsolverDnXsyevjGetResidual(handle, params, resDn))
  call hipsolverCheck(hipsolverDnXsyevjGetSweeps(handle, params, sweepsDn))

  call hipCheck(hipMemcpy(hW, dW, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hInfo, dInfo, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
     write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"
     STOP 1
  end if

  ! ---- Assertions -----------------------------------------------------------
  if (sweeps1 /= 1) then
     write(*,*) "FAILED! max_sweeps=1 was not honoured, GetSweeps returned ", sweeps1
     STOP 1
  end if

  if (.not. (sweeps2 > sweeps1 .and. sweeps2 <= 100)) then
     write(*,*) "FAILED! second run sweeps out of range, got ", sweeps2, " vs ", sweeps1
     STOP 1
  end if

  if (res1 <= 1.0d-3) then
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
