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
! hipSOLVER Rf parameter getters, each output poisoned and checked against its default.
! On ROCm the hipsolverRfSet* entry points return HIPSOLVER_STATUS_NOT_SUPPORTED,
! so only the defaults are queried, not a set/get round-trip.
!!!!!!!!!!!!!!
!
program hipsolver_rf_params
  use iso_c_binding
  use hip
  use hipsolver

  implicit none

  type(c_ptr) :: rf

  integer(kind(HIPSOLVERRF_MATRIX_FORMAT_CSR)) :: fmt
  integer(kind(HIPSOLVERRF_UNIT_DIAGONAL_STORED_L)) :: diag
  integer(kind(HIPSOLVERRF_NUMERIC_BOOST_NOT_USED)) :: report
  integer(kind(HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF)) :: fast
  integer(kind(HIPSOLVERRF_FACTORIZATION_ALG0)) :: falg
  integer(kind(HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1)) :: salg
  real(c_double) :: zero, boost

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_rf_params' (Fortran 2003 interfaces) - "

  ! hipsolverRfCreate takes its handle as a bare type(c_ptr) output argument.
  rf = c_null_ptr
  call hipsolverCheck(hipsolverRfCreate(rf))

  if (.not. c_associated(rf)) then
     write(*,*) "FAILED! hipsolverRfCreate left the handle null"
     STOP 1
  end if

  ! Matrix format / unit diagonal storage: default is CSR with L unit diagonal.
  fmt = -1
  diag = -1
  call hipsolverCheck(hipsolverRfGetMatrixFormat(rf, fmt, diag))

  if (fmt /= HIPSOLVERRF_MATRIX_FORMAT_CSR) then
     write(*,*) "FAILED! hipsolverRfGetMatrixFormat returned myFormat =", fmt, &
                ", expected HIPSOLVERRF_MATRIX_FORMAT_CSR =", HIPSOLVERRF_MATRIX_FORMAT_CSR
     STOP 1
  end if

  if (diag /= HIPSOLVERRF_UNIT_DIAGONAL_STORED_L) then
     write(*,*) "FAILED! hipsolverRfGetMatrixFormat returned diag =", diag, &
                ", expected HIPSOLVERRF_UNIT_DIAGONAL_STORED_L =", HIPSOLVERRF_UNIT_DIAGONAL_STORED_L
     STOP 1
  end if

  ! Numeric properties: numeric boosting is disabled by default, so both the
  ! zero threshold and the boost value come back as exactly 0.
  zero = -1.0d0
  boost = -1.0d0
  call hipsolverCheck(hipsolverRfGetNumericProperties(rf, zero, boost))

  if (zero /= 0.0d0) then
     write(*,*) "FAILED! hipsolverRfGetNumericProperties returned zero =", zero, ", expected 0.0"
     STOP 1
  end if

  if (boost /= 0.0d0) then
     write(*,*) "FAILED! hipsolverRfGetNumericProperties returned boost =", boost, ", expected 0.0"
     STOP 1
  end if

  ! Consistent with boost == 0 above, the boost report must say "not used".
  report = -1
  call hipsolverCheck(hipsolverRfGetNumericBoostReport(rf, report))

  if (report /= HIPSOLVERRF_NUMERIC_BOOST_NOT_USED) then
     write(*,*) "FAILED! hipsolverRfGetNumericBoostReport returned report =", report, &
                ", expected HIPSOLVERRF_NUMERIC_BOOST_NOT_USED =", HIPSOLVERRF_NUMERIC_BOOST_NOT_USED
     STOP 1
  end if

  ! Fast reset-values mode is off by default.
  fast = -1
  call hipsolverCheck(hipsolverRfGetResetValuesFastMode(rf, fast))

  if (fast /= HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF) then
     write(*,*) "FAILED! hipsolverRfGetResetValuesFastMode returned fastMode =", fast, &
                ", expected HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF =", HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF
     STOP 1
  end if

  ! Algorithms: ALG0 (= 0) for factorization, ALG1 (= 1) for triangular solve.
  ! The two expected values differ, and neither equals the -1 poison.
  falg = -1
  salg = -1
  call hipsolverCheck(hipsolverRfGet_Algs(rf, falg, salg))

  if (falg /= HIPSOLVERRF_FACTORIZATION_ALG0) then
     write(*,*) "FAILED! hipsolverRfGet_Algs returned fact_alg =", falg, &
                ", expected HIPSOLVERRF_FACTORIZATION_ALG0 =", HIPSOLVERRF_FACTORIZATION_ALG0
     STOP 1
  end if

  if (salg /= HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1) then
     write(*,*) "FAILED! hipsolverRfGet_Algs returned solve_alg =", salg, &
                ", expected HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1 =", HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1
     STOP 1
  end if

  call hipsolverCheck(hipsolverRfDestroy(rf))

  write(*,"(a,i0,a,i0,a,i0,a,i0)") "PASSED! hipsolverRf defaults: format=", fmt, " diag=", diag, &
                                   " fact_alg=", falg, " solve_alg=", salg

end program hipsolver_rf_params
