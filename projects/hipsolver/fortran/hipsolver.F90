!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

module hipsolver
  use, intrinsic :: iso_c_binding
  implicit none

  ! hipsolverStatus_t
  enum, bind(c)
    enumerator :: HIPSOLVER_STATUS_SUCCESS = 0
    enumerator :: HIPSOLVER_STATUS_NOT_INITIALIZED = 1
    enumerator :: HIPSOLVER_STATUS_ALLOC_FAILED = 2
    enumerator :: HIPSOLVER_STATUS_INVALID_VALUE = 3
    enumerator :: HIPSOLVER_STATUS_MAPPING_ERROR = 4
    enumerator :: HIPSOLVER_STATUS_EXECUTION_FAILED = 5
    enumerator :: HIPSOLVER_STATUS_INTERNAL_ERROR = 6
    enumerator :: HIPSOLVER_STATUS_NOT_SUPPORTED = 7
    enumerator :: HIPSOLVER_STATUS_ARCH_MISMATCH = 8
    enumerator :: HIPSOLVER_STATUS_HANDLE_IS_NULLPTR = 9
    enumerator :: HIPSOLVER_STATUS_INVALID_ENUM = 10
    enumerator :: HIPSOLVER_STATUS_UNKNOWN = 11
    enumerator :: HIPSOLVER_STATUS_ZERO_PIVOT = 12
    enumerator :: HIPSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 13
  end enum

  ! hipblasOperation_t
  enum, bind(c)
    enumerator :: HIPSOLVER_OP_N = 111
    enumerator :: HIPSOLVER_OP_T = 112
    enumerator :: HIPSOLVER_OP_C = 113
  end enum

  ! hipblasFillMode_t
  enum, bind(c)
    enumerator :: HIPSOLVER_FILL_MODE_UPPER = 121
    enumerator :: HIPSOLVER_FILL_MODE_LOWER = 122
    enumerator :: HIPSOLVER_FILL_MODE_FULL = 123
  end enum

  ! hipblasDiagType_t
  enum, bind(c)
    enumerator :: HIPSOLVER_DIAG_NON_UNIT = 131
    enumerator :: HIPSOLVER_DIAG_UNIT = 132
  end enum

  ! hipblasSideMode_t
  enum, bind(c)
    enumerator :: HIPSOLVER_SIDE_LEFT = 141
    enumerator :: HIPSOLVER_SIDE_RIGHT = 142
    enumerator :: HIPSOLVER_SIDE_BOTH = 143
  end enum

  ! hipsolverEigMode_t
  enum, bind(c)
    enumerator :: HIPSOLVER_EIG_MODE_NOVECTOR = 201
    enumerator :: HIPSOLVER_EIG_MODE_VECTOR = 202
  end enum

  ! hipsolverEigType_t
  enum, bind(c)
    enumerator :: HIPSOLVER_EIG_TYPE_1 = 211
    enumerator :: HIPSOLVER_EIG_TYPE_2 = 212
    enumerator :: HIPSOLVER_EIG_TYPE_3 = 213
  end enum

  ! hipsolverEigRange_t
  enum, bind(c)
    enumerator :: HIPSOLVER_EIG_RANGE_ALL = 221
    enumerator :: HIPSOLVER_EIG_RANGE_V = 222
    enumerator :: HIPSOLVER_EIG_RANGE_I = 223
  end enum

  ! hipsolverDeterministicMode_t
  enum, bind(c)
    enumerator :: HIPSOLVER_DETERMINISTIC_RESULTS = 241
    enumerator :: HIPSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS = 242
  end enum

  ! hipsolverDirectMode_t
  enum, bind(c)
    enumerator :: HIPSOLVER_DIRECT_FORWARD = 251
    enumerator :: HIPSOLVER_DIRECT_BACKWARD = 252
  end enum

  ! hipsolverStorevMode_t
  enum, bind(c)
    enumerator :: HIPSOLVER_STOREV_COLUMNWISE = 261
    enumerator :: HIPSOLVER_STOREV_ROWWISE = 262
  end enum

  ! hipsolverAlgMode_t
  enum, bind(c)
    enumerator :: HIPSOLVER_ALG_0 = 231
    enumerator :: HIPSOLVER_ALG_1 = 232
  end enum

  ! hipsolverDnFunction_t
  enum, bind(c)
    enumerator :: HIPSOLVERDN_GETRF = 0
  end enum

  ! hipsolverRfFactorization_t
  enum, bind(c)
    enumerator :: HIPSOLVERRF_FACTORIZATION_ALG0 = 0
    enumerator :: HIPSOLVERRF_FACTORIZATION_ALG1 = 1
    enumerator :: HIPSOLVERRF_FACTORIZATION_ALG2 = 2
  end enum

  ! hipsolverRfMatrixFormat_t
  enum, bind(c)
    enumerator :: HIPSOLVERRF_MATRIX_FORMAT_CSR = 0
    enumerator :: HIPSOLVERRF_MATRIX_FORMAT_CSC = 1
  end enum

  ! hipsolverRfNumericBoostReport_t
  enum, bind(c)
    enumerator :: HIPSOLVERRF_NUMERIC_BOOST_NOT_USED = 0
    enumerator :: HIPSOLVERRF_NUMERIC_BOOST_USED = 1
  end enum

  ! hipsolverRfResetValuesFastMode_t
  enum, bind(c)
    enumerator :: HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0
    enumerator :: HIPSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1
  end enum

  ! hipsolverRfTriangularSolve_t
  enum, bind(c)
    enumerator :: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1
    enumerator :: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2
    enumerator :: HIPSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3
  end enum

  ! hipsolverRfUnitDiagonal_t
  enum, bind(c)
    enumerator :: HIPSOLVERRF_UNIT_DIAGONAL_STORED_L = 0
    enumerator :: HIPSOLVERRF_UNIT_DIAGONAL_STORED_U = 1
    enumerator :: HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2
    enumerator :: HIPSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3
  end enum

  integer(c_int), parameter :: hipsolverVersionMajor = 3
  integer(c_int), parameter :: hipsolverVersionMinor = 8
  integer(c_int), parameter :: hipsolverVersionPatch = 0


  interface

    !---------------------------------------------
    ! hipsolverCreate
    !---------------------------------------------
    function hipsolverCreate(handle) &
       result(Create) &
       bind(C, name="hipsolverCreate")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Create
    end function hipsolverCreate

    !---------------------------------------------
    ! hipsolverDestroy
    !---------------------------------------------
    function hipsolverDestroy(handle) &
       result(Destroy) &
       bind(C, name="hipsolverDestroy")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Destroy
    end function hipsolverDestroy

    !---------------------------------------------
    ! hipsolverSetStream
    !---------------------------------------------
    function hipsolverSetStream(handle, streamId) &
       result(SetStream) &
       bind(C, name="hipsolverSetStream")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: streamId
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SetStream
    end function hipsolverSetStream

    !---------------------------------------------
    ! hipsolverGetStream
    !---------------------------------------------
    function hipsolverGetStream(handle, streamId) &
       result(GetStream) &
       bind(C, name="hipsolverGetStream")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr) :: streamId
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: GetStream
    end function hipsolverGetStream

    !---------------------------------------------
    ! hipsolverSetDeterministicMode
    !---------------------------------------------
    function hipsolverSetDeterministicMode(handle, mode) &
       result(SetDeterministicMode) &
       bind(C, name="hipsolverSetDeterministicMode")
       import :: c_ptr, HIPSOLVER_DETERMINISTIC_RESULTS, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_DETERMINISTIC_RESULTS)), value :: mode
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SetDeterministicMode
    end function hipsolverSetDeterministicMode

    !---------------------------------------------
    ! hipsolverGetDeterministicMode
    !---------------------------------------------
    function hipsolverGetDeterministicMode(handle, mode) &
       result(GetDeterministicMode) &
       bind(C, name="hipsolverGetDeterministicMode")
       import :: c_ptr, HIPSOLVER_DETERMINISTIC_RESULTS, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_DETERMINISTIC_RESULTS)) :: mode
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: GetDeterministicMode
    end function hipsolverGetDeterministicMode

    !---------------------------------------------
    ! hipsolverCreateGesvdjInfo
    !---------------------------------------------
    function hipsolverCreateGesvdjInfo(myInfo) &
       result(CreateGesvdjInfo) &
       bind(C, name="hipsolverCreateGesvdjInfo")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CreateGesvdjInfo
    end function hipsolverCreateGesvdjInfo

    !---------------------------------------------
    ! hipsolverDestroyGesvdjInfo
    !---------------------------------------------
    function hipsolverDestroyGesvdjInfo(myInfo) &
       result(DestroyGesvdjInfo) &
       bind(C, name="hipsolverDestroyGesvdjInfo")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DestroyGesvdjInfo
    end function hipsolverDestroyGesvdjInfo

    !---------------------------------------------
    ! hipsolverXgesvdjSetMaxSweeps
    !---------------------------------------------
    function hipsolverXgesvdjSetMaxSweeps(myInfo, max_sweeps) &
       result(XgesvdjSetMaxSweeps) &
       bind(C, name="hipsolverXgesvdjSetMaxSweeps")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(c_int), value :: max_sweeps
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XgesvdjSetMaxSweeps
    end function hipsolverXgesvdjSetMaxSweeps

    !---------------------------------------------
    ! hipsolverXgesvdjSetSortEig
    !---------------------------------------------
    function hipsolverXgesvdjSetSortEig(myInfo, sort_eig) &
       result(XgesvdjSetSortEig) &
       bind(C, name="hipsolverXgesvdjSetSortEig")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(c_int), value :: sort_eig
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XgesvdjSetSortEig
    end function hipsolverXgesvdjSetSortEig

    !---------------------------------------------
    ! hipsolverXgesvdjSetTolerance
    !---------------------------------------------
    function hipsolverXgesvdjSetTolerance(myInfo, tolerance) &
       result(XgesvdjSetTolerance) &
       bind(C, name="hipsolverXgesvdjSetTolerance")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       real(c_double), value :: tolerance
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XgesvdjSetTolerance
    end function hipsolverXgesvdjSetTolerance

    !---------------------------------------------
    ! hipsolverXgesvdjGetResidual
    !---------------------------------------------
    function hipsolverXgesvdjGetResidual(handle, myInfo, residual) &
       result(XgesvdjGetResidual) &
       bind(C, name="hipsolverXgesvdjGetResidual")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: myInfo
       real(c_double) :: residual
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XgesvdjGetResidual
    end function hipsolverXgesvdjGetResidual

    !---------------------------------------------
    ! hipsolverXgesvdjGetSweeps
    !---------------------------------------------
    function hipsolverXgesvdjGetSweeps(handle, myInfo, executed_sweeps) &
       result(XgesvdjGetSweeps) &
       bind(C, name="hipsolverXgesvdjGetSweeps")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: myInfo
       integer(c_int) :: executed_sweeps
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XgesvdjGetSweeps
    end function hipsolverXgesvdjGetSweeps

    !---------------------------------------------
    ! hipsolverCreateSyevjInfo
    !---------------------------------------------
    function hipsolverCreateSyevjInfo(myInfo) &
       result(CreateSyevjInfo) &
       bind(C, name="hipsolverCreateSyevjInfo")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CreateSyevjInfo
    end function hipsolverCreateSyevjInfo

    !---------------------------------------------
    ! hipsolverDestroySyevjInfo
    !---------------------------------------------
    function hipsolverDestroySyevjInfo(myInfo) &
       result(DestroySyevjInfo) &
       bind(C, name="hipsolverDestroySyevjInfo")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DestroySyevjInfo
    end function hipsolverDestroySyevjInfo

    !---------------------------------------------
    ! hipsolverXsyevjSetMaxSweeps
    !---------------------------------------------
    function hipsolverXsyevjSetMaxSweeps(myInfo, max_sweeps) &
       result(XsyevjSetMaxSweeps) &
       bind(C, name="hipsolverXsyevjSetMaxSweeps")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(c_int), value :: max_sweeps
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XsyevjSetMaxSweeps
    end function hipsolverXsyevjSetMaxSweeps

    !---------------------------------------------
    ! hipsolverXsyevjSetSortEig
    !---------------------------------------------
    function hipsolverXsyevjSetSortEig(myInfo, sort_eig) &
       result(XsyevjSetSortEig) &
       bind(C, name="hipsolverXsyevjSetSortEig")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(c_int), value :: sort_eig
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XsyevjSetSortEig
    end function hipsolverXsyevjSetSortEig

    !---------------------------------------------
    ! hipsolverXsyevjSetTolerance
    !---------------------------------------------
    function hipsolverXsyevjSetTolerance(myInfo, tolerance) &
       result(XsyevjSetTolerance) &
       bind(C, name="hipsolverXsyevjSetTolerance")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       real(c_double), value :: tolerance
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XsyevjSetTolerance
    end function hipsolverXsyevjSetTolerance

    !---------------------------------------------
    ! hipsolverXsyevjGetResidual
    !---------------------------------------------
    function hipsolverXsyevjGetResidual(handle, myInfo, residual) &
       result(XsyevjGetResidual) &
       bind(C, name="hipsolverXsyevjGetResidual")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: myInfo
       real(c_double) :: residual
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XsyevjGetResidual
    end function hipsolverXsyevjGetResidual

    !---------------------------------------------
    ! hipsolverXsyevjGetSweeps
    !---------------------------------------------
    function hipsolverXsyevjGetSweeps(handle, myInfo, executed_sweeps) &
       result(XsyevjGetSweeps) &
       bind(C, name="hipsolverXsyevjGetSweeps")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: myInfo
       integer(c_int) :: executed_sweeps
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: XsyevjGetSweeps
    end function hipsolverXsyevjGetSweeps

    !---------------------------------------------
    ! hipsolverSorgbr_bufferSize
    !---------------------------------------------
    function hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
       result(Sorgbr_bufferSize) &
       bind(C, name="hipsolverSorgbr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sorgbr_bufferSize
    end function hipsolverSorgbr_bufferSize

    !---------------------------------------------
    ! hipsolverDorgbr_bufferSize
    !---------------------------------------------
    function hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
       result(Dorgbr_bufferSize) &
       bind(C, name="hipsolverDorgbr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dorgbr_bufferSize
    end function hipsolverDorgbr_bufferSize

    !---------------------------------------------
    ! hipsolverCungbr_bufferSize
    !---------------------------------------------
    function hipsolverCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
       result(Cungbr_bufferSize) &
       bind(C, name="hipsolverCungbr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cungbr_bufferSize
    end function hipsolverCungbr_bufferSize

    !---------------------------------------------
    ! hipsolverZungbr_bufferSize
    !---------------------------------------------
    function hipsolverZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
       result(Zungbr_bufferSize) &
       bind(C, name="hipsolverZungbr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zungbr_bufferSize
    end function hipsolverZungbr_bufferSize

    !---------------------------------------------
    ! hipsolverSorgbr
    !---------------------------------------------
    function hipsolverSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(Sorgbr) &
       bind(C, name="hipsolverSorgbr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sorgbr
    end function hipsolverSorgbr

    !---------------------------------------------
    ! hipsolverDorgbr
    !---------------------------------------------
    function hipsolverDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(Dorgbr) &
       bind(C, name="hipsolverDorgbr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dorgbr
    end function hipsolverDorgbr

    !---------------------------------------------
    ! hipsolverCungbr
    !---------------------------------------------
    function hipsolverCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(Cungbr) &
       bind(C, name="hipsolverCungbr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cungbr
    end function hipsolverCungbr

    !---------------------------------------------
    ! hipsolverZungbr
    !---------------------------------------------
    function hipsolverZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(Zungbr) &
       bind(C, name="hipsolverZungbr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zungbr
    end function hipsolverZungbr

    !---------------------------------------------
    ! hipsolverSorgqr_bufferSize
    !---------------------------------------------
    function hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
       result(Sorgqr_bufferSize) &
       bind(C, name="hipsolverSorgqr_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sorgqr_bufferSize
    end function hipsolverSorgqr_bufferSize

    !---------------------------------------------
    ! hipsolverDorgqr_bufferSize
    !---------------------------------------------
    function hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
       result(Dorgqr_bufferSize) &
       bind(C, name="hipsolverDorgqr_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dorgqr_bufferSize
    end function hipsolverDorgqr_bufferSize

    !---------------------------------------------
    ! hipsolverCungqr_bufferSize
    !---------------------------------------------
    function hipsolverCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
       result(Cungqr_bufferSize) &
       bind(C, name="hipsolverCungqr_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cungqr_bufferSize
    end function hipsolverCungqr_bufferSize

    !---------------------------------------------
    ! hipsolverZungqr_bufferSize
    !---------------------------------------------
    function hipsolverZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
       result(Zungqr_bufferSize) &
       bind(C, name="hipsolverZungqr_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zungqr_bufferSize
    end function hipsolverZungqr_bufferSize

    !---------------------------------------------
    ! hipsolverSorgqr
    !---------------------------------------------
    function hipsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(Sorgqr) &
       bind(C, name="hipsolverSorgqr")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sorgqr
    end function hipsolverSorgqr

    !---------------------------------------------
    ! hipsolverDorgqr
    !---------------------------------------------
    function hipsolverDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(Dorgqr) &
       bind(C, name="hipsolverDorgqr")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dorgqr
    end function hipsolverDorgqr

    !---------------------------------------------
    ! hipsolverCungqr
    !---------------------------------------------
    function hipsolverCungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(Cungqr) &
       bind(C, name="hipsolverCungqr")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cungqr
    end function hipsolverCungqr

    !---------------------------------------------
    ! hipsolverZungqr
    !---------------------------------------------
    function hipsolverZungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(Zungqr) &
       bind(C, name="hipsolverZungqr")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zungqr
    end function hipsolverZungqr

    !---------------------------------------------
    ! hipsolverSorgtr_bufferSize
    !---------------------------------------------
    function hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
       result(Sorgtr_bufferSize) &
       bind(C, name="hipsolverSorgtr_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sorgtr_bufferSize
    end function hipsolverSorgtr_bufferSize

    !---------------------------------------------
    ! hipsolverDorgtr_bufferSize
    !---------------------------------------------
    function hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
       result(Dorgtr_bufferSize) &
       bind(C, name="hipsolverDorgtr_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dorgtr_bufferSize
    end function hipsolverDorgtr_bufferSize

    !---------------------------------------------
    ! hipsolverCungtr_bufferSize
    !---------------------------------------------
    function hipsolverCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
       result(Cungtr_bufferSize) &
       bind(C, name="hipsolverCungtr_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cungtr_bufferSize
    end function hipsolverCungtr_bufferSize

    !---------------------------------------------
    ! hipsolverZungtr_bufferSize
    !---------------------------------------------
    function hipsolverZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
       result(Zungtr_bufferSize) &
       bind(C, name="hipsolverZungtr_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zungtr_bufferSize
    end function hipsolverZungtr_bufferSize

    !---------------------------------------------
    ! hipsolverSorgtr
    !---------------------------------------------
    function hipsolverSorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo) &
       result(Sorgtr) &
       bind(C, name="hipsolverSorgtr")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sorgtr
    end function hipsolverSorgtr

    !---------------------------------------------
    ! hipsolverDorgtr
    !---------------------------------------------
    function hipsolverDorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo) &
       result(Dorgtr) &
       bind(C, name="hipsolverDorgtr")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dorgtr
    end function hipsolverDorgtr

    !---------------------------------------------
    ! hipsolverCungtr
    !---------------------------------------------
    function hipsolverCungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo) &
       result(Cungtr) &
       bind(C, name="hipsolverCungtr")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cungtr
    end function hipsolverCungtr

    !---------------------------------------------
    ! hipsolverZungtr
    !---------------------------------------------
    function hipsolverZungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo) &
       result(Zungtr) &
       bind(C, name="hipsolverZungtr")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zungtr
    end function hipsolverZungtr

    !---------------------------------------------
    ! hipsolverSormqr_bufferSize
    !---------------------------------------------
    function hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
       result(Sormqr_bufferSize) &
       bind(C, name="hipsolverSormqr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sormqr_bufferSize
    end function hipsolverSormqr_bufferSize

    !---------------------------------------------
    ! hipsolverDormqr_bufferSize
    !---------------------------------------------
    function hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
       result(Dormqr_bufferSize) &
       bind(C, name="hipsolverDormqr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dormqr_bufferSize
    end function hipsolverDormqr_bufferSize

    !---------------------------------------------
    ! hipsolverCunmqr_bufferSize
    !---------------------------------------------
    function hipsolverCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
       result(Cunmqr_bufferSize) &
       bind(C, name="hipsolverCunmqr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cunmqr_bufferSize
    end function hipsolverCunmqr_bufferSize

    !---------------------------------------------
    ! hipsolverZunmqr_bufferSize
    !---------------------------------------------
    function hipsolverZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork) &
       result(Zunmqr_bufferSize) &
       bind(C, name="hipsolverZunmqr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zunmqr_bufferSize
    end function hipsolverZunmqr_bufferSize

    !---------------------------------------------
    ! hipsolverSormqr
    !---------------------------------------------
    function hipsolverSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &
                             devInfo) &
       result(Sormqr) &
       bind(C, name="hipsolverSormqr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sormqr
    end function hipsolverSormqr

    !---------------------------------------------
    ! hipsolverDormqr
    !---------------------------------------------
    function hipsolverDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &
                             devInfo) &
       result(Dormqr) &
       bind(C, name="hipsolverDormqr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dormqr
    end function hipsolverDormqr

    !---------------------------------------------
    ! hipsolverCunmqr
    !---------------------------------------------
    function hipsolverCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &
                             devInfo) &
       result(Cunmqr) &
       bind(C, name="hipsolverCunmqr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cunmqr
    end function hipsolverCunmqr

    !---------------------------------------------
    ! hipsolverZunmqr
    !---------------------------------------------
    function hipsolverZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &
                             devInfo) &
       result(Zunmqr) &
       bind(C, name="hipsolverZunmqr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zunmqr
    end function hipsolverZunmqr

    !---------------------------------------------
    ! hipsolverSormtr_bufferSize
    !---------------------------------------------
    function hipsolverSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &
                                        lwork) &
       result(Sormtr_bufferSize) &
       bind(C, name="hipsolverSormtr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sormtr_bufferSize
    end function hipsolverSormtr_bufferSize

    !---------------------------------------------
    ! hipsolverDormtr_bufferSize
    !---------------------------------------------
    function hipsolverDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &
                                        lwork) &
       result(Dormtr_bufferSize) &
       bind(C, name="hipsolverDormtr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dormtr_bufferSize
    end function hipsolverDormtr_bufferSize

    !---------------------------------------------
    ! hipsolverCunmtr_bufferSize
    !---------------------------------------------
    function hipsolverCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &
                                        lwork) &
       result(Cunmtr_bufferSize) &
       bind(C, name="hipsolverCunmtr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cunmtr_bufferSize
    end function hipsolverCunmtr_bufferSize

    !---------------------------------------------
    ! hipsolverZunmtr_bufferSize
    !---------------------------------------------
    function hipsolverZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &
                                        lwork) &
       result(Zunmtr_bufferSize) &
       bind(C, name="hipsolverZunmtr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zunmtr_bufferSize
    end function hipsolverZunmtr_bufferSize

    !---------------------------------------------
    ! hipsolverSormtr
    !---------------------------------------------
    function hipsolverSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, &
                             devInfo) &
       result(Sormtr) &
       bind(C, name="hipsolverSormtr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sormtr
    end function hipsolverSormtr

    !---------------------------------------------
    ! hipsolverDormtr
    !---------------------------------------------
    function hipsolverDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, &
                             devInfo) &
       result(Dormtr) &
       bind(C, name="hipsolverDormtr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dormtr
    end function hipsolverDormtr

    !---------------------------------------------
    ! hipsolverCunmtr
    !---------------------------------------------
    function hipsolverCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, &
                             devInfo) &
       result(Cunmtr) &
       bind(C, name="hipsolverCunmtr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cunmtr
    end function hipsolverCunmtr

    !---------------------------------------------
    ! hipsolverZunmtr
    !---------------------------------------------
    function hipsolverZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, &
                             devInfo) &
       result(Zunmtr) &
       bind(C, name="hipsolverZunmtr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zunmtr
    end function hipsolverZunmtr

    !---------------------------------------------
    ! hipsolverSgebrd_bufferSize
    !---------------------------------------------
    function hipsolverSgebrd_bufferSize(handle, m, n, lwork) &
       result(Sgebrd_bufferSize) &
       bind(C, name="hipsolverSgebrd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgebrd_bufferSize
    end function hipsolverSgebrd_bufferSize

    !---------------------------------------------
    ! hipsolverDgebrd_bufferSize
    !---------------------------------------------
    function hipsolverDgebrd_bufferSize(handle, m, n, lwork) &
       result(Dgebrd_bufferSize) &
       bind(C, name="hipsolverDgebrd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgebrd_bufferSize
    end function hipsolverDgebrd_bufferSize

    !---------------------------------------------
    ! hipsolverCgebrd_bufferSize
    !---------------------------------------------
    function hipsolverCgebrd_bufferSize(handle, m, n, lwork) &
       result(Cgebrd_bufferSize) &
       bind(C, name="hipsolverCgebrd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgebrd_bufferSize
    end function hipsolverCgebrd_bufferSize

    !---------------------------------------------
    ! hipsolverZgebrd_bufferSize
    !---------------------------------------------
    function hipsolverZgebrd_bufferSize(handle, m, n, lwork) &
       result(Zgebrd_bufferSize) &
       bind(C, name="hipsolverZgebrd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgebrd_bufferSize
    end function hipsolverZgebrd_bufferSize

    !---------------------------------------------
    ! hipsolverSgebrd
    !---------------------------------------------
    function hipsolverSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo) &
       result(Sgebrd) &
       bind(C, name="hipsolverSgebrd")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgebrd
    end function hipsolverSgebrd

    !---------------------------------------------
    ! hipsolverDgebrd
    !---------------------------------------------
    function hipsolverDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo) &
       result(Dgebrd) &
       bind(C, name="hipsolverDgebrd")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgebrd
    end function hipsolverDgebrd

    !---------------------------------------------
    ! hipsolverCgebrd
    !---------------------------------------------
    function hipsolverCgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo) &
       result(Cgebrd) &
       bind(C, name="hipsolverCgebrd")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgebrd
    end function hipsolverCgebrd

    !---------------------------------------------
    ! hipsolverZgebrd
    !---------------------------------------------
    function hipsolverZgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo) &
       result(Zgebrd) &
       bind(C, name="hipsolverZgebrd")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgebrd
    end function hipsolverZgebrd

    !---------------------------------------------
    ! hipsolverSSgels_bufferSize
    !---------------------------------------------
    function hipsolverSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
       result(SSgels_bufferSize) &
       bind(C, name="hipsolverSSgels_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SSgels_bufferSize
    end function hipsolverSSgels_bufferSize

    !---------------------------------------------
    ! hipsolverDDgels_bufferSize
    !---------------------------------------------
    function hipsolverDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
       result(DDgels_bufferSize) &
       bind(C, name="hipsolverDDgels_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DDgels_bufferSize
    end function hipsolverDDgels_bufferSize

    !---------------------------------------------
    ! hipsolverCCgels_bufferSize
    !---------------------------------------------
    function hipsolverCCgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
       result(CCgels_bufferSize) &
       bind(C, name="hipsolverCCgels_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CCgels_bufferSize
    end function hipsolverCCgels_bufferSize

    !---------------------------------------------
    ! hipsolverZZgels_bufferSize
    !---------------------------------------------
    function hipsolverZZgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork) &
       result(ZZgels_bufferSize) &
       bind(C, name="hipsolverZZgels_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZZgels_bufferSize
    end function hipsolverZZgels_bufferSize

    !---------------------------------------------
    ! hipsolverSSgels
    !---------------------------------------------
    function hipsolverSSgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, &
                             devInfo) &
       result(SSgels) &
       bind(C, name="hipsolverSSgels")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SSgels
    end function hipsolverSSgels

    !---------------------------------------------
    ! hipsolverDDgels
    !---------------------------------------------
    function hipsolverDDgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, &
                             devInfo) &
       result(DDgels) &
       bind(C, name="hipsolverDDgels")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DDgels
    end function hipsolverDDgels

    !---------------------------------------------
    ! hipsolverCCgels
    !---------------------------------------------
    function hipsolverCCgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, &
                             devInfo) &
       result(CCgels) &
       bind(C, name="hipsolverCCgels")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CCgels
    end function hipsolverCCgels

    !---------------------------------------------
    ! hipsolverZZgels
    !---------------------------------------------
    function hipsolverZZgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, &
                             devInfo) &
       result(ZZgels) &
       bind(C, name="hipsolverZZgels")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZZgels
    end function hipsolverZZgels

    !---------------------------------------------
    ! hipsolverSgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(Sgeqrf_bufferSize) &
       bind(C, name="hipsolverSgeqrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgeqrf_bufferSize
    end function hipsolverSgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverDgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(Dgeqrf_bufferSize) &
       bind(C, name="hipsolverDgeqrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgeqrf_bufferSize
    end function hipsolverDgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverCgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverCgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(Cgeqrf_bufferSize) &
       bind(C, name="hipsolverCgeqrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgeqrf_bufferSize
    end function hipsolverCgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverZgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverZgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(Zgeqrf_bufferSize) &
       bind(C, name="hipsolverZgeqrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgeqrf_bufferSize
    end function hipsolverZgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverSgeqrf
    !---------------------------------------------
    function hipsolverSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo) &
       result(Sgeqrf) &
       bind(C, name="hipsolverSgeqrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgeqrf
    end function hipsolverSgeqrf

    !---------------------------------------------
    ! hipsolverDgeqrf
    !---------------------------------------------
    function hipsolverDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo) &
       result(Dgeqrf) &
       bind(C, name="hipsolverDgeqrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgeqrf
    end function hipsolverDgeqrf

    !---------------------------------------------
    ! hipsolverCgeqrf
    !---------------------------------------------
    function hipsolverCgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo) &
       result(Cgeqrf) &
       bind(C, name="hipsolverCgeqrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgeqrf
    end function hipsolverCgeqrf

    !---------------------------------------------
    ! hipsolverZgeqrf
    !---------------------------------------------
    function hipsolverZgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo) &
       result(Zgeqrf) &
       bind(C, name="hipsolverZgeqrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgeqrf
    end function hipsolverZgeqrf

    !---------------------------------------------
    ! hipsolverSSgesv_bufferSize
    !---------------------------------------------
    function hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, lwork) &
       result(SSgesv_bufferSize) &
       bind(C, name="hipsolverSSgesv_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SSgesv_bufferSize
    end function hipsolverSSgesv_bufferSize

    !---------------------------------------------
    ! hipsolverDDgesv_bufferSize
    !---------------------------------------------
    function hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, lwork) &
       result(DDgesv_bufferSize) &
       bind(C, name="hipsolverDDgesv_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DDgesv_bufferSize
    end function hipsolverDDgesv_bufferSize

    !---------------------------------------------
    ! hipsolverCCgesv_bufferSize
    !---------------------------------------------
    function hipsolverCCgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, lwork) &
       result(CCgesv_bufferSize) &
       bind(C, name="hipsolverCCgesv_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CCgesv_bufferSize
    end function hipsolverCCgesv_bufferSize

    !---------------------------------------------
    ! hipsolverZZgesv_bufferSize
    !---------------------------------------------
    function hipsolverZZgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, lwork) &
       result(ZZgesv_bufferSize) &
       bind(C, name="hipsolverZZgesv_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZZgesv_bufferSize
    end function hipsolverZZgesv_bufferSize

    !---------------------------------------------
    ! hipsolverSSgesv
    !---------------------------------------------
    function hipsolverSSgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, &
                             niters, devInfo) &
       result(SSgesv) &
       bind(C, name="hipsolverSSgesv")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SSgesv
    end function hipsolverSSgesv

    !---------------------------------------------
    ! hipsolverDDgesv
    !---------------------------------------------
    function hipsolverDDgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, &
                             niters, devInfo) &
       result(DDgesv) &
       bind(C, name="hipsolverDDgesv")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DDgesv
    end function hipsolverDDgesv

    !---------------------------------------------
    ! hipsolverCCgesv
    !---------------------------------------------
    function hipsolverCCgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, &
                             niters, devInfo) &
       result(CCgesv) &
       bind(C, name="hipsolverCCgesv")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CCgesv
    end function hipsolverCCgesv

    !---------------------------------------------
    ! hipsolverZZgesv
    !---------------------------------------------
    function hipsolverZZgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, &
                             niters, devInfo) &
       result(ZZgesv) &
       bind(C, name="hipsolverZZgesv")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZZgesv
    end function hipsolverZZgesv

    !---------------------------------------------
    ! hipsolverSgesvd_bufferSize
    !---------------------------------------------
    function hipsolverSgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
       result(Sgesvd_bufferSize) &
       bind(C, name="hipsolverSgesvd_bufferSize")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgesvd_bufferSize
    end function hipsolverSgesvd_bufferSize

    !---------------------------------------------
    ! hipsolverDgesvd_bufferSize
    !---------------------------------------------
    function hipsolverDgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
       result(Dgesvd_bufferSize) &
       bind(C, name="hipsolverDgesvd_bufferSize")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgesvd_bufferSize
    end function hipsolverDgesvd_bufferSize

    !---------------------------------------------
    ! hipsolverCgesvd_bufferSize
    !---------------------------------------------
    function hipsolverCgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
       result(Cgesvd_bufferSize) &
       bind(C, name="hipsolverCgesvd_bufferSize")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgesvd_bufferSize
    end function hipsolverCgesvd_bufferSize

    !---------------------------------------------
    ! hipsolverZgesvd_bufferSize
    !---------------------------------------------
    function hipsolverZgesvd_bufferSize(handle, jobu, jobv, m, n, lwork) &
       result(Zgesvd_bufferSize) &
       bind(C, name="hipsolverZgesvd_bufferSize")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgesvd_bufferSize
    end function hipsolverZgesvd_bufferSize

    !---------------------------------------------
    ! hipsolverSgesvd
    !---------------------------------------------
    function hipsolverSgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                             rwork, devInfo) &
       result(Sgesvd) &
       bind(C, name="hipsolverSgesvd")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: rwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgesvd
    end function hipsolverSgesvd

    !---------------------------------------------
    ! hipsolverDgesvd
    !---------------------------------------------
    function hipsolverDgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                             rwork, devInfo) &
       result(Dgesvd) &
       bind(C, name="hipsolverDgesvd")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: rwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgesvd
    end function hipsolverDgesvd

    !---------------------------------------------
    ! hipsolverCgesvd
    !---------------------------------------------
    function hipsolverCgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                             rwork, devInfo) &
       result(Cgesvd) &
       bind(C, name="hipsolverCgesvd")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: rwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgesvd
    end function hipsolverCgesvd

    !---------------------------------------------
    ! hipsolverZgesvd
    !---------------------------------------------
    function hipsolverZgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                             rwork, devInfo) &
       result(Zgesvd) &
       bind(C, name="hipsolverZgesvd")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: rwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgesvd
    end function hipsolverZgesvd

    !---------------------------------------------
    ! hipsolverSgesvdj_bufferSize
    !---------------------------------------------
    function hipsolverSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &
                                         lwork, params) &
       result(Sgesvdj_bufferSize) &
       bind(C, name="hipsolverSgesvdj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgesvdj_bufferSize
    end function hipsolverSgesvdj_bufferSize

    !---------------------------------------------
    ! hipsolverDgesvdj_bufferSize
    !---------------------------------------------
    function hipsolverDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &
                                         lwork, params) &
       result(Dgesvdj_bufferSize) &
       bind(C, name="hipsolverDgesvdj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgesvdj_bufferSize
    end function hipsolverDgesvdj_bufferSize

    !---------------------------------------------
    ! hipsolverCgesvdj_bufferSize
    !---------------------------------------------
    function hipsolverCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &
                                         lwork, params) &
       result(Cgesvdj_bufferSize) &
       bind(C, name="hipsolverCgesvdj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgesvdj_bufferSize
    end function hipsolverCgesvdj_bufferSize

    !---------------------------------------------
    ! hipsolverZgesvdj_bufferSize
    !---------------------------------------------
    function hipsolverZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &
                                         lwork, params) &
       result(Zgesvdj_bufferSize) &
       bind(C, name="hipsolverZgesvdj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgesvdj_bufferSize
    end function hipsolverZgesvdj_bufferSize

    !---------------------------------------------
    ! hipsolverSgesvdj
    !---------------------------------------------
    function hipsolverSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                              devInfo, params) &
       result(Sgesvdj) &
       bind(C, name="hipsolverSgesvdj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgesvdj
    end function hipsolverSgesvdj

    !---------------------------------------------
    ! hipsolverDgesvdj
    !---------------------------------------------
    function hipsolverDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                              devInfo, params) &
       result(Dgesvdj) &
       bind(C, name="hipsolverDgesvdj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgesvdj
    end function hipsolverDgesvdj

    !---------------------------------------------
    ! hipsolverCgesvdj
    !---------------------------------------------
    function hipsolverCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                              devInfo, params) &
       result(Cgesvdj) &
       bind(C, name="hipsolverCgesvdj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgesvdj
    end function hipsolverCgesvdj

    !---------------------------------------------
    ! hipsolverZgesvdj
    !---------------------------------------------
    function hipsolverZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                              devInfo, params) &
       result(Zgesvdj) &
       bind(C, name="hipsolverZgesvdj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgesvdj
    end function hipsolverZgesvdj

    !---------------------------------------------
    ! hipsolverSgesvdjBatched_bufferSize
    !---------------------------------------------
    function hipsolverSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &
                                                lwork, params, batch_count) &
       result(SgesvdjBatched_bufferSize) &
       bind(C, name="hipsolverSgesvdjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SgesvdjBatched_bufferSize
    end function hipsolverSgesvdjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDgesvdjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &
                                                lwork, params, batch_count) &
       result(DgesvdjBatched_bufferSize) &
       bind(C, name="hipsolverDgesvdjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DgesvdjBatched_bufferSize
    end function hipsolverDgesvdjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverCgesvdjBatched_bufferSize
    !---------------------------------------------
    function hipsolverCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &
                                                lwork, params, batch_count) &
       result(CgesvdjBatched_bufferSize) &
       bind(C, name="hipsolverCgesvdjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CgesvdjBatched_bufferSize
    end function hipsolverCgesvdjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverZgesvdjBatched_bufferSize
    !---------------------------------------------
    function hipsolverZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &
                                                lwork, params, batch_count) &
       result(ZgesvdjBatched_bufferSize) &
       bind(C, name="hipsolverZgesvdjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZgesvdjBatched_bufferSize
    end function hipsolverZgesvdjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverSgesvdjBatched
    !---------------------------------------------
    function hipsolverSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                     devInfo, params, batch_count) &
       result(SgesvdjBatched) &
       bind(C, name="hipsolverSgesvdjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SgesvdjBatched
    end function hipsolverSgesvdjBatched

    !---------------------------------------------
    ! hipsolverDgesvdjBatched
    !---------------------------------------------
    function hipsolverDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                     devInfo, params, batch_count) &
       result(DgesvdjBatched) &
       bind(C, name="hipsolverDgesvdjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DgesvdjBatched
    end function hipsolverDgesvdjBatched

    !---------------------------------------------
    ! hipsolverCgesvdjBatched
    !---------------------------------------------
    function hipsolverCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                     devInfo, params, batch_count) &
       result(CgesvdjBatched) &
       bind(C, name="hipsolverCgesvdjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CgesvdjBatched
    end function hipsolverCgesvdjBatched

    !---------------------------------------------
    ! hipsolverZgesvdjBatched
    !---------------------------------------------
    function hipsolverZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                     devInfo, params, batch_count) &
       result(ZgesvdjBatched) &
       bind(C, name="hipsolverZgesvdjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZgesvdjBatched
    end function hipsolverZgesvdjBatched

    !---------------------------------------------
    ! hipsolverSgetrf_bufferSize
    !---------------------------------------------
    function hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(Sgetrf_bufferSize) &
       bind(C, name="hipsolverSgetrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgetrf_bufferSize
    end function hipsolverSgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverDgetrf_bufferSize
    !---------------------------------------------
    function hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(Dgetrf_bufferSize) &
       bind(C, name="hipsolverDgetrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgetrf_bufferSize
    end function hipsolverDgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverCgetrf_bufferSize
    !---------------------------------------------
    function hipsolverCgetrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(Cgetrf_bufferSize) &
       bind(C, name="hipsolverCgetrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgetrf_bufferSize
    end function hipsolverCgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverZgetrf_bufferSize
    !---------------------------------------------
    function hipsolverZgetrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(Zgetrf_bufferSize) &
       bind(C, name="hipsolverZgetrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgetrf_bufferSize
    end function hipsolverZgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverSgetrf
    !---------------------------------------------
    function hipsolverSgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo) &
       result(Sgetrf) &
       bind(C, name="hipsolverSgetrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgetrf
    end function hipsolverSgetrf

    !---------------------------------------------
    ! hipsolverDgetrf
    !---------------------------------------------
    function hipsolverDgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo) &
       result(Dgetrf) &
       bind(C, name="hipsolverDgetrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgetrf
    end function hipsolverDgetrf

    !---------------------------------------------
    ! hipsolverCgetrf
    !---------------------------------------------
    function hipsolverCgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo) &
       result(Cgetrf) &
       bind(C, name="hipsolverCgetrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgetrf
    end function hipsolverCgetrf

    !---------------------------------------------
    ! hipsolverZgetrf
    !---------------------------------------------
    function hipsolverZgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo) &
       result(Zgetrf) &
       bind(C, name="hipsolverZgetrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgetrf
    end function hipsolverZgetrf

    !---------------------------------------------
    ! hipsolverSgetrfBatched_bufferSize
    !---------------------------------------------
    function hipsolverSgetrfBatched_bufferSize(handle, m, n, A, lda, strideP, lwork, batch_count) &
       result(SgetrfBatched_bufferSize) &
       bind(C, name="hipsolverSgetrfBatched_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int), value :: strideP
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SgetrfBatched_bufferSize
    end function hipsolverSgetrfBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDgetrfBatched_bufferSize
    !---------------------------------------------
    function hipsolverDgetrfBatched_bufferSize(handle, m, n, A, lda, strideP, lwork, batch_count) &
       result(DgetrfBatched_bufferSize) &
       bind(C, name="hipsolverDgetrfBatched_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int), value :: strideP
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DgetrfBatched_bufferSize
    end function hipsolverDgetrfBatched_bufferSize

    !---------------------------------------------
    ! hipsolverCgetrfBatched_bufferSize
    !---------------------------------------------
    function hipsolverCgetrfBatched_bufferSize(handle, m, n, A, lda, strideP, lwork, batch_count) &
       result(CgetrfBatched_bufferSize) &
       bind(C, name="hipsolverCgetrfBatched_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int), value :: strideP
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CgetrfBatched_bufferSize
    end function hipsolverCgetrfBatched_bufferSize

    !---------------------------------------------
    ! hipsolverZgetrfBatched_bufferSize
    !---------------------------------------------
    function hipsolverZgetrfBatched_bufferSize(handle, m, n, A, lda, strideP, lwork, batch_count) &
       result(ZgetrfBatched_bufferSize) &
       bind(C, name="hipsolverZgetrfBatched_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int), value :: strideP
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZgetrfBatched_bufferSize
    end function hipsolverZgetrfBatched_bufferSize

    !---------------------------------------------
    ! hipsolverSgetrfBatched
    !---------------------------------------------
    function hipsolverSgetrfBatched(handle, m, n, A, lda, work, lwork, devIpiv, strideP, devInfo, &
                                    batch_count) &
       result(SgetrfBatched) &
       bind(C, name="hipsolverSgetrfBatched")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devIpiv
       integer(c_int), value :: strideP
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SgetrfBatched
    end function hipsolverSgetrfBatched

    !---------------------------------------------
    ! hipsolverDgetrfBatched
    !---------------------------------------------
    function hipsolverDgetrfBatched(handle, m, n, A, lda, work, lwork, devIpiv, strideP, devInfo, &
                                    batch_count) &
       result(DgetrfBatched) &
       bind(C, name="hipsolverDgetrfBatched")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devIpiv
       integer(c_int), value :: strideP
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DgetrfBatched
    end function hipsolverDgetrfBatched

    !---------------------------------------------
    ! hipsolverCgetrfBatched
    !---------------------------------------------
    function hipsolverCgetrfBatched(handle, m, n, A, lda, work, lwork, devIpiv, strideP, devInfo, &
                                    batch_count) &
       result(CgetrfBatched) &
       bind(C, name="hipsolverCgetrfBatched")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devIpiv
       integer(c_int), value :: strideP
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CgetrfBatched
    end function hipsolverCgetrfBatched

    !---------------------------------------------
    ! hipsolverZgetrfBatched
    !---------------------------------------------
    function hipsolverZgetrfBatched(handle, m, n, A, lda, work, lwork, devIpiv, strideP, devInfo, &
                                    batch_count) &
       result(ZgetrfBatched) &
       bind(C, name="hipsolverZgetrfBatched")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devIpiv
       integer(c_int), value :: strideP
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZgetrfBatched
    end function hipsolverZgetrfBatched

    !---------------------------------------------
    ! hipsolverSgetrs_bufferSize
    !---------------------------------------------
    function hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, lwork) &
       result(Sgetrs_bufferSize) &
       bind(C, name="hipsolverSgetrs_bufferSize")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgetrs_bufferSize
    end function hipsolverSgetrs_bufferSize

    !---------------------------------------------
    ! hipsolverDgetrs_bufferSize
    !---------------------------------------------
    function hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, lwork) &
       result(Dgetrs_bufferSize) &
       bind(C, name="hipsolverDgetrs_bufferSize")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgetrs_bufferSize
    end function hipsolverDgetrs_bufferSize

    !---------------------------------------------
    ! hipsolverCgetrs_bufferSize
    !---------------------------------------------
    function hipsolverCgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, lwork) &
       result(Cgetrs_bufferSize) &
       bind(C, name="hipsolverCgetrs_bufferSize")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgetrs_bufferSize
    end function hipsolverCgetrs_bufferSize

    !---------------------------------------------
    ! hipsolverZgetrs_bufferSize
    !---------------------------------------------
    function hipsolverZgetrs_bufferSize(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, lwork) &
       result(Zgetrs_bufferSize) &
       bind(C, name="hipsolverZgetrs_bufferSize")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgetrs_bufferSize
    end function hipsolverZgetrs_bufferSize

    !---------------------------------------------
    ! hipsolverSgetrs
    !---------------------------------------------
    function hipsolverSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, work, lwork, &
                             devInfo) &
       result(Sgetrs) &
       bind(C, name="hipsolverSgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Sgetrs
    end function hipsolverSgetrs

    !---------------------------------------------
    ! hipsolverDgetrs
    !---------------------------------------------
    function hipsolverDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, work, lwork, &
                             devInfo) &
       result(Dgetrs) &
       bind(C, name="hipsolverDgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dgetrs
    end function hipsolverDgetrs

    !---------------------------------------------
    ! hipsolverCgetrs
    !---------------------------------------------
    function hipsolverCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, work, lwork, &
                             devInfo) &
       result(Cgetrs) &
       bind(C, name="hipsolverCgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cgetrs
    end function hipsolverCgetrs

    !---------------------------------------------
    ! hipsolverZgetrs
    !---------------------------------------------
    function hipsolverZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, work, lwork, &
                             devInfo) &
       result(Zgetrs) &
       bind(C, name="hipsolverZgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zgetrs
    end function hipsolverZgetrs

    !---------------------------------------------
    ! hipsolverSpotrf_bufferSize
    !---------------------------------------------
    function hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(Spotrf_bufferSize) &
       bind(C, name="hipsolverSpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Spotrf_bufferSize
    end function hipsolverSpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverDpotrf_bufferSize
    !---------------------------------------------
    function hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(Dpotrf_bufferSize) &
       bind(C, name="hipsolverDpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dpotrf_bufferSize
    end function hipsolverDpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverCpotrf_bufferSize
    !---------------------------------------------
    function hipsolverCpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(Cpotrf_bufferSize) &
       bind(C, name="hipsolverCpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cpotrf_bufferSize
    end function hipsolverCpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverZpotrf_bufferSize
    !---------------------------------------------
    function hipsolverZpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(Zpotrf_bufferSize) &
       bind(C, name="hipsolverZpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zpotrf_bufferSize
    end function hipsolverZpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverSpotrf
    !---------------------------------------------
    function hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(Spotrf) &
       bind(C, name="hipsolverSpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Spotrf
    end function hipsolverSpotrf

    !---------------------------------------------
    ! hipsolverDpotrf
    !---------------------------------------------
    function hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(Dpotrf) &
       bind(C, name="hipsolverDpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dpotrf
    end function hipsolverDpotrf

    !---------------------------------------------
    ! hipsolverCpotrf
    !---------------------------------------------
    function hipsolverCpotrf(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(Cpotrf) &
       bind(C, name="hipsolverCpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cpotrf
    end function hipsolverCpotrf

    !---------------------------------------------
    ! hipsolverZpotrf
    !---------------------------------------------
    function hipsolverZpotrf(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(Zpotrf) &
       bind(C, name="hipsolverZpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zpotrf
    end function hipsolverZpotrf

    !---------------------------------------------
    ! hipsolverSpotrfBatched_bufferSize
    !---------------------------------------------
    function hipsolverSpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
       result(SpotrfBatched_bufferSize) &
       bind(C, name="hipsolverSpotrfBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpotrfBatched_bufferSize
    end function hipsolverSpotrfBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDpotrfBatched_bufferSize
    !---------------------------------------------
    function hipsolverDpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
       result(DpotrfBatched_bufferSize) &
       bind(C, name="hipsolverDpotrfBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DpotrfBatched_bufferSize
    end function hipsolverDpotrfBatched_bufferSize

    !---------------------------------------------
    ! hipsolverCpotrfBatched_bufferSize
    !---------------------------------------------
    function hipsolverCpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
       result(CpotrfBatched_bufferSize) &
       bind(C, name="hipsolverCpotrfBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CpotrfBatched_bufferSize
    end function hipsolverCpotrfBatched_bufferSize

    !---------------------------------------------
    ! hipsolverZpotrfBatched_bufferSize
    !---------------------------------------------
    function hipsolverZpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, batch_count) &
       result(ZpotrfBatched_bufferSize) &
       bind(C, name="hipsolverZpotrfBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZpotrfBatched_bufferSize
    end function hipsolverZpotrfBatched_bufferSize

    !---------------------------------------------
    ! hipsolverSpotrfBatched
    !---------------------------------------------
    function hipsolverSpotrfBatched(handle, uplo, n, A, lda, work, lwork, devInfo, batch_count) &
       result(SpotrfBatched) &
       bind(C, name="hipsolverSpotrfBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpotrfBatched
    end function hipsolverSpotrfBatched

    !---------------------------------------------
    ! hipsolverDpotrfBatched
    !---------------------------------------------
    function hipsolverDpotrfBatched(handle, uplo, n, A, lda, work, lwork, devInfo, batch_count) &
       result(DpotrfBatched) &
       bind(C, name="hipsolverDpotrfBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DpotrfBatched
    end function hipsolverDpotrfBatched

    !---------------------------------------------
    ! hipsolverCpotrfBatched
    !---------------------------------------------
    function hipsolverCpotrfBatched(handle, uplo, n, A, lda, work, lwork, devInfo, batch_count) &
       result(CpotrfBatched) &
       bind(C, name="hipsolverCpotrfBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CpotrfBatched
    end function hipsolverCpotrfBatched

    !---------------------------------------------
    ! hipsolverZpotrfBatched
    !---------------------------------------------
    function hipsolverZpotrfBatched(handle, uplo, n, A, lda, work, lwork, devInfo, batch_count) &
       result(ZpotrfBatched) &
       bind(C, name="hipsolverZpotrfBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZpotrfBatched
    end function hipsolverZpotrfBatched

    !---------------------------------------------
    ! hipsolverSpotri_bufferSize
    !---------------------------------------------
    function hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(Spotri_bufferSize) &
       bind(C, name="hipsolverSpotri_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Spotri_bufferSize
    end function hipsolverSpotri_bufferSize

    !---------------------------------------------
    ! hipsolverDpotri_bufferSize
    !---------------------------------------------
    function hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(Dpotri_bufferSize) &
       bind(C, name="hipsolverDpotri_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dpotri_bufferSize
    end function hipsolverDpotri_bufferSize

    !---------------------------------------------
    ! hipsolverCpotri_bufferSize
    !---------------------------------------------
    function hipsolverCpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(Cpotri_bufferSize) &
       bind(C, name="hipsolverCpotri_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cpotri_bufferSize
    end function hipsolverCpotri_bufferSize

    !---------------------------------------------
    ! hipsolverZpotri_bufferSize
    !---------------------------------------------
    function hipsolverZpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(Zpotri_bufferSize) &
       bind(C, name="hipsolverZpotri_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zpotri_bufferSize
    end function hipsolverZpotri_bufferSize

    !---------------------------------------------
    ! hipsolverSpotri
    !---------------------------------------------
    function hipsolverSpotri(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(Spotri) &
       bind(C, name="hipsolverSpotri")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Spotri
    end function hipsolverSpotri

    !---------------------------------------------
    ! hipsolverDpotri
    !---------------------------------------------
    function hipsolverDpotri(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(Dpotri) &
       bind(C, name="hipsolverDpotri")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dpotri
    end function hipsolverDpotri

    !---------------------------------------------
    ! hipsolverCpotri
    !---------------------------------------------
    function hipsolverCpotri(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(Cpotri) &
       bind(C, name="hipsolverCpotri")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cpotri
    end function hipsolverCpotri

    !---------------------------------------------
    ! hipsolverZpotri
    !---------------------------------------------
    function hipsolverZpotri(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(Zpotri) &
       bind(C, name="hipsolverZpotri")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zpotri
    end function hipsolverZpotri

    !---------------------------------------------
    ! hipsolverSpotrs_bufferSize
    !---------------------------------------------
    function hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
       result(Spotrs_bufferSize) &
       bind(C, name="hipsolverSpotrs_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Spotrs_bufferSize
    end function hipsolverSpotrs_bufferSize

    !---------------------------------------------
    ! hipsolverDpotrs_bufferSize
    !---------------------------------------------
    function hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
       result(Dpotrs_bufferSize) &
       bind(C, name="hipsolverDpotrs_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dpotrs_bufferSize
    end function hipsolverDpotrs_bufferSize

    !---------------------------------------------
    ! hipsolverCpotrs_bufferSize
    !---------------------------------------------
    function hipsolverCpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
       result(Cpotrs_bufferSize) &
       bind(C, name="hipsolverCpotrs_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cpotrs_bufferSize
    end function hipsolverCpotrs_bufferSize

    !---------------------------------------------
    ! hipsolverZpotrs_bufferSize
    !---------------------------------------------
    function hipsolverZpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork) &
       result(Zpotrs_bufferSize) &
       bind(C, name="hipsolverZpotrs_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zpotrs_bufferSize
    end function hipsolverZpotrs_bufferSize

    !---------------------------------------------
    ! hipsolverSpotrs
    !---------------------------------------------
    function hipsolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo) &
       result(Spotrs) &
       bind(C, name="hipsolverSpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Spotrs
    end function hipsolverSpotrs

    !---------------------------------------------
    ! hipsolverDpotrs
    !---------------------------------------------
    function hipsolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo) &
       result(Dpotrs) &
       bind(C, name="hipsolverDpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dpotrs
    end function hipsolverDpotrs

    !---------------------------------------------
    ! hipsolverCpotrs
    !---------------------------------------------
    function hipsolverCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo) &
       result(Cpotrs) &
       bind(C, name="hipsolverCpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cpotrs
    end function hipsolverCpotrs

    !---------------------------------------------
    ! hipsolverZpotrs
    !---------------------------------------------
    function hipsolverZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo) &
       result(Zpotrs) &
       bind(C, name="hipsolverZpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zpotrs
    end function hipsolverZpotrs

    !---------------------------------------------
    ! hipsolverSpotrsBatched_bufferSize
    !---------------------------------------------
    function hipsolverSpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, &
                                               batch_count) &
       result(SpotrsBatched_bufferSize) &
       bind(C, name="hipsolverSpotrsBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpotrsBatched_bufferSize
    end function hipsolverSpotrsBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDpotrsBatched_bufferSize
    !---------------------------------------------
    function hipsolverDpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, &
                                               batch_count) &
       result(DpotrsBatched_bufferSize) &
       bind(C, name="hipsolverDpotrsBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DpotrsBatched_bufferSize
    end function hipsolverDpotrsBatched_bufferSize

    !---------------------------------------------
    ! hipsolverCpotrsBatched_bufferSize
    !---------------------------------------------
    function hipsolverCpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, &
                                               batch_count) &
       result(CpotrsBatched_bufferSize) &
       bind(C, name="hipsolverCpotrsBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CpotrsBatched_bufferSize
    end function hipsolverCpotrsBatched_bufferSize

    !---------------------------------------------
    ! hipsolverZpotrsBatched_bufferSize
    !---------------------------------------------
    function hipsolverZpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, &
                                               batch_count) &
       result(ZpotrsBatched_bufferSize) &
       bind(C, name="hipsolverZpotrsBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZpotrsBatched_bufferSize
    end function hipsolverZpotrsBatched_bufferSize

    !---------------------------------------------
    ! hipsolverSpotrsBatched
    !---------------------------------------------
    function hipsolverSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo, &
                                    batch_count) &
       result(SpotrsBatched) &
       bind(C, name="hipsolverSpotrsBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpotrsBatched
    end function hipsolverSpotrsBatched

    !---------------------------------------------
    ! hipsolverDpotrsBatched
    !---------------------------------------------
    function hipsolverDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo, &
                                    batch_count) &
       result(DpotrsBatched) &
       bind(C, name="hipsolverDpotrsBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DpotrsBatched
    end function hipsolverDpotrsBatched

    !---------------------------------------------
    ! hipsolverCpotrsBatched
    !---------------------------------------------
    function hipsolverCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo, &
                                    batch_count) &
       result(CpotrsBatched) &
       bind(C, name="hipsolverCpotrsBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CpotrsBatched
    end function hipsolverCpotrsBatched

    !---------------------------------------------
    ! hipsolverZpotrsBatched
    !---------------------------------------------
    function hipsolverZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, devInfo, &
                                    batch_count) &
       result(ZpotrsBatched) &
       bind(C, name="hipsolverZpotrsBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZpotrsBatched
    end function hipsolverZpotrsBatched

    !---------------------------------------------
    ! hipsolverSsyevd_bufferSize
    !---------------------------------------------
    function hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork) &
       result(Ssyevd_bufferSize) &
       bind(C, name="hipsolverSsyevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssyevd_bufferSize
    end function hipsolverSsyevd_bufferSize

    !---------------------------------------------
    ! hipsolverDsyevd_bufferSize
    !---------------------------------------------
    function hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork) &
       result(Dsyevd_bufferSize) &
       bind(C, name="hipsolverDsyevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsyevd_bufferSize
    end function hipsolverDsyevd_bufferSize

    !---------------------------------------------
    ! hipsolverCheevd_bufferSize
    !---------------------------------------------
    function hipsolverCheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork) &
       result(Cheevd_bufferSize) &
       bind(C, name="hipsolverCheevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cheevd_bufferSize
    end function hipsolverCheevd_bufferSize

    !---------------------------------------------
    ! hipsolverZheevd_bufferSize
    !---------------------------------------------
    function hipsolverZheevd_bufferSize(handle, jobz, uplo, n, A, lda, D, lwork) &
       result(Zheevd_bufferSize) &
       bind(C, name="hipsolverZheevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zheevd_bufferSize
    end function hipsolverZheevd_bufferSize

    !---------------------------------------------
    ! hipsolverSsyevd
    !---------------------------------------------
    function hipsolverSsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo) &
       result(Ssyevd) &
       bind(C, name="hipsolverSsyevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssyevd
    end function hipsolverSsyevd

    !---------------------------------------------
    ! hipsolverDsyevd
    !---------------------------------------------
    function hipsolverDsyevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo) &
       result(Dsyevd) &
       bind(C, name="hipsolverDsyevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsyevd
    end function hipsolverDsyevd

    !---------------------------------------------
    ! hipsolverCheevd
    !---------------------------------------------
    function hipsolverCheevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo) &
       result(Cheevd) &
       bind(C, name="hipsolverCheevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cheevd
    end function hipsolverCheevd

    !---------------------------------------------
    ! hipsolverZheevd
    !---------------------------------------------
    function hipsolverZheevd(handle, jobz, uplo, n, A, lda, D, work, lwork, devInfo) &
       result(Zheevd) &
       bind(C, name="hipsolverZheevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zheevd
    end function hipsolverZheevd

    !---------------------------------------------
    ! hipsolverSsyevdx_bufferSize
    !---------------------------------------------
    function hipsolverSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, &
                                         nev, W, lwork) &
       result(Ssyevdx_bufferSize) &
       bind(C, name="hipsolverSsyevdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssyevdx_bufferSize
    end function hipsolverSsyevdx_bufferSize

    !---------------------------------------------
    ! hipsolverDsyevdx_bufferSize
    !---------------------------------------------
    function hipsolverDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, &
                                         nev, W, lwork) &
       result(Dsyevdx_bufferSize) &
       bind(C, name="hipsolverDsyevdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsyevdx_bufferSize
    end function hipsolverDsyevdx_bufferSize

    !---------------------------------------------
    ! hipsolverCheevdx_bufferSize
    !---------------------------------------------
    function hipsolverCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, &
                                         nev, W, lwork) &
       result(Cheevdx_bufferSize) &
       bind(C, name="hipsolverCheevdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cheevdx_bufferSize
    end function hipsolverCheevdx_bufferSize

    !---------------------------------------------
    ! hipsolverZheevdx_bufferSize
    !---------------------------------------------
    function hipsolverZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, &
                                         nev, W, lwork) &
       result(Zheevdx_bufferSize) &
       bind(C, name="hipsolverZheevdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zheevdx_bufferSize
    end function hipsolverZheevdx_bufferSize

    !---------------------------------------------
    ! hipsolverSsyevdx
    !---------------------------------------------
    function hipsolverSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, &
                              lwork, devInfo) &
       result(Ssyevdx) &
       bind(C, name="hipsolverSsyevdx")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssyevdx
    end function hipsolverSsyevdx

    !---------------------------------------------
    ! hipsolverDsyevdx
    !---------------------------------------------
    function hipsolverDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, &
                              lwork, devInfo) &
       result(Dsyevdx) &
       bind(C, name="hipsolverDsyevdx")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsyevdx
    end function hipsolverDsyevdx

    !---------------------------------------------
    ! hipsolverCheevdx
    !---------------------------------------------
    function hipsolverCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, &
                              lwork, devInfo) &
       result(Cheevdx) &
       bind(C, name="hipsolverCheevdx")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cheevdx
    end function hipsolverCheevdx

    !---------------------------------------------
    ! hipsolverZheevdx
    !---------------------------------------------
    function hipsolverZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, &
                              lwork, devInfo) &
       result(Zheevdx) &
       bind(C, name="hipsolverZheevdx")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zheevdx
    end function hipsolverZheevdx

    !---------------------------------------------
    ! hipsolverSsyevj_bufferSize
    !---------------------------------------------
    function hipsolverSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
       result(Ssyevj_bufferSize) &
       bind(C, name="hipsolverSsyevj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssyevj_bufferSize
    end function hipsolverSsyevj_bufferSize

    !---------------------------------------------
    ! hipsolverDsyevj_bufferSize
    !---------------------------------------------
    function hipsolverDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
       result(Dsyevj_bufferSize) &
       bind(C, name="hipsolverDsyevj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsyevj_bufferSize
    end function hipsolverDsyevj_bufferSize

    !---------------------------------------------
    ! hipsolverCheevj_bufferSize
    !---------------------------------------------
    function hipsolverCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
       result(Cheevj_bufferSize) &
       bind(C, name="hipsolverCheevj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cheevj_bufferSize
    end function hipsolverCheevj_bufferSize

    !---------------------------------------------
    ! hipsolverZheevj_bufferSize
    !---------------------------------------------
    function hipsolverZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
       result(Zheevj_bufferSize) &
       bind(C, name="hipsolverZheevj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zheevj_bufferSize
    end function hipsolverZheevj_bufferSize

    !---------------------------------------------
    ! hipsolverSsyevj
    !---------------------------------------------
    function hipsolverSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params) &
       result(Ssyevj) &
       bind(C, name="hipsolverSsyevj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssyevj
    end function hipsolverSsyevj

    !---------------------------------------------
    ! hipsolverDsyevj
    !---------------------------------------------
    function hipsolverDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params) &
       result(Dsyevj) &
       bind(C, name="hipsolverDsyevj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsyevj
    end function hipsolverDsyevj

    !---------------------------------------------
    ! hipsolverCheevj
    !---------------------------------------------
    function hipsolverCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params) &
       result(Cheevj) &
       bind(C, name="hipsolverCheevj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Cheevj
    end function hipsolverCheevj

    !---------------------------------------------
    ! hipsolverZheevj
    !---------------------------------------------
    function hipsolverZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params) &
       result(Zheevj) &
       bind(C, name="hipsolverZheevj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zheevj
    end function hipsolverZheevj

    !---------------------------------------------
    ! hipsolverSsyevjBatched_bufferSize
    !---------------------------------------------
    function hipsolverSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, &
                                               batch_count) &
       result(SsyevjBatched_bufferSize) &
       bind(C, name="hipsolverSsyevjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SsyevjBatched_bufferSize
    end function hipsolverSsyevjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDsyevjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, &
                                               batch_count) &
       result(DsyevjBatched_bufferSize) &
       bind(C, name="hipsolverDsyevjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DsyevjBatched_bufferSize
    end function hipsolverDsyevjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverCheevjBatched_bufferSize
    !---------------------------------------------
    function hipsolverCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, &
                                               batch_count) &
       result(CheevjBatched_bufferSize) &
       bind(C, name="hipsolverCheevjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CheevjBatched_bufferSize
    end function hipsolverCheevjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverZheevjBatched_bufferSize
    !---------------------------------------------
    function hipsolverZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, &
                                               batch_count) &
       result(ZheevjBatched_bufferSize) &
       bind(C, name="hipsolverZheevjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZheevjBatched_bufferSize
    end function hipsolverZheevjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverSsyevjBatched
    !---------------------------------------------
    function hipsolverSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, &
                                    params, batch_count) &
       result(SsyevjBatched) &
       bind(C, name="hipsolverSsyevjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SsyevjBatched
    end function hipsolverSsyevjBatched

    !---------------------------------------------
    ! hipsolverDsyevjBatched
    !---------------------------------------------
    function hipsolverDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, &
                                    params, batch_count) &
       result(DsyevjBatched) &
       bind(C, name="hipsolverDsyevjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DsyevjBatched
    end function hipsolverDsyevjBatched

    !---------------------------------------------
    ! hipsolverCheevjBatched
    !---------------------------------------------
    function hipsolverCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, &
                                    params, batch_count) &
       result(CheevjBatched) &
       bind(C, name="hipsolverCheevjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: CheevjBatched
    end function hipsolverCheevjBatched

    !---------------------------------------------
    ! hipsolverZheevjBatched
    !---------------------------------------------
    function hipsolverZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, &
                                    params, batch_count) &
       result(ZheevjBatched) &
       bind(C, name="hipsolverZheevjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: ZheevjBatched
    end function hipsolverZheevjBatched

    !---------------------------------------------
    ! hipsolverSsygvd_bufferSize
    !---------------------------------------------
    function hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
       result(Ssygvd_bufferSize) &
       bind(C, name="hipsolverSsygvd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssygvd_bufferSize
    end function hipsolverSsygvd_bufferSize

    !---------------------------------------------
    ! hipsolverDsygvd_bufferSize
    !---------------------------------------------
    function hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
       result(Dsygvd_bufferSize) &
       bind(C, name="hipsolverDsygvd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsygvd_bufferSize
    end function hipsolverDsygvd_bufferSize

    !---------------------------------------------
    ! hipsolverChegvd_bufferSize
    !---------------------------------------------
    function hipsolverChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
       result(Chegvd_bufferSize) &
       bind(C, name="hipsolverChegvd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Chegvd_bufferSize
    end function hipsolverChegvd_bufferSize

    !---------------------------------------------
    ! hipsolverZhegvd_bufferSize
    !---------------------------------------------
    function hipsolverZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
       result(Zhegvd_bufferSize) &
       bind(C, name="hipsolverZhegvd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zhegvd_bufferSize
    end function hipsolverZhegvd_bufferSize

    !---------------------------------------------
    ! hipsolverSsygvd
    !---------------------------------------------
    function hipsolverSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                             devInfo) &
       result(Ssygvd) &
       bind(C, name="hipsolverSsygvd")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssygvd
    end function hipsolverSsygvd

    !---------------------------------------------
    ! hipsolverDsygvd
    !---------------------------------------------
    function hipsolverDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                             devInfo) &
       result(Dsygvd) &
       bind(C, name="hipsolverDsygvd")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsygvd
    end function hipsolverDsygvd

    !---------------------------------------------
    ! hipsolverChegvd
    !---------------------------------------------
    function hipsolverChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                             devInfo) &
       result(Chegvd) &
       bind(C, name="hipsolverChegvd")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Chegvd
    end function hipsolverChegvd

    !---------------------------------------------
    ! hipsolverZhegvd
    !---------------------------------------------
    function hipsolverZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                             devInfo) &
       result(Zhegvd) &
       bind(C, name="hipsolverZhegvd")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zhegvd
    end function hipsolverZhegvd

    !---------------------------------------------
    ! hipsolverSsygvdx_bufferSize
    !---------------------------------------------
    function hipsolverSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, &
                                         vu, il, iu, nev, W, lwork) &
       result(Ssygvdx_bufferSize) &
       bind(C, name="hipsolverSsygvdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssygvdx_bufferSize
    end function hipsolverSsygvdx_bufferSize

    !---------------------------------------------
    ! hipsolverDsygvdx_bufferSize
    !---------------------------------------------
    function hipsolverDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, &
                                         vu, il, iu, nev, W, lwork) &
       result(Dsygvdx_bufferSize) &
       bind(C, name="hipsolverDsygvdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsygvdx_bufferSize
    end function hipsolverDsygvdx_bufferSize

    !---------------------------------------------
    ! hipsolverChegvdx_bufferSize
    !---------------------------------------------
    function hipsolverChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, &
                                         vu, il, iu, nev, W, lwork) &
       result(Chegvdx_bufferSize) &
       bind(C, name="hipsolverChegvdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Chegvdx_bufferSize
    end function hipsolverChegvdx_bufferSize

    !---------------------------------------------
    ! hipsolverZhegvdx_bufferSize
    !---------------------------------------------
    function hipsolverZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, &
                                         vu, il, iu, nev, W, lwork) &
       result(Zhegvdx_bufferSize) &
       bind(C, name="hipsolverZhegvdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zhegvdx_bufferSize
    end function hipsolverZhegvdx_bufferSize

    !---------------------------------------------
    ! hipsolverSsygvdx
    !---------------------------------------------
    function hipsolverSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, &
                              nev, W, work, lwork, devInfo) &
       result(Ssygvdx) &
       bind(C, name="hipsolverSsygvdx")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssygvdx
    end function hipsolverSsygvdx

    !---------------------------------------------
    ! hipsolverDsygvdx
    !---------------------------------------------
    function hipsolverDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, &
                              nev, W, work, lwork, devInfo) &
       result(Dsygvdx) &
       bind(C, name="hipsolverDsygvdx")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsygvdx
    end function hipsolverDsygvdx

    !---------------------------------------------
    ! hipsolverChegvdx
    !---------------------------------------------
    function hipsolverChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, &
                              nev, W, work, lwork, devInfo) &
       result(Chegvdx) &
       bind(C, name="hipsolverChegvdx")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Chegvdx
    end function hipsolverChegvdx

    !---------------------------------------------
    ! hipsolverZhegvdx
    !---------------------------------------------
    function hipsolverZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, &
                              nev, W, work, lwork, devInfo) &
       result(Zhegvdx) &
       bind(C, name="hipsolverZhegvdx")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zhegvdx
    end function hipsolverZhegvdx

    !---------------------------------------------
    ! hipsolverSsygvj_bufferSize
    !---------------------------------------------
    function hipsolverSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, &
                                        params) &
       result(Ssygvj_bufferSize) &
       bind(C, name="hipsolverSsygvj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssygvj_bufferSize
    end function hipsolverSsygvj_bufferSize

    !---------------------------------------------
    ! hipsolverDsygvj_bufferSize
    !---------------------------------------------
    function hipsolverDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, &
                                        params) &
       result(Dsygvj_bufferSize) &
       bind(C, name="hipsolverDsygvj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsygvj_bufferSize
    end function hipsolverDsygvj_bufferSize

    !---------------------------------------------
    ! hipsolverChegvj_bufferSize
    !---------------------------------------------
    function hipsolverChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, &
                                        params) &
       result(Chegvj_bufferSize) &
       bind(C, name="hipsolverChegvj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Chegvj_bufferSize
    end function hipsolverChegvj_bufferSize

    !---------------------------------------------
    ! hipsolverZhegvj_bufferSize
    !---------------------------------------------
    function hipsolverZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, &
                                        params) &
       result(Zhegvj_bufferSize) &
       bind(C, name="hipsolverZhegvj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zhegvj_bufferSize
    end function hipsolverZhegvj_bufferSize

    !---------------------------------------------
    ! hipsolverSsygvj
    !---------------------------------------------
    function hipsolverSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                             devInfo, params) &
       result(Ssygvj) &
       bind(C, name="hipsolverSsygvj")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssygvj
    end function hipsolverSsygvj

    !---------------------------------------------
    ! hipsolverDsygvj
    !---------------------------------------------
    function hipsolverDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                             devInfo, params) &
       result(Dsygvj) &
       bind(C, name="hipsolverDsygvj")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsygvj
    end function hipsolverDsygvj

    !---------------------------------------------
    ! hipsolverChegvj
    !---------------------------------------------
    function hipsolverChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                             devInfo, params) &
       result(Chegvj) &
       bind(C, name="hipsolverChegvj")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Chegvj
    end function hipsolverChegvj

    !---------------------------------------------
    ! hipsolverZhegvj
    !---------------------------------------------
    function hipsolverZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                             devInfo, params) &
       result(Zhegvj) &
       bind(C, name="hipsolverZhegvj")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zhegvj
    end function hipsolverZhegvj

    !---------------------------------------------
    ! hipsolverSsytrd_bufferSize
    !---------------------------------------------
    function hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
       result(Ssytrd_bufferSize) &
       bind(C, name="hipsolverSsytrd_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssytrd_bufferSize
    end function hipsolverSsytrd_bufferSize

    !---------------------------------------------
    ! hipsolverDsytrd_bufferSize
    !---------------------------------------------
    function hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
       result(Dsytrd_bufferSize) &
       bind(C, name="hipsolverDsytrd_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsytrd_bufferSize
    end function hipsolverDsytrd_bufferSize

    !---------------------------------------------
    ! hipsolverChetrd_bufferSize
    !---------------------------------------------
    function hipsolverChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
       result(Chetrd_bufferSize) &
       bind(C, name="hipsolverChetrd_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Chetrd_bufferSize
    end function hipsolverChetrd_bufferSize

    !---------------------------------------------
    ! hipsolverZhetrd_bufferSize
    !---------------------------------------------
    function hipsolverZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
       result(Zhetrd_bufferSize) &
       bind(C, name="hipsolverZhetrd_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zhetrd_bufferSize
    end function hipsolverZhetrd_bufferSize

    !---------------------------------------------
    ! hipsolverSsytrd
    !---------------------------------------------
    function hipsolverSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo) &
       result(Ssytrd) &
       bind(C, name="hipsolverSsytrd")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssytrd
    end function hipsolverSsytrd

    !---------------------------------------------
    ! hipsolverDsytrd
    !---------------------------------------------
    function hipsolverDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo) &
       result(Dsytrd) &
       bind(C, name="hipsolverDsytrd")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsytrd
    end function hipsolverDsytrd

    !---------------------------------------------
    ! hipsolverChetrd
    !---------------------------------------------
    function hipsolverChetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo) &
       result(Chetrd) &
       bind(C, name="hipsolverChetrd")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Chetrd
    end function hipsolverChetrd

    !---------------------------------------------
    ! hipsolverZhetrd
    !---------------------------------------------
    function hipsolverZhetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo) &
       result(Zhetrd) &
       bind(C, name="hipsolverZhetrd")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zhetrd
    end function hipsolverZhetrd

    !---------------------------------------------
    ! hipsolverSsytrf_bufferSize
    !---------------------------------------------
    function hipsolverSsytrf_bufferSize(handle, n, A, lda, lwork) &
       result(Ssytrf_bufferSize) &
       bind(C, name="hipsolverSsytrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssytrf_bufferSize
    end function hipsolverSsytrf_bufferSize

    !---------------------------------------------
    ! hipsolverDsytrf_bufferSize
    !---------------------------------------------
    function hipsolverDsytrf_bufferSize(handle, n, A, lda, lwork) &
       result(Dsytrf_bufferSize) &
       bind(C, name="hipsolverDsytrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsytrf_bufferSize
    end function hipsolverDsytrf_bufferSize

    !---------------------------------------------
    ! hipsolverCsytrf_bufferSize
    !---------------------------------------------
    function hipsolverCsytrf_bufferSize(handle, n, A, lda, lwork) &
       result(Csytrf_bufferSize) &
       bind(C, name="hipsolverCsytrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Csytrf_bufferSize
    end function hipsolverCsytrf_bufferSize

    !---------------------------------------------
    ! hipsolverZsytrf_bufferSize
    !---------------------------------------------
    function hipsolverZsytrf_bufferSize(handle, n, A, lda, lwork) &
       result(Zsytrf_bufferSize) &
       bind(C, name="hipsolverZsytrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zsytrf_bufferSize
    end function hipsolverZsytrf_bufferSize

    !---------------------------------------------
    ! hipsolverSsytrf
    !---------------------------------------------
    function hipsolverSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo) &
       result(Ssytrf) &
       bind(C, name="hipsolverSsytrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Ssytrf
    end function hipsolverSsytrf

    !---------------------------------------------
    ! hipsolverDsytrf
    !---------------------------------------------
    function hipsolverDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo) &
       result(Dsytrf) &
       bind(C, name="hipsolverDsytrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Dsytrf
    end function hipsolverDsytrf

    !---------------------------------------------
    ! hipsolverCsytrf
    !---------------------------------------------
    function hipsolverCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo) &
       result(Csytrf) &
       bind(C, name="hipsolverCsytrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Csytrf
    end function hipsolverCsytrf

    !---------------------------------------------
    ! hipsolverZsytrf
    !---------------------------------------------
    function hipsolverZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo) &
       result(Zsytrf) &
       bind(C, name="hipsolverZsytrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: Zsytrf
    end function hipsolverZsytrf

    !---------------------------------------------
    ! hipsolverDnCreate
    !---------------------------------------------
    function hipsolverDnCreate(handle) &
       result(DnCreate) &
       bind(C, name="hipsolverDnCreate")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCreate
    end function hipsolverDnCreate

    !---------------------------------------------
    ! hipsolverDnDestroy
    !---------------------------------------------
    function hipsolverDnDestroy(handle) &
       result(DnDestroy) &
       bind(C, name="hipsolverDnDestroy")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDestroy
    end function hipsolverDnDestroy

    !---------------------------------------------
    ! hipsolverDnSetStream
    !---------------------------------------------
    function hipsolverDnSetStream(handle, streamId) &
       result(DnSetStream) &
       bind(C, name="hipsolverDnSetStream")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: streamId
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSetStream
    end function hipsolverDnSetStream

    !---------------------------------------------
    ! hipsolverDnGetStream
    !---------------------------------------------
    function hipsolverDnGetStream(handle, streamId) &
       result(DnGetStream) &
       bind(C, name="hipsolverDnGetStream")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr) :: streamId
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnGetStream
    end function hipsolverDnGetStream

    !---------------------------------------------
    ! hipsolverDnSetDeterministicMode
    !---------------------------------------------
    function hipsolverDnSetDeterministicMode(handle, mode) &
       result(DnSetDeterministicMode) &
       bind(C, name="hipsolverDnSetDeterministicMode")
       import :: c_ptr, HIPSOLVER_DETERMINISTIC_RESULTS, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_DETERMINISTIC_RESULTS)), value :: mode
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSetDeterministicMode
    end function hipsolverDnSetDeterministicMode

    !---------------------------------------------
    ! hipsolverDnGetDeterministicMode
    !---------------------------------------------
    function hipsolverDnGetDeterministicMode(handle, mode) &
       result(DnGetDeterministicMode) &
       bind(C, name="hipsolverDnGetDeterministicMode")
       import :: c_ptr, HIPSOLVER_DETERMINISTIC_RESULTS, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_DETERMINISTIC_RESULTS)) :: mode
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnGetDeterministicMode
    end function hipsolverDnGetDeterministicMode

    !---------------------------------------------
    ! hipsolverDnCreateGesvdjInfo
    !---------------------------------------------
    function hipsolverDnCreateGesvdjInfo(myInfo) &
       result(DnCreateGesvdjInfo) &
       bind(C, name="hipsolverDnCreateGesvdjInfo")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCreateGesvdjInfo
    end function hipsolverDnCreateGesvdjInfo

    !---------------------------------------------
    ! hipsolverDnDestroyGesvdjInfo
    !---------------------------------------------
    function hipsolverDnDestroyGesvdjInfo(myInfo) &
       result(DnDestroyGesvdjInfo) &
       bind(C, name="hipsolverDnDestroyGesvdjInfo")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDestroyGesvdjInfo
    end function hipsolverDnDestroyGesvdjInfo

    !---------------------------------------------
    ! hipsolverDnXgesvdjSetMaxSweeps
    !---------------------------------------------
    function hipsolverDnXgesvdjSetMaxSweeps(myInfo, max_sweeps) &
       result(DnXgesvdjSetMaxSweeps) &
       bind(C, name="hipsolverDnXgesvdjSetMaxSweeps")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(c_int), value :: max_sweeps
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgesvdjSetMaxSweeps
    end function hipsolverDnXgesvdjSetMaxSweeps

    !---------------------------------------------
    ! hipsolverDnXgesvdjSetSortEig
    !---------------------------------------------
    function hipsolverDnXgesvdjSetSortEig(myInfo, sort_eig) &
       result(DnXgesvdjSetSortEig) &
       bind(C, name="hipsolverDnXgesvdjSetSortEig")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(c_int), value :: sort_eig
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgesvdjSetSortEig
    end function hipsolverDnXgesvdjSetSortEig

    !---------------------------------------------
    ! hipsolverDnXgesvdjSetTolerance
    !---------------------------------------------
    function hipsolverDnXgesvdjSetTolerance(myInfo, tolerance) &
       result(DnXgesvdjSetTolerance) &
       bind(C, name="hipsolverDnXgesvdjSetTolerance")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       real(c_double), value :: tolerance
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgesvdjSetTolerance
    end function hipsolverDnXgesvdjSetTolerance

    !---------------------------------------------
    ! hipsolverDnXgesvdjGetResidual
    !---------------------------------------------
    function hipsolverDnXgesvdjGetResidual(handle, myInfo, residual) &
       result(DnXgesvdjGetResidual) &
       bind(C, name="hipsolverDnXgesvdjGetResidual")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: myInfo
       real(c_double) :: residual
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgesvdjGetResidual
    end function hipsolverDnXgesvdjGetResidual

    !---------------------------------------------
    ! hipsolverDnXgesvdjGetSweeps
    !---------------------------------------------
    function hipsolverDnXgesvdjGetSweeps(handle, myInfo, executed_sweeps) &
       result(DnXgesvdjGetSweeps) &
       bind(C, name="hipsolverDnXgesvdjGetSweeps")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: myInfo
       integer(c_int) :: executed_sweeps
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgesvdjGetSweeps
    end function hipsolverDnXgesvdjGetSweeps

    !---------------------------------------------
    ! hipsolverDnCreateSyevjInfo
    !---------------------------------------------
    function hipsolverDnCreateSyevjInfo(myInfo) &
       result(DnCreateSyevjInfo) &
       bind(C, name="hipsolverDnCreateSyevjInfo")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCreateSyevjInfo
    end function hipsolverDnCreateSyevjInfo

    !---------------------------------------------
    ! hipsolverDnDestroySyevjInfo
    !---------------------------------------------
    function hipsolverDnDestroySyevjInfo(myInfo) &
       result(DnDestroySyevjInfo) &
       bind(C, name="hipsolverDnDestroySyevjInfo")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDestroySyevjInfo
    end function hipsolverDnDestroySyevjInfo

    !---------------------------------------------
    ! hipsolverDnXsyevjSetMaxSweeps
    !---------------------------------------------
    function hipsolverDnXsyevjSetMaxSweeps(myInfo, max_sweeps) &
       result(DnXsyevjSetMaxSweeps) &
       bind(C, name="hipsolverDnXsyevjSetMaxSweeps")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(c_int), value :: max_sweeps
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevjSetMaxSweeps
    end function hipsolverDnXsyevjSetMaxSweeps

    !---------------------------------------------
    ! hipsolverDnXsyevjSetSortEig
    !---------------------------------------------
    function hipsolverDnXsyevjSetSortEig(myInfo, sort_eig) &
       result(DnXsyevjSetSortEig) &
       bind(C, name="hipsolverDnXsyevjSetSortEig")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       integer(c_int), value :: sort_eig
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevjSetSortEig
    end function hipsolverDnXsyevjSetSortEig

    !---------------------------------------------
    ! hipsolverDnXsyevjSetTolerance
    !---------------------------------------------
    function hipsolverDnXsyevjSetTolerance(myInfo, tolerance) &
       result(DnXsyevjSetTolerance) &
       bind(C, name="hipsolverDnXsyevjSetTolerance")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: myInfo
       real(c_double), value :: tolerance
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevjSetTolerance
    end function hipsolverDnXsyevjSetTolerance

    !---------------------------------------------
    ! hipsolverDnXsyevjGetResidual
    !---------------------------------------------
    function hipsolverDnXsyevjGetResidual(handle, myInfo, residual) &
       result(DnXsyevjGetResidual) &
       bind(C, name="hipsolverDnXsyevjGetResidual")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: myInfo
       real(c_double) :: residual
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevjGetResidual
    end function hipsolverDnXsyevjGetResidual

    !---------------------------------------------
    ! hipsolverDnXsyevjGetSweeps
    !---------------------------------------------
    function hipsolverDnXsyevjGetSweeps(handle, myInfo, executed_sweeps) &
       result(DnXsyevjGetSweeps) &
       bind(C, name="hipsolverDnXsyevjGetSweeps")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: myInfo
       integer(c_int) :: executed_sweeps
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevjGetSweeps
    end function hipsolverDnXsyevjGetSweeps

    !---------------------------------------------
    ! hipsolverDnSorgbr_bufferSize
    !---------------------------------------------
    function hipsolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
       result(DnSorgbr_bufferSize) &
       bind(C, name="hipsolverDnSorgbr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSorgbr_bufferSize
    end function hipsolverDnSorgbr_bufferSize

    !---------------------------------------------
    ! hipsolverDnDorgbr_bufferSize
    !---------------------------------------------
    function hipsolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
       result(DnDorgbr_bufferSize) &
       bind(C, name="hipsolverDnDorgbr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDorgbr_bufferSize
    end function hipsolverDnDorgbr_bufferSize

    !---------------------------------------------
    ! hipsolverDnCungbr_bufferSize
    !---------------------------------------------
    function hipsolverDnCungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
       result(DnCungbr_bufferSize) &
       bind(C, name="hipsolverDnCungbr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCungbr_bufferSize
    end function hipsolverDnCungbr_bufferSize

    !---------------------------------------------
    ! hipsolverDnZungbr_bufferSize
    !---------------------------------------------
    function hipsolverDnZungbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork) &
       result(DnZungbr_bufferSize) &
       bind(C, name="hipsolverDnZungbr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZungbr_bufferSize
    end function hipsolverDnZungbr_bufferSize

    !---------------------------------------------
    ! hipsolverDnSorgbr
    !---------------------------------------------
    function hipsolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(DnSorgbr) &
       bind(C, name="hipsolverDnSorgbr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSorgbr
    end function hipsolverDnSorgbr

    !---------------------------------------------
    ! hipsolverDnDorgbr
    !---------------------------------------------
    function hipsolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(DnDorgbr) &
       bind(C, name="hipsolverDnDorgbr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDorgbr
    end function hipsolverDnDorgbr

    !---------------------------------------------
    ! hipsolverDnCungbr
    !---------------------------------------------
    function hipsolverDnCungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(DnCungbr) &
       bind(C, name="hipsolverDnCungbr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCungbr
    end function hipsolverDnCungbr

    !---------------------------------------------
    ! hipsolverDnZungbr
    !---------------------------------------------
    function hipsolverDnZungbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(DnZungbr) &
       bind(C, name="hipsolverDnZungbr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZungbr
    end function hipsolverDnZungbr

    !---------------------------------------------
    ! hipsolverDnSorgqr_bufferSize
    !---------------------------------------------
    function hipsolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
       result(DnSorgqr_bufferSize) &
       bind(C, name="hipsolverDnSorgqr_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSorgqr_bufferSize
    end function hipsolverDnSorgqr_bufferSize

    !---------------------------------------------
    ! hipsolverDnDorgqr_bufferSize
    !---------------------------------------------
    function hipsolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
       result(DnDorgqr_bufferSize) &
       bind(C, name="hipsolverDnDorgqr_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDorgqr_bufferSize
    end function hipsolverDnDorgqr_bufferSize

    !---------------------------------------------
    ! hipsolverDnCungqr_bufferSize
    !---------------------------------------------
    function hipsolverDnCungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
       result(DnCungqr_bufferSize) &
       bind(C, name="hipsolverDnCungqr_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCungqr_bufferSize
    end function hipsolverDnCungqr_bufferSize

    !---------------------------------------------
    ! hipsolverDnZungqr_bufferSize
    !---------------------------------------------
    function hipsolverDnZungqr_bufferSize(handle, m, n, k, A, lda, tau, lwork) &
       result(DnZungqr_bufferSize) &
       bind(C, name="hipsolverDnZungqr_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZungqr_bufferSize
    end function hipsolverDnZungqr_bufferSize

    !---------------------------------------------
    ! hipsolverDnSorgqr
    !---------------------------------------------
    function hipsolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(DnSorgqr) &
       bind(C, name="hipsolverDnSorgqr")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSorgqr
    end function hipsolverDnSorgqr

    !---------------------------------------------
    ! hipsolverDnDorgqr
    !---------------------------------------------
    function hipsolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(DnDorgqr) &
       bind(C, name="hipsolverDnDorgqr")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDorgqr
    end function hipsolverDnDorgqr

    !---------------------------------------------
    ! hipsolverDnCungqr
    !---------------------------------------------
    function hipsolverDnCungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(DnCungqr) &
       bind(C, name="hipsolverDnCungqr")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCungqr
    end function hipsolverDnCungqr

    !---------------------------------------------
    ! hipsolverDnZungqr
    !---------------------------------------------
    function hipsolverDnZungqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo) &
       result(DnZungqr) &
       bind(C, name="hipsolverDnZungqr")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZungqr
    end function hipsolverDnZungqr

    !---------------------------------------------
    ! hipsolverDnSorgtr_bufferSize
    !---------------------------------------------
    function hipsolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
       result(DnSorgtr_bufferSize) &
       bind(C, name="hipsolverDnSorgtr_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSorgtr_bufferSize
    end function hipsolverDnSorgtr_bufferSize

    !---------------------------------------------
    ! hipsolverDnDorgtr_bufferSize
    !---------------------------------------------
    function hipsolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
       result(DnDorgtr_bufferSize) &
       bind(C, name="hipsolverDnDorgtr_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDorgtr_bufferSize
    end function hipsolverDnDorgtr_bufferSize

    !---------------------------------------------
    ! hipsolverDnCungtr_bufferSize
    !---------------------------------------------
    function hipsolverDnCungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
       result(DnCungtr_bufferSize) &
       bind(C, name="hipsolverDnCungtr_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCungtr_bufferSize
    end function hipsolverDnCungtr_bufferSize

    !---------------------------------------------
    ! hipsolverDnZungtr_bufferSize
    !---------------------------------------------
    function hipsolverDnZungtr_bufferSize(handle, uplo, n, A, lda, tau, lwork) &
       result(DnZungtr_bufferSize) &
       bind(C, name="hipsolverDnZungtr_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZungtr_bufferSize
    end function hipsolverDnZungtr_bufferSize

    !---------------------------------------------
    ! hipsolverDnSorgtr
    !---------------------------------------------
    function hipsolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo) &
       result(DnSorgtr) &
       bind(C, name="hipsolverDnSorgtr")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSorgtr
    end function hipsolverDnSorgtr

    !---------------------------------------------
    ! hipsolverDnDorgtr
    !---------------------------------------------
    function hipsolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo) &
       result(DnDorgtr) &
       bind(C, name="hipsolverDnDorgtr")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDorgtr
    end function hipsolverDnDorgtr

    !---------------------------------------------
    ! hipsolverDnCungtr
    !---------------------------------------------
    function hipsolverDnCungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo) &
       result(DnCungtr) &
       bind(C, name="hipsolverDnCungtr")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCungtr
    end function hipsolverDnCungtr

    !---------------------------------------------
    ! hipsolverDnZungtr
    !---------------------------------------------
    function hipsolverDnZungtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo) &
       result(DnZungtr) &
       bind(C, name="hipsolverDnZungtr")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZungtr
    end function hipsolverDnZungtr

    !---------------------------------------------
    ! hipsolverDnSormqr_bufferSize
    !---------------------------------------------
    function hipsolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &
                                          lwork) &
       result(DnSormqr_bufferSize) &
       bind(C, name="hipsolverDnSormqr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSormqr_bufferSize
    end function hipsolverDnSormqr_bufferSize

    !---------------------------------------------
    ! hipsolverDnDormqr_bufferSize
    !---------------------------------------------
    function hipsolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &
                                          lwork) &
       result(DnDormqr_bufferSize) &
       bind(C, name="hipsolverDnDormqr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDormqr_bufferSize
    end function hipsolverDnDormqr_bufferSize

    !---------------------------------------------
    ! hipsolverDnCunmqr_bufferSize
    !---------------------------------------------
    function hipsolverDnCunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &
                                          lwork) &
       result(DnCunmqr_bufferSize) &
       bind(C, name="hipsolverDnCunmqr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCunmqr_bufferSize
    end function hipsolverDnCunmqr_bufferSize

    !---------------------------------------------
    ! hipsolverDnZunmqr_bufferSize
    !---------------------------------------------
    function hipsolverDnZunmqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, &
                                          lwork) &
       result(DnZunmqr_bufferSize) &
       bind(C, name="hipsolverDnZunmqr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZunmqr_bufferSize
    end function hipsolverDnZunmqr_bufferSize

    !---------------------------------------------
    ! hipsolverDnSormqr
    !---------------------------------------------
    function hipsolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &
                               devInfo) &
       result(DnSormqr) &
       bind(C, name="hipsolverDnSormqr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSormqr
    end function hipsolverDnSormqr

    !---------------------------------------------
    ! hipsolverDnDormqr
    !---------------------------------------------
    function hipsolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &
                               devInfo) &
       result(DnDormqr) &
       bind(C, name="hipsolverDnDormqr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDormqr
    end function hipsolverDnDormqr

    !---------------------------------------------
    ! hipsolverDnCunmqr
    !---------------------------------------------
    function hipsolverDnCunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &
                               devInfo) &
       result(DnCunmqr) &
       bind(C, name="hipsolverDnCunmqr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCunmqr
    end function hipsolverDnCunmqr

    !---------------------------------------------
    ! hipsolverDnZunmqr
    !---------------------------------------------
    function hipsolverDnZunmqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, &
                               devInfo) &
       result(DnZunmqr) &
       bind(C, name="hipsolverDnZunmqr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: k
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZunmqr
    end function hipsolverDnZunmqr

    !---------------------------------------------
    ! hipsolverDnSormtr_bufferSize
    !---------------------------------------------
    function hipsolverDnSormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &
                                          lwork) &
       result(DnSormtr_bufferSize) &
       bind(C, name="hipsolverDnSormtr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSormtr_bufferSize
    end function hipsolverDnSormtr_bufferSize

    !---------------------------------------------
    ! hipsolverDnDormtr_bufferSize
    !---------------------------------------------
    function hipsolverDnDormtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &
                                          lwork) &
       result(DnDormtr_bufferSize) &
       bind(C, name="hipsolverDnDormtr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDormtr_bufferSize
    end function hipsolverDnDormtr_bufferSize

    !---------------------------------------------
    ! hipsolverDnCunmtr_bufferSize
    !---------------------------------------------
    function hipsolverDnCunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &
                                          lwork) &
       result(DnCunmtr_bufferSize) &
       bind(C, name="hipsolverDnCunmtr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCunmtr_bufferSize
    end function hipsolverDnCunmtr_bufferSize

    !---------------------------------------------
    ! hipsolverDnZunmtr_bufferSize
    !---------------------------------------------
    function hipsolverDnZunmtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, &
                                          lwork) &
       result(DnZunmtr_bufferSize) &
       bind(C, name="hipsolverDnZunmtr_bufferSize")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZunmtr_bufferSize
    end function hipsolverDnZunmtr_bufferSize

    !---------------------------------------------
    ! hipsolverDnSormtr
    !---------------------------------------------
    function hipsolverDnSormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, &
                               devInfo) &
       result(DnSormtr) &
       bind(C, name="hipsolverDnSormtr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSormtr
    end function hipsolverDnSormtr

    !---------------------------------------------
    ! hipsolverDnDormtr
    !---------------------------------------------
    function hipsolverDnDormtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, &
                               devInfo) &
       result(DnDormtr) &
       bind(C, name="hipsolverDnDormtr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDormtr
    end function hipsolverDnDormtr

    !---------------------------------------------
    ! hipsolverDnCunmtr
    !---------------------------------------------
    function hipsolverDnCunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, &
                               devInfo) &
       result(DnCunmtr) &
       bind(C, name="hipsolverDnCunmtr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCunmtr
    end function hipsolverDnCunmtr

    !---------------------------------------------
    ! hipsolverDnZunmtr
    !---------------------------------------------
    function hipsolverDnZunmtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, &
                               devInfo) &
       result(DnZunmtr) &
       bind(C, name="hipsolverDnZunmtr")
       import :: c_ptr, HIPSOLVER_SIDE_LEFT, HIPSOLVER_FILL_MODE_UPPER, HIPSOLVER_OP_N, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_SIDE_LEFT)), value :: side
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: C
       integer(c_int), value :: ldc
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZunmtr
    end function hipsolverDnZunmtr

    !---------------------------------------------
    ! hipsolverDnSgebrd_bufferSize
    !---------------------------------------------
    function hipsolverDnSgebrd_bufferSize(handle, m, n, lwork) &
       result(DnSgebrd_bufferSize) &
       bind(C, name="hipsolverDnSgebrd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgebrd_bufferSize
    end function hipsolverDnSgebrd_bufferSize

    !---------------------------------------------
    ! hipsolverDnDgebrd_bufferSize
    !---------------------------------------------
    function hipsolverDnDgebrd_bufferSize(handle, m, n, lwork) &
       result(DnDgebrd_bufferSize) &
       bind(C, name="hipsolverDnDgebrd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgebrd_bufferSize
    end function hipsolverDnDgebrd_bufferSize

    !---------------------------------------------
    ! hipsolverDnCgebrd_bufferSize
    !---------------------------------------------
    function hipsolverDnCgebrd_bufferSize(handle, m, n, lwork) &
       result(DnCgebrd_bufferSize) &
       bind(C, name="hipsolverDnCgebrd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgebrd_bufferSize
    end function hipsolverDnCgebrd_bufferSize

    !---------------------------------------------
    ! hipsolverDnZgebrd_bufferSize
    !---------------------------------------------
    function hipsolverDnZgebrd_bufferSize(handle, m, n, lwork) &
       result(DnZgebrd_bufferSize) &
       bind(C, name="hipsolverDnZgebrd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgebrd_bufferSize
    end function hipsolverDnZgebrd_bufferSize

    !---------------------------------------------
    ! hipsolverDnSgebrd
    !---------------------------------------------
    function hipsolverDnSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo) &
       result(DnSgebrd) &
       bind(C, name="hipsolverDnSgebrd")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgebrd
    end function hipsolverDnSgebrd

    !---------------------------------------------
    ! hipsolverDnDgebrd
    !---------------------------------------------
    function hipsolverDnDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo) &
       result(DnDgebrd) &
       bind(C, name="hipsolverDnDgebrd")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgebrd
    end function hipsolverDnDgebrd

    !---------------------------------------------
    ! hipsolverDnCgebrd
    !---------------------------------------------
    function hipsolverDnCgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo) &
       result(DnCgebrd) &
       bind(C, name="hipsolverDnCgebrd")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgebrd
    end function hipsolverDnCgebrd

    !---------------------------------------------
    ! hipsolverDnZgebrd
    !---------------------------------------------
    function hipsolverDnZgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, devInfo) &
       result(DnZgebrd) &
       bind(C, name="hipsolverDnZgebrd")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tauq
       type(c_ptr), value :: taup
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgebrd
    end function hipsolverDnZgebrd

    !---------------------------------------------
    ! hipsolverDnSSgels_bufferSize
    !---------------------------------------------
    function hipsolverDnSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork) &
       result(DnSSgels_bufferSize) &
       bind(C, name="hipsolverDnSSgels_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSSgels_bufferSize
    end function hipsolverDnSSgels_bufferSize

    !---------------------------------------------
    ! hipsolverDnDDgels_bufferSize
    !---------------------------------------------
    function hipsolverDnDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork) &
       result(DnDDgels_bufferSize) &
       bind(C, name="hipsolverDnDDgels_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDDgels_bufferSize
    end function hipsolverDnDDgels_bufferSize

    !---------------------------------------------
    ! hipsolverDnCCgels_bufferSize
    !---------------------------------------------
    function hipsolverDnCCgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork) &
       result(DnCCgels_bufferSize) &
       bind(C, name="hipsolverDnCCgels_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCCgels_bufferSize
    end function hipsolverDnCCgels_bufferSize

    !---------------------------------------------
    ! hipsolverDnZZgels_bufferSize
    !---------------------------------------------
    function hipsolverDnZZgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork) &
       result(DnZZgels_bufferSize) &
       bind(C, name="hipsolverDnZZgels_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZZgels_bufferSize
    end function hipsolverDnZZgels_bufferSize

    !---------------------------------------------
    ! hipsolverDnSSgels
    !---------------------------------------------
    function hipsolverDnSSgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, &
                               devInfo) &
       result(DnSSgels) &
       bind(C, name="hipsolverDnSSgels")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSSgels
    end function hipsolverDnSSgels

    !---------------------------------------------
    ! hipsolverDnDDgels
    !---------------------------------------------
    function hipsolverDnDDgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, &
                               devInfo) &
       result(DnDDgels) &
       bind(C, name="hipsolverDnDDgels")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDDgels
    end function hipsolverDnDDgels

    !---------------------------------------------
    ! hipsolverDnCCgels
    !---------------------------------------------
    function hipsolverDnCCgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, &
                               devInfo) &
       result(DnCCgels) &
       bind(C, name="hipsolverDnCCgels")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCCgels
    end function hipsolverDnCCgels

    !---------------------------------------------
    ! hipsolverDnZZgels
    !---------------------------------------------
    function hipsolverDnZZgels(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, &
                               devInfo) &
       result(DnZZgels) &
       bind(C, name="hipsolverDnZZgels")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZZgels
    end function hipsolverDnZZgels

    !---------------------------------------------
    ! hipsolverDnSgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(DnSgeqrf_bufferSize) &
       bind(C, name="hipsolverDnSgeqrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgeqrf_bufferSize
    end function hipsolverDnSgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnDgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(DnDgeqrf_bufferSize) &
       bind(C, name="hipsolverDnDgeqrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgeqrf_bufferSize
    end function hipsolverDnDgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnCgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverDnCgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(DnCgeqrf_bufferSize) &
       bind(C, name="hipsolverDnCgeqrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgeqrf_bufferSize
    end function hipsolverDnCgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnZgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverDnZgeqrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(DnZgeqrf_bufferSize) &
       bind(C, name="hipsolverDnZgeqrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgeqrf_bufferSize
    end function hipsolverDnZgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnSgeqrf
    !---------------------------------------------
    function hipsolverDnSgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo) &
       result(DnSgeqrf) &
       bind(C, name="hipsolverDnSgeqrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgeqrf
    end function hipsolverDnSgeqrf

    !---------------------------------------------
    ! hipsolverDnDgeqrf
    !---------------------------------------------
    function hipsolverDnDgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo) &
       result(DnDgeqrf) &
       bind(C, name="hipsolverDnDgeqrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgeqrf
    end function hipsolverDnDgeqrf

    !---------------------------------------------
    ! hipsolverDnCgeqrf
    !---------------------------------------------
    function hipsolverDnCgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo) &
       result(DnCgeqrf) &
       bind(C, name="hipsolverDnCgeqrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgeqrf
    end function hipsolverDnCgeqrf

    !---------------------------------------------
    ! hipsolverDnZgeqrf
    !---------------------------------------------
    function hipsolverDnZgeqrf(handle, m, n, A, lda, tau, work, lwork, devInfo) &
       result(DnZgeqrf) &
       bind(C, name="hipsolverDnZgeqrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgeqrf
    end function hipsolverDnZgeqrf

    !---------------------------------------------
    ! hipsolverDnSSgesv_bufferSize
    !---------------------------------------------
    function hipsolverDnSSgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, &
                                          lwork) &
       result(DnSSgesv_bufferSize) &
       bind(C, name="hipsolverDnSSgesv_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSSgesv_bufferSize
    end function hipsolverDnSSgesv_bufferSize

    !---------------------------------------------
    ! hipsolverDnDDgesv_bufferSize
    !---------------------------------------------
    function hipsolverDnDDgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, &
                                          lwork) &
       result(DnDDgesv_bufferSize) &
       bind(C, name="hipsolverDnDDgesv_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDDgesv_bufferSize
    end function hipsolverDnDDgesv_bufferSize

    !---------------------------------------------
    ! hipsolverDnCCgesv_bufferSize
    !---------------------------------------------
    function hipsolverDnCCgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, &
                                          lwork) &
       result(DnCCgesv_bufferSize) &
       bind(C, name="hipsolverDnCCgesv_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCCgesv_bufferSize
    end function hipsolverDnCCgesv_bufferSize

    !---------------------------------------------
    ! hipsolverDnZZgesv_bufferSize
    !---------------------------------------------
    function hipsolverDnZZgesv_bufferSize(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, &
                                          lwork) &
       result(DnZZgesv_bufferSize) &
       bind(C, name="hipsolverDnZZgesv_bufferSize")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZZgesv_bufferSize
    end function hipsolverDnZZgesv_bufferSize

    !---------------------------------------------
    ! hipsolverDnSSgesv
    !---------------------------------------------
    function hipsolverDnSSgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, &
                               niters, devInfo) &
       result(DnSSgesv) &
       bind(C, name="hipsolverDnSSgesv")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSSgesv
    end function hipsolverDnSSgesv

    !---------------------------------------------
    ! hipsolverDnDDgesv
    !---------------------------------------------
    function hipsolverDnDDgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, &
                               niters, devInfo) &
       result(DnDDgesv) &
       bind(C, name="hipsolverDnDDgesv")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDDgesv
    end function hipsolverDnDDgesv

    !---------------------------------------------
    ! hipsolverDnCCgesv
    !---------------------------------------------
    function hipsolverDnCCgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, &
                               niters, devInfo) &
       result(DnCCgesv) &
       bind(C, name="hipsolverDnCCgesv")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCCgesv
    end function hipsolverDnCCgesv

    !---------------------------------------------
    ! hipsolverDnZZgesv
    !---------------------------------------------
    function hipsolverDnZZgesv(handle, n, nrhs, A, lda, devIpiv, B, ldb, X, ldx, work, lwork, &
                               niters, devInfo) &
       result(DnZZgesv) &
       bind(C, name="hipsolverDnZZgesv")
       import :: c_ptr, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: X
       integer(c_int), value :: ldx
       type(c_ptr), value :: work
       integer(c_size_t), value :: lwork
       integer(c_int) :: niters
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZZgesv
    end function hipsolverDnZZgesv

    !---------------------------------------------
    ! hipsolverDnSgesvd_bufferSize
    !---------------------------------------------
    function hipsolverDnSgesvd_bufferSize(handle, m, n, lwork) &
       result(DnSgesvd_bufferSize) &
       bind(C, name="hipsolverDnSgesvd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgesvd_bufferSize
    end function hipsolverDnSgesvd_bufferSize

    !---------------------------------------------
    ! hipsolverDnDgesvd_bufferSize
    !---------------------------------------------
    function hipsolverDnDgesvd_bufferSize(handle, m, n, lwork) &
       result(DnDgesvd_bufferSize) &
       bind(C, name="hipsolverDnDgesvd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgesvd_bufferSize
    end function hipsolverDnDgesvd_bufferSize

    !---------------------------------------------
    ! hipsolverDnCgesvd_bufferSize
    !---------------------------------------------
    function hipsolverDnCgesvd_bufferSize(handle, m, n, lwork) &
       result(DnCgesvd_bufferSize) &
       bind(C, name="hipsolverDnCgesvd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgesvd_bufferSize
    end function hipsolverDnCgesvd_bufferSize

    !---------------------------------------------
    ! hipsolverDnZgesvd_bufferSize
    !---------------------------------------------
    function hipsolverDnZgesvd_bufferSize(handle, m, n, lwork) &
       result(DnZgesvd_bufferSize) &
       bind(C, name="hipsolverDnZgesvd_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgesvd_bufferSize
    end function hipsolverDnZgesvd_bufferSize

    !---------------------------------------------
    ! hipsolverDnSgesvd
    !---------------------------------------------
    function hipsolverDnSgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                               rwork, devInfo) &
       result(DnSgesvd) &
       bind(C, name="hipsolverDnSgesvd")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: rwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgesvd
    end function hipsolverDnSgesvd

    !---------------------------------------------
    ! hipsolverDnDgesvd
    !---------------------------------------------
    function hipsolverDnDgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                               rwork, devInfo) &
       result(DnDgesvd) &
       bind(C, name="hipsolverDnDgesvd")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: rwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgesvd
    end function hipsolverDnDgesvd

    !---------------------------------------------
    ! hipsolverDnCgesvd
    !---------------------------------------------
    function hipsolverDnCgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                               rwork, devInfo) &
       result(DnCgesvd) &
       bind(C, name="hipsolverDnCgesvd")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: rwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgesvd
    end function hipsolverDnCgesvd

    !---------------------------------------------
    ! hipsolverDnZgesvd
    !---------------------------------------------
    function hipsolverDnZgesvd(handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                               rwork, devInfo) &
       result(DnZgesvd) &
       bind(C, name="hipsolverDnZgesvd")
       import :: c_ptr, c_char, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       character(c_char), value :: jobu
       character(c_char), value :: jobv
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: rwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgesvd
    end function hipsolverDnZgesvd

    !---------------------------------------------
    ! hipsolverDnSgesvdj_bufferSize
    !---------------------------------------------
    function hipsolverDnSgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &
                                           lwork, params) &
       result(DnSgesvdj_bufferSize) &
       bind(C, name="hipsolverDnSgesvdj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgesvdj_bufferSize
    end function hipsolverDnSgesvdj_bufferSize

    !---------------------------------------------
    ! hipsolverDnDgesvdj_bufferSize
    !---------------------------------------------
    function hipsolverDnDgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &
                                           lwork, params) &
       result(DnDgesvdj_bufferSize) &
       bind(C, name="hipsolverDnDgesvdj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgesvdj_bufferSize
    end function hipsolverDnDgesvdj_bufferSize

    !---------------------------------------------
    ! hipsolverDnCgesvdj_bufferSize
    !---------------------------------------------
    function hipsolverDnCgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &
                                           lwork, params) &
       result(DnCgesvdj_bufferSize) &
       bind(C, name="hipsolverDnCgesvdj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgesvdj_bufferSize
    end function hipsolverDnCgesvdj_bufferSize

    !---------------------------------------------
    ! hipsolverDnZgesvdj_bufferSize
    !---------------------------------------------
    function hipsolverDnZgesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, &
                                           lwork, params) &
       result(DnZgesvdj_bufferSize) &
       bind(C, name="hipsolverDnZgesvdj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgesvdj_bufferSize
    end function hipsolverDnZgesvdj_bufferSize

    !---------------------------------------------
    ! hipsolverDnSgesvdj
    !---------------------------------------------
    function hipsolverDnSgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                devInfo, params) &
       result(DnSgesvdj) &
       bind(C, name="hipsolverDnSgesvdj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgesvdj
    end function hipsolverDnSgesvdj

    !---------------------------------------------
    ! hipsolverDnDgesvdj
    !---------------------------------------------
    function hipsolverDnDgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                devInfo, params) &
       result(DnDgesvdj) &
       bind(C, name="hipsolverDnDgesvdj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgesvdj
    end function hipsolverDnDgesvdj

    !---------------------------------------------
    ! hipsolverDnCgesvdj
    !---------------------------------------------
    function hipsolverDnCgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                devInfo, params) &
       result(DnCgesvdj) &
       bind(C, name="hipsolverDnCgesvdj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgesvdj
    end function hipsolverDnCgesvdj

    !---------------------------------------------
    ! hipsolverDnZgesvdj
    !---------------------------------------------
    function hipsolverDnZgesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                devInfo, params) &
       result(DnZgesvdj) &
       bind(C, name="hipsolverDnZgesvdj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: econ
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgesvdj
    end function hipsolverDnZgesvdj

    !---------------------------------------------
    ! hipsolverDnSgesvdjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnSgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &
                                                  lwork, params, batch_count) &
       result(DnSgesvdjBatched_bufferSize) &
       bind(C, name="hipsolverDnSgesvdjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgesvdjBatched_bufferSize
    end function hipsolverDnSgesvdjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnDgesvdjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnDgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &
                                                  lwork, params, batch_count) &
       result(DnDgesvdjBatched_bufferSize) &
       bind(C, name="hipsolverDnDgesvdjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgesvdjBatched_bufferSize
    end function hipsolverDnDgesvdjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnCgesvdjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnCgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &
                                                  lwork, params, batch_count) &
       result(DnCgesvdjBatched_bufferSize) &
       bind(C, name="hipsolverDnCgesvdjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgesvdjBatched_bufferSize
    end function hipsolverDnCgesvdjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnZgesvdjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnZgesvdjBatched_bufferSize(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, &
                                                  lwork, params, batch_count) &
       result(DnZgesvdjBatched_bufferSize) &
       bind(C, name="hipsolverDnZgesvdjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgesvdjBatched_bufferSize
    end function hipsolverDnZgesvdjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnSgesvdjBatched
    !---------------------------------------------
    function hipsolverDnSgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                       devInfo, params, batch_count) &
       result(DnSgesvdjBatched) &
       bind(C, name="hipsolverDnSgesvdjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgesvdjBatched
    end function hipsolverDnSgesvdjBatched

    !---------------------------------------------
    ! hipsolverDnDgesvdjBatched
    !---------------------------------------------
    function hipsolverDnDgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                       devInfo, params, batch_count) &
       result(DnDgesvdjBatched) &
       bind(C, name="hipsolverDnDgesvdjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgesvdjBatched
    end function hipsolverDnDgesvdjBatched

    !---------------------------------------------
    ! hipsolverDnCgesvdjBatched
    !---------------------------------------------
    function hipsolverDnCgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                       devInfo, params, batch_count) &
       result(DnCgesvdjBatched) &
       bind(C, name="hipsolverDnCgesvdjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgesvdjBatched
    end function hipsolverDnCgesvdjBatched

    !---------------------------------------------
    ! hipsolverDnZgesvdjBatched
    !---------------------------------------------
    function hipsolverDnZgesvdjBatched(handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, &
                                       devInfo, params, batch_count) &
       result(DnZgesvdjBatched) &
       bind(C, name="hipsolverDnZgesvdjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: S
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgesvdjBatched
    end function hipsolverDnZgesvdjBatched

    !---------------------------------------------
    ! hipsolverDnSgesvdaStridedBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnSgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, &
                                                         strideA, S, strideS, U, ldu, strideU, V, &
                                                         ldv, strideV, lwork, batch_count) &
       result(DnSgesvdaStridedBatched_bufferSize) &
       bind(C, name="hipsolverDnSgesvdaStridedBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, c_int64_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: rank
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgesvdaStridedBatched_bufferSize
    end function hipsolverDnSgesvdaStridedBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnDgesvdaStridedBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnDgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, &
                                                         strideA, S, strideS, U, ldu, strideU, V, &
                                                         ldv, strideV, lwork, batch_count) &
       result(DnDgesvdaStridedBatched_bufferSize) &
       bind(C, name="hipsolverDnDgesvdaStridedBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, c_int64_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: rank
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgesvdaStridedBatched_bufferSize
    end function hipsolverDnDgesvdaStridedBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnCgesvdaStridedBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnCgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, &
                                                         strideA, S, strideS, U, ldu, strideU, V, &
                                                         ldv, strideV, lwork, batch_count) &
       result(DnCgesvdaStridedBatched_bufferSize) &
       bind(C, name="hipsolverDnCgesvdaStridedBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, c_int64_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: rank
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgesvdaStridedBatched_bufferSize
    end function hipsolverDnCgesvdaStridedBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnZgesvdaStridedBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnZgesvdaStridedBatched_bufferSize(handle, jobz, rank, m, n, A, lda, &
                                                         strideA, S, strideS, U, ldu, strideU, V, &
                                                         ldv, strideV, lwork, batch_count) &
       result(DnZgesvdaStridedBatched_bufferSize) &
       bind(C, name="hipsolverDnZgesvdaStridedBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, c_int64_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: rank
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       integer(c_int) :: lwork
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgesvdaStridedBatched_bufferSize
    end function hipsolverDnZgesvdaStridedBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnSgesvdaStridedBatched
    !---------------------------------------------
    function hipsolverDnSgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, &
                                              strideS, U, ldu, strideU, V, ldv, strideV, work, &
                                              lwork, devInfo, hRnrmF, batch_count) &
       result(DnSgesvdaStridedBatched) &
       bind(C, name="hipsolverDnSgesvdaStridedBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, c_int64_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: rank
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: hRnrmF
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgesvdaStridedBatched
    end function hipsolverDnSgesvdaStridedBatched

    !---------------------------------------------
    ! hipsolverDnDgesvdaStridedBatched
    !---------------------------------------------
    function hipsolverDnDgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, &
                                              strideS, U, ldu, strideU, V, ldv, strideV, work, &
                                              lwork, devInfo, hRnrmF, batch_count) &
       result(DnDgesvdaStridedBatched) &
       bind(C, name="hipsolverDnDgesvdaStridedBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, c_int64_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: rank
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: hRnrmF
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgesvdaStridedBatched
    end function hipsolverDnDgesvdaStridedBatched

    !---------------------------------------------
    ! hipsolverDnCgesvdaStridedBatched
    !---------------------------------------------
    function hipsolverDnCgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, &
                                              strideS, U, ldu, strideU, V, ldv, strideV, work, &
                                              lwork, devInfo, hRnrmF, batch_count) &
       result(DnCgesvdaStridedBatched) &
       bind(C, name="hipsolverDnCgesvdaStridedBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, c_int64_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: rank
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: hRnrmF
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgesvdaStridedBatched
    end function hipsolverDnCgesvdaStridedBatched

    !---------------------------------------------
    ! hipsolverDnZgesvdaStridedBatched
    !---------------------------------------------
    function hipsolverDnZgesvdaStridedBatched(handle, jobz, rank, m, n, A, lda, strideA, S, &
                                              strideS, U, ldu, strideU, V, ldv, strideV, work, &
                                              lwork, devInfo, hRnrmF, batch_count) &
       result(DnZgesvdaStridedBatched) &
       bind(C, name="hipsolverDnZgesvdaStridedBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int, c_int64_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(c_int), value :: rank
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int64_t), value :: strideA
       type(c_ptr), value :: S
       integer(c_int64_t), value :: strideS
       type(c_ptr), value :: U
       integer(c_int), value :: ldu
       integer(c_int64_t), value :: strideU
       type(c_ptr), value :: V
       integer(c_int), value :: ldv
       integer(c_int64_t), value :: strideV
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: hRnrmF
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgesvdaStridedBatched
    end function hipsolverDnZgesvdaStridedBatched

    !---------------------------------------------
    ! hipsolverDnSgetrf_bufferSize
    !---------------------------------------------
    function hipsolverDnSgetrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(DnSgetrf_bufferSize) &
       bind(C, name="hipsolverDnSgetrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgetrf_bufferSize
    end function hipsolverDnSgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnDgetrf_bufferSize
    !---------------------------------------------
    function hipsolverDnDgetrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(DnDgetrf_bufferSize) &
       bind(C, name="hipsolverDnDgetrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgetrf_bufferSize
    end function hipsolverDnDgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnCgetrf_bufferSize
    !---------------------------------------------
    function hipsolverDnCgetrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(DnCgetrf_bufferSize) &
       bind(C, name="hipsolverDnCgetrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgetrf_bufferSize
    end function hipsolverDnCgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnZgetrf_bufferSize
    !---------------------------------------------
    function hipsolverDnZgetrf_bufferSize(handle, m, n, A, lda, lwork) &
       result(DnZgetrf_bufferSize) &
       bind(C, name="hipsolverDnZgetrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgetrf_bufferSize
    end function hipsolverDnZgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnSgetrf
    !---------------------------------------------
    function hipsolverDnSgetrf(handle, m, n, A, lda, work, devIpiv, devInfo) &
       result(DnSgetrf) &
       bind(C, name="hipsolverDnSgetrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgetrf
    end function hipsolverDnSgetrf

    !---------------------------------------------
    ! hipsolverDnDgetrf
    !---------------------------------------------
    function hipsolverDnDgetrf(handle, m, n, A, lda, work, devIpiv, devInfo) &
       result(DnDgetrf) &
       bind(C, name="hipsolverDnDgetrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgetrf
    end function hipsolverDnDgetrf

    !---------------------------------------------
    ! hipsolverDnCgetrf
    !---------------------------------------------
    function hipsolverDnCgetrf(handle, m, n, A, lda, work, devIpiv, devInfo) &
       result(DnCgetrf) &
       bind(C, name="hipsolverDnCgetrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgetrf
    end function hipsolverDnCgetrf

    !---------------------------------------------
    ! hipsolverDnZgetrf
    !---------------------------------------------
    function hipsolverDnZgetrf(handle, m, n, A, lda, work, devIpiv, devInfo) &
       result(DnZgetrf) &
       bind(C, name="hipsolverDnZgetrf")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: m
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgetrf
    end function hipsolverDnZgetrf

    !---------------------------------------------
    ! hipsolverDnSgetrs
    !---------------------------------------------
    function hipsolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo) &
       result(DnSgetrs) &
       bind(C, name="hipsolverDnSgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSgetrs
    end function hipsolverDnSgetrs

    !---------------------------------------------
    ! hipsolverDnDgetrs
    !---------------------------------------------
    function hipsolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo) &
       result(DnDgetrs) &
       bind(C, name="hipsolverDnDgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDgetrs
    end function hipsolverDnDgetrs

    !---------------------------------------------
    ! hipsolverDnCgetrs
    !---------------------------------------------
    function hipsolverDnCgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo) &
       result(DnCgetrs) &
       bind(C, name="hipsolverDnCgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCgetrs
    end function hipsolverDnCgetrs

    !---------------------------------------------
    ! hipsolverDnZgetrs
    !---------------------------------------------
    function hipsolverDnZgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo) &
       result(DnZgetrs) &
       bind(C, name="hipsolverDnZgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devIpiv
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZgetrs
    end function hipsolverDnZgetrs

    !---------------------------------------------
    ! hipsolverDnSpotrf_bufferSize
    !---------------------------------------------
    function hipsolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(DnSpotrf_bufferSize) &
       bind(C, name="hipsolverDnSpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSpotrf_bufferSize
    end function hipsolverDnSpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnDpotrf_bufferSize
    !---------------------------------------------
    function hipsolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(DnDpotrf_bufferSize) &
       bind(C, name="hipsolverDnDpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDpotrf_bufferSize
    end function hipsolverDnDpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnCpotrf_bufferSize
    !---------------------------------------------
    function hipsolverDnCpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(DnCpotrf_bufferSize) &
       bind(C, name="hipsolverDnCpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCpotrf_bufferSize
    end function hipsolverDnCpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnZpotrf_bufferSize
    !---------------------------------------------
    function hipsolverDnZpotrf_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(DnZpotrf_bufferSize) &
       bind(C, name="hipsolverDnZpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZpotrf_bufferSize
    end function hipsolverDnZpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnSpotrf
    !---------------------------------------------
    function hipsolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(DnSpotrf) &
       bind(C, name="hipsolverDnSpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSpotrf
    end function hipsolverDnSpotrf

    !---------------------------------------------
    ! hipsolverDnDpotrf
    !---------------------------------------------
    function hipsolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(DnDpotrf) &
       bind(C, name="hipsolverDnDpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDpotrf
    end function hipsolverDnDpotrf

    !---------------------------------------------
    ! hipsolverDnCpotrf
    !---------------------------------------------
    function hipsolverDnCpotrf(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(DnCpotrf) &
       bind(C, name="hipsolverDnCpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCpotrf
    end function hipsolverDnCpotrf

    !---------------------------------------------
    ! hipsolverDnZpotrf
    !---------------------------------------------
    function hipsolverDnZpotrf(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(DnZpotrf) &
       bind(C, name="hipsolverDnZpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZpotrf
    end function hipsolverDnZpotrf

    !---------------------------------------------
    ! hipsolverDnSpotrfBatched
    !---------------------------------------------
    function hipsolverDnSpotrfBatched(handle, uplo, n, A, lda, devInfo, batch_count) &
       result(DnSpotrfBatched) &
       bind(C, name="hipsolverDnSpotrfBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSpotrfBatched
    end function hipsolverDnSpotrfBatched

    !---------------------------------------------
    ! hipsolverDnDpotrfBatched
    !---------------------------------------------
    function hipsolverDnDpotrfBatched(handle, uplo, n, A, lda, devInfo, batch_count) &
       result(DnDpotrfBatched) &
       bind(C, name="hipsolverDnDpotrfBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDpotrfBatched
    end function hipsolverDnDpotrfBatched

    !---------------------------------------------
    ! hipsolverDnCpotrfBatched
    !---------------------------------------------
    function hipsolverDnCpotrfBatched(handle, uplo, n, A, lda, devInfo, batch_count) &
       result(DnCpotrfBatched) &
       bind(C, name="hipsolverDnCpotrfBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCpotrfBatched
    end function hipsolverDnCpotrfBatched

    !---------------------------------------------
    ! hipsolverDnZpotrfBatched
    !---------------------------------------------
    function hipsolverDnZpotrfBatched(handle, uplo, n, A, lda, devInfo, batch_count) &
       result(DnZpotrfBatched) &
       bind(C, name="hipsolverDnZpotrfBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZpotrfBatched
    end function hipsolverDnZpotrfBatched

    !---------------------------------------------
    ! hipsolverDnSpotri_bufferSize
    !---------------------------------------------
    function hipsolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(DnSpotri_bufferSize) &
       bind(C, name="hipsolverDnSpotri_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSpotri_bufferSize
    end function hipsolverDnSpotri_bufferSize

    !---------------------------------------------
    ! hipsolverDnDpotri_bufferSize
    !---------------------------------------------
    function hipsolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(DnDpotri_bufferSize) &
       bind(C, name="hipsolverDnDpotri_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDpotri_bufferSize
    end function hipsolverDnDpotri_bufferSize

    !---------------------------------------------
    ! hipsolverDnCpotri_bufferSize
    !---------------------------------------------
    function hipsolverDnCpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(DnCpotri_bufferSize) &
       bind(C, name="hipsolverDnCpotri_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCpotri_bufferSize
    end function hipsolverDnCpotri_bufferSize

    !---------------------------------------------
    ! hipsolverDnZpotri_bufferSize
    !---------------------------------------------
    function hipsolverDnZpotri_bufferSize(handle, uplo, n, A, lda, lwork) &
       result(DnZpotri_bufferSize) &
       bind(C, name="hipsolverDnZpotri_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZpotri_bufferSize
    end function hipsolverDnZpotri_bufferSize

    !---------------------------------------------
    ! hipsolverDnSpotri
    !---------------------------------------------
    function hipsolverDnSpotri(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(DnSpotri) &
       bind(C, name="hipsolverDnSpotri")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSpotri
    end function hipsolverDnSpotri

    !---------------------------------------------
    ! hipsolverDnDpotri
    !---------------------------------------------
    function hipsolverDnDpotri(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(DnDpotri) &
       bind(C, name="hipsolverDnDpotri")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDpotri
    end function hipsolverDnDpotri

    !---------------------------------------------
    ! hipsolverDnCpotri
    !---------------------------------------------
    function hipsolverDnCpotri(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(DnCpotri) &
       bind(C, name="hipsolverDnCpotri")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCpotri
    end function hipsolverDnCpotri

    !---------------------------------------------
    ! hipsolverDnZpotri
    !---------------------------------------------
    function hipsolverDnZpotri(handle, uplo, n, A, lda, work, lwork, devInfo) &
       result(DnZpotri) &
       bind(C, name="hipsolverDnZpotri")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZpotri
    end function hipsolverDnZpotri

    !---------------------------------------------
    ! hipsolverDnSpotrs
    !---------------------------------------------
    function hipsolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) &
       result(DnSpotrs) &
       bind(C, name="hipsolverDnSpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSpotrs
    end function hipsolverDnSpotrs

    !---------------------------------------------
    ! hipsolverDnDpotrs
    !---------------------------------------------
    function hipsolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) &
       result(DnDpotrs) &
       bind(C, name="hipsolverDnDpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDpotrs
    end function hipsolverDnDpotrs

    !---------------------------------------------
    ! hipsolverDnCpotrs
    !---------------------------------------------
    function hipsolverDnCpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) &
       result(DnCpotrs) &
       bind(C, name="hipsolverDnCpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCpotrs
    end function hipsolverDnCpotrs

    !---------------------------------------------
    ! hipsolverDnZpotrs
    !---------------------------------------------
    function hipsolverDnZpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo) &
       result(DnZpotrs) &
       bind(C, name="hipsolverDnZpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZpotrs
    end function hipsolverDnZpotrs

    !---------------------------------------------
    ! hipsolverDnSpotrsBatched
    !---------------------------------------------
    function hipsolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo, batch_count) &
       result(DnSpotrsBatched) &
       bind(C, name="hipsolverDnSpotrsBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSpotrsBatched
    end function hipsolverDnSpotrsBatched

    !---------------------------------------------
    ! hipsolverDnDpotrsBatched
    !---------------------------------------------
    function hipsolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo, batch_count) &
       result(DnDpotrsBatched) &
       bind(C, name="hipsolverDnDpotrsBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDpotrsBatched
    end function hipsolverDnDpotrsBatched

    !---------------------------------------------
    ! hipsolverDnCpotrsBatched
    !---------------------------------------------
    function hipsolverDnCpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo, batch_count) &
       result(DnCpotrsBatched) &
       bind(C, name="hipsolverDnCpotrsBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCpotrsBatched
    end function hipsolverDnCpotrsBatched

    !---------------------------------------------
    ! hipsolverDnZpotrsBatched
    !---------------------------------------------
    function hipsolverDnZpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo, batch_count) &
       result(DnZpotrsBatched) &
       bind(C, name="hipsolverDnZpotrsBatched")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       integer(c_int), value :: nrhs
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: devInfo
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZpotrsBatched
    end function hipsolverDnZpotrsBatched

    !---------------------------------------------
    ! hipsolverDnSsyevd_bufferSize
    !---------------------------------------------
    function hipsolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) &
       result(DnSsyevd_bufferSize) &
       bind(C, name="hipsolverDnSsyevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsyevd_bufferSize
    end function hipsolverDnSsyevd_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsyevd_bufferSize
    !---------------------------------------------
    function hipsolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) &
       result(DnDsyevd_bufferSize) &
       bind(C, name="hipsolverDnDsyevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsyevd_bufferSize
    end function hipsolverDnDsyevd_bufferSize

    !---------------------------------------------
    ! hipsolverDnCheevd_bufferSize
    !---------------------------------------------
    function hipsolverDnCheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) &
       result(DnCheevd_bufferSize) &
       bind(C, name="hipsolverDnCheevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCheevd_bufferSize
    end function hipsolverDnCheevd_bufferSize

    !---------------------------------------------
    ! hipsolverDnZheevd_bufferSize
    !---------------------------------------------
    function hipsolverDnZheevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork) &
       result(DnZheevd_bufferSize) &
       bind(C, name="hipsolverDnZheevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZheevd_bufferSize
    end function hipsolverDnZheevd_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsyevd
    !---------------------------------------------
    function hipsolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo) &
       result(DnSsyevd) &
       bind(C, name="hipsolverDnSsyevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsyevd
    end function hipsolverDnSsyevd

    !---------------------------------------------
    ! hipsolverDnDsyevd
    !---------------------------------------------
    function hipsolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo) &
       result(DnDsyevd) &
       bind(C, name="hipsolverDnDsyevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsyevd
    end function hipsolverDnDsyevd

    !---------------------------------------------
    ! hipsolverDnCheevd
    !---------------------------------------------
    function hipsolverDnCheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo) &
       result(DnCheevd) &
       bind(C, name="hipsolverDnCheevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCheevd
    end function hipsolverDnCheevd

    !---------------------------------------------
    ! hipsolverDnZheevd
    !---------------------------------------------
    function hipsolverDnZheevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo) &
       result(DnZheevd) &
       bind(C, name="hipsolverDnZheevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZheevd
    end function hipsolverDnZheevd

    !---------------------------------------------
    ! hipsolverDnSsyevdx_bufferSize
    !---------------------------------------------
    function hipsolverDnSsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, &
                                           nev, W, lwork) &
       result(DnSsyevdx_bufferSize) &
       bind(C, name="hipsolverDnSsyevdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsyevdx_bufferSize
    end function hipsolverDnSsyevdx_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsyevdx_bufferSize
    !---------------------------------------------
    function hipsolverDnDsyevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, &
                                           nev, W, lwork) &
       result(DnDsyevdx_bufferSize) &
       bind(C, name="hipsolverDnDsyevdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsyevdx_bufferSize
    end function hipsolverDnDsyevdx_bufferSize

    !---------------------------------------------
    ! hipsolverDnCheevdx_bufferSize
    !---------------------------------------------
    function hipsolverDnCheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, &
                                           nev, W, lwork) &
       result(DnCheevdx_bufferSize) &
       bind(C, name="hipsolverDnCheevdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCheevdx_bufferSize
    end function hipsolverDnCheevdx_bufferSize

    !---------------------------------------------
    ! hipsolverDnZheevdx_bufferSize
    !---------------------------------------------
    function hipsolverDnZheevdx_bufferSize(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, &
                                           nev, W, lwork) &
       result(DnZheevdx_bufferSize) &
       bind(C, name="hipsolverDnZheevdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZheevdx_bufferSize
    end function hipsolverDnZheevdx_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsyevdx
    !---------------------------------------------
    function hipsolverDnSsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, &
                                work, lwork, devInfo) &
       result(DnSsyevdx) &
       bind(C, name="hipsolverDnSsyevdx")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsyevdx
    end function hipsolverDnSsyevdx

    !---------------------------------------------
    ! hipsolverDnDsyevdx
    !---------------------------------------------
    function hipsolverDnDsyevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, &
                                work, lwork, devInfo) &
       result(DnDsyevdx) &
       bind(C, name="hipsolverDnDsyevdx")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsyevdx
    end function hipsolverDnDsyevdx

    !---------------------------------------------
    ! hipsolverDnCheevdx
    !---------------------------------------------
    function hipsolverDnCheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, &
                                work, lwork, devInfo) &
       result(DnCheevdx) &
       bind(C, name="hipsolverDnCheevdx")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCheevdx
    end function hipsolverDnCheevdx

    !---------------------------------------------
    ! hipsolverDnZheevdx
    !---------------------------------------------
    function hipsolverDnZheevdx(handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, &
                                work, lwork, devInfo) &
       result(DnZheevdx) &
       bind(C, name="hipsolverDnZheevdx")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZheevdx
    end function hipsolverDnZheevdx

    !---------------------------------------------
    ! hipsolverDnSsyevj_bufferSize
    !---------------------------------------------
    function hipsolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
       result(DnSsyevj_bufferSize) &
       bind(C, name="hipsolverDnSsyevj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsyevj_bufferSize
    end function hipsolverDnSsyevj_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsyevj_bufferSize
    !---------------------------------------------
    function hipsolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
       result(DnDsyevj_bufferSize) &
       bind(C, name="hipsolverDnDsyevj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsyevj_bufferSize
    end function hipsolverDnDsyevj_bufferSize

    !---------------------------------------------
    ! hipsolverDnCheevj_bufferSize
    !---------------------------------------------
    function hipsolverDnCheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
       result(DnCheevj_bufferSize) &
       bind(C, name="hipsolverDnCheevj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCheevj_bufferSize
    end function hipsolverDnCheevj_bufferSize

    !---------------------------------------------
    ! hipsolverDnZheevj_bufferSize
    !---------------------------------------------
    function hipsolverDnZheevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params) &
       result(DnZheevj_bufferSize) &
       bind(C, name="hipsolverDnZheevj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZheevj_bufferSize
    end function hipsolverDnZheevj_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsyevj
    !---------------------------------------------
    function hipsolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params) &
       result(DnSsyevj) &
       bind(C, name="hipsolverDnSsyevj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsyevj
    end function hipsolverDnSsyevj

    !---------------------------------------------
    ! hipsolverDnDsyevj
    !---------------------------------------------
    function hipsolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params) &
       result(DnDsyevj) &
       bind(C, name="hipsolverDnDsyevj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsyevj
    end function hipsolverDnDsyevj

    !---------------------------------------------
    ! hipsolverDnCheevj
    !---------------------------------------------
    function hipsolverDnCheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params) &
       result(DnCheevj) &
       bind(C, name="hipsolverDnCheevj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCheevj
    end function hipsolverDnCheevj

    !---------------------------------------------
    ! hipsolverDnZheevj
    !---------------------------------------------
    function hipsolverDnZheevj(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params) &
       result(DnZheevj) &
       bind(C, name="hipsolverDnZheevj")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZheevj
    end function hipsolverDnZheevj

    !---------------------------------------------
    ! hipsolverDnSsyevjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnSsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, &
                                                 batch_count) &
       result(DnSsyevjBatched_bufferSize) &
       bind(C, name="hipsolverDnSsyevjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsyevjBatched_bufferSize
    end function hipsolverDnSsyevjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsyevjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnDsyevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, &
                                                 batch_count) &
       result(DnDsyevjBatched_bufferSize) &
       bind(C, name="hipsolverDnDsyevjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsyevjBatched_bufferSize
    end function hipsolverDnDsyevjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnCheevjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnCheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, &
                                                 batch_count) &
       result(DnCheevjBatched_bufferSize) &
       bind(C, name="hipsolverDnCheevjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCheevjBatched_bufferSize
    end function hipsolverDnCheevjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnZheevjBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnZheevjBatched_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params, &
                                                 batch_count) &
       result(DnZheevjBatched_bufferSize) &
       bind(C, name="hipsolverDnZheevjBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZheevjBatched_bufferSize
    end function hipsolverDnZheevjBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsyevjBatched
    !---------------------------------------------
    function hipsolverDnSsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, &
                                      params, batch_count) &
       result(DnSsyevjBatched) &
       bind(C, name="hipsolverDnSsyevjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsyevjBatched
    end function hipsolverDnSsyevjBatched

    !---------------------------------------------
    ! hipsolverDnDsyevjBatched
    !---------------------------------------------
    function hipsolverDnDsyevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, &
                                      params, batch_count) &
       result(DnDsyevjBatched) &
       bind(C, name="hipsolverDnDsyevjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsyevjBatched
    end function hipsolverDnDsyevjBatched

    !---------------------------------------------
    ! hipsolverDnCheevjBatched
    !---------------------------------------------
    function hipsolverDnCheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, &
                                      params, batch_count) &
       result(DnCheevjBatched) &
       bind(C, name="hipsolverDnCheevjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCheevjBatched
    end function hipsolverDnCheevjBatched

    !---------------------------------------------
    ! hipsolverDnZheevjBatched
    !---------------------------------------------
    function hipsolverDnZheevjBatched(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, &
                                      params, batch_count) &
       result(DnZheevjBatched) &
       bind(C, name="hipsolverDnZheevjBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(c_int), value :: batch_count
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZheevjBatched
    end function hipsolverDnZheevjBatched

    !---------------------------------------------
    ! hipsolverDnSsygvd_bufferSize
    !---------------------------------------------
    function hipsolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
       result(DnSsygvd_bufferSize) &
       bind(C, name="hipsolverDnSsygvd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsygvd_bufferSize
    end function hipsolverDnSsygvd_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsygvd_bufferSize
    !---------------------------------------------
    function hipsolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
       result(DnDsygvd_bufferSize) &
       bind(C, name="hipsolverDnDsygvd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsygvd_bufferSize
    end function hipsolverDnDsygvd_bufferSize

    !---------------------------------------------
    ! hipsolverDnChegvd_bufferSize
    !---------------------------------------------
    function hipsolverDnChegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
       result(DnChegvd_bufferSize) &
       bind(C, name="hipsolverDnChegvd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnChegvd_bufferSize
    end function hipsolverDnChegvd_bufferSize

    !---------------------------------------------
    ! hipsolverDnZhegvd_bufferSize
    !---------------------------------------------
    function hipsolverDnZhegvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork) &
       result(DnZhegvd_bufferSize) &
       bind(C, name="hipsolverDnZhegvd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZhegvd_bufferSize
    end function hipsolverDnZhegvd_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsygvd
    !---------------------------------------------
    function hipsolverDnSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                               devInfo) &
       result(DnSsygvd) &
       bind(C, name="hipsolverDnSsygvd")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsygvd
    end function hipsolverDnSsygvd

    !---------------------------------------------
    ! hipsolverDnDsygvd
    !---------------------------------------------
    function hipsolverDnDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                               devInfo) &
       result(DnDsygvd) &
       bind(C, name="hipsolverDnDsygvd")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsygvd
    end function hipsolverDnDsygvd

    !---------------------------------------------
    ! hipsolverDnChegvd
    !---------------------------------------------
    function hipsolverDnChegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                               devInfo) &
       result(DnChegvd) &
       bind(C, name="hipsolverDnChegvd")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnChegvd
    end function hipsolverDnChegvd

    !---------------------------------------------
    ! hipsolverDnZhegvd
    !---------------------------------------------
    function hipsolverDnZhegvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                               devInfo) &
       result(DnZhegvd) &
       bind(C, name="hipsolverDnZhegvd")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZhegvd
    end function hipsolverDnZhegvd

    !---------------------------------------------
    ! hipsolverDnSsygvdx_bufferSize
    !---------------------------------------------
    function hipsolverDnSsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, &
                                           vl, vu, il, iu, nev, W, lwork) &
       result(DnSsygvdx_bufferSize) &
       bind(C, name="hipsolverDnSsygvdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsygvdx_bufferSize
    end function hipsolverDnSsygvdx_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsygvdx_bufferSize
    !---------------------------------------------
    function hipsolverDnDsygvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, &
                                           vl, vu, il, iu, nev, W, lwork) &
       result(DnDsygvdx_bufferSize) &
       bind(C, name="hipsolverDnDsygvdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsygvdx_bufferSize
    end function hipsolverDnDsygvdx_bufferSize

    !---------------------------------------------
    ! hipsolverDnChegvdx_bufferSize
    !---------------------------------------------
    function hipsolverDnChegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, &
                                           vl, vu, il, iu, nev, W, lwork) &
       result(DnChegvdx_bufferSize) &
       bind(C, name="hipsolverDnChegvdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnChegvdx_bufferSize
    end function hipsolverDnChegvdx_bufferSize

    !---------------------------------------------
    ! hipsolverDnZhegvdx_bufferSize
    !---------------------------------------------
    function hipsolverDnZhegvdx_bufferSize(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, &
                                           vl, vu, il, iu, nev, W, lwork) &
       result(DnZhegvdx_bufferSize) &
       bind(C, name="hipsolverDnZhegvdx_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZhegvdx_bufferSize
    end function hipsolverDnZhegvdx_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsygvdx
    !---------------------------------------------
    function hipsolverDnSsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, &
                                iu, nev, W, work, lwork, devInfo) &
       result(DnSsygvdx) &
       bind(C, name="hipsolverDnSsygvdx")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsygvdx
    end function hipsolverDnSsygvdx

    !---------------------------------------------
    ! hipsolverDnDsygvdx
    !---------------------------------------------
    function hipsolverDnDsygvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, &
                                iu, nev, W, work, lwork, devInfo) &
       result(DnDsygvdx) &
       bind(C, name="hipsolverDnDsygvdx")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsygvdx
    end function hipsolverDnDsygvdx

    !---------------------------------------------
    ! hipsolverDnChegvdx
    !---------------------------------------------
    function hipsolverDnChegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, &
                                iu, nev, W, work, lwork, devInfo) &
       result(DnChegvdx) &
       bind(C, name="hipsolverDnChegvdx")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_float), value :: vl
       real(c_float), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnChegvdx
    end function hipsolverDnChegvdx

    !---------------------------------------------
    ! hipsolverDnZhegvdx
    !---------------------------------------------
    function hipsolverDnZhegvdx(handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, &
                                iu, nev, W, work, lwork, devInfo) &
       result(DnZhegvdx) &
       bind(C, name="hipsolverDnZhegvdx")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_EIG_RANGE_ALL, &
                 HIPSOLVER_FILL_MODE_UPPER, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_EIG_RANGE_ALL)), value :: range
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       real(c_double), value :: vl
       real(c_double), value :: vu
       integer(c_int), value :: il
       integer(c_int), value :: iu
       integer(c_int) :: nev
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZhegvdx
    end function hipsolverDnZhegvdx

    !---------------------------------------------
    ! hipsolverDnSsygvj_bufferSize
    !---------------------------------------------
    function hipsolverDnSsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, &
                                          params) &
       result(DnSsygvj_bufferSize) &
       bind(C, name="hipsolverDnSsygvj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsygvj_bufferSize
    end function hipsolverDnSsygvj_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsygvj_bufferSize
    !---------------------------------------------
    function hipsolverDnDsygvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, &
                                          params) &
       result(DnDsygvj_bufferSize) &
       bind(C, name="hipsolverDnDsygvj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsygvj_bufferSize
    end function hipsolverDnDsygvj_bufferSize

    !---------------------------------------------
    ! hipsolverDnChegvj_bufferSize
    !---------------------------------------------
    function hipsolverDnChegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, &
                                          params) &
       result(DnChegvj_bufferSize) &
       bind(C, name="hipsolverDnChegvj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnChegvj_bufferSize
    end function hipsolverDnChegvj_bufferSize

    !---------------------------------------------
    ! hipsolverDnZhegvj_bufferSize
    !---------------------------------------------
    function hipsolverDnZhegvj_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, &
                                          params) &
       result(DnZhegvj_bufferSize) &
       bind(C, name="hipsolverDnZhegvj_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       integer(c_int) :: lwork
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZhegvj_bufferSize
    end function hipsolverDnZhegvj_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsygvj
    !---------------------------------------------
    function hipsolverDnSsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                               devInfo, params) &
       result(DnSsygvj) &
       bind(C, name="hipsolverDnSsygvj")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsygvj
    end function hipsolverDnSsygvj

    !---------------------------------------------
    ! hipsolverDnDsygvj
    !---------------------------------------------
    function hipsolverDnDsygvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                               devInfo, params) &
       result(DnDsygvj) &
       bind(C, name="hipsolverDnDsygvj")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsygvj
    end function hipsolverDnDsygvj

    !---------------------------------------------
    ! hipsolverDnChegvj
    !---------------------------------------------
    function hipsolverDnChegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                               devInfo, params) &
       result(DnChegvj) &
       bind(C, name="hipsolverDnChegvj")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnChegvj
    end function hipsolverDnChegvj

    !---------------------------------------------
    ! hipsolverDnZhegvj
    !---------------------------------------------
    function hipsolverDnZhegvj(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, &
                               devInfo, params) &
       result(DnZhegvj) &
       bind(C, name="hipsolverDnZhegvj")
       import :: c_ptr, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                 c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_EIG_TYPE_1)), value :: itype
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: B
       integer(c_int), value :: ldb
       type(c_ptr), value :: W
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZhegvj
    end function hipsolverDnZhegvj

    !---------------------------------------------
    ! hipsolverDnSsytrd_bufferSize
    !---------------------------------------------
    function hipsolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
       result(DnSsytrd_bufferSize) &
       bind(C, name="hipsolverDnSsytrd_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsytrd_bufferSize
    end function hipsolverDnSsytrd_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsytrd_bufferSize
    !---------------------------------------------
    function hipsolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
       result(DnDsytrd_bufferSize) &
       bind(C, name="hipsolverDnDsytrd_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsytrd_bufferSize
    end function hipsolverDnDsytrd_bufferSize

    !---------------------------------------------
    ! hipsolverDnChetrd_bufferSize
    !---------------------------------------------
    function hipsolverDnChetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
       result(DnChetrd_bufferSize) &
       bind(C, name="hipsolverDnChetrd_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnChetrd_bufferSize
    end function hipsolverDnChetrd_bufferSize

    !---------------------------------------------
    ! hipsolverDnZhetrd_bufferSize
    !---------------------------------------------
    function hipsolverDnZhetrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork) &
       result(DnZhetrd_bufferSize) &
       bind(C, name="hipsolverDnZhetrd_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZhetrd_bufferSize
    end function hipsolverDnZhetrd_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsytrd
    !---------------------------------------------
    function hipsolverDnSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo) &
       result(DnSsytrd) &
       bind(C, name="hipsolverDnSsytrd")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsytrd
    end function hipsolverDnSsytrd

    !---------------------------------------------
    ! hipsolverDnDsytrd
    !---------------------------------------------
    function hipsolverDnDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo) &
       result(DnDsytrd) &
       bind(C, name="hipsolverDnDsytrd")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsytrd
    end function hipsolverDnDsytrd

    !---------------------------------------------
    ! hipsolverDnChetrd
    !---------------------------------------------
    function hipsolverDnChetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo) &
       result(DnChetrd) &
       bind(C, name="hipsolverDnChetrd")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnChetrd
    end function hipsolverDnChetrd

    !---------------------------------------------
    ! hipsolverDnZhetrd
    !---------------------------------------------
    function hipsolverDnZhetrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, devInfo) &
       result(DnZhetrd) &
       bind(C, name="hipsolverDnZhetrd")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: D
       type(c_ptr), value :: E
       type(c_ptr), value :: tau
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZhetrd
    end function hipsolverDnZhetrd

    !---------------------------------------------
    ! hipsolverDnSsytrf_bufferSize
    !---------------------------------------------
    function hipsolverDnSsytrf_bufferSize(handle, n, A, lda, lwork) &
       result(DnSsytrf_bufferSize) &
       bind(C, name="hipsolverDnSsytrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsytrf_bufferSize
    end function hipsolverDnSsytrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnDsytrf_bufferSize
    !---------------------------------------------
    function hipsolverDnDsytrf_bufferSize(handle, n, A, lda, lwork) &
       result(DnDsytrf_bufferSize) &
       bind(C, name="hipsolverDnDsytrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsytrf_bufferSize
    end function hipsolverDnDsytrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnCsytrf_bufferSize
    !---------------------------------------------
    function hipsolverDnCsytrf_bufferSize(handle, n, A, lda, lwork) &
       result(DnCsytrf_bufferSize) &
       bind(C, name="hipsolverDnCsytrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCsytrf_bufferSize
    end function hipsolverDnCsytrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnZsytrf_bufferSize
    !---------------------------------------------
    function hipsolverDnZsytrf_bufferSize(handle, n, A, lda, lwork) &
       result(DnZsytrf_bufferSize) &
       bind(C, name="hipsolverDnZsytrf_bufferSize")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       integer(c_int) :: lwork
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZsytrf_bufferSize
    end function hipsolverDnZsytrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnSsytrf
    !---------------------------------------------
    function hipsolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo) &
       result(DnSsytrf) &
       bind(C, name="hipsolverDnSsytrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSsytrf
    end function hipsolverDnSsytrf

    !---------------------------------------------
    ! hipsolverDnDsytrf
    !---------------------------------------------
    function hipsolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo) &
       result(DnDsytrf) &
       bind(C, name="hipsolverDnDsytrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDsytrf
    end function hipsolverDnDsytrf

    !---------------------------------------------
    ! hipsolverDnCsytrf
    !---------------------------------------------
    function hipsolverDnCsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo) &
       result(DnCsytrf) &
       bind(C, name="hipsolverDnCsytrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCsytrf
    end function hipsolverDnCsytrf

    !---------------------------------------------
    ! hipsolverDnZsytrf
    !---------------------------------------------
    function hipsolverDnZsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo) &
       result(DnZsytrf) &
       bind(C, name="hipsolverDnZsytrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int), value :: n
       type(c_ptr), value :: A
       integer(c_int), value :: lda
       type(c_ptr), value :: ipiv
       type(c_ptr), value :: work
       integer(c_int), value :: lwork
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnZsytrf
    end function hipsolverDnZsytrf

    !---------------------------------------------
    ! hipsolverDnCreateParams
    !---------------------------------------------
    function hipsolverDnCreateParams(params) &
       result(DnCreateParams) &
       bind(C, name="hipsolverDnCreateParams")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnCreateParams
    end function hipsolverDnCreateParams

    !---------------------------------------------
    ! hipsolverDnDestroyParams
    !---------------------------------------------
    function hipsolverDnDestroyParams(params) &
       result(DnDestroyParams) &
       bind(C, name="hipsolverDnDestroyParams")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnDestroyParams
    end function hipsolverDnDestroyParams

    !---------------------------------------------
    ! hipsolverDnSetAdvOptions
    !---------------------------------------------
    function hipsolverDnSetAdvOptions(params, func, alg) &
       result(DnSetAdvOptions) &
       bind(C, name="hipsolverDnSetAdvOptions")
       import :: c_ptr, HIPSOLVERDN_GETRF, HIPSOLVER_ALG_0, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: params
       integer(kind(HIPSOLVERDN_GETRF)), value :: func
       integer(kind(HIPSOLVER_ALG_0)), value :: alg
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnSetAdvOptions
    end function hipsolverDnSetAdvOptions

    !---------------------------------------------
    ! hipsolverDnXgeev_bufferSize
    !---------------------------------------------
    function hipsolverDnXgeev_bufferSize(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, &
                                         dataTypeW, W, dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, &
                                         computeType, lworkOnDevice, lworkOnHost) &
       result(DnXgeev_bufferSize) &
       bind(C, name="hipsolverDnXgeev_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int64_t, c_int, c_size_t, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobvl
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobvr
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeW
       type(c_ptr), value :: W
       integer(c_int), value :: dataTypeVL
       type(c_ptr), value :: VL
       integer(c_int64_t), value :: ldvl
       integer(c_int), value :: dataTypeVR
       type(c_ptr), value :: VR
       integer(c_int64_t), value :: ldvr
       integer(c_int), value :: computeType
       integer(c_size_t) :: lworkOnDevice
       integer(c_size_t) :: lworkOnHost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgeev_bufferSize
    end function hipsolverDnXgeev_bufferSize

    !---------------------------------------------
    ! hipsolverDnXgeev
    !---------------------------------------------
    function hipsolverDnXgeev(handle, params, jobvl, jobvr, n, dataTypeA, A, lda, dataTypeW, W, &
                              dataTypeVL, VL, ldvl, dataTypeVR, VR, ldvr, computeType, &
                              workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo) &
       result(DnXgeev) &
       bind(C, name="hipsolverDnXgeev")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, c_int64_t, c_int, c_size_t, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobvl
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobvr
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeW
       type(c_ptr), value :: W
       integer(c_int), value :: dataTypeVL
       type(c_ptr), value :: VL
       integer(c_int64_t), value :: ldvl
       integer(c_int), value :: dataTypeVR
       type(c_ptr), value :: VR
       integer(c_int64_t), value :: ldvr
       integer(c_int), value :: computeType
       type(c_ptr), value :: workOnDevice
       integer(c_size_t), value :: lworkOnDevice
       type(c_ptr), value :: workOnHost
       integer(c_size_t), value :: lworkOnHost
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgeev
    end function hipsolverDnXgeev

    !---------------------------------------------
    ! hipsolverDnXgeqrf_bufferSize
    !---------------------------------------------
    function hipsolverDnXgeqrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, &
                                          tau, computeType, lworkOnDevice, lworkOnHost) &
       result(DnXgeqrf_bufferSize) &
       bind(C, name="hipsolverDnXgeqrf_bufferSize")
       import :: c_ptr, c_int64_t, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeTau
       type(c_ptr), value :: tau
       integer(c_int), value :: computeType
       integer(c_size_t) :: lworkOnDevice
       integer(c_size_t) :: lworkOnHost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgeqrf_bufferSize
    end function hipsolverDnXgeqrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnXgeqrf
    !---------------------------------------------
    function hipsolverDnXgeqrf(handle, params, m, n, dataTypeA, A, lda, dataTypeTau, tau, &
                               computeType, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, &
                               devInfo) &
       result(DnXgeqrf) &
       bind(C, name="hipsolverDnXgeqrf")
       import :: c_ptr, c_int64_t, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeTau
       type(c_ptr), value :: tau
       integer(c_int), value :: computeType
       type(c_ptr), value :: workOnDevice
       integer(c_size_t), value :: lworkOnDevice
       type(c_ptr), value :: workOnHost
       integer(c_size_t), value :: lworkOnHost
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgeqrf
    end function hipsolverDnXgeqrf

    !---------------------------------------------
    ! hipsolverDnXgetrf_bufferSize
    !---------------------------------------------
    function hipsolverDnXgetrf_bufferSize(handle, params, m, n, dataTypeA, A, lda, computeType, &
                                          lworkOnDevice, lworkOnHost) &
       result(DnXgetrf_bufferSize) &
       bind(C, name="hipsolverDnXgetrf_bufferSize")
       import :: c_ptr, c_int64_t, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: computeType
       integer(c_size_t) :: lworkOnDevice
       integer(c_size_t) :: lworkOnHost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgetrf_bufferSize
    end function hipsolverDnXgetrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnXgetrf
    !---------------------------------------------
    function hipsolverDnXgetrf(handle, params, m, n, dataTypeA, A, lda, devIpiv, computeType, &
                               workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo) &
       result(DnXgetrf) &
       bind(C, name="hipsolverDnXgetrf")
       import :: c_ptr, c_int64_t, c_int, c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(c_int64_t), value :: m
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: devIpiv
       integer(c_int), value :: computeType
       type(c_ptr), value :: workOnDevice
       integer(c_size_t), value :: lworkOnDevice
       type(c_ptr), value :: workOnHost
       integer(c_size_t), value :: lworkOnHost
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgetrf
    end function hipsolverDnXgetrf

    !---------------------------------------------
    ! hipsolverDnXgetrs
    !---------------------------------------------
    function hipsolverDnXgetrs(handle, params, trans, n, nrhs, dataTypeA, A, lda, devIpiv, &
                               dataTypeB, B, ldb, devInfo) &
       result(DnXgetrs) &
       bind(C, name="hipsolverDnXgetrs")
       import :: c_ptr, HIPSOLVER_OP_N, c_int64_t, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_OP_N)), value :: trans
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: devIpiv
       integer(c_int), value :: dataTypeB
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXgetrs
    end function hipsolverDnXgetrs

    !---------------------------------------------
    ! hipsolverDnXlarft_bufferSize
    !---------------------------------------------
    function hipsolverDnXlarft_bufferSize(handle, params, myDirect, storev, n, k, dataTypeV, V, &
                                          ldv, dataTypeTau, tau, dataTypeT, T, ldt, computeType, &
                                          lworkOnDevice, lworkOnHost) &
       result(DnXlarft_bufferSize) &
       bind(C, name="hipsolverDnXlarft_bufferSize")
       import :: c_ptr, HIPSOLVER_DIRECT_FORWARD, HIPSOLVER_STOREV_COLUMNWISE, c_int64_t, c_int, &
                 c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_DIRECT_FORWARD)), value :: myDirect
       integer(kind(HIPSOLVER_STOREV_COLUMNWISE)), value :: storev
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: k
       integer(c_int), value :: dataTypeV
       type(c_ptr), value :: V
       integer(c_int64_t), value :: ldv
       integer(c_int), value :: dataTypeTau
       type(c_ptr), value :: tau
       integer(c_int), value :: dataTypeT
       type(c_ptr), value :: T
       integer(c_int64_t), value :: ldt
       integer(c_int), value :: computeType
       integer(c_size_t) :: lworkOnDevice
       integer(c_size_t) :: lworkOnHost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXlarft_bufferSize
    end function hipsolverDnXlarft_bufferSize

    !---------------------------------------------
    ! hipsolverDnXlarft
    !---------------------------------------------
    function hipsolverDnXlarft(handle, params, myDirect, storev, n, k, dataTypeV, V, ldv, &
                               dataTypeTau, tau, dataTypeT, T, ldt, computeType, workOnDevice, &
                               lworkOnDevice, workOnHost, lworkOnHost) &
       result(DnXlarft) &
       bind(C, name="hipsolverDnXlarft")
       import :: c_ptr, HIPSOLVER_DIRECT_FORWARD, HIPSOLVER_STOREV_COLUMNWISE, c_int64_t, c_int, &
                 c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_DIRECT_FORWARD)), value :: myDirect
       integer(kind(HIPSOLVER_STOREV_COLUMNWISE)), value :: storev
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: k
       integer(c_int), value :: dataTypeV
       type(c_ptr), value :: V
       integer(c_int64_t), value :: ldv
       integer(c_int), value :: dataTypeTau
       type(c_ptr), value :: tau
       integer(c_int), value :: dataTypeT
       type(c_ptr), value :: T
       integer(c_int64_t), value :: ldt
       integer(c_int), value :: computeType
       type(c_ptr), value :: workOnDevice
       integer(c_size_t), value :: lworkOnDevice
       type(c_ptr), value :: workOnHost
       integer(c_size_t), value :: lworkOnHost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXlarft
    end function hipsolverDnXlarft

    !---------------------------------------------
    ! hipsolverDnXpotrf_bufferSize
    !---------------------------------------------
    function hipsolverDnXpotrf_bufferSize(handle, params, uplo, n, dataTypeA, A, lda, computeType, &
                                          lworkOnDevice, lworkOnHost) &
       result(DnXpotrf_bufferSize) &
       bind(C, name="hipsolverDnXpotrf_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, c_size_t, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: computeType
       integer(c_size_t) :: lworkOnDevice
       integer(c_size_t) :: lworkOnHost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXpotrf_bufferSize
    end function hipsolverDnXpotrf_bufferSize

    !---------------------------------------------
    ! hipsolverDnXpotrf
    !---------------------------------------------
    function hipsolverDnXpotrf(handle, params, uplo, n, dataTypeA, A, lda, computeType, &
                               workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, myInfo) &
       result(DnXpotrf) &
       bind(C, name="hipsolverDnXpotrf")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, c_size_t, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: computeType
       type(c_ptr), value :: workOnDevice
       integer(c_size_t), value :: lworkOnDevice
       type(c_ptr), value :: workOnHost
       integer(c_size_t), value :: lworkOnHost
       type(c_ptr), value :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXpotrf
    end function hipsolverDnXpotrf

    !---------------------------------------------
    ! hipsolverDnXpotrs
    !---------------------------------------------
    function hipsolverDnXpotrs(handle, params, uplo, n, nrhs, dataTypeA, A, lda, dataTypeB, B, &
                               ldb, myInfo) &
       result(DnXpotrs) &
       bind(C, name="hipsolverDnXpotrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeB
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       type(c_ptr), value :: myInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXpotrs
    end function hipsolverDnXpotrs

    !---------------------------------------------
    ! hipsolverDnXsyevd_bufferSize
    !---------------------------------------------
    function hipsolverDnXsyevd_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, &
                                          dataTypeW, W, computeType, lworkOnDevice, lworkOnHost) &
       result(DnXsyevd_bufferSize) &
       bind(C, name="hipsolverDnXsyevd_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, &
                 c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeW
       type(c_ptr), value :: W
       integer(c_int), value :: computeType
       integer(c_size_t) :: lworkOnDevice
       integer(c_size_t) :: lworkOnHost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevd_bufferSize
    end function hipsolverDnXsyevd_bufferSize

    !---------------------------------------------
    ! hipsolverDnXsyevd
    !---------------------------------------------
    function hipsolverDnXsyevd(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, W, &
                               computeType, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, &
                               devInfo) &
       result(DnXsyevd) &
       bind(C, name="hipsolverDnXsyevd")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, &
                 c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeW
       type(c_ptr), value :: W
       integer(c_int), value :: computeType
       type(c_ptr), value :: workOnDevice
       integer(c_size_t), value :: lworkOnDevice
       type(c_ptr), value :: workOnHost
       integer(c_size_t), value :: lworkOnHost
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevd
    end function hipsolverDnXsyevd

    !---------------------------------------------
    ! hipsolverDnXsyevBatched_bufferSize
    !---------------------------------------------
    function hipsolverDnXsyevBatched_bufferSize(handle, params, jobz, uplo, n, dataTypeA, A, lda, &
                                                dataTypeW, W, computeType, lworkOnDevice, &
                                                lworkOnHost, batchSize) &
       result(DnXsyevBatched_bufferSize) &
       bind(C, name="hipsolverDnXsyevBatched_bufferSize")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, &
                 c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeW
       type(c_ptr), value :: W
       integer(c_int), value :: computeType
       integer(c_size_t) :: lworkOnDevice
       integer(c_size_t) :: lworkOnHost
       integer(c_int64_t), value :: batchSize
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevBatched_bufferSize
    end function hipsolverDnXsyevBatched_bufferSize

    !---------------------------------------------
    ! hipsolverDnXsyevBatched
    !---------------------------------------------
    function hipsolverDnXsyevBatched(handle, params, jobz, uplo, n, dataTypeA, A, lda, dataTypeW, &
                                     W, computeType, workOnDevice, lworkOnDevice, workOnHost, &
                                     lworkOnHost, devInfo, batchSize) &
       result(DnXsyevBatched) &
       bind(C, name="hipsolverDnXsyevBatched")
       import :: c_ptr, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, &
                 c_size_t, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: params
       integer(kind(HIPSOLVER_EIG_MODE_NOVECTOR)), value :: jobz
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       integer(c_int), value :: dataTypeW
       type(c_ptr), value :: W
       integer(c_int), value :: computeType
       type(c_ptr), value :: workOnDevice
       integer(c_size_t), value :: lworkOnDevice
       type(c_ptr), value :: workOnHost
       integer(c_size_t), value :: lworkOnHost
       type(c_ptr), value :: devInfo
       integer(c_int64_t), value :: batchSize
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsyevBatched
    end function hipsolverDnXsyevBatched

    !---------------------------------------------
    ! hipsolverDnXsytrs_bufferSize
    !---------------------------------------------
    function hipsolverDnXsytrs_bufferSize(handle, uplo, n, nrhs, dataTypeA, A, lda, devIpiv, &
                                          dataTypeB, B, ldb, lworkOnDevice, lworkOnHost) &
       result(DnXsytrs_bufferSize) &
       bind(C, name="hipsolverDnXsytrs_bufferSize")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, c_size_t, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: devIpiv
       integer(c_int), value :: dataTypeB
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       integer(c_size_t) :: lworkOnDevice
       integer(c_size_t) :: lworkOnHost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsytrs_bufferSize
    end function hipsolverDnXsytrs_bufferSize

    !---------------------------------------------
    ! hipsolverDnXsytrs
    !---------------------------------------------
    function hipsolverDnXsytrs(handle, uplo, n, nrhs, dataTypeA, A, lda, devIpiv, dataTypeB, B, &
                               ldb, workOnDevice, lworkOnDevice, workOnHost, lworkOnHost, devInfo) &
       result(DnXsytrs) &
       bind(C, name="hipsolverDnXsytrs")
       import :: c_ptr, HIPSOLVER_FILL_MODE_UPPER, c_int64_t, c_int, c_size_t, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_FILL_MODE_UPPER)), value :: uplo
       integer(c_int64_t), value :: n
       integer(c_int64_t), value :: nrhs
       integer(c_int), value :: dataTypeA
       type(c_ptr), value :: A
       integer(c_int64_t), value :: lda
       type(c_ptr), value :: devIpiv
       integer(c_int), value :: dataTypeB
       type(c_ptr), value :: B
       integer(c_int64_t), value :: ldb
       type(c_ptr), value :: workOnDevice
       integer(c_size_t), value :: lworkOnDevice
       type(c_ptr), value :: workOnHost
       integer(c_size_t), value :: lworkOnHost
       type(c_ptr), value :: devInfo
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: DnXsytrs
    end function hipsolverDnXsytrs

    !---------------------------------------------
    ! hipsolverRfCreate
    !---------------------------------------------
    function hipsolverRfCreate(handle) &
       result(RfCreate) &
       bind(C, name="hipsolverRfCreate")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfCreate
    end function hipsolverRfCreate

    !---------------------------------------------
    ! hipsolverRfDestroy
    !---------------------------------------------
    function hipsolverRfDestroy(handle) &
       result(RfDestroy) &
       bind(C, name="hipsolverRfDestroy")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfDestroy
    end function hipsolverRfDestroy

    !---------------------------------------------
    ! hipsolverRfSetupDevice
    !---------------------------------------------
    function hipsolverRfSetupDevice(n, nnzA, csrRowPtrA, csrColIndA, csrValA, nnzL, csrRowPtrL, &
                                    csrColIndL, csrValL, nnzU, csrRowPtrU, csrColIndU, csrValU, P, &
                                    Q, handle) &
       result(RfSetupDevice) &
       bind(C, name="hipsolverRfSetupDevice")
       import :: c_int, c_ptr, HIPSOLVER_STATUS_SUCCESS
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: csrRowPtrA
       type(c_ptr), value :: csrColIndA
       type(c_ptr), value :: csrValA
       integer(c_int), value :: nnzL
       type(c_ptr), value :: csrRowPtrL
       type(c_ptr), value :: csrColIndL
       type(c_ptr), value :: csrValL
       integer(c_int), value :: nnzU
       type(c_ptr), value :: csrRowPtrU
       type(c_ptr), value :: csrColIndU
       type(c_ptr), value :: csrValU
       type(c_ptr), value :: P
       type(c_ptr), value :: Q
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfSetupDevice
    end function hipsolverRfSetupDevice

    !---------------------------------------------
    ! hipsolverRfSetupHost
    !---------------------------------------------
    function hipsolverRfSetupHost(n, nnzA, h_csrRowPtrA, h_csrColIndA, h_csrValA, nnzL, &
                                  h_csrRowPtrL, h_csrColIndL, h_csrValL, nnzU, h_csrRowPtrU, &
                                  h_csrColIndU, h_csrValU, h_P, h_Q, handle) &
       result(RfSetupHost) &
       bind(C, name="hipsolverRfSetupHost")
       import :: c_int, c_ptr, HIPSOLVER_STATUS_SUCCESS
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: h_csrRowPtrA
       type(c_ptr), value :: h_csrColIndA
       type(c_ptr), value :: h_csrValA
       integer(c_int), value :: nnzL
       type(c_ptr), value :: h_csrRowPtrL
       type(c_ptr), value :: h_csrColIndL
       type(c_ptr), value :: h_csrValL
       integer(c_int), value :: nnzU
       type(c_ptr), value :: h_csrRowPtrU
       type(c_ptr), value :: h_csrColIndU
       type(c_ptr), value :: h_csrValU
       type(c_ptr), value :: h_P
       type(c_ptr), value :: h_Q
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfSetupHost
    end function hipsolverRfSetupHost

    !---------------------------------------------
    ! hipsolverRfAccessBundledFactorsDevice
    !---------------------------------------------
    function hipsolverRfAccessBundledFactorsDevice(handle, nnzM, Mp, Mi, Mx) &
       result(RfAccessBundledFactorsDevice) &
       bind(C, name="hipsolverRfAccessBundledFactorsDevice")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int) :: nnzM
       type(c_ptr) :: Mp
       type(c_ptr) :: Mi
       type(c_ptr) :: Mx
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfAccessBundledFactorsDevice
    end function hipsolverRfAccessBundledFactorsDevice

    !---------------------------------------------
    ! hipsolverRfAnalyze
    !---------------------------------------------
    function hipsolverRfAnalyze(handle) &
       result(RfAnalyze) &
       bind(C, name="hipsolverRfAnalyze")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfAnalyze
    end function hipsolverRfAnalyze

    !---------------------------------------------
    ! hipsolverRfExtractBundledFactorsHost
    !---------------------------------------------
    function hipsolverRfExtractBundledFactorsHost(handle, h_nnzM, h_Mp, h_Mi, h_Mx) &
       result(RfExtractBundledFactorsHost) &
       bind(C, name="hipsolverRfExtractBundledFactorsHost")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int) :: h_nnzM
       type(c_ptr) :: h_Mp
       type(c_ptr) :: h_Mi
       type(c_ptr) :: h_Mx
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfExtractBundledFactorsHost
    end function hipsolverRfExtractBundledFactorsHost

    !---------------------------------------------
    ! hipsolverRfExtractSplitFactorsHost
    !---------------------------------------------
    function hipsolverRfExtractSplitFactorsHost(handle, h_nnzL, h_Lp, h_Li, h_Lx, h_nnzU, h_Up, &
                                                h_Ui, h_Ux) &
       result(RfExtractSplitFactorsHost) &
       bind(C, name="hipsolverRfExtractSplitFactorsHost")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int) :: h_nnzL
       type(c_ptr) :: h_Lp
       type(c_ptr) :: h_Li
       type(c_ptr) :: h_Lx
       integer(c_int) :: h_nnzU
       type(c_ptr) :: h_Up
       type(c_ptr) :: h_Ui
       type(c_ptr) :: h_Ux
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfExtractSplitFactorsHost
    end function hipsolverRfExtractSplitFactorsHost

    !---------------------------------------------
    ! hipsolverRfGet_Algs
    !---------------------------------------------
    function hipsolverRfGet_Algs(handle, fact_alg, solve_alg) &
       result(RfGet_Algs) &
       bind(C, name="hipsolverRfGet_Algs")
       import :: c_ptr, HIPSOLVERRF_FACTORIZATION_ALG0, HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVERRF_FACTORIZATION_ALG0)) :: fact_alg
       integer(kind(HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1)) :: solve_alg
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfGet_Algs
    end function hipsolverRfGet_Algs

    !---------------------------------------------
    ! hipsolverRfGetMatrixFormat
    !---------------------------------------------
    function hipsolverRfGetMatrixFormat(handle, myFormat, diag) &
       result(RfGetMatrixFormat) &
       bind(C, name="hipsolverRfGetMatrixFormat")
       import :: c_ptr, HIPSOLVERRF_MATRIX_FORMAT_CSR, HIPSOLVERRF_UNIT_DIAGONAL_STORED_L, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVERRF_MATRIX_FORMAT_CSR)) :: myFormat
       integer(kind(HIPSOLVERRF_UNIT_DIAGONAL_STORED_L)) :: diag
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfGetMatrixFormat
    end function hipsolverRfGetMatrixFormat

    !---------------------------------------------
    ! hipsolverRfGetNumericBoostReport
    !---------------------------------------------
    function hipsolverRfGetNumericBoostReport(handle, report) &
       result(RfGetNumericBoostReport) &
       bind(C, name="hipsolverRfGetNumericBoostReport")
       import :: c_ptr, HIPSOLVERRF_NUMERIC_BOOST_NOT_USED, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVERRF_NUMERIC_BOOST_NOT_USED)) :: report
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfGetNumericBoostReport
    end function hipsolverRfGetNumericBoostReport

    !---------------------------------------------
    ! hipsolverRfGetNumericProperties
    !---------------------------------------------
    function hipsolverRfGetNumericProperties(handle, zero, boost) &
       result(RfGetNumericProperties) &
       bind(C, name="hipsolverRfGetNumericProperties")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       real(c_double) :: zero
       real(c_double) :: boost
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfGetNumericProperties
    end function hipsolverRfGetNumericProperties

    !---------------------------------------------
    ! hipsolverRfGetResetValuesFastMode
    !---------------------------------------------
    function hipsolverRfGetResetValuesFastMode(handle, fastMode) &
       result(RfGetResetValuesFastMode) &
       bind(C, name="hipsolverRfGetResetValuesFastMode")
       import :: c_ptr, HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF)) :: fastMode
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfGetResetValuesFastMode
    end function hipsolverRfGetResetValuesFastMode

    !---------------------------------------------
    ! hipsolverRfRefactor
    !---------------------------------------------
    function hipsolverRfRefactor(handle) &
       result(RfRefactor) &
       bind(C, name="hipsolverRfRefactor")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfRefactor
    end function hipsolverRfRefactor

    !---------------------------------------------
    ! hipsolverRfResetValues
    !---------------------------------------------
    function hipsolverRfResetValues(n, nnzA, csrRowPtrA, csrColIndA, csrValA, P, Q, handle) &
       result(RfResetValues) &
       bind(C, name="hipsolverRfResetValues")
       import :: c_int, c_ptr, HIPSOLVER_STATUS_SUCCESS
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: csrRowPtrA
       type(c_ptr), value :: csrColIndA
       type(c_ptr), value :: csrValA
       type(c_ptr), value :: P
       type(c_ptr), value :: Q
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfResetValues
    end function hipsolverRfResetValues

    !---------------------------------------------
    ! hipsolverRfSetAlgs
    !---------------------------------------------
    function hipsolverRfSetAlgs(handle, fact_alg, solve_alg) &
       result(RfSetAlgs) &
       bind(C, name="hipsolverRfSetAlgs")
       import :: c_ptr, HIPSOLVERRF_FACTORIZATION_ALG0, HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVERRF_FACTORIZATION_ALG0)), value :: fact_alg
       integer(kind(HIPSOLVERRF_TRIANGULAR_SOLVE_ALG1)), value :: solve_alg
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfSetAlgs
    end function hipsolverRfSetAlgs

    !---------------------------------------------
    ! hipsolverRfSetMatrixFormat
    !---------------------------------------------
    function hipsolverRfSetMatrixFormat(handle, myFormat, diag) &
       result(RfSetMatrixFormat) &
       bind(C, name="hipsolverRfSetMatrixFormat")
       import :: c_ptr, HIPSOLVERRF_MATRIX_FORMAT_CSR, HIPSOLVERRF_UNIT_DIAGONAL_STORED_L, &
                 HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVERRF_MATRIX_FORMAT_CSR)), value :: myFormat
       integer(kind(HIPSOLVERRF_UNIT_DIAGONAL_STORED_L)), value :: diag
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfSetMatrixFormat
    end function hipsolverRfSetMatrixFormat

    !---------------------------------------------
    ! hipsolverRfSetNumericProperties
    !---------------------------------------------
    function hipsolverRfSetNumericProperties(handle, effective_zero, boost_val) &
       result(RfSetNumericProperties) &
       bind(C, name="hipsolverRfSetNumericProperties")
       import :: c_ptr, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       real(c_double), value :: effective_zero
       real(c_double), value :: boost_val
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfSetNumericProperties
    end function hipsolverRfSetNumericProperties

    !---------------------------------------------
    ! hipsolverRfSetResetValuesFastMode
    !---------------------------------------------
    function hipsolverRfSetResetValuesFastMode(handle, fastMode) &
       result(RfSetResetValuesFastMode) &
       bind(C, name="hipsolverRfSetResetValuesFastMode")
       import :: c_ptr, HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVERRF_RESET_VALUES_FAST_MODE_OFF)), value :: fastMode
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfSetResetValuesFastMode
    end function hipsolverRfSetResetValuesFastMode

    !---------------------------------------------
    ! hipsolverRfSolve
    !---------------------------------------------
    function hipsolverRfSolve(handle, P, Q, nrhs, Temp, ldt, XF, ldxf) &
       result(RfSolve) &
       bind(C, name="hipsolverRfSolve")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: P
       type(c_ptr), value :: Q
       integer(c_int), value :: nrhs
       type(c_ptr), value :: Temp
       integer(c_int), value :: ldt
       type(c_ptr), value :: XF
       integer(c_int), value :: ldxf
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfSolve
    end function hipsolverRfSolve

    !---------------------------------------------
    ! hipsolverRfBatchSetupHost
    !---------------------------------------------
    function hipsolverRfBatchSetupHost(batchSize, n, nnzA, h_csrRowPtrA, h_csrColIndA, &
                                       h_csrValA_array, nnzL, h_csrRowPtrL, h_csrColIndL, &
                                       h_csrValL, nnzU, h_csrRowPtrU, h_csrColIndU, h_csrValU, &
                                       h_P, h_Q, handle) &
       result(RfBatchSetupHost) &
       bind(C, name="hipsolverRfBatchSetupHost")
       import :: c_int, c_ptr, HIPSOLVER_STATUS_SUCCESS
       integer(c_int), value :: batchSize
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: h_csrRowPtrA
       type(c_ptr), value :: h_csrColIndA
       type(c_ptr), value :: h_csrValA_array
       integer(c_int), value :: nnzL
       type(c_ptr), value :: h_csrRowPtrL
       type(c_ptr), value :: h_csrColIndL
       type(c_ptr), value :: h_csrValL
       integer(c_int), value :: nnzU
       type(c_ptr), value :: h_csrRowPtrU
       type(c_ptr), value :: h_csrColIndU
       type(c_ptr), value :: h_csrValU
       type(c_ptr), value :: h_P
       type(c_ptr), value :: h_Q
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfBatchSetupHost
    end function hipsolverRfBatchSetupHost

    !---------------------------------------------
    ! hipsolverRfBatchAnalyze
    !---------------------------------------------
    function hipsolverRfBatchAnalyze(handle) &
       result(RfBatchAnalyze) &
       bind(C, name="hipsolverRfBatchAnalyze")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfBatchAnalyze
    end function hipsolverRfBatchAnalyze

    !---------------------------------------------
    ! hipsolverRfBatchRefactor
    !---------------------------------------------
    function hipsolverRfBatchRefactor(handle) &
       result(RfBatchRefactor) &
       bind(C, name="hipsolverRfBatchRefactor")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfBatchRefactor
    end function hipsolverRfBatchRefactor

    !---------------------------------------------
    ! hipsolverRfBatchResetValues
    !---------------------------------------------
    function hipsolverRfBatchResetValues(batchSize, n, nnzA, csrRowPtrA, csrColIndA, &
                                         csrValA_array, P, Q, handle) &
       result(RfBatchResetValues) &
       bind(C, name="hipsolverRfBatchResetValues")
       import :: c_int, c_ptr, HIPSOLVER_STATUS_SUCCESS
       integer(c_int), value :: batchSize
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: csrRowPtrA
       type(c_ptr), value :: csrColIndA
       type(c_ptr), value :: csrValA_array
       type(c_ptr), value :: P
       type(c_ptr), value :: Q
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfBatchResetValues
    end function hipsolverRfBatchResetValues

    !---------------------------------------------
    ! hipsolverRfBatchSolve
    !---------------------------------------------
    function hipsolverRfBatchSolve(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf) &
       result(RfBatchSolve) &
       bind(C, name="hipsolverRfBatchSolve")
       import :: c_ptr, c_int, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: P
       type(c_ptr), value :: Q
       integer(c_int), value :: nrhs
       type(c_ptr), value :: Temp
       integer(c_int), value :: ldt
       type(c_ptr), value :: XF_array
       integer(c_int), value :: ldxf
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfBatchSolve
    end function hipsolverRfBatchSolve

    !---------------------------------------------
    ! hipsolverRfBatchZeroPivot
    !---------------------------------------------
    function hipsolverRfBatchZeroPivot(handle, position) &
       result(RfBatchZeroPivot) &
       bind(C, name="hipsolverRfBatchZeroPivot")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: position
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: RfBatchZeroPivot
    end function hipsolverRfBatchZeroPivot

    !---------------------------------------------
    ! hipsolverSpCreate
    !---------------------------------------------
    function hipsolverSpCreate(handle) &
       result(SpCreate) &
       bind(C, name="hipsolverSpCreate")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr) :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpCreate
    end function hipsolverSpCreate

    !---------------------------------------------
    ! hipsolverSpDestroy
    !---------------------------------------------
    function hipsolverSpDestroy(handle) &
       result(SpDestroy) &
       bind(C, name="hipsolverSpDestroy")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpDestroy
    end function hipsolverSpDestroy

    !---------------------------------------------
    ! hipsolverSpSetStream
    !---------------------------------------------
    function hipsolverSpSetStream(handle, streamId) &
       result(SpSetStream) &
       bind(C, name="hipsolverSpSetStream")
       import :: c_ptr, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       type(c_ptr), value :: streamId
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpSetStream
    end function hipsolverSpSetStream

    !---------------------------------------------
    ! hipsolverSpScsrlsvchol
    !---------------------------------------------
    function hipsolverSpScsrlsvchol(handle, n, nnzA, descrA, csrVal, csrRowPtr, csrColInd, b, &
                                    tolerance, reorder, x, singularity) &
       result(SpScsrlsvchol) &
       bind(C, name="hipsolverSpScsrlsvchol")
       import :: c_ptr, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: descrA
       type(c_ptr), value :: csrVal
       type(c_ptr), value :: csrRowPtr
       type(c_ptr), value :: csrColInd
       type(c_ptr), value :: b
       real(c_float), value :: tolerance
       integer(c_int), value :: reorder
       type(c_ptr), value :: x
       integer(c_int) :: singularity
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpScsrlsvchol
    end function hipsolverSpScsrlsvchol

    !---------------------------------------------
    ! hipsolverSpDcsrlsvchol
    !---------------------------------------------
    function hipsolverSpDcsrlsvchol(handle, n, nnzA, descrA, csrVal, csrRowPtr, csrColInd, b, &
                                    tolerance, reorder, x, singularity) &
       result(SpDcsrlsvchol) &
       bind(C, name="hipsolverSpDcsrlsvchol")
       import :: c_ptr, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: descrA
       type(c_ptr), value :: csrVal
       type(c_ptr), value :: csrRowPtr
       type(c_ptr), value :: csrColInd
       type(c_ptr), value :: b
       real(c_double), value :: tolerance
       integer(c_int), value :: reorder
       type(c_ptr), value :: x
       integer(c_int) :: singularity
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpDcsrlsvchol
    end function hipsolverSpDcsrlsvchol

    !---------------------------------------------
    ! hipsolverSpScsrlsvcholHost
    !---------------------------------------------
    function hipsolverSpScsrlsvcholHost(handle, n, nnzA, descrA, csrVal, csrRowPtr, csrColInd, b, &
                                        tolerance, reorder, x, singularity) &
       result(SpScsrlsvcholHost) &
       bind(C, name="hipsolverSpScsrlsvcholHost")
       import :: c_ptr, c_int, c_float, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: descrA
       type(c_ptr), value :: csrVal
       type(c_ptr), value :: csrRowPtr
       type(c_ptr), value :: csrColInd
       type(c_ptr), value :: b
       real(c_float), value :: tolerance
       integer(c_int), value :: reorder
       type(c_ptr), value :: x
       integer(c_int) :: singularity
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpScsrlsvcholHost
    end function hipsolverSpScsrlsvcholHost

    !---------------------------------------------
    ! hipsolverSpDcsrlsvcholHost
    !---------------------------------------------
    function hipsolverSpDcsrlsvcholHost(handle, n, nnzA, descrA, csrVal, csrRowPtr, csrColInd, b, &
                                        tolerance, reorder, x, singularity) &
       result(SpDcsrlsvcholHost) &
       bind(C, name="hipsolverSpDcsrlsvcholHost")
       import :: c_ptr, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnzA
       type(c_ptr), value :: descrA
       type(c_ptr), value :: csrVal
       type(c_ptr), value :: csrRowPtr
       type(c_ptr), value :: csrColInd
       type(c_ptr), value :: b
       real(c_double), value :: tolerance
       integer(c_int), value :: reorder
       type(c_ptr), value :: x
       integer(c_int) :: singularity
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpDcsrlsvcholHost
    end function hipsolverSpDcsrlsvcholHost

    !---------------------------------------------
    ! hipsolverSpScsrlsvqr
    !---------------------------------------------
    function hipsolverSpScsrlsvqr(handle, n, nnz, descrA, csrVal, csrRowPts, csrColInd, b, &
                                  tolerance, reorder, x, singularity) &
       result(SpScsrlsvqr) &
       bind(C, name="hipsolverSpScsrlsvqr")
       import :: c_ptr, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnz
       type(c_ptr), value :: descrA
       type(c_ptr), value :: csrVal
       type(c_ptr), value :: csrRowPts
       type(c_ptr), value :: csrColInd
       type(c_ptr), value :: b
       real(c_double), value :: tolerance
       integer(c_int), value :: reorder
       type(c_ptr), value :: x
       integer(c_int) :: singularity
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpScsrlsvqr
    end function hipsolverSpScsrlsvqr

    !---------------------------------------------
    ! hipsolverSpDcsrlsvqr
    !---------------------------------------------
    function hipsolverSpDcsrlsvqr(handle, n, nnz, descrA, csrVal, csrRowPts, csrColInd, b, &
                                  tolerance, reorder, x, singularity) &
       result(SpDcsrlsvqr) &
       bind(C, name="hipsolverSpDcsrlsvqr")
       import :: c_ptr, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnz
       type(c_ptr), value :: descrA
       type(c_ptr), value :: csrVal
       type(c_ptr), value :: csrRowPts
       type(c_ptr), value :: csrColInd
       type(c_ptr), value :: b
       real(c_double), value :: tolerance
       integer(c_int), value :: reorder
       type(c_ptr), value :: x
       integer(c_int) :: singularity
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpDcsrlsvqr
    end function hipsolverSpDcsrlsvqr

    !---------------------------------------------
    ! hipsolverSpCcsrlsvqr
    !---------------------------------------------
    function hipsolverSpCcsrlsvqr(handle, n, nnz, descrA, csrVal, csrRowPts, csrColInd, b, &
                                  tolerance, reorder, x, singularity) &
       result(SpCcsrlsvqr) &
       bind(C, name="hipsolverSpCcsrlsvqr")
       import :: c_ptr, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnz
       type(c_ptr), value :: descrA
       type(c_ptr), value :: csrVal
       type(c_ptr), value :: csrRowPts
       type(c_ptr), value :: csrColInd
       type(c_ptr), value :: b
       real(c_double), value :: tolerance
       integer(c_int), value :: reorder
       type(c_ptr), value :: x
       integer(c_int) :: singularity
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpCcsrlsvqr
    end function hipsolverSpCcsrlsvqr

    !---------------------------------------------
    ! hipsolverSpZcsrlsvqr
    !---------------------------------------------
    function hipsolverSpZcsrlsvqr(handle, n, nnz, descrA, csrVal, csrRowPts, csrColInd, b, &
                                  tolerance, reorder, x, singularity) &
       result(SpZcsrlsvqr) &
       bind(C, name="hipsolverSpZcsrlsvqr")
       import :: c_ptr, c_int, c_double, HIPSOLVER_STATUS_SUCCESS
       type(c_ptr), value :: handle
       integer(c_int), value :: n
       integer(c_int), value :: nnz
       type(c_ptr), value :: descrA
       type(c_ptr), value :: csrVal
       type(c_ptr), value :: csrRowPts
       type(c_ptr), value :: csrColInd
       type(c_ptr), value :: b
       real(c_double), value :: tolerance
       integer(c_int), value :: reorder
       type(c_ptr), value :: x
       integer(c_int) :: singularity
       integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: SpZcsrlsvqr
    end function hipsolverSpZcsrlsvqr

  end interface


  contains

    subroutine hipsolverCheck(status)
      implicit none
      integer(kind(HIPSOLVER_STATUS_SUCCESS)) :: status
      if (status /= HIPSOLVER_STATUS_SUCCESS) then
        write (*, *) "HIPSOLVER ERROR: code = ", status
        stop 1
      end if
    end subroutine hipsolverCheck
end module hipsolver
