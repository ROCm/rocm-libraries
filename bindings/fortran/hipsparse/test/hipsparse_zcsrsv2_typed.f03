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
! hipsparse Zcsrsv2 example (sparse triangular solve v2, double complex, Fortran 2003)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Solves the lower-triangular system L*x = alpha*f for x using the legacy
! csrsv2 API (bufferSize -> analysis -> solve, with a mat descriptor and a
! csrsv2Info object). The right-hand side f is built on the host from a known
! complex solution (f = L*x) so the recovered x can be checked directly.
!
!   L (lower, CSR) has entries: (1,1), (2,1),(2,2), (3,1),(3,2),(3,3)
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
! The same check as hipsparse_zcsrsv2.f03, through the _typed form: alpha is passed as a
! Fortran variable, here a named constant, rather than with c_loc(), so the
! generic resolves to the _typed module procedure, which hands the C API its
! address. The handle is in host pointer mode, as that form requires.
program hipsparse_zcsrsv2_typed
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 6
  integer(c_int), target :: hRowPtr(4) = (/0, 1, 3, 6/)
  integer(c_int), target :: hColInd(6) = (/0, 0, 1, 0, 1, 2/)
  complex(c_double_complex), target :: hVal(6) = (/ &
    (2.0d0,1.0d0), (1.0d0,0.0d0), (2.0d0,1.0d0), (3.0d0,0.0d0), (1.0d0,0.0d0), (2.0d0,1.0d0)/)
  complex(c_double_complex), target :: hF(3)
  complex(c_double_complex), target :: hX(3)
  complex(c_double_complex) :: hExp(3) = (/(1.0d0,1.0d0), (2.0d0,-1.0d0), (3.0d0,0.0d0)/)
  ! Not 1: a dropped or mis-read alpha must change the solution.
  complex(c_double_complex), parameter :: alpha = (2.0d0,1.0d0)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descrA = c_null_ptr
  type(c_ptr) :: info = c_null_ptr
  type(c_ptr) :: dRowPtr, dColInd, dVal, dF, dX, dBuf
  integer(c_int) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zcsrsv2_typed' (Fortran 2003 interfaces) - "

  ! f = L*x (row by row from the CSR structure above).
  hF(1) = hVal(1)*hExp(1)
  hF(2) = hVal(2)*hExp(1) + hVal(3)*hExp(2)
  hF(3) = hVal(4)*hExp(1) + hVal(5)*hExp(2) + hVal(6)*hExp(3)

  call hipCheck(hipMalloc(dRowPtr, int(m+1,c_size_t) * 4))
  call hipCheck(hipMalloc(dColInd, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dVal,    int(nnz,c_size_t) * 16))
  call hipCheck(hipMalloc(dF,      int(m,c_size_t)   * 16))
  call hipCheck(hipMalloc(dX,      int(m,c_size_t)   * 16))
  call hipCheck(hipMemcpy(dRowPtr, c_loc(hRowPtr(1)), int(m+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dColInd, c_loc(hColInd(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dVal,    c_loc(hVal(1)),    int(nnz,c_size_t) * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dF,      c_loc(hF(1)),      int(m,c_size_t)   * 16, hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateMatDescr(descrA))
  call hipsparseCheck(hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL))
  call hipsparseCheck(hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO))
  call hipsparseCheck(hipsparseSetMatFillMode(descrA, HIPSPARSE_FILL_MODE_LOWER))
  call hipsparseCheck(hipsparseSetMatDiagType(descrA, HIPSPARSE_DIAG_TYPE_NON_UNIT))
  call hipsparseCheck(hipsparseCreateCsrsv2Info(info))

  call hipsparseCheck(hipsparseZcsrsv2_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       m, nnz, descrA, dVal, dRowPtr, dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, int(max(bufSize,1),c_size_t)))
  call hipsparseCheck(hipsparseZcsrsv2_analysis(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       m, nnz, descrA, dVal, dRowPtr, dColInd, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipsparseCheck(hipsparseZcsrsv2_solve(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       m, nnz, alpha, descrA, dVal, dRowPtr, dColInd, info, dF, dX, &
       HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hX(1)), dX, int(m,c_size_t) * 16, hipMemcpyDeviceToHost))

  do i = 1, m
     if (.not. (abs(hX(i) - alpha*hExp(i)) <= 1.0d-11)) then
        write(*,*) "FAILED! x(", i, ") = ", hX(i), " expected ", alpha*hExp(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroyCsrsv2Info(info))
  call hipsparseCheck(hipsparseDestroyMatDescr(descrA))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dF)); call hipCheck(hipFree(dX)); call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program hipsparse_zcsrsv2_typed
