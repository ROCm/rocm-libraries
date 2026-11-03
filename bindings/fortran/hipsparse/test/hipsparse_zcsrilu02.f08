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
! hipsparse Zcsrilu02 example (incomplete LU, double complex)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Computes the ILU(0) factorization of a complex tridiagonal matrix in place
! using the legacy csrilu02 API (bufferSize -> analysis -> compute, with a mat
! descriptor and a csrilu02Info object). A tridiagonal matrix has no fill-in, so
! ILU(0) equals the exact LU factorization; the overwritten CSR values are
! checked against the factors computed on the host with the same recurrence.
!!!!!!!!!!!!!!
!
program hipsparse_zcsrilu02
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 7
  integer(c_int) :: hRowPtr(4) = (/0, 2, 5, 7/)
  integer(c_int) :: hColInd(7) = (/0, 1, 0, 1, 2, 1, 2/)
  complex(c_double_complex) :: hVal(7) = (/ &
    (4.0d0,1.0d0), (1.0d0,0.0d0), (1.0d0,0.0d0), (4.0d0,1.0d0), (1.0d0,0.0d0), (1.0d0,0.0d0), (4.0d0,1.0d0)/)
  complex(c_double_complex) :: hOut(7)
  complex(c_double_complex) :: hExp(7)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descrA = c_null_ptr
  type(c_ptr) :: info = c_null_ptr
  integer(c_int), pointer :: dRowPtr(:), dColInd(:)
  complex(c_double_complex), pointer :: dVal(:)
  type(c_ptr) :: dBuf
  integer(c_int) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zcsrilu02' (Fortran 2008 interfaces) - "

  ! Exact LU of the tridiagonal (no fill-in), stored in CSR order.
  hExp(1) = hVal(1)
  hExp(2) = hVal(2)
  hExp(3) = hVal(3)/hExp(1)
  hExp(4) = hVal(4) - hExp(3)*hExp(2)
  hExp(5) = hVal(5)
  hExp(6) = hVal(6)/hExp(4)
  hExp(7) = hVal(7) - hExp(6)*hExp(5)

  call hipCheck(hipMalloc(dRowPtr, source=hRowPtr))
  call hipCheck(hipMalloc(dColInd, source=hColInd))
  call hipCheck(hipMalloc(dVal,    source=hVal))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateMatDescr(descrA))
  call hipsparseCheck(hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL))
  call hipsparseCheck(hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO))
  call hipsparseCheck(hipsparseCreateCsrilu02Info(info))

  call hipsparseCheck(hipsparseZcsrilu02_bufferSize(handle, m, nnz, descrA, dVal, dRowPtr, &
       dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, int(max(bufSize,1),c_size_t)))
  call hipsparseCheck(hipsparseZcsrilu02_analysis(handle, m, nnz, descrA, dVal, dRowPtr, &
       dColInd, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipsparseCheck(hipsparseZcsrilu02(handle, m, nnz, descrA, dVal, dRowPtr, dColInd, &
       info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hOut, dVal, hipMemcpyDeviceToHost))

  do i = 1, nnz
     if (.not. (abs(hOut(i) - hExp(i)) <= 1.0d-11)) then
        write(*,*) "FAILED! val(", i, ") = ", hOut(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroyCsrilu02Info(info))
  call hipsparseCheck(hipsparseDestroyMatDescr(descrA))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program hipsparse_zcsrilu02
