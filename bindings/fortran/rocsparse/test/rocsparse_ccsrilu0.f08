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
! rocsparse ccsrilu0 example (incomplete LU, single complex)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Computes the ILU(0) factorization of a sparse matrix in place
! (buffer_size -> analysis -> compute, with a mat descriptor and a mat info
! object). For a tridiagonal matrix there is no fill-in, so ILU(0) equals the
! exact LU factorization and the overwritten CSR values are checked against the
! factors (unit lower L, upper U packed into one array) computed on the host from
! the same values. A is complex, non-Hermitian and diagonally dominant:
!
!   A = [ 4+i  1-i   0    ]   l21 = a21/a11, u22 = a22 - l21*a12
!       [ 2+i  5-2i  1+2i ]   l32 = a32/u22, u33 = a33 - l32*a23
!       [ 0    1-i   6+i  ]
!!!!!!!!!!!!!!
!
program ccsrilu0
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 7
  integer(c_int) :: hRowPtr(4) = (/0, 2, 5, 7/)
  integer(c_int) :: hColInd(7) = (/0, 1, 0, 1, 2, 1, 2/)
  complex(c_float_complex) :: hVal(7) = (/ &
    (4.0,1.0), (1.0,-1.0), (2.0,1.0), (5.0,-2.0), &
    (1.0,2.0), (1.0,-1.0), (6.0,1.0)/)
  complex(c_float_complex) :: hOut(7)
  complex(c_float_complex) :: hExp(7)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descr = c_null_ptr, info = c_null_ptr
  integer(c_int), pointer :: dRowPtr(:), dColInd(:)
  complex(c_float_complex), pointer :: dVal(:)
  type(c_ptr) :: dBuf
  integer(c_size_t) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_ccsrilu0' (Fortran 2008 interfaces) - "

  hExp(1) = hVal(1)
  hExp(2) = hVal(2)
  hExp(3) = hVal(3) / hExp(1)
  hExp(4) = hVal(4) - hExp(3)*hExp(2)
  hExp(5) = hVal(5)
  hExp(6) = hVal(6) / hExp(4)
  hExp(7) = hVal(7) - hExp(6)*hExp(5)

  call hipCheck(hipMalloc(dRowPtr, source=hRowPtr))
  call hipCheck(hipMalloc(dColInd, source=hColInd))
  call hipCheck(hipMalloc(dVal,    source=hVal))

  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr))
  call rocsparseCheck(rocsparse_create_mat_info(info))

  call rocsparseCheck(rocsparse_ccsrilu0_buffer_size(handle, m, nnz, descr, dVal, dRowPtr, &
       dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, max(bufSize, 1_c_size_t)))
  call rocsparseCheck(rocsparse_ccsrilu0_analysis(handle, m, nnz, descr, dVal, dRowPtr, dColInd, &
       info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, dBuf))
  call rocsparseCheck(rocsparse_ccsrilu0(handle, m, nnz, descr, dVal, dRowPtr, dColInd, &
       info, rocsparse_solve_policy_auto, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hOut, dVal, hipMemcpyDeviceToHost))

  do i = 1, nnz
     if (.not. (abs(hOut(i) - hExp(i)) <= 1.0e-4)) then
        write(*,*) "FAILED! val(", i, ") = ", hOut(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_mat_info(info))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program ccsrilu0
