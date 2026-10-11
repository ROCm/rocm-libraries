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
! rocsparse scsrilu0 example (incomplete LU, single, Fortran 2003)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Computes the ILU(0) factorization of a sparse matrix in place
! (buffer_size -> analysis -> compute, with a mat descriptor and a mat info
! object). For a tridiagonal matrix there is no fill-in, so ILU(0) equals the
! exact LU factorization and the overwritten CSR values are checked against the
! hand-computed factors (unit lower L, upper U packed into one array).
!
!   A = [ 4 1 0 ]     LU (in place) = [ 4    1    0   ]
!       [ 1 4 1 ]                     [ 1/4  15/4 1   ]
!       [ 0 1 4 ]                     [ 0    4/15 56/15]
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program zcsrilu0
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 7
  integer(c_int), target :: hRowPtr(4) = (/0, 2, 5, 7/)
  integer(c_int), target :: hColInd(7) = (/0, 1, 0, 1, 2, 1, 2/)
  complex(c_double_complex), target :: hVal(7) = (/ &
    (4.0d0,0.0d0), (1.0d0,0.0d0), (1.0d0,0.0d0), (4.0d0,0.0d0), &
    (1.0d0,0.0d0), (1.0d0,0.0d0), (4.0d0,0.0d0)/)
  complex(c_double_complex), target :: hOut(7)
  complex(c_double_complex) :: hExp(7)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descr = c_null_ptr, info = c_null_ptr
  type(c_ptr) :: dRowPtr, dColInd, dVal, dBuf
  integer(c_size_t) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_zcsrilu0' (Fortran 2003 interfaces) - "

  hExp(1) = (4.0d0,0.0d0)
  hExp(2) = (1.0d0,0.0d0)
  hExp(3) = (1.0d0,0.0d0)/(4.0d0,0.0d0)
  hExp(4) = (4.0d0,0.0d0) - ((1.0d0,0.0d0)/(4.0d0,0.0d0))*(1.0d0,0.0d0)
  hExp(5) = (1.0d0,0.0d0)
  hExp(6) = (1.0d0,0.0d0)/hExp(4)
  hExp(7) = (4.0d0,0.0d0) - hExp(6)*(1.0d0,0.0d0)

  call hipCheck(hipMalloc(dRowPtr, int(m+1,c_size_t) * 4))
  call hipCheck(hipMalloc(dColInd, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dVal,    int(nnz,c_size_t) * 16))
  call hipCheck(hipMemcpy(dRowPtr, c_loc(hRowPtr(1)), int(m+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dColInd, c_loc(hColInd(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dVal,    c_loc(hVal(1)),    int(nnz,c_size_t) * 16, hipMemcpyHostToDevice))

  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr))
  call rocsparseCheck(rocsparse_create_mat_info(info))

  call rocsparseCheck(rocsparse_zcsrilu0_buffer_size(handle, m, nnz, descr, dVal, dRowPtr, &
       dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, max(bufSize, 1_c_size_t)))
  call rocsparseCheck(rocsparse_zcsrilu0_analysis(handle, m, nnz, descr, dVal, dRowPtr, dColInd, &
       info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, dBuf))
  call rocsparseCheck(rocsparse_zcsrilu0(handle, m, nnz, descr, dVal, dRowPtr, dColInd, &
       info, rocsparse_solve_policy_auto, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hOut(1)), dVal, int(nnz,c_size_t) * 16, hipMemcpyDeviceToHost))

  do i = 1, nnz
     if (.not. (abs(hOut(i) - hExp(i)) <= 1.0d-10)) then
        write(*,*) "FAILED! val(", i, ") = ", hOut(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_mat_info(info))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program zcsrilu0
