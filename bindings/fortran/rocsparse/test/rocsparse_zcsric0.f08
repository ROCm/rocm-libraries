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
! rocsparse zcsric0 example (incomplete Cholesky, double complex)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Computes the IC(0) factorization of a Hermitian positive definite sparse matrix
! in place (buffer_size -> analysis -> compute, with a mat descriptor and a mat
! info object). For an HPD tridiagonal matrix there is no fill-in, so IC(0)
! equals the exact Cholesky factor L (A = L*L^H). csric0 overwrites the
! lower-triangular part (including the diagonal) with L; the strict upper part is
! left unchanged. The lower/diagonal CSR entries are checked against the exact
! complex Cholesky factor computed on the host from the same values:
!
!   A = [ 4    1+i  0    ]   l11 = sqrt(a11), l21 = a21/l11, l22 = sqrt(a22 - |l21|^2)
!       [ 1-i  4    1-2i ]   l32 = a32/l22,   l33 = sqrt(a33 - |l32|^2)
!       [ 0    1+2i 5    ]
!!!!!!!!!!!!!!
!
program zcsric0
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 7
  integer(c_int) :: hRowPtr(4) = (/0, 2, 5, 7/)
  integer(c_int) :: hColInd(7) = (/0, 1, 0, 1, 2, 1, 2/)
  complex(c_double_complex) :: hVal(7) = (/ &
    (4.0d0,0.0d0), (1.0d0,1.0d0), (1.0d0,-1.0d0), (4.0d0,0.0d0), &
    (1.0d0,-2.0d0), (1.0d0,2.0d0), (5.0d0,0.0d0)/)
  complex(c_double_complex) :: hOut(7)
  complex(c_double_complex) :: L11, L21, L22, L32, L33
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descr = c_null_ptr, info = c_null_ptr
  integer(c_int), pointer :: dRowPtr(:), dColInd(:)
  complex(c_double_complex), pointer :: dVal(:)
  type(c_ptr) :: dBuf
  integer(c_size_t) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_zcsric0' (Fortran 2008 interfaces) - "

  L11 = sqrt(real(hVal(1)))
  L21 = hVal(3) / L11
  L22 = sqrt(real(hVal(4)) - abs(L21)**2)
  L32 = hVal(6) / L22
  L33 = sqrt(real(hVal(7)) - abs(L32)**2)

  call hipCheck(hipMalloc(dRowPtr, source=hRowPtr))
  call hipCheck(hipMalloc(dColInd, source=hColInd))
  call hipCheck(hipMalloc(dVal,    source=hVal))

  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr))
  call rocsparseCheck(rocsparse_create_mat_info(info))

  call rocsparseCheck(rocsparse_zcsric0_buffer_size(handle, m, nnz, descr, dVal, dRowPtr, &
       dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, max(bufSize, 1_c_size_t)))
  call rocsparseCheck(rocsparse_zcsric0_analysis(handle, m, nnz, descr, dVal, dRowPtr, dColInd, &
       info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, dBuf))
  call rocsparseCheck(rocsparse_zcsric0(handle, m, nnz, descr, dVal, dRowPtr, dColInd, &
       info, rocsparse_solve_policy_auto, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hOut, dVal, hipMemcpyDeviceToHost))

  if (.not. (abs(hOut(1) - L11) <= 1.0d-10 .and. abs(hOut(3) - L21) <= 1.0d-10 .and. &
      abs(hOut(4) - L22) <= 1.0d-10 .and. abs(hOut(6) - L32) <= 1.0d-10 .and. &
      abs(hOut(7) - L33) <= 1.0d-10)) then
     write(*,*) "FAILED! L = ", hOut(1), hOut(3), hOut(4), hOut(6), hOut(7), &
                " expected ", L11, L21, L22, L32, L33
     call exit(1)
  end if
  call rocsparseCheck(rocsparse_destroy_mat_info(info))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program zcsric0
