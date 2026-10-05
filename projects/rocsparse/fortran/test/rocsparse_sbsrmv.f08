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
! rocsparse sbsrmv example (block-sparse matrix-vector multiply, single)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Computes y = alpha * A * x + beta * y for a BSR matrix A. Here A is
! block-diagonal with mb=nb=2 blocks of block_dim=2 (a 4x4 dense equivalent):
!   A = [ B1  0  ]   B1 = [1 2]   B2 = [5 6]
!       [ 0   B2 ]        [3 4]        [7 8]
! Blocks are stored row-major (rocsparse_direction_row). The result is checked
! against the dense reference y = A_dense * x.
!!!!!!!!!!!!!!
!
program sbsrmv
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: mb = 2, nb = 2, nnzb = 2, block_dim = 2
  integer(c_int), parameter :: mdim = mb * block_dim
  real(c_float) :: hVal(8) = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0/)
  integer(c_int) :: hRowPtr(3) = (/0, 1, 2/)
  integer(c_int) :: hColInd(2) = (/0, 1/)
  real(c_float) :: hX(4) = (/1.0, 2.0, 3.0, 4.0/)
  real(c_float) :: hY(4) = (/0.0, 0.0, 0.0, 0.0/)
  real(c_float) :: hRef(4)
  real(c_float), target :: alpha = 1.0, beta = 0.0
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descr = c_null_ptr, info = c_null_ptr
  real(c_float), pointer :: dVal(:), dX(:), dY(:)
  integer(c_int), pointer :: dRowPtr(:), dColInd(:)
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_sbsrmv' (Fortran 2008 interfaces) - "

  hRef(1) = 1.0*hX(1) + 2.0*hX(2)
  hRef(2) = 3.0*hX(1) + 4.0*hX(2)
  hRef(3) = 5.0*hX(3) + 6.0*hX(4)
  hRef(4) = 7.0*hX(3) + 8.0*hX(4)

  call hipCheck(hipMalloc(dVal,    source=hVal))
  call hipCheck(hipMalloc(dRowPtr, source=hRowPtr))
  call hipCheck(hipMalloc(dColInd, source=hColInd))
  call hipCheck(hipMalloc(dX,      source=hX))
  call hipCheck(hipMalloc(dY,      source=hY))

  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr))
  call rocsparseCheck(rocsparse_create_mat_info(info))
  call rocsparseCheck(rocsparse_sbsrmv(handle, rocsparse_direction_row, rocsparse_operation_none, &
       mb, nb, nnzb, alpha, descr, dVal, dRowPtr, dColInd, block_dim, info, dX, beta, dY))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hY, dY, hipMemcpyDeviceToHost))

  do i = 1, mdim
     if (.not. (abs(hY(i) - hRef(i)) <= 1.0e-5)) then
        write(*,*) "FAILED! y(", i, ") = ", hY(i), " expected ", hRef(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_mat_info(info))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dVal)); call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd))
  call hipCheck(hipFree(dX)); call hipCheck(hipFree(dY))
  write(*,*) "PASSED!"
end program sbsrmv
