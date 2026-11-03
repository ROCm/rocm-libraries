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
! rocsparse ssctr example (scatter x_val -> y[x_ind], single)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Scatters the packed vector x_val into the dense vector y at the sparse index
! set x_ind, then checks the resulting dense vector.
!!!!!!!!!!!!!!
!
program ssctr
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: n = 5, nnz = 3
  real(c_float) :: hXval(3) = (/100.0, 200.0, 300.0/)
  integer(c_int) :: hXind(3) = (/0, 2, 4/)
  real(c_float) :: hY(5) = (/0.0, 0.0, 0.0, 0.0, 0.0/)
  real(c_float) :: hExp(5) = (/100.0, 0.0, 200.0, 0.0, 300.0/)
  type(c_ptr) :: handle = c_null_ptr
  real(c_float), pointer :: dXval(:), dY(:)
  integer(c_int), pointer :: dXind(:)
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_ssctr' (Fortran 2008 interfaces) - "
  call hipCheck(hipMalloc(dXval, source=hXval))
  call hipCheck(hipMalloc(dXind, source=hXind))
  call hipCheck(hipMalloc(dY,    source=hY))
  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_ssctr(handle, nnz, dXval, dXind, dY, rocsparse_index_base_zero))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hY, dY, hipMemcpyDeviceToHost))
  do i = 1, n
     if (.not. (abs(hY(i) - hExp(i)) <= 1.0e-6)) then
        write(*,*) "FAILED! y(", i, ") = ", hY(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dXval)); call hipCheck(hipFree(dXind)); call hipCheck(hipFree(dY))
  write(*,*) "PASSED!"
end program ssctr
