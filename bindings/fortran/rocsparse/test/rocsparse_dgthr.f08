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
! rocsparse dgthr example (gather y[x_ind] -> x_val, double)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Gathers the entries of a dense vector y at the sparse index set x_ind into the
! packed vector x_val, then checks the gathered values.
!!!!!!!!!!!!!!
!
program dgthr
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: n = 5, nnz = 3
  real(c_double) :: hY(5) = (/10.0d0, 20.0d0, 30.0d0, 40.0d0, 50.0d0/)
  integer(c_int) :: hXind(3) = (/0, 2, 4/)
  real(c_double) :: hXval(3)
  real(c_double) :: hExp(3) = (/10.0d0, 30.0d0, 50.0d0/)
  type(c_ptr) :: handle = c_null_ptr
  real(c_double), pointer :: dY(:), dXval(:)
  integer(c_int), pointer :: dXind(:)
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_dgthr' (Fortran 2008 interfaces) - "
  call hipCheck(hipMalloc(dY,    source=hY))
  call hipCheck(hipMalloc(dXind, source=hXind))
  call hipCheck(hipMalloc(dXval, mold=hXval))
  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_dgthr(handle, nnz, dY, dXval, dXind, rocsparse_index_base_zero))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(hXval, dXval, hipMemcpyDeviceToHost))
  do i = 1, nnz
     if (.not. (abs(hXval(i) - hExp(i)) <= 1.0d-12)) then
        write(*,*) "FAILED! x_val(", i, ") = ", hXval(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dY)); call hipCheck(hipFree(dXind)); call hipCheck(hipFree(dXval))
  write(*,*) "PASSED!"
end program dgthr
