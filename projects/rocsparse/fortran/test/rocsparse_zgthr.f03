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
! rocsparse sgthr example (gather y[x_ind] -> x_val, single, Fortran 2003)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Gathers the entries of a dense vector y at the sparse index set x_ind into the
! packed vector x_val, then checks the gathered values.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program zgthr
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: n = 5, nnz = 3
  complex(c_double_complex), target :: hY(5) = (/(10.0d0,1.0d0), (20.0d0,2.0d0), (30.0d0,3.0d0), (40.0d0,4.0d0), (50.0d0,5.0d0)/)
  integer(c_int), target :: hXind(3) = (/0, 2, 4/)
  complex(c_double_complex), target :: hXval(3)
  complex(c_double_complex) :: hExp(3) = (/(10.0d0,1.0d0), (30.0d0,3.0d0), (50.0d0,5.0d0)/)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dY, dXind, dXval
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_zgthr' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(dY,    int(n,c_size_t)   * 16))
  call hipCheck(hipMalloc(dXind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dXval, int(nnz,c_size_t) * 16))
  call hipCheck(hipMemcpy(dY,    c_loc(hY(1)),    int(n,c_size_t)   * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dXind, c_loc(hXind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_zgthr(handle, nnz, dY, dXval, dXind, rocsparse_index_base_zero))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hXval(1)), dXval, int(nnz,c_size_t) * 16, hipMemcpyDeviceToHost))
  do i = 1, nnz
     if (.not. (abs(hXval(i) - hExp(i)) <= 1.0d-12)) then
        write(*,*) "FAILED! x_val(", i, ") = ", hXval(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dY)); call hipCheck(hipFree(dXind)); call hipCheck(hipFree(dXval))
  write(*,*) "PASSED!"
end program zgthr
