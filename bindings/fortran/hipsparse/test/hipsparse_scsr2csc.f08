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
! hipsparse scsr2csc example (single-precision CSR -> CSC / sparse transpose)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Converting A from CSR to CSC is the CSR of A**T. We check the resulting
! csc_col_ptr / csc_row_ind / csc_val against the known transpose. The hipSPARSE
! legacy csr2csc needs no external workspace.
!!!!!!!!!!!!!!
!
program hipsparse_scsr2csc
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  ! 3x3 CSR (0-based): row0:(0,0)=1,(0,2)=2  row1:(1,1)=3  row2:(2,0)=4,(2,2)=5
  integer(c_int), parameter :: M = 3, N = 3, nnz = 5
  integer(c_int) :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_float)  :: h_csr_val(5)     = (/1, 2, 3, 4, 5/)
  integer(c_int) :: h_exp_col_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_exp_row_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_float)  :: h_exp_val(5)     = (/1, 4, 3, 2, 5/)
  integer(c_int) :: h_csc_col_ptr(4), h_csc_row_ind(5)
  real(c_float)  :: h_csc_val(5)
  integer(c_int), pointer :: d_csr_row_ptr(:), d_csr_col_ind(:), d_csc_col_ptr(:), d_csc_row_ind(:)
  real(c_float),  pointer :: d_csr_val(:), d_csc_val(:)
  type(c_ptr) :: handle = c_null_ptr
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10 * epsilon(error_max)
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_scsr2csc' (Fortran 2008 interfaces) - "
  call hipCheck(hipMalloc(d_csr_row_ptr, source=h_csr_row_ptr))
  call hipCheck(hipMalloc(d_csr_col_ind, source=h_csr_col_ind))
  call hipCheck(hipMalloc(d_csr_val,     source=h_csr_val))
  call hipCheck(hipMalloc(d_csc_col_ptr, mold=h_csc_col_ptr))
  call hipCheck(hipMalloc(d_csc_row_ind, mold=h_csc_row_ind))
  call hipCheck(hipMalloc(d_csc_val,     mold=h_csc_val))
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseScsr2csc(handle, M, N, nnz, &
                         d_csr_val, d_csr_row_ptr, d_csr_col_ind, &
                         d_csc_val, d_csc_row_ind, d_csc_col_ptr, &
                         HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_INDEX_BASE_ZERO))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(h_csc_col_ptr, d_csc_col_ptr, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(h_csc_row_ind, d_csc_row_ind, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(h_csc_val,     d_csc_val,     hipMemcpyDeviceToHost))
  do i = 1,N+1
    if(h_csc_col_ptr(i) /= h_exp_col_ptr(i)) then
        write(*,*) "FAILED! csc_col_ptr(", i, ") = ", h_csc_col_ptr(i), " expected ", h_exp_col_ptr(i); call exit(1)
    end if
  end do
  do i = 1,nnz
    if(h_csc_row_ind(i) /= h_exp_row_ind(i)) then
        write(*,*) "FAILED! csc_row_ind(", i, ") = ", h_csc_row_ind(i); call exit(1)
    end if
    error = abs(h_csc_val(i) - h_exp_val(i))
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! csc_val(", i, ") = ", h_csc_val(i); call exit(1)
    end if
  end do
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr)); call hipCheck(hipFree(d_csr_col_ind)); call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_csc_col_ptr)); call hipCheck(hipFree(d_csc_row_ind)); call hipCheck(hipFree(d_csc_val))
  write(*,*) "PASSED!"
end program hipsparse_scsr2csc
