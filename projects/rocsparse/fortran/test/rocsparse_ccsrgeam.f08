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

!!!!!!!!!!!!!/
! ccsrgeam example (single-precision complex sparse matrix addition, C = alpha*A + beta*B)
! see: https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/extra.html
!
! Two-phase flow: csrgeam_nnz (fills row_ptr_C and nnz_C) -> allocate
! col_ind_C/val_C -> csrgeam (computes C). csrgeam needs no workspace buffer.
! Here C = A + B (alpha = beta = 1), checked against the known sum.
!
! NOTE: the mat-descr/array arguments are c_ptr-only, so device buffers are
! passed via c_loc(...).
!!!!!!!!!!!!!!/
!
program ccsrgeam
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  ! A (3x3) CSR (0-based); B = complex diagonal. Diagonals merge in C = A + B.
  integer(c_int), parameter :: M = 3, N = 3, nnz_A = 5, nnz_B = 3

  integer(c_int) :: h_row_ptr_A(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_col_ind_A(5) = (/0, 2, 1, 0, 2/)
  complex(c_float_complex) :: h_val_A(5) = (/ (1.,1.),(2.,-1.),(3.,0.),(4.,2.),(5.,-1.) /)

  integer(c_int) :: h_row_ptr_B(4) = (/0, 1, 2, 3/)
  integer(c_int) :: h_col_ind_B(3) = (/0, 1, 2/)
  complex(c_float_complex) :: h_val_B(3) = (/ (10.,1.),(20.,-2.),(30.,3.) /)

  ! Expected C = A + B (pattern of A; diag entries (0,0),(1,1),(2,2) get B added):
  !   C(0,0)=1+10i +10+1i = 11+2i ; C(0,2)=2-1i ; C(1,1)=3+20-2i=23-2i ;
  !   C(2,0)=4+2i ; C(2,2)=5-1i+30+3i=35+2i
  integer(c_int) :: h_exp_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_exp_col_ind(5) = (/0, 2, 1, 0, 2/)
  complex(c_float_complex) :: h_exp_val(5) = (/ (11.,2.),(2.,-1.),(23.,-2.),(4.,2.),(35.,2.) /)

  complex(c_float_complex), target :: alpha = (1.0,0.0), beta = (1.0,0.0)

  integer(c_int) :: h_row_ptr_C(4)
  integer(c_int), target :: nnz_C

  integer(c_int), pointer :: d_row_ptr_A(:), d_col_ind_A(:)
  complex(c_float_complex), pointer :: d_val_A(:)
  integer(c_int), pointer :: d_row_ptr_B(:), d_col_ind_B(:)
  complex(c_float_complex), pointer :: d_val_B(:)
  integer(c_int), pointer :: d_row_ptr_C(:), d_col_ind_C(:)
  complex(c_float_complex), pointer :: d_val_C(:)

  type(c_ptr) :: handle, descr_A, descr_B, descr_C

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_ccsrgeam' (Fortran 2008 interfaces) - "

  ! Create handle and matrix descriptors
  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_A))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_B))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_C))

  ! Copy A and B to device; allocate row_ptr_C + nnz_C
  call hipCheck(hipMalloc(d_row_ptr_A, source=h_row_ptr_A))
  call hipCheck(hipMalloc(d_col_ind_A, source=h_col_ind_A))
  call hipCheck(hipMalloc(d_val_A,     source=h_val_A))
  call hipCheck(hipMalloc(d_row_ptr_B, source=h_row_ptr_B))
  call hipCheck(hipMalloc(d_col_ind_B, source=h_col_ind_B))
  call hipCheck(hipMalloc(d_val_B,     source=h_val_B))
  call hipCheck(hipMalloc(d_row_ptr_C, mold=h_row_ptr_C))

  ! Phase 1: compute the sparsity of C (row_ptr_C + total nnz_C)
  call rocsparseCheck(rocsparse_csrgeam_nnz(handle, M, N, &
                          descr_A, nnz_A, c_loc(d_row_ptr_A), c_loc(d_col_ind_A), &
                          descr_B, nnz_B, c_loc(d_row_ptr_B), c_loc(d_col_ind_B), &
                          descr_C, c_loc(d_row_ptr_C), c_loc(nnz_C)))

  ! Allocate C column indices and values now that nnz_C is known
  call hipCheck(hipMalloc(d_col_ind_C, dims=(/nnz_C/)))
  call hipCheck(hipMalloc(d_val_C,     dims=(/nnz_C/)))

  ! Phase 2: compute the values of C
  call rocsparseCheck(rocsparse_ccsrgeam(handle, M, N, c_loc(alpha), &
                          descr_A, nnz_A, c_loc(d_val_A), c_loc(d_row_ptr_A), c_loc(d_col_ind_A), &
                          c_loc(beta), descr_B, nnz_B, c_loc(d_val_B), c_loc(d_row_ptr_B), c_loc(d_col_ind_B), &
                          descr_C, c_loc(d_val_C), c_loc(d_row_ptr_C), c_loc(d_col_ind_C)))
  call hipCheck(hipDeviceSynchronize())

  ! Copy the C structure back to host
  call hipCheck(hipMemcpy(h_row_ptr_C, d_row_ptr_C, hipMemcpyDeviceToHost))

  ! Verify nnz and row pointers
  if(nnz_C /= 5) then
    write(*,*) "FAILED! nnz_C = ", nnz_C, " expected 5"
    call exit(1)
  end if
  do i = 1,N+1
    if(h_row_ptr_C(i) /= h_exp_row_ptr(i)) then
        write(*,*) "FAILED! row_ptr_C(", i, ") = ", h_row_ptr_C(i), " expected ", h_exp_row_ptr(i)
        call exit(1)
    end if
  end do

  ! Verify column indices and values
  block
    integer(c_int) :: h_col_ind_C(nnz_C)
    complex(c_float_complex) :: h_val_C(nnz_C)
    call hipCheck(hipMemcpy(h_col_ind_C, d_col_ind_C, hipMemcpyDeviceToHost))
    call hipCheck(hipMemcpy(h_val_C,     d_val_C,     hipMemcpyDeviceToHost))
    do i = 1,nnz_C
      if(h_col_ind_C(i) /= h_exp_col_ind(i)) then
          write(*,*) "FAILED! col_ind_C(", i, ") = ", h_col_ind_C(i), " expected ", h_exp_col_ind(i)
          call exit(1)
      end if
      error = abs(h_val_C(i) - h_exp_val(i)) / max(abs(h_exp_val(i)), 1.0)
      if(.not. (error .le. error_max)) then
          write(*,*) "FAILED! val_C(", i, ") = ", h_val_C(i), " expected ", h_exp_val(i)
          call exit(1)
      end if
    end do
  end block

  ! Clean up
  call hipCheck(hipFree(d_col_ind_C))
  call hipCheck(hipFree(d_val_C))
  call hipCheck(hipFree(d_row_ptr_A))
  call hipCheck(hipFree(d_col_ind_A))
  call hipCheck(hipFree(d_val_A))
  call hipCheck(hipFree(d_row_ptr_B))
  call hipCheck(hipFree(d_col_ind_B))
  call hipCheck(hipFree(d_val_B))
  call hipCheck(hipFree(d_row_ptr_C))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_A))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_B))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_C))
  call rocsparseCheck(rocsparse_destroy_handle(handle))

  write(*,*) "PASSED!"

end program ccsrgeam
