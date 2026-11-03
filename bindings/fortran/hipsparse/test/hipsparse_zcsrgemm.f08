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
! zcsrgemm example (double-complex sparse-matrix sparse-matrix multiply,
! C = A*B)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Two-phase flow with the classic hipSPARSE API: XcsrgemmNnz fills row_ptr_C and
! the total nnz, then Xcsrgemm computes the values. Here B = A, so C = A*A;
! the expected values come from a dense matmul of the same A.
!
! NOTE: the mat-descr/array arguments are c_ptr-only, so device buffers are
! passed via c_loc(...). nnzTotal is a host int (hipSPARSE default pointer mode).
!!!!!!!!!!!!!!/
!
program hipsparse_zcsrgemm
  use iso_c_binding
  use hip
  use hipsparse

  implicit none
  integer :: i

  ! Sparse A (3x3) in CSR (0-based); complex values.
  ! A = [[a11,0,a13],[0,a22,0],[a31,0,a33]]
  integer(c_int), parameter :: M = 3, N = 3, K = 3, nnz_A = 5

  integer(c_int) :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  complex(c_double_complex) :: h_csr_val(5) = &
    (/ (1.d0,1.d0),(2.d0,-1.d0),(3.d0,0.d0),(4.d0,2.d0),(5.d0,-1.d0) /)

  ! Expected C (built from a dense product of A)
  integer(c_int) :: h_exp_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_exp_col_ind(5) = (/0, 2, 1, 0, 2/)
  complex(c_double_complex) :: h_exp_val(5)
  complex(c_double_complex) :: A_dense(3,3), C_dense(3,3)

  integer(c_int) :: h_row_ptr_C(4)
  integer(c_int), target :: nnz_C

  integer(c_int), pointer :: d_csr_row_ptr(:), d_csr_col_ind(:)
  complex(c_double_complex), pointer :: d_csr_val(:)
  integer(c_int), pointer :: d_row_ptr_C(:), d_col_ind_C(:)
  complex(c_double_complex), pointer :: d_val_C(:)

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descr_A, descr_B, descr_C

  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zcsrgemm' (Fortran 2008 interfaces) - "

  ! Build the expected product densely
  A_dense = (0.d0, 0.d0)
  A_dense(1,1) = (1.d0,1.d0); A_dense(1,3) = (2.d0,-1.d0)
  A_dense(2,2) = (3.d0,0.d0)
  A_dense(3,1) = (4.d0,2.d0); A_dense(3,3) = (5.d0,-1.d0)
  C_dense = matmul(A_dense, A_dense)
  ! CSR order of the pattern: (1,1),(1,3),(2,2),(3,1),(3,3)  [1-based]
  h_exp_val(1) = C_dense(1,1)
  h_exp_val(2) = C_dense(1,3)
  h_exp_val(3) = C_dense(2,2)
  h_exp_val(4) = C_dense(3,1)
  h_exp_val(5) = C_dense(3,3)

  ! Create handle and matrix descriptors
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateMatDescr(descr_A))
  call hipsparseCheck(hipsparseCreateMatDescr(descr_B))
  call hipsparseCheck(hipsparseCreateMatDescr(descr_C))

  ! Allocate device memory and copy A (B aliases A)
  call hipCheck(hipMalloc(d_csr_row_ptr, source=h_csr_row_ptr))
  call hipCheck(hipMalloc(d_csr_col_ind, source=h_csr_col_ind))
  call hipCheck(hipMalloc(d_csr_val,     source=h_csr_val))
  call hipCheck(hipMalloc(d_row_ptr_C,   mold=h_row_ptr_C))

  ! Phase 1: compute the sparsity of C (row_ptr_C + total nnz_C)
  call hipsparseCheck(hipsparseXcsrgemmNnz(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          M, N, K, descr_A, nnz_A, c_loc(d_csr_row_ptr(1)), c_loc(d_csr_col_ind(1)), &
                          descr_B, nnz_A, c_loc(d_csr_row_ptr(1)), c_loc(d_csr_col_ind(1)), &
                          descr_C, c_loc(d_row_ptr_C(1)), c_loc(nnz_C)))

  ! Allocate C column indices and values now that nnz_C is known
  call hipCheck(hipMalloc(d_col_ind_C, dims=(/nnz_C/)))
  call hipCheck(hipMalloc(d_val_C,     dims=(/nnz_C/)))

  ! Phase 2: compute the values of C
  call hipsparseCheck(hipsparseZcsrgemm(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          M, N, K, descr_A, nnz_A, c_loc(d_csr_val(1)), c_loc(d_csr_row_ptr(1)), c_loc(d_csr_col_ind(1)), &
                          descr_B, nnz_A, c_loc(d_csr_val(1)), c_loc(d_csr_row_ptr(1)), c_loc(d_csr_col_ind(1)), &
                          descr_C, c_loc(d_val_C(1)), c_loc(d_row_ptr_C(1)), c_loc(d_col_ind_C(1))))
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
    complex(c_double_complex) :: h_val_C(nnz_C)
    call hipCheck(hipMemcpy(h_col_ind_C, d_col_ind_C, hipMemcpyDeviceToHost))
    call hipCheck(hipMemcpy(h_val_C,     d_val_C,     hipMemcpyDeviceToHost))
    do i = 1,nnz_C
      if(h_col_ind_C(i) /= h_exp_col_ind(i)) then
          write(*,*) "FAILED! col_ind_C(", i, ") = ", h_col_ind_C(i), " expected ", h_exp_col_ind(i)
          call exit(1)
      end if
      error = abs(h_val_C(i) - h_exp_val(i)) / max(abs(h_exp_val(i)), 1.0_c_double)
      if(.not. (error .le. error_max)) then
          write(*,*) "FAILED! val_C(", i, ") = ", h_val_C(i), " expected ", h_exp_val(i)
          call exit(1)
      end if
    end do
  end block

  ! Clean up
  call hipCheck(hipFree(d_col_ind_C))
  call hipCheck(hipFree(d_val_C))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_row_ptr_C))
  call hipsparseCheck(hipsparseDestroyMatDescr(descr_A))
  call hipsparseCheck(hipsparseDestroyMatDescr(descr_B))
  call hipsparseCheck(hipsparseDestroyMatDescr(descr_C))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsparse_zcsrgemm
