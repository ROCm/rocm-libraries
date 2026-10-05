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
! zsptrsm example (double-complex sparse triangular solve with multiple rhs)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Uses the generic SpSM API on a lower-triangular L with a dense rhs matrix.
! Self-verifying: pick a known Y, form X = L*Y (dense), solve L*C = X, and
! confirm C recovers Y. hipSPARSE flow: createDescr -> bufferSize -> analysis
! -> solve.
!
! NOTE: descriptor/array arguments are c_ptr-only, so device buffers and the
! attribute values are passed via c_loc(...). bufferSize is a c_ptr to a host
! size_t.
!!!!!!!!!!!!!!/
!
program hipsparse_zsptrsm
  use iso_c_binding
  use hip
  use hipsparse

  implicit none
  integer :: i, j

  ! Lower-triangular L (3x3) in CSR (0-based); complex values
  integer(c_int), parameter :: M = 3, nrhs = 2, nnz = 6

  integer(c_int) :: h_csr_row_ptr(4) = (/0, 1, 3, 6/)
  integer(c_int) :: h_csr_col_ind(6) = (/0, 0, 1, 0, 1, 2/)
  complex(c_double_complex) :: h_csr_val(6) = &
    (/ (2.d0,1.d0), (1.d0,0.d0),(3.d0,-1.d0), (4.d0,1.d0),(5.d0,0.d0),(6.d0,2.d0) /)

  complex(c_double_complex) :: h_Y(3,2) = reshape( &
    (/ (1.d0,1.d0),(2.d0,-1.d0),(3.d0,0.d0), (4.d0,0.d0),(5.d0,2.d0),(6.d0,-1.d0) /), (/3, 2/))  ! known solution
  complex(c_double_complex) :: h_X(3,2)          ! rhs = L*Y
  complex(c_double_complex) :: h_C(3,2)          ! recovered solution
  complex(c_double_complex) :: L_dense(3,3)

  complex(c_double_complex), target :: alpha = (1.0d0,0.0d0)
  integer(kind(HIPSPARSE_FILL_MODE_LOWER)), target :: fill = HIPSPARSE_FILL_MODE_LOWER
  integer(kind(HIPSPARSE_DIAG_TYPE_NON_UNIT)), target :: diag = HIPSPARSE_DIAG_TYPE_NON_UNIT

  integer(c_int), pointer :: d_csr_row_ptr(:), d_csr_col_ind(:)
  complex(c_double_complex), pointer :: d_csr_val(:)
  complex(c_double_complex), pointer :: d_X(:,:), d_C(:,:)

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: matL, matB, matC, spsmDescr, d_buffer
  integer(c_size_t), target :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zsptrsm' (Fortran 2008 interfaces) - "

  ! Build dense L and the consistent rhs X = L*Y on the host
  L_dense = (0.0d0, 0.0d0)
  L_dense(1,1) = (2.d0,1.d0)
  L_dense(2,1) = (1.d0,0.d0); L_dense(2,2) = (3.d0,-1.d0)
  L_dense(3,1) = (4.d0,1.d0); L_dense(3,2) = (5.d0,0.d0); L_dense(3,3) = (6.d0,2.d0)
  h_X = matmul(L_dense, h_Y)

  ! Allocate device memory and copy inputs
  call hipCheck(hipMalloc(d_csr_row_ptr, source=h_csr_row_ptr))
  call hipCheck(hipMalloc(d_csr_col_ind, source=h_csr_col_ind))
  call hipCheck(hipMalloc(d_csr_val,     source=h_csr_val))
  call hipCheck(hipMalloc(d_X,           source=h_X))
  call hipCheck(hipMalloc(d_C,           mold=h_C))

  ! Create handle, CSR descriptor for L (lower / non-unit diag), dense matrices
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateCsr(matL, int(M,c_int64_t), int(M,c_int64_t), int(nnz,c_int64_t), &
                          c_loc(d_csr_row_ptr(1)), c_loc(d_csr_col_ind(1)), c_loc(d_csr_val(1)), &
                          HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_C_64F))
  call hipsparseCheck(hipsparseSpMatSetAttribute(matL, HIPSPARSE_SPMAT_FILL_MODE, c_loc(fill), int(4,c_size_t)))
  call hipsparseCheck(hipsparseSpMatSetAttribute(matL, HIPSPARSE_SPMAT_DIAG_TYPE, c_loc(diag), int(4,c_size_t)))
  call hipsparseCheck(hipsparseCreateDnMat(matB, int(M,c_int64_t), int(nrhs,c_int64_t), int(M,c_int64_t), &
                          c_loc(d_X(1,1)), HIP_C_64F, HIPSPARSE_ORDER_COLUMN))
  call hipsparseCheck(hipsparseCreateDnMat(matC, int(M,c_int64_t), int(nrhs,c_int64_t), int(M,c_int64_t), &
                          c_loc(d_C(1,1)), HIP_C_64F, HIPSPARSE_ORDER_COLUMN))
  call hipsparseCheck(hipsparseSpSM_createDescr(spsmDescr))

  ! Stage 1: workspace size
  call hipsparseCheck(hipsparseSpSM_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          c_loc(alpha), matL, matB, matC, HIP_C_64F, HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr, buffer_size))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: analysis
  call hipsparseCheck(hipsparseSpSM_analysis(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          c_loc(alpha), matL, matB, matC, HIP_C_64F, HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr, d_buffer))

  ! Stage 3: solve
  call hipsparseCheck(hipsparseSpSM_solve(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          c_loc(alpha), matL, matB, matC, HIP_C_64F, HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr, d_buffer))

  ! Copy the recovered solution back
  call hipCheck(hipMemcpy(h_C, d_C, hipMemcpyDeviceToHost))

  ! Verify C == Y
  do j = 1,nrhs
    do i = 1,M
      error = abs(h_C(i,j) - h_Y(i,j)) / max(abs(h_Y(i,j)), 1.0_c_double)
      if(.not. (error .le. error_max)) then
          write(*,*) "FAILED! Error bigger than max! Error = ", error, " C(", i, ",", j, ") = ", h_C(i,j)
          call exit(1)
      end if
    end do
  end do

  ! Clean up
  call hipsparseCheck(hipsparseSpSM_destroyDescr(spsmDescr))
  call hipsparseCheck(hipsparseDestroyDnMat(matB))
  call hipsparseCheck(hipsparseDestroyDnMat(matC))
  call hipsparseCheck(hipsparseDestroySpMat(matL))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_X))
  call hipCheck(hipFree(d_C))
  call hipCheck(hipFree(d_buffer))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsparse_zsptrsm
