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
! dsptrsv example (double-precision sparse triangular solve, op(A)*y = alpha*x)
! see: https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Uses the generic SpSV API on a lower-triangular L. Self-verifying: pick a
! known y, form x = L*y, solve L*y' = x, and confirm y' recovers y. The
! triangular structure is set via SpMatSetAttribute (fill_mode + diag_type).
! hipSPARSE flow: createDescr -> bufferSize -> analysis -> solve.
!
! NOTE: descriptor/array arguments are c_ptr-only, so device buffers and the
! attribute values are passed via c_loc(...). bufferSize is a c_ptr to a host
! size_t.
!!!!!!!!!!!!!!/
!
program hipsparse_dsptrsv
  use iso_c_binding
  use hip
  use hipsparse

  implicit none
  integer :: i

  ! Lower-triangular L (3x3) in CSR (0-based):  L = [[2,0,0],[1,3,0],[4,5,6]]
  integer(c_int), parameter :: M = 3, N = 3, nnz = 6

  integer(c_int) :: h_csr_row_ptr(4) = (/0, 1, 3, 6/)
  integer(c_int) :: h_csr_col_ind(6) = (/0, 0, 1, 0, 1, 2/)
  real(c_double) :: h_csr_val(6)     = (/2, 1, 3, 4, 5, 6/)

  real(c_double) :: h_y(3) = (/1, 2, 3/)   ! known solution
  real(c_double) :: h_x(3)                  ! rhs = L*y
  real(c_double) :: h_yout(3)               ! recovered solution

  real(c_double), target :: alpha = 1.0_c_double
  integer(kind(HIPSPARSE_FILL_MODE_LOWER)), target :: fill = HIPSPARSE_FILL_MODE_LOWER
  integer(kind(HIPSPARSE_DIAG_TYPE_NON_UNIT)), target :: diag = HIPSPARSE_DIAG_TYPE_NON_UNIT

  integer(c_int), pointer :: d_csr_row_ptr(:), d_csr_col_ind(:)
  real(c_double), pointer :: d_csr_val(:), d_x(:), d_y(:)

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: matL, vecX, vecY, spsvDescr, d_buffer
  integer(c_size_t), target :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_dsptrsv' (Fortran 2008 interfaces) - "

  ! Build a consistent rhs so that L*y = x:  x1=2, x2=1+6=7, x3=4+10+18=32
  h_x(1) = 2.0_c_double
  h_x(2) = 7.0_c_double
  h_x(3) = 32.0_c_double

  ! Allocate device memory and copy inputs
  call hipCheck(hipMalloc(d_csr_row_ptr, source=h_csr_row_ptr))
  call hipCheck(hipMalloc(d_csr_col_ind, source=h_csr_col_ind))
  call hipCheck(hipMalloc(d_csr_val,     source=h_csr_val))
  call hipCheck(hipMalloc(d_x,           source=h_x))
  call hipCheck(hipMalloc(d_y,           mold=h_yout))

  ! Create handle, CSR descriptor for L (lower / non-unit diag), dense vectors
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateCsr(matL, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
                          c_loc(d_csr_row_ptr(1)), c_loc(d_csr_col_ind(1)), c_loc(d_csr_val(1)), &
                          HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F))
  call hipsparseCheck(hipsparseSpMatSetAttribute(matL, HIPSPARSE_SPMAT_FILL_MODE, c_loc(fill), int(4,c_size_t)))
  call hipsparseCheck(hipsparseSpMatSetAttribute(matL, HIPSPARSE_SPMAT_DIAG_TYPE, c_loc(diag), int(4,c_size_t)))
  call hipsparseCheck(hipsparseCreateDnVec(vecX, int(M,c_int64_t), c_loc(d_x(1)), HIP_R_64F))
  call hipsparseCheck(hipsparseCreateDnVec(vecY, int(M,c_int64_t), c_loc(d_y(1)), HIP_R_64F))
  call hipsparseCheck(hipsparseSpSV_createDescr(spsvDescr))

  ! Stage 1: workspace size
  call hipsparseCheck(hipsparseSpSV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matL, vecX, vecY, &
                          HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr, buffer_size))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: analysis
  call hipsparseCheck(hipsparseSpSV_analysis(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matL, vecX, vecY, &
                          HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr, d_buffer))

  ! Stage 3: solve
  call hipsparseCheck(hipsparseSpSV_solve(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matL, vecX, vecY, &
                          HIP_R_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr))

  ! Copy the recovered solution back
  call hipCheck(hipMemcpy(h_yout, d_y, hipMemcpyDeviceToHost))

  ! Verify y' == y
  do i = 1,M
    error = abs(h_yout(i) - h_y(i)) / max(abs(h_y(i)), 1.0_c_double)
    if(.not. (error .le. error_max)) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " y(", i, ") = ", h_yout(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipsparseCheck(hipsparseSpSV_destroyDescr(spsvDescr))
  call hipsparseCheck(hipsparseDestroyDnVec(vecX))
  call hipsparseCheck(hipsparseDestroyDnVec(vecY))
  call hipsparseCheck(hipsparseDestroySpMat(matL))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_x))
  call hipCheck(hipFree(d_y))
  call hipCheck(hipFree(d_buffer))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsparse_dsptrsv
