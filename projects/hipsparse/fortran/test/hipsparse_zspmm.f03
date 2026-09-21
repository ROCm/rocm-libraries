!!!!!!!!!!!!!!
! hipsparse SpMM example (double complex, C = alpha*A*B + beta*C, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Generic API: build a CSR descriptor for the sparse A and dense-matrix
! descriptors for B and C, query the workspace with SpMM_bufferSize, then run
! SpMM. Result is checked against a dense host reference (matmul(A_dense, B)).
! Dense matrices are column-major (HIPSPARSE_ORDER_COL).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_zspmm
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i, j

  integer(c_int), parameter :: M = 3, K = 3, Ncol = 2, nnz = 5
  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  complex(c_double_complex), target :: h_csr_val(5) = (/ &
    (1.0d0,1.0d0), (2.0d0,0.0d0), (3.0d0,1.0d0), (4.0d0,0.0d0), (5.0d0,1.0d0)/)
  complex(c_double_complex), target :: h_B(3,2) = reshape((/ &
    (1.0d0,0.0d0), (2.0d0,1.0d0), (3.0d0,0.0d0), (4.0d0,1.0d0), (5.0d0,0.0d0), (6.0d0,1.0d0)/), (/3,2/))
  complex(c_double_complex), target :: h_C(3,2)
  complex(c_double_complex) :: h_Adense(3,3), h_expected(3,2)
  complex(c_double_complex), target :: alpha = (1.0d0,0.0d0), beta = (0.0d0,0.0d0)

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val, d_B, d_C
  type(c_ptr) :: handle, matA, matB, matC, d_buffer
  integer(c_size_t) :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-11

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zspmm' (Fortran 2003 interfaces) - "

  h_Adense = (0.0d0,0.0d0)
  h_Adense(1,1) = (1.0d0,1.0d0); h_Adense(1,3) = (2.0d0,0.0d0)
  h_Adense(2,2) = (3.0d0,1.0d0)
  h_Adense(3,1) = (4.0d0,0.0d0); h_Adense(3,3) = (5.0d0,1.0d0)
  h_expected = matmul(h_Adense, h_B)

  call hipCheck(hipMalloc(d_csr_row_ptr, int(M+1,c_size_t)    * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, int(nnz,c_size_t)    * 4))
  call hipCheck(hipMalloc(d_csr_val,     int(nnz,c_size_t)    * 16))
  call hipCheck(hipMalloc(d_B,           int(K*Ncol,c_size_t) * 16))
  call hipCheck(hipMalloc(d_C,           int(M*Ncol,c_size_t) * 16))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), int(M+1,c_size_t)    * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), int(nnz,c_size_t)    * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     int(nnz,c_size_t)    * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_B,           c_loc(h_B(1,1)),         int(K*Ncol,c_size_t) * 16, hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateCsr(matA, int(M,c_int64_t), int(K,c_int64_t), int(nnz,c_int64_t), &
       d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_C_64F))
  call hipsparseCheck(hipsparseCreateDnMat(matB, int(K,c_int64_t), int(Ncol,c_int64_t), int(K,c_int64_t), &
       d_B, HIP_C_64F, HIPSPARSE_ORDER_COL))
  call hipsparseCheck(hipsparseCreateDnMat(matC, int(M,c_int64_t), int(Ncol,c_int64_t), int(M,c_int64_t), &
       d_C, HIP_C_64F, HIPSPARSE_ORDER_COL))

  call hipsparseCheck(hipsparseSpMM_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matA, matB, c_loc(beta), matC, &
       HIP_C_64F, HIPSPARSE_SPMM_ALG_DEFAULT, buffer_size))
  ! hipSPARSE requires a null buffer when the queried size is 0; a non-null
  ! (dummy) pointer makes SpMM return HIPSPARSE_STATUS_INVALID_VALUE.
  d_buffer = c_null_ptr
  if (buffer_size > 0) call hipCheck(hipMalloc(d_buffer, buffer_size))
  call hipsparseCheck(hipsparseSpMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matA, matB, c_loc(beta), matC, &
       HIP_C_64F, HIPSPARSE_SPMM_ALG_DEFAULT, d_buffer))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(h_C(1,1)), d_C, int(M*Ncol,c_size_t) * 16, hipMemcpyDeviceToHost))

  do j = 1, Ncol
    do i = 1, M
      error = abs(h_C(i,j) - h_expected(i,j)) / max(abs(h_expected(i,j)), 1.0_c_double)
      if(error .gt. error_max) then
          write(*,*) "FAILED! C(", i, j, ") = ", h_C(i,j), " expected ", h_expected(i,j); call exit(1)
      end if
    end do
  end do

  call hipsparseCheck(hipsparseDestroyDnMat(matB))
  call hipsparseCheck(hipsparseDestroyDnMat(matC))
  call hipsparseCheck(hipsparseDestroySpMat(matA))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr)); call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val)); call hipCheck(hipFree(d_B)); call hipCheck(hipFree(d_C))
  if (c_associated(d_buffer)) call hipCheck(hipFree(d_buffer))
  write(*,*) "PASSED!"
end program hipsparse_zspmm
