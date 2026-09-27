!!!!!!!!!!!!!/
! dsptrsm example (double-precision sparse triangular solve with multiple rhs,
! Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Uses the generic SpSM API on a lower-triangular L with a dense rhs matrix.
! Self-verifying: pick a known Y, form X = L*Y, solve L*C = X, and confirm C
! recovers Y. Triangular structure set via SpMatSetAttribute.
! hipSPARSE flow: createDescr -> bufferSize -> analysis -> solve.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc. Descriptor/array args are c_ptr-only.
!!!!!!!!!!!!!!/
!
program dsptrsm
  use iso_c_binding
  use hip
  use hipsparse

  implicit none
  integer :: i, j

  ! Lower-triangular L (3x3) in CSR (0-based):  L = [[2,0,0],[1,3,0],[4,5,6]]
  integer(c_int), parameter :: M = 3, nrhs = 2, nnz = 6

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 1, 3, 6/)
  integer(c_int), target :: h_csr_col_ind(6) = (/0, 0, 1, 0, 1, 2/)
  real(c_double), target :: h_csr_val(6)     = (/2, 1, 3, 4, 5, 6/)

  real(c_double) :: h_Y(3,2) = reshape((/1, 2, 3, 4, 5, 6/), (/3, 2/))  ! known solution
  real(c_double), target :: h_X(3,2)          ! rhs = L*Y
  real(c_double), target :: h_C(3,2)          ! recovered solution
  real(c_double) :: L_dense(3,3)

  real(c_double), target :: alpha = 1.0_c_double
  integer(kind(HIPSPARSE_FILL_MODE_LOWER)), target :: fill = HIPSPARSE_FILL_MODE_LOWER
  integer(kind(HIPSPARSE_DIAG_TYPE_NON_UNIT)), target :: diag = HIPSPARSE_DIAG_TYPE_NON_UNIT

  integer(c_size_t) :: size_rp = 4, size_nz = 6, size_mat = 6

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: matL, matB, matC, spsmDescr, d_buffer
  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val, d_X, d_C
  integer(c_size_t), target :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_dsptrsm' (Fortran 2003 interfaces) - "

  ! Build dense L and the consistent rhs X = L*Y on the host
  L_dense = 0.0_c_double
  L_dense(1,1) = 2.0_c_double
  L_dense(2,1) = 1.0_c_double; L_dense(2,2) = 3.0_c_double
  L_dense(3,1) = 4.0_c_double; L_dense(3,2) = 5.0_c_double; L_dense(3,3) = 6.0_c_double
  h_X = matmul(L_dense, h_Y)

  ! Allocate device memory and copy inputs
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_nz * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_nz * 8))
  call hipCheck(hipMalloc(d_X,           size_mat * 8))
  call hipCheck(hipMalloc(d_C,           size_mat * 8))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_nz * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_nz * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_X,           c_loc(h_X(1,1)),         size_mat * 8, hipMemcpyHostToDevice))

  ! Create handle, CSR descriptor for L (lower / non-unit diag), dense matrices
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateCsr(matL, int(M,c_int64_t), int(M,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F))
  call hipsparseCheck(hipsparseSpMatSetAttribute(matL, HIPSPARSE_SPMAT_FILL_MODE, c_loc(fill), int(4,c_size_t)))
  call hipsparseCheck(hipsparseSpMatSetAttribute(matL, HIPSPARSE_SPMAT_DIAG_TYPE, c_loc(diag), int(4,c_size_t)))
  call hipsparseCheck(hipsparseCreateDnMat(matB, int(M,c_int64_t), int(nrhs,c_int64_t), int(M,c_int64_t), &
                          d_X, HIP_R_64F, HIPSPARSE_ORDER_COLUMN))
  call hipsparseCheck(hipsparseCreateDnMat(matC, int(M,c_int64_t), int(nrhs,c_int64_t), int(M,c_int64_t), &
                          d_C, HIP_R_64F, HIPSPARSE_ORDER_COLUMN))
  call hipsparseCheck(hipsparseSpSM_createDescr(spsmDescr))

  ! Stage 1: workspace size
  call hipsparseCheck(hipsparseSpSM_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          c_loc(alpha), matL, matB, matC, HIP_R_64F, HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr, buffer_size))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: analysis
  call hipsparseCheck(hipsparseSpSM_analysis(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          c_loc(alpha), matL, matB, matC, HIP_R_64F, HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr, d_buffer))

  ! Stage 3: solve
  call hipsparseCheck(hipsparseSpSM_solve(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          c_loc(alpha), matL, matB, matC, HIP_R_64F, HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr, d_buffer))

  ! Copy the recovered solution back
  call hipCheck(hipMemcpy(c_loc(h_C(1,1)), d_C, size_mat * 8, hipMemcpyDeviceToHost))

  ! Verify C == Y
  do j = 1,nrhs
    do i = 1,M
      error = abs(h_C(i,j) - h_Y(i,j)) / max(abs(h_Y(i,j)), 1.0_c_double)
      if(error .gt. error_max) then
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

end program dsptrsm
