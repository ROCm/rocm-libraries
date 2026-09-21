!!!!!!!!!!!!!!
! hipsparse SpMV example (double complex, y = alpha*A*x + beta*y, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Generic API: build a CSR descriptor for A and dense-vector descriptors for x
! and y, query the workspace with SpMV_bufferSize, then run SpMV. Result is
! checked against a dense host reference. Uses HIPSPARSE_SPMV_ALG_DEFAULT.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_zspmv
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i

  integer(c_int), parameter :: M = 3, N = 3, nnz = 5
  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  complex(c_double_complex), target :: h_csr_val(5) = (/ &
    (1.0d0,1.0d0), (2.0d0,0.0d0), (3.0d0,1.0d0), (4.0d0,0.0d0), (5.0d0,1.0d0)/)
  complex(c_double_complex), target :: h_x(3) = (/(1.0d0,0.0d0), (2.0d0,1.0d0), (3.0d0,0.0d0)/)
  complex(c_double_complex), target :: h_y(3)
  complex(c_double_complex) :: h_dense(3,3), h_expected(3)
  complex(c_double_complex), target :: alpha = (1.0d0,0.0d0), beta = (0.0d0,0.0d0)

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val, d_x, d_y
  type(c_ptr) :: handle, matA, vecX, vecY, d_buffer
  integer(c_size_t) :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-11

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zspmv' (Fortran 2003 interfaces) - "

  h_dense = (0.0d0,0.0d0)
  h_dense(1,1) = (1.0d0,1.0d0); h_dense(1,3) = (2.0d0,0.0d0)
  h_dense(2,2) = (3.0d0,1.0d0)
  h_dense(3,1) = (4.0d0,0.0d0); h_dense(3,3) = (5.0d0,1.0d0)
  h_expected = matmul(h_dense, h_x)

  call hipCheck(hipMalloc(d_csr_row_ptr, int(M+1,c_size_t) * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(d_csr_val,     int(nnz,c_size_t) * 16))
  call hipCheck(hipMalloc(d_x,           int(N,c_size_t)   * 16))
  call hipCheck(hipMalloc(d_y,           int(M,c_size_t)   * 16))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), int(M+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     int(nnz,c_size_t) * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_x,           c_loc(h_x(1)),           int(N,c_size_t)   * 16, hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateCsr(matA, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
       d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_C_64F))
  call hipsparseCheck(hipsparseCreateDnVec(vecX, int(N,c_int64_t), d_x, HIP_C_64F))
  call hipsparseCheck(hipsparseCreateDnVec(vecY, int(M,c_int64_t), d_y, HIP_C_64F))

  call hipsparseCheck(hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       c_loc(alpha), matA, vecX, c_loc(beta), vecY, HIP_C_64F, HIPSPARSE_SPMV_ALG_DEFAULT, buffer_size))
  ! hipSPARSE requires a null buffer when the queried size is 0; a non-null
  ! (dummy) pointer makes SpMV return HIPSPARSE_STATUS_INVALID_VALUE.
  d_buffer = c_null_ptr
  if (buffer_size > 0) call hipCheck(hipMalloc(d_buffer, buffer_size))
  call hipsparseCheck(hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       c_loc(alpha), matA, vecX, c_loc(beta), vecY, HIP_C_64F, HIPSPARSE_SPMV_ALG_DEFAULT, d_buffer))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(h_y(1)), d_y, int(M,c_size_t) * 16, hipMemcpyDeviceToHost))

  do i = 1, M
    error = abs(h_y(i) - h_expected(i)) / max(abs(h_expected(i)), 1.0_c_double)
    if(error .gt. error_max) then
        write(*,*) "FAILED! y(", i, ") = ", h_y(i), " expected ", h_expected(i); call exit(1)
    end if
  end do

  call hipsparseCheck(hipsparseDestroyDnVec(vecX))
  call hipsparseCheck(hipsparseDestroyDnVec(vecY))
  call hipsparseCheck(hipsparseDestroySpMat(matA))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr)); call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val)); call hipCheck(hipFree(d_x)); call hipCheck(hipFree(d_y))
  if (c_associated(d_buffer)) call hipCheck(hipFree(d_buffer))
  write(*,*) "PASSED!"
end program hipsparse_zspmv
