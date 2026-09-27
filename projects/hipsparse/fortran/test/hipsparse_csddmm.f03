!!!!!!!!!!!!!!
! hipsparse SDDMM example (c, sampled dense-dense matmul, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! SDDMM computes C = alpha * (A * B) .* spy(C) + beta * C: the dense product
! A*B is evaluated only at the nonzero positions of the sparse (CSR) C. Generic
! API with three stages (SDDMM_bufferSize -> SDDMM_preprocess -> SDDMM).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly to the descriptor constructors / SDDMM.
!!!!!!!!!!!!!!
!
program hipsparse_csddmm
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i

  integer(c_int), parameter :: M = 3, N = 2, K = 3, nnz = 4

  complex(c_float_complex), target :: h_A(3,3) = reshape((/ &
       (1.,0.),(4.,0.),(7.,0.), &
       (2.,0.),(5.,0.),(8.,0.), &
       (3.,0.),(6.,0.),(10.,0.) /), (/3,3/))
  complex(c_float_complex), target :: h_B(3,2) = reshape((/ &
       (1.,0.),(3.,0.),(5.,0.), &
       (2.,0.),(4.,0.),(6.,0.) /), (/3,2/))

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 1, 2, 4/)
  integer(c_int), target :: h_csr_col_ind(4) = (/0, 1, 0, 1/)
  complex(c_float_complex), target :: h_csr_val(4)     = (/(0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0)/)

  complex(c_float_complex) :: h_AB(3,2), h_expected(4)
  complex(c_float_complex), target :: alpha = (1.0,0.0), beta = (0.0,0.0)

  integer(c_size_t) :: size_rp = 4, size_ci = 4, size_v = 4
  integer(c_size_t) :: size_A = 9, size_B = 6

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val, d_A, d_B
  type(c_ptr) :: handle, matA, matB, matC, d_buffer
  integer(c_size_t) :: buffer_size

  real :: error
  real, parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_csddmm' (Fortran 2003 interfaces) - "

  h_AB = matmul(h_A, h_B)
  h_expected(1) = h_AB(1,1)
  h_expected(2) = h_AB(2,2)
  h_expected(3) = h_AB(3,1)
  h_expected(4) = h_AB(3,2)

  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 8))
  call hipCheck(hipMalloc(d_A,           size_A * 8))
  call hipCheck(hipMalloc(d_B,           size_B * 8))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_A,           c_loc(h_A(1,1)),         size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_B,           c_loc(h_B(1,1)),         size_B * 8, hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateDnMat(matA, int(M,c_int64_t), int(K,c_int64_t), int(M,c_int64_t), &
       d_A, HIP_C_32F, HIPSPARSE_ORDER_COL))
  call hipsparseCheck(hipsparseCreateDnMat(matB, int(K,c_int64_t), int(N,c_int64_t), int(K,c_int64_t), &
       d_B, HIP_C_32F, HIPSPARSE_ORDER_COL))
  call hipsparseCheck(hipsparseCreateCsr(matC, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
       d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_C_32F))

  call hipsparseCheck(hipsparseSDDMM_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matA, matB, c_loc(beta), matC, &
       HIP_C_32F, HIPSPARSE_SDDMM_ALG_DEFAULT, buffer_size))
  d_buffer = c_null_ptr
  if (buffer_size > 0) call hipCheck(hipMalloc(d_buffer, buffer_size))
  call hipsparseCheck(hipsparseSDDMM_preprocess(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matA, matB, c_loc(beta), matC, &
       HIP_C_32F, HIPSPARSE_SDDMM_ALG_DEFAULT, d_buffer))
  call hipsparseCheck(hipsparseSDDMM(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matA, matB, c_loc(beta), matC, &
       HIP_C_32F, HIPSPARSE_SDDMM_ALG_DEFAULT, d_buffer))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(h_csr_val(1)), d_csr_val, size_v * 8, hipMemcpyDeviceToHost))

  do i = 1, nnz
    error = abs(h_csr_val(i) - h_expected(i)) / max(abs(h_expected(i)), 1.0)
    if(error .gt. error_max) then
        write(*,*) "FAILED! val(", i, ") = ", h_csr_val(i), " expected ", h_expected(i); call exit(1)
    end if
  end do

  call hipsparseCheck(hipsparseDestroyDnMat(matA))
  call hipsparseCheck(hipsparseDestroyDnMat(matB))
  call hipsparseCheck(hipsparseDestroySpMat(matC))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr)); call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val)); call hipCheck(hipFree(d_A)); call hipCheck(hipFree(d_B))
  if (c_associated(d_buffer)) call hipCheck(hipFree(d_buffer))
  write(*,*) "PASSED!"
end program hipsparse_csddmm
