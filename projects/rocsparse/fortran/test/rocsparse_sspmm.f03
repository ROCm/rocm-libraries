!!!!!!!!!!!!!/
! sspmm example (single-precision sparse-matrix times dense-matrix, C = alpha*A*B + beta*C, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
!
! Uses the generic API: build a CSR descriptor for A and dense-matrix
! descriptors for B and C, then run the three spmm stages
! (buffer_size -> preprocess -> compute). Result is checked against A*B.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly to the descriptor constructors / spmm; host data is moved
! with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program sspmm
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i, j

  ! Sparse A (3x3) in CSR (0-based):
  !   A = [[1,0,2],[0,3,0],[4,0,5]]
  integer(c_int), parameter :: M = 3, K = 3, Ncol = 2, nnz = 5

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_float), target :: h_csr_val(5)     = (/1, 2, 3, 4, 5/)

  ! Dense B (3x2), column-major
  real(c_float), target :: h_B(3,2) = reshape((/1, 2, 3, 4, 5, 6/), (/3, 2/))
  real(c_float), target :: h_C(3,2)
  ! Expected C = A*B
  real(c_float) :: h_expected(3,2) = reshape((/7, 6, 19, 16, 15, 46/), (/3, 2/))

  real(c_float), target :: alpha = 1.0, beta = 0.0

  integer(c_size_t) :: size_rp = 4, size_ci = 5, size_v = 5
  integer(c_size_t) :: size_B = 6, size_C = 6

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_B, d_C

  type(c_ptr) :: handle, matA, matB, matC, d_buffer
  integer(c_size_t) :: buffer_size

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_sspmm' (Fortran 2003 interfaces) - "

  ! Allocate device memory and copy inputs to device
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 4))
  call hipCheck(hipMalloc(d_B,           size_B * 4))
  call hipCheck(hipMalloc(d_C,           size_C * 4))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 4,  hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_B,           c_loc(h_B(1,1)),         size_B * 4,  hipMemcpyHostToDevice))

  ! Create rocSPARSE handle
  call rocsparseCheck(rocsparse_create_handle(handle))

  ! Descriptors: CSR for A, dense (column-major) for B and C
  call rocsparseCheck(rocsparse_create_csr_descr(matA, int(M,c_int64_t), int(K,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          rocsparse_indextype_i32, rocsparse_indextype_i32, &
                          rocsparse_index_base_zero, rocsparse_datatype_f32_r))
  call rocsparseCheck(rocsparse_create_dnmat_descr(matB, int(K,c_int64_t), int(Ncol,c_int64_t), int(K,c_int64_t), &
                          d_B, rocsparse_datatype_f32_r, rocsparse_order_column))
  call rocsparseCheck(rocsparse_create_dnmat_descr(matC, int(M,c_int64_t), int(Ncol,c_int64_t), int(M,c_int64_t), &
                          d_C, rocsparse_datatype_f32_r, rocsparse_order_column))

  ! Stage 1: query workspace size
  call rocsparseCheck(rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none, c_loc(alpha), &
                          matA, matB, c_loc(beta), matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, &
                          rocsparse_spmm_stage_buffer_size, buffer_size, c_null_ptr))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: preprocess
  call rocsparseCheck(rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none, c_loc(alpha), &
                          matA, matB, c_loc(beta), matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, &
                          rocsparse_spmm_stage_preprocess, buffer_size, d_buffer))

  ! Stage 3: compute
  call rocsparseCheck(rocsparse_spmm(handle, rocsparse_operation_none, rocsparse_operation_none, c_loc(alpha), &
                          matA, matB, c_loc(beta), matC, rocsparse_datatype_f32_r, rocsparse_spmm_alg_default, &
                          rocsparse_spmm_stage_compute, buffer_size, d_buffer))

  ! Copy result back to host
  call hipCheck(hipMemcpy(c_loc(h_C(1,1)), d_C, size_C * 4, hipMemcpyDeviceToHost))

  ! Verify C == A*B
  do j = 1,Ncol
    do i = 1,M
        error = abs(h_C(i,j) - h_expected(i,j)) / max(abs(h_expected(i,j)), 1.0)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " at (", i, ",", j, ")"
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_B))
  call hipCheck(hipFree(d_C))
  call hipCheck(hipFree(d_buffer))

  write(*,*) "PASSED!"

end program sspmm
