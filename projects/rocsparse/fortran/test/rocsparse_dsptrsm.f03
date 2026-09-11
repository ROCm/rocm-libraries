!!!!!!!!!!!!!/
! dsptrsm example (double-precision sparse triangular solve with multiple rhs, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
!
! Uses the generic spsm API on a lower-triangular L with a dense rhs matrix.
! Self-verifying: pick a known Y, form X = L*Y, solve L*C = X, and confirm C
! recovers Y. Triangular structure set via spmat_set_attribute. Three stages:
! buffer_size -> preprocess -> compute.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly; host data (and the attribute values) are moved/passed with
! hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program dsptrsm
  use iso_c_binding
  use hip
  use rocsparse

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

  integer(kind(rocsparse_fill_mode_lower)), target :: fill = rocsparse_fill_mode_lower
  integer(kind(rocsparse_diag_type_non_unit)), target :: diag = rocsparse_diag_type_non_unit

  integer(c_size_t) :: size_rp = 4, size_ci = 6, size_v = 6
  integer(c_size_t) :: size_X = 6, size_C = 6

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_X, d_C

  type(c_ptr) :: handle, matL, matB, matC, d_buffer
  integer(c_size_t) :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_dsptrsm' (Fortran 2003 interfaces) - "

  ! Build dense L and the consistent rhs X = L*Y on the host
  L_dense = 0.0_c_double
  L_dense(1,1) = 2.0_c_double
  L_dense(2,1) = 1.0_c_double; L_dense(2,2) = 3.0_c_double
  L_dense(3,1) = 4.0_c_double; L_dense(3,2) = 5.0_c_double; L_dense(3,3) = 6.0_c_double
  h_X = matmul(L_dense, h_Y)

  ! Allocate device memory and copy inputs
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 8))
  call hipCheck(hipMalloc(d_X,           size_X * 8))
  call hipCheck(hipMalloc(d_C,           size_C * 8))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 8,  hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_X,           c_loc(h_X(1,1)),         size_X * 8,  hipMemcpyHostToDevice))

  ! Create rocSPARSE handle
  call rocsparseCheck(rocsparse_create_handle(handle))

  ! CSR descriptor for L, marked lower-triangular / non-unit-diagonal
  call rocsparseCheck(rocsparse_create_csr_descr(matL, int(M,c_int64_t), int(M,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          rocsparse_indextype_i32, rocsparse_indextype_i32, &
                          rocsparse_index_base_zero, rocsparse_datatype_f64_r))
  call rocsparseCheck(rocsparse_spmat_set_attribute(matL, rocsparse_spmat_fill_mode, c_loc(fill), int(4,c_size_t)))
  call rocsparseCheck(rocsparse_spmat_set_attribute(matL, rocsparse_spmat_diag_type, c_loc(diag), int(4,c_size_t)))

  ! Dense-matrix descriptors: B is the rhs (X), C is the solution
  call rocsparseCheck(rocsparse_create_dnmat_descr(matB, int(M,c_int64_t), int(nrhs,c_int64_t), int(M,c_int64_t), &
                          d_X, rocsparse_datatype_f64_r, rocsparse_order_column))
  call rocsparseCheck(rocsparse_create_dnmat_descr(matC, int(M,c_int64_t), int(nrhs,c_int64_t), int(M,c_int64_t), &
                          d_C, rocsparse_datatype_f64_r, rocsparse_order_column))

  ! Stage 1: workspace size
  call rocsparseCheck(rocsparse_spsm(handle, rocsparse_operation_none, rocsparse_operation_none, c_loc(alpha), &
                          matL, matB, matC, rocsparse_datatype_f64_r, rocsparse_spsm_alg_default, &
                          rocsparse_spsm_stage_buffer_size, buffer_size, c_null_ptr))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: preprocess (analysis)
  call rocsparseCheck(rocsparse_spsm(handle, rocsparse_operation_none, rocsparse_operation_none, c_loc(alpha), &
                          matL, matB, matC, rocsparse_datatype_f64_r, rocsparse_spsm_alg_default, &
                          rocsparse_spsm_stage_preprocess, buffer_size, d_buffer))

  ! Stage 3: solve
  call rocsparseCheck(rocsparse_spsm(handle, rocsparse_operation_none, rocsparse_operation_none, c_loc(alpha), &
                          matL, matB, matC, rocsparse_datatype_f64_r, rocsparse_spsm_alg_default, &
                          rocsparse_spsm_stage_compute, buffer_size, d_buffer))

  ! Copy the recovered solution back
  call hipCheck(hipMemcpy(c_loc(h_C(1,1)), d_C, size_C * 8, hipMemcpyDeviceToHost))

  ! Verify C == Y
  do j = 1,nrhs
    do i = 1,M
        error = abs(h_C(i,j) - h_Y(i,j)) / max(abs(h_Y(i,j)), 1.0_c_double)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " at (", i, ",", j, ")"
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call rocsparseCheck(rocsparse_destroy_spmat_descr(matL))
  call rocsparseCheck(rocsparse_destroy_dnmat_descr(matB))
  call rocsparseCheck(rocsparse_destroy_dnmat_descr(matC))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_X))
  call hipCheck(hipFree(d_C))
  call hipCheck(hipFree(d_buffer))

  write(*,*) "PASSED!"

end program dsptrsm
