!!!!!!!!!!!!!/
! scsrgemm example (single-precision sparse-matrix sparse-matrix multiply, C = alpha*A*B, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/extra.html
!
! Two-phase flow: csrgemm_buffer_size -> csrgemm_nnz (fills row_ptr_C and
! nnz_C) -> allocate col_ind_C/val_C -> csrgemm (computes C). Here B = A, so
! C = A*A, checked against the known product. The optional D term is unused
! (descr_D / arrays passed as null, beta = 0).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly; host data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program scsrgemm
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  ! Sparse A (3x3) in CSR (0-based):  A = [[1,0,2],[0,3,0],[4,0,5]]
  ! C = A*A = [[9,0,12],[0,9,0],[24,0,33]]
  integer(c_int), parameter :: M = 3, N = 3, K = 3, nnz_A = 5

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_float), target :: h_csr_val(5)     = (/1, 2, 3, 4, 5/)

  ! Expected C
  integer(c_int) :: h_exp_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_exp_col_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_float) :: h_exp_val(5)     = (/9, 12, 9, 24, 33/)

  real(c_float), target :: alpha = 1.0, beta = 0.0

  integer(c_int), target :: h_row_ptr_C(4)
  integer(c_int), target :: nnz_C

  integer(c_size_t) :: size_rp = 4, size_ci = 5, size_v = 5

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_row_ptr_C, d_col_ind_C, d_val_C
  type(c_ptr) :: d_nnz_C

  type(c_ptr) :: handle, descr_A, descr_B, descr_C, info_C, d_buffer
  integer(c_size_t) :: buffer_size

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_scsrgemm' (Fortran 2003 interfaces) - "

  ! Create handle, matrix descriptors and the csrgemm info object
  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_A))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_B))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_C))
  call rocsparseCheck(rocsparse_create_mat_info(info_C))

  ! Copy A to device (B aliases A) and allocate row_ptr_C + nnz_C
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 4))
  call hipCheck(hipMalloc(d_row_ptr_C,   size_rp * 4))
  call hipCheck(hipMalloc(d_nnz_C,       4_c_size_t))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 4,  hipMemcpyHostToDevice))

  ! Phase 0: workspace size
  call rocsparseCheck(rocsparse_scsrgemm_buffer_size(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          M, N, K, alpha, descr_A, nnz_A, d_csr_row_ptr, d_csr_col_ind, &
                          descr_B, nnz_A, d_csr_row_ptr, d_csr_col_ind, &
                          beta, c_null_ptr, 0, c_null_ptr, c_null_ptr, info_C, buffer_size))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Phase 1: compute the sparsity of C (row_ptr_C + total nnz_C)
  call rocsparseCheck(rocsparse_csrgemm_nnz(handle, rocsparse_operation_none, rocsparse_operation_none, M, N, K, &
                          descr_A, nnz_A, d_csr_row_ptr, d_csr_col_ind, &
                          descr_B, nnz_A, d_csr_row_ptr, d_csr_col_ind, &
                          c_null_ptr, 0, c_null_ptr, c_null_ptr, &
                          descr_C, d_row_ptr_C, d_nnz_C, info_C, d_buffer))
  call hipCheck(hipMemcpy(c_loc(nnz_C), d_nnz_C, 4_c_size_t, hipMemcpyDeviceToHost))

  ! Allocate C column indices and values now that nnz_C is known
  call hipCheck(hipMalloc(d_col_ind_C, int(nnz_C,c_size_t) * 4))
  call hipCheck(hipMalloc(d_val_C,     int(nnz_C,c_size_t) * 4))

  ! Phase 2: compute the values of C
  call rocsparseCheck(rocsparse_scsrgemm(handle, rocsparse_operation_none, rocsparse_operation_none, M, N, K, alpha, &
                          descr_A, nnz_A, d_csr_val, d_csr_row_ptr, d_csr_col_ind, &
                          descr_B, nnz_A, d_csr_val, d_csr_row_ptr, d_csr_col_ind, &
                          beta, c_null_ptr, 0, c_null_ptr, c_null_ptr, c_null_ptr, &
                          descr_C, d_val_C, d_row_ptr_C, d_col_ind_C, info_C, d_buffer))
  call hipCheck(hipDeviceSynchronize())

  ! Copy the C structure back to host
  call hipCheck(hipMemcpy(c_loc(h_row_ptr_C(1)), d_row_ptr_C, size_rp * 4, hipMemcpyDeviceToHost))

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
    integer(c_int), target :: h_col_ind_C(nnz_C)
    real(c_float),  target :: h_val_C(nnz_C)
    call hipCheck(hipMemcpy(c_loc(h_col_ind_C(1)), d_col_ind_C, int(nnz_C,c_size_t) * 4, hipMemcpyDeviceToHost))
    call hipCheck(hipMemcpy(c_loc(h_val_C(1)),     d_val_C,     int(nnz_C,c_size_t) * 4, hipMemcpyDeviceToHost))
    do i = 1,nnz_C
      if(h_col_ind_C(i) /= h_exp_col_ind(i)) then
          write(*,*) "FAILED! col_ind_C(", i, ") = ", h_col_ind_C(i), " expected ", h_exp_col_ind(i)
          call exit(1)
      end if
      error = abs(h_val_C(i) - h_exp_val(i)) / max(abs(h_exp_val(i)), 1.0)
      if(error .gt. error_max) then
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
  call hipCheck(hipFree(d_nnz_C))
  call hipCheck(hipFree(d_buffer))
  call rocsparseCheck(rocsparse_destroy_mat_info(info_C))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_A))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_B))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_C))
  call rocsparseCheck(rocsparse_destroy_handle(handle))

  write(*,*) "PASSED!"

end program scsrgemm
