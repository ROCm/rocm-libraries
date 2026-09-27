!!!!!!!!!!!!!/
! ssddmm example (sampled dense-dense matrix multiplication, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
!
! SDDMM computes C = alpha * (A * B) .* spy(C) + beta * C, where A and B are
! dense and C is sparse (CSR): the dense product A*B is only evaluated at the
! nonzero positions of C. Uses the generic API with the three sddmm stages
! (buffer_size -> preprocess -> compute).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly to the descriptor constructors / sddmm.
!!!!!!!!!!!!!!/
!
program ssddmm
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  integer(c_int), parameter :: M = 3, N = 2, K = 3, nnz = 4

  real(c_float), target :: h_A(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3,3/))
  real(c_float), target :: h_B(3,2) = reshape((/1, 3, 5, 2, 4, 6/), (/3,2/))

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 1, 2, 4/)
  integer(c_int), target :: h_csr_col_ind(4) = (/0, 1, 0, 1/)
  real(c_float), target :: h_csr_val(4)     = (/0, 0, 0, 0/)

  real(c_float) :: h_AB(3,2), h_expected(4)
  real(c_float), target :: alpha = 1.0_c_float, beta = 0.0_c_float

  integer(c_size_t) :: size_rp = 4, size_ci = 4, size_v = 4
  integer(c_size_t) :: size_A = 9, size_B = 6

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_A, d_B

  type(c_ptr) :: handle, matA, matB, matC, d_buffer
  integer(c_size_t) :: buffer_size

  real :: error
  real, parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_ssddmm' (Fortran 2003 interfaces) - "

  h_AB = matmul(h_A, h_B)
  h_expected(1) = h_AB(1,1)
  h_expected(2) = h_AB(2,2)
  h_expected(3) = h_AB(3,1)
  h_expected(4) = h_AB(3,2)

  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 4))
  call hipCheck(hipMalloc(d_A,           size_A * 4))
  call hipCheck(hipMalloc(d_B,           size_B * 4))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_A,           c_loc(h_A(1,1)),         size_A * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_B,           c_loc(h_B(1,1)),         size_B * 4, hipMemcpyHostToDevice))

  call rocsparseCheck(rocsparse_create_handle(handle))

  call rocsparseCheck(rocsparse_create_dnmat_descr(matA, int(M,c_int64_t), int(K,c_int64_t), int(M,c_int64_t), &
                          d_A, rocsparse_datatype_f32_r, rocsparse_order_column))
  call rocsparseCheck(rocsparse_create_dnmat_descr(matB, int(K,c_int64_t), int(N,c_int64_t), int(K,c_int64_t), &
                          d_B, rocsparse_datatype_f32_r, rocsparse_order_column))
  call rocsparseCheck(rocsparse_create_csr_descr(matC, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          rocsparse_indextype_i32, rocsparse_indextype_i32, &
                          rocsparse_index_base_zero, rocsparse_datatype_f32_r))

  call rocsparseCheck(rocsparse_sddmm_buffer_size(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          c_loc(alpha), matA, matB, c_loc(beta), matC, rocsparse_datatype_f32_r, &
                          rocsparse_sddmm_alg_default, buffer_size))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  call rocsparseCheck(rocsparse_sddmm_preprocess(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          c_loc(alpha), matA, matB, c_loc(beta), matC, rocsparse_datatype_f32_r, &
                          rocsparse_sddmm_alg_default, d_buffer))

  call rocsparseCheck(rocsparse_sddmm(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          c_loc(alpha), matA, matB, c_loc(beta), matC, rocsparse_datatype_f32_r, &
                          rocsparse_sddmm_alg_default, d_buffer))

  call hipCheck(hipMemcpy(c_loc(h_csr_val(1)), d_csr_val, size_v * 4, hipMemcpyDeviceToHost))

  do i = 1,nnz
    error = abs(h_csr_val(i) - h_expected(i)) / max(abs(h_expected(i)), 1.0)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " at nnz ", i
        call exit(1)
    end if
  end do

  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_A))
  call hipCheck(hipFree(d_B))
  call hipCheck(hipFree(d_buffer))

  write(*,*) "PASSED!"

end program ssddmm
