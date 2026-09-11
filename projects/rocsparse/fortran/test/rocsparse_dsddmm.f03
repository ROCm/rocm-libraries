!!!!!!!!!!!!!/
! dsddmm example (double-precision sampled dense-dense matrix multiplication, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
!
! SDDMM computes C = alpha * (A * B) .* spy(C) + beta * C, where A and B are
! dense and C is sparse (CSR): the dense product A*B is only evaluated at the
! nonzero positions of C. Uses the generic API with the three sddmm stages
! (buffer_size -> preprocess -> compute). Result is checked against a host
! reference that samples matmul(A,B) at C's pattern.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly to the descriptor constructors / sddmm; host data is moved
! with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program dsddmm
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  ! Dense A (M x K) and B (K x N), column-major. Sparse C (M x N) in CSR.
  integer(c_int), parameter :: M = 3, N = 2, K = 3, nnz = 4

  ! A = [[1,2,3],[4,5,6],[7,8,10]]  (column-major storage)
  real(c_double), target :: h_A(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3,3/))
  ! B = [[1,2],[3,4],[5,6]]         (column-major storage)
  real(c_double), target :: h_B(3,2) = reshape((/1, 3, 5, 2, 4, 6/), (/3,2/))

  ! Sparse C pattern (0-based CSR), C = [[*,0],[0,*],[*,*]]: nonzeros at
  ! (0,0),(1,1),(2,0),(2,1). Values start at 0 (beta = 0 so they are ignored).
  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 1, 2, 4/)
  integer(c_int), target :: h_csr_col_ind(4) = (/0, 1, 0, 1/)
  real(c_double), target :: h_csr_val(4)     = (/0, 0, 0, 0/)

  real(c_double) :: h_AB(3,2), h_expected(4)
  real(c_double), target :: alpha = 1.0_c_double, beta = 0.0_c_double

  integer(c_size_t) :: size_rp = 4, size_ci = 4, size_v = 4
  integer(c_size_t) :: size_A = 9, size_B = 6

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_A, d_B

  type(c_ptr) :: handle, matA, matB, matC, d_buffer
  integer(c_size_t) :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_dsddmm' (Fortran 2003 interfaces) - "

  ! Host reference: sample matmul(A,B) at C's nonzero pattern (row-major walk of CSR)
  h_AB = matmul(h_A, h_B)
  h_expected(1) = h_AB(1,1)   ! (0,0)
  h_expected(2) = h_AB(2,2)   ! (1,1)
  h_expected(3) = h_AB(3,1)   ! (2,0)
  h_expected(4) = h_AB(3,2)   ! (2,1)

  ! Allocate device memory and copy inputs to device
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 8))
  call hipCheck(hipMalloc(d_A,           size_A * 8))
  call hipCheck(hipMalloc(d_B,           size_B * 8))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 8,  hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_A,           c_loc(h_A(1,1)),         size_A * 8,  hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_B,           c_loc(h_B(1,1)),         size_B * 8,  hipMemcpyHostToDevice))

  ! Create rocSPARSE handle
  call rocsparseCheck(rocsparse_create_handle(handle))

  ! Descriptors: dense (column-major) for A and B, CSR for C
  call rocsparseCheck(rocsparse_create_dnmat_descr(matA, int(M,c_int64_t), int(K,c_int64_t), int(M,c_int64_t), &
                          d_A, rocsparse_datatype_f64_r, rocsparse_order_column))
  call rocsparseCheck(rocsparse_create_dnmat_descr(matB, int(K,c_int64_t), int(N,c_int64_t), int(K,c_int64_t), &
                          d_B, rocsparse_datatype_f64_r, rocsparse_order_column))
  call rocsparseCheck(rocsparse_create_csr_descr(matC, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          rocsparse_indextype_i32, rocsparse_indextype_i32, &
                          rocsparse_index_base_zero, rocsparse_datatype_f64_r))

  ! Stage 1: query workspace size
  call rocsparseCheck(rocsparse_sddmm_buffer_size(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          c_loc(alpha), matA, matB, c_loc(beta), matC, rocsparse_datatype_f64_r, &
                          rocsparse_sddmm_alg_default, buffer_size))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: preprocess
  call rocsparseCheck(rocsparse_sddmm_preprocess(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          c_loc(alpha), matA, matB, c_loc(beta), matC, rocsparse_datatype_f64_r, &
                          rocsparse_sddmm_alg_default, d_buffer))

  ! Stage 3: compute
  call rocsparseCheck(rocsparse_sddmm(handle, rocsparse_operation_none, rocsparse_operation_none, &
                          c_loc(alpha), matA, matB, c_loc(beta), matC, rocsparse_datatype_f64_r, &
                          rocsparse_sddmm_alg_default, d_buffer))

  ! Copy the sparse values back to host
  call hipCheck(hipMemcpy(c_loc(h_csr_val(1)), d_csr_val, size_v * 8, hipMemcpyDeviceToHost))

  ! Verify sampled values
  do i = 1,nnz
    error = abs(h_csr_val(i) - h_expected(i)) / max(abs(h_expected(i)), 1.0_c_double)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " at nnz ", i, &
                   " got ", h_csr_val(i), " expected ", h_expected(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call rocsparseCheck(rocsparse_destroy_dnmat_descr(matA))
  call rocsparseCheck(rocsparse_destroy_dnmat_descr(matB))
  call rocsparseCheck(rocsparse_destroy_spmat_descr(matC))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_A))
  call hipCheck(hipFree(d_B))
  call hipCheck(hipFree(d_buffer))

  write(*,*) "PASSED!"

end program dsddmm
