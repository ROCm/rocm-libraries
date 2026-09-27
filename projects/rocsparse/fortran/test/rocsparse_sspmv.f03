!!!!!!!!!!!!!/
! sspmv example (single-precision sparse-matrix dense-vector multiply, y = alpha*A*x + beta*y, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
!
! Uses the generic API: build a CSR descriptor for A and dense-vector
! descriptors for x and y, then run the three spmv stages
! (buffer_size -> preprocess -> compute). Result is checked against A*x.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly to the descriptor constructors / spmv; host data is moved
! with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program sspmv
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  ! Sparse A (3x3) in CSR (0-based):  A = [[1,0,2],[0,3,0],[4,0,5]]
  integer(c_int), parameter :: M = 3, N = 3, nnz = 5

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_float), target :: h_csr_val(5)     = (/1, 2, 3, 4, 5/)

  real(c_float), target :: h_x(3) = (/1, 2, 3/)
  real(c_float), target :: h_y(3)
  real(c_float) :: h_expected(3) = (/7, 6, 19/)   ! A*x

  real(c_float), target :: alpha = 1.0, beta = 0.0

  integer(c_size_t) :: size_rp = 4, size_ci = 5, size_v = 5
  integer(c_size_t) :: size_x = 3, size_y = 3

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_x, d_y

  type(c_ptr) :: handle, matA, vecX, vecY, d_buffer
  integer(c_size_t) :: buffer_size

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_sspmv' (Fortran 2003 interfaces) - "

  ! Allocate device memory and copy inputs to device
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 4))
  call hipCheck(hipMalloc(d_x,           size_x * 4))
  call hipCheck(hipMalloc(d_y,           size_y * 4))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 4,  hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_x,           c_loc(h_x(1)),           size_x * 4,  hipMemcpyHostToDevice))

  ! Create rocSPARSE handle
  call rocsparseCheck(rocsparse_create_handle(handle))

  ! Descriptors: CSR for A, dense vectors for x and y
  call rocsparseCheck(rocsparse_create_csr_descr(matA, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          rocsparse_indextype_i32, rocsparse_indextype_i32, &
                          rocsparse_index_base_zero, rocsparse_datatype_f32_r))
  call rocsparseCheck(rocsparse_create_dnvec_descr(vecX, int(N,c_int64_t), d_x, rocsparse_datatype_f32_r))
  call rocsparseCheck(rocsparse_create_dnvec_descr(vecY, int(M,c_int64_t), d_y, rocsparse_datatype_f32_r))

  ! Stage 1: query workspace size
  call rocsparseCheck(rocsparse_spmv(handle, rocsparse_operation_none, c_loc(alpha), matA, vecX, c_loc(beta), vecY, &
                          rocsparse_datatype_f32_r, rocsparse_spmv_alg_default, &
                          rocsparse_spmv_stage_buffer_size, buffer_size, c_null_ptr))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: preprocess
  call rocsparseCheck(rocsparse_spmv(handle, rocsparse_operation_none, c_loc(alpha), matA, vecX, c_loc(beta), vecY, &
                          rocsparse_datatype_f32_r, rocsparse_spmv_alg_default, &
                          rocsparse_spmv_stage_preprocess, buffer_size, d_buffer))

  ! Stage 3: compute
  call rocsparseCheck(rocsparse_spmv(handle, rocsparse_operation_none, c_loc(alpha), matA, vecX, c_loc(beta), vecY, &
                          rocsparse_datatype_f32_r, rocsparse_spmv_alg_default, &
                          rocsparse_spmv_stage_compute, buffer_size, d_buffer))

  ! Copy result back to host
  call hipCheck(hipMemcpy(c_loc(h_y(1)), d_y, size_y * 4, hipMemcpyDeviceToHost))

  ! Verify y == A*x
  do i = 1,M
    error = abs(h_y(i) - h_expected(i)) / max(abs(h_expected(i)), 1.0)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " y(", i, ") = ", h_y(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_x))
  call hipCheck(hipFree(d_y))
  call hipCheck(hipFree(d_buffer))

  write(*,*) "PASSED!"

end program sspmv
