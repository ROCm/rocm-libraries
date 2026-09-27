!!!!!!!!!!!!!/
! ssptrsv example (single-precision sparse triangular solve, op(A)*y = alpha*x, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
!
! Uses the generic spsv API on a lower-triangular L. Self-verifying: pick a
! known y, form x = L*y, solve L*y' = x, and confirm y' recovers y.
! The triangular structure is set via spmat_set_attribute (fill_mode +
! diag_type). Three stages: buffer_size -> preprocess -> compute.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly; host data (and the attribute values) are moved/passed with
! hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program ssptrsv
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  ! Lower-triangular L (3x3) in CSR (0-based):
  !   L = [[2,0,0],[1,3,0],[4,5,6]]
  integer(c_int), parameter :: M = 3, N = 3, nnz = 6

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 1, 3, 6/)
  integer(c_int), target :: h_csr_col_ind(6) = (/0, 0, 1, 0, 1, 2/)
  real(c_float), target :: h_csr_val(6)     = (/2, 1, 3, 4, 5, 6/)

  real(c_float) :: h_y(3) = (/1, 2, 3/)   ! known solution
  real(c_float), target :: h_x(3)                  ! rhs = L*y
  real(c_float), target :: h_yout(3)               ! recovered solution

  real(c_float), target :: alpha = 1.0

  integer(kind(rocsparse_fill_mode_lower)), target :: fill = rocsparse_fill_mode_lower
  integer(kind(rocsparse_diag_type_non_unit)), target :: diag = rocsparse_diag_type_non_unit

  integer(c_size_t) :: size_rp = 4, size_ci = 6, size_v = 6
  integer(c_size_t) :: size_x = 3, size_y = 3

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_x, d_y

  type(c_ptr) :: handle, matL, vecX, vecY, d_buffer
  integer(c_size_t) :: buffer_size

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_ssptrsv' (Fortran 2003 interfaces) - "

  ! Build a consistent rhs so that L*y = x:  x1=2, x2=1+6=7, x3=4+10+18=32
  h_x(1) = 2.0
  h_x(2) = 7.0
  h_x(3) = 32.0

  ! Allocate device memory and copy inputs
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

  ! CSR descriptor for L, marked lower-triangular / non-unit-diagonal
  call rocsparseCheck(rocsparse_create_csr_descr(matL, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          rocsparse_indextype_i32, rocsparse_indextype_i32, &
                          rocsparse_index_base_zero, rocsparse_datatype_f32_r))
  call rocsparseCheck(rocsparse_spmat_set_attribute(matL, rocsparse_spmat_fill_mode, c_loc(fill), int(4,c_size_t)))
  call rocsparseCheck(rocsparse_spmat_set_attribute(matL, rocsparse_spmat_diag_type, c_loc(diag), int(4,c_size_t)))

  ! Dense-vector descriptors for the rhs (x) and the solution (y)
  call rocsparseCheck(rocsparse_create_dnvec_descr(vecX, int(M,c_int64_t), d_x, rocsparse_datatype_f32_r))
  call rocsparseCheck(rocsparse_create_dnvec_descr(vecY, int(M,c_int64_t), d_y, rocsparse_datatype_f32_r))

  ! Stage 1: workspace size
  call rocsparseCheck(rocsparse_spsv(handle, rocsparse_operation_none, c_loc(alpha), matL, vecX, vecY, &
                          rocsparse_datatype_f32_r, rocsparse_spsv_alg_default, &
                          rocsparse_spsv_stage_buffer_size, buffer_size, c_null_ptr))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: preprocess (analysis)
  call rocsparseCheck(rocsparse_spsv(handle, rocsparse_operation_none, c_loc(alpha), matL, vecX, vecY, &
                          rocsparse_datatype_f32_r, rocsparse_spsv_alg_default, &
                          rocsparse_spsv_stage_preprocess, buffer_size, d_buffer))

  ! Stage 3: solve
  call rocsparseCheck(rocsparse_spsv(handle, rocsparse_operation_none, c_loc(alpha), matL, vecX, vecY, &
                          rocsparse_datatype_f32_r, rocsparse_spsv_alg_default, &
                          rocsparse_spsv_stage_compute, buffer_size, d_buffer))

  ! Copy the recovered solution back
  call hipCheck(hipMemcpy(c_loc(h_yout(1)), d_y, size_y * 4, hipMemcpyDeviceToHost))

  ! Verify y' == y
  do i = 1,M
    error = abs(h_yout(i) - h_y(i)) / max(abs(h_y(i)), 1.0)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " y(", i, ") = ", h_yout(i)
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

end program ssptrsv
