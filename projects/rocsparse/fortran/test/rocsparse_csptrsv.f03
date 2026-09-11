!!!!!!!!!!!!!/
! csptrsv example (single-precision complex sparse triangular solve, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html
!
! Uses the generic spsv API on a lower-triangular L. Self-verifying: pick a
! known y, form x = L*y (dense product on host), solve L*y' = x, and confirm
! y' recovers y. Triangular structure set via spmat_set_attribute.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count and
! passed directly; host data (and the attribute values) are moved/passed with
! hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program csptrsv
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  ! Lower-triangular L (3x3) in CSR (0-based); complex values
  integer(c_int), parameter :: M = 3, N = 3, nnz = 6

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 1, 3, 6/)
  integer(c_int), target :: h_csr_col_ind(6) = (/0, 0, 1, 0, 1, 2/)
  complex(c_float_complex), target :: h_csr_val(6) = (/ (2.,1.), (1.,0.),(3.,-1.), (4.,1.),(5.,0.),(6.,2.) /)

  complex(c_float_complex) :: h_y(3) = (/ (1.,1.),(2.,-1.),(3.,0.) /)  ! known solution
  complex(c_float_complex), target :: h_x(3)          ! rhs = L*y
  complex(c_float_complex), target :: h_yout(3)       ! recovered solution
  complex(c_float_complex) :: L_dense(3,3)

  complex(c_float_complex), target :: alpha = (1.0,0.0)

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

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_csptrsv' (Fortran 2003 interfaces) - "

  ! Build dense L and form the consistent rhs x = L*y on the host
  L_dense = (0.0, 0.0)
  L_dense(1,1) = (2.,1.)
  L_dense(2,1) = (1.,0.); L_dense(2,2) = (3.,-1.)
  L_dense(3,1) = (4.,1.); L_dense(3,2) = (5.,0.); L_dense(3,3) = (6.,2.)
  h_x = matmul(L_dense, h_y)

  ! Allocate device memory and copy inputs
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 8))
  call hipCheck(hipMalloc(d_x,           size_x * 8))
  call hipCheck(hipMalloc(d_y,           size_y * 8))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 8,  hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_x,           c_loc(h_x(1)),           size_x * 8,  hipMemcpyHostToDevice))

  ! Create rocSPARSE handle
  call rocsparseCheck(rocsparse_create_handle(handle))

  ! CSR descriptor for L, marked lower-triangular / non-unit-diagonal
  call rocsparseCheck(rocsparse_create_csr_descr(matL, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          rocsparse_indextype_i32, rocsparse_indextype_i32, &
                          rocsparse_index_base_zero, rocsparse_datatype_f32_c))
  call rocsparseCheck(rocsparse_spmat_set_attribute(matL, rocsparse_spmat_fill_mode, c_loc(fill), int(4,c_size_t)))
  call rocsparseCheck(rocsparse_spmat_set_attribute(matL, rocsparse_spmat_diag_type, c_loc(diag), int(4,c_size_t)))

  ! Dense-vector descriptors for the rhs (x) and the solution (y)
  call rocsparseCheck(rocsparse_create_dnvec_descr(vecX, int(M,c_int64_t), d_x, rocsparse_datatype_f32_c))
  call rocsparseCheck(rocsparse_create_dnvec_descr(vecY, int(M,c_int64_t), d_y, rocsparse_datatype_f32_c))

  ! Stage 1: workspace size
  call rocsparseCheck(rocsparse_spsv(handle, rocsparse_operation_none, c_loc(alpha), matL, vecX, vecY, &
                          rocsparse_datatype_f32_c, rocsparse_spsv_alg_default, &
                          rocsparse_spsv_stage_buffer_size, buffer_size, c_null_ptr))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: preprocess (analysis)
  call rocsparseCheck(rocsparse_spsv(handle, rocsparse_operation_none, c_loc(alpha), matL, vecX, vecY, &
                          rocsparse_datatype_f32_c, rocsparse_spsv_alg_default, &
                          rocsparse_spsv_stage_preprocess, buffer_size, d_buffer))

  ! Stage 3: solve
  call rocsparseCheck(rocsparse_spsv(handle, rocsparse_operation_none, c_loc(alpha), matL, vecX, vecY, &
                          rocsparse_datatype_f32_c, rocsparse_spsv_alg_default, &
                          rocsparse_spsv_stage_compute, buffer_size, d_buffer))

  ! Copy the recovered solution back
  call hipCheck(hipMemcpy(c_loc(h_yout(1)), d_y, size_y * 8, hipMemcpyDeviceToHost))

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

end program csptrsv
