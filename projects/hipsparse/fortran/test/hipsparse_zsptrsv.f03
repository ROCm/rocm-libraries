!!!!!!!!!!!!!/
! zsptrsv example (double-complex sparse triangular solve, op(A)*y = alpha*x,
! Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Uses the generic SpSV API on a lower-triangular L. Self-verifying: pick a
! known y, form x = L*y (dense), solve L*y' = x, and confirm y' recovers y.
! hipSPARSE flow: createDescr -> bufferSize -> analysis -> solve.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc. Descriptor/array args are c_ptr-only.
!!!!!!!!!!!!!!/
!
program zsptrsv
  use iso_c_binding
  use hip
  use hipsparse

  implicit none
  integer :: i

  ! Lower-triangular L (3x3) in CSR (0-based); complex values
  integer(c_int), parameter :: M = 3, N = 3, nnz = 6

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 1, 3, 6/)
  integer(c_int), target :: h_csr_col_ind(6) = (/0, 0, 1, 0, 1, 2/)
  complex(c_double_complex), target :: h_csr_val(6) = &
    (/ (2.d0,1.d0), (1.d0,0.d0),(3.d0,-1.d0), (4.d0,1.d0),(5.d0,0.d0),(6.d0,2.d0) /)

  complex(c_double_complex) :: h_y(3) = (/ (1.d0,1.d0),(2.d0,-1.d0),(3.d0,0.d0) /)  ! known solution
  complex(c_double_complex), target :: h_x(3)          ! rhs = L*y
  complex(c_double_complex), target :: h_yout(3)       ! recovered solution
  complex(c_double_complex) :: L_dense(3,3)

  complex(c_double_complex), target :: alpha = (1.0d0,0.0d0)
  integer(kind(HIPSPARSE_FILL_MODE_LOWER)), target :: fill = HIPSPARSE_FILL_MODE_LOWER
  integer(kind(HIPSPARSE_DIAG_TYPE_NON_UNIT)), target :: diag = HIPSPARSE_DIAG_TYPE_NON_UNIT

  integer(c_size_t) :: size_rp = 4, size_nz = 6, size_v = 3

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: matL, vecX, vecY, spsvDescr, d_buffer
  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val, d_x, d_y
  integer(c_size_t), target :: buffer_size

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zsptrsv' (Fortran 2003 interfaces) - "

  ! Build a consistent rhs densely so that L*y = x
  L_dense = (0.d0, 0.d0)
  L_dense(1,1) = (2.d0,1.d0)
  L_dense(2,1) = (1.d0,0.d0); L_dense(2,2) = (3.d0,-1.d0)
  L_dense(3,1) = (4.d0,1.d0); L_dense(3,2) = (5.d0,0.d0); L_dense(3,3) = (6.d0,2.d0)
  h_x = matmul(L_dense, h_y)

  ! Allocate device memory and copy inputs
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_nz * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_nz * 16))
  call hipCheck(hipMalloc(d_x,           size_v * 16))
  call hipCheck(hipMalloc(d_y,           size_v * 16))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_nz * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_nz * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_x,           c_loc(h_x(1)),           size_v * 16, hipMemcpyHostToDevice))

  ! Create handle, CSR descriptor for L (lower / non-unit diag), dense vectors
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateCsr(matL, int(M,c_int64_t), int(N,c_int64_t), int(nnz,c_int64_t), &
                          d_csr_row_ptr, d_csr_col_ind, d_csr_val, &
                          HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_C_64F))
  call hipsparseCheck(hipsparseSpMatSetAttribute(matL, HIPSPARSE_SPMAT_FILL_MODE, c_loc(fill), int(4,c_size_t)))
  call hipsparseCheck(hipsparseSpMatSetAttribute(matL, HIPSPARSE_SPMAT_DIAG_TYPE, c_loc(diag), int(4,c_size_t)))
  call hipsparseCheck(hipsparseCreateDnVec(vecX, int(M,c_int64_t), d_x, HIP_C_64F))
  call hipsparseCheck(hipsparseCreateDnVec(vecY, int(M,c_int64_t), d_y, HIP_C_64F))
  call hipsparseCheck(hipsparseSpSV_createDescr(spsvDescr))

  ! Stage 1: workspace size
  call hipsparseCheck(hipsparseSpSV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matL, vecX, vecY, &
                          HIP_C_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr, buffer_size))
  call hipCheck(hipMalloc(d_buffer, max(buffer_size, 1_c_size_t)))

  ! Stage 2: analysis
  call hipsparseCheck(hipsparseSpSV_analysis(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matL, vecX, vecY, &
                          HIP_C_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr, d_buffer))

  ! Stage 3: solve
  call hipsparseCheck(hipsparseSpSV_solve(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, c_loc(alpha), matL, vecX, vecY, &
                          HIP_C_64F, HIPSPARSE_SPSV_ALG_DEFAULT, spsvDescr))

  ! Copy the recovered solution back
  call hipCheck(hipMemcpy(c_loc(h_yout(1)), d_y, size_v * 16, hipMemcpyDeviceToHost))

  ! Verify y' == y
  do i = 1,M
    error = abs(h_yout(i) - h_y(i)) / max(abs(h_y(i)), 1.0_c_double)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " y(", i, ") = ", h_yout(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipsparseCheck(hipsparseSpSV_destroyDescr(spsvDescr))
  call hipsparseCheck(hipsparseDestroyDnVec(vecX))
  call hipsparseCheck(hipsparseDestroyDnVec(vecY))
  call hipsparseCheck(hipsparseDestroySpMat(matL))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_x))
  call hipCheck(hipFree(d_y))
  call hipCheck(hipFree(d_buffer))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program zsptrsv
