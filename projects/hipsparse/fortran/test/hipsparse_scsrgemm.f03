!!!!!!!!!!!!!/
! scsrgemm example (single-precision sparse-matrix sparse-matrix multiply,
! C = A*B, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Two-phase flow with the classic hipSPARSE API: XcsrgemmNnz fills row_ptr_C and
! the total nnz, then Xcsrgemm computes the values. Here B = A, so C = A*A,
! checked against the known product.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc. nnzTotal is a host int (hipSPARSE default
! pointer mode). csrgemm's mat-descr/array args are c_ptr-only.
!!!!!!!!!!!!!!/
!
program scsrgemm
  use iso_c_binding
  use hip
  use hipsparse

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

  integer(c_int), target :: h_row_ptr_C(4)
  integer(c_int), target :: h_col_ind_C(5)
  real(c_float), target :: h_val_C(5)
  integer(c_int) :: nnz_C

  integer(c_size_t) :: size_rp = 4, size_nz = 5

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descr_A, descr_B, descr_C
  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_row_ptr_C, d_col_ind_C, d_val_C

  real(c_float) :: error
  real(c_float), parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_scsrgemm' (Fortran 2003 interfaces) - "

  ! Create handle and matrix descriptors
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateMatDescr(descr_A))
  call hipsparseCheck(hipsparseCreateMatDescr(descr_B))
  call hipsparseCheck(hipsparseCreateMatDescr(descr_C))

  ! Allocate device memory and copy A (B aliases A)
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_nz * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_nz * 4))
  call hipCheck(hipMalloc(d_row_ptr_C,   size_rp * 4))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_nz * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_nz * 4, hipMemcpyHostToDevice))

  ! Phase 1: compute the sparsity of C (row_ptr_C + total nnz_C)
  call hipsparseCheck(hipsparseXcsrgemmNnz(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          M, N, K, descr_A, nnz_A, d_csr_row_ptr, d_csr_col_ind, &
                          descr_B, nnz_A, d_csr_row_ptr, d_csr_col_ind, &
                          descr_C, d_row_ptr_C, nnz_C))

  ! Allocate C column indices and values now that nnz_C is known
  call hipCheck(hipMalloc(d_col_ind_C, int(nnz_C,c_size_t) * 4))
  call hipCheck(hipMalloc(d_val_C,     int(nnz_C,c_size_t) * 4))

  ! Phase 2: compute the values of C
  call hipsparseCheck(hipsparseScsrgemm(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
                          M, N, K, descr_A, nnz_A, d_csr_val, d_csr_row_ptr, d_csr_col_ind, &
                          descr_B, nnz_A, d_csr_val, d_csr_row_ptr, d_csr_col_ind, &
                          descr_C, d_val_C, d_row_ptr_C, d_col_ind_C))
  call hipCheck(hipDeviceSynchronize())

  ! Copy the C structure back to host
  call hipCheck(hipMemcpy(c_loc(h_row_ptr_C(1)), d_row_ptr_C, size_rp * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(h_col_ind_C(1)), d_col_ind_C, int(nnz_C,c_size_t) * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(h_val_C(1)),     d_val_C,     int(nnz_C,c_size_t) * 4, hipMemcpyDeviceToHost))

  ! Verify nnz, row pointers, column indices and values
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
  do i = 1,nnz_C
    if(h_col_ind_C(i) /= h_exp_col_ind(i)) then
        write(*,*) "FAILED! col_ind_C(", i, ") = ", h_col_ind_C(i), " expected ", h_exp_col_ind(i)
        call exit(1)
    end if
    error = abs(h_val_C(i) - h_exp_val(i)) / max(abs(h_exp_val(i)), 1.0_c_float)
    if(error .gt. error_max) then
        write(*,*) "FAILED! val_C(", i, ") = ", h_val_C(i), " expected ", h_exp_val(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(d_col_ind_C))
  call hipCheck(hipFree(d_val_C))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_row_ptr_C))
  call hipsparseCheck(hipsparseDestroyMatDescr(descr_A))
  call hipsparseCheck(hipsparseDestroyMatDescr(descr_B))
  call hipsparseCheck(hipsparseDestroyMatDescr(descr_C))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program scsrgemm
