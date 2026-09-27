!!!!!!!!!!!!!/
! dcsr2csc example (double-precision CSR -> CSC conversion / sparse transpose,
! Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Converting A from CSR to CSC is equivalent to producing the CSR of A**T.
! We check the resulting csc_col_ptr / csc_row_ind / csc_val against the known
! transpose.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc. csr2csc args are c_ptr-only.
!!!!!!!!!!!!!!/
!
program dcsr2csc
  use iso_c_binding
  use hip
  use hipsparse

  implicit none
  integer :: i

  ! 3x3 sparse matrix in CSR (0-based):  A = [[1,0,2],[0,3,0],[4,0,5]]
  integer(c_int), parameter :: M = 3, N = 3, nnz = 5

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_double), target :: h_csr_val(5)     = (/1, 2, 3, 4, 5/)

  ! Expected CSC (= CSR of the transpose)
  integer(c_int) :: h_exp_col_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_exp_row_ind(5) = (/0, 2, 1, 0, 2/)
  real(c_double) :: h_exp_val(5)     = (/1, 4, 3, 2, 5/)

  integer(c_int), target :: h_csc_col_ptr(4)
  integer(c_int), target :: h_csc_row_ind(5)
  real(c_double), target :: h_csc_val(5)

  integer(c_size_t) :: size_rp = 4, size_nz = 5

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_csc_col_ptr, d_csc_row_ind, d_csc_val

  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'hipsparse_dcsr2csc' (Fortran 2003 interfaces) - "

  ! Allocate device memory and copy the CSR matrix to device
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_nz * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_nz * 8))
  call hipCheck(hipMalloc(d_csc_col_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csc_row_ind, size_nz * 4))
  call hipCheck(hipMalloc(d_csc_val,     size_nz * 8))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_nz * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_nz * 8, hipMemcpyHostToDevice))

  ! Create handle and convert CSR -> CSC (numeric: also permute values)
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseDcsr2csc(handle, M, N, nnz, &
                          d_csr_val, d_csr_row_ptr, d_csr_col_ind, &
                          d_csc_val, d_csc_row_ind, d_csc_col_ptr, &
                          HIPSPARSE_ACTION_NUMERIC, HIPSPARSE_INDEX_BASE_ZERO))
  call hipCheck(hipDeviceSynchronize())

  ! Copy the result back to host
  call hipCheck(hipMemcpy(c_loc(h_csc_col_ptr(1)), d_csc_col_ptr, size_rp * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(h_csc_row_ind(1)), d_csc_row_ind, size_nz * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(h_csc_val(1)),     d_csc_val,     size_nz * 8, hipMemcpyDeviceToHost))

  ! Verify structure and values
  do i = 1,N+1
    if(h_csc_col_ptr(i) /= h_exp_col_ptr(i)) then
        write(*,*) "FAILED! csc_col_ptr(", i, ") = ", h_csc_col_ptr(i), " expected ", h_exp_col_ptr(i)
        call exit(1)
    end if
  end do
  do i = 1,nnz
    if(h_csc_row_ind(i) /= h_exp_row_ind(i)) then
        write(*,*) "FAILED! csc_row_ind(", i, ") = ", h_csc_row_ind(i), " expected ", h_exp_row_ind(i)
        call exit(1)
    end if
    error = abs(h_csc_val(i) - h_exp_val(i))
    if(error .gt. error_max) then
        write(*,*) "FAILED! csc_val(", i, ") = ", h_csc_val(i), " expected ", h_exp_val(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_csc_col_ptr))
  call hipCheck(hipFree(d_csc_row_ind))
  call hipCheck(hipFree(d_csc_val))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dcsr2csc
