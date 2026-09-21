!!!!!!!!!!!!!!
! hipsparse Xcsr2coo example (CSR row pointers -> COO row indices, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Expands the CSR row-pointer array into one row index per nonzero and checks
! it against the expected COO row indices.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_xcsr2coo
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: M = 3, nnz = 5
  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_exp_coo_row(5) = (/0, 0, 1, 2, 2/)
  integer(c_int), target :: h_coo_row(5)
  integer(c_size_t) :: size_rp = 4, size_nz = 5
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: d_csr_row_ptr, d_coo_row
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_xcsr2coo' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_coo_row,     size_nz * 4))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseXcsr2coo(handle, d_csr_row_ptr, nnz, M, d_coo_row, HIPSPARSE_INDEX_BASE_ZERO))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(h_coo_row(1)), d_coo_row, size_nz * 4, hipMemcpyDeviceToHost))
  do i = 1, nnz
     if (h_coo_row(i) /= h_exp_coo_row(i)) then
        write(*,*) "FAILED! coo_row(", i, ") = ", h_coo_row(i), " expected ", h_exp_coo_row(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_csr_row_ptr)); call hipCheck(hipFree(d_coo_row))
  write(*,*) "PASSED!"
end program hipsparse_xcsr2coo
