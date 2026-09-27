!!!!!!!!!!!!!!
! hipsparse Xcoo2csr example (COO row indices -> CSR row pointers, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Compresses the per-nonzero COO row-index array into a CSR row-pointer array
! and checks it against the expected offsets. Inverse of Xcsr2coo.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_xcoo2csr
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: M = 3, nnz = 5
  integer(c_int), target :: h_coo_row(5) = (/0, 0, 1, 2, 2/)
  integer(c_int) :: h_exp_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_row_ptr(4)
  integer(c_size_t) :: size_rp = 4, size_nz = 5
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: d_coo_row, d_csr_row_ptr
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_xcoo2csr' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(d_coo_row,     size_nz * 4))
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMemcpy(d_coo_row, c_loc(h_coo_row(1)), size_nz * 4, hipMemcpyHostToDevice))
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseXcoo2csr(handle, d_coo_row, nnz, M, d_csr_row_ptr, HIPSPARSE_INDEX_BASE_ZERO))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(h_csr_row_ptr(1)), d_csr_row_ptr, size_rp * 4, hipMemcpyDeviceToHost))
  do i = 1, M + 1
     if (h_csr_row_ptr(i) /= h_exp_csr_row_ptr(i)) then
        write(*,*) "FAILED! csr_row_ptr(", i, ") = ", h_csr_row_ptr(i), " expected ", h_exp_csr_row_ptr(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(d_coo_row)); call hipCheck(hipFree(d_csr_row_ptr))
  write(*,*) "PASSED!"
end program hipsparse_xcoo2csr
