!!!!!!!!!!!!!/
! csr2coo example (CSR -> COO row-index conversion, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/conversion.html
!
! csr2coo expands the CSR row-pointer array into per-nonzero COO row indices.
! It is integer-only (no s/d/c/z variants).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program csr2coo
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  integer(c_int), parameter :: M = 3      ! number of rows
  integer(c_int), parameter :: nnz = 4    ! number of nonzeros

  ! CSR row-pointer for a 3x3 matrix with 2, 1, 1 nonzeros per row (0-based)
  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 4/)
  integer(c_int), target :: h_coo_row_ind(4)
  integer(c_int) :: h_expected(4) = (/0, 0, 1, 2/)

  integer(c_size_t) :: size_rp = 4, size_ci = 4

  type(c_ptr) :: d_csr_row_ptr
  type(c_ptr) :: d_coo_row_ind

  type(c_ptr) :: handle

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_csr2coo' (Fortran 2003 interfaces) - "

  ! Allocate device memory and copy the row-pointer to device
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_coo_row_ind, size_ci * 4))
  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))

  ! Create rocSPARSE handle
  call rocsparseCheck(rocsparse_create_handle(handle))

  ! Convert CSR row pointers to COO row indices
  call rocsparseCheck(rocsparse_csr2coo(handle, d_csr_row_ptr, nnz, M, &
                                        d_coo_row_ind, rocsparse_index_base_zero))

  ! Copy the result back to host
  call hipCheck(hipMemcpy(c_loc(h_coo_row_ind(1)), d_coo_row_ind, size_ci * 4, hipMemcpyDeviceToHost))

  ! Verify against the expected COO row indices
  do i = 1,nnz
    if(h_coo_row_ind(i) /= h_expected(i)) then
        write(*,*) "FAILED! coo_row_ind(", i, ") = ", h_coo_row_ind(i), " expected ", h_expected(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_coo_row_ind))

  write(*,*) "PASSED!"

end program csr2coo
