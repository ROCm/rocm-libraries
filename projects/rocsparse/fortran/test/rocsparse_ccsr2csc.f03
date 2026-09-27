!!!!!!!!!!!!!/
! ccsr2csc example (single-precision complex CSR -> CSC conversion / sparse transpose, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/conversion.html
!
! Converting A from CSR to CSC is equivalent to producing the CSR of A**T
! (csr2csc permutes values without conjugation). We check the resulting
! csc_col_ptr / csc_row_ind / csc_val against the known transpose.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program ccsr2csc
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  ! 3x3 sparse matrix in CSR (0-based); values complex
  integer(c_int), parameter :: M = 3, N = 3, nnz = 5

  integer(c_int), target :: h_csr_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_csr_col_ind(5) = (/0, 2, 1, 0, 2/)
  complex(c_float_complex), target :: h_csr_val(5) = (/ (1.,1.),(2.,-1.),(3.,2.),(4.,0.),(5.,-2.) /)

  ! Expected CSC (= CSR of the transpose): value order permuted as 1,4,3,2,5
  integer(c_int) :: h_exp_col_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_exp_row_ind(5) = (/0, 2, 1, 0, 2/)
  complex(c_float_complex) :: h_exp_val(5) = (/ (1.,1.),(4.,0.),(3.,2.),(2.,-1.),(5.,-2.) /)

  integer(c_int), target :: h_csc_col_ptr(4)
  integer(c_int), target :: h_csc_row_ind(5)
  complex(c_float_complex), target :: h_csc_val(5)

  integer(c_size_t) :: size_rp = 4, size_ci = 5, size_v = 5
  integer(c_size_t) :: size_cp = 4, size_ri = 5

  type(c_ptr) :: d_csr_row_ptr, d_csr_col_ind, d_csr_val
  type(c_ptr) :: d_csc_col_ptr, d_csc_row_ind, d_csc_val

  type(c_ptr) :: handle, d_buffer
  integer(c_size_t) :: buffer_size

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_ccsr2csc' (Fortran 2003 interfaces) - "

  ! Allocate device memory and copy the CSR matrix to device
  call hipCheck(hipMalloc(d_csr_row_ptr, size_rp * 4))
  call hipCheck(hipMalloc(d_csr_col_ind, size_ci * 4))
  call hipCheck(hipMalloc(d_csr_val,     size_v * 8))
  call hipCheck(hipMalloc(d_csc_col_ptr, size_cp * 4))
  call hipCheck(hipMalloc(d_csc_row_ind, size_ri * 4))
  call hipCheck(hipMalloc(d_csc_val,     size_v * 8))

  call hipCheck(hipMemcpy(d_csr_row_ptr, c_loc(h_csr_row_ptr(1)), size_rp * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_col_ind, c_loc(h_csr_col_ind(1)), size_ci * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_csr_val,     c_loc(h_csr_val(1)),     size_v * 8,  hipMemcpyHostToDevice))

  ! Create rocSPARSE handle
  call rocsparseCheck(rocsparse_create_handle(handle))

  ! Query and allocate the required workspace
  call rocsparseCheck(rocsparse_csr2csc_buffer_size(handle, M, N, nnz, &
                          d_csr_row_ptr, d_csr_col_ind, rocsparse_action_numeric, buffer_size))
  call hipCheck(hipMalloc(d_buffer, buffer_size))

  ! Convert CSR -> CSC (numeric: also permute values)
  call rocsparseCheck(rocsparse_ccsr2csc(handle, M, N, nnz, &
                          d_csr_val, d_csr_row_ptr, d_csr_col_ind, &
                          d_csc_val, d_csc_row_ind, d_csc_col_ptr, &
                          rocsparse_action_numeric, rocsparse_index_base_zero, d_buffer))

  ! Copy the result back to host
  call hipCheck(hipMemcpy(c_loc(h_csc_col_ptr(1)), d_csc_col_ptr, size_cp * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(h_csc_row_ind(1)), d_csc_row_ind, size_ri * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(h_csc_val(1)),     d_csc_val,     size_v * 8,  hipMemcpyDeviceToHost))

  ! Verify structure
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
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(d_csr_row_ptr))
  call hipCheck(hipFree(d_csr_col_ind))
  call hipCheck(hipFree(d_csr_val))
  call hipCheck(hipFree(d_csc_col_ptr))
  call hipCheck(hipFree(d_csc_row_ind))
  call hipCheck(hipFree(d_csc_val))
  call hipCheck(hipFree(d_buffer))

  write(*,*) "PASSED!"

end program ccsr2csc
