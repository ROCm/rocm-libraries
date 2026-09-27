!!!!!!!!!!!!!/
! ccsrgeam example (single-precision complex sparse matrix addition, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/extra.html
!
! Two-phase flow: csrgeam_nnz (fills row_ptr_C and nnz_C) -> allocate
! col_ind_C/val_C -> csrgeam (computes C = alpha*A + beta*B). csrgeam needs no
! workspace buffer.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program ccsrgeam
  use iso_c_binding
  use hip
  use rocsparse

  implicit none
  integer :: i

  ! A = [[1,0,2],[0,3,0],[4,0,5]] ; B = diag(10,20,30) ; C = A + B
  integer(c_int), parameter :: M = 3, N = 3, nnz_A = 5, nnz_B = 3

  integer(c_int), target :: h_row_ptr_A(4) = (/0, 2, 3, 5/)
  integer(c_int), target :: h_col_ind_A(5) = (/0, 2, 1, 0, 2/)
  complex(c_float_complex), target :: h_val_A(5) = (/ (1.,1.),(2.,-1.),(3.,0.),(4.,2.),(5.,-1.) /)

  integer(c_int), target :: h_row_ptr_B(4) = (/0, 1, 2, 3/)
  integer(c_int), target :: h_col_ind_B(3) = (/0, 1, 2/)
  complex(c_float_complex), target :: h_val_B(3) = (/ (10.,1.),(20.,-2.),(30.,3.) /)

  ! Expected C
  integer(c_int) :: h_exp_row_ptr(4) = (/0, 2, 3, 5/)
  integer(c_int) :: h_exp_col_ind(5) = (/0, 2, 1, 0, 2/)
  complex(c_float_complex) :: h_exp_val(5) = (/ (11.,2.),(2.,-1.),(23.,-2.),(4.,2.),(35.,2.) /)

  complex(c_float_complex) :: alpha = (1.0,0.0), beta = (1.0,0.0)

  integer(c_int), target :: h_row_ptr_C(4)
  integer(c_int), target :: nnz_C

  integer(c_size_t) :: size_rpA = 4, size_ciA = 5, size_vA = 5
  integer(c_size_t) :: size_rpB = 4, size_ciB = 3, size_vB = 3

  type(c_ptr) :: d_row_ptr_A, d_col_ind_A, d_val_A
  type(c_ptr) :: d_row_ptr_B, d_col_ind_B, d_val_B
  type(c_ptr) :: d_row_ptr_C, d_col_ind_C, d_val_C
  type(c_ptr) :: d_nnz_C

  type(c_ptr) :: handle, descr_A, descr_B, descr_C

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'rocsparse_ccsrgeam' (Fortran 2003 interfaces) - "

  ! Create handle and matrix descriptors
  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_A))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_B))
  call rocsparseCheck(rocsparse_create_mat_descr(descr_C))

  ! Allocate device memory
  call hipCheck(hipMalloc(d_row_ptr_A, size_rpA * 4))
  call hipCheck(hipMalloc(d_col_ind_A, size_ciA * 4))
  call hipCheck(hipMalloc(d_val_A,     size_vA * 8))
  call hipCheck(hipMalloc(d_row_ptr_B, size_rpB * 4))
  call hipCheck(hipMalloc(d_col_ind_B, size_ciB * 4))
  call hipCheck(hipMalloc(d_val_B,     size_vB * 8))
  call hipCheck(hipMalloc(d_row_ptr_C, size_rpA * 4))
  call hipCheck(hipMalloc(d_nnz_C,     4_c_size_t))

  ! Copy A and B to device
  call hipCheck(hipMemcpy(d_row_ptr_A, c_loc(h_row_ptr_A(1)), size_rpA * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_col_ind_A, c_loc(h_col_ind_A(1)), size_ciA * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_val_A,     c_loc(h_val_A(1)),     size_vA * 8,  hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_row_ptr_B, c_loc(h_row_ptr_B(1)), size_rpB * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_col_ind_B, c_loc(h_col_ind_B(1)), size_ciB * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(d_val_B,     c_loc(h_val_B(1)),     size_vB * 8,  hipMemcpyHostToDevice))

  ! Phase 1: compute the sparsity of C (row_ptr_C + total nnz_C)
  call rocsparseCheck(rocsparse_csrgeam_nnz(handle, M, N, &
                          descr_A, nnz_A, d_row_ptr_A, d_col_ind_A, &
                          descr_B, nnz_B, d_row_ptr_B, d_col_ind_B, &
                          descr_C, d_row_ptr_C, d_nnz_C))
  call hipCheck(hipMemcpy(c_loc(nnz_C), d_nnz_C, 4_c_size_t, hipMemcpyDeviceToHost))

  ! Allocate C column indices and values now that nnz_C is known
  call hipCheck(hipMalloc(d_col_ind_C, int(nnz_C,c_size_t) * 4))
  call hipCheck(hipMalloc(d_val_C,     int(nnz_C,c_size_t) * 8))

  ! Phase 2: compute the values of C
  call rocsparseCheck(rocsparse_ccsrgeam(handle, M, N, alpha, &
                          descr_A, nnz_A, d_val_A, d_row_ptr_A, d_col_ind_A, &
                          beta, descr_B, nnz_B, d_val_B, d_row_ptr_B, d_col_ind_B, &
                          descr_C, d_val_C, d_row_ptr_C, d_col_ind_C))
  call hipCheck(hipDeviceSynchronize())

  ! Copy the C structure back to host
  call hipCheck(hipMemcpy(c_loc(h_row_ptr_C(1)), d_row_ptr_C, size_rpA * 4, hipMemcpyDeviceToHost))

  ! Verify nnz and row pointers
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

  ! Verify column indices and values
  block
    integer(c_int), target :: h_col_ind_C(nnz_C)
    complex(c_float_complex), target :: h_val_C(nnz_C)
    call hipCheck(hipMemcpy(c_loc(h_col_ind_C(1)), d_col_ind_C, int(nnz_C,c_size_t) * 4, hipMemcpyDeviceToHost))
    call hipCheck(hipMemcpy(c_loc(h_val_C(1)),     d_val_C,     int(nnz_C,c_size_t) * 8, hipMemcpyDeviceToHost))
    do i = 1,nnz_C
      if(h_col_ind_C(i) /= h_exp_col_ind(i)) then
          write(*,*) "FAILED! col_ind_C(", i, ") = ", h_col_ind_C(i), " expected ", h_exp_col_ind(i)
          call exit(1)
      end if
      error = abs(h_val_C(i) - h_exp_val(i)) / max(abs(h_exp_val(i)), 1.0)
      if(error .gt. error_max) then
          write(*,*) "FAILED! val_C(", i, ") = ", h_val_C(i), " expected ", h_exp_val(i)
          call exit(1)
      end if
    end do
  end block

  ! Clean up
  call hipCheck(hipFree(d_col_ind_C))
  call hipCheck(hipFree(d_val_C))
  call hipCheck(hipFree(d_row_ptr_A))
  call hipCheck(hipFree(d_col_ind_A))
  call hipCheck(hipFree(d_val_A))
  call hipCheck(hipFree(d_row_ptr_B))
  call hipCheck(hipFree(d_col_ind_B))
  call hipCheck(hipFree(d_val_B))
  call hipCheck(hipFree(d_row_ptr_C))
  call hipCheck(hipFree(d_nnz_C))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_A))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_B))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr_C))
  call rocsparseCheck(rocsparse_destroy_handle(handle))

  write(*,*) "PASSED!"

end program ccsrgeam
