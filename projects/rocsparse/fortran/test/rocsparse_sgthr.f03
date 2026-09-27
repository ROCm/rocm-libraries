!!!!!!!!!!!!!!
! rocsparse sgthr example (gather y[x_ind] -> x_val, single, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Gathers the entries of a dense vector y at the sparse index set x_ind into the
! packed vector x_val, then checks the gathered values.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program sgthr
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: n = 5, nnz = 3
  real(c_float), target :: hY(5) = (/10.0, 20.0, 30.0, 40.0, 50.0/)
  integer(c_int), target :: hXind(3) = (/0, 2, 4/)
  real(c_float), target :: hXval(3)
  real(c_float) :: hExp(3) = (/10.0, 30.0, 50.0/)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dY, dXind, dXval
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_sgthr' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(dY,    int(n,c_size_t)   * 4))
  call hipCheck(hipMalloc(dXind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dXval, int(nnz,c_size_t) * 4))
  call hipCheck(hipMemcpy(dY,    c_loc(hY(1)),    int(n,c_size_t)   * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dXind, c_loc(hXind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_sgthr(handle, nnz, dY, dXval, dXind, rocsparse_index_base_zero))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hXval(1)), dXval, int(nnz,c_size_t) * 4, hipMemcpyDeviceToHost))
  do i = 1, nnz
     if (abs(hXval(i) - hExp(i)) > 1.0e-6) then
        write(*,*) "FAILED! x_val(", i, ") = ", hXval(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dY)); call hipCheck(hipFree(dXind)); call hipCheck(hipFree(dXval))
  write(*,*) "PASSED!"
end program sgthr
