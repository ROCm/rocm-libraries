!!!!!!!!!!!!!!
! hipsparse Cgthr example (gather y[xInd] -> xVal, single complex, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Gathers the entries of a dense vector y at the sparse index set xInd into the
! packed vector xVal, then checks the gathered values.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_cgthr
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: n = 5, nnz = 3
  complex(c_float_complex), target :: hY(5) = (/ &
    (10.0,1.0), (20.0,2.0), (30.0,3.0), (40.0,4.0), (50.0,5.0)/)
  integer(c_int), target :: hXind(3) = (/0, 2, 4/)
  complex(c_float_complex), target :: hXval(3)
  complex(c_float_complex) :: hExp(3) = (/(10.0,1.0), (30.0,3.0), (50.0,5.0)/)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dY, dXind, dXval
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_cgthr' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(dY,    int(n,c_size_t)   * 8))
  call hipCheck(hipMalloc(dXind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dXval, int(nnz,c_size_t) * 8))
  call hipCheck(hipMemcpy(dY,    c_loc(hY(1)),    int(n,c_size_t)   * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dXind, c_loc(hXind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCgthr(handle, nnz, dY, dXval, dXind, HIPSPARSE_INDEX_BASE_ZERO))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hXval(1)), dXval, int(nnz,c_size_t) * 8, hipMemcpyDeviceToHost))
  do i = 1, nnz
     if (abs(hXval(i) - hExp(i)) > 1.0e-6) then
        write(*,*) "FAILED! xVal(", i, ") = ", hXval(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(dY)); call hipCheck(hipFree(dXind)); call hipCheck(hipFree(dXval))
  write(*,*) "PASSED!"
end program hipsparse_cgthr
