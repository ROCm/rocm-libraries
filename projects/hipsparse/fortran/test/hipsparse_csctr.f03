!!!!!!!!!!!!!!
! hipsparse Csctr example (scatter xVal -> y[xInd], single complex, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Scatters the packed vector xVal into the dense vector y at the sparse index
! set xInd, then checks the resulting dense vector.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_csctr
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: n = 5, nnz = 3
  complex(c_float_complex), target :: hXval(3) = (/(100.0,1.0), (200.0,2.0), (300.0,3.0)/)
  integer(c_int), target :: hXind(3) = (/0, 2, 4/)
  complex(c_float_complex), target :: hY(5) = (/ &
    (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0), (0.0,0.0)/)
  complex(c_float_complex) :: hExp(5) = (/ &
    (100.0,1.0), (0.0,0.0), (200.0,2.0), (0.0,0.0), (300.0,3.0)/)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dXval, dXind, dY
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_csctr' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(dXval, int(nnz,c_size_t) * 8))
  call hipCheck(hipMalloc(dXind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dY,    int(n,c_size_t)   * 8))
  call hipCheck(hipMemcpy(dXval, c_loc(hXval(1)), int(nnz,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dXind, c_loc(hXind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dY,    c_loc(hY(1)),    int(n,c_size_t)   * 8, hipMemcpyHostToDevice))
  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCsctr(handle, nnz, dXval, dXind, dY, HIPSPARSE_INDEX_BASE_ZERO))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hY(1)), dY, int(n,c_size_t) * 8, hipMemcpyDeviceToHost))
  do i = 1, n
     if (abs(hY(i) - hExp(i)) > 1.0e-6) then
        write(*,*) "FAILED! y(", i, ") = ", hY(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(dXval)); call hipCheck(hipFree(dXind)); call hipCheck(hipFree(dY))
  write(*,*) "PASSED!"
end program hipsparse_csctr
