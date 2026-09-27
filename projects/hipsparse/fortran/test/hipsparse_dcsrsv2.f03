!!!!!!!!!!!!!!
! hipsparse Dcsrsv2 example (sparse triangular solve v2, single, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Solves the lower-triangular system L*x = alpha*f for x using the legacy
! csrsv2 API (bufferSize -> analysis -> solve, with a mat descriptor and a
! csrsv2Info object). The right-hand side f is built from a known solution so
! the recovered x can be checked directly.
!
!   L = [ 2 0 0 ]              x = [ 1 ]        f = L*x = [ 2  ]
!       [ 1 2 0 ]                  [ 2 ]                  [ 5  ]
!       [ 3 1 2 ]                  [ 3 ]                  [ 11 ]
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_dcsrsv2
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 6
  integer(c_int), target :: hRowPtr(4) = (/0, 1, 3, 6/)
  integer(c_int), target :: hColInd(6) = (/0, 0, 1, 0, 1, 2/)
  real(c_double), target :: hVal(6) = (/2.0d0, 1.0d0, 2.0d0, 3.0d0, 1.0d0, 2.0d0/)
  real(c_double), target :: hF(3) = (/2.0d0, 5.0d0, 11.0d0/)
  real(c_double), target :: hX(3)
  real(c_double) :: hExp(3) = (/1.0d0, 2.0d0, 3.0d0/)
  real(c_double) :: alpha = 1.0d0
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descrA = c_null_ptr
  type(c_ptr) :: info = c_null_ptr
  type(c_ptr) :: dRowPtr, dColInd, dVal, dF, dX, dBuf
  integer(c_int) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_dcsrsv2' (Fortran 2003 interfaces) - "

  call hipCheck(hipMalloc(dRowPtr, int(m+1,c_size_t) * 4))
  call hipCheck(hipMalloc(dColInd, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dVal,    int(nnz,c_size_t) * 8))
  call hipCheck(hipMalloc(dF,      int(m,c_size_t)   * 8))
  call hipCheck(hipMalloc(dX,      int(m,c_size_t)   * 8))
  call hipCheck(hipMemcpy(dRowPtr, c_loc(hRowPtr(1)), int(m+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dColInd, c_loc(hColInd(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dVal,    c_loc(hVal(1)),    int(nnz,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dF,      c_loc(hF(1)),      int(m,c_size_t)   * 8, hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateMatDescr(descrA))
  call hipsparseCheck(hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL))
  call hipsparseCheck(hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO))
  call hipsparseCheck(hipsparseSetMatFillMode(descrA, HIPSPARSE_FILL_MODE_LOWER))
  call hipsparseCheck(hipsparseSetMatDiagType(descrA, HIPSPARSE_DIAG_TYPE_NON_UNIT))
  call hipsparseCheck(hipsparseCreateCsrsv2Info(info))

  call hipsparseCheck(hipsparseDcsrsv2_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       m, nnz, descrA, dVal, dRowPtr, dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, int(max(bufSize,1),c_size_t)))
  call hipsparseCheck(hipsparseDcsrsv2_analysis(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       m, nnz, descrA, dVal, dRowPtr, dColInd, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipsparseCheck(hipsparseDcsrsv2_solve(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &
       m, nnz, alpha, descrA, dVal, dRowPtr, dColInd, info, dF, dX, &
       HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hX(1)), dX, int(m,c_size_t) * 8, hipMemcpyDeviceToHost))

  do i = 1, m
     if (abs(hX(i) - hExp(i)) > 1.0d-12) then
        write(*,*) "FAILED! x(", i, ") = ", hX(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroyCsrsv2Info(info))
  call hipsparseCheck(hipsparseDestroyMatDescr(descrA))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dF)); call hipCheck(hipFree(dX)); call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program hipsparse_dcsrsv2
