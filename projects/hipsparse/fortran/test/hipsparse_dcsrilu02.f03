!!!!!!!!!!!!!!
! hipsparse Dcsrilu02 example (incomplete LU, single, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Computes the ILU(0) factorization of a sparse matrix in place using the legacy
! csrilu02 API (bufferSize -> analysis -> compute, with a mat descriptor and a
! csrilu02Info object). For a tridiagonal matrix there is no fill-in, so ILU(0)
! equals the exact LU factorization and the overwritten CSR values can be checked
! against the hand-computed factors (unit lower L, upper U packed into one array).
!
!   A = [ 4 1 0 ]     LU (in place) = [ 4    1    0   ]   (l21=1/4, l32=4/15,
!       [ 1 4 1 ]                     [ 1/4  15/4 1   ]    u22=15/4, u33=56/15)
!       [ 0 1 4 ]                     [ 0    4/15 56/15]
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_dcsrilu02
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 7
  integer(c_int), target :: hRowPtr(4) = (/0, 2, 5, 7/)
  integer(c_int), target :: hColInd(7) = (/0, 1, 0, 1, 2, 1, 2/)
  real(c_double), target :: hVal(7) = (/4.0d0, 1.0d0, 1.0d0, 4.0d0, 1.0d0, 1.0d0, 4.0d0/)
  real(c_double), target :: hOut(7)
  real(c_double) :: hExp(7)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descrA = c_null_ptr
  type(c_ptr) :: info = c_null_ptr
  type(c_ptr) :: dRowPtr, dColInd, dVal, dBuf
  integer(c_int) :: bufSize, pivot
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_dcsrilu02' (Fortran 2003 interfaces) - "

  ! Hand-computed LU (no fill-in for tridiagonal): stored in CSR order matching
  ! hColInd: (1,1)=4, (1,2)=1, (2,1)=l21, (2,2)=u22, (2,3)=1, (3,2)=l32, (3,3)=u33
  hExp(1) = 4.0d0
  hExp(2) = 1.0d0
  hExp(3) = 1.0d0/4.0d0
  hExp(4) = 4.0d0 - (1.0d0/4.0d0)*1.0d0
  hExp(5) = 1.0d0
  hExp(6) = 1.0d0/hExp(4)
  hExp(7) = 4.0d0 - hExp(6)*1.0d0

  call hipCheck(hipMalloc(dRowPtr, int(m+1,c_size_t) * 4))
  call hipCheck(hipMalloc(dColInd, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dVal,    int(nnz,c_size_t) * 8))
  call hipCheck(hipMemcpy(dRowPtr, c_loc(hRowPtr(1)), int(m+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dColInd, c_loc(hColInd(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dVal,    c_loc(hVal(1)),    int(nnz,c_size_t) * 8, hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateMatDescr(descrA))
  call hipsparseCheck(hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL))
  call hipsparseCheck(hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO))
  call hipsparseCheck(hipsparseCreateCsrilu02Info(info))

  call hipsparseCheck(hipsparseDcsrilu02_bufferSize(handle, m, nnz, descrA, dVal, dRowPtr, &
       dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, int(max(bufSize,1),c_size_t)))
  call hipsparseCheck(hipsparseDcsrilu02_analysis(handle, m, nnz, descrA, dVal, dRowPtr, &
       dColInd, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipsparseCheck(hipsparseDcsrilu02(handle, m, nnz, descrA, dVal, dRowPtr, dColInd, &
       info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hOut(1)), dVal, int(nnz,c_size_t) * 8, hipMemcpyDeviceToHost))

  do i = 1, nnz
     if (abs(hOut(i) - hExp(i)) > 1.0d-12) then
        write(*,*) "FAILED! val(", i, ") = ", hOut(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroyCsrilu02Info(info))
  call hipsparseCheck(hipsparseDestroyMatDescr(descrA))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program hipsparse_dcsrilu02
