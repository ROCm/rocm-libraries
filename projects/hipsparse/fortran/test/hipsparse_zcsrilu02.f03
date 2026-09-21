!!!!!!!!!!!!!!
! hipsparse Zcsrilu02 example (incomplete LU, single complex, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSPARSE/en/latest/
!
! Computes the ILU(0) factorization of a complex tridiagonal matrix in place
! using the legacy csrilu02 API (bufferSize -> analysis -> compute, with a mat
! descriptor and a csrilu02Info object). A tridiagonal matrix has no fill-in, so
! ILU(0) equals the exact LU factorization; the overwritten CSR values are
! checked against the factors computed on the host with the same recurrence.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program hipsparse_zcsrilu02
  use iso_c_binding
  use hip
  use hipsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 7
  integer(c_int), target :: hRowPtr(4) = (/0, 2, 5, 7/)
  integer(c_int), target :: hColInd(7) = (/0, 1, 0, 1, 2, 1, 2/)
  complex(c_double_complex), target :: hVal(7) = (/ &
    (4.0d0,1.0d0), (1.0d0,0.0d0), (1.0d0,0.0d0), (4.0d0,1.0d0), (1.0d0,0.0d0), (1.0d0,0.0d0), (4.0d0,1.0d0)/)
  complex(c_double_complex), target :: hOut(7)
  complex(c_double_complex) :: hExp(7)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descrA = c_null_ptr
  type(c_ptr) :: info = c_null_ptr
  type(c_ptr) :: dRowPtr, dColInd, dVal, dBuf
  integer(c_int) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'hipsparse_zcsrilu02' (Fortran 2003 interfaces) - "

  ! Exact LU of the tridiagonal (no fill-in), stored in CSR order.
  hExp(1) = hVal(1)                         ! u11
  hExp(2) = hVal(2)                         ! u12
  hExp(3) = hVal(3)/hExp(1)                 ! l21
  hExp(4) = hVal(4) - hExp(3)*hExp(2)       ! u22
  hExp(5) = hVal(5)                         ! u23
  hExp(6) = hVal(6)/hExp(4)                 ! l32
  hExp(7) = hVal(7) - hExp(6)*hExp(5)       ! u33

  call hipCheck(hipMalloc(dRowPtr, int(m+1,c_size_t) * 4))
  call hipCheck(hipMalloc(dColInd, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dVal,    int(nnz,c_size_t) * 16))
  call hipCheck(hipMemcpy(dRowPtr, c_loc(hRowPtr(1)), int(m+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dColInd, c_loc(hColInd(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dVal,    c_loc(hVal(1)),    int(nnz,c_size_t) * 16, hipMemcpyHostToDevice))

  call hipsparseCheck(hipsparseCreate(handle))
  call hipsparseCheck(hipsparseCreateMatDescr(descrA))
  call hipsparseCheck(hipsparseSetMatType(descrA, HIPSPARSE_MATRIX_TYPE_GENERAL))
  call hipsparseCheck(hipsparseSetMatIndexBase(descrA, HIPSPARSE_INDEX_BASE_ZERO))
  call hipsparseCheck(hipsparseCreateCsrilu02Info(info))

  call hipsparseCheck(hipsparseZcsrilu02_bufferSize(handle, m, nnz, descrA, dVal, dRowPtr, &
       dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, int(max(bufSize,1),c_size_t)))
  call hipsparseCheck(hipsparseZcsrilu02_analysis(handle, m, nnz, descrA, dVal, dRowPtr, &
       dColInd, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipsparseCheck(hipsparseZcsrilu02(handle, m, nnz, descrA, dVal, dRowPtr, dColInd, &
       info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hOut(1)), dVal, int(nnz,c_size_t) * 16, hipMemcpyDeviceToHost))

  do i = 1, nnz
     if (abs(hOut(i) - hExp(i)) > 1.0d-11) then
        write(*,*) "FAILED! val(", i, ") = ", hOut(i), " expected ", hExp(i); call exit(1)
     end if
  end do
  call hipsparseCheck(hipsparseDestroyCsrilu02Info(info))
  call hipsparseCheck(hipsparseDestroyMatDescr(descrA))
  call hipsparseCheck(hipsparseDestroy(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program hipsparse_zcsrilu02
