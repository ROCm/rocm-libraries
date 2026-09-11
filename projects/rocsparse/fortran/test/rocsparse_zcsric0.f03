!!!!!!!!!!!!!!
! rocsparse scsric0 example (incomplete Cholesky, single, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Computes the IC(0) factorization of an SPD sparse matrix in place
! (buffer_size -> analysis -> compute, with a mat descriptor and a mat info
! object). For an SPD tridiagonal matrix there is no fill-in, so IC(0) equals the
! exact Cholesky factor L (A = L*L^T). csric0 overwrites the lower-triangular
! part (including the diagonal) with L; the strict upper part is left unchanged.
! The lower/diagonal CSR entries are checked against the hand-computed L.
!
!   A = [ 4 1 0 ]   L = [ 2       0        0      ]
!       [ 1 4 1 ]       [ 1/2     sqrt(15)/2   0  ]
!       [ 0 1 4 ]       [ 0       2/sqrt(15) l33  ]
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program zcsric0
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, nnz = 7
  ! CSR order: (1,1),(1,2), (2,1),(2,2),(2,3), (3,2),(3,3)
  integer(c_int), target :: hRowPtr(4) = (/0, 2, 5, 7/)
  integer(c_int), target :: hColInd(7) = (/0, 1, 0, 1, 2, 1, 2/)
  complex(c_double_complex), target :: hVal(7) = (/ &
    (4.0d0,0.0d0), (1.0d0,0.0d0), (1.0d0,0.0d0), (4.0d0,0.0d0), &
    (1.0d0,0.0d0), (1.0d0,0.0d0), (4.0d0,0.0d0)/)
  complex(c_double_complex), target :: hOut(7)
  real(c_double) :: L11, L21, L22, L32, L33
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descr = c_null_ptr, info = c_null_ptr
  type(c_ptr) :: dRowPtr, dColInd, dVal, dBuf
  integer(c_size_t) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_zcsric0' (Fortran 2003 interfaces) - "

  ! Exact Cholesky of the SPD tridiagonal (lower factor L).
  L11 = sqrt(4.0d0)
  L21 = 1.0d0 / L11
  L22 = sqrt(4.0d0 - L21*L21)
  L32 = 1.0d0 / L22
  L33 = sqrt(4.0d0 - L32*L32)

  call hipCheck(hipMalloc(dRowPtr, int(m+1,c_size_t) * 4))
  call hipCheck(hipMalloc(dColInd, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dVal,    int(nnz,c_size_t) * 16))
  call hipCheck(hipMemcpy(dRowPtr, c_loc(hRowPtr(1)), int(m+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dColInd, c_loc(hColInd(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dVal,    c_loc(hVal(1)),    int(nnz,c_size_t) * 16, hipMemcpyHostToDevice))

  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr))
  call rocsparseCheck(rocsparse_create_mat_info(info))

  call rocsparseCheck(rocsparse_zcsric0_buffer_size(handle, m, nnz, descr, dVal, dRowPtr, &
       dColInd, info, bufSize))
  call hipCheck(hipMalloc(dBuf, max(bufSize, 1_c_size_t)))
  call rocsparseCheck(rocsparse_zcsric0_analysis(handle, m, nnz, descr, dVal, dRowPtr, dColInd, &
       info, rocsparse_analysis_policy_reuse, rocsparse_solve_policy_auto, dBuf))
  call rocsparseCheck(rocsparse_zcsric0(handle, m, nnz, descr, dVal, dRowPtr, dColInd, &
       info, rocsparse_solve_policy_auto, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hOut(1)), dVal, int(nnz,c_size_t) * 16, hipMemcpyDeviceToHost))

  ! Check the lower/diagonal entries (positions 1,3,4,6,7 in CSR order).
  if (abs(hOut(1) - L11) > 1.0d-10 .or. abs(hOut(3) - L21) > 1.0d-10 .or. &
      abs(hOut(4) - L22) > 1.0d-10 .or. abs(hOut(6) - L32) > 1.0d-10 .or. &
      abs(hOut(7) - L33) > 1.0d-10) then
     write(*,*) "FAILED! L = ", hOut(1), hOut(3), hOut(4), hOut(6), hOut(7), &
                " expected ", L11, L21, L22, L32, L33
     call exit(1)
  end if
  call rocsparseCheck(rocsparse_destroy_mat_info(info))
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd)); call hipCheck(hipFree(dVal))
  call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program zcsric0
