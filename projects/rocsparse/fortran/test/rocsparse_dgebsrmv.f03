!!!!!!!!!!!!!!
! rocsparse sgebsrmv example (general block-sparse matrix-vector multiply, single, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Computes y = alpha * A * x + beta * y for a BSR matrix A. Here A is
! block-diagonal with mb=nb=2 blocks of block_dim=2 (a 4x4 dense equivalent):
!   A = [ B1  0  ]   B1 = [1 2]   B2 = [5 6]
!       [ 0   B2 ]        [3 4]        [7 8]
! Blocks are stored row-major (rocsparse_direction_row). The result is checked
! against the dense reference y = A_dense * x.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc. alpha/beta are host scalars (by reference).
!!!!!!!!!!!!!!
!
program dgebsrmv
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: mb = 2, nb = 2, nnzb = 2, block_dim = 2
  integer(c_int), parameter :: mdim = mb * block_dim   ! 4
  ! two 2x2 blocks, each stored row-major: B1=[1,2,3,4], B2=[5,6,7,8]
  real(c_double), target :: hVal(8) = (/1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0, 7.0d0, 8.0d0/)
  integer(c_int), target :: hRowPtr(3) = (/0, 1, 2/)   ! one block per block-row
  integer(c_int), target :: hColInd(2) = (/0, 1/)      ! block cols 0 and 1
  real(c_double), target :: hX(4) = (/1.0d0, 2.0d0, 3.0d0, 4.0d0/)
  real(c_double), target :: hY(4) = (/0.0d0, 0.0d0, 0.0d0, 0.0d0/)
  real(c_double) :: hRef(4)
  real(c_double) :: alpha = 1.0d0, beta = 0.0d0
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: descr = c_null_ptr
  type(c_ptr) :: dVal, dRowPtr, dColInd, dX, dY
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_dgebsrmv' (Fortran 2003 interfaces) - "

  ! Dense reference: y = A * x, A = blkdiag([1 2;3 4],[5 6;7 8]).
  hRef(1) = 1.0d0*hX(1) + 2.0d0*hX(2)
  hRef(2) = 3.0d0*hX(1) + 4.0d0*hX(2)
  hRef(3) = 5.0d0*hX(3) + 6.0d0*hX(4)
  hRef(4) = 7.0d0*hX(3) + 8.0d0*hX(4)

  call hipCheck(hipMalloc(dVal,    int(8,c_size_t) * 8))
  call hipCheck(hipMalloc(dRowPtr, int(mb+1,c_size_t) * 4))
  call hipCheck(hipMalloc(dColInd, int(nnzb,c_size_t) * 4))
  call hipCheck(hipMalloc(dX,      int(mdim,c_size_t) * 8))
  call hipCheck(hipMalloc(dY,      int(mdim,c_size_t) * 8))
  call hipCheck(hipMemcpy(dVal,    c_loc(hVal(1)),    int(8,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dRowPtr, c_loc(hRowPtr(1)), int(mb+1,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dColInd, c_loc(hColInd(1)), int(nnzb,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dX,      c_loc(hX(1)),      int(mdim,c_size_t) * 8, hipMemcpyHostToDevice))

  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_create_mat_descr(descr))
  call rocsparseCheck(rocsparse_dgebsrmv(handle, rocsparse_direction_row, rocsparse_operation_none, &
       mb, nb, nnzb, alpha, descr, dVal, dRowPtr, dColInd, block_dim, block_dim, dX, beta, dY))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hY(1)), dY, int(mdim,c_size_t) * 8, hipMemcpyDeviceToHost))

  do i = 1, mdim
     if (abs(hY(i) - hRef(i)) > 1.0d-11) then
        write(*,*) "FAILED! y(", i, ") = ", hY(i), " expected ", hRef(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_mat_descr(descr))
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dVal)); call hipCheck(hipFree(dRowPtr)); call hipCheck(hipFree(dColInd))
  call hipCheck(hipFree(dX)); call hipCheck(hipFree(dY))
  write(*,*) "PASSED!"
end program dgebsrmv
