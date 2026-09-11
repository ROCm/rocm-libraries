!!!!!!!!!!!!!/
! dgesv example (double-precision general linear solver, A*X = B, Fortran 2003 interfaces)
! Solves A*X = B by LU factorization with partial pivoting; X overwrites B.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. rocSOLVER writes `info` to device
! memory, so dInfo is a device allocation passed directly (already a c_ptr).
!!!!!!!!!!!!!!/
!
program dgesv
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i

  ! A (column-major) and right-hand side B chosen so the solution is X = [1,1,1].
  real(c_double), target :: hA(3,3) = reshape((/2, 1, 0,  1, 2, 1,  0, 1, 2/), (/3, 3/))
  real(c_double), target :: hB(3,1) = reshape((/3, 4, 3/), (/3, 1/))
  real(c_double) :: hX_ref(3) = (/1, 1, 1/)
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: nrhs = 1
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldb = 3
  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_B = size(hB)

  type(c_ptr) :: dA     ! GPU buffer for A
  type(c_ptr) :: dB     ! GPU buffer for B (holds X on output)
  type(c_ptr) :: dIpiv  ! GPU buffer for pivots
  type(c_ptr) :: dInfo  ! GPU buffer for info (rocSOLVER writes to device)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dgesv' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA,    size_A * 8))
  call hipCheck(hipMalloc(dB,    size_B * 8))
  call hipCheck(hipMalloc(dIpiv, int(N, c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), size_B * 8, hipMemcpyHostToDevice))

  ! Solve A*X = B on the device (X overwrites B)
  call hipCheck(rocsolver_dgesv(handle, N, nrhs, dA, lda, dIpiv, dB, ldb, dInfo))

  ! Copy the solution back to the host
  call hipCheck(hipMemcpy(c_loc(hB(1,1)), dB, size_B * 8, hipMemcpyDeviceToHost))

  ! Check the solution
  do i = 1,3
    error = abs(hB(i,1) - hX_ref(i)) / max(abs(hX_ref(i)), 1.0_c_double)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " hB(", i, ",1) = ", hB(i,1)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dgesv
