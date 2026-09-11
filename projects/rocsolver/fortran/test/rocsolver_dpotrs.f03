!!!!!!!!!!!!!/
! dpotrs example (double-precision Cholesky solve, Fortran 2003 interfaces)
! see: https:!www.netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational.html
!
! Self-verifying: pick a known solution x, form b = A*x, factorize A with
! potrf, solve A*X = b with potrs, and confirm X recovers x.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. potrf writes `info` to device memory,
! so dInfo is a device allocation passed directly (already a c_ptr).
!!!!!!!!!!!!!!/
!
program dpotrs
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i ! index for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldb = 3
  integer(c_int), parameter :: nrhs = 1

  ! Symmetric positive-definite input (column-major)
  real(c_double), target :: hA(3,3) = reshape((/4, 12, -16, 12, 37, -43, -16, -43, 98/), (/3, 3/))
  real(c_double), target :: hX(3,1) = reshape((/1, 2, 3/), (/3, 1/))  ! known solution
  real(c_double), target :: hB(3,1)                                    ! RHS = A*x

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_B = size(hB)

  type(c_ptr) :: dA
  type(c_ptr) :: dB
  type(c_ptr) :: dInfo

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dpotrs' (Fortran 2003 interfaces) - "

  ! Build a consistent RHS so that A*x = b
  hB = matmul(hA, hX)

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dB, size_B * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), size_B * 8, hipMemcpyHostToDevice))

  ! Factorize (Cholesky), then solve A*X = B in place (B overwritten with solution)
  call hipCheck(rocsolver_dpotrf(handle, rocblas_fill_upper, N, dA, lda, dInfo))
  call hipCheck(rocsolver_dpotrs(handle, rocblas_fill_upper, N, nrhs, dA, lda, dB, ldb))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hB(1,1)), dB, size_B * 8, hipMemcpyDeviceToHost))

  ! Verify the recovered solution matches x
  do i = 1,N
    error = abs(hB(i,1) - hX(i,1)) / max(abs(hX(i,1)), 1.0_c_double)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " X(", i, ") = ", hB(i,1)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dpotrs
