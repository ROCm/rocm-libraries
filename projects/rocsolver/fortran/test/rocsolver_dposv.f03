!!!!!!!!!!!!!!
! dposv example (rocSOLVER)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Solves the symmetric positive-definite system A*X = B (X overwrites B).
! rocSOLVER writes `info` to DEVICE memory, so it is backed by a device
! allocation and passed as a device pointer.
!!!!!!!!!!!!!!
!
program dposv
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: n = 3, nrhs = 1, lda = 3, ldb = 3

  ! SPD matrix A (column-major) and RHS B chosen so the solution is X = [1,1,1].
  real(c_double), target :: hA(n,n) = reshape([ &
      2.0d0, 1.0d0, 0.0d0, &
      1.0d0, 2.0d0, 1.0d0, &
      0.0d0, 1.0d0, 2.0d0], [n,n])
  real(c_double), target :: hB(n,nrhs) = reshape([3.0d0, 4.0d0, 3.0d0], [n,nrhs])
  real(c_double) :: hX_ref(n) = [1.0d0, 1.0d0, 1.0d0]

  integer(c_size_t) :: size_A = n*n
  integer(c_size_t) :: size_B = n*nrhs

  type(c_ptr) :: dA, dB, dInfo
  type(c_ptr) :: handle ! rocblas_handle

  integer :: i
  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dposv' (Fortran 2003 interfaces) - "

  call hipCheck(hipMalloc(dA,    size_A * 8))
  call hipCheck(hipMalloc(dB,    size_B * 8))
  call hipCheck(hipMalloc(dInfo, int(4, c_size_t)))

  call rocblasCheck(rocblas_create_handle(handle))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), size_B * 8, hipMemcpyHostToDevice))

  ! Device buffers passed as type(c_ptr) (resolves to the raw bind(c) interface).
  call rocsolverCheck(rocsolver_dposv(handle, rocblas_fill_upper, n, nrhs, dA, lda, dB, ldb, dInfo))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hB(1,1)), dB, size_B * 8, hipMemcpyDeviceToHost))

  do i = 1, n
     error = abs(hB(i,1) - hX_ref(i)) / max(abs(hX_ref(i)), 1.0d0)
     if (error > error_max) then
        write(*,*) "FAILED! X(", i, ") = ", hB(i,1), " expected ", hX_ref(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dposv
