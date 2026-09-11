!!!!!!!!!!!!!!
! dlatrd example (rocSOLVER)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/auxiliary.html
!
! Reduces the first k columns of a symmetric matrix A to symmetric tridiagonal
! form (uplo = lower). The device buffers are passed as type(c_ptr), which
! resolves to the raw bind(c) interface. tau is now a type(c_ptr) there (it used
! to be a scalar by reference), so a device tau pointer can be supplied at all.
!!!!!!!!!!!!!!
!
program dlatrd
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: n = 4, k = 2, lda = 4, ldw = 4

  ! Symmetric 4x4 matrix (column-major). Its first sub-column A(2:n,1) = [2,3,6]
  ! has 2-norm 7, which is the quantity we check below.
  real(c_double), target :: hA(n,n) = reshape([ &
      10.0d0,  2.0d0,  3.0d0,  6.0d0, &
       2.0d0, 11.0d0,  1.0d0,  0.0d0, &
       3.0d0,  1.0d0, 12.0d0,  2.0d0, &
       6.0d0,  0.0d0,  2.0d0, 13.0d0], [n,n])
  real(c_double), target :: hE(n-1) = 0.0d0

  integer(c_size_t) :: size_A   = n*n
  integer(c_size_t) :: size_E   = n-1
  integer(c_size_t) :: size_tau = n-1
  integer(c_size_t) :: size_W   = n*k

  type(c_ptr) :: dA, dE, dtau, dW
  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: expected, error
  real(c_double), parameter :: rtol = 1.0d-10

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dlatrd' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA,   size_A   * 8))
  call hipCheck(hipMalloc(dE,   size_E   * 8))
  call hipCheck(hipMalloc(dtau, size_tau * 8))
  call hipCheck(hipMalloc(dW,   size_W   * 8))

  call rocblasCheck(rocblas_create_handle(handle))

  ! Copy the input matrix from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Reduce the first k columns to tridiagonal form. Device buffers are passed as
  ! type(c_ptr) (resolves to the raw bind(c) interface).
  call rocsolverCheck(rocsolver_dlatrd(handle, rocblas_fill_lower, n, k, dA, lda, dE, dtau, dW, ldw))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hE(1)), dE, size_E * 8, hipMemcpyDeviceToHost))

  ! The first Householder reflector maps the original sub-column A(2:n,1) onto
  ! [E(1), 0, ..., 0], so |E(1)| == ||A(2:n,1)||_2, independent of the sign
  ! convention rocSOLVER uses for the reflector.
  expected = sqrt(2.0d0**2 + 3.0d0**2 + 6.0d0**2)   ! = 7
  error = abs(abs(hE(1)) - expected) / expected
  if (error > rtol) then
     write(*,*) "FAILED! |E(1)| = ", abs(hE(1)), " expected ", expected
     call exit(1)
  end if

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dtau))
  call hipCheck(hipFree(dW))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dlatrd
