!!!!!!!!!!!!!!
! dsyev example (rocSOLVER)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Computes the eigenvalues of a real symmetric matrix. rocSOLVER writes `info`
! to DEVICE memory, so it is backed by a device allocation and passed as a
! device pointer.
!!!!!!!!!!!!!!
!
program dsyev
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: n = 4, lda = 4

  ! Symmetric 4x4 matrix (column-major); its trace is 10+11+12+13 = 46.
  real(c_double), target :: hA(n,n) = reshape([ &
      10.0d0,  2.0d0,  3.0d0,  6.0d0, &
       2.0d0, 11.0d0,  1.0d0,  0.0d0, &
       3.0d0,  1.0d0, 12.0d0,  2.0d0, &
       6.0d0,  0.0d0,  2.0d0, 13.0d0], [n,n])
  real(c_double), target :: hD(n) = 0.0d0   ! eigenvalues

  integer(c_size_t) :: size_A = n*n
  integer(c_size_t) :: size_D = n
  integer(c_size_t) :: size_E = n

  type(c_ptr) :: dA, dD, dE, dInfo
  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: trace_A, error
  real(c_double), parameter :: rtol = 1.0d-9

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsyev' (Fortran 2003 interfaces) - "

  trace_A = hA(1,1) + hA(2,2) + hA(3,3) + hA(4,4)   ! = 46

  call hipCheck(hipMalloc(dA,    size_A * 8))
  call hipCheck(hipMalloc(dD,    size_D * 8))
  call hipCheck(hipMalloc(dE,    size_E * 8))
  call hipCheck(hipMalloc(dInfo, int(4, c_size_t)))

  call rocblasCheck(rocblas_create_handle(handle))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Device buffers passed as type(c_ptr) (resolves to the raw bind(c) interface).
  call rocsolverCheck(rocsolver_dsyev(handle, rocblas_evect_none, rocblas_fill_lower, n, dA, lda, &
                                      dD, dE, dInfo))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD, size_D * 8, hipMemcpyDeviceToHost))

  ! An orthogonal diagonalization preserves the trace: sum of eigenvalues =
  ! trace(A). This is convention-independent (order of eigenvalues irrelevant).
  error = abs(sum(hD) - trace_A) / abs(trace_A)
  if (error > rtol) then
     write(*,*) "FAILED! sum(eigenvalues) = ", sum(hD), " expected trace(A) = ", trace_A
     call exit(1)
  end if

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dD))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dsyev
