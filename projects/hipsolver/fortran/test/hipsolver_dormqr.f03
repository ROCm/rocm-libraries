!!!!!!!!!!!!!/
! hipsolverDormqr example (double-precision multiply by Q from a QR
! factorization, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: factorize A with geqrf to obtain Q (as Householder vectors),
! then form Q*C with ormqr. Q is orthogonal, so it preserves the Frobenius norm:
! ||Q*C||_F = ||C||_F. Norm preservation is sign-convention independent, so no
! reference matrix is needed.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. tau and devInfo are passed by reference
! through c_f_pointer views of their device allocations.
!!!!!!!!!!!!!!/
!
program dormqr
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 2   ! number of columns of C
  integer(c_int), parameter :: K = 3   ! number of reflectors
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldc = 3

  ! Matrix to factorize (column-major) and a separate C to multiply
  real(c_double), target :: hA(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3, 3/))
  real(c_double), target :: hC(3,2) = reshape((/1, 2, 3, 4, 5, 6/), (/3, 2/))
  real(c_double) :: norm_in, norm_out

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_C = size(hC)
  integer(c_size_t) :: size_tau = 3

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA, dC, dTau, dInfo, dWork
  real(c_double), pointer :: dTau_p(:)
  integer(c_int), pointer :: dInfo_p(:)
  integer(c_int) :: lwork_qr, lwork_mq, lwork

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverDormqr' (Fortran 2003 interfaces) - "

  ! Norm of the input C (Frobenius)
  norm_in = sqrt(sum(hC*hC))

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dC, size_C * 8))
  call hipCheck(hipMalloc(dTau, size_tau * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dTau, dTau_p, (/int(size_tau)/))
  call c_f_pointer(dInfo, dInfo_p, (/1/))

  ! Copy inputs to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), size_C * 8, hipMemcpyHostToDevice))

  ! Workspace big enough for both geqrf and ormqr
  call hipsolverCheck(hipsolverDgeqrf_bufferSize(handle, M, K, dA, lda, lwork_qr))
  call hipsolverCheck(hipsolverDormqr_bufferSize(handle, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, &
       M, N, K, dA, lda, dTau_p(1), dC, ldc, lwork_mq))
  lwork = max(lwork_qr, lwork_mq)
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  ! Factorize A = Q*R (Q stored as reflectors), then form C <- Q*C in place
  call hipsolverCheck(hipsolverDgeqrf(handle, M, K, dA, lda, dTau_p(1), dWork, lwork, dInfo_p(1)))
  call hipsolverCheck(hipsolverDormqr(handle, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, &
       M, N, K, dA, lda, dTau_p(1), dC, ldc, dWork, lwork, dInfo_p(1)))

  ! Copy the transformed C back to host
  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, size_C * 8, hipMemcpyDeviceToHost))

  ! Verify ||Q*C||_F = ||C||_F (Q is orthogonal)
  norm_out = sqrt(sum(hC*hC))
  error = abs(norm_out - norm_in) / max(norm_in, 1.0_c_double)
  if(error .gt. error_max) then
      write(*,*) "FAILED! Norm not preserved! ||C|| = ", norm_in, " ||Q*C|| = ", norm_out
      call exit(1)
  end if

  ! Clean up
  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dC))
  call hipCheck(hipFree(dTau))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dormqr
