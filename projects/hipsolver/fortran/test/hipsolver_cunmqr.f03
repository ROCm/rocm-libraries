!!!!!!!!!!!!!/
! hipsolverCunmqr example (single-complex multiply by Q from a QR
! factorization, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: factorize A with geqrf to obtain Q (as Householder vectors),
! then form Q*C with unmqr. Q is unitary, so it preserves the Frobenius norm:
! ||Q*C||_F = ||C||_F. Norm preservation is phase-convention independent, so no
! reference matrix is needed.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. tau and devInfo are passed by reference
! through c_f_pointer views of their device allocations.
!!!!!!!!!!!!!!/
!
program cunmqr
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 2
  integer(c_int), parameter :: K = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldc = 3

  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (1.0,1.0), (4.0,0.0), (7.0,0.0), &
    (2.0,0.0), (5.0,1.0), (8.0,0.0), &
    (3.0,0.0), (6.0,0.0), (10.0,1.0)/), (/3, 3/))
  complex(c_float_complex), target :: hC(3,2) = reshape((/ &
    (1.0,1.0), (2.0,0.0), (3.0,-1.0), (4.0,0.0), (5.0,2.0), (6.0,0.0)/), (/3, 2/))
  real(c_float) :: norm_in, norm_out

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_C = size(hC)
  integer(c_size_t) :: size_tau = 3

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA, dC, dTau, dInfo, dWork
  complex(c_float_complex), pointer :: dTau_p(:)
  integer(c_int), pointer :: dInfo_p(:)
  integer(c_int) :: lwork_qr, lwork_mq, lwork

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverCunmqr' (Fortran 2003 interfaces) - "

  norm_in = sqrt(sum(abs(hC)**2))

  call hipsolverCheck(hipsolverCreate(handle))

  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dC, size_C * 8))
  call hipCheck(hipMalloc(dTau, size_tau * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dTau, dTau_p, (/int(size_tau)/))
  call c_f_pointer(dInfo, dInfo_p, (/1/))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), size_C * 8, hipMemcpyHostToDevice))

  call hipsolverCheck(hipsolverCgeqrf_bufferSize(handle, M, K, dA, lda, lwork_qr))
  call hipsolverCheck(hipsolverCunmqr_bufferSize(handle, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, &
       M, N, K, dA, lda, dTau_p(1), dC, ldc, lwork_mq))
  lwork = max(lwork_qr, lwork_mq)
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  call hipsolverCheck(hipsolverCgeqrf(handle, M, K, dA, lda, dTau_p(1), dWork, lwork, dInfo_p(1)))
  call hipsolverCheck(hipsolverCunmqr(handle, HIPSOLVER_SIDE_LEFT, HIPSOLVER_OP_N, &
       M, N, K, dA, lda, dTau_p(1), dC, ldc, dWork, lwork, dInfo_p(1)))

  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, size_C * 8, hipMemcpyDeviceToHost))

  norm_out = sqrt(sum(abs(hC)**2))
  error = abs(norm_out - norm_in) / max(norm_in, 1.0_c_float)
  if(error .gt. error_max) then
      write(*,*) "FAILED! Norm not preserved! ||C|| = ", norm_in, " ||Q*C|| = ", norm_out
      call exit(1)
  end if

  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dC))
  call hipCheck(hipFree(dTau))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program cunmqr
