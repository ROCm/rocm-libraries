!!!!!!!!!!!!!!
! cgesvd example (complex singular value decomposition, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Computes the singular values of A (jobu = jobv = 'N') and checks the
! convention-independent invariant sum(sigma_i^2) == ||A||_F^2.
!!!!!!!!!!!!!!
!
program hipsolver_cgesvd
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer(c_int), parameter :: M = 2, N = 2, lda = 2, ldu = 2, ldv = 2
  integer(c_int), parameter :: mn = 2
  complex(c_float_complex), target :: hA(M,N) = reshape((/ (1.,0.),(3.,0.),(2.,0.),(4.,0.) /), (/M,N/))
  real(c_float), target :: hS(mn)
  type(c_ptr) :: dA, dS, dU, dV, dWork, dRwork, dInfo, handle = c_null_ptr
  integer(c_int) :: lwork
  integer(c_size_t) :: szA = M*N, szS = mn, szU = M*M, szV = N*N, szR = mn
  real(c_float) :: frob, ssum, error
  real(c_float), parameter :: rtol = 1.0e-5
  integer :: i
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_cgesvd' (Fortran 2003 interfaces) - "
  call hipsolverCheck(hipsolverCreate(handle))
  call hipCheck(hipMalloc(dA,    szA * 8))
  call hipCheck(hipMalloc(dS,    szS * 4))
  call hipCheck(hipMalloc(dU,    szU * 8))
  call hipCheck(hipMalloc(dV,    szV * 8))
  call hipCheck(hipMalloc(dRwork, szR * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), szA * 8, hipMemcpyHostToDevice))
  call hipsolverCheck(hipsolverCgesvd_bufferSize(handle, 'N', 'N', M, N, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 8, 1_c_size_t)))
  call hipsolverCheck(hipsolverCgesvd(handle, 'N', 'N', M, N, dA, lda, dS, &
                                      dU, ldu, dV, ldv, dWork, lwork, dRwork, dInfo))
  call hipCheck(hipMemcpy(c_loc(hS(1)), dS, szS * 4, hipMemcpyDeviceToHost))
  frob = 1.0 + 9.0 + 4.0 + 16.0
  ssum = 0.0
  do i = 1, mn
     ssum = ssum + hS(i)**2
  end do
  error = abs(ssum - frob) / frob
  if (error > rtol) then
     write(*,*) "FAILED! sum(sigma^2) = ", ssum, " expected ||A||_F^2 = ", frob
     call exit(1)
  end if
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dS)); call hipCheck(hipFree(dU))
  call hipCheck(hipFree(dV)); call hipCheck(hipFree(dRwork)); call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  write(*,*) "PASSED!"
end program hipsolver_cgesvd
