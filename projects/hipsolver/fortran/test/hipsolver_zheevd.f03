!!!!!!!!!!!!!!
! zheevd example (complex Hermitian eigenvalues, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Checks sum(eigenvalues) == trace(A). f2003 style: device buffers are type(c_ptr).
!!!!!!!!!!!!!!
!
program hipsolver_zheevd
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer(c_int), parameter :: N = 4, lda = 4
  complex(c_double_complex), target :: hA(N,N) = reshape((/ &
      (10.,0.),(2.,0.),(3.,0.),(6.,0.), (2.,0.),(11.,0.),(1.,0.),(0.,0.), &
      (3.,0.),(1.,0.),(12.,0.),(2.,0.), (6.,0.),(0.,0.),(2.,0.),(13.,0.) /), (/N,N/))
  real(c_double), target :: hD(N) = 0.0
  integer(c_size_t) :: szA = 16, szD = 4
  type(c_ptr) :: dA, dD, dWork, handle = c_null_ptr
  integer(c_int), pointer :: dInfo(:)
  integer(c_int) :: lwork
  real(c_double) :: trace_A, error
  real(c_double), parameter :: rtol = 1.0e-5
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_zheevd' (Fortran 2003 interfaces) - "
  trace_A = real(hA(1,1)) + real(hA(2,2)) + real(hA(3,3)) + real(hA(4,4))
  call hipsolverCheck(hipsolverCreate(handle))
  call hipCheck(hipMalloc(dA, szA * 16))
  call hipCheck(hipMalloc(dD, szD * 8))
  call hipCheck(hipMalloc(dInfo, 1))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), szA * 16, hipMemcpyHostToDevice))
  call hipsolverCheck(hipsolverZheevd_bufferSize(handle, HIPSOLVER_EIG_MODE_NOVECTOR, &
                                                 HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dD, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 16, 1_c_size_t)))
  call hipsolverCheck(hipsolverZheevd(handle, HIPSOLVER_EIG_MODE_NOVECTOR, HIPSOLVER_FILL_MODE_UPPER, &
                                      N, dA, lda, dD, dWork, lwork, dInfo(1)))
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD, szD * 8, hipMemcpyDeviceToHost))
  error = abs(sum(hD) - trace_A) / abs(trace_A)
  if (error > rtol) then
     write(*,*) "FAILED! sum(eigenvalues) = ", sum(hD), " expected trace = ", trace_A
     call exit(1)
  end if
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dD)); call hipCheck(hipFree(dInfo)); call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  write(*,*) "PASSED!"
end program hipsolver_zheevd
