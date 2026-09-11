!!!!!!!!!!!!!!
! zsytrf example (Bunch-Kaufman symmetric factorization, single complex, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/
!
! Factorizes a complex SYMMETRIC matrix (A = A^T, not Hermitian) and checks the
! algorithm succeeded (info == 0). rocSOLVER writes info to DEVICE memory.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program zsytrf
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer(c_int), parameter :: N = 4, lda = 4
  complex(c_double_complex), target :: hA(N,N) = reshape((/ &
      (10.0d0,1.0d0), ( 2.0d0,0.0d0), ( 3.0d0,0.0d0), ( 6.0d0,0.0d0), &
      ( 2.0d0,0.0d0), (11.0d0,1.0d0), ( 1.0d0,0.0d0), ( 0.0d0,0.0d0), &
      ( 3.0d0,0.0d0), ( 1.0d0,0.0d0), (12.0d0,1.0d0), ( 2.0d0,0.0d0), &
      ( 6.0d0,0.0d0), ( 0.0d0,0.0d0), ( 2.0d0,0.0d0), (13.0d0,1.0d0)/), (/N,N/))
  integer(c_int), target :: hInfo(1)
  type(c_ptr) :: dA, dIpiv, dInfo
  type(c_ptr) :: handle
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_zsytrf' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(dA, int(N*N,c_size_t) * 16))
  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(N*N,c_size_t) * 16, hipMemcpyHostToDevice))
  call hipCheck(rocblas_create_handle(handle))
  call hipCheck(rocsolver_zsytrf(handle, rocblas_fill_upper, N, dA, lda, dIpiv, dInfo))
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
     write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dIpiv)); call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program zsytrf
