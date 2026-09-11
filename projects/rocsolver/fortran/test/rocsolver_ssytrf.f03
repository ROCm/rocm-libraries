!!!!!!!!!!!!!!
! ssytrf example (Bunch-Kaufman symmetric factorization, double, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/
!
! Factorizes a symmetric matrix and checks the algorithm succeeded (info == 0).
! rocSOLVER writes info to DEVICE memory, so it is read back from a device
! allocation (this is what the test primarily exercises).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program ssytrf
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer(c_int), parameter :: N = 4, lda = 4
  real(c_float), target :: hA(N,N) = reshape((/ &
      10.0,  2.0,  3.0,  6.0, &
       2.0, 11.0,  1.0,  0.0, &
       3.0,  1.0, 12.0,  2.0, &
       6.0,  0.0,  2.0, 13.0/), (/N,N/))
  integer(c_int), target :: hInfo(1)
  type(c_ptr) :: dA, dIpiv, dInfo
  type(c_ptr) :: handle
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_ssytrf' (Fortran 2003 interfaces) - "
  call hipCheck(hipMalloc(dA, int(N*N,c_size_t) * 4))
  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(N*N,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(rocblas_create_handle(handle))
  call hipCheck(rocsolver_ssytrf(handle, rocblas_fill_upper, N, dA, lda, dIpiv, dInfo))
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  if (hInfo(1) /= 0) then
     write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dIpiv)); call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program ssytrf
