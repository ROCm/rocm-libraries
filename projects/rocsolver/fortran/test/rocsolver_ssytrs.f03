!!!!!!!!!!!!!!
! ssytrs example (solve A*X = B for a symmetric matrix, double, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/
!
! Self-verifying: factorize a symmetric A with sytrf (Bunch-Kaufman), then solve
! A*x = b with sytrs for a right-hand side built from a known solution, and check
! that the recovered x matches. rocSOLVER writes info to DEVICE memory.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program ssytrs
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer :: i
  integer(c_int), parameter :: N = 4, nrhs = 1, lda = 4, ldb = 4
  real(c_float), target :: hA(N,N) = reshape((/ &
      10.0,  2.0,  3.0,  6.0, &
       2.0, 11.0,  1.0,  0.0, &
       3.0,  1.0, 12.0,  2.0, &
       6.0,  0.0,  2.0, 13.0/), (/N,N/))
  real(c_float) :: hX(N) = (/1.0, 2.0, 3.0, 4.0/)   ! known solution
  real(c_float), target :: hB(N)                            ! RHS = A*x
  integer(c_int), target :: hInfo(1)
  type(c_ptr) :: dA, dB, dIpiv, dInfo
  type(c_ptr) :: handle
  real(c_float), parameter :: error_max = 1.0e-5
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_ssytrs' (Fortran 2003 interfaces) - "

  hB = matmul(hA, hX)

  call hipCheck(hipMalloc(dA, int(N*N,c_size_t) * 4))
  call hipCheck(hipMalloc(dB, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(N*N,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1)),   int(N,c_size_t) * 4, hipMemcpyHostToDevice))

  call hipCheck(rocblas_create_handle(handle))
  call hipCheck(rocsolver_ssytrf(handle, rocblas_fill_upper, N, dA, lda, dIpiv, dInfo))
  call hipCheck(rocsolver_ssytrs(handle, rocblas_fill_upper, N, nrhs, dA, lda, dIpiv, dB, ldb))
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hB(1)), dB, int(N,c_size_t) * 4, hipMemcpyDeviceToHost))

  if (hInfo(1) /= 0) then
     write(*,*) "FAILED! info = ", hInfo(1), " (expected 0)"; call exit(1)
  end if
  do i = 1, N
     if (abs(hB(i) - hX(i)) > error_max) then
        write(*,*) "FAILED! x(", i, ") = ", hB(i), " expected ", hX(i); call exit(1)
     end if
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dIpiv)); call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program ssytrs
