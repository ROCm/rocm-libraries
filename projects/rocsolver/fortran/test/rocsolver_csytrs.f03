!!!!!!!!!!!!!!
! csytrs example (solve A*X = B for a symmetric matrix, single complex, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/
!
! Self-verifying: factorize a complex SYMMETRIC A (A = A^T) with sytrf, then solve
! A*x = b with sytrs for a right-hand side built from a known solution, and check
! that the recovered x matches. rocSOLVER writes info to DEVICE memory.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program csytrs
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver
  implicit none
  integer :: i
  integer(c_int), parameter :: N = 4, nrhs = 1, lda = 4, ldb = 4
  complex(c_float_complex), target :: hA(N,N) = reshape((/ &
      (10.0,1.0), ( 2.0,0.0), ( 3.0,0.0), ( 6.0,0.0), &
      ( 2.0,0.0), (11.0,1.0), ( 1.0,0.0), ( 0.0,0.0), &
      ( 3.0,0.0), ( 1.0,0.0), (12.0,1.0), ( 2.0,0.0), &
      ( 6.0,0.0), ( 0.0,0.0), ( 2.0,0.0), (13.0,1.0)/), (/N,N/))
  complex(c_float_complex) :: hX(N) = (/(1.0,1.0), (2.0,-1.0), (3.0,0.0), (4.0,2.0)/)
  complex(c_float_complex), target :: hB(N)
  integer(c_int), target :: hInfo(1)
  type(c_ptr) :: dA, dB, dIpiv, dInfo
  type(c_ptr) :: handle
  real(c_float), parameter :: error_max = 1.0e-4
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_csytrs' (Fortran 2003 interfaces) - "

  hB = matmul(hA, hX)

  call hipCheck(hipMalloc(dA, int(N*N,c_size_t) * 8))
  call hipCheck(hipMalloc(dB, int(N,c_size_t) * 8))
  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(N*N,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1)),   int(N,c_size_t) * 8, hipMemcpyHostToDevice))

  call hipCheck(rocblas_create_handle(handle))
  call hipCheck(rocsolver_csytrf(handle, rocblas_fill_upper, N, dA, lda, dIpiv, dInfo))
  call hipCheck(rocsolver_csytrs(handle, rocblas_fill_upper, N, nrhs, dA, lda, dIpiv, dB, ldb))
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hB(1)), dB, int(N,c_size_t) * 8, hipMemcpyDeviceToHost))

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
end program csytrs
