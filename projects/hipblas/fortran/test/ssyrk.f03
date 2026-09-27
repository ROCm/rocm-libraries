program hip_ssyrk
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! C := alpha*A*A**T + beta*C, upper triangle. A = [1 0; 2 3] gives
  ! A*A**T = [1 2; 2 13], so the referenced upper triangle is 1, 2, 13.
  integer, parameter :: ld = 2
  integer(c_size_t) :: Nbytes
  real(c_float), parameter :: alpha = 1.0, beta = 0.0

  real(c_float), target :: hA(ld,ld) = reshape([1.0, 2.0, 0.0, 3.0], [ld,ld])
  real(c_float), target :: hC(ld,ld)
  real(c_float) :: expected(ld,ld) = reshape([1.0, 0.0, 2.0, 13.0], [ld,ld])
  type(c_ptr) :: dA = c_null_ptr, dC = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'ssyrk' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  hC = 0.0

  Nbytes = int(ld*ld, c_size_t) * 4
  call hipCheck(hipMalloc(dA, Nbytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dC, Nbytes))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), Nbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasSsyrk(handle, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, ld, ld, &
       alpha, dA, ld, beta, dC, ld))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, Nbytes, hipMemcpyDeviceToHost))

  do j = 1, ld
    do i = 1, j
      error = abs(expected(i,j) - hC(i,j))
      if (error > error_max) then
        write(*,*) "FAILED! error = ", error, " at ", i, j
        call exit(1)
      end if
    end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dC))
  call hipblasCheck(hipblasDestroy(handle))

  write(*,*) "PASSED!"

end program hip_ssyrk
