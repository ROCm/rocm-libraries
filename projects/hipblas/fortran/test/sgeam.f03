program hip_sgeam
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! C := alpha*op(A) + beta*op(B) with alpha = beta = 1 and no
  ! transposition, so C is the elementwise sum A + B.
  integer, parameter :: ld = 2
  integer(c_size_t) :: Nbytes
  real(c_float), parameter :: alpha = 1.0, beta = 1.0

  real(c_float), target :: hA(ld,ld) = reshape([1.0, 2.0, 3.0, 4.0], [ld,ld])
  real(c_float), target :: hB(ld,ld) = reshape([10.0, 20.0, 30.0, 40.0], [ld,ld])
  real(c_float), target :: hC(ld,ld)
  real(c_float) :: expected(ld,ld) = reshape([11.0, 22.0, 33.0, 44.0], [ld,ld])
  type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr, dC = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'sgeam' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  hC = 0.0

  Nbytes = int(ld*ld, c_size_t) * 4
  call hipCheck(hipMalloc(dA, Nbytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dB, Nbytes))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dC, Nbytes))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), Nbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasSgeam(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, ld, ld, &
       alpha, dA, ld, beta, dB, ld, dC, ld))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, Nbytes, hipMemcpyDeviceToHost))

  do j = 1, ld
    do i = 1, ld
      error = abs(expected(i,j) - hC(i,j))
      if (error > error_max) then
        write(*,*) "FAILED! error = ", error, " at ", i, j
        call exit(1)
      end if
    end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dC))
  call hipblasCheck(hipblasDestroy(handle))

  write(*,*) "PASSED!"

end program hip_sgeam
