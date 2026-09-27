program hip_ctrmm
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! C := alpha*op(A)*B with A upper triangular (out-of-place trmm).
  ! A = [1 2; 0 3] and B = I, so C = A.
  integer, parameter :: ld = 2
  integer(c_size_t) :: Nbytes
  complex(c_float_complex), parameter :: alpha = (1.0, 0.0)

  complex(c_float_complex), target :: hA(ld,ld) = reshape([(1.0, 0.0), (0.0, 0.0), (2.0, 0.0), (3.0, 0.0)], [ld,ld])
  complex(c_float_complex), target :: hB(ld,ld) = reshape([(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)], [ld,ld])
  complex(c_float_complex), target :: hC(ld,ld)
  complex(c_float_complex) :: expected(ld,ld) = reshape([(1.0, 0.0), (0.0, 0.0), (2.0, 0.0), (3.0, 0.0)], [ld,ld])
  type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr, dC = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'ctrmm' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  hC = (0.0, 0.0)

  Nbytes = int(ld*ld, c_size_t) * 8
  call hipCheck(hipMalloc(dA, Nbytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dB, Nbytes))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMalloc(dC, Nbytes))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), Nbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasCtrmm(handle, HIPBLAS_SIDE_LEFT, HIPBLAS_FILL_MODE_UPPER, HIPBLAS_OP_N, &
       HIPBLAS_DIAG_NON_UNIT, ld, ld, alpha, dA, ld, dB, ld, dC, ld))
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

end program hip_ctrmm
