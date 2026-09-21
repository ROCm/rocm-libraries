program hip_ztrsm
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_SIDE_LEFT)),       parameter :: side   = HIPBLAS_SIDE_LEFT
  integer(kind(HIPBLAS_FILL_MODE_LOWER)), parameter :: uplo   = HIPBLAS_FILL_MODE_LOWER
  integer(kind(HIPBLAS_OP_N)),            parameter :: transA = HIPBLAS_OP_N
  integer(kind(HIPBLAS_DIAG_NON_UNIT)),   parameter :: diag   = HIPBLAS_DIAG_NON_UNIT
  integer, parameter :: m = 512, n = 512
  integer, parameter :: bytes_per_element = 16
  complex(kind=8), parameter :: alpha = (2.d0, 0.d0)
  integer(c_size_t) :: NAbytes, NBbytes
  complex(kind=8), allocatable, target, dimension(:) :: hA, hB
  complex(kind=8), parameter :: x_exact = (1.d0, 0.d0)
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)
  type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j

  write(*,"(a)",advance="no") "-- Running test 'ZTRSM' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  NAbytes = int(m, c_size_t) * int(m, c_size_t) * bytes_per_element
  NBbytes = int(m, c_size_t) * int(n, c_size_t) * bytes_per_element
  allocate(hA(m*m), hB(m*n))

  hA = (0.d0, 0.d0)
  do j = 1, m
    do i = j, m
      hA(i + (j-1)*m) = (1.d0, 0.d0)
    end do
  end do
  do j = 1, n
    do i = 1, m
      hB(i + (j-1)*m) = cmplx(dble(i) / 2.d0, 0., kind=8)
    end do
  end do

  call hipCheck(hipMalloc(dA, NAbytes))
  call hipCheck(hipMalloc(dB, NBbytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), NAbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1)), NBbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasZtrsm(handle, side, uplo, transA, diag, m, n, alpha, dA, m, dB, m))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hB(1)), dB, NBbytes, hipMemcpyDeviceToHost))

  do i = 1, m*n
    error = abs((x_exact - hB(i)) / x_exact)
    if (error > error_max) then
      write(*,*) "FAILED! Error bigger than max! Error = ", error, " hB(", i, ") = ", hB(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipblasCheck(hipblasDestroy(handle))
  deallocate(hA, hB)
  write(*,*) "PASSED!"

end program hip_ztrsm
