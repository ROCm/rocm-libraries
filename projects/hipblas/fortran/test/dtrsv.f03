program hip_dtrsv
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer(kind(HIPBLAS_FILL_MODE_UPPER)), parameter :: uplo   = HIPBLAS_FILL_MODE_LOWER
  integer(kind(HIPBLAS_OP_N)),            parameter :: transA = HIPBLAS_OP_N
  integer(kind(HIPBLAS_DIAG_NON_UNIT)),   parameter :: diag   = HIPBLAS_DIAG_NON_UNIT
  integer, parameter :: m = 1024
  integer, parameter :: bytes_per_element = 8
  integer(c_size_t) :: NAbytes, Nxbytes
  double precision, allocatable, target, dimension(:) :: hA, hx
  double precision, parameter :: x_exact = 1.d0
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)
  type(c_ptr) :: dA = c_null_ptr, dx = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j

  write(*,"(a)",advance="no") "-- Running test 'DTRSV' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  NAbytes = int(m, c_size_t) * int(m, c_size_t) * bytes_per_element
  Nxbytes = int(m, c_size_t) * bytes_per_element
  allocate(hA(m*m), hx(m))

  hA = 0.d0
  do j = 1, m
    do i = j, m
      hA(i + (j-1)*m) = 1.d0
    end do
  end do
  do i = 1, m
    hx(i) = dble(i)
  end do

  call hipCheck(hipMalloc(dA, NAbytes))
  call hipCheck(hipMalloc(dx, Nxbytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), NAbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nxbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasDtrsv(handle, uplo, transA, diag, m, dA, m, dx, 1))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nxbytes, hipMemcpyDeviceToHost))

  do i = 1, m
    error = abs((x_exact - hx(i)) / x_exact)
    if (error > error_max) then
      write(*,*) "FAILED! Error bigger than max! Error = ", error, " hx(", i, ") = ", hx(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dx))
  call hipblasCheck(hipblasDestroy(handle))
  deallocate(hA, hx)
  write(*,*) "PASSED!"

end program hip_dtrsv
