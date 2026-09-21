program hipblas_zgemv_test
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  complex(kind=8), parameter :: alpha = (1.1d0, 0.0d0), beta = (0.9d0, 0.0d0)
  integer, parameter :: m = 512, n = 512
  integer, parameter :: bytes_per_element = 16
  integer(c_size_t) :: NAbytes, Nxbytes, Nybytes
  complex(kind=8), allocatable, target, dimension(:) :: hA, hx, hy
  complex(kind=8) :: y_exact
  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error)
  type(c_ptr) :: dA = c_null_ptr, dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'ZGEMV' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  NAbytes = int(m*n, c_size_t) * bytes_per_element
  Nxbytes = int(n,   c_size_t) * bytes_per_element
  Nybytes = int(m,   c_size_t) * bytes_per_element
  allocate(hA(m*n), hx(n), hy(m))
  hA = (1.d0, 0.d0);  hx = (1.d0, 0.d0);  hy = (1.d0, 0.d0)
  y_exact = alpha * n + beta

  call hipCheck(hipMalloc(dA, NAbytes))
  call hipCheck(hipMalloc(dx, Nxbytes))
  call hipCheck(hipMalloc(dy, Nybytes))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), NAbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(hy(1)), Nybytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasZgemv(handle, HIPBLAS_OP_N, m, n, alpha, dA, m, dx, 1, beta, dy, 1))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hy(1)), dy, Nybytes, hipMemcpyDeviceToHost))

  do i = 1, m
    error = abs((y_exact - hy(i)) / y_exact)
    if (error > error_max) then
      write(*,*) "FAILED! Error bigger than max! Error = ", error
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipblasCheck(hipblasDestroy(handle))
  deallocate(hA, hx, hy)
  write(*,*) "PASSED!"

end program hipblas_zgemv_test
