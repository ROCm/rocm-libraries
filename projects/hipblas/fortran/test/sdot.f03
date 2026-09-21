program hip_sdot
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  integer, parameter :: n = 10240
  integer, parameter :: bytes_per_element = 4
  integer(c_size_t) :: Nbytes
  real(kind=4), allocatable, target, dimension(:) :: hx, hy
  real(kind=4), target :: res
  real(kind=4) :: res_exact, error
  real(kind=4), parameter :: error_max = 10*epsilon(error)
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  write(*,"(a)",advance="no") "-- Running test 'SDOT' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  Nbytes = int(n, c_size_t) * bytes_per_element
  allocate(hx(n), hy(n))
  hx = 1.0
  hy = 2.0
  res_exact = 2.0 * n

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMalloc(dy, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(hy(1)), Nbytes, hipMemcpyHostToDevice))

  res = 0.0
  call hipblasCheck(hipblasSdot(handle, n, dx, 1, dy, 1, c_loc(res)))
  call hipCheck(hipDeviceSynchronize())

  error = abs((res_exact - res) / res_exact)
  if (error > error_max) then
    write(*,*) "FAILED! error = ", error, " result = ", res
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipblasCheck(hipblasDestroy(handle))
  deallocate(hx, hy)
  write(*,*) "PASSED!"

end program hip_sdot
