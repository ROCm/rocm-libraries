program hip_scasum
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! asum(x) with x = 1 over n elements, so the result is n.
  integer, parameter :: n = 1024
  integer(c_size_t) :: Nbytes
  complex(c_float_complex), allocatable, target, dimension(:) :: hx
  real(c_float), target :: res
  real(c_float) :: res_exact, error
  real(c_float), parameter :: error_max = 10*epsilon(error)
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  write(*,"(a)",advance="no") "-- Running test 'scasum' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(hx(n))
  hx = (1.0, 0.0)
  res_exact = real(n, kind=kind(res_exact))

  Nbytes = int(n, c_size_t) * 8
  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  res = 0.0
  call hipblasCheck(hipblasScasum(handle, n, dx, 1, c_loc(res)))
  call hipCheck(hipDeviceSynchronize())

  error = abs((res_exact - res) / res_exact)
  if (error > error_max) then
    write(*,*) "FAILED! error = ", error, " result = ", res
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipblasCheck(hipblasDestroy(handle))

  write(*,*) "PASSED!"

end program hip_scasum
  