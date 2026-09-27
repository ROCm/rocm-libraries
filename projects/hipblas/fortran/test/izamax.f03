program hip_izamax
  use iso_c_binding
  use hip
  use hipblas

  implicit none

  ! The largest |x| sits at index 5, and hipBLAS returns a 1-based index.
  integer, parameter :: n = 8
  integer(c_size_t) :: Nbytes
  complex(c_double_complex), allocatable, target, dimension(:) :: hx
  integer(c_int), target :: res
  integer(c_int), parameter :: res_exact = 5
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  write(*,"(a)",advance="no") "-- Running test 'izamax' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(hx(n))
  hx = (1.0, 0.0)
  hx(5) = (10.0, 0.0)

  Nbytes = int(n, c_size_t) * 16
  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  res = 0
  call hipblasCheck(hipblasIzamax(handle, n, dx, 1, c_loc(res)))
  call hipCheck(hipDeviceSynchronize())

  if (res /= res_exact) then
    write(*,*) "FAILED! wrong index: result = ", res, " expected ", res_exact
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipblasCheck(hipblasDestroy(handle))

  write(*,*) "PASSED!"

end program hip_izamax
