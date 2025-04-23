program hip_sswap
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none
  
  integer :: n = 6
  integer :: i
  real(kind=4), allocatable, target, dimension(:) :: x, y, x_exact, y_exact
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  real :: error
  real, parameter :: error_max = 10*epsilon(error)
  
  integer(c_size_t) :: Nxbytes, Nybytes
  integer, parameter :: bytes_per_element = 4 !float precision
 
  Nxbytes = n * bytes_per_element
  Nybytes = n * bytes_per_element
  
  allocate(x(n))
  allocate(y(n))
  allocate(x_exact(n))
  allocate(y_exact(n))
  
  do i = 1,n
    x(i) = i
    x_exact(i) = x(i)
    y(i) = 2 * i
    y_exact(i) = y(i)
  end do
  
  write(*,"(a)",advance="no") "-- Running test 'SSWAP' (Fortran 2003 interfaces) - "
  
  call hipblasCheck(hipblasCreate(handle))
  
  call hipCheck(hipMalloc(dx,Nxbytes))  
  call hipCheck(hipMalloc(dy,Nybytes))
  
  call hipCheck(hipMemcpy(dx, c_loc(x(1)), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y(1)), Nybytes, hipMemcpyHostToDevice))
  
  call hipblasCheck(hipblasSswap(handle,n,dx,1,dy,1))
  
  call hipCheck(hipMemcpy(c_loc(x(1)), dx, Nxbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(y(1)), dy, Nybytes, hipMemcpyDeviceToHost))
  
  do i = 1,n
    error = MAX(abs((y_exact(i) - x(i))/y_exact(i)), abs((x_exact(i) - y(i))/x_exact(i)))
      if( error > error_max )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error
        call exit(1)
      end if
  end do
  
  call hipblasCheck(hipblasDestroy(handle))
  
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  
  deallocate(x)
  deallocate(y)
  
  write(*,*) "PASSED!"
  
end program hip_sswap