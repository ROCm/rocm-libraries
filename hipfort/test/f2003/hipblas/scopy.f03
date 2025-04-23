program hip_scopy
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none

  integer :: n = 6
  type(c_ptr) :: handle = c_null_ptr
  integer :: j
  real, allocatable, target, dimension(:) :: x, y
  integer, parameter :: bytes_per_element = 4 !float precision
 
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr

  integer(c_size_t) :: Nxbytes
  integer(c_size_t) :: Nybytes

  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  Nxbytes = n * bytes_per_element
  Nybytes = n * bytes_per_element
  allocate(x(n))
  allocate(y(n))

  do j = 1,n
    x(j) = j
   ! write(*,*) "value of x(j)" , x(j)
  end do

  write(*,"(a)",advance="no") "-- Running test 'Scopy' (Fortran 2003 interfaces) - "

  do j = 1,n
    y(j) = x(j)
   ! write(*,*) "value of y(j)" , y(j)
  end do
  
  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx,Nxbytes))
  call hipCheck(hipMalloc(dy,Nybytes))

  call hipCheck(hipMemcpy(dx, c_loc(x(1)), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y(1)), Nybytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasScopy(handle,n,dx,1,dy,1))

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(y(1)), dy, Nybytes, hipMemcpyDeviceToHost))

  do j = 1,n
    error = abs(y(j) - x(j))
      if( error > error_max )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error
        call exit(1)
      end if
  end do

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(x)
  deallocate(y)

  write(*,*) "PASSED!"
  
end program hip_scopy
