program hip_saxpy
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas


  implicit none

  integer :: n = 6
  type(c_ptr) :: handle = c_null_ptr
  integer :: j
  real, allocatable, target, dimension(:) :: x, y, y_exact

  real, parameter :: alpha = 2.0
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr

  integer, parameter :: bytes_per_element = 4 !float precision

  integer(c_size_t) :: Nxbytes
  integer(c_size_t) :: Nybytes

  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  Nxbytes = n * bytes_per_element
  Nybytes = n * bytes_per_element
  allocate(x(n))
  allocate(y(n))
  allocate(y_exact(n))

  do j = 1,n
    x(j) = j
    y(j) = j
  end do

  do j = 1,n
    y_exact(j) = alpha*x(j) + y(j)
  end do
 
  write(*,"(a)",advance="no") "-- Running test 'SAXPY' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx,Nxbytes))
  call hipCheck(hipMalloc(dy,Nybytes))

  call hipCheck(hipMemcpy(dx, c_loc(x(1)), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y(1)), Nybytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasSaxpy(handle,n,alpha,dx,1,dy,1))

  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(y(1)), dy, Nybytes, hipMemcpyDeviceToHost))

  do j = 1,n
    error = abs((y_exact(j) - y(j))/y_exact(j))
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

end program hip_saxpy
