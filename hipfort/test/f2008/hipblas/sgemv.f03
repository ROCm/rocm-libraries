program hip_sgemv
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none

  integer :: m = 6
  integer :: n = 5
  integer :: i, j

  real, parameter :: alpha = 1.0
  real, parameter :: beta = 0.0

  type(c_ptr) :: handle = c_null_ptr

  real(kind=4), allocatable, target, dimension(:) :: a, x, y

  real(kind=4), pointer, dimension(:) :: da, dx, dy

  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  allocate(x(n))
  allocate(y(m))
  allocate(a(m*n))

  a(:) = 1.0
  x(:) = 1.0
  y(:) = 1.0

  write(*,"(a)",advance="no") "-- Running test 'SGEMV' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  call hipCheck(hipMalloc(dx,source=x))
  call hipCheck(hipMalloc(dy,source=y))
  call hipCheck(hipMalloc(da,source=a))

  call hipCheck(hipblasSgemv(handle,HIPBLAS_OP_N,m,n,alpha,da,m,dx,1,beta,dy,1))

  call hipCheck(hipMemcpy(y, dy, hipMemcpyDeviceToHost))

  do i = 1,m
    error = abs(5.0 - y(i))
      if( error > 10*epsilon(error) )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, "y(i) = ", y(i)
        call exit(1)
     end if
  end do

  call hipblasCheck(hipblasDestroy(handle))

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))

  deallocate(a,x,y)

  write(*,*) "PASSED!"

end program hip_sgemv

