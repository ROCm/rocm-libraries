! HIPBLAS, i.e. cuBLAS and rocBLAS, assumes column-major matrix memory layouts.
! Hence, no matrix must be transposed when interfacing with Fortran.
program hip_sger
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none
  
  integer :: m = 6 
  integer :: n = 5
  real, parameter :: alpha = 2.0
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  
  real :: error
  real, parameter :: error_max = 10*epsilon(error)

  real(kind=4), allocatable, target, dimension(:,:) :: a
  real(kind=4), allocatable, target, dimension(:) :: x, y
 
  type(c_ptr) :: da = c_null_ptr, dx = c_null_ptr, dy = c_null_ptr
  
  integer(c_size_t) :: Nabytes, Nxbytes, Nybytes
  integer, parameter :: bytes_per_element = 4 !float precision
  
  Nxbytes = m * bytes_per_element
  Nybytes = n * bytes_per_element
  Nabytes = m * n * bytes_per_element
  
  allocate(x(m))
  allocate(y(n))
  allocate(a(m,n))
  
  do i = 1,m
    do j = 1,n
      a(i,j) = 1.0 
    end do
  end do
  
  do i = 1,m
    x(i) = 1.0
  end do

  do i = 1,n
    y(i) = 1.0
  end do
 

  write(*,"(a)",advance="no") "-- Running test 'SGER' (Fortran 2008 interfaces) - "
  
  call hipblasCheck(hipblasCreate(handle))
  
  call hipCheck(hipMalloc(dx,Nxbytes))  
  call hipCheck(hipMalloc(dy,Nybytes))
  call hipCheck(hipMalloc(da,Nabytes))
  
  !call hipCheck(hipblasSetMatrix(m,n,bytes_per_element,a,m,da,m))
  !call hipCheck(hipblasSetVector(m,bytes_per_element,x,1,dx,1))
  !call hipCheck(hipblasSetVector(n,bytes_per_element,y,1,dy,1))
  
  call hipCheck(hipMemcpy(da, c_loc(a), Nabytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dx, c_loc(x), Nxbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(y), Nybytes, hipMemcpyHostToDevice)) 

  call hipCheck(hipblasSger(handle,m,n,alpha,dx,1,dy,1,da,m))
  
  !call hipCheck(hipblasGetMatrix(m,n,bytes_per_element,da,m,a,m));
  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(a), da, Nabytes, hipMemcpyDeviceToHost))

  !do i=1,m
  !  do j = 1,n
  !    write(*,*) a(i,j) 
  !  end do
  !end do

  do i = 1,m
  do j = 1,n
    error = abs(3.0 - a(i,j))
      if( error > 10*epsilon(error) )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, "a(i,j) = ", a(i,j)
        call exit(1)
     end if
  end do
  end do

  call hipblasCheck(hipblasDestroy(handle))

  call hipCheck(hipFree(da))
  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  
  deallocate(a)
  deallocate(x)
  deallocate(y)
  
  write(*,*) "PASSED!"
  
end program hip_sger
