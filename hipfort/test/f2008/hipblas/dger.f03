program hip_dger

  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none

  integer, parameter ::  m = 100, n = 100
  double precision, parameter ::  alpha = 1.1d0

  double precision, allocatable, target, dimension(:)   :: hx, hy
  double precision, allocatable, target, dimension(:,:) :: hA

  type(c_ptr) :: handle = c_null_ptr
  
  double precision, pointer, dimension(:)   :: dx, dy
  double precision, pointer, dimension(:,:) :: dA

  integer :: i,j
  double precision :: error

  write(*,"(a)",advance="no") "-- Running test 'dger' (Fortran 2008 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(hx(m))
  allocate(hy(n))
  allocate(ha(m,n))

  hx(:)   = 1.d0
  hy(:)   = 1.d0
  hA(:,:) = 1.d0

  ! Allocate device memory
  call hipCheck(hipMalloc(dx,m))
  call hipCheck(hipMalloc(dy,n))
  call hipCheck(hipMalloc(dA,m,n))

  !Transfer from host to device
  call hipCheck(hipMemcpy(dx, hx, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, hy, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dA, hA, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasDger(handle,m,n,alpha,dx,1,dy,1,dA,m))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(hA, dA, hipMemcpyDeviceToHost))

  do j = 1,n
    do i = 1,m
     error = abs(2.1d0 - hA(i,j))
     if( error > 10*epsilon(error) )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, "hA(i,j) = ", hA(i,j)
        call exit(1)
     end if
   end do
  end do

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipCheck(hipFree(dA))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(hx,hy,hA)

  write(*,*) "PASSED!"

end program hip_dger
