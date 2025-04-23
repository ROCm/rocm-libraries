program hip_dger

  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none

  integer, parameter ::  m = 100, n = 100
  double precision, parameter ::  alpha = 1.1d0
  integer, parameter :: bytes_per_element = 8 !double precision

  double precision, allocatable, target, dimension(:) :: hx, hy, hA

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr, dA = c_null_ptr
  integer(c_size_t), parameter :: Nbytes = m*bytes_per_element

  integer :: i
  double precision :: error

  write(*,"(a)",advance="no") "-- Running test 'dger' (Fortran 2003 interfaces) - "

  call hipblasCheck(hipblasCreate(handle))

  allocate(hx(m))
  allocate(hy(n))
  allocate(ha(m*n))

  hx(:) = 1.d0
  hy(:) = 1.d0
  hA(:) = 1.d0

  ! Allocate device memory
  call hipCheck(hipMalloc(dx,Nbytes))
  call hipCheck(hipMalloc(dy,Nbytes))
  call hipCheck(hipMalloc(dA,Nbytes*n))

  !Transfer from host to device
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dy, c_loc(hy(1)), Nbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), Nbytes*n, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasDger(handle,m,n,alpha,dx,1,dy,1,dA,m))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(hA(1)), dA, Nbytes*n, hipMemcpyDeviceToHost))

  do i = 1,m*n
     error = abs(2.1d0 - hA(i))
     if( error > 10*epsilon(error) )then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, "hA(i) = ", hA(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  call hipCheck(hipFree(dA))

  call hipblasCheck(hipblasDestroy(handle))

  deallocate(hx)
  deallocate(hy)
  deallocate(hA)

  write(*,*) "PASSED!"

end program hip_dger
