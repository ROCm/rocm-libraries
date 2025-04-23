program hip_dscal

  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipblas

  implicit none

  integer, parameter :: N = 10240;
  integer, parameter :: bytes_per_element = 8 !double precision
  integer(c_size_t), parameter :: Nbytes = N*bytes_per_element

  double precision, parameter :: alpha = 10.d0

  type(c_ptr) :: dx = c_null_ptr

  double precision,allocatable,target,dimension(:) :: hx
  double precision,allocatable,dimension(:) ::hx_scaled

  double precision :: error
  double precision, parameter :: error_max = 10*epsilon(error_max)

  type(c_ptr) :: hip_blas_handle = c_null_ptr

  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'dscal' (Fortran 2003 interfaces) - "
  allocate(hx(N))
  allocate(hx_scaled(N))

  hx(:) = 10.d0

  hx_scaled = alpha*hx

  call hipblasCheck(hipblasCreate(hip_blas_handle))

  call hipCheck(hipMalloc(dx,Nbytes))

   ! Transfer data from host to device memory
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  call hipblasCheck(hipblasDscal(hip_blas_handle, N, alpha, dx, 1))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))

  call hipCheck(hipFree(dx))

  ! Verification
  do i = 1,N
     error = abs(hx(i) - hx_scaled(i) )
     if( error .gt. error_max ) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " hx(i) = ", hx(i)
        call exit
     endif
  end do

  call hipblasCheck(hipblasDestroy(hip_blas_handle))

  deallocate(hx_scaled)
  deallocate(hx)

  write(*,*) "PASSED!"

end program hip_dscal