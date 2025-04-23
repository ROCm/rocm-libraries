program hipfft_example
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipfft

  implicit none

  integer(c_size_t), parameter :: N = 16
  integer(c_size_t), parameter :: Nbytes = N * 8 * 2

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:) :: hx
  integer(c_int) :: direction = HIPFFT_FORWARD
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer :: i
  integer(kind(HIPFFT_SUCCESS)) :: ierr
  double precision :: error
  double precision, parameter :: error_max = epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'hipFFT' (Fortran 2003 interfaces) - "

  allocate(lengths(3))
  lengths(1) = N

  allocate(hx(N))
  hx(:)%x = 1
  hx(:)%y = -1

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  call hipfftCheck(hipfftPlan1d(plan, int(N, 4), HIPFFT_Z2Z, 1))

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  

  call hipCheck(hipDeviceSynchronize())


  call hipCheck(hipMemcpy(c_loc(hx(1)),dx,Nbytes,hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! Using the C++ version of this as the "gold".
  ! first components were \pm 16 and the remaining componenents
  ! were zero, so the sum of each component pair should be zero
  do i = 1,N
     error = abs(hx(i) % x + hx(i) % y)
     if(error > error_max)then
        write(*,*) "FAILED! Error = ", error, "hx(i)%x = ", hx(i)%x, "hx(i)%y = "
     end if
  end do

  deallocate(hx)
  deallocate(lengths)

  
  call hipfftcheck( hipfftDestroy(plan))

  write(*,*) "PASSED!"

end program hipfft_example