program hipfft_example
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_hipfft

  implicit none

  integer(c_int), parameter :: N = 16

  complex(8), allocatable, dimension(:) :: hx
  integer(c_int) :: direction = HIPFFT_FORWARD
  complex(8), pointer, dimension(:) :: dx
  type(c_ptr) :: plan = c_null_ptr
  integer(c_size_t)            :: lengths(3)
  integer(c_size_t), parameter :: one = 1
  integer :: i
  integer(kind(HIPFFT_SUCCESS)) :: ierr
  double precision :: error
  double precision, parameter :: error_max = epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'hipFFT' (Fortran 2008 interfaces) - "

  lengths(1) = N

  allocate(hx(N))
  hx(:) = (1, -1)

  call hipCheck(hipMalloc(dx, source=hx))

  call hipfftCheck(hipfftPlan1d(plan, N, HIPFFT_Z2Z, 1))

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, direction))
  
  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(hx,dx,hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! Using the C++ version of this as the "gold".
  ! first components were \pm 16 and the remaining componenents
  ! were zero, so the sum of each component pair should be zero
  do i = 1,N
     error = abs(DBLE(hx(i)) + AIMAG(hx(i)))
     if(error > error_max)then
        write(*,*) "FAILED! Error = ", error, "hx(i)%x = ", DBLE(hx(i)), "hx(i)%y = ", AIMAG(hx(i))
     end if
  end do

  deallocate(hx)

  call hipfftcheck( hipfftDestroy(plan))

  write(*,*) "PASSED!"
end program hipfft_example
