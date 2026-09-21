program hipfft_c2c_1d_c
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N = 16
  integer(c_size_t), parameter :: Nbytes = int(N,c_size_t)*4*2

  type float2
     real :: x
     real :: y
  end type float2

  type(float2), allocatable, target, dimension(:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer :: i
  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'hipFFT C2C 1D single (c)' (Fortran 2003 interfaces) - "

  allocate(hx(N))
  allocate(hx_input(N))
  do i = 1, N
     hx(i)%x = real(i)
     hx(i)%y = real(N - i)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! A single hipFFT plan covers both directions: the transform direction is
  ! passed to hipfftExecC2C rather than baked into the plan.
  call hipfftCheck(hipfftPlan1d(plan, N, HIPFFT_C2C, 1))

  call hipfftCheck(hipfftExecC2C(plan, dx, dx, HIPFFT_FORWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftExecC2C(plan, dx, dx, HIPFFT_BACKWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! hipFFT is unnormalized, so forward+inverse yields N times the original input.
  do i = 1, N
     error = abs(hx(i)%x - N * hx_input(i)%x) + abs(hx(i)%y - N * hx_input(i)%y)
     if (error > error_max * N) then
        write(*,*) "FAILED! i=", i, " error=", error
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)

  write(*,*) "PASSED!"

end program hipfft_c2c_1d_c
