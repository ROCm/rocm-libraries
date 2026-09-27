program rocfft_c2c_2d_z
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: Nx = 4, Ny = 8
  integer(c_size_t), parameter :: Ntot = Nx * Ny
  integer(c_size_t), parameter :: Nbytes = Ntot*8*2

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: two = 2
  integer(c_size_t), parameter :: one = 1
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'rocFFT C2C 2D double (z)' (Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_setup())

  allocate(lengths(2))
  lengths(1) = Nx
  lengths(2) = Ny

  allocate(hx(Ntot))
  allocate(hx_input(Ntot))
  do i = 1, Ntot
     hx(i)%x = dble(i)
     hx(i)%y = dble(Ntot - i)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! Forward 2D transform (in-place).
  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      two,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_execute(plan_fwd, dx, c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_fwd))

  ! Inverse 2D transform (in-place). rocFFT unnormalized -> Ntot*input.
  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_inverse,&
                                      rocfft_precision_double,&
                                      two,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_execute(plan_bwd, dx, c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_bwd))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! After forward+inverse the data should equal Ntot times the original input.
  do i = 1, Ntot
     error = abs(hx(i)%x - Ntot * hx_input(i)%x) + abs(hx(i)%y - Ntot * hx_input(i)%y)
     if (error > error_max * Ntot) then
        write(*,*) "FAILED! i=", i, " error=", error
        call rocfftCheck(rocfft_cleanup())
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED!"

end program rocfft_c2c_2d_z
