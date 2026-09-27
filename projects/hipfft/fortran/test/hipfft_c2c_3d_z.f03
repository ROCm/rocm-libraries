program hipfft_c2c_3d_z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: Nx = 4, Ny = 4, Nz = 8
  integer(c_int), parameter :: Ntot = Nx * Ny * Nz
  integer(c_size_t), parameter :: Nbytes = int(Ntot,c_size_t)*8*2

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'hipFFT C2C 3D double (z)' (Fortran 2003 interfaces) - "

  allocate(hx(Ntot))
  allocate(hx_input(Ntot))
  do i = 1, Ntot
     hx(i)%x = dble(i)
     hx(i)%y = dble(Ntot - i)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! In-place 3D C2C plan over an Nx*Ny*Nz complex double volume; the direction
  ! is supplied at exec time rather than at plan creation.
  call hipfftCheck(hipfftPlan3d(plan, Nx, Ny, Nz, HIPFFT_Z2Z))

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, HIPFFT_FORWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, HIPFFT_BACKWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! hipFFT is unnormalized, so forward+inverse yields Ntot times the original input.
  do i = 1, Ntot
     error = abs(hx(i)%x - Ntot * hx_input(i)%x) + abs(hx(i)%y - Ntot * hx_input(i)%y)
     if (error > error_max * Ntot) then
        write(*,*) "FAILED! i=", i, " error=", error
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)

  write(*,*) "PASSED!"

end program hipfft_c2c_3d_z
