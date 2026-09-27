program hipfft_planmany_2d_z2z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  ! Advanced-interface 2D batched C2C transform. The data layout follows the
  ! C convention used by hipfftPlanMany: the last dimension (Ny) is contiguous.
  integer(c_int), parameter :: rank = 2
  integer(c_int), parameter :: Nx = 4, Ny = 5
  integer(c_int), parameter :: howmany = 3
  integer(c_int), parameter :: Ntot = Nx * Ny
  integer(c_size_t), parameter :: Nbytes = int(Ntot,c_size_t) * howmany * 16

  integer(c_int), target :: n(rank) = [Nx, Ny]
  integer(c_int), target :: inembed(rank) = [Nx, Ny]
  integer(c_int), target :: onembed(rank) = [Nx, Ny]
  integer(c_int), parameter :: istride = 1, ostride = 1
  integer(c_int), parameter :: idist = Nx * Ny, odist = Nx * Ny

  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-8

  complex(c_double_complex), allocatable, target, dimension(:) :: hx
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer :: b, i, j, kx, ky, pos
  complex(c_double_complex) :: expected
  double precision :: error, max_error

  write(*,"(a)",advance="no") "-- Running test 'hipFFT PlanMany C2C 2D batched (z)' (Fortran 2003 interfaces) - "

  allocate(hx(Ntot * howmany))

  ! Batch b holds the pure harmonic exp(2*pi*i*(kx*ii/Nx + ky*jj/Ny)) with
  ! kx = b mod Nx and ky = (b+1) mod Ny, so the forward transform must place
  ! Nx*Ny in bin (kx,ky) of that batch and zero everywhere else. A wrong
  ! stride, embed or batch distance moves energy into other bins and fails.
  do b = 0, howmany - 1
     kx = mod(b, Nx)
     ky = mod(b + 1, Ny)
     do i = 0, Nx - 1
        do j = 0, Ny - 1
           pos = b * idist + i * inembed(2) + j
           hx(pos + 1) = exp(cmplx(0.0d0, &
                2.0d0 * pi * (dble(kx * i) / dble(Nx) + dble(ky * j) / dble(Ny)), &
                kind=c_double_complex))
        end do
     end do
  end do

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  call hipfftCheck(hipfftPlanMany(plan, rank, c_loc(n), c_loc(inembed), istride, idist, &
       c_loc(onembed), ostride, odist, HIPFFT_Z2Z, howmany))

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, HIPFFT_FORWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  max_error = 0.0d0
  do b = 0, howmany - 1
     kx = mod(b, Nx)
     ky = mod(b + 1, Ny)
     do i = 0, Nx - 1
        do j = 0, Ny - 1
           pos = b * odist + i * onembed(2) + j
           if (i == kx .and. j == ky) then
              expected = cmplx(dble(Ntot), 0.0d0, kind=c_double_complex)
           else
              expected = (0.0d0, 0.0d0)
           end if
           error = abs(hx(pos + 1) - expected)
           max_error = max(max_error, error)
        end do
     end do
  end do

  if (max_error > tol * Ntot) then
     write(*,*) "FAILED! max error = ", max_error
     STOP 1
  end if

  deallocate(hx)

  write(*,*) "PASSED!"

end program hipfft_planmany_2d_z2z
