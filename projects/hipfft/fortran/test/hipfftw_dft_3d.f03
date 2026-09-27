program hipfftw_dft_3d_test
  use iso_c_binding
  use hip
  use hipfftw

  implicit none

  integer(c_int), parameter :: NX = 4, NY = 4, NZ = 2
  integer(c_int), parameter :: kx = 1, ky = 2, kz = 1
  integer(c_size_t), parameter :: Nbytes = NX * NY * NZ * 16  ! sizeof(double complex) = 16
  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-10

  complex(c_double_complex), allocatable, target, dimension(:) :: hx, hresult
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer :: ix, iy, iz, kxi, kyi, kzi
  double precision :: error, max_error
  complex(c_double_complex) :: wx, wy, wz, expected

  write(*,"(a)",advance="no") "-- Running test 'hipfftw_dft_3d' (Fortran 2003) - "

  ! Signal: h[ix,iy,iz] = exp(2*pi*i*(kx*ix/NX + ky*iy/NY + kz*iz/NZ)).
  ! Layout: C row-major, NZ contiguous. Index: ix*NY*NZ + iy*NZ + iz (0-indexed).
  ! Expected 3D DFT: output bin (kx,ky,kz) = NX*NY*NZ, all others zero.
  wx = cmplx(0.0d0, 2.0d0 * pi * dble(kx) / dble(NX), kind=c_double_complex)
  wy = cmplx(0.0d0, 2.0d0 * pi * dble(ky) / dble(NY), kind=c_double_complex)
  wz = cmplx(0.0d0, 2.0d0 * pi * dble(kz) / dble(NZ), kind=c_double_complex)
  allocate(hx(NX*NY*NZ), hresult(NX*NY*NZ))
  do ix = 0, NX-1
    do iy = 0, NY-1
      do iz = 0, NZ-1
        hx(ix*NY*NZ + iy*NZ + iz + 1) = &
            exp(wx*dble(ix)) * exp(wy*dble(iy)) * exp(wz*dble(iz))
      end do
    end do
  end do

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMalloc(dy, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! n0=NX (slowest), n1=NY, n2=NZ (fastest); C row-major, no reversal needed.
  plan = fftw_plan_dft_3d(NX, NY, NZ, dx, dy, FFTW_FORWARD, FFTW_ESTIMATE)
  call fftw_execute_dft(plan, dx, dy)
  call fftw_destroy_plan(plan)

  call hipCheck(hipMemcpy(c_loc(hresult(1)), dy, Nbytes, hipMemcpyDeviceToHost))

  ! Verify: output bin (kx,ky,kz) should equal NX*NY*NZ; all others zero.
  max_error = 0.0d0
  do kxi = 0, NX-1
    do kyi = 0, NY-1
      do kzi = 0, NZ-1
        if (kxi == kx .and. kyi == ky .and. kzi == kz) then
          expected = cmplx(dble(NX*NY*NZ), 0.0d0, kind=c_double_complex)
        else
          expected = cmplx(0.0d0, 0.0d0, kind=c_double_complex)
        end if
        error = abs(hresult(kxi*NY*NZ + kyi*NZ + kzi + 1) - expected)
        max_error = max(max_error, error)
      end do
    end do
  end do

  if (max_error > tol * dble(NX*NY*NZ)) then
    write(*,*) "FAILED! max error = ", max_error
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  deallocate(hx, hresult)

  write(*,*) "PASSED!"
end program hipfftw_dft_3d_test
