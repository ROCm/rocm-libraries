program hipfftw_dft_2d_test
  use iso_c_binding
  use hip
  use hipfftw

  implicit none

  integer(c_int), parameter :: NX = 4, NY = 8
  integer(c_int), parameter :: kx = 1, ky = 3
  integer(c_size_t), parameter :: Nbytes = NX * NY * 16  ! sizeof(double complex) = 16
  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-10

  complex(c_double_complex), allocatable, target, dimension(:) :: hx, hresult
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  integer :: ii, jj, kxi, kyi
  double precision :: error, max_error
  complex(c_double_complex) :: wx, wy, expected

  write(*,"(a)",advance="no") "-- Running test 'hipfftw_dft_2d' (Fortran 2003) - "

  ! Signal: h[ii,jj] = exp(2*pi*i*(kx*ii/NX + ky*jj/NY)), C row-major (jj contiguous).
  ! Linear index: ii*NY + jj (0-indexed).
  ! Expected 2D DFT: output bin (kx,ky) = NX*NY, all others zero.
  wx = cmplx(0.0d0, 2.0d0 * pi * dble(kx) / dble(NX), kind=c_double_complex)
  wy = cmplx(0.0d0, 2.0d0 * pi * dble(ky) / dble(NY), kind=c_double_complex)
  allocate(hx(NX*NY), hresult(NX*NY))
  do ii = 0, NX-1
    do jj = 0, NY-1
      hx(ii*NY + jj + 1) = exp(wx*dble(ii)) * exp(wy*dble(jj))
    end do
  end do

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMalloc(dy, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! n0=NX (slow), n1=NY (fast); C row-major, no dimension reversal needed.
  plan = fftw_plan_dft_2d(NX, NY, dx, dy, FFTW_FORWARD, FFTW_ESTIMATE)
  call fftw_execute_dft(plan, dx, dy)
  call fftw_destroy_plan(plan)

  call hipCheck(hipMemcpy(c_loc(hresult(1)), dy, Nbytes, hipMemcpyDeviceToHost))

  ! Verify: output bin (kx,ky) should equal NX*NY; all others zero.
  max_error = 0.0d0
  do kxi = 0, NX-1
    do kyi = 0, NY-1
      if (kxi == kx .and. kyi == ky) then
        expected = cmplx(dble(NX*NY), 0.0d0, kind=c_double_complex)
      else
        expected = cmplx(0.0d0, 0.0d0, kind=c_double_complex)
      end if
      error = abs(hresult(kxi*NY + kyi + 1) - expected)
      max_error = max(max_error, error)
    end do
  end do

  if (max_error > tol * dble(NX*NY)) then
    write(*,*) "FAILED! max error = ", max_error
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  deallocate(hx, hresult)

  write(*,*) "PASSED!"
end program hipfftw_dft_2d_test
