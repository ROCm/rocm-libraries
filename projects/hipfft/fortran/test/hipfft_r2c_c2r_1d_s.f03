program hipfft_r2c_c2r_1d_s
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N = 16
  integer(c_int), parameter :: Ncomplex = N/2 + 1
  integer(c_size_t), parameter :: Rbytes = int(N,c_size_t)*4
  integer(c_size_t), parameter :: Cbytes = int(Ncomplex,c_size_t)*4*2

  real(4), allocatable, target, dimension(:) :: hr, hr_input
  type(c_ptr) :: dr = c_null_ptr
  type(c_ptr) :: dc = c_null_ptr
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  integer :: i
  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'hipFFT R2C/C2R 1D single (s)' (Fortran 2003 interfaces) - "

  allocate(hr(N))
  allocate(hr_input(N))
  do i = 1, N
     hr(i) = real(i) + real(mod(i,3)) - real(mod(i,7))
  end do
  hr_input(:) = hr(:)

  call hipCheck(hipMalloc(dr, Rbytes))
  ! The complex (Hermitian) output holds N/2+1 elements due to Hermitian symmetry.
  call hipCheck(hipMalloc(dc, Cbytes))
  call hipCheck(hipMemcpy(dr, c_loc(hr(1)), Rbytes, hipMemcpyHostToDevice))

  ! Forward real-to-complex (out-of-place): dr -> dc.
  call hipfftCheck(hipfftPlan1d(plan_fwd, N, HIPFFT_R2C, 1))
  call hipfftCheck(hipfftExecR2C(plan_fwd, dr, dc))
  call hipCheck(hipDeviceSynchronize())
  call hipfftCheck(hipfftDestroy(plan_fwd))

  ! Inverse complex-to-real (out-of-place): dc -> dr. hipFFT unnormalized -> N*input.
  call hipfftCheck(hipfftPlan1d(plan_bwd, N, HIPFFT_C2R, 1))
  call hipfftCheck(hipfftExecC2R(plan_bwd, dc, dr))
  call hipCheck(hipDeviceSynchronize())
  call hipfftCheck(hipfftDestroy(plan_bwd))

  call hipCheck(hipMemcpy(c_loc(hr(1)), dr, Rbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dr))
  call hipCheck(hipFree(dc))

  ! After forward+inverse the real data should equal N times the original input.
  do i = 1, N
     error = abs(hr(i) - N * hr_input(i))
     if (error > error_max * N) then
        write(*,*) "FAILED! i=", i, " error=", error
        STOP 1
     end if
  end do

  deallocate(hr)
  deallocate(hr_input)

  write(*,*) "PASSED!"

end program hipfft_r2c_c2r_1d_s
