program rocfft_r2c_c2r_1d_s
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: N = 16
  integer(c_size_t), parameter :: Ncomplex = N/2 + 1
  integer(c_size_t), parameter :: Rbytes = N*4
  integer(c_size_t), parameter :: Cbytes = Ncomplex*4*2

  real(4), allocatable, target, dimension(:) :: hr, hr_input
  type(c_ptr) :: dr = c_null_ptr
  type(c_ptr) :: dc = c_null_ptr
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer :: i
  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'rocFFT R2C/C2R 1D single (s)' (Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_setup())

  allocate(lengths(1))
  lengths(1) = N

  allocate(hr(N))
  allocate(hr_input(N))
  do i = 1, N
     hr(i) = real(i) + real(mod(i,3)) - real(mod(i,7))
  end do
  hr_input(:) = hr(:)

  call hipCheck(hipMalloc(dr, Rbytes))
  call hipCheck(hipMalloc(dc, Cbytes))
  call hipCheck(hipMemcpy(dr, c_loc(hr(1)), Rbytes, hipMemcpyHostToDevice))

  ! Forward real-to-complex (out-of-place): dr -> dc.
  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_notinplace,&
                                      rocfft_transform_type_real_forward,&
                                      rocfft_precision_single,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_execute(plan_fwd, dr, dc, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_fwd))

  ! Inverse complex-to-real (out-of-place): dc -> dr. rocFFT unnormalized -> N*input.
  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_notinplace,&
                                      rocfft_transform_type_real_inverse,&
                                      rocfft_precision_single,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_execute(plan_bwd, dc, dr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_bwd))

  call hipCheck(hipMemcpy(c_loc(hr(1)), dr, Rbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dr))
  call hipCheck(hipFree(dc))

  ! After forward+inverse the real data should equal N times the original input.
  do i = 1, N
     error = abs(hr(i) - N * hr_input(i))
     if (error > error_max * N) then
        write(*,*) "FAILED! i=", i, " error=", error
        call rocfftCheck(rocfft_cleanup())
        STOP 1
     end if
  end do

  deallocate(hr)
  deallocate(hr_input)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED!"

end program rocfft_r2c_c2r_1d_s
