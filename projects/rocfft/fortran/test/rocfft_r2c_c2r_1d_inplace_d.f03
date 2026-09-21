! Fortran counterpart of the in-place path of the rocFFT sample
! clients/samples/rocfft/rocfft_example_realcomplex.cpp: an in-place real
! transform, where the real buffer is padded so that it can also hold the
! Hermitian-symmetric spectrum.
program rocfft_r2c_c2r_1d_inplace_d
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: N = 16
  integer(c_size_t), parameter :: Ncomplex = N/2 + 1
  ! An in-place real transform needs room for N/2+1 complex values, that is
  ! two extra reals in the contiguous dimension.
  integer(c_size_t), parameter :: Npad = 2 * Ncomplex
  integer(c_size_t), parameter :: Nbytes = Npad*8

  real(8), allocatable, target, dimension(:) :: hr, hr_input
  type(c_ptr) :: dr = c_null_ptr
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  type(c_ptr) :: desc_fwd = c_null_ptr
  type(c_ptr) :: desc_bwd = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), target :: strides(1)
  integer(c_size_t), parameter :: one = 1
  integer(c_size_t), parameter :: zero = 0
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'rocFFT R2C/C2R 1D in-place double (d)' &
                              &(Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_setup())

  allocate(lengths(1))
  lengths(1) = N
  strides(1) = 1

  allocate(hr(Npad))
  allocate(hr_input(Npad))
  hr(:) = 0.0d0
  do i = 1, N
     hr(i) = dble(i) + dble(mod(i,3)) - dble(mod(i,7))
  end do
  hr_input(:) = hr(:)

  call hipCheck(hipMalloc(dr, Nbytes))
  call hipCheck(hipMemcpy(dr, c_loc(hr(1)), Nbytes, hipMemcpyHostToDevice))

  ! Forward: real input, Hermitian-interleaved output, both in the same buffer.
  call rocfftCheck(rocfft_plan_description_create(desc_fwd))
  call rocfftCheck(rocfft_plan_description_set_data_layout(desc_fwd,&
                       rocfft_array_type_real,&
                       rocfft_array_type_hermitian_interleaved,&
                       c_null_ptr, c_null_ptr,&
                       one, c_loc(strides(1)), zero,&
                       one, c_loc(strides(1)), zero))
  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_real_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      desc_fwd))
  call rocfftCheck(rocfft_execute(plan_fwd, dr, c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_fwd))
  call rocfftCheck(rocfft_plan_description_destroy(desc_fwd))

  ! Inverse: the array types are swapped. rocFFT is unnormalized, so the round
  ! trip yields N*input.
  call rocfftCheck(rocfft_plan_description_create(desc_bwd))
  call rocfftCheck(rocfft_plan_description_set_data_layout(desc_bwd,&
                       rocfft_array_type_hermitian_interleaved,&
                       rocfft_array_type_real,&
                       c_null_ptr, c_null_ptr,&
                       one, c_loc(strides(1)), zero,&
                       one, c_loc(strides(1)), zero))
  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_real_inverse,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      desc_bwd))
  call rocfftCheck(rocfft_execute(plan_bwd, dr, c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_bwd))
  call rocfftCheck(rocfft_plan_description_destroy(desc_bwd))

  call hipCheck(hipMemcpy(c_loc(hr(1)), dr, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dr))

  ! Only the first N reals hold the signal; the padding is scratch space.
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

end program rocfft_r2c_c2r_1d_inplace_d
