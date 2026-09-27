! Demonstrates rocfft_plan_get_print, the plan introspection helper described in
! the rocFFT "Working with rocFFT" guide. It writes a human-readable summary of
! a plan to stdout, which is the quickest way to check how rocFFT interpreted
! the lengths, strides and distances of a plan description.
program rocfft_plan_print_z
  use iso_c_binding
  use iso_fortran_env, only: output_unit
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: Nx = 4
  integer(c_size_t), parameter :: Ny = 6
  integer(c_size_t), parameter :: N = Nx*Ny
  integer(c_size_t), parameter :: Nbytes = N*8*2

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:,:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer(c_size_t), parameter :: two = 2
  integer :: i, j
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  call rocfftCheck(rocfft_setup())

  ! lengths starts with the fastest-varying dimension, matching Fortran's
  ! column-major storage.
  allocate(lengths(2))
  lengths(1) = Nx
  lengths(2) = Ny

  allocate(hx(Nx,Ny))
  allocate(hx_input(Nx,Ny))
  do j = 1, Ny
     do i = 1, Nx
        hx(i,j)%x = dble(i)
        hx(i,j)%y = dble(j)
     end do
  end do
  hx_input(:,:) = hx(:,:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1,1)), Nbytes, hipMemcpyHostToDevice))

  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      two,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))

  ! rocfft_plan_get_print writes to stdout from C, so flush the Fortran unit
  ! first to keep the two output streams in order.
  write(*,"(a)") "-- rocfft_plan_get_print output for a 2D complex plan:"
  flush(output_unit)
  call rocfftCheck(rocfft_plan_get_print(plan_fwd))

  write(*,"(a)",advance="no") "-- Running test 'rocFFT plan print double (z)' &
                              &(Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_inverse,&
                                      rocfft_precision_double,&
                                      two,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))

  call rocfftCheck(rocfft_execute(plan_fwd, dx, c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_execute(plan_bwd, dx, c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())

  call rocfftCheck(rocfft_plan_destroy(plan_fwd))
  call rocfftCheck(rocfft_plan_destroy(plan_bwd))

  call hipCheck(hipMemcpy(c_loc(hx(1,1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! rocFFT is unnormalized, so the round trip yields Nx*Ny*input.
  do j = 1, Ny
     do i = 1, Nx
        error = abs(hx(i,j)%x - N * hx_input(i,j)%x) &
              + abs(hx(i,j)%y - N * hx_input(i,j)%y)
        if (error > error_max * N) then
           write(*,*) "FAILED! i=", i, " j=", j, " error=", error
           call rocfftCheck(rocfft_cleanup())
           STOP 1
        end if
     end do
  end do

  deallocate(hx)
  deallocate(hx_input)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED!"

end program rocfft_plan_print_z
