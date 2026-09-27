! Fortran counterpart of clients/tests/default_callbacks_test.cpp: register the
! load and store callbacks of an execution info handle and transform with them.
! Fortran cannot take the address of a device function, so the callbacks are the
! null (default) ones, which rocFFT documents as clearing any registration.
program rocfft_callback_z
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  integer(c_size_t), parameter :: N = 1024
  integer(c_size_t), parameter :: Nbytes = N*8*2

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  type(c_ptr) :: info = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer(kind(rocfft_status_success)) :: stat
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'rocFFT callbacks double (z)' &
                              &(Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_setup())

  allocate(lengths(1))
  lengths(1) = N

  allocate(hx(N))
  allocate(hx_input(N))
  do i = 1, N
     hx(i)%x = dble(mod(i,7)) - 3.0d0
     hx(i)%y = dble(mod(i,5)) - 2.0d0
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  call rocfftCheck(rocfft_execution_info_create(info))

  call rocfftCheck(rocfft_execution_info_set_load_callback(info, c_null_ptr, c_null_ptr, &
                                                           0_c_size_t))
  call rocfftCheck(rocfft_execution_info_set_store_callback(info, c_null_ptr, c_null_ptr, &
                                                            0_c_size_t))

  ! rocFFT allocates no LDS for callbacks, so a shared memory request is rejected.
  stat = rocfft_execution_info_set_load_callback(info, c_null_ptr, c_null_ptr, 16_c_size_t)
  if (stat /= rocfft_status_invalid_arg_value) then
     write(*,*) "FAILED! load callback with shared memory returned", stat, &
                " expected", int(rocfft_status_invalid_arg_value)
     call rocfftCheck(rocfft_cleanup())
     STOP 1
  end if

  stat = rocfft_execution_info_set_store_callback(info, c_null_ptr, c_null_ptr, 16_c_size_t)
  if (stat /= rocfft_status_invalid_arg_value) then
     write(*,*) "FAILED! store callback with shared memory returned", stat, &
                " expected", int(rocfft_status_invalid_arg_value)
     call rocfftCheck(rocfft_cleanup())
     STOP 1
  end if

  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_execute(plan_fwd, dx, c_null_ptr, info))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_fwd))

  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_inverse,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_execute(plan_bwd, dx, c_null_ptr, info))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_plan_destroy(plan_bwd))

  call rocfftCheck(rocfft_execution_info_destroy(info))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! The default callbacks are plain loads and stores, so the unnormalized round
  ! trip still yields N*input.
  do i = 1, N
     error = abs(hx(i)%x - N * hx_input(i)%x) + abs(hx(i)%y - N * hx_input(i)%y)
     if (error > error_max * N) then
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

end program rocfft_callback_z
