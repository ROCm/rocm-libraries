! Fortran counterpart of the work-buffer handling in the rocFFT sample
! clients/samples/rocfft/rocfft_example_complexcomplex.cpp: query the scratch
! memory a plan needs and supply it through an execution info handle.
program rocfft_work_buffer_z
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  ! Large transforms are split into several kernels and need scratch memory.
  integer(c_size_t), parameter :: N = 262144
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
  type(c_ptr) :: dwork = c_null_ptr
  integer(c_size_t) :: work_fwd = 0
  integer(c_size_t) :: work_bwd = 0
  integer(c_size_t) :: work_bytes = 0
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'rocFFT work buffer double (z)' &
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

  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))
  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_inverse,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))

  ! Ask each plan how much scratch memory it needs. rocFFT allocates the work
  ! buffer itself when none is supplied; providing one explicitly lets the
  ! application control its lifetime and share a single buffer between plans.
  call rocfftCheck(rocfft_plan_get_work_buffer_size(plan_fwd, work_fwd))
  call rocfftCheck(rocfft_plan_get_work_buffer_size(plan_bwd, work_bwd))
  work_bytes = max(work_fwd, work_bwd)

  call rocfftCheck(rocfft_execution_info_create(info))
  if (work_bytes > 0) then
     call hipCheck(hipMalloc(dwork, work_bytes))
     call rocfftCheck(rocfft_execution_info_set_work_buffer(info, dwork, work_bytes))
  end if

  call rocfftCheck(rocfft_execute(plan_fwd, dx, c_null_ptr, info))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_execute(plan_bwd, dx, c_null_ptr, info))
  call hipCheck(hipDeviceSynchronize())

  call rocfftCheck(rocfft_execution_info_destroy(info))
  if (work_bytes > 0) then
     call hipCheck(hipFree(dwork))
  end if
  call rocfftCheck(rocfft_plan_destroy(plan_fwd))
  call rocfftCheck(rocfft_plan_destroy(plan_bwd))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! rocFFT is unnormalized, so the round trip yields N*input.
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

end program rocfft_work_buffer_z
