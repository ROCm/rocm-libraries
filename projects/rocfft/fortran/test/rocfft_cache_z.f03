! Demonstrates the compiled-kernel cache described in the rocFFT
! "Runtime compilation" guide: rocFFT compiles the kernels a plan needs when the
! plan is created, and the resulting cache can be serialized, moved and loaded
! back into another process.
program rocfft_cache_z
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  ! A length that is not covered by the kernels built into rocFFT, so that plan
  ! creation has to compile kernels at runtime and the cache is not empty.
  integer(c_size_t), parameter :: N = 10007
  integer(c_size_t), parameter :: Nbytes = N*8*2

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan_fwd = c_null_ptr
  type(c_ptr) :: plan_bwd = c_null_ptr
  type(c_ptr) :: cache = c_null_ptr
  integer(c_size_t), target :: cache_bytes = 0
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'rocFFT kernel cache double (z)' &
                              &(Fortran 2003 interfaces) - "

  call rocfftCheck(rocfft_setup())

  allocate(lengths(1))
  lengths(1) = N

  allocate(hx(N))
  allocate(hx_input(N))
  do i = 1, N
     hx(i)%x = dble(mod(i,11)) - 5.0d0
     hx(i)%y = dble(mod(i,7)) - 3.0d0
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! Creating the plan populates the in-memory kernel cache.
  call rocfftCheck(rocfft_plan_create(plan_fwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))

  ! Copy the cache into a buffer that rocFFT allocates. The buffer address is
  ! written to the first argument, its size in bytes to the second, which is
  ! why the length is passed as the address of a size_t variable.
  call rocfftCheck(rocfft_cache_serialize(cache, c_loc(cache_bytes)))
  if (.not. c_associated(cache) .or. cache_bytes == 0) then
     write(*,*) "FAILED! empty kernel cache"
     call rocfftCheck(rocfft_cleanup())
     STOP 1
  end if

  ! The buffer can be written to a file and loaded in another process. Loading
  ! it back replaces matching kernels and leaves the rest of the cache alone,
  ! so plans created afterwards reuse the kernels instead of recompiling them.
  call rocfftCheck(rocfft_cache_deserialize(cache, cache_bytes))

  ! The buffer belongs to the caller once it has been handed out.
  call rocfftCheck(rocfft_cache_buffer_free(cache))
  cache = c_null_ptr

  call rocfftCheck(rocfft_plan_create(plan_bwd,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_inverse,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths(1)),&
                                      one,&
                                      c_null_ptr))

  call rocfftCheck(rocfft_execute(plan_fwd, dx, c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())
  call rocfftCheck(rocfft_execute(plan_bwd, dx, c_null_ptr, c_null_ptr))
  call hipCheck(hipDeviceSynchronize())

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

  write(*,*) "PASSED! kernel cache size: ", cache_bytes, " bytes"

end program rocfft_cache_z
