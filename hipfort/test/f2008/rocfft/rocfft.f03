program rocfft_example
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_rocfft

  implicit none

  integer(c_size_t), parameter :: N=16

  complex(8), allocatable, target, dimension(:) :: hx
  complex(8), pointer, dimension(:) :: dx => null()
  type(c_ptr) :: plan = c_null_ptr
  integer(c_size_t), allocatable, target, dimension(:) :: lengths
  integer(c_size_t), parameter :: one = 1
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = epsilon(error)


  allocate(lengths(3))
  lengths(1) = N
  allocate(hx(N))
  hx(:) = (1., -1.)
  
  call hipCheck(hipMalloc(dx,source=hx))
  
  write(*,"(a)",advance="no") "-- Running test 'rocFFT' (Fortran 2008 interfaces) - "

  call rocfftCheck(rocfft_setup())
  call rocfftCheck(rocfft_plan_create(plan,&
                                      rocfft_placement_inplace,&
                                      rocfft_transform_type_complex_forward,&
                                      rocfft_precision_double,&
                                      one,&
                                      c_loc(lengths),&
                                      one,&
                                      c_null_ptr))

  call rocfftCheck(rocfft_execute(plan,c_loc(dx),c_null_ptr,c_null_ptr))

  call hipCheck(hipDeviceSynchronize())

  call rocfftCheck(rocfft_plan_destroy(plan))

  call hipCheck(hipMemcpy(hx,dx,hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! Using the C++ version of this as the "gold".
  ! first components were \pm 16 and the remaining componenents
  ! were zero, so the sum of each component pair should be zero
  do i = 1,N
     error = DBLE(hx(i)) + AIMAG(hx(i))
     if(error > error_max)then
        write(*,*) "FAILED! Error = ", error, "hx(i)=",hx(i)
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(lengths)

  call rocfftCheck(rocfft_cleanup())

  write(*,*) "PASSED!"

end program rocfft_example
