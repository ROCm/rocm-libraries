program hipfft_c2c_1d_batched_z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N = 16
  integer(c_int), parameter :: Nbatch = 4
  integer(c_int), parameter :: Ntot = N * Nbatch
  integer(c_size_t), parameter :: Nbytes = int(Ntot,c_size_t)*8*2

  ! one_i avoids ambiguity: the raw hipfftPlanMany_ interface requires integer(c_int)
  ! VALUE arguments, and a bare literal '1' would have the default integer kind.
  integer(c_int), parameter :: one_i = 1

  type double2
     double precision :: x
     double precision :: y
  end type double2

  type(double2), allocatable, target, dimension(:) :: hx, hx_input
  type(c_ptr) :: dx = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  ! nlen is the per-dimension transform length array passed by address to
  ! hipfftPlanMany (it cannot be called "n": Fortran is case-insensitive and N
  ! is already the transform length parameter).
  integer(c_int), target :: nlen(1)
  integer :: i
  double precision :: error
  double precision, parameter :: error_max = 1.0d-8

  write(*,"(a)",advance="no") "-- Running test 'hipFFT C2C 1D batched double (z)' (Fortran 2003 interfaces) - "

  nlen(1) = N

  allocate(hx(Ntot))
  allocate(hx_input(Ntot))
  do i = 1, Ntot
     hx(i)%x = dble(i)
     hx(i)%y = dble(Ntot - i)
  end do
  hx_input(:) = hx(:)

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  ! NULL inembed/onembed selects the simple contiguous batched layout: each
  ! transform spans N consecutive complex doubles (istride=1, idist=N,
  ! ostride=1, odist=N). Passing c_loc(n) and c_null_ptr routes to the raw
  ! hipfftPlanMany_ C interface rather than the Fortran array overloads.
  ! Unlike rocFFT, a single plan handle serves both directions; the direction
  ! is supplied at exec time via hipfftExecZ2Z.
  call hipfftCheck(hipfftPlanMany(plan, one_i, c_loc(nlen), &
                                  c_null_ptr, one_i, N, &
                                  c_null_ptr, one_i, N, &
                                  HIPFFT_Z2Z, Nbatch))

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, HIPFFT_FORWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, HIPFFT_BACKWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))

  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))

  ! hipFFT is unnormalized: forward+inverse yields N (per-transform length,
  ! not Ntot) times the original input for each element.
  do i = 1, Ntot
     error = abs(hx(i)%x - N * hx_input(i)%x) + abs(hx(i)%y - N * hx_input(i)%y)
     if (error > error_max * N) then
        write(*,*) "FAILED! i=", i, " error=", error
        STOP 1
     end if
  end do

  deallocate(hx)
  deallocate(hx_input)

  write(*,*) "PASSED!"

end program hipfft_c2c_1d_batched_z
