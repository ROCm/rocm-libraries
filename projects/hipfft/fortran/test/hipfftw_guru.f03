program hipfftw_guru_test
  use iso_c_binding
  use hip
  use hipfftw

  implicit none

  integer(c_int), parameter :: N = 16, HOWMANY = 3
  integer(c_size_t), parameter :: Nbytes = N * HOWMANY * 16  ! sizeof(double complex) = 16
  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-10

  complex(c_double_complex), allocatable, target, dimension(:) :: hx, hresult
  type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
  type(c_ptr) :: plan = c_null_ptr
  type(fftw_iodim), target :: dims(1), howmany_dims(1)
  integer :: b, j, k
  double precision :: error, max_error
  complex(c_double_complex) :: w, expected

  write(*,"(a)",advance="no") "-- Running test 'hipfftw_guru' (Fortran 2003) - "

  ! 1D batched C2C: HOWMANY batches of length N, contiguous (stride=1, dist=N).
  ! Batch b (0-indexed): x[j] = exp(2*pi*i*(b+1)*j/N).
  ! Expected forward DFT: output bin b+1 of batch b = N, all others zero.
  allocate(hx(N*HOWMANY), hresult(N*HOWMANY))
  do b = 0, HOWMANY-1
    w = cmplx(0.0d0, 2.0d0 * pi * dble(b+1) / dble(N), kind=c_double_complex)
    do j = 0, N-1
      hx(b*N + j + 1) = exp(w * dble(j))
    end do
  end do

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMalloc(dy, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  dims(1) = fftw_iodim(N, 1, 1)         ! n=N, is=1, os=1
  howmany_dims(1) = fftw_iodim(HOWMANY, N, N)  ! n=HOWMANY, is=N, os=N
  ! The dims/howmany_dims dummies are declared scalar type(fftw_iodim); pass the
  ! first element of each array so the callee receives the base address of
  ! the contiguous struct array — looks like an element but acts as a pointer.
  plan = fftw_plan_guru_dft(1, dims(1), 1, howmany_dims(1), &
      dx, dy, FFTW_FORWARD, FFTW_ESTIMATE)
  call fftw_execute_dft(plan, dx, dy)
  call fftw_destroy_plan(plan)

  call hipCheck(hipMemcpy(c_loc(hresult(1)), dy, Nbytes, hipMemcpyDeviceToHost))

  ! Verify: for batch b, output bin b+1 should equal N; all others zero.
  max_error = 0.0d0
  do b = 0, HOWMANY-1
    do k = 0, N-1
      if (k == b+1) then
        expected = cmplx(dble(N), 0.0d0, kind=c_double_complex)
      else
        expected = cmplx(0.0d0, 0.0d0, kind=c_double_complex)
      end if
      error = abs(hresult(b*N + k + 1) - expected)
      max_error = max(max_error, error)
    end do
  end do

  if (max_error > tol * dble(N)) then
    write(*,*) "FAILED! max error = ", max_error
    call exit(1)
  end if

  call hipCheck(hipFree(dx))
  call hipCheck(hipFree(dy))
  deallocate(hx, hresult)

  write(*,*) "PASSED!"
end program hipfftw_guru_test
