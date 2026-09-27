program hipfft_makeplanmany_z
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N      = 16
  integer(c_int), parameter :: Nbatch = 3
  integer(c_int), parameter :: Ntot   = N * Nbatch
  integer(c_size_t), parameter :: Nbytes = int(Ntot, c_size_t) * 16

  integer(c_int), parameter :: rank_i = 1
  integer(c_int), parameter :: one_i  = 1
  integer(c_int), parameter :: zero_i = 0

  double precision, parameter :: pi  = 4.0d0 * atan(1.0d0)
  double precision, parameter :: tol = 1.0d-8

  ! n/inembed/onembed passed by address to the raw C interface (c_loc)
  integer(c_int), target :: nlen(1)
  integer(c_int), target :: in_emb(1)
  integer(c_int), target :: out_emb(1)

  complex(c_double_complex), allocatable, target, dimension(:) :: hx
  type(c_ptr) :: dx      = c_null_ptr
  type(c_ptr) :: plan    = c_null_ptr
  type(c_ptr) :: workBuf = c_null_ptr

  integer(c_size_t)        :: workSize, workSize2
  double precision         :: max_error, error
  complex(c_double_complex) :: expected
  integer :: b, j, pos, kb

  write(*,"(a)",advance="no") &
    "-- Running test 'hipFFT MakePlanMany 1D batched (z)' (Fortran 2003 interfaces) - "

  nlen(1)    = N
  in_emb(1)  = N
  out_emb(1) = N

  allocate(hx(Ntot))

  ! Batch b holds exp(2*pi*i*kb*j/N) with kb=b+1; forward FFT places N in bin kb.
  do b = 0, Nbatch - 1
    kb = b + 1
    do j = 0, N - 1
      pos = b * N + j
      hx(pos + 1) = exp(cmplx(0.0d0, 2.0d0 * pi * dble(kb * j) / dble(N), &
                               kind=c_double_complex))
    end do
  end do

  call hipCheck(hipMalloc(dx, Nbytes))
  call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

  call hipfftCheck(hipfftCreate(plan))
  ! Disable auto-allocation so we can supply a manual work buffer.
  call hipfftCheck(hipfftSetAutoAllocation(plan, zero_i))
  call hipfftCheck(hipfftMakePlanMany(plan, rank_i, c_loc(nlen), &
       c_loc(in_emb), one_i, N, c_loc(out_emb), one_i, N, &
       HIPFFT_Z2Z, Nbatch, workSize))

  ! GetSizeMany on the same configured plan must agree with MakePlanMany.
  call hipfftCheck(hipfftGetSizeMany(plan, rank_i, c_loc(nlen), &
       c_loc(in_emb), one_i, N, c_loc(out_emb), one_i, N, &
       HIPFFT_Z2Z, Nbatch, workSize2))

  if (workSize2 /= workSize) then
    write(*,*) "FAILED! workSize mismatch: MakePlanMany=", workSize, &
               " GetSizeMany=", workSize2
    STOP 1
  end if

  ! A plan needing no scratch is valid; skip alloc/set in that case.
  if (workSize > 0_c_size_t) then
    call hipCheck(hipMalloc(workBuf, workSize))
    call hipfftCheck(hipfftSetWorkArea(plan, workBuf))
  end if

  call hipfftCheck(hipfftExecZ2Z(plan, dx, dx, HIPFFT_FORWARD))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan))
  call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx))
  if (workSize > 0_c_size_t) call hipCheck(hipFree(workBuf))

  max_error = 0.0d0
  do b = 0, Nbatch - 1
    kb = b + 1
    do j = 0, N - 1
      pos = b * N + j
      if (j == kb) then
        expected = cmplx(dble(N), 0.0d0, kind=c_double_complex)
      else
        expected = (0.0d0, 0.0d0)
      end if
      error = abs(hx(pos + 1) - expected)
      max_error = max(max_error, error)
    end do
  end do

  if (max_error > tol * N) then
    write(*,*) "FAILED! max error = ", max_error
    STOP 1
  end if

  deallocate(hx)

  write(*,*) "PASSED!"

end program hipfft_makeplanmany_z
