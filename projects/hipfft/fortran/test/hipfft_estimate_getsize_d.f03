program hipfft_estimate_getsize_d
  use iso_c_binding
  use hip
  use hipfft

  implicit none

  integer(c_int), parameter :: N1d   = 32
  integer(c_int), parameter :: Nout  = N1d / 2 + 1    ! D2Z output: N/2+1 complex elements
  integer(c_int), parameter :: Nx2   = 8,  Ny2 = 4
  integer(c_int), parameter :: Nx3   = 4,  Ny3 = 4,  Nz3 = 2
  integer(c_int), parameter :: one_i = 1

  integer(c_size_t), parameter :: Nbytes_r = int(N1d, c_size_t) * 8
  integer(c_size_t), parameter :: Nbytes_c = int(Nout, c_size_t) * 16

  integer(c_size_t) :: workEst1d, workGS1d
  integer(c_size_t) :: workEst2d, workGS2d
  integer(c_size_t) :: workEst3d, workGS3d
  integer(c_size_t) :: workTmp

  type(c_ptr) :: plan1d = c_null_ptr
  type(c_ptr) :: plan2d = c_null_ptr
  type(c_ptr) :: plan3d = c_null_ptr
  type(c_ptr) :: dx_r   = c_null_ptr
  type(c_ptr) :: dx_c   = c_null_ptr

  real(c_double), allocatable, target, dimension(:)            :: hrx
  complex(c_double_complex), allocatable, target, dimension(:) :: hcx

  double precision, parameter :: tol = 1.0d-8
  integer :: i

  write(*,"(a)",advance="no") &
    "-- Running test 'hipFFT Estimate/GetSize D2Z (d)' (Fortran 2003 interfaces) - "

  ! --- 1D D2Z ---
  call hipfftCheck(hipfftEstimate1d(N1d, HIPFFT_D2Z, one_i, workEst1d))

  call hipfftCheck(hipfftCreate(plan1d))
  call hipfftCheck(hipfftMakePlan1d(plan1d, N1d, HIPFFT_D2Z, one_i, workGS1d))
  ! GetSize1d and GetSize provide alternative query paths on the same configured plan.
  call hipfftCheck(hipfftGetSize1d(plan1d, N1d, HIPFFT_D2Z, one_i, workTmp))
  call hipfftCheck(hipfftGetSize(plan1d, workTmp))

  ! hipfftEstimate* is a pre-plan upper bound; hipfftGetSize* is the accurate post-config value.
  if (workEst1d < workGS1d) then
    write(*,*) "FAILED! Estimate1d returned less than GetSize1d"
    STOP 1
  end if

  ! --- 2D Z2Z ---
  call hipfftCheck(hipfftEstimate2d(Nx2, Ny2, HIPFFT_Z2Z, workEst2d))

  call hipfftCheck(hipfftCreate(plan2d))
  call hipfftCheck(hipfftMakePlan2d(plan2d, Nx2, Ny2, HIPFFT_Z2Z, workGS2d))
  call hipfftCheck(hipfftGetSize2d(plan2d, Nx2, Ny2, HIPFFT_Z2Z, workTmp))

  if (workEst2d < workGS2d) then
    write(*,*) "FAILED! Estimate2d returned less than GetSize2d"
    STOP 1
  end if

  ! --- 3D Z2Z ---
  call hipfftCheck(hipfftEstimate3d(Nx3, Ny3, Nz3, HIPFFT_Z2Z, workEst3d))

  call hipfftCheck(hipfftCreate(plan3d))
  call hipfftCheck(hipfftMakePlan3d(plan3d, Nx3, Ny3, Nz3, HIPFFT_Z2Z, workGS3d))
  call hipfftCheck(hipfftGetSize3d(plan3d, Nx3, Ny3, Nz3, HIPFFT_Z2Z, workTmp))

  if (workEst3d < workGS3d) then
    write(*,*) "FAILED! Estimate3d returned less than GetSize3d"
    STOP 1
  end if

  ! --- 1D D2Z transform: verify DC bin equals sum of input ---
  allocate(hrx(N1d), hcx(Nout))
  do i = 1, N1d
    hrx(i) = 1.0d0
  end do

  call hipCheck(hipMalloc(dx_r, Nbytes_r))
  call hipCheck(hipMalloc(dx_c, Nbytes_c))
  call hipCheck(hipMemcpy(dx_r, c_loc(hrx(1)), Nbytes_r, hipMemcpyHostToDevice))

  ! plan1d was configured by MakePlan1d (auto-alloc ON by default); ready to execute.
  call hipfftCheck(hipfftExecD2Z(plan1d, dx_r, dx_c))
  call hipCheck(hipDeviceSynchronize())

  call hipfftCheck(hipfftDestroy(plan1d))
  call hipfftCheck(hipfftDestroy(plan2d))
  call hipfftCheck(hipfftDestroy(plan3d))

  call hipCheck(hipMemcpy(c_loc(hcx(1)), dx_c, Nbytes_c, hipMemcpyDeviceToHost))
  call hipCheck(hipFree(dx_r))
  call hipCheck(hipFree(dx_c))

  ! DC bin = sum of real inputs = N1d (all ones).
  if (abs(hcx(1) - cmplx(dble(N1d), 0.0d0, kind=c_double_complex)) > tol * N1d) then
    write(*,*) "FAILED! DC bin error"
    STOP 1
  end if

  deallocate(hrx, hcx)

  write(*,*) "PASSED!"

end program hipfft_estimate_getsize_d
