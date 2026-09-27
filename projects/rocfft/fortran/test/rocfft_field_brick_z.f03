program rocfft_field_brick_z
  use iso_c_binding
  use hip
  use rocfft

  implicit none

  ! 2D transform: N0 x N1, decomposed over 2 devices along dim-1
  integer(c_size_t), parameter :: N0 = 8, N1 = 8
  integer(c_size_t), parameter :: Ntot = N0 * N1
  integer(c_size_t), parameter :: nbytes_elem = 16    ! complex(8) = 16 bytes
  integer(c_size_t), parameter :: dim_wb = 3          ! 2 transform dims + 1 batch
  integer(c_size_t), parameter :: two = 2, one = 1

  ! rocFFT handles
  type(c_ptr) :: description = c_null_ptr
  type(c_ptr) :: infield = c_null_ptr, outfield = c_null_ptr
  type(c_ptr) :: abrick = c_null_ptr
  type(c_ptr) :: gpu_plan = c_null_ptr
  type(c_ptr) :: planinfo = c_null_ptr

  ! Per-brick GPU buffers and work buffers (2 bricks / 2 devices)
  type(c_ptr) :: gpu_in(2), gpu_out(2), work_buf(2)
  integer(c_size_t) :: wbuf_size

  ! Brick geometry (computed per brick in the loop)
  integer(c_size_t) :: brick_len1, brick_lower1, brick_upper1
  integer(c_size_t), target :: blower(3), bupper(3), bstride(3)

  ! Plan lengths
  integer(c_size_t), allocatable, target :: lengths(:)

  ! Host staging buffer
  complex(kind=8), allocatable, target :: hbuf(:)

  ! Verification
  double precision, parameter :: tol = 1.0d-8
  double precision :: pi2, err
  integer(c_int) :: ndev
  integer :: idx, ix, iy, pos

  gpu_in(1) = c_null_ptr  ;  gpu_in(2) = c_null_ptr
  gpu_out(1) = c_null_ptr ;  gpu_out(2) = c_null_ptr
  work_buf(1) = c_null_ptr;  work_buf(2) = c_null_ptr

  write(*,"(a)",advance="no") &
    "-- Running test 'rocFFT field/brick 2D z' (Fortran 2003 interfaces) - "

  ! Mandatory guard: need at least 2 GPUs
  call hipCheck(hipGetDeviceCount(ndev))
  if (ndev < 2_c_int) then
    write(*,*) "SKIPPED (requires 2 GPUs)"
    stop
  end if

  ! The field/brick API is marked experimental in the rocFFT headers.
  call rocfftCheck(rocfft_setup())

  pi2 = 8.0d0 * atan(1.0d0)

  ! Plan description: null strides/offsets because the field defines the layout
  call rocfftCheck(rocfft_plan_description_create(description))
  call rocfftCheck(rocfft_plan_description_set_data_layout( &
    description, &
    rocfft_array_type_complex_interleaved, &
    rocfft_array_type_complex_interleaved, &
    c_null_ptr, c_null_ptr, 0_c_size_t, c_null_ptr, 0_c_size_t, &
    0_c_size_t, c_null_ptr, 0_c_size_t))

  ! Build input field: split N1 across 2 devices (C++ sample formula)
  call rocfftCheck(rocfft_field_create(infield))
  do idx = 0, 1
    brick_len1 = N1 / 2 + &
      merge(1_c_size_t, 0_c_size_t, int(idx,c_size_t) < mod(N1, 2_c_size_t))
    brick_lower1 = int(idx,c_size_t) * (N1/2) + &
      min(int(idx,c_size_t), mod(N1, 2_c_size_t))
    brick_upper1 = brick_lower1 + brick_len1

    blower  = [0_c_size_t,  brick_lower1, 0_c_size_t]
    bupper  = [N0,           brick_upper1, 1_c_size_t]
    bstride = [1_c_size_t,  N0,           brick_len1]

    ! rocfft_brick_create takes c_ptr,value for lower/upper/stride; no f2008
    ! rank-N wrapper exists, so c_loc is used in both dialect files.
    call rocfftCheck(rocfft_brick_create(abrick, &
      c_loc(blower(1)), c_loc(bupper(1)), c_loc(bstride(1)), &
      dim_wb, int(idx, c_int)))
    call rocfftCheck(rocfft_field_add_brick(infield, abrick))
    call rocfftCheck(rocfft_brick_destroy(abrick))
    abrick = c_null_ptr

    call hipCheck(hipSetDevice(int(idx, c_int)))
    call hipCheck(hipMalloc(gpu_in(idx+1), N0 * brick_len1 * nbytes_elem))

    ! Fill with known harmonic exp(2*pi*i*(1*ix/N0 + 2*iy/N1))
    allocate(hbuf(N0 * brick_len1))
    do iy = int(brick_lower1), int(brick_upper1) - 1
      do ix = 0, int(N0) - 1
        pos = ix + (iy - int(brick_lower1)) * int(N0) + 1
        hbuf(pos) = exp(cmplx(0.0d0, &
          pi2*(dble(ix)/dble(N0) + 2.0d0*dble(iy)/dble(N1)), kind=8))
      end do
    end do
    call hipCheck(hipMemcpy(gpu_in(idx+1), c_loc(hbuf(1)), &
      N0 * brick_len1 * nbytes_elem, hipMemcpyHostToDevice))
    deallocate(hbuf)
  end do

  call rocfftCheck(rocfft_plan_description_add_infield(description, infield))
  call rocfftCheck(rocfft_field_destroy(infield))
  infield = c_null_ptr

  ! Build output field (same decomposition as input)
  call rocfftCheck(rocfft_field_create(outfield))
  do idx = 0, 1
    brick_len1 = N1 / 2 + &
      merge(1_c_size_t, 0_c_size_t, int(idx,c_size_t) < mod(N1, 2_c_size_t))
    brick_lower1 = int(idx,c_size_t) * (N1/2) + &
      min(int(idx,c_size_t), mod(N1, 2_c_size_t))
    brick_upper1 = brick_lower1 + brick_len1

    blower  = [0_c_size_t,  brick_lower1, 0_c_size_t]
    bupper  = [N0,           brick_upper1, 1_c_size_t]
    bstride = [1_c_size_t,  N0,           brick_len1]

    call rocfftCheck(rocfft_brick_create(abrick, &
      c_loc(blower(1)), c_loc(bupper(1)), c_loc(bstride(1)), &
      dim_wb, int(idx, c_int)))
    call rocfftCheck(rocfft_field_add_brick(outfield, abrick))
    call rocfftCheck(rocfft_brick_destroy(abrick))
    abrick = c_null_ptr

    call hipCheck(hipSetDevice(int(idx, c_int)))
    call hipCheck(hipMalloc(gpu_out(idx+1), N0 * brick_len1 * nbytes_elem))
  end do

  call rocfftCheck(rocfft_plan_description_add_outfield(description, outfield))
  call rocfftCheck(rocfft_field_destroy(outfield))
  outfield = c_null_ptr

  ! Create not-in-place forward double plan on device 0
  call hipCheck(hipSetDevice(0_c_int))
  allocate(lengths(2))
  lengths(1) = N0
  lengths(2) = N1
  call rocfftCheck(rocfft_plan_create(gpu_plan, &
    rocfft_placement_notinplace, &
    rocfft_transform_type_complex_forward, &
    rocfft_precision_double, &
    two, c_loc(lengths(1)), one, description))

  ! Per-device work buffer allocation (mirrors C++ sample loop)
  do idx = 0, 1
    call hipCheck(hipSetDevice(int(idx, c_int)))
    wbuf_size = 0_c_size_t
    call rocfftCheck(rocfft_plan_get_work_buffer_size(gpu_plan, wbuf_size))
    if (wbuf_size > 0_c_size_t) then
      if (.not. c_associated(planinfo)) &
        call rocfftCheck(rocfft_execution_info_create(planinfo))
      call hipCheck(hipMalloc(work_buf(idx+1), wbuf_size))
      call rocfftCheck(rocfft_execution_info_set_work_buffer( &
        planinfo, work_buf(idx+1), wbuf_size))
    end if
  end do

  ! Execute: in_buffer / out_buffer are arrays of per-brick device pointers
  call hipCheck(hipSetDevice(0_c_int))
  call rocfftCheck(rocfft_execute(gpu_plan, gpu_in(1), gpu_out(1), planinfo))
  call hipCheck(hipDeviceSynchronize())

  ! Verify: forward FFT of exp(2pi*i*(1*ix/N0 + 2*iy/N1)) must give Ntot at
  ! bin (kx=1, ky=2) and zero at every other bin.
  do idx = 0, 1
    brick_len1 = N1 / 2 + &
      merge(1_c_size_t, 0_c_size_t, int(idx,c_size_t) < mod(N1, 2_c_size_t))
    brick_lower1 = int(idx,c_size_t) * (N1/2) + &
      min(int(idx,c_size_t), mod(N1, 2_c_size_t))
    brick_upper1 = brick_lower1 + brick_len1

    allocate(hbuf(N0 * brick_len1))
    call hipCheck(hipSetDevice(int(idx, c_int)))
    call hipCheck(hipMemcpy(c_loc(hbuf(1)), gpu_out(idx+1), &
      N0 * brick_len1 * nbytes_elem, hipMemcpyDeviceToHost))
    do iy = int(brick_lower1), int(brick_upper1) - 1
      do ix = 0, int(N0) - 1
        pos = ix + (iy - int(brick_lower1)) * int(N0) + 1
        if (ix == 1 .and. iy == 2) then
          err = abs(hbuf(pos) - cmplx(dble(Ntot), 0.0d0, kind=8))
        else
          err = abs(hbuf(pos))
        end if
        if (err > tol * dble(Ntot)) then
          write(*,*) "FAILED! bin=(", ix, ",", iy, &
                     ") got=", hbuf(pos), " err=", err
          call rocfftCheck(rocfft_cleanup())
          STOP 1
        end if
      end do
    end do
    deallocate(hbuf)
  end do

  ! Teardown in sample order: info, description, plan, cleanup, buffers
  if (c_associated(planinfo)) &
    call rocfftCheck(rocfft_execution_info_destroy(planinfo))
  call rocfftCheck(rocfft_plan_description_destroy(description))
  call rocfftCheck(rocfft_plan_destroy(gpu_plan))
  call rocfftCheck(rocfft_cleanup())

  do idx = 0, 1
    call hipCheck(hipSetDevice(int(idx, c_int)))
    if (c_associated(work_buf(idx+1))) &
      call hipCheck(hipFree(work_buf(idx+1)))
    call hipCheck(hipFree(gpu_in(idx+1)))
    call hipCheck(hipFree(gpu_out(idx+1)))
  end do

  deallocate(lengths)

  write(*,*) "PASSED!"

end program rocfft_field_brick_z
