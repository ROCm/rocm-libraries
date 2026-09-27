! =============================================================================
! GPU hipfftw test for the *_many interfaces
! =============================================================================
!
! This program validates hipfort's fftw_plan_many_dft, fftw_plan_many_dft_r2c,
! and fftw_plan_many_dft_c2r wrappers using GPU device memory.
!
! KEY CONVENTION: These are the generated bindings to the FFTW C API, which
! takes dimensions in C order (last index fastest, row-major) and does NOT
! reverse them. For a Fortran array z(NX, NY) with NX fastest, pass the
! reversed dimension list n = [NY, NX] (and likewise for the embed arrays).
! =============================================================================
program hipfftw_many_test
  use iso_c_binding
  use hip
  use hipfftw
  implicit none

  double precision, parameter :: tol = 1.0d-12
  double precision, parameter :: pi = 4.0d0 * atan(1.0d0)
  integer :: nfail, seed_size
  integer, allocatable :: seed(:)

  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  seed = 42
  call random_seed(put=seed)
  deallocate(seed)

  nfail = 0
  write(*,'(a)') "=== hipfftw many GPU Tests ==="

  call test_1d_c2c_interleaved(nfail)
  call test_2d_c2c_padded_embed(nfail)
  call test_1d_c2r_roundtrip(nfail)
  call test_many_vs_individual_c2c(nfail)

  write(*,*)
  if (nfail > 0) then
    write(*,'(i0,a)') nfail, " test(s) FAILED"
    call exit(1)
  else
    write(*,*) "PASSED!"
  end if

contains

  subroutine report(max_err, tolerance, nfail)
    double precision, intent(in) :: max_err, tolerance
    integer, intent(inout) :: nfail
    if (max_err > tolerance) then
      write(*,'(a,es12.4,a)') "FAILED (max_err=", max_err, ")"
      nfail = nfail + 1
    else
      write(*,'(a,es12.4,a)') "PASSED (max_err=", max_err, ")"
    end if
  end subroutine

  ! ===========================================================================
  ! 1D C2C, interleaved batches (batch-major layout)
  ! N=8, howmany=3, istride=howmany=3, idist=1
  ! Data layout: in(j*3+b+1) = signal(j) for transform b, element j
  ! Signal: x[j] = exp(2*pi*i*j/N) + 2*exp(2*pi*i*3*j/N)
  ! Expected DFT: X[1]=N, X[3]=2N, rest 0
  ! ===========================================================================
  subroutine test_1d_c2c_interleaved(nfail)
    integer, intent(inout) :: nfail
    integer(c_int), parameter :: N = 8, howmany = 3
    integer(c_size_t), parameter :: total_bytes = N * howmany * 16
    complex(c_double_complex), allocatable, target :: hx(:), hresult(:)
    type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
    type(c_ptr) :: plan = c_null_ptr
    integer(c_int), target :: n_arr(1), ie(1), oe(1)
    integer :: j, b
    double precision :: max_err, err
    complex(c_double_complex) :: w, expected

    write(*,'(a)',advance="no") "  1D C2C interleaved batches:    "
    allocate(hx(N*howmany), hresult(N*howmany))
    hx = cmplx(0d0, 0d0, kind=c_double_complex)

    w = cmplx(0d0, 2d0*pi/dble(N), kind=c_double_complex)
    do b = 0, howmany-1
      do j = 0, N-1
        hx(j*howmany + b + 1) = exp(w*dble(j)) + 2d0*exp(3d0*w*dble(j))
      end do
    end do

    call hipCheck(hipMalloc(dx, total_bytes))
    call hipCheck(hipMalloc(dy, total_bytes))
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), total_bytes, hipMemcpyHostToDevice))

    n_arr = [N]; ie = [N]; oe = [N]
    plan = fftw_plan_many_dft(1, c_loc(n_arr), howmany, &
        dx, c_loc(ie), howmany, 1, dy, c_loc(oe), howmany, 1, FFTW_FORWARD, FFTW_ESTIMATE)
    call fftw_execute_dft(plan, dx, dy)
    call fftw_destroy_plan(plan)

    call hipCheck(hipMemcpy(c_loc(hresult(1)), dy, total_bytes, hipMemcpyDeviceToHost))

    max_err = 0d0
    do b = 0, howmany-1
      do j = 0, N-1
        if (j == 1) then
          expected = cmplx(dble(N), 0d0, kind=c_double_complex)
        else if (j == 3) then
          expected = cmplx(2d0*dble(N), 0d0, kind=c_double_complex)
        else
          expected = cmplx(0d0, 0d0, kind=c_double_complex)
        end if
        err = abs(hresult(j*howmany + b + 1) - expected)
        max_err = max(max_err, err)
      end do
    end do

    call report(max_err, tol, nfail)
    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    deallocate(hx, hresult)
  end subroutine

  ! ===========================================================================
  ! 2D C2C with padded embed (embed != n)
  ! NX=4 (fast in Fortran), NY=6 (slow), LDX=8 (padded), howmany=2
  ! Fortran array: z(LDX, NY, howmany), transform z(1:NX, 1:NY, :)
  !
  ! Generated bindings call the FFTW C API directly (no dimension reversal),
  ! so pass dimensions in C order: n=[NY,NX], inembed=[NY,LDX].
  !
  ! Signal: z(ix,iy) = exp(2*pi*i*(ix-1)/NX) * exp(2*pi*i*2*(iy-1)/NY)
  ! Expected 2D DFT (0-indexed): Z(kx=1,ky=2) = NX*NY, rest 0
  ! ===========================================================================
  subroutine test_2d_c2c_padded_embed(nfail)
    integer, intent(inout) :: nfail
    integer(c_int), parameter :: NX = 4, NY = 6, LDX = 8, howmany = 2
    integer(c_size_t), parameter :: total_bytes = LDX * NY * howmany * 16
    complex(c_double_complex), allocatable, target :: hx(:), hresult(:)
    type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
    type(c_ptr) :: plan = c_null_ptr
    integer(c_int), target :: n_arr(2), ie(2), oe(2)
    integer :: ix, iy, kx, ky, b
    double precision :: max_err, err
    complex(c_double_complex) :: wx, wy, expected

    write(*,'(a)',advance="no") "  2D C2C padded embed:           "
    allocate(hx(LDX*NY*howmany), hresult(LDX*NY*howmany))
    hx = cmplx(0d0, 0d0, kind=c_double_complex)

    wx = cmplx(0d0, 2d0*pi/dble(NX), kind=c_double_complex)
    wy = cmplx(0d0, 2d0*pi/dble(NY), kind=c_double_complex)
    do b = 0, howmany-1
      do iy = 0, NY-1
        do ix = 0, NX-1
          hx(b*LDX*NY + iy*LDX + ix + 1) = exp(wx*dble(ix)) * exp(2d0*wy*dble(iy))
        end do
      end do
    end do

    call hipCheck(hipMalloc(dx, total_bytes))
    call hipCheck(hipMalloc(dy, total_bytes))
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), total_bytes, hipMemcpyHostToDevice))

    ! C order (reversed from Fortran): fastest dimension last.
    n_arr = [NY, NX]
    ie = [NY, LDX]
    oe = [NY, LDX]
    plan = fftw_plan_many_dft(2, c_loc(n_arr), howmany, &
        dx, c_loc(ie), 1, LDX*NY, dy, c_loc(oe), 1, LDX*NY, FFTW_FORWARD, FFTW_ESTIMATE)
    call fftw_execute_dft(plan, dx, dy)
    call fftw_destroy_plan(plan)

    call hipCheck(hipMemcpy(c_loc(hresult(1)), dy, total_bytes, hipMemcpyDeviceToHost))

    max_err = 0d0
    do b = 0, howmany-1
      do ky = 0, NY-1
        do kx = 0, NX-1
          if (kx == 1 .and. ky == 2) then
            expected = cmplx(dble(NX*NY), 0d0, kind=c_double_complex)
          else
            expected = cmplx(0d0, 0d0, kind=c_double_complex)
          end if
          err = abs(hresult(b*LDX*NY + ky*LDX + kx + 1) - expected)
          max_err = max(max_err, err)
        end do
      end do
    end do

    call report(max_err, tol, nfail)
    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    deallocate(hx, hresult)
  end subroutine

  ! ===========================================================================
  ! 1D C2R round-trip: c2r(r2c(x)) = N * x
  ! Tests both fftw_plan_many_dft_r2c and fftw_plan_many_dft_c2r.
  ! Uses random real input (any real signal is valid for R2C).
  ! ===========================================================================
  subroutine test_1d_c2r_roundtrip(nfail)
    integer, intent(inout) :: nfail
    integer(c_int), parameter :: N = 16, Nc = N/2+1, howmany = 3
    integer(c_size_t), parameter :: rbytes = N * howmany * 8
    integer(c_size_t), parameter :: cbytes = Nc * howmany * 16
    real(c_double), allocatable, target :: hx(:), hresult(:)
    type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr, dz = c_null_ptr
    type(c_ptr) :: plan = c_null_ptr
    integer(c_int), target :: n_arr(1), ie_r(1), oe_c(1), ie_c(1), oe_r(1)
    integer :: j, b
    double precision :: max_err, err

    write(*,'(a)',advance="no") "  1D R2C/C2R round-trip:         "
    allocate(hx(N*howmany), hresult(N*howmany))

    call random_number(hx)

    call hipCheck(hipMalloc(dx, rbytes))
    call hipCheck(hipMalloc(dy, cbytes))
    call hipCheck(hipMalloc(dz, rbytes))
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), rbytes, hipMemcpyHostToDevice))

    n_arr = [N]; ie_r = [N]; oe_c = [Nc]; ie_c = [Nc]; oe_r = [N]

    plan = fftw_plan_many_dft_r2c(1, c_loc(n_arr), howmany, &
        dx, c_loc(ie_r), 1, N, dy, c_loc(oe_c), 1, Nc, FFTW_ESTIMATE)
    call fftw_execute_dft_r2c(plan, dx, dy)
    call fftw_destroy_plan(plan)

    plan = fftw_plan_many_dft_c2r(1, c_loc(n_arr), howmany, &
        dy, c_loc(ie_c), 1, Nc, dz, c_loc(oe_r), 1, N, FFTW_ESTIMATE)
    call fftw_execute_dft_c2r(plan, dy, dz)
    call fftw_destroy_plan(plan)

    call hipCheck(hipMemcpy(c_loc(hresult(1)), dz, rbytes, hipMemcpyDeviceToHost))

    max_err = 0d0
    do b = 0, howmany-1
      do j = 0, N-1
        err = abs(hresult(b*N + j + 1) - dble(N) * hx(b*N + j + 1))
        max_err = max(max_err, err)
      end do
    end do

    call report(max_err, tol, nfail)
    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    call hipCheck(hipFree(dz))
    deallocate(hx, hresult)
  end subroutine

  ! ===========================================================================
  ! 1D C2C: many_dft vs individual dft_1d per batch on GPU
  ! Uses random complex input (any complex signal is valid for C2C).
  ! ===========================================================================
  subroutine test_many_vs_individual_c2c(nfail)
    integer, intent(inout) :: nfail
    integer(c_int), parameter :: N = 16, howmany = 5
    integer(c_size_t), parameter :: total_bytes = N * howmany * 16
    integer(c_size_t), parameter :: slice_bytes = N * 16
    complex(c_double_complex), allocatable, target :: hx(:), hout_many(:), hout_ind(:)
    type(c_ptr) :: dx = c_null_ptr, dy = c_null_ptr
    type(c_ptr) :: d_sin = c_null_ptr, d_sout = c_null_ptr
    type(c_ptr) :: plan = c_null_ptr
    integer(c_int), target :: n_arr(1), ie(1), oe(1)
    integer :: j
    double precision :: max_err, err
    real(c_double) :: rr(N*howmany), ri(N*howmany)
    integer :: b

    write(*,'(a)',advance="no") "  1D C2C many vs individual:     "
    allocate(hx(N*howmany), hout_many(N*howmany), hout_ind(N*howmany))

    call random_number(rr)
    call random_number(ri)
    do j = 1, N*howmany
      hx(j) = cmplx(rr(j), ri(j), kind=c_double_complex)
    end do

    call hipCheck(hipMalloc(dx, total_bytes))
    call hipCheck(hipMalloc(dy, total_bytes))
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), total_bytes, hipMemcpyHostToDevice))

    n_arr = [N]; ie = [N]; oe = [N]
    plan = fftw_plan_many_dft(1, c_loc(n_arr), howmany, &
        dx, c_loc(ie), 1, N, dy, c_loc(oe), 1, N, FFTW_FORWARD, FFTW_ESTIMATE)
    call fftw_execute_dft(plan, dx, dy)
    call fftw_destroy_plan(plan)
    call hipCheck(hipMemcpy(c_loc(hout_many(1)), dy, total_bytes, hipMemcpyDeviceToHost))
    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))

    call hipCheck(hipMalloc(d_sin, slice_bytes))
    call hipCheck(hipMalloc(d_sout, slice_bytes))

    do b = 0, howmany-1
      call hipCheck(hipMemcpy(d_sin, c_loc(hx(b*N+1)), slice_bytes, hipMemcpyHostToDevice))
      plan = fftw_plan_dft_1d(N, d_sin, d_sout, FFTW_FORWARD, FFTW_ESTIMATE)
      call fftw_execute_dft(plan, d_sin, d_sout)
      call fftw_destroy_plan(plan)
      call hipCheck(hipMemcpy(c_loc(hout_ind(b*N+1)), d_sout, slice_bytes, hipMemcpyDeviceToHost))
    end do
    call hipCheck(hipFree(d_sin))
    call hipCheck(hipFree(d_sout))

    max_err = 0d0
    do j = 1, N*howmany
      err = abs(hout_many(j) - hout_ind(j))
      max_err = max(max_err, err)
    end do

    call report(max_err, tol, nfail)
    deallocate(hx, hout_many, hout_ind)
  end subroutine

end program hipfftw_many_test
