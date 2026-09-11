!!!!!!!!!!!!!/
! hipsolverDgeqrf example (double-precision QR factorization, Fortran 2003
! interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: geqrf overwrites the upper triangle of A with R. Because
! A = Q*R with Q orthogonal, A**T * A = R**T * R, so we recover R from the
! output, form R**T * R, and compare against A0**T * A0 computed on the host.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. The generic hipsolverDgeqrf resolves
! to the raw bind(c) interface, whose tau and devInfo arguments are passed by
! reference; we view the device allocations with c_f_pointer so a scalar/array
! target can be handed to those arguments.
!!!!!!!!!!!!!!/
!
program dgeqrf
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j, l ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  ! Input matrix (column-major)
  real(c_double), target :: hA(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3, 3/))
  real(c_double) :: hA0(3,3)      ! original kept for verification
  real(c_double) :: R(3,3), lhs, rhs

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_tau = 3

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA, dTau, dInfo, dWork
  real(c_double), pointer :: dTau_p(:)
  integer(c_int), pointer :: dInfo_p(:)
  integer(c_int) :: lwork

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverDgeqrf' (Fortran 2003 interfaces) - "

  hA0 = hA ! keep original for the A**T*A = R**T*R check

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dTau, size_tau * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dTau, dTau_p, (/int(size_tau)/))
  call c_f_pointer(dInfo, dInfo_p, (/1/))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Query workspace size and allocate it
  call hipsolverCheck(hipsolverDgeqrf_bufferSize(handle, M, N, dA, lda, lwork))
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  ! Compute the QR factorization
  call hipsolverCheck(hipsolverDgeqrf(handle, M, N, dA, lda, dTau_p(1), dWork, lwork, dInfo_p(1)))

  ! Copy the factorized matrix back to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))

  ! Extract R (upper triangle of the geqrf output)
  R = 0.0_c_double
  do j = 1,N
    do i = 1,j
      R(i,j) = hA(i,j)
    end do
  end do

  ! Verify A0**T * A0 = R**T * R
  do j = 1,N
    do i = 1,N
      lhs = 0.0_c_double
      rhs = 0.0_c_double
      do l = 1,M
        lhs = lhs + hA0(l,i) * hA0(l,j)
        rhs = rhs + R(l,i) * R(l,j)
      end do
      error = abs(lhs - rhs)
      if(error .gt. error_max) then
          write(*,*) "FAILED! Error bigger than max! Error = ", error, " (", i, ",", j, ")"
          call exit(1)
      end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dTau))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dgeqrf
