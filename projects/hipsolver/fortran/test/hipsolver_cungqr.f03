!!!!!!!!!!!!!/
! hipsolverCungqr example (single-complex generation of Q from a QR
! factorization, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: factorize A with geqrf, generate the unitary factor Q with
! ungqr, and confirm Q**H * Q = I. Unitarity is phase-convention independent,
! so no reference matrix is needed.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. tau and devInfo are passed by reference
! through c_f_pointer views of their device allocations.
!!!!!!!!!!!!!!/
!
program cungqr
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: K = 3
  integer(c_int), parameter :: lda = 3

  ! Input matrix (column-major)
  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (1.0,1.0), (4.0,0.0), (7.0,0.0), &
    (2.0,0.0), (5.0,1.0), (8.0,0.0), &
    (3.0,0.0), (6.0,0.0), (10.0,1.0)/), (/3, 3/))
  complex(c_float_complex) :: gram

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_tau = 3

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA, dTau, dInfo, dWork
  complex(c_float_complex), pointer :: dTau_p(:)
  integer(c_int), pointer :: dInfo_p(:)
  integer(c_int) :: lwork_qr, lwork_un, lwork

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverCungqr' (Fortran 2003 interfaces) - "

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dTau, size_tau * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dTau, dTau_p, (/int(size_tau)/))
  call c_f_pointer(dInfo, dInfo_p, (/1/))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Workspace big enough for both geqrf and ungqr
  call hipsolverCheck(hipsolverCgeqrf_bufferSize(handle, M, N, dA, lda, lwork_qr))
  call hipsolverCheck(hipsolverCungqr_bufferSize(handle, M, N, K, dA, lda, dTau_p(1), lwork_un))
  lwork = max(lwork_qr, lwork_un)
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  ! Factorize A = Q*R, then form the explicit Q in place
  call hipsolverCheck(hipsolverCgeqrf(handle, M, N, dA, lda, dTau_p(1), dWork, lwork, dInfo_p(1)))
  call hipsolverCheck(hipsolverCungqr(handle, M, N, K, dA, lda, dTau_p(1), dWork, lwork, dInfo_p(1)))

  ! Copy Q back to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))

  ! Verify Q**H * Q = I
  do j = 1,N
    do i = 1,N
      gram = sum(conjg(hA(:,i)) * hA(:,j))
      if(i .eq. j) then
        error = abs(gram - (1.0_c_float, 0.0_c_float))
      else
        error = abs(gram)
      end if
      if(error .gt. error_max) then
          write(*,*) "FAILED! Q not unitary! Error = ", error, " (", i, ",", j, ")"
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

end program cungqr
