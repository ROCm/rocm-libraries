!!!!!!!!!!!!!/
! hipsolverCpotrs example (single-complex Cholesky solve, Fortran 2003
! interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: pick a known solution x, form b = A*x for a Hermitian positive
! definite A, factorize with potrf, solve A*X = b with potrs, and confirm X
! recovers x.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. devInfo is passed by reference through a
! c_f_pointer view of its device allocation.
!!!!!!!!!!!!!!/
!
program cpotrs
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i ! index for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: nrhs = 1
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldb = 3

  ! Hermitian positive definite input (column-major)
  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (4.0,0.0), (1.0,1.0), (0.0,0.0), &
    (1.0,-1.0), (5.0,0.0), (0.0,-1.0), &
    (0.0,0.0), (0.0,1.0), (6.0,0.0)/), (/3, 3/))
  complex(c_float_complex), target :: hX(3,1) = reshape((/ &
    (1.0,1.0), (2.0,-1.0), (3.0,2.0)/), (/3, 1/))  ! known solution
  complex(c_float_complex), target :: hB(3,1)                 ! RHS = A*x

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_B = size(hB)

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA, dB, dInfo, dWork
  integer(c_int), pointer :: dInfo_p(:)
  integer(c_int) :: lwork_f, lwork_s, lwork

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverCpotrs' (Fortran 2003 interfaces) - "

  ! Build a consistent RHS so that A*x = b
  hB = matmul(hA, hX)

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dB, size_B * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dInfo, dInfo_p, (/1/))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), size_B * 8, hipMemcpyHostToDevice))

  ! Workspace big enough for both potrf and potrs
  call hipsolverCheck(hipsolverCpotrf_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, N, dA, lda, lwork_f))
  call hipsolverCheck(hipsolverCpotrs_bufferSize(handle, HIPSOLVER_FILL_MODE_LOWER, N, nrhs, dA, lda, dB, ldb, lwork_s))
  lwork = max(lwork_f, lwork_s)
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  ! Factorize A = L*L**H, then solve A*X = B in place
  call hipsolverCheck(hipsolverCpotrf(handle, HIPSOLVER_FILL_MODE_LOWER, N, dA, lda, dWork, lwork, dInfo_p(1)))
  call hipsolverCheck(hipsolverCpotrs(handle, HIPSOLVER_FILL_MODE_LOWER, N, nrhs, dA, lda, dB, ldb, dWork, lwork, dInfo_p(1)))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hB(1,1)), dB, size_B * 8, hipMemcpyDeviceToHost))

  ! Verify the recovered solution matches x
  do i = 1,N
    error = abs(hB(i,1) - hX(i,1)) / max(abs(hX(i,1)), 1.0_c_float)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " X(", i, ") = ", hB(i,1)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program cpotrs
