!!!!!!!!!!!!!/
! sgebrd example (single-precision bidiagonal reduction, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! gebrd reduces A to bidiagonal form B = Q**T * A * P. We check the diagonal D
! and superdiagonal E of B against reference values, and confirm the reduction
! preserves the Frobenius norm (sum(D**2)+sum(E**2) = ||A||_F**2), which holds
! because Q and P are orthogonal.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program sgebrd
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i ! index for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  ! Input matrix (column-major)
  real(c_float), target :: hA(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3, 3/))
  real(c_float), target :: hD(3)      ! diagonal of B
  real(c_float), target :: hE(2)      ! superdiagonal of B
  real(c_float), target :: hTauq(3)   ! Householder scalars (Q)
  real(c_float), target :: hTaup(3)   ! Householder scalars (P)

  ! Reference bidiagonal entries (from rocSOLVER)
  real(c_float), parameter :: refD(3) = &
    (/-8.1240386963_c_float, -1.7704836130_c_float, 0.2085723728_c_float/)
  real(c_float), parameter :: refE(2) = &
    (/15.3213052750_c_float, 0.2818795741_c_float/)
  real(c_float), parameter :: normA2 = 304.0_c_float

  integer(c_size_t) :: sz = 9

  type(c_ptr) :: dA, dD, dE, dTauq, dTaup
  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_sgebrd' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, sz * 4))
  call hipCheck(hipMalloc(dD, 3_c_size_t * 4))
  call hipCheck(hipMalloc(dE, 2_c_size_t * 4))
  call hipCheck(hipMalloc(dTauq, 3_c_size_t * 4))
  call hipCheck(hipMalloc(dTaup, 3_c_size_t * 4))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sz * 4, hipMemcpyHostToDevice))

  ! Reduce A to bidiagonal form
  call hipCheck(rocsolver_sgebrd(handle, M, N, dA, lda, dD, dE, dTauq, dTaup))

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD, 3_c_size_t * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hE(1)), dE, 2_c_size_t * 4, hipMemcpyDeviceToHost))

  ! Verify the bidiagonal entries against the reference
  do i = 1,3
    error = abs(hD(i) - refD(i))
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " D(", i, ") = ", hD(i)
        call exit(1)
    end if
  end do
  do i = 1,2
    error = abs(hE(i) - refE(i))
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " E(", i, ") = ", hE(i)
        call exit(1)
    end if
  end do

  ! Cross-check the norm-preserving invariant
  error = abs(sum(hD*hD) + sum(hE*hE) - normA2)
  if(error .gt. 1.0e-2) then
      write(*,*) "FAILED! Norm not preserved! Error = ", error
      call exit(1)
  end if

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dD))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dTauq))
  call hipCheck(hipFree(dTaup))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program sgebrd
