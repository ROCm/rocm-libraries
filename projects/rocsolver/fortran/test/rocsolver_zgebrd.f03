!!!!!!!!!!!!!/
! zgebrd example (double-complex bidiagonal reduction, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! gebrd reduces A to (real) bidiagonal form; D holds the diagonal and E the
! superdiagonal. We check D and E against reference values and confirm the
! reduction preserves the Frobenius norm (sum(D**2)+sum(E**2) = ||A||_F**2),
! which holds because Q and P are unitary.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program zgebrd
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
  complex(c_double_complex), target :: hA(3,3) = reshape((/ &
    (1.0d0,1.0d0), (4.0d0,0.0d0), (7.0d0,0.0d0), &
    (2.0d0,0.0d0), (5.0d0,1.0d0), (8.0d0,0.0d0), &
    (3.0d0,0.0d0), (6.0d0,0.0d0), (10.0d0,1.0d0)/), (/3, 3/))
  real(c_double), target :: hD(3)                ! diagonal of B (real)
  real(c_double), target :: hE(2)                ! superdiagonal of B (real)
  complex(c_double_complex), target :: hTauq(3)  ! Householder scalars (Q)
  complex(c_double_complex), target :: hTaup(3)  ! Householder scalars (P)

  ! Reference bidiagonal entries (from rocSOLVER)
  real(c_double), parameter :: refD(3) = &
    (/-8.1853527718724504_c_double, -2.7123138593754832_c_double, 1.0369566243275448_c_double/)
  real(c_double), parameter :: refE(2) = &
    (/15.2163502007819158_c_double, -0.1753882968361314_c_double/)
  real(c_double), parameter :: normA2 = 307.0_c_double

  integer(c_size_t) :: sz = 9

  type(c_ptr) :: dA, dD, dE, dTauq, dTaup
  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_zgebrd' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, sz * 16))
  call hipCheck(hipMalloc(dD, 3_c_size_t * 8))
  call hipCheck(hipMalloc(dE, 2_c_size_t * 8))
  call hipCheck(hipMalloc(dTauq, 3_c_size_t * 16))
  call hipCheck(hipMalloc(dTaup, 3_c_size_t * 16))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sz * 16, hipMemcpyHostToDevice))

  ! Reduce A to bidiagonal form
  call hipCheck(rocsolver_zgebrd(handle, M, N, dA, lda, dD, dE, dTauq, dTaup))

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD, 3_c_size_t * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hE(1)), dE, 2_c_size_t * 8, hipMemcpyDeviceToHost))

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
  if(error .gt. 1.0d-6) then
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

end program zgebrd
