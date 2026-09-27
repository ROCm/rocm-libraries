!!!!!!!!!!!!!/
! cgebrd example (single-complex bidiagonal reduction, Fortran 2003 interfaces)
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
program cgebrd
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
  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (1.0,1.0), (4.0,0.0), (7.0,0.0), &
    (2.0,0.0), (5.0,1.0), (8.0,0.0), &
    (3.0,0.0), (6.0,0.0), (10.0,1.0)/), (/3, 3/))
  real(c_float), target :: hD(3)                ! diagonal of B (real)
  real(c_float), target :: hE(2)                ! superdiagonal of B (real)
  complex(c_float_complex), target :: hTauq(3)  ! Householder scalars (Q)
  complex(c_float_complex), target :: hTaup(3)  ! Householder scalars (P)

  ! Reference bidiagonal entries (from rocSOLVER)
  real(c_float), parameter :: refD(3) = &
    (/-8.1853523254_c_float, -2.7123162746_c_float, 1.0369565487_c_float/)
  real(c_float), parameter :: refE(2) = &
    (/15.2163524628_c_float, -0.1753883064_c_float/)
  real(c_float), parameter :: normA2 = 307.0_c_float

  integer(c_size_t) :: sz = 9

  type(c_ptr) :: dA, dD, dE, dTauq, dTaup
  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_cgebrd' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, sz * 8))
  call hipCheck(hipMalloc(dD, 3_c_size_t * 4))
  call hipCheck(hipMalloc(dE, 2_c_size_t * 4))
  call hipCheck(hipMalloc(dTauq, 3_c_size_t * 8))
  call hipCheck(hipMalloc(dTaup, 3_c_size_t * 8))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sz * 8, hipMemcpyHostToDevice))

  ! Reduce A to bidiagonal form
  call hipCheck(rocsolver_cgebrd(handle, M, N, dA, lda, dD, dE, dTauq, dTaup))

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

end program cgebrd
