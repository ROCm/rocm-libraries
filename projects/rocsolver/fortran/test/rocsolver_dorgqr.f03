!!!!!!!!!!!!!/
! dorgqr example (double-precision generation of Q from a QR factorization,
! Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: factorize A with geqrf, generate the orthogonal factor Q with
! orgqr, and confirm Q**T * Q = I. Orthogonality is sign-convention independent,
! so no reference matrix is needed.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program dorgqr
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: K = 3
  integer(c_int), parameter :: lda = 3

  ! Input matrix (column-major)
  real(c_double), target :: hA(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3, 3/))
  real(c_double), target :: hIpiv(3)   ! Householder scalars
  real(c_double) :: gram               ! entry of Q**T * Q

  integer(c_size_t) :: sz = 9
  integer(c_size_t) :: szp = 3

  type(c_ptr) :: dA, dIpiv
  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dorgqr' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, sz * 8))
  call hipCheck(hipMalloc(dIpiv, szp * 8))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sz * 8, hipMemcpyHostToDevice))

  ! Factorize A = Q*R, then form the explicit Q in place
  call hipCheck(rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv))
  call hipCheck(rocsolver_dorgqr(handle, M, N, K, dA, lda, dIpiv))

  ! Copy Q back to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, sz * 8, hipMemcpyDeviceToHost))

  ! Verify Q**T * Q = I
  do j = 1,N
    do i = 1,N
      gram = sum(hA(:,i) * hA(:,j))
      if(i .eq. j) then
        error = abs(gram - 1.0_c_double)
      else
        error = abs(gram)
      end if
      if(error .gt. error_max) then
          write(*,*) "FAILED! Q not orthogonal! Error = ", error, " (", i, ",", j, ")"
          call exit(1)
      end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dorgqr
