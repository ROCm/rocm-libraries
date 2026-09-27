!!!!!!!!!!!!!/
! sormqr example (single-precision multiplication by Q from a QR factorization,
! Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: factorize A with geqrf to obtain Q (as Householder vectors),
! then apply Q and Q**T successively to a matrix C. Since Q**T * Q = I, the
! result must recover the original C, so no reference matrix is needed.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program sormqr
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 2
  integer(c_int), parameter :: K = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldc = 3

  ! Matrix to factorize (column-major) and a separate C to multiply
  real(c_float), target :: hA(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3, 3/))
  real(c_float), target :: hC(3,2) = reshape((/1, 2, 3, 4, 5, 6/), (/3, 2/))
  real(c_float) :: hC0(3,2)   ! original C kept for verification
  real(c_float), target :: hIpiv(3)

  integer(c_size_t) :: sza = 9
  integer(c_size_t) :: szc = 6
  integer(c_size_t) :: szp = 3

  type(c_ptr) :: dA, dC, dIpiv
  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_sormqr' (Fortran 2003 interfaces) - "

  hC0 = hC ! keep original C for the round-trip check

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, sza * 4))
  call hipCheck(hipMalloc(dC, szc * 4))
  call hipCheck(hipMalloc(dIpiv, szp * 4))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sza * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), szc * 4, hipMemcpyHostToDevice))

  ! Factorize to get Q, then apply C <- Q*C followed by C <- Q**T*C
  call hipCheck(rocsolver_sgeqrf(handle, M, K, dA, lda, dIpiv))
  call hipCheck(rocsolver_sormqr(handle, rocblas_side_left, rocblas_operation_none, &
       M, N, K, dA, lda, dIpiv, dC, ldc))
  call hipCheck(rocsolver_sormqr(handle, rocblas_side_left, rocblas_operation_transpose, &
       M, N, K, dA, lda, dIpiv, dC, ldc))

  ! Copy result back to host
  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, szc * 4, hipMemcpyDeviceToHost))

  ! Verify Q**T * Q * C = C
  do j = 1,N
    do i = 1,M
      error = abs(hC(i,j) - hC0(i,j))
      if(error .gt. error_max) then
          write(*,*) "FAILED! Round trip mismatch! Error = ", error, " (", i, ",", j, ")"
          call exit(1)
      end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dC))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program sormqr
