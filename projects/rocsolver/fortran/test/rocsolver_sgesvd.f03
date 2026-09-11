!!!!!!!!!!!!!/
! sgesvd example (single-precision singular value decomposition, Fortran 2003 interfaces)
! see: https:!www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
!
! Self-verifying: compute A = U * S * V (rocSOLVER returns V as V**T), then
! confirm the reconstruction matches the original A.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. gesvd writes `info` to device memory,
! so dInfo is a device allocation passed directly (already a c_ptr).
! For real matrices S and E are real; A, U, V are real.
!!!!!!!!!!!!!!/
!
program sgesvd
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldu = 3
  integer(c_int), parameter :: ldv = 3

  ! Input matrix (column-major) and a copy kept for verification
  real(c_float), target :: hA(3,3) = reshape((/4, 6, -4, -51, 167, 24, 4, -68, -41/), (/3, 3/))
  real(c_float) :: hA0(3,3)
  real(c_float), target :: hS(3)      ! singular values
  real(c_float), target :: hU(3,3)    ! left singular vectors
  real(c_float), target :: hV(3,3)    ! right singular vectors (stored as V**T)
  real(c_float), target :: hE(2)      ! superdiagonal workspace (min(M,N)-1)
  real(c_float) :: recon(3,3), Sd(3,3)

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_S = size(hS)
  integer(c_size_t) :: size_U = size(hU)
  integer(c_size_t) :: size_V = size(hV)
  integer(c_size_t) :: size_E = size(hE)

  type(c_ptr) :: dA
  type(c_ptr) :: dS
  type(c_ptr) :: dU
  type(c_ptr) :: dV
  type(c_ptr) :: dE
  type(c_ptr) :: dInfo

  type(c_ptr) :: handle ! rocblas_handle

  real :: error
  ! single precision: relative tolerance
  real, parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_sgesvd' (Fortran 2003 interfaces) - "

  hA0 = hA ! keep the original for the reconstruction check

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 4))
  call hipCheck(hipMalloc(dS, size_S * 4))
  call hipCheck(hipMalloc(dU, size_U * 4))
  call hipCheck(hipMalloc(dV, size_V * 4))
  call hipCheck(hipMalloc(dE, size_E * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 4, hipMemcpyHostToDevice))

  ! Compute the full SVD on the device
  call hipCheck(rocsolver_sgesvd(handle, rocblas_svect_all, rocblas_svect_all, M, N, dA, lda, &
                                 dS, dU, ldu, dV, ldv, dE, rocblas_outofplace, dInfo))

  ! Copy factors back to host
  call hipCheck(hipMemcpy(c_loc(hS(1)), dS, size_S * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hU(1,1)), dU, size_U * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hV(1,1)), dV, size_V * 4, hipMemcpyDeviceToHost))

  ! Reconstruct A = U * diag(S) * V (V is already V**T from rocSOLVER)
  Sd = 0.0
  do i = 1,N
    Sd(i,i) = hS(i)
  end do
  recon = matmul(hU, matmul(Sd, hV))

  ! Verify reconstruction matches the original
  do j = 1,N
    do i = 1,M
        error = abs(recon(i,j) - hA0(i,j)) / max(abs(hA0(i,j)), 1.0)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " at (", i, ",", j, ")"
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dS))
  call hipCheck(hipFree(dU))
  call hipCheck(hipFree(dV))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program sgesvd
