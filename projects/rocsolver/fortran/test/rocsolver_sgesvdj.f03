!!!!!!!!!!!!!/
! sgesvdj example (single-precision Jacobi SVD, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: compute the singular value decomposition A = U*S*V**T with the
! Jacobi method and confirm the factors reconstruct the original matrix. Using
! the reconstruction avoids sign/order ambiguity in the singular vectors.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. gesvdj writes residual, n_sweeps and
! info to device memory, so those are device allocations passed directly.
!!!!!!!!!!!!!!/
!
program sgesvdj
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
  integer(c_int), parameter :: max_sweeps = 100
  real(c_float), parameter :: abstol = 0.0_c_float

  ! Nonsingular input matrix (column-major)
  real(c_float), target :: hA(3,3) = reshape((/1, 4, 7, 2, 5, 8, 3, 6, 10/), (/3, 3/))
  real(c_float) :: hA0(3,3)          ! original kept for verification
  real(c_float), target :: hS(3)     ! singular values
  real(c_float), target :: hU(3,3)   ! left singular vectors
  real(c_float), target :: hV(3,3)   ! right singular vectors
  real(c_float) :: recon(3,3)

  integer(c_size_t) :: sz = 9
  integer(c_size_t) :: szv = 3

  type(c_ptr) :: dA, dS, dU, dV
  type(c_ptr) :: dResidual, dNsweeps, dInfo

  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_sgesvdj' (Fortran 2003 interfaces) - "

  hA0 = hA ! keep original for the reconstruction check

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, sz * 4))
  call hipCheck(hipMalloc(dS, szv * 4))
  call hipCheck(hipMalloc(dU, sz * 4))
  call hipCheck(hipMalloc(dV, sz * 4))
  call hipCheck(hipMalloc(dResidual, 4_c_size_t))
  call hipCheck(hipMalloc(dNsweeps, 4_c_size_t))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sz * 4, hipMemcpyHostToDevice))

  ! Compute the singular value decomposition
  call hipCheck(rocsolver_sgesvdj(handle, rocblas_svect_singular, rocblas_svect_singular, &
       M, N, dA, lda, abstol, dResidual, max_sweeps, dNsweeps, dS, dU, ldu, dV, ldv, dInfo))

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hS(1)), dS, szv * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hU(1,1)), dU, sz * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hV(1,1)), dV, sz * 4, hipMemcpyDeviceToHost))

  ! Reconstruct A = U * diag(S) * V**T and compare with the original
  do j = 1,N
    do i = 1,M
      recon(i,j) = sum(hU(i,:) * hS(:) * hV(:,j))
    end do
  end do

  do j = 1,N
    do i = 1,M
      error = abs(recon(i,j) - hA0(i,j))
      if(error .gt. error_max) then
          write(*,*) "FAILED! Error bigger than max! Error = ", error, " (", i, ",", j, ")"
          call exit(1)
      end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dS))
  call hipCheck(hipFree(dU))
  call hipCheck(hipFree(dV))
  call hipCheck(hipFree(dResidual))
  call hipCheck(hipFree(dNsweeps))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program sgesvdj
