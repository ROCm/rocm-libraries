!!!!!!!!!!!!!/
! ssyevj example (single-precision symmetric Jacobi eigensolver, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: with evect=original, A is overwritten with the eigenvectors
! (as columns) and W holds the eigenvalues. We confirm A0*v_k = lambda_k*v_k.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. syevj writes residual, n_sweeps and
! info to device memory, so those are device allocations passed directly.
!!!!!!!!!!!!!!/
!
program ssyevj
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, k ! indices for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: max_sweeps = 100
  real(c_float), parameter :: abstol = 0.0_c_float

  ! Symmetric input (column-major); eigenvalues are 2-sqrt(2), 2, 2+sqrt(2)
  real(c_float), target :: hA(3,3) = reshape((/2, -1, 0, -1, 2, -1, 0, -1, 2/), (/3, 3/))
  real(c_float) :: hA0(3,3)   ! original kept for verification
  real(c_float), target :: hW(3)      ! eigenvalues
  real(c_float) :: lhs(3), rhs(3)

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_W = size(hW)

  type(c_ptr) :: dA
  type(c_ptr) :: dW
  type(c_ptr) :: dResidual
  type(c_ptr) :: dNsweeps
  type(c_ptr) :: dInfo

  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_ssyevj' (Fortran 2003 interfaces) - "

  hA0 = hA ! keep original for the A*v = lambda*v check

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 4))
  call hipCheck(hipMalloc(dW, size_W * 4))
  call hipCheck(hipMalloc(dResidual, 4_c_size_t))
  call hipCheck(hipMalloc(dNsweeps, 4_c_size_t))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 4, hipMemcpyHostToDevice))

  ! Compute eigenvalues and eigenvectors (A overwritten with eigenvectors)
  call hipCheck(rocsolver_ssyevj(handle, rocblas_esort_ascending, rocblas_evect_original, &
       rocblas_fill_upper, N, dA, lda, abstol, dResidual, max_sweeps, dNsweeps, dW, dInfo))

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, size_W * 4, hipMemcpyDeviceToHost))

  ! Verify A0 * v_k = lambda_k * v_k for each eigenpair
  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hW(k) * hA(:,k)
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dResidual))
  call hipCheck(hipFree(dNsweeps))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program ssyevj
