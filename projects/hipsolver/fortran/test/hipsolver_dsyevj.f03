!!!!!!!!!!!!!/
! hipsolverDsyevj example (double-precision Jacobi symmetric eigensolver,
! Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: with jobz=vector, A is overwritten with the eigenvectors (as
! columns) and W holds the eigenvalues. We confirm A0*v_k = lambda_k*v_k.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. syevj takes a syevjInfo params handle,
! an explicit workspace sized by _bufferSize, and a device devInfo (viewed with
! c_f_pointer so it can be passed by reference).
!!!!!!!!!!!!!!/
!
program dsyevj
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, k ! indices for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  ! Symmetric input (column-major); eigenvalues are 2-sqrt(2), 2, 2+sqrt(2)
  real(c_double), target :: hA(3,3) = reshape((/2, -1, 0, -1, 2, -1, 0, -1, 2/), (/3, 3/))
  real(c_double) :: hA0(3,3)   ! original kept for verification
  real(c_double), target :: hW(3)      ! eigenvalues
  real(c_double) :: lhs(3), rhs(3)

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_W = size(hW)

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: params = c_null_ptr
  type(c_ptr) :: dA, dW, dInfo, dWork
  integer(c_int), pointer :: dInfo_p(:)
  integer(c_int) :: lwork

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverDsyevj' (Fortran 2003 interfaces) - "

  hA0 = hA ! keep original for the A*v = lambda*v check

  call hipsolverCheck(hipsolverCreate(handle))
  call hipsolverCheck(hipsolverCreateSyevjInfo(params))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dW, size_W * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dInfo, dInfo_p, (/1/))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Query workspace size and allocate it
  call hipsolverCheck(hipsolverDsyevj_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dW, lwork, params))
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 8))

  ! Compute eigenvalues and eigenvectors (A overwritten with eigenvectors)
  call hipsolverCheck(hipsolverDsyevj(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dW, dWork, lwork, dInfo_p(1), params))

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, size_W * 8, hipMemcpyDeviceToHost))

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
  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroySyevjInfo(params))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dsyevj
