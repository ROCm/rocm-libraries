!!!!!!!!!!!!!/
! dsygvd example (generalized symmetric-definite eigensolver, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! sygvd solves A*x = lambda*B*x (itype=eform_ax). f2003 style: device buffers are
! type(c_ptr). Self-verifying: A0*v_k = lambda_k * B0*v_k.
!!!!!!!!!!!!!!/
!
program dsygvd
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, k

  integer(c_int), parameter :: N = 3, lda = 3, ldb = 3

  real(c_double), target :: hA(3,3) = reshape((/2, -1, 0, -1, 2, -1, 0, -1, 2/), (/3, 3/))
  real(c_double), target :: hB(3,3) = reshape((/2, 0, 0, 0, 2, 0, 0, 0, 2/), (/3, 3/))
  real(c_double) :: hA0(3,3), hB0(3,3)
  real(c_double), target :: hD(3)
  real(c_double), target :: hE(3)
  real(c_double) :: lhs(3), rhs(3)
  integer(c_int), target :: hInfo

  integer(c_size_t) :: sizeA = 9, sizeB = 9, sizeD = 3

  type(c_ptr) :: dA, dB, dD, dE, dInfo
  type(c_ptr) :: handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsygvd' (Fortran 2003 interfaces) - "

  hA0 = hA
  hB0 = hB

  call hipCheck(hipMalloc(dA, sizeA * 8))
  call hipCheck(hipMalloc(dB, sizeB * 8))
  call hipCheck(hipMalloc(dD, sizeD * 8))
  call hipCheck(hipMalloc(dE, sizeD * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sizeA * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), sizeB * 8, hipMemcpyHostToDevice))

  call hipCheck(rocblas_create_handle(handle))

  call hipCheck(rocsolver_dsygvd(handle, rocblas_eform_ax, rocblas_evect_original, &
       rocblas_fill_upper, N, dA, lda, dB, ldb, dD, dE, dInfo))

  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, sizeA * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD, sizeD * 8, hipMemcpyDeviceToHost))

  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hD(k) * matmul(hB0, hA(:,k))
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k; call exit(1)
        end if
    end do
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dD)); call hipCheck(hipFree(dE)); call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dsygvd
