!!!!!!!!!!!!!/
! dsygvdx example (partial generalized symmetric-definite eigensolver, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! sygvdx solves a selected subset of A*x = lambda*B*x (itype=eform_ax). erange=index
! with il=1, iu=N requests the full spectrum. f2003 style: device buffers are
! type(c_ptr). Self-verifying: A0*v_k = lambda_k * B0*v_k.
!!!!!!!!!!!!!!/
!
program dsygvdx
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, k

  integer(c_int), parameter :: N = 3, lda = 3, ldb = 3, ldz = 3

  real(c_double), target :: hA(3,3) = reshape((/2, -1, 0, -1, 2, -1, 0, -1, 2/), (/3, 3/))
  real(c_double), target :: hB(3,3) = reshape((/2, 0, 0, 0, 2, 0, 0, 0, 2/), (/3, 3/))
  real(c_double) :: hA0(3,3), hB0(3,3)
  real(c_double), target :: hW(3)
  real(c_double), target :: hZ(3,3)
  integer(c_int), target :: hNev, hInfo
  real(c_double) :: lhs(3), rhs(3)

  integer(c_size_t) :: sizeA = 9, sizeB = 9, sizeW = 3, sizeZ = 9

  type(c_ptr) :: dA, dB, dW, dZ, dInfo
  type(c_ptr) :: dNev
  type(c_ptr) :: handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dsygvdx' (Fortran 2003 interfaces) - "

  hA0 = hA
  hB0 = hB

  call hipCheck(hipMalloc(dA, sizeA * 8))
  call hipCheck(hipMalloc(dB, sizeB * 8))
  call hipCheck(hipMalloc(dW, sizeW * 8))
  call hipCheck(hipMalloc(dZ, sizeZ * 8))
  call hipCheck(hipMalloc(dNev, 4_c_size_t))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sizeA * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), sizeB * 8, hipMemcpyHostToDevice))

  call hipCheck(rocblas_create_handle(handle))

  call hipCheck(rocsolver_dsygvdx(handle, rocblas_eform_ax, rocblas_evect_original, &
       rocblas_erange_index, rocblas_fill_upper, N, dA, lda, dB, ldb, &
       0.0d0, 0.0d0, 1, N, dNev, dW, dZ, ldz, dInfo))

  call hipCheck(hipMemcpy(c_loc(hNev), dNev, 4_c_size_t, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, sizeW * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hZ(1,1)), dZ, sizeZ * 8, hipMemcpyDeviceToHost))

  if (hNev /= N) then
    write(*,*) "FAILED! nev = ", hNev, " expected ", N; call exit(1)
  end if

  do k = 1,N
    lhs = matmul(hA0, hZ(:,k))
    rhs = hW(k) * matmul(hB0, hZ(:,k))
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k; call exit(1)
        end if
    end do
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dB)); call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dZ)); call hipCheck(hipFree(dNev)); call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dsygvdx
