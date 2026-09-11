!!!!!!!!!!!!!/
! zheevdx example (complex partial Hermitian eigensolver, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! heevdx computes a selected subset of eigenvalues/eigenvectors. erange=index with
! il=1, iu=N requests the full spectrum. f2003 style: device buffers are
! type(c_ptr). Self-verifying: A0*v_k = lambda_k*v_k.
!!!!!!!!!!!!!!/
!
program zheevdx
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, k

  integer(c_int), parameter :: N = 3, lda = 3, ldz = 3

  complex(c_double_complex), target :: hA(3,3) = reshape((/ &
       (2.,0.),(-1.,0.),(0.,0.), (-1.,0.),(2.,0.),(-1.,0.), (0.,0.),(-1.,0.),(2.,0.) /), (/3,3/))
  complex(c_double_complex) :: hA0(3,3)
  real(c_double), target :: hW(3)
  complex(c_double_complex), target :: hZ(3,3)
  integer(c_int), target :: hNev, hInfo
  complex(c_double_complex) :: lhs(3), rhs(3)

  integer(c_size_t) :: sizeA = 9, sizeW = 3, sizeZ = 9

  type(c_ptr) :: dA, dW, dZ, dNev, dInfo
  type(c_ptr) :: handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0e-5
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_zheevdx' (Fortran 2003 interfaces) - "

  hA0 = hA

  call hipCheck(hipMalloc(dA, sizeA * 16))
  call hipCheck(hipMalloc(dW, sizeW * 8))
  call hipCheck(hipMalloc(dZ, sizeZ * 16))
  call hipCheck(hipMalloc(dNev, 4_c_size_t))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sizeA * 16, hipMemcpyHostToDevice))

  call hipCheck(rocblas_create_handle(handle))

  call hipCheck(rocsolver_zheevdx(handle, rocblas_evect_original, rocblas_erange_index, &
       rocblas_fill_upper, N, dA, lda, 0.0d0, 0.0d0, 1, N, dNev, dW, dZ, ldz, dInfo))

  call hipCheck(hipMemcpy(c_loc(hNev), dNev, 4_c_size_t, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, sizeW * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hZ(1,1)), dZ, sizeZ * 16, hipMemcpyDeviceToHost))

  if (hNev /= N) then
    write(*,*) "FAILED! nev = ", hNev, " expected ", N; call exit(1)
  end if

  do k = 1,N
    lhs = matmul(hA0, hZ(:,k))
    rhs = hW(k) * hZ(:,k)
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k; call exit(1)
        end if
    end do
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dW)); call hipCheck(hipFree(dZ))
  call hipCheck(hipFree(dNev)); call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program zheevdx
