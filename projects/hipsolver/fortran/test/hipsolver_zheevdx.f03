!!!!!!!!!!!!!!
! zheevdx example (complex partial Hermitian eigensolver, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! heevdx computes a selected subset of eigenvalues/eigenvectors (range=all -> full
! spectrum). f2003 style: device buffers are type(c_ptr); nev is a device-backed
! integer. Self-verifying: A0*v_k = lambda_k*v_k.
!!!!!!!!!!!!!!
!
program hipsolver_zheevdx
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer :: i, k

  integer(c_int), parameter :: N = 3, lda = 3

  complex(c_double_complex), target :: hA(3,3) = reshape((/ &
       (2.,0.),(-1.,0.),(0.,0.), (-1.,0.),(2.,0.),(-1.,0.), (0.,0.),(-1.,0.),(2.,0.) /), (/3,3/))
  complex(c_double_complex) :: hA0(3,3)
  real(c_double), target :: hW(3)
  integer(c_int), target :: hNev
  complex(c_double_complex) :: lhs(3), rhs(3)

  integer(c_size_t) :: sizeA = 9, sizeW = 3
  integer(c_int) :: nevBuf

  type(c_ptr) :: dA, dW
  integer(c_int), pointer :: dNev(:), dInfo(:)
  type(c_ptr) :: dWork, handle = c_null_ptr
  integer(c_int) :: lwork

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0e-5

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_zheevdx' (Fortran 2003 interfaces) - "

  hA0 = hA

  call hipCheck(hipMalloc(dA, sizeA * 16))
  call hipCheck(hipMalloc(dW, sizeW * 8))
  call hipCheck(hipMalloc(dNev, 1))
  call hipCheck(hipMalloc(dInfo, 1))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sizeA * 16, hipMemcpyHostToDevice))

  call hipsolverCheck(hipsolverCreate(handle))

  call hipsolverCheck(hipsolverZheevdx_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_EIG_RANGE_ALL, HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, &
       0.0d0, 0.0d0, 1, N, nevBuf, dW, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 16, 1_c_size_t)))

  call hipsolverCheck(hipsolverZheevdx(handle, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_EIG_RANGE_ALL, HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, &
       0.0d0, 0.0d0, 1, N, dNev(1), dW, dWork, lwork, dInfo(1)))

  call hipCheck(hipMemcpy(c_loc(hNev), c_loc(dNev(1)), 4_c_size_t, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, sizeA * 16, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, sizeW * 8, hipMemcpyDeviceToHost))

  if (hNev /= N) then
    write(*,*) "FAILED! nev = ", hNev, " expected ", N; call exit(1)
  end if

  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hW(k) * hA(:,k)
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k; call exit(1)
        end if
    end do
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dNev)); call hipCheck(hipFree(dInfo)); call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsolver_zheevdx
