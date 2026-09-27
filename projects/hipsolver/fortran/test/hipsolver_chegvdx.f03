!!!!!!!!!!!!!!
! chegvdx example (complex partial generalized Hermitian-definite eigensolver, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! hegvdx solves a selected subset of A*x = lambda*B*x (itype=1, range=all). f2003
! style: device buffers are type(c_ptr); nev is a host integer, devInfo is
! device-backed. Self-verifying: A0*v_k = lambda_k * B0*v_k.
!!!!!!!!!!!!!!
!
program hipsolver_chegvdx
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer :: i, k

  integer(c_int), parameter :: N = 3, lda = 3, ldb = 3

  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
       (2.,0.),(-1.,0.),(0.,0.), (-1.,0.),(2.,0.),(-1.,0.), (0.,0.),(-1.,0.),(2.,0.) /), (/3,3/))
  complex(c_float_complex), target :: hB(3,3) = reshape((/ &
       (2.,0.),(0.,0.),(0.,0.), (0.,0.),(2.,0.),(0.,0.), (0.,0.),(0.,0.),(2.,0.) /), (/3,3/))
  complex(c_float_complex) :: hA0(3,3), hB0(3,3)
  real(c_float), target :: hW(3)
  integer(c_int) :: hNev, nevBuf
  complex(c_float_complex) :: lhs(3), rhs(3)

  integer(c_size_t) :: sizeA = 9, sizeB = 9, sizeW = 3

  type(c_ptr) :: dA, dB, dW
  integer(c_int), pointer :: dInfo(:)
  type(c_ptr) :: dWork, handle = c_null_ptr
  integer(c_int) :: lwork

  real(c_float) :: error
  real(c_float), parameter :: error_max = 1.0e-5

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_chegvdx' (Fortran 2003 interfaces) - "

  hA0 = hA
  hB0 = hB

  call hipCheck(hipMalloc(dA, sizeA * 8))
  call hipCheck(hipMalloc(dB, sizeB * 8))
  call hipCheck(hipMalloc(dW, sizeW * 4))
  call hipCheck(hipMalloc(dInfo, 1))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sizeA * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), sizeB * 8, hipMemcpyHostToDevice))

  call hipsolverCheck(hipsolverCreate(handle))

  call hipsolverCheck(hipsolverChegvdx_bufferSize(handle, HIPSOLVER_EIG_TYPE_1, &
       HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_EIG_RANGE_ALL, HIPSOLVER_FILL_MODE_UPPER, N, &
       dA, lda, dB, ldb, 0.0, 0.0, 1, N, nevBuf, dW, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 8, 1_c_size_t)))

  call hipsolverCheck(hipsolverChegvdx(handle, HIPSOLVER_EIG_TYPE_1, HIPSOLVER_EIG_MODE_VECTOR, &
       HIPSOLVER_EIG_RANGE_ALL, HIPSOLVER_FILL_MODE_UPPER, N, dA, lda, dB, ldb, &
       0.0, 0.0, 1, N, hNev, dW, dWork, lwork, dInfo(1)))

  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, sizeA * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hW(1)), dW, sizeW * 4, hipMemcpyDeviceToHost))

  if (hNev /= N) then
    write(*,*) "FAILED! nev = ", hNev, " expected ", N; call exit(1)
  end if

  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hW(k) * matmul(hB0, hA(:,k))
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k; call exit(1)
        end if
    end do
  end do

  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dB)); call hipCheck(hipFree(dW))
  call hipCheck(hipFree(dInfo)); call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsolver_chegvdx
