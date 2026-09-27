!!!!!!!!!!!!!/
! hipsolverZgesvdj example (double-complex Jacobi SVD, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Self-verifying: compute the singular value decomposition A = U*S*V**H with the
! Jacobi method and confirm the factors reconstruct the original matrix. Using
! the reconstruction avoids sign/phase ambiguity in the singular vectors.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. gesvdj takes a gesvdjInfo params handle,
! an explicit workspace sized by _bufferSize, and a device devInfo (viewed with
! c_f_pointer so it can be passed by reference).
!!!!!!!!!!!!!!/
!
program zgesvdj
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldu = 3
  integer(c_int), parameter :: ldv = 3
  integer(c_int), parameter :: econ = 0

  ! Nonsingular input matrix (column-major)
  complex(c_double_complex), target :: hA(3,3) = reshape((/ &
    (1.0d0,1.0d0), (4.0d0,0.0d0), (7.0d0,0.0d0), &
    (2.0d0,0.0d0), (5.0d0,1.0d0), (8.0d0,0.0d0), &
    (3.0d0,0.0d0), (6.0d0,0.0d0), (10.0d0,1.0d0)/), (/3, 3/))
  complex(c_double_complex) :: hA0(3,3)          ! original kept for verification
  real(c_double), target :: hS(3)                ! singular values (real)
  complex(c_double_complex), target :: hU(3,3)   ! left singular vectors
  complex(c_double_complex), target :: hV(3,3)   ! right singular vectors
  complex(c_double_complex) :: recon(3,3)

  integer(c_size_t) :: sz = 9
  integer(c_size_t) :: szv = 3

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: params = c_null_ptr
  type(c_ptr) :: dA, dS, dU, dV, dInfo, dWork
  integer(c_int), pointer :: dInfo_p(:)
  integer(c_int) :: lwork

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'hipsolverZgesvdj' (Fortran 2003 interfaces) - "

  hA0 = hA ! keep original for the reconstruction check

  call hipsolverCheck(hipsolverCreate(handle))
  call hipsolverCheck(hipsolverCreateGesvdjInfo(params))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, sz * 16))
  call hipCheck(hipMalloc(dS, szv * 8))
  call hipCheck(hipMalloc(dU, sz * 16))
  call hipCheck(hipMalloc(dV, sz * 16))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dInfo, dInfo_p, (/1/))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), sz * 16, hipMemcpyHostToDevice))

  ! Query workspace size and allocate it
  call hipsolverCheck(hipsolverZgesvdj_bufferSize(handle, HIPSOLVER_EIG_MODE_VECTOR, econ, &
       M, N, dA, lda, dS, dU, ldu, dV, ldv, lwork, params))
  call hipCheck(hipMalloc(dWork, int(lwork,c_size_t) * 16))

  ! Compute the singular value decomposition
  call hipsolverCheck(hipsolverZgesvdj(handle, HIPSOLVER_EIG_MODE_VECTOR, econ, &
       M, N, dA, lda, dS, dU, ldu, dV, ldv, dWork, lwork, dInfo_p(1), params))

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hS(1)), dS, szv * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hU(1,1)), dU, sz * 16, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hV(1,1)), dV, sz * 16, hipMemcpyDeviceToHost))

  ! Reconstruct A = U * diag(S) * V**H and compare with the original
  do j = 1,N
    do i = 1,M
      recon(i,j) = sum(hU(i,:) * hS(:) * conjg(hV(j,:)))
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
  call hipCheck(hipFree(dWork))
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dS))
  call hipCheck(hipFree(dU))
  call hipCheck(hipFree(dV))
  call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroyGesvdjInfo(params))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program zgesvdj
