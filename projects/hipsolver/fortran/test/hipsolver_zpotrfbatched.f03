!!!!!!!!!!!!!!
! hipsolver cpotrfBatched example (batched Cholesky factorization, single complex, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! Factorizes a batch of Hermitian positive-definite matrices. The batched API
! takes A as an array of device pointers that itself lives in DEVICE memory: each
! matrix is allocated on the device, their device addresses are collected in a
! host array, and that array is copied to a device buffer whose address (by
! value) is passed as A. Checks info == 0 for every batch entry and the Cholesky
! diagonal (real for an HPD matrix).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!
!
program zpotrfbatched
  use iso_c_binding
  use hip
  use hipsolver
  implicit none
  integer :: b
  integer(c_int), parameter :: N = 3, lda = 3, batch = 2
  ! HPD A = diag(4, 9, 16) -> upper Cholesky diag = 2, 3, 4 (real)
  complex(c_double_complex), target :: hA(N,N) = reshape((/ &
      (4.0d0,0.0d0), (0.0d0,0.0d0), (0.0d0,0.0d0), &
      (0.0d0,0.0d0), (9.0d0,0.0d0), (0.0d0,0.0d0), &
      (0.0d0,0.0d0), (0.0d0,0.0d0),(16.0d0,0.0d0)/), (/N,N/))
  complex(c_double_complex), target :: hOut(N,N)
  integer(c_int), target :: hInfo(batch)
  type(c_ptr), target :: hostPtrs(batch)
  type(c_ptr) :: dA1, dA2, dPtrArray, dWork, dInfo
  type(c_ptr) :: handle
  integer(c_int), pointer :: dInfo_p(:)
  integer(c_int) :: lwork
  integer(c_size_t) :: ptrbytes
  write(*,"(a)",advance="no") "-- Running test 'hipsolver_zpotrfbatched' (Fortran 2003 interfaces) - "

  call hipsolverCheck(hipsolverCreate(handle))
  call hipCheck(hipMalloc(dA1, int(N*N,c_size_t) * 16))
  call hipCheck(hipMalloc(dA2, int(N*N,c_size_t) * 16))
  call hipCheck(hipMemcpy(dA1, c_loc(hA(1,1)), int(N*N,c_size_t) * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dA2, c_loc(hA(1,1)), int(N*N,c_size_t) * 16, hipMemcpyHostToDevice))

  hostPtrs(1) = dA1
  hostPtrs(2) = dA2
  ptrbytes = int(batch,c_size_t) * c_sizeof(c_null_ptr)
  call hipCheck(hipMalloc(dPtrArray, ptrbytes))
  call hipCheck(hipMemcpy(dPtrArray, c_loc(hostPtrs), ptrbytes, hipMemcpyHostToDevice))

  call hipCheck(hipMalloc(dInfo, int(batch,c_size_t) * 4))
  call c_f_pointer(dInfo, dInfo_p, (/batch/))

  call hipsolverCheck(hipsolverZpotrfBatched_bufferSize(handle, HIPSOLVER_FILL_MODE_UPPER, &
       N, dPtrArray, lda, lwork, batch))
  call hipCheck(hipMalloc(dWork, int(max(lwork,1),c_size_t) * 16))
  call hipsolverCheck(hipsolverZpotrfBatched(handle, HIPSOLVER_FILL_MODE_UPPER, &
       N, dPtrArray, lda, dWork, lwork, dInfo_p(1), batch))
  call hipCheck(hipDeviceSynchronize())

  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, int(batch,c_size_t) * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hOut(1,1)), dA1, int(N*N,c_size_t) * 16, hipMemcpyDeviceToHost))

  do b = 1, batch
     if (hInfo(b) /= 0) then
        write(*,*) "FAILED! info(", b, ") = ", hInfo(b), " (expected 0)"; call exit(1)
     end if
  end do
  if (abs(real(hOut(1,1))-2.0d0) > 1.0d-9 .or. abs(real(hOut(2,2))-3.0d0) > 1.0d-9 .or. &
      abs(real(hOut(3,3))-4.0d0) > 1.0d-9) then
     write(*,*) "FAILED! chol diag = ", real(hOut(1,1)), real(hOut(2,2)), real(hOut(3,3)), " expected 2 3 4"
     call exit(1)
  end if

  call hipCheck(hipFree(dA1)); call hipCheck(hipFree(dA2)); call hipCheck(hipFree(dPtrArray))
  call hipCheck(hipFree(dWork)); call hipCheck(hipFree(dInfo))
  call hipsolverCheck(hipsolverDestroy(handle)); call hipCheck(hipDeviceReset())
  write(*,*) "PASSED!"
end program zpotrfbatched
