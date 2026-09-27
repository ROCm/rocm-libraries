!!!!!!!!!!!!!/
! hipsolver sgetrf example (single-precision LU factorization, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/hipSOLVER/en/latest/
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!
! NOTE: hipSOLVER getrf needs an explicit workspace (hipsolverSgetrf_bufferSize
! -> hipMalloc(dWork)). Its devInfo output is written to DEVICE memory: the
! binding types devInfo as an integer(c_int), so it is backed by a device
! allocation (dInfo), viewed via c_f_pointer, and passed as dInfo_p(1).
!!!!!!!!!!!!!!/
!
program hipsolver_sgetrf
  use iso_c_binding
  use hip
  use hipsolver

  implicit none
  integer :: i, j

  integer(c_int), parameter :: M = 3, N = 3, lda = 3

  ! Input matrix (column-major) and the expected packed LU (same reference as
  ! the rocSOLVER sgetrf test; hipSOLVER wraps rocSOLVER)
  real(c_float), target :: hA(3,3) = reshape((/12, 6, -4, -51, 167, 24, 4, -68, -41/), (/3, 3/))
  real(c_float) :: hResult(3,3) = reshape((/&
    12.0000000000000000d0,   0.500000000000000000d0,  -0.333333333333333315d0,&
   -51.0000000000000000d0, 192.500000000000000d0,      0.363636363636363688d-01,&
     4.00000000000000000d0, -70.0000000000000000d0,   -37.1212121212121176d0/), shape(hResult))
  integer(c_int) :: hIpiv_ref(3) = (/1, 2, 3/)

  integer(c_int), target :: hIpiv(3), hInfo
  integer(c_size_t) :: size_A = 9

  type(c_ptr) :: dA, dIpiv, dWork, dInfo   ! dInfo: device memory for devInfo
  integer(c_int), pointer :: dInfo_p(:)    ! typed view of dInfo for the by-ref arg
  type(c_ptr) :: handle = c_null_ptr
  integer(c_int) :: lwork

  real :: error
  real, parameter :: error_max = 1.0e-4

  write(*,"(a)",advance="no") "-- Running test 'hipsolver_sgetrf' (Fortran 2003 interfaces) - "

  call hipsolverCheck(hipsolverCreate(handle))

  ! Allocate device memory and copy A to device
  call hipCheck(hipMalloc(dA, size_A * 4))
  call hipCheck(hipMalloc(dIpiv, int(N,c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))
  call c_f_pointer(dInfo, dInfo_p, shape=[1])   ! typed device view for the by-ref devInfo arg
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 4, hipMemcpyHostToDevice))

  ! Query and allocate the workspace
  call hipsolverCheck(hipsolverSgetrf_bufferSize(handle, M, N, dA, lda, lwork))
  call hipCheck(hipMalloc(dWork, max(int(lwork,c_size_t) * 4, 1_c_size_t)))

  ! Compute the LU factorization
  call hipsolverCheck(hipsolverSgetrf(handle, M, N, dA, lda, dWork, lwork, dIpiv, dInfo_p(1)))

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hIpiv(1)), dIpiv, int(N,c_size_t) * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hInfo), dInfo, 4_c_size_t, hipMemcpyDeviceToHost))

  ! Check info
  if(hInfo /= 0) then
    write(*,*) "FAILED! info = ", hInfo, " (expected 0)"
    call exit(1)
  end if

  ! Check factor values
  do j = 1,size(hA,2)
    do i = 1,size(hA,1)
        error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hA(", i, ",", j, ") = ", hA(i,j)
            call exit(1)
        end if
    end do
  end do

  ! Check pivots
  do i = 1,3
    if(hIpiv(i) .ne. hIpiv_ref(i)) then
        write(*,*) "FAILED! Pivot mismatch at ", i, " got ", hIpiv(i), " expected ", hIpiv_ref(i)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call hipCheck(hipFree(dWork))
  call hipsolverCheck(hipsolverDestroy(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program hipsolver_sgetrf
