!!!!!!!!!!!!!/
! dpotrf example (double-precision Cholesky factorization, Fortran 2003 interfaces)
! see: https:!www.netlib.org/lapack/explore-html/d1/d7a/group__double_p_ocomputational_ga2f55f604a6003d03b5cd4a0adcfb9e07.html
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. rocSOLVER writes `info` to device
! memory, so dInfo is a device allocation passed directly (already a c_ptr).
!
! With rocblas_fill_upper, the Cholesky factor is written to the upper
! triangle; the lower triangle keeps the original input values.
!!!!!!!!!!!!!!/
!
program dpotrf
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  ! Symmetric positive-definite input (column-major)
  real(c_double), target :: hA(3,3) = reshape((/4, 12, -16, 12, 37, -43, -16, -43, 98/), (/3, 3/))
  ! Reference from rocSOLVER (upper = Cholesky factor, lower = original), column-major
  real(c_double), target :: hResult(3,3) = reshape((/&
     2.0000000000000000d0, 12.0000000000000000d0, -16.0000000000000000d0, &
     6.0000000000000000d0,  1.0000000000000000d0, -43.0000000000000000d0, &
    -8.0000000000000000d0,  5.0000000000000000d0,   3.0000000000000000d0/), (/3, 3/))
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  integer(c_size_t) :: size_A = size(hA)

  type(c_ptr) :: dA     ! GPU buffer for A
  type(c_ptr) :: dInfo  ! GPU buffer for info (rocSOLVER writes to device)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 10 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dpotrf' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Compute the Cholesky factorization on the device (upper triangle)
  call hipCheck(rocsolver_dpotrf(handle, rocblas_fill_upper, N, dA, lda, dInfo))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))

  ! Check factor values
  do j = 1,size(hA,2)
    do i = 1,size(hA,1)
        error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0_c_double)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hA(", i, ",", j, ") = ", hA(i,j)
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dpotrf
