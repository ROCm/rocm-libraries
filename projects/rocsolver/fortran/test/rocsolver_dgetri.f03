!!!!!!!!!!!!!/
! dgetri example (double-precision inverse of a general matrix, Fortran 2003 interfaces)
! Computes inv(A) from the LU factorization produced by getrf; inv(A)
! overwrites A.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. rocSOLVER writes `info` to device
! memory, so dInfo is a device allocation passed directly (already a c_ptr).
!!!!!!!!!!!!!!/
!
program dgetri
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j

  ! A (column-major), chosen with det(A) = 1 so inv(A) is integer.
  real(c_double), target :: hA(3,3) = reshape((/1, 0, 5,  2, 1, 6,  3, 4, 0/), (/3, 3/))
  ! Reference inverse (column-major)
  real(c_double), target :: hResult(3,3) = reshape((/-24, 20, -5,  18, -15, 4,  5, -4, 1/), (/3, 3/))
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_size_t) :: size_A = size(hA)

  type(c_ptr) :: dA     ! GPU buffer for A (holds inv(A) on output)
  type(c_ptr) :: dIpiv  ! GPU buffer for pivots
  type(c_ptr) :: dInfo  ! GPU buffer for info (rocSOLVER writes to device)

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dgetri' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA,    size_A * 8))
  call hipCheck(hipMalloc(dIpiv, int(N, c_size_t) * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! LU factorization, then invert in place
  call hipCheck(rocsolver_dgetrf(handle, N, N, dA, lda, dIpiv, dInfo))
  call hipCheck(rocsolver_dgetri(handle, N,    dA, lda, dIpiv, dInfo))

  ! Copy the inverse back to the host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))

  ! Check the inverse
  do j = 1,3
    do i = 1,3
        error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0_c_double)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hA(", i, ",", j, ") = ", hA(i,j)
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dgetri
