!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! rocsolver_dgetrf_strided_batched example / unit test
!
! Computes the LU factorization of a batch of matrices on the GPU.
!
! This test guards the fix for the batched LU "info" argument (myInfo):
! in rocSOLVER the batched routines write an array of `batch_count` integers
! to a *device* pointer, so the Fortran binding must declare `myInfo` as
! type(c_ptr). Previously it was `integer(c_int)`, which can only represent a
! single scalar passed by reference and cannot point at a device array.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
program dgetrf_strided_batched
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: M = 3, N = 3, lda = 3
  integer(c_int), parameter :: batch_count = 2
  integer(c_int), parameter :: sz_piv = min(M, N)
  integer(c_int64_t), parameter :: strideA = int(lda, c_int64_t) * int(N, c_int64_t)
  integer(c_int64_t), parameter :: strideP = int(sz_piv, c_int64_t)

  ! Two invertible 3x3 matrices stored back-to-back (column-major).
  real(c_double), target :: hA(lda, N*batch_count)
  integer(c_int), target :: hInfo(batch_count)
  integer(c_int), target :: hIpiv(sz_piv*batch_count)

  integer(c_size_t) :: size_A    = lda*N*batch_count
  integer(c_size_t) :: size_Ipiv = sz_piv*batch_count
  integer(c_size_t) :: size_Info = batch_count

  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA     ! GPU buffer for the strided matrices
  type(c_ptr) :: dIpiv  ! GPU buffer for the pivot indices
  type(c_ptr) :: dInfo  ! GPU buffer for the info array

  integer :: b

  write(*,"(a)",advance="no") &
    "-- Running test 'rocsolver_dgetrf_strided_batched' (Fortran 2003 interfaces) - "

  hA(:, 1:N)       = reshape((/ 12.d0, -51.d0,   4.d0, &
                                 6.d0, 167.d0, -68.d0, &
                                -4.d0,  24.d0, -41.d0/), (/lda, N/))
  hA(:, N+1:2*N)   = reshape((/  2.d0,   0.d0,   0.d0, &
                                 0.d0,   3.d0,   0.d0, &
                                 0.d0,   0.d0,   5.d0/), (/lda, N/))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA,    size_A    * 8))  ! c_double
  call hipCheck(hipMalloc(dIpiv, size_Ipiv * 4))  ! c_int
  call hipCheck(hipMalloc(dInfo, size_Info * 4))  ! c_int

  ! Copy the input matrices to the device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Compute the batched LU factorization on the device.
  ! `dInfo` is a device pointer to an array of `batch_count` ints, which is
  ! exactly what the myInfo -> c_ptr binding fix enables.
  call hipCheck(rocsolver_dgetrf_strided_batched(handle, M, N, dA, lda, strideA, &
                                                 dIpiv, strideP, dInfo, batch_count))

  ! Copy the info array back to the host
  call hipCheck(hipMemcpy(c_loc(hInfo(1)), dInfo, size_Info * 4, hipMemcpyDeviceToHost))

  ! For invertible inputs rocSOLVER sets info(b) == 0 for every batch entry.
  do b = 1, batch_count
    if (hInfo(b) /= 0) then
      write(*,*) "FAILED! info(", b, ") = ", hInfo(b), " (expected 0)"
      call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dgetrf_strided_batched
