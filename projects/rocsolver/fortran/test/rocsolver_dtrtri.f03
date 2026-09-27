!!!!!!!!!!!!!!
! dtrtri example (rocSOLVER)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Inverts an upper-triangular matrix in place and checks U * U^-1 == I.
! rocSOLVER writes `info` to DEVICE memory, so it is backed by a device
! allocation and passed as c_loc(dInfo).
!!!!!!!!!!!!!!
!
program dtrtri
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: n = 3, lda = 3

  ! Upper-triangular matrix (lower part zero), column-major:
  !   [2 1 1; 0 2 1; 0 0 2]
  real(c_double), target :: hU(n,n) = reshape([ &
      2.0d0, 0.0d0, 0.0d0, &
      1.0d0, 2.0d0, 0.0d0, &
      1.0d0, 1.0d0, 2.0d0], [n,n])
  real(c_double), target :: hUinv(n,n) = 0.0d0

  integer(c_size_t) :: size_A = n*n

  type(c_ptr) :: dA, dInfo
  type(c_ptr) :: handle ! rocblas_handle

  integer :: i, j, l
  real(c_double) :: prod, error
  real(c_double), parameter :: error_max = 1.0d-10

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dtrtri' (Fortran 2003 interfaces) - "

  call hipCheck(hipMalloc(dA,    size_A * 8))
  call hipCheck(hipMalloc(dInfo, int(4, c_size_t)))

  call rocblasCheck(rocblas_create_handle(handle))

  call hipCheck(hipMemcpy(dA, c_loc(hU(1,1)), size_A * 8, hipMemcpyHostToDevice))

  call rocsolverCheck(rocsolver_dtrtri(handle, rocblas_fill_upper, rocblas_diagonal_non_unit, &
                                       n, dA, lda, dInfo))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hUinv(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))

  ! U and U^-1 are both upper triangular (zero below), so U * U^-1 == I exactly.
  do j = 1, n
     do i = 1, n
        prod = 0.0d0
        do l = 1, n
           prod = prod + hU(i,l) * hUinv(l,j)
        end do
        error = abs(prod - merge(1.0d0, 0.0d0, i == j))
        if (error > error_max) then
           write(*,*) "FAILED! (U*Uinv)(", i, ",", j, ") = ", prod
           call exit(1)
        end if
     end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dInfo))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dtrtri
