!!!!!!!!!!!!!!
! rocsparse sgemvi example (dense matrix * sparse vector, single, Fortran 2003)
! see: https:!rocm.docs.amd.com/projects/rocSPARSE/en/latest/
!
! Computes y = alpha * A * x + beta * y, where A is a dense m-by-n matrix and x
! is a sparse vector (nnz values x_val at indices x_ind). The result is checked
! against a dense host reference (alpha * matmul(A, x_dense) + beta * y).
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host data
! is moved with hipMemcpy + c_loc. alpha/beta are host scalars (by reference).
!!!!!!!!!!!!!!
!
program dgemvi
  use iso_c_binding
  use hip
  use rocsparse
  implicit none
  integer :: i
  integer(c_int), parameter :: m = 3, n = 4, lda = 3, nnz = 2
  real(c_double), target :: hA(3,4) = reshape((/ &
    1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0, 7.0d0, 8.0d0, 9.0d0, 10.0d0, 11.0d0, 12.0d0/), (/3,4/))
  real(c_double), target :: hXval(2) = (/2.0d0, 3.0d0/)
  integer(c_int), target :: hXind(2) = (/0, 2/)
  real(c_double), target :: hY(3) = (/1.0d0, 1.0d0, 1.0d0/)
  real(c_double) :: alpha = 2.0d0, beta = 3.0d0
  real(c_double) :: xDense(4), hRef(3)
  type(c_ptr) :: handle = c_null_ptr
  type(c_ptr) :: dA, dXval, dXind, dY, dBuf
  integer(c_size_t) :: bufSize
  write(*,"(a)",advance="no") "-- Running test 'rocsparse_dgemvi' (Fortran 2003 interfaces) - "

  xDense = 0.0d0
  do i = 1, nnz
     xDense(hXind(i) + 1) = hXval(i)
  end do
  hRef = alpha * matmul(hA, xDense) + beta * hY

  call hipCheck(hipMalloc(dA,    int(m*n,c_size_t) * 8))
  call hipCheck(hipMalloc(dXval, int(nnz,c_size_t) * 8))
  call hipCheck(hipMalloc(dXind, int(nnz,c_size_t) * 4))
  call hipCheck(hipMalloc(dY,    int(m,c_size_t)   * 8))
  call hipCheck(hipMemcpy(dA,    c_loc(hA(1,1)),  int(m*n,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dXval, c_loc(hXval(1)), int(nnz,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dXind, c_loc(hXind(1)), int(nnz,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dY,    c_loc(hY(1)),    int(m,c_size_t)   * 8, hipMemcpyHostToDevice))

  call rocsparseCheck(rocsparse_create_handle(handle))
  call rocsparseCheck(rocsparse_dgemvi_buffer_size(handle, rocsparse_operation_none, m, n, nnz, bufSize))
  call hipCheck(hipMalloc(dBuf, max(bufSize, 1_c_size_t)))
  call rocsparseCheck(rocsparse_dgemvi(handle, rocsparse_operation_none, m, n, alpha, &
       dA, lda, nnz, dXval, dXind, beta, dY, rocsparse_index_base_zero, dBuf))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hY(1)), dY, int(m,c_size_t) * 8, hipMemcpyDeviceToHost))

  do i = 1, m
     if (abs(hY(i) - hRef(i)) > 1.0d-11) then
        write(*,*) "FAILED! y(", i, ") = ", hY(i), " expected ", hRef(i); call exit(1)
     end if
  end do
  call rocsparseCheck(rocsparse_destroy_handle(handle))
  call hipCheck(hipFree(dA)); call hipCheck(hipFree(dXval)); call hipCheck(hipFree(dXind))
  call hipCheck(hipFree(dY)); call hipCheck(hipFree(dBuf))
  write(*,*) "PASSED!"
end program dgemvi
