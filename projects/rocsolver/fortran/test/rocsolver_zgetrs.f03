!!!!!!!!!!!!!/
! zgetrs example (double-precision complex LU solve, Fortran 2003 interfaces)
! see: https:!www.netlib.org/lapack/explore-html/d6/d49/group__double_g_ecomputational_gafa35ce1d7865b80563bbed6317050ad7.html
!
! Self-verifying: pick a known solution x, form b = A*x, factorize A with
! getrf, solve A*X = b with getrs, and confirm X recovers x.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. getrf writes `info` to device memory,
! so dInfo is a device allocation passed directly (already a c_ptr).
!!!!!!!!!!!!!!/
!
program zgetrs
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i ! index for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldb = 3
  integer(c_int), parameter :: nrhs = 1

  ! Nonsingular input matrix (column-major)
  complex(c_double_complex), target :: hA(3,3) = reshape((/ &
    (12.,1.),(6.,-2.),(-4.,3.), (-51.,2.),(167.,-1.),(24.,4.), (4.,-3.),(-68.,2.),(-41.,1.) /), (/3, 3/))
  complex(c_double_complex), target :: hX(3,1) = reshape((/ (1.,1.), (2.,-1.), (3.,2.) /), (/3, 1/))  ! known solution
  complex(c_double_complex), target :: hB(3,1)                                                          ! RHS = A*x

  integer(c_int), target :: hIpiv(3)
  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_B = size(hB)
  integer(c_size_t) :: size_Ipiv = size(hIpiv)

  type(c_ptr) :: dA
  type(c_ptr) :: dB
  type(c_ptr) :: dIpiv
  type(c_ptr) :: dInfo

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 100 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_zgetrs' (Fortran 2003 interfaces) - "

  ! Build a consistent RHS so that A*x = b
  hB = matmul(hA, hX)

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 16))
  call hipCheck(hipMalloc(dB, size_B * 16))
  call hipCheck(hipMalloc(dIpiv, size_Ipiv * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 16, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), size_B * 16, hipMemcpyHostToDevice))

  ! Factorize, then solve A*X = B in place (B is overwritten with the solution)
  call hipCheck(rocsolver_zgetrf(handle, N, N, dA, lda, dIpiv, dInfo))
  call hipCheck(rocsolver_zgetrs(handle, rocblas_operation_none, N, nrhs, dA, lda, dIpiv, dB, ldb))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hB(1,1)), dB, size_B * 16, hipMemcpyDeviceToHost))

  ! Verify the recovered solution matches x
  do i = 1,N
    error = abs(hB(i,1) - hX(i,1)) / max(abs(hX(i,1)), 1.0_c_double)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " X(", i, ") = ", hB(i,1)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program zgetrs
