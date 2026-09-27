!!!!!!!!!!!!!/
! sgeqrf example (single-precision QR factorization, Fortran 2003 interfaces)
! see: http:!www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc.
!!!!!!!!!!!!!!/
!
program sgeqrf
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  ! Define our input data
  real(c_float), target :: hA(3,3) = reshape((/12, 6, -4, -51, 167, 24, 4, -68, -41/), (/3, 3/))
  real(c_float), target :: hResult(3,3) = reshape((/&
  -14.000000000000000,       -21.000000000000000,        14.000000000000002,&
  0.23076923076923078,       -175.00000000000000,        70.000000000000000,&
 -0.15384615384615385,        5.5555555555555559E-002,  -35.000000000000000/), shape(hResult), order=(/2,1/))
  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  real(c_float), target :: hIpiv(3) ! CPU buffer for Householder scalars
  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_Ipiv = size(hIpiv)

  type(c_ptr) :: dA     ! GPU buffer for A
  type(c_ptr) :: dIpiv  ! GPU buffer for Householder scalars

  type(c_ptr) :: handle ! rocblas_handle

  real :: error
  ! single precision: use a relative tolerance (absolute error on
  ! magnitude-~175 entries exceeds 10*epsilon in single precision)
  real, parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_sgeqrf' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 4))
  call hipCheck(hipMalloc(dIpiv, size_Ipiv * 4))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 4, hipMemcpyHostToDevice))

  ! Compute the QR factorization on the device
  call hipCheck(rocsolver_sgeqrf(handle, M, N, dA, lda, dIpiv))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 4, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hIpiv(1)), dIpiv, size_Ipiv * 4, hipMemcpyDeviceToHost))

  ! Output results
  do j = 1,size(hA,2)
    do i = 1,size(hA,1)
        error = abs(hA(i,j) - hResult(i,j)) / max(abs(hResult(i,j)), 1.0)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hA(", i, ",", j, ") = ", hA(i,j)
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program sgeqrf
