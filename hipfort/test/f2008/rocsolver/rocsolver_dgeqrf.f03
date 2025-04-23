!!!!!!!!!!!!!/
! dgeqrf example
! see: http:!www.netlib.org/lapack/explore-html/df/dc5/group__variants_g_ecomputational_ga3766ea903391b5cf9008132f7440ec7b.html
!!!!!!!!!!!!!!/
!
program dgeqrf
  use iso_c_binding
  use hipfort
  use hipfort_check
  use hipfort_rocblas
  use hipfort_rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  ! Define our input data
  real(c_double) :: hA(3,3) = reshape((/12, 6, -4, -51, 167, 24, 4, -68, -41/), (/3, 3/))
  real(c_double) :: hResult(3,3) = reshape((/&
  -14.000000000000000,       -21.000000000000000,        14.000000000000002,&
  0.23076923076923078,       -175.00000000000000,        70.000000000000000,&
 -0.15384615384615385,        5.5555555555555559E-002,  -35.000000000000000/), shape(hResult), order=(/2,1/))
  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  real(c_double) :: hIpiv(3) ! CPU buffer for Householder scalars

  real(c_double), pointer :: dA(:,:)   ! GPU buffer for A
  real(c_double), pointer :: dIpiv(:)  ! GPU buffer for Householder scalars

  type(c_ptr) :: handle ! rocblas_handle
    
  real :: error
  real, parameter :: error_max = 10 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dgeqrf' (Fortran 2008 interfaces) - "

  ! Allocate device-side memory & copy memory from host to device
  call hipCheck(hipMalloc(dA,    source=hA))
  call hipCheck(hipMalloc(dIpiv, source=hIpiv))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Compute the QR factorization on the devi ce
  call hipCheck(rocsolver_dgeqrf(handle, M, N, dA, lda, dIpiv))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(hA,    dA,    hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(hIpiv, dIpiv, hipMemcpyDeviceToHost))

  ! Output results
  do j = 1,size(hA,2)
    do i = 1,size(hA,1)
        error = abs(hA(i,j) - hResult(i,j))
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hA(", i, ",", j, ") = ", hA(i,j)
            call exit
        end if
      ! print *, (hA(i,j), j=1,size(hA,2))
      ! print *, (hResult(i,j), j=1,size(hA,2))
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dIpiv))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())
    
  write(*,*) "PASSED!"

end program dgeqrf
