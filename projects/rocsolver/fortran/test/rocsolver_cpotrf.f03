!!!!!!!!!!!!!/
! cpotrf example (single-precision complex Cholesky factorization, Fortran 2003 interfaces)
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
program cpotrf
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  ! Hermitian positive-definite input (column-major)
  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (25.,0.),(1.,2.),(3.,-1.), (1.,-2.),(18.,0.),(2.,3.), (3.,1.),(2.,-3.),(30.,0.) /), (/3, 3/))
  ! Reference from rocSOLVER (upper = Cholesky factor, lower = original), column-major
  complex(c_float_complex), target :: hResult(3,3) = reshape((/ &
    ( 0.500000000000000000E+01, -0.000000000000000000E+00), &
    ( 0.100000000000000000E+01,  0.200000000000000000E+01), &
    ( 0.300000000000000000E+01, -0.100000000000000000E+01), &
    ( 0.200000002980232239E+00, -0.400000005960464478E+00), &
    ( 0.421900463104248047E+01, -0.000000000000000000E+00), &
    ( 0.200000000000000000E+01,  0.300000000000000000E+01), &
    ( 0.600000023841857910E+00,  0.200000002980232239E+00), &
    ( 0.464564561843872070E+00, -0.777434527873992920E+00), &
    ( 0.536467838287353516E+01, -0.000000000000000000E+00) /), (/3, 3/))
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  integer(c_size_t) :: size_A = size(hA)

  type(c_ptr) :: dA     ! GPU buffer for A
  type(c_ptr) :: dInfo  ! GPU buffer for info (rocSOLVER writes to device)

  type(c_ptr) :: handle ! rocblas_handle

  real :: error
  ! single precision complex: relative tolerance
  real, parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_cpotrf' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Compute the Cholesky factorization on the device (upper triangle)
  call hipCheck(rocsolver_cpotrf(handle, rocblas_fill_upper, N, dA, lda, dInfo))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))

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

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program cpotrf
