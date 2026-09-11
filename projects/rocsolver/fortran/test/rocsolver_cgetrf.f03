!!!!!!!!!!!!!/
! cgetrf example (single-precision complex LU factorization, Fortran 2003 interfaces)
! see: https:!www.netlib.org/lapack/explore-html/dd/d9a/group__double_g_ecomputational_ga0019443faea08275ca60a734d0593e60.html
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. rocSOLVER writes `info` to device
! memory, so dInfo is a device allocation passed directly (already a c_ptr).
!!!!!!!!!!!!!!/
!
program cgetrf
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, j ! indices for iterating over results

  ! Define our input data (column-major)
  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (12.,1.),(6.,-2.),(-4.,3.), (-51.,2.),(167.,-1.),(24.,4.), (4.,-3.),(-68.,2.),(-41.,1.) /), (/3, 3/))
  ! Reference: packed LU factors from rocSOLVER (column-major, matching hA layout)
  complex(c_float_complex), target :: hResult(3,3) = reshape((/ &
    ( 0.120000000000000000E+02,  0.100000000000000000E+01), &
    ( 0.482758611440658569E+00, -0.206896543502807617E+00), &
    (-0.310344815254211426E+00,  0.275862067937850952E+00), &
    (-0.510000000000000000E+02,  0.200000000000000000E+01), &
    ( 0.191206893920898438E+03, -0.125172405242919922E+02), &
    ( 0.390604361891746521E-01,  0.100302793085575104E+00), &
    ( 0.400000000000000000E+01, -0.300000000000000000E+01), &
    (-0.693103485107421875E+02,  0.427586174011230469E+01), &
    (-0.374500350952148438E+02,  0.575052165985107422E+01) /), (/3, 3/))
  integer(c_int) :: hIpiv_ref(3) = (/1, 2, 3/)
  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  integer(c_int), target :: hIpiv(3) ! CPU buffer for pivot indices
  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_Ipiv = size(hIpiv)

  type(c_ptr) :: dA     ! GPU buffer for A
  type(c_ptr) :: dIpiv  ! GPU buffer for pivots
  type(c_ptr) :: dInfo  ! GPU buffer for info (rocSOLVER writes to device)

  type(c_ptr) :: handle ! rocblas_handle

  real :: error
  ! single precision complex: relative tolerance
  real, parameter :: error_max = 1.0e-4
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_cgetrf' (Fortran 2003 interfaces) - "

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dIpiv, size_Ipiv * 4))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))

  ! Compute the LU factorization on the device
  call hipCheck(rocsolver_cgetrf(handle, M, N, dA, lda, dIpiv, dInfo))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 8, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hIpiv(1)), dIpiv, size_Ipiv * 4, hipMemcpyDeviceToHost))

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
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program cgetrf
