!!!!!!!!!!!!!/
! zheevd example (double-precision Hermitian eigensolver, Fortran 2003 interfaces)
! see: https:!www.netlib.org/lapack/explore-html/d9/de3/group__complex_h_eeigen.html
!
! Self-verifying: with evect=original, A is overwritten with the eigenvectors
! (as columns) and D holds the (real) eigenvalues. We confirm A0*v_k = lambda_k*v_k.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. heevd writes `info` to device memory,
! so dInfo is a device allocation passed directly (already a c_ptr).
! For Hermitian matrices A is complex (16 bytes) but the eigenvalues D (and
! workspace E) are REAL (8 bytes).
!!!!!!!!!!!!!!/
!
program zheevd
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i, k ! indices for iterating over results

  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: lda = 3

  ! Hermitian input (column-major)
  complex(c_double_complex), target :: hA(3,3) = reshape((/ &
    (2.,0.),(0.,-1.),(0.,0.), (0.,1.),(2.,0.),(0.,-1.), (0.,0.),(0.,1.),(2.,0.) /), (/3, 3/))
  complex(c_double_complex) :: hA0(3,3)   ! original kept for verification
  real(c_double), target :: hD(3)                  ! eigenvalues (real)
  real(c_double), target :: hE(3)                  ! workspace (real)
  complex(c_double_complex) :: lhs(3), rhs(3)

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_D = size(hD)
  integer(c_size_t) :: size_E = size(hE)

  type(c_ptr) :: dA
  type(c_ptr) :: dD
  type(c_ptr) :: dE
  type(c_ptr) :: dInfo

  type(c_ptr) :: handle ! rocblas_handle

  real(c_double) :: error
  real(c_double), parameter :: error_max = 1.0d-9
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_zheevd' (Fortran 2003 interfaces) - "

  hA0 = hA ! keep original for the A*v = lambda*v check

  ! Allocate device-side memory (A complex = 16 bytes, D/E real = 8 bytes)
  call hipCheck(hipMalloc(dA, size_A * 16))
  call hipCheck(hipMalloc(dD, size_D * 8))
  call hipCheck(hipMalloc(dE, size_E * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 16, hipMemcpyHostToDevice))

  ! Compute eigenvalues and eigenvectors (A overwritten with eigenvectors)
  call hipCheck(rocsolver_zheevd(handle, rocblas_evect_original, rocblas_fill_upper, N, dA, lda, dD, dE, dInfo))

  ! Copy results back to host
  call hipCheck(hipMemcpy(c_loc(hA(1,1)), dA, size_A * 16, hipMemcpyDeviceToHost))
  call hipCheck(hipMemcpy(c_loc(hD(1)), dD, size_D * 8, hipMemcpyDeviceToHost))

  ! Verify A0 * v_k = lambda_k * v_k for each eigenpair
  do k = 1,N
    lhs = matmul(hA0, hA(:,k))
    rhs = hD(k) * hA(:,k)
    do i = 1,N
        error = abs(lhs(i) - rhs(i))
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " eigenpair ", k
            call exit(1)
        end if
    end do
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dD))
  call hipCheck(hipFree(dE))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program zheevd
