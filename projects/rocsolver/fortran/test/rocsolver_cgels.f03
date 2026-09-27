!!!!!!!!!!!!!/
! cgels example (single-complex least-squares solve, Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/lapack.html
!
! Self-verifying: pick a known solution x, form b = A*x for a square
! nonsingular A, solve the least-squares problem with gels, and confirm the
! overwritten B recovers x.
!
! f2003 style: device buffers are type(c_ptr) allocated by byte count; host
! data is moved with hipMemcpy + c_loc. gels writes `info` to device memory,
! so dInfo is a device allocation passed directly (already a c_ptr).
!!!!!!!!!!!!!!/
!
program cgels
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none
  integer :: i ! index for iterating over results

  integer(c_int), parameter :: M = 3
  integer(c_int), parameter :: N = 3
  integer(c_int), parameter :: nrhs = 1
  integer(c_int), parameter :: lda = 3
  integer(c_int), parameter :: ldb = 3

  ! Nonsingular input matrix (column-major)
  complex(c_float_complex), target :: hA(3,3) = reshape((/ &
    (1.0,1.0), (2.0,0.0), (1.0,0.0), &
    (2.0,0.0), (5.0,1.0), (0.0,0.0), &
    (3.0,0.0), (3.0,0.0), (8.0,1.0)/), (/3, 3/))
  complex(c_float_complex), target :: hX(3,1) = reshape((/ &
    (1.0,1.0), (2.0,-1.0), (3.0,2.0)/), (/3, 1/))  ! known solution
  complex(c_float_complex), target :: hB(3,1)                 ! RHS = A*x

  integer(c_size_t) :: size_A = size(hA)
  integer(c_size_t) :: size_B = size(hB)

  type(c_ptr) :: dA
  type(c_ptr) :: dB
  type(c_ptr) :: dInfo

  type(c_ptr) :: handle ! rocblas_handle

  real(c_float) :: error
  real(c_float), parameter :: error_max = 100 * epsilon(error_max)
  !
  write(*,"(a)",advance="no") "-- Running test 'rocsolver_cgels' (Fortran 2003 interfaces) - "

  ! Build a consistent RHS so that A*x = b
  hB = matmul(hA, hX)

  ! Allocate device-side memory
  call hipCheck(hipMalloc(dA, size_A * 8))
  call hipCheck(hipMalloc(dB, size_B * 8))
  call hipCheck(hipMalloc(dInfo, 4_c_size_t))

  ! Create rocBLAS handle
  call hipCheck(rocblas_create_handle(handle))

  ! Copy memory from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), size_A * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), size_B * 8, hipMemcpyHostToDevice))

  ! Solve min || A*X - B || in place (B is overwritten with the solution)
  call hipCheck(rocsolver_cgels(handle, rocblas_operation_none, M, N, nrhs, dA, lda, dB, ldb, dInfo))

  ! Copy result from device to host
  call hipCheck(hipMemcpy(c_loc(hB(1,1)), dB, size_B * 8, hipMemcpyDeviceToHost))

  ! Verify the recovered solution matches x
  do i = 1,N
    error = abs(hB(i,1) - hX(i,1)) / max(abs(hX(i,1)), 1.0_c_float)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " X(", i, ") = ", hB(i,1)
        call exit(1)
    end if
  end do

  ! Clean up
  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dInfo))
  call hipCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program cgels
