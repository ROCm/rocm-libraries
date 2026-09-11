!!!!!!!!!!!!!!
! dlarft example (rocSOLVER)
! see: https:!rocm.docs.amd.com/projects/rocSOLVER/en/latest/reference/auxiliary.html
!
! Forms the triangular factor T of a block Householder reflector
! H = I - V T V**T from the reflectors V and their scalar factors tau. The
! device buffers are passed as type(c_ptr), which resolves to the raw bind(c)
! interface. tau is now a type(c_ptr) there (it used to be a scalar).
!!!!!!!!!!!!!!
!
program dlarft
  use iso_c_binding
  use hip
  use rocblas
  use rocsolver

  implicit none

  integer(c_int), parameter :: order = 4, k = 3, ldv = 4, ldt = 3

  ! V holds the Householder vectors column-wise (unit diagonal implicit).
  real(c_double), target :: hV(order,k) = reshape([ &
      1.0d0,  0.3d0,  0.2d0,  0.1d0, &
      0.0d0,  1.0d0,  0.4d0,  0.2d0, &
      0.0d0,  0.0d0,  1.0d0,  0.5d0], [order,k])
  real(c_double), target :: htau(k) = [1.5d0, 1.2d0, 1.8d0]
  real(c_double), target :: hT(k,k) = 0.0d0

  integer(c_size_t) :: size_V   = order*k
  integer(c_size_t) :: size_tau = k
  integer(c_size_t) :: size_T   = k*k

  type(c_ptr) :: dV, dtau, dT
  type(c_ptr) :: handle ! rocblas_handle

  integer :: i
  real(c_double) :: error
  real(c_double), parameter :: rtol = 1.0d-12

  write(*,"(a)",advance="no") "-- Running test 'rocsolver_dlarft' (Fortran 2003 interfaces) - "

  call hipCheck(hipMalloc(dV,   size_V   * 8))
  call hipCheck(hipMalloc(dtau, size_tau * 8))
  call hipCheck(hipMalloc(dT,   size_T   * 8))

  call rocblasCheck(rocblas_create_handle(handle))

  call hipCheck(hipMemcpy(dV,   c_loc(hV(1,1)),  size_V   * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dtau, c_loc(htau(1)),  size_tau * 8, hipMemcpyHostToDevice))

  ! Device buffers passed as type(c_ptr) (resolves to the raw bind(c) interface).
  call rocsolverCheck(rocsolver_dlarft(handle, rocblas_forward_direction, rocblas_column_wise, &
                                       order, k, dV, ldv, dtau, dT, ldt))

  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hT(1,1)), dT, size_T * 8, hipMemcpyDeviceToHost))

  ! For the forward, column-wise variant T is upper triangular with T(i,i)=tau(i).
  do i = 1, k
     error = abs(hT(i,i) - htau(i))
     if (error > rtol) then
        write(*,*) "FAILED! T(", i, ",", i, ") = ", hT(i,i), " expected tau(", i, ") = ", htau(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dV))
  call hipCheck(hipFree(dtau))
  call hipCheck(hipFree(dT))
  call rocblasCheck(rocblas_destroy_handle(handle))
  call hipCheck(hipDeviceReset())

  write(*,*) "PASSED!"

end program dlarft
