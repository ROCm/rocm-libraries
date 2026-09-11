!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2026 Advanced Micro Devices, Inc.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program rocblas_ctrsm_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  integer(kind(rocblas_side_left)),         parameter :: side   = rocblas_side_left
  integer(kind(rocblas_fill_lower)),        parameter :: uplo   = rocblas_fill_lower
  integer(kind(rocblas_operation_none)),    parameter :: transA = rocblas_operation_none
  integer(kind(rocblas_diagonal_non_unit)), parameter :: diag   = rocblas_diagonal_non_unit

  integer, parameter :: m = 512, n = 512

  complex(kind=4), allocatable, target, dimension(:) :: hA, hB
  complex(c_float_complex), parameter :: alpha = (2., 0.)
  complex(kind=4), parameter :: x_exact = (1., 0.)

  type(c_ptr) :: dA = c_null_ptr, dB = c_null_ptr
  type(c_ptr) :: handle = c_null_ptr

  integer, parameter :: bytes_per_element = 8 ! single-complex (2x single)
  integer(c_size_t) :: NAbytes, NBbytes

  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10*epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'CTRSM' (Fortran 2003 interfaces) - "

  ! Create rocblas handle and set host pointer mode for host alpha
  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, 0))

  NAbytes = int(m, c_size_t) * int(m, c_size_t) * bytes_per_element
  NBbytes = int(m, c_size_t) * int(n, c_size_t) * bytes_per_element

  allocate(hA(m*m), hB(m*n))

  ! Lower-triangular A with all ones on/below the diagonal; upper part zeroed
  hA(:) = (0., 0.)
  do j = 1, m
    do i = j, m
      hA(i + (j-1)*m) = (1., 0.)
    end do
  end do
  ! RHS B(i,j) = i/2 = alpha^-1 * i  ->  exact solution X(i,j) = 1
  do j = 1, n
    do i = 1, m
      hB(i + (j-1)*m) = cmplx(real(i) / 2., 0.)
    end do
  end do

  ! Allocate device memory
  call hipCheck(hipMalloc(dA, NAbytes))
  call hipCheck(hipMalloc(dB, NBbytes))

  ! Transfer from host to device
  call hipCheck(hipMemcpy(dA, c_loc(hA(1)), NAbytes, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1)), NBbytes, hipMemcpyHostToDevice))

  ! Solve A * X = alpha * B, in place: dB holds B on entry, X on exit
  call rocblasCheck(rocblas_ctrsm(handle, side, uplo, transA, diag, m, n, alpha, dA, m, dB, m))

  call hipCheck(hipDeviceSynchronize())

  ! Transfer data back to host memory
  call hipCheck(hipMemcpy(c_loc(hB(1)), dB, NBbytes, hipMemcpyDeviceToHost))

  do i = 1, m*n
    error = abs((x_exact - hB(i))/x_exact)
    if( error > error_max )then
      write(*,*) "FAILED! Error bigger than max! Error = ", error, " hB(", i, ") = ", hB(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))

  call rocblasCheck(rocblas_destroy_handle(handle))

  deallocate(hA, hB)

  write(*,*) "PASSED!"

end program rocblas_ctrsm_test
