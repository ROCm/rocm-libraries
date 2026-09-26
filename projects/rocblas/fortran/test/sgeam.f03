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


program rocblas_sgeam_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  ! C := alpha*op(A) + beta*op(B), both non-transposed, alpha = beta = 1,
  ! so C is the elementwise sum A + B.
  integer, parameter :: m = 2, n = 2
  real(c_float), parameter :: alpha = 1.0, beta = 1.0

  real(c_float), target :: hA(m,n) = reshape([1.0, 2.0, 3.0, 4.0], [m,n])
  real(c_float), target :: hB(m,n) = reshape([10.0, 20.0, 30.0, 40.0], [m,n])
  real(c_float), target :: hC(m,n)
  real(c_float) :: expected(m,n) = reshape([11.0, 22.0, 33.0, 44.0], [m,n])

  type(c_ptr) :: dA, dB, dC
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10 * epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'sgeam' (Fortran 2003 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, 0)) ! host pointer mode

  hC = 0.0

  call hipCheck(hipMalloc(dA, int(m*n,c_size_t) * 4))
  call hipCheck(hipMalloc(dB, int(m*n,c_size_t) * 4))
  call hipCheck(hipMalloc(dC, int(m*n,c_size_t) * 4))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(m*n,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), int(m*n,c_size_t) * 4, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), int(m*n,c_size_t) * 4, hipMemcpyHostToDevice))

  call rocblasCheck(rocblas_sgeam(handle, rocblas_operation_none, rocblas_operation_none, m, n, &
       alpha, dA, m, beta, dB, m, dC, m))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, int(m*n,c_size_t) * 4, hipMemcpyDeviceToHost))

  do j = 1, n
    do i = 1, m
      error = abs(expected(i,j) - hC(i,j))
      if (error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " at ", i, j
        call exit(1)
      end if
    end do
  end do

  call hipCheck(hipFree(dA))
  call hipCheck(hipFree(dB))
  call hipCheck(hipFree(dC))
  call rocblasCheck(rocblas_destroy_handle(handle))

  write(*,*) "PASSED!"

end program rocblas_sgeam_test
