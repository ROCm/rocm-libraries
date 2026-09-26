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


program rocblas_chemm_test

  use iso_c_binding
  use hip
  use rocblas

  implicit none

  ! C := alpha*A*B + beta*C with A Hermitian (upper triangle referenced).
  ! A = [1 2; 2 3], B = I, alpha = 1, beta = 0, so C = A. The data is
  ! real-valued, so the symmetric and Hermitian forms agree.
  integer, parameter :: m = 2, n = 2
  complex(c_float_complex), parameter :: alpha = (1.0, 0.0), beta = (0.0, 0.0)

  complex(c_float_complex), target :: hA(m,m) = reshape([(1.0, 0.0), (2.0, 0.0), (2.0, 0.0), (3.0, 0.0)], [m,m])
  complex(c_float_complex), target :: hB(m,n) = reshape([(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)], [m,n])
  complex(c_float_complex), target :: hC(m,n)
  complex(c_float_complex) :: expected(m,n) = reshape([(1.0, 0.0), (2.0, 0.0), (2.0, 0.0), (3.0, 0.0)], [m,n])

  type(c_ptr) :: dA, dB, dC
  type(c_ptr) :: handle = c_null_ptr
  integer :: i, j
  real(c_float) :: error
  real(c_float), parameter :: error_max = 10 * epsilon(error)

  write(*,"(a)",advance="no") "-- Running test 'chemm' (Fortran 2003 interfaces) - "

  call rocblasCheck(rocblas_create_handle(handle))
  call rocblasCheck(rocblas_set_pointer_mode(handle, 0)) ! host pointer mode

  hC = (0.0, 0.0)

  call hipCheck(hipMalloc(dA, int(m*n,c_size_t) * 8))
  call hipCheck(hipMalloc(dB, int(m*n,c_size_t) * 8))
  call hipCheck(hipMalloc(dC, int(m*n,c_size_t) * 8))
  call hipCheck(hipMemcpy(dA, c_loc(hA(1,1)), int(m*n,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dB, c_loc(hB(1,1)), int(m*n,c_size_t) * 8, hipMemcpyHostToDevice))
  call hipCheck(hipMemcpy(dC, c_loc(hC(1,1)), int(m*n,c_size_t) * 8, hipMemcpyHostToDevice))

  call rocblasCheck(rocblas_chemm(handle, rocblas_side_left, rocblas_fill_upper, m, n, &
       alpha, dA, m, dB, m, beta, dC, m))
  call hipCheck(hipDeviceSynchronize())
  call hipCheck(hipMemcpy(c_loc(hC(1,1)), dC, int(m*n,c_size_t) * 8, hipMemcpyDeviceToHost))

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

end program rocblas_chemm_test
