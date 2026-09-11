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

program rocblas_zscal_test

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer, parameter :: N = 12000
    integer, parameter :: bytes_per_element = 16 ! double-precision complex
    integer(c_size_t), parameter :: Nbytes = N * bytes_per_element

    ! A non-zero imaginary part is deliberate: with a real-valued alpha, zscal
    ! and zdscal would compute identical results and this test could not tell
    ! them apart. Both cross terms of the complex multiply contribute here.
    complex(c_double_complex),target :: alpha = (1.5d0, 0.5d0)

    type(c_ptr) :: dx = c_null_ptr

    complex(c_double_complex),allocatable,target,dimension(:) :: hx
    complex(c_double_complex),allocatable,target,dimension(:) :: hres

    complex(c_double_complex) :: expected
    real(c_double) :: error
    real(c_double), parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    integer :: i

    write(*,"(a)",advance="no") "-- Running test 'zscal' (Fortran 2003 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))

    ! Allocate host-side memory
    allocate(hx(N))
    allocate(hres(N))

    ! Allocate device-side memory
    call hipCheck(hipMalloc(dx, Nbytes))

    ! Initialize host memory. Opposite signs on the components so a component
    ! swap is caught. (1.5 + 0.5i)*(i - i*i) = 2i - i*i, exact in binary FP.
    do i = 1, N
        hx(i) = cmplx(i, -i, kind=c_double_complex)
    end do

    ! Transfer data from host to device memory
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

    ! Call rocblas function. scal is in-place: dx is both input and output.
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, 0))
    call rocblasCheck(rocblas_zscal(rocblas_handle, N, alpha, dx, 1))
    call hipCheck(hipDeviceSynchronize())

    ! Transfer data back into a separate array so hx stays pristine for
    ! verification; scal overwrote its input device-side.
    call hipCheck(hipMemcpy(c_loc(hres(1)), dx, Nbytes, hipMemcpyDeviceToHost))

    ! Verification
    do i = 1, N
        expected = alpha * hx(i)
        error = abs((expected - hres(i)) / expected)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hres(", i, ") = ", hres(i)
            call exit(1)
        end if
    end do

    ! Cleanup
    call hipCheck(hipFree(dx))
    deallocate(hx, hres)
    call rocblasCheck(rocblas_destroy_handle(rocblas_handle))

    write(*,*) "PASSED!"

end program rocblas_zscal_test
