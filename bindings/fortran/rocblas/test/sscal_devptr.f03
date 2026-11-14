!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
!
! SPDX-License-Identifier: MIT
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

!!!!!!!!!!!!!!
! rocblas sscal example, device pointer mode (x = alpha*x)
!
! Device-pointer-mode variant of sscal.f03. The handle is switched to
! rocblas_pointer_mode_device and alpha is staged in device memory, so rocBLAS
! reads the scalar with a device load rather than a host one. That is what
! exercises the type(c_ptr),value scalar binding end to end; the host-mode
! sibling only proves it compiles.
!
! The expected value is still computed from the HOST alpha, so a library that
! read the scalar from host memory would produce a wrong answer rather than pass.
!!!!!!!!!!!!!!
!
program rocblas_sscal_devptr_test

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer, parameter :: N = 12000
    integer, parameter :: bytes_per_element = 4 ! single precision
    integer(c_size_t), parameter :: Nbytes = N * bytes_per_element

    real(c_float),target :: alpha = 12.5

    ! Device-resident copy of the dual-mode scalar; the library reads it from
    ! device memory because the handle is in device pointer mode.
    type(c_ptr) :: d_alpha = c_null_ptr

    type(c_ptr) :: dx = c_null_ptr

    real(c_float),allocatable,target,dimension(:) :: hx
    real(c_float),allocatable,target,dimension(:) :: hres

    real(c_float) :: expected
    real(c_float) :: error
    real(c_float), parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    integer :: i

    write(*,"(a)",advance="no") "-- Running test 'sscal_devptr' (Fortran 2003 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))

    ! Allocate host-side memory
    allocate(hx(N))
    allocate(hres(N))

    ! Allocate device-side memory
    call hipCheck(hipMalloc(dx, Nbytes))

    ! Stage alpha in device memory for device pointer mode
    call hipCheck(hipMalloc(d_alpha, c_sizeof(alpha)))
    call hipCheck(hipMemcpy(d_alpha, c_loc(alpha), c_sizeof(alpha), hipMemcpyHostToDevice))

    ! Initialize host memory. alpha*i = 12.5*i is exact in binary floating
    ! point for i <= N (25*12000 < 2**24), so a correct result has zero error.
    do i = 1, N
        hx(i) = i
    end do

    ! Transfer data from host to device memory
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))

    ! Call rocblas function. scal is in-place: dx is both input and output.
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, rocblas_pointer_mode_device))
    call rocblasCheck(rocblas_sscal(rocblas_handle, N, d_alpha, dx, 1))
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
    call hipCheck(hipFree(d_alpha))
    deallocate(hx, hres)
    call rocblasCheck(rocblas_destroy_handle(rocblas_handle))

    write(*,*) "PASSED!"

end program rocblas_sscal_devptr_test
