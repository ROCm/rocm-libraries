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

program rocblas_zdotc_test

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer, parameter :: N = 10240
    integer, parameter :: bytes_per_element = 16 ! double complex
    integer(c_size_t), parameter :: Nbytes = N * bytes_per_element

    type(c_ptr) :: dx = c_null_ptr
    type(c_ptr) :: dy = c_null_ptr

    complex(kind=8),allocatable,target,dimension(:) :: hx
    complex(kind=8),allocatable,target,dimension(:) :: hy
    complex(kind=8),target :: res

    complex(kind=8) :: res_exact
    double precision :: error
    double precision, parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    write(*,"(a)",advance="no") "-- Running test 'zdotc' (Fortran 2003 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, 0)) ! host pointer mode

    ! Allocate host-side memory
    allocate(hx(N))
    allocate(hy(N))

    ! Initialize host memory
    hx = (1.d0, 1.d0)                                 ! x = 1 + i
    hy = (2.d0, 0.d0)                                 ! y = 2
    res_exact = cmplx(2.d0*N, -2.d0*N, kind=8)       ! zdotc -> (2n, -2n)

    ! Allocate device-side memory
    call hipCheck(hipMalloc(dx, Nbytes))
    call hipCheck(hipMalloc(dy, Nbytes))

    ! Transfer data from host to device memory
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(dy, c_loc(hy(1)), Nbytes, hipMemcpyHostToDevice))

    ! Call rocblas function
    res = (0.d0, 0.d0)
    call rocblasCheck(rocblas_zdotc(rocblas_handle, N, dx, 1, dy, 1, c_loc(res)))
    call hipCheck(hipDeviceSynchronize()) ! res now valid host-side

    ! Verification
    error = abs((res_exact - res) / res_exact)
    if(error .gt. error_max) then
        write(*,*) "FAILED! Error bigger than max! Error = ", error, " result = ", res
        call exit(1)
    end if

    ! Cleanup
    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    deallocate(hx, hy)
    call rocblasCheck(rocblas_destroy_handle(rocblas_handle))

    write(*,*) "PASSED!"

end program rocblas_zdotc_test
