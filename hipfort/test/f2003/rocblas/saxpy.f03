!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc.
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

program rocblas_saxpy_test

    use iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_rocblas

    implicit none

    integer, parameter :: N = 12000
    integer, parameter :: bytes_per_element = 4 ! single precision
    integer(c_size_t), parameter :: Nbytes = N * bytes_per_element

    real(c_float), target :: alpha = 12.5

    type(c_ptr) :: dx = c_null_ptr
    type(c_ptr) :: dy = c_null_ptr

    real,allocatable,target,dimension(:) :: hx
    real,allocatable,target,dimension(:) :: hy
    real,allocatable,target,dimension(:) :: hz

    real :: error
    real :: result
    real, parameter :: error_max = 10 * epsilon(error_max)

    type(c_ptr) :: rocblas_handle

    integer :: i

    write(*,"(a)",advance="no") "-- Running test 'saxpy' (Fortran 2003 interfaces) - "

    ! Create rocblas handle
    call rocblasCheck(rocblas_create_handle(rocblas_handle))

    ! Allocate host-side memory
    allocate(hx(N))
    allocate(hy(N))
    allocate(hz(N))

    ! Allocate device-side memory
    call hipCheck(hipMalloc(dx, Nbytes))
    call hipCheck(hipMalloc(dy, Nbytes))

    ! Initialize host memory
    do i = 1, n
        hx(i) = i
        hy(i) = n - i
        hz(i) = n - i
    end do
    
    ! Transfer data from host to deivce memory
    call hipCheck(hipMemcpy(dx, c_loc(hx(1)), Nbytes, hipMemcpyHostToDevice))
    call hipCheck(hipmemcpy(dy, c_loc(hy(1)), Nbytes, hipMemcpyHostToDevice))

    ! Call rocblas function
    call rocblasCheck(rocblas_set_pointer_mode(rocblas_handle, 0))
    call rocblasCheck(rocblas_saxpy(rocblas_handle, N, alpha, dx, 1, dy, 1))
    call hipCheck(hipDeviceSynchronize())

    ! Transfer data back to host memory
    call hipcheck(hipMemcpy(c_loc(hy(1)), dy, Nbytes, hipMemcpyDeviceToHost))

    ! Verification
    do i = 1, N
        result = alpha * hx(i) + hz(i)
        error = abs(hy(i) - result)
        if(error .gt. error_max) then
            write(*,*) "FAILED! Error bigger than max! Error = ", error, " hy(", i, ") = ", hy(i)
            call exit
        end if
    end do

    ! Cleanup
    call hipCheck(hipFree(dx))
    call hipCheck(hipFree(dy))
    deallocate(hx, hy, hz)
    call rocblasCheck(rocblas_destroy_handle(rocblas_handle))

    write(*,*) "PASSED!"

end program rocblas_saxpy_test