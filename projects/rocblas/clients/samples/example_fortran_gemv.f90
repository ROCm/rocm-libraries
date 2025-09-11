!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

subroutine HIP_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: hip error'
        stop
    end if
end subroutine HIP_CHECK

subroutine ROCBLAS_CHECK(stat)
    use iso_c_binding

    implicit none

    integer(c_int) :: stat

    if(stat /= 0) then
        write(*,*) 'Error: rocblas error'
        stop
    endif
end subroutine ROCBLAS_CHECK


program example_fortran_gemv
    use iso_c_binding
    use rocblas
    use rocblas_enums

    implicit none

    ! TODO: hip workaround until plugin is ready.
    interface
        function hipMalloc(ptr, size) &
                result(c_int) &
                bind(c, name = 'hipMalloc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
            integer(c_size_t), value :: size
        end function hipMalloc
    end interface

    interface
        function hipFree(ptr) &
                result(c_int) &
                bind(c, name = 'hipFree')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: ptr
        end function hipFree
    end interface

    interface
        function hipMemcpy(dst, src, size, kind) &
                result(c_int) &
                bind(c, name = 'hipMemcpy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dst
            type(c_ptr), intent(in), value :: src
            integer(c_size_t), value :: size
            integer(c_int), value :: kind
        end function hipMemcpy
    end interface

    interface
        function hipMemset(dst, val, size) &
                result(c_int) &
                bind(c, name = 'hipMemset')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dst
            integer(c_int), value :: val
            integer(c_size_t), value :: size
        end function hipMemset
    end interface

    interface
        function hipDeviceSynchronize() &
                result(c_int) &
                bind(c, name = 'hipDeviceSynchronize')
            use iso_c_binding
            implicit none
        end function hipDeviceSynchronize
    end interface

    interface
        function hipDeviceReset() &
                result(c_int) &
                bind(c, name = 'hipDeviceReset')
            use iso_c_binding
            implicit none
        end function hipDeviceReset
    end interface
    ! TODO end


    integer tbegin(8)
    integer tend(8)
    real(8) timing
    logical :: failure_in_gemv = .FALSE.
    real(c_double) :: res

    integer(c_int) :: n = 12000
    integer(c_int) :: m = 5000
    integer(c_int) :: lda = 6500
    integer(c_int) :: incx = 1
    integer(c_int) :: incy = 1
    real(c_double), target :: alpha = 12.5
    real(c_double), target :: beta = 2.0
    integer(kind(rocblas_operation_none)), parameter :: transA = rocblas_operation_none

    integer(c_int) :: size_A = 6500 * 12000
    integer(c_int) :: size_x = 12000
    integer(c_int) :: size_y = 5000

    real(8), dimension(:), allocatable, target :: hA
    real(8), dimension(:), allocatable, target :: hx
    real(8), dimension(:), allocatable, target :: hy

    type(c_ptr), target :: dA
    type(c_ptr), target :: dx
    type(c_ptr), target :: dy

    real :: gpu_time_used = 0.0

    integer(c_int) :: i, element

    ! Create rocBLAS handle
    type(c_ptr), target :: handle
    call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))

    ! Allocate host-side memory
    allocate(hA(size_A))
    allocate(hx(size_x))
    allocate(hy(size_y))


    ! Allocate device-side memory
    call HIP_CHECK(hipMalloc(c_loc(dA), int(size_A, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(c_loc(dx), int(size_x, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(c_loc(dy), int(size_y, c_size_t) * 8))

    ! Initialize host memory
    ! Using constant matrices so result is easy to check
    do i = 1, size_A
        hA(i) = 2
    end do

    do i = 1, size_x
        hx(i) = 3
    end do

    do i = 1, size_y
        hy(i) = 4
    end do
    res = alpha * 2 * 3 * size_x + beta * 4

    ! Copy memory from host to device
    call HIP_CHECK(hipMemcpy(dA, c_loc(hA), int(size_A, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(dx, c_loc(hx), int(size_x, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(dy, c_loc(hy), int(size_y, c_size_t) * 8, 1))

    ! Begin time
    call date_and_time(values = tbegin)

    ! Call rocblas_gemv
    call ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, 0))
    call ROCBLAS_CHECK(rocblas_dgemv(handle, transA, m, n, c_loc(alpha), dA,&
                                     lda, dx, 1, c_loc(beta), dy, 1))
    call HIP_CHECK(hipDeviceSynchronize())

    ! Stop time
    call date_and_time(values = tend)

    ! Copy output from device to host
    call HIP_CHECK(hipMemcpy(c_loc(hy), dy, int(size_y, c_size_t) * 8, 2))

    do element = 1, size_y
        if(res .ne. hy(element)) then
            failure_in_gemv = .true.
            write(*,*) '[rocblas_dgemv] ERROR: ', res, '!=', hy(element)
        end if
    end do

    ! Calculate time
    tbegin = tend - tbegin
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    write(*,fmt='(A,F0.2,A)') '[rocblas_dgemv] took ', timing, ' msec'

    if(failure_in_gemv) then
        write(*,*) 'DGEMV TEST FAIL'
    else
        write(*,*) 'DGEMV TEST PASS'
    end if

    ! Cleanup
    call HIP_CHECK(hipFree(dA))
    call HIP_CHECK(hipFree(dx))
    call HIP_CHECK(hipFree(dy))
    deallocate(hA, hx, hy)
    call ROCBLAS_CHECK(rocblas_destroy_handle(handle))
    call HIP_CHECK(hipDeviceReset())

end program example_fortran_gemv
