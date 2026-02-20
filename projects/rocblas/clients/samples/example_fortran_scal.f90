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


program example_fortran_scal
    use iso_c_binding
    use rocblas

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
    logical :: failure_in_scal = .FALSE.
    real(c_float) :: res

    integer(c_int) :: n = 10240
    real(c_float), target :: alpha = 2

    real(4), dimension(:), allocatable, target :: hx
    real(4), dimension(:), allocatable, target :: hz
    type(c_ptr), target :: dx

    real :: gpu_time_used = 0.0

    integer(c_int) :: i, element

    ! Create rocBLAS handle
    type(c_ptr), target :: handle
    call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))

    ! Allocate host-side memory
    allocate(hx(n))
    allocate(hz(n))

    ! Allocate device-side memory
    call HIP_CHECK(hipMalloc(c_loc(dx), int(n, c_size_t) * 4))

    ! Initialize host memory
    do i = 1, n
        hx(i) = i
        hz(i) = i
    end do

    ! Copy memory from host to device
    call HIP_CHECK(hipMemcpy(dx, c_loc(hx), int(n, c_size_t) * 4, 1))

    ! Begin time
    call date_and_time(values = tbegin)

    ! Call rocblas_scal
    call ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, 0))
    call ROCBLAS_CHECK(rocblas_sscal(handle, n, c_loc(alpha), dx, 1))
    call HIP_CHECK(hipDeviceSynchronize())

    ! Stop time
    call date_and_time(values = tend)

    ! Copy output from device to host
    call HIP_CHECK(hipMemcpy(c_loc(hx), dx, int(n, c_size_t) * 4, 2))

    do element = 1, n
        res = alpha * hz(element)
        if(res .ne. hx(element)) then
            failure_in_scal = .true.
            write(*,*) '[rocblas_sscal] ERROR: ', res, '!=', hx(element)
        end if
    end do

    ! Calculate time
    tbegin = tend - tbegin
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    write(*,fmt='(A,F0.2,A)') '[rocblas_sscal] took ', timing, ' msec'

    if(failure_in_scal) then
        write(*,*) 'SSCAL TEST FAIL'
    else
        write(*,*) 'SSCAL TEST PASS'
    end if

    ! Cleanup
    call HIP_CHECK(hipFree(dx))
    deallocate(hx, hz)
    call ROCBLAS_CHECK(rocblas_destroy_handle(handle))
    call HIP_CHECK(hipDeviceReset())

end program example_fortran_scal
