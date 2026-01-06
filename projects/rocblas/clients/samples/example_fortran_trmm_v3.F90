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

module procedures
    implicit none
contains
    subroutine trmm_reference(side, uplo, transA, diag, m, n, alpha, A, lda, B, ldb, C, ldc)
    use rocblas_enums
        integer(kind(        rocblas_side_left)) ::   side
        integer(kind(       rocblas_fill_upper)) ::   uplo
        integer(kind(   rocblas_operation_none)) :: transA
        integer(kind(rocblas_diagonal_non_unit)) ::   diag
        integer :: m, n, lda, ldb, ldc
        real(8), dimension(:) :: A, B, C
        integer i1, i2, i3
        real(8) alpha, t

        integer :: As1, As2, Bs1, Bs2, Cs1, Cs2
        if(transA .eq. rocblas_operation_none) then
            As1 = 1
            As2 = lda
        else
            As1 = lda
            As2 = 1
        end if

        if(side .eq. rocblas_side_left) then
            do i1 = 1, m
                do i2 = 1, n
                    t = 0.0
                    do i3 = 1, m
                        if( i1.eq.i3 .and. diag.eq.rocblas_diagonal_unit) then
                            t = t + B(i3 + (i2-1) * ldb)
                        else if( (i3.gt.i1 .and. uplo.eq.rocblas_fill_upper) .or. &
                                 (i1.gt.i3 .and. uplo.eq.rocblas_fill_lower) .or. &
                                 (i1.eq.i3 .and. diag.eq.rocblas_diagonal_non_unit)) then
                                 if(transA.eq.rocblas_operation_none)then
                                     t = t + A(i1*As1 + (i3-1)*As2) * B(i3 + (i2-1)*ldb)
                                 else
                                     t = t + A((i1-1)*As1 + i3*As2) * B(i3 + (i2-1)*ldb)
                                 endif
                        end if
                    end do
                    C(i1 + (i2-1)*ldc) = alpha * t
                end do
            end do
        end if

        if(side .eq. rocblas_side_right) then
            do i1 = 1, m
                do i2 = 1, n
                    t = 0.0
                    do i3 = 1, n
                        if( i2.eq.i3 .and. diag.eq.rocblas_diagonal_unit) then
                            t = t + B(i1 + (i3-1) * ldb)
                        else if( (i2.gt.i3 .and. uplo.eq.rocblas_fill_upper) .or. &
                                 (i3.gt.i2 .and. uplo.eq.rocblas_fill_lower) .or. &
                                 (i3.eq.i2 .and. diag.eq.rocblas_diagonal_non_unit)) then
                                 if(transA.eq.rocblas_operation_none) then
                                     t = t + A(i3*As1 + (i2-1)*As2) * B(i1 + (i3-1)*ldb)
                                 else
                                     t = t + A((i3-1)*As1 + i2*As2) * B(i1 + (i3-1)*ldb)
                                 endif
                        end if
                    end do
                    C(i1 + (i2-1)*ldc) = alpha * t
                end do
            end do
        end if

        return
    end subroutine trmm_reference
end module

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

program example_fortran_trmm
    use iso_c_binding
    use rocblas
    use rocblas_enums
    use procedures

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
    real(8) timing, max_relative_error, relative_error
    logical :: failure_in_gemv = .FALSE.
    real(c_double) :: res

    integer(c_int) :: n = 4
    integer(c_int) :: m = 4
    integer(c_int) :: lda = 4
    integer(c_int) :: ldb = 4
    integer(c_int) :: ldc = 4
    real(c_double), target :: alpha = 1.0
    integer(kind(        rocblas_side_left)), parameter ::   side = rocblas_side_left
    integer(kind(       rocblas_fill_upper)), parameter ::   uplo = rocblas_fill_upper
    integer(kind(   rocblas_operation_none)), parameter :: transA = rocblas_operation_none
    integer(kind(rocblas_diagonal_non_unit)), parameter ::   diag = rocblas_diagonal_non_unit

    integer(c_int) :: size_A = 4 * 4
    integer(c_int) :: size_B = 4 * 4
    integer(c_int) :: size_C = 4 * 4

    real(8), dimension(:), allocatable, target :: hA, hA_gold
    real(8), dimension(:), allocatable, target :: hB, hB_gold
    real(8), dimension(:), allocatable, target :: hC, hC_gold

    type(c_ptr), target :: dA
    type(c_ptr), target :: dB
    type(c_ptr), target :: dC

    real :: gpu_time_used = 0.0

    integer(c_int) :: i, element

    ! Create rocBLAS handle
    type(c_ptr), target :: handle
    call ROCBLAS_CHECK(rocblas_create_handle(c_loc(handle)))

    ! Allocate host-side memory
    allocate(hA(size_A), hA_gold(size_A))
    allocate(hB(size_B), hB_gold(size_B))
    allocate(hC(size_C), hC_gold(size_C))


    ! Allocate device-side memory
    call HIP_CHECK(hipMalloc(c_loc(dA), int(size_A, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(c_loc(dB), int(size_B, c_size_t) * 8))
    call HIP_CHECK(hipMalloc(c_loc(dC), int(size_C, c_size_t) * 8))

    ! Initialize host memory
    ! Using constant matrices so result is easy to check
    do i = 1, size_A
        hA(i) = i
        hA_gold(i) = hA(i)
    end do
    do i = 1, size_B
        hB(i) = i
        hB_gold(i) = hB(i)
    end do
    do i = 1, size_C
        hC(i) = i
        hC_gold(i) = hC(i)
    end do

!   res = alpha * 2 * 3 * size_x + beta * 4

    ! Copy memory from host to device
    call HIP_CHECK(hipMemcpy(dA, c_loc(hA), int(size_A, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(dB, c_loc(hB), int(size_B, c_size_t) * 8, 1))
    call HIP_CHECK(hipMemcpy(dC, c_loc(hC), int(size_C, c_size_t) * 8, 1))

    ! Begin time
    call date_and_time(values = tbegin)

    ! Call rocblas_dtrmm
    call ROCBLAS_CHECK(rocblas_set_pointer_mode(handle, 0))
#ifdef ROCBLAS_V3
#define rocblas_dtrmm rocblas_dtrmm_outofplace
    call ROCBLAS_CHECK(rocblas_dtrmm(handle, side, uplo, transA, diag, m, n,&
                                     c_loc(alpha), dA, lda, dB, ldb, dC, ldc))
#else
    call ROCBLAS_CHECK(rocblas_dtrmm(handle, side, uplo, transA, diag, m, n,&
                                     c_loc(alpha), dA, lda, dB, ldb         ))
#endif
    call HIP_CHECK(hipDeviceSynchronize())

    ! Stop time
    call date_and_time(values = tend)

    ! Copy output from device to host
    call HIP_CHECK(hipMemcpy(c_loc(hC), dC, int(size_C, c_size_t) * 8, 2))

    call trmm_reference(side, uplo, transA, diag, m, n, alpha, hA_gold, lda, hB_gold, ldb, hC_gold, ldc)

    max_relative_error = 0
    do i = 1, size_A
        if(hc_gold(i).eq.0)then
            relative_error = hc(i)
        else
            relative_error = (hc_gold(i) - hc(i)) / hc_gold(i)
            if(relative_error.lt.0) then
                relative_error = - relative_error
            endif
        endif
        if(relative_error.gt.max_relative_error)then
            max_relative_error = relative_error
        endif
    end do

    ! Calculate time
    tbegin = tend - tbegin
    timing = (0.001d0 * tbegin(8) + tbegin(7) + 60d0 * tbegin(6) + 3600d0 * tbegin(5)) / 200d0 * 1000d0
    write(*,fmt='(A,F0.2,A)') '[rocblas_dtrmm] took ', timing, ' msec'

    if(max_relative_error.gt.0) then
        write(*,*) 'DTRMM TEST FAIL'
        write(*,*) 'relative error =', max_relative_error
    else
        write(*,*) 'DTRMM TEST PASS'
    end if

    ! Cleanup
    call HIP_CHECK(hipFree(dA))
    call HIP_CHECK(hipFree(dB))
    call HIP_CHECK(hipFree(dC))
    deallocate(hA, hB, hC)
    call ROCBLAS_CHECK(rocblas_destroy_handle(handle))
    call HIP_CHECK(hipDeviceReset())

end program example_fortran_trmm
