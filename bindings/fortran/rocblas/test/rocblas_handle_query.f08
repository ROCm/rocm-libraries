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

! Exercises the rocBLAS handle query routines, i.e. the getters whose output
! scalar is now a plain Fortran dummy argument instead of a type(c_ptr),value.
!
! Three independent properties are asserted:
!   (a) rocblas_get_stream returns, bit for bit, the HIP stream that was handed
!       to rocblas_set_stream.  'stream' is passed by value, 'got' is a bare
!       type(c_ptr) that rocBLAS fills in, so it is passed directly.
!   (b) rocblas_pointer_to_mode classifies a genuine hipMalloc'd allocation as
!       rocblas_pointer_mode_device and a plain Fortran host array as
!       rocblas_pointer_mode_host.  It is a pure query function returning the
!       mode (not a status), so it is not wrapped in rocblasCheck.
!   (c) The int64 batch stride setters/getters round-trip: a fresh handle
!       reports 0, the values written with rocblas_set_batch_{alpha,beta}_stride
!       are read back by the matching getters (bare integer(c_int64_t) dummies),
!       and resetting them to 0 leaves the handle in its documented default.
!
! Any mismatch prints "FAILED! ..." and stops with exit code 1; otherwise the
! test prints "PASSED!" together with the strides it recovered.
program rocblas_handle_query

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer, parameter :: N = 256

    type(c_ptr) :: handle = c_null_ptr
    type(c_ptr) :: stream = c_null_ptr
    type(c_ptr) :: got = c_null_ptr

    real(c_double), allocatable, target, dimension(:) :: hx
    real(c_double), pointer, dimension(:) :: d_x

    integer(kind(rocblas_pointer_mode_host)) :: mode
    integer(c_int64_t) :: astride, bstride

    write(*,"(a)",advance="no") "-- Running test 'rocblas_handle_query' &
                                &(Fortran 2008 interfaces) - "

    call rocblasCheck(rocblas_create_handle(handle))
    call hipCheck(hipStreamCreate(stream))

    ! (a) Stream round-trip: set by value, get into a bare c_ptr.
    call rocblasCheck(rocblas_set_stream(handle, stream))
    call rocblasCheck(rocblas_get_stream(handle, got))

    if (transfer(got, 0_c_intptr_t) /= transfer(stream, 0_c_intptr_t)) then
        write(*,*) "FAILED! rocblas_get_stream did not return the stream that was set"
        STOP 1
    end if

    ! (b) Pointer classification against real device and host memory.
    allocate(hx(N))
    hx = 1.0_c_double
    call hipCheck(hipMalloc(d_x, source=hx))

    mode = rocblas_pointer_to_mode(c_loc(d_x(1)))
    if (mode /= rocblas_pointer_mode_device) then
        write(*,*) "FAILED! rocblas_pointer_to_mode misclassified the device allocation, mode = ", mode
        call hipCheck(hipFree(d_x))
        STOP 1
    end if

    mode = rocblas_pointer_to_mode(c_loc(hx(1)))
    if (mode /= rocblas_pointer_mode_host) then
        write(*,*) "FAILED! rocblas_pointer_to_mode misclassified the host array, mode = ", mode
        call hipCheck(hipFree(d_x))
        STOP 1
    end if

    call hipCheck(hipFree(d_x))

    ! (c) Batch stride round-trip; the getters take bare integer(c_int64_t).
    astride = -1_c_int64_t
    call rocblasCheck(rocblas_get_batch_alpha_stride(handle, astride))
    if (astride /= 0_c_int64_t) then
        write(*,*) "FAILED! default batch alpha stride is not 0, got ", astride
        STOP 1
    end if

    bstride = -1_c_int64_t
    call rocblasCheck(rocblas_get_batch_beta_stride(handle, bstride))
    if (bstride /= 0_c_int64_t) then
        write(*,*) "FAILED! default batch beta stride is not 0, got ", bstride
        STOP 1
    end if

    call rocblasCheck(rocblas_set_batch_alpha_stride(handle, 4_c_int64_t))
    astride = -1_c_int64_t
    call rocblasCheck(rocblas_get_batch_alpha_stride(handle, astride))
    if (astride /= 4_c_int64_t) then
        write(*,*) "FAILED! batch alpha stride did not round-trip, expected 4, got ", astride
        STOP 1
    end if

    call rocblasCheck(rocblas_set_batch_beta_stride(handle, 7_c_int64_t))
    bstride = -1_c_int64_t
    call rocblasCheck(rocblas_get_batch_beta_stride(handle, bstride))
    if (bstride /= 7_c_int64_t) then
        write(*,*) "FAILED! batch beta stride did not round-trip, expected 7, got ", bstride
        STOP 1
    end if

    ! The strides are modal handle state: restore the documented default.
    call rocblasCheck(rocblas_set_batch_alpha_stride(handle, 0_c_int64_t))
    call rocblasCheck(rocblas_set_batch_beta_stride(handle, 0_c_int64_t))
    call rocblasCheck(rocblas_get_batch_alpha_stride(handle, astride))
    call rocblasCheck(rocblas_get_batch_beta_stride(handle, bstride))
    if (astride /= 0_c_int64_t .or. bstride /= 0_c_int64_t) then
        write(*,*) "FAILED! batch strides were not restored to 0, got ", astride, bstride
        STOP 1
    end if

    call rocblasCheck(rocblas_destroy_handle(handle))
    call hipCheck(hipStreamDestroy(stream))

    write(*,*) "PASSED! stream round-trip ok, pointer modes host/device ok, batch strides: ", astride, bstride

end program rocblas_handle_query
