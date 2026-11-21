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
! rocSPARSE handle creation with an optional error descriptor (Fortran 2003)
!
! rocsparse_handle_create and rocsparse_handle_destroy take a rocsparse_error*
! that may be NULL. p_error is an OPTIONAL argument of the generic: left out,
! NULL reaches C; passed, C receives the address of the caller's descriptor,
! which it may fill in when the call fails.
!!!!!!!!!!!!!!

program rocsparse_handle_create_test

    use iso_c_binding
    use hip
    use rocsparse

    implicit none

    type(c_ptr) :: handle = c_null_ptr
    type(c_ptr) :: stream = c_null_ptr
    type(c_ptr) :: err = c_null_ptr
    integer(kind(rocsparse_status_success)) :: stat

    write(*,"(a)",advance="no") "-- Running test 'rocsparse_handle_create' (Fortran 2003 interfaces) - "

    call hipCheck(hipStreamCreate(stream))

!   Without the error descriptor: NULL is passed for p_error.
    call rocsparseCheck(rocsparse_handle_create(handle, stream))
    if (.not. c_associated(handle)) then
        write(*,*) "FAILED! rocsparse_handle_create returned a null handle"
        call exit(1)
    end if
    call rocsparseCheck(rocsparse_handle_destroy(handle))

!   With it: a successful call leaves it unset.
    handle = c_null_ptr
    call rocsparseCheck(rocsparse_handle_create(handle, stream, err))
    if (.not. c_associated(handle) .or. c_associated(err)) then
        write(*,*) "FAILED! rocsparse_handle_create with p_error"
        call exit(1)
    end if
    call rocsparseCheck(rocsparse_handle_destroy(handle, err))
    if (c_associated(err)) then
        write(*,*) "FAILED! rocsparse_handle_destroy set p_error on success"
        call exit(1)
    end if

!   A failing call (null descriptors), with and without p_error. The library
!   may or may not fill in a descriptor for an argument error; free it if so.
    call rocsparseCheck(rocsparse_handle_create(handle, stream))
    stat = rocsparse_spmat_scale(handle, c_null_ptr, c_null_ptr, c_null_ptr, err)
    if (stat == rocsparse_status_success) then
        write(*,*) "FAILED! rocsparse_spmat_scale(null descriptors) with p_error succeeded"
        call exit(1)
    end if
    if (c_associated(err)) call rocsparseCheck(rocsparse_destroy_error(err))
    stat = rocsparse_spmat_scale(handle, c_null_ptr, c_null_ptr, c_null_ptr)
    if (stat == rocsparse_status_success) then
        write(*,*) "FAILED! rocsparse_spmat_scale(null descriptors) succeeded"
        call exit(1)
    end if
    call rocsparseCheck(rocsparse_handle_destroy(handle))

    call hipCheck(hipStreamDestroy(stream))

    write(*,*) "PASSED!"

end program rocsparse_handle_create_test
