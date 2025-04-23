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

program rocsparse_ddoti_test

    use iso_c_binding
    use hipfort
    use hipfort_check
    use hipfort_rocsparse

    implicit none

    integer, target :: h_xind(3)
    real(8), target :: h_xval(3), h_y(9)
    real(8), target :: h_dot

    type(c_ptr) :: d_xind
    type(c_ptr) :: d_xval
    type(c_ptr) :: d_y
    type(c_ptr) :: d_dot

    integer :: i
    integer(c_int) :: M, nnz

    type(c_ptr) :: handle
  
    write(*,"(a)",advance="no") "-- Running test 'ddoti' (Fortran 2003 interfaces) - "

!   Input data

!   Number of rows
    M = 9

!   Number of non-zero entries
    nnz = 3

!   Fill structures
    h_xind = (/0, 3, 5/)
    h_xval = (/1, 2, 3/)
    h_y    = (/1, 2, 3, 4, 5, 6, 7, 8, 9/)

!   Allocate device memory
    call hipCheck(hipMalloc(d_xind, (int(nnz, c_size_t) + 1) * 4))
    call hipCheck(hipMalloc(d_xval, int(nnz, c_size_t) * 8))
    call hipCheck(hipMalloc(d_y, int(M, c_size_t) * 8))
    call hipCheck(hipMalloc(d_dot, int(8, c_size_t)))

!   Copy host data to device
    call hipCheck(hipMemcpy(d_xind, c_loc(h_xind(1)), (int(nnz, c_size_t) + 1) * 4, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(d_xval, c_loc(h_xval(1)), int(nnz, c_size_t) * 8, hipMemcpyHostToDevice))
    call hipCheck(hipMemcpy(d_y, c_loc(h_y(1)), int(M, c_size_t) * 8, hipMemcpyHostToDevice))

!   Create rocSPARSE handle
    call rocsparseCheck(rocsparse_create_handle(handle))
    call rocsparseCheck(rocsparse_set_pointer_mode(handle, rocsparse_pointer_mode_device))

!   Call ddoti
    call rocsparseCheck(rocsparse_ddoti(handle, &
                                        nnz, &
                                        d_xval, &
                                        d_xind, &
                                        d_y, &
                                        d_dot, &
                                        rocsparse_index_base_zero))

!   Copy result back to host
    call hipCheck(hipMemcpy(c_loc(h_dot), d_dot, int(8, c_size_t), hipMemcpyDeviceToHost))

!   Verification
    if(h_dot /= 27d0) then
        write(*,*) 'FAILED!'
        call exit
    end if

!   Clear rocSPARSE
    call rocsparseCheck(rocsparse_destroy_handle(handle))

!   Clear device memory
    call hipCheck(hipFree(d_xind))
    call hipCheck(hipFree(d_xval))
    call hipCheck(hipFree(d_y))
    call hipCheck(hipFree(d_dot))

!   Print success
    write(*,*) 'PASSED!'

end program rocsparse_ddoti_test