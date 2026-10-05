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

program rocsparse_zgpsv_interleaved_batch_test

    use iso_c_binding
    use hip
    use rocsparse

    implicit none

!   Pentadiagonal system size and (single) batch layout
    integer(c_int), parameter :: M = 6
    integer(c_int), parameter :: batch_count = 1
    integer(c_int), parameter :: batch_stride = 1

!   Five diagonals of the pentadiagonal system (interleaved, batch_count = 1)
!     ds : lower diagonal at distance 2 (first two entries are zero)
!     dl : lower diagonal            (first entry is zero)
!     d  : main diagonal
!     du : upper diagonal            (last entry is zero)
!     dw : upper diagonal at distance 2 (last two entries are zero)
    complex(8) :: h_ds(M), h_dl(M), h_d(M), h_du(M), h_dw(M)
    complex(8) :: h_x(M), h_x_exact(M)
    complex(8) :: A(M,M)

    complex(8), pointer :: d_ds(:), d_dl(:), d_d(:), d_du(:), d_dw(:), d_x(:)
    type(c_ptr) :: d_buffer

    integer(c_size_t) :: buffer_size
    integer :: i
    real(8) :: err

    type(c_ptr) :: handle

    write(*,"(a)",advance="no") "-- Running test 'zgpsv_interleaved_batch' (Fortran 2008 interfaces) - "

!   Build a diagonally dominant pentadiagonal matrix
    h_ds = (0.0d0, 0.0d0)
    h_dl = (0.0d0, 0.0d0)
    h_d  = (0.0d0, 0.0d0)
    h_du = (0.0d0, 0.0d0)
    h_dw = (0.0d0, 0.0d0)

    do i = 1, M
        h_d(i) = (10.0d0, 1.0d0)
    end do
    do i = 2, M
        h_dl(i) = (-1.0d0, 0.0d0)
    end do
    do i = 1, M - 1
        h_du(i) = (-1.0d0, 0.0d0)
    end do
    do i = 3, M
        h_ds(i) = (-2.0d0, 1.0d0)
    end do
    do i = 1, M - 2
        h_dw(i) = (-2.0d0, -1.0d0)
    end do

!   Known solution and matching right-hand side (rhs = A * x_exact)
    do i = 1, M
        h_x_exact(i) = cmplx(real(i, 8), -real(i, 8), 8)
    end do

    A = (0.0d0, 0.0d0)
    do i = 1, M
        A(i,i) = h_d(i)
        if (i >= 2)     A(i,i-1) = h_dl(i)
        if (i >= 3)     A(i,i-2) = h_ds(i)
        if (i <= M - 1) A(i,i+1) = h_du(i)
        if (i <= M - 2) A(i,i+2) = h_dw(i)
    end do
    h_x = matmul(A, h_x_exact)

!   Allocate device memory and copy host data to device
    call hipCheck(hipMalloc(d_ds, source=h_ds))
    call hipCheck(hipMalloc(d_dl, source=h_dl))
    call hipCheck(hipMalloc(d_d,  source=h_d))
    call hipCheck(hipMalloc(d_du, source=h_du))
    call hipCheck(hipMalloc(d_dw, source=h_dw))
    call hipCheck(hipMalloc(d_x,  source=h_x))

!   Create rocSPARSE handle
    call rocsparseCheck(rocsparse_create_handle(handle))

!   Query the required temporary buffer size
    call rocsparseCheck(rocsparse_zgpsv_interleaved_batch_buffer_size(handle, &
                                                                      rocsparse_gpsv_interleaved_alg_qr, &
                                                                      M, &
                                                                      d_ds, d_dl, d_d, &
                                                                      d_du, d_dw, &
                                                                      d_x, &
                                                                      batch_count, &
                                                                      batch_stride, &
                                                                      buffer_size))

    call hipCheck(hipMalloc(d_buffer, buffer_size))

!   Solve the batched pentadiagonal system (x is overwritten with the solution)
    call rocsparseCheck(rocsparse_zgpsv_interleaved_batch(handle, &
                                                          rocsparse_gpsv_interleaved_alg_qr, &
                                                          M, &
                                                          d_ds, d_dl, d_d, &
                                                          d_du, d_dw, &
                                                          d_x, &
                                                          batch_count, &
                                                          batch_stride, &
                                                          d_buffer))

!   Copy the solution back to the host
    call hipCheck(hipMemcpy(h_x, d_x, hipMemcpyDeviceToHost))

!   Verification
    err = 0.0d0
    do i = 1, M
        err = max(err, abs(h_x(i) - h_x_exact(i)))
    end do

    if (.not. (err <= 1.0d-8)) then
        write(*,*) 'FAILED! max error =', err
        call exit(1)
    end if

!   Clear rocSPARSE
    call rocsparseCheck(rocsparse_destroy_handle(handle))

!   Clear device memory
    call hipCheck(hipFree(d_ds))
    call hipCheck(hipFree(d_dl))
    call hipCheck(hipFree(d_d))
    call hipCheck(hipFree(d_du))
    call hipCheck(hipFree(d_dw))
    call hipCheck(hipFree(d_x))
    call hipCheck(hipFree(d_buffer))

!   Print success
    write(*,*) 'PASSED!'

end program rocsparse_zgpsv_interleaved_batch_test
