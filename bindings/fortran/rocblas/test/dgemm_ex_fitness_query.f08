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

! rocblas_set_solution_fitness_query: `fitness` is an optional host double.
! Present, the library keeps its address in the handle and sets it at once to
! -huge (the lowest double); absent, the binding passes NULL, which turns the
! query off. While the query is on, gemm_ex is a query: on the Tensile path it
! only selects a solution, writes that solution's fitness and launches nothing
! (tensile_host.cpp). A problem that takes another path (hipBLASLt, a source
! kernel, a build without Tensile) leaves fitness at -huge, and whether D is
! then computed is not specified, so the test does not fail on that:
!   1. present: fitness, preset to a sentinel, is -huge after the call;
!      gemm_ex succeeds; if fitness was written (Tensile), D is untouched,
!      otherwise D is either untouched or the right product.
!   2. absent: fitness, reset to the sentinel, is touched neither by the call
!      nor by gemm_ex, and gemm_ex computes D = alpha*A*B + beta*C.
! On gfx1101 with ROCm 10.2, step 1 takes the Tensile path: fitness is written
! and D is left untouched.
program rocblas_dgemm_ex_fitness_query_test

    use iso_c_binding
    use hip
    use rocblas

    implicit none

    integer(c_int), parameter :: m = 3, n = 2, k = 4
    real(c_double), parameter :: sentinel = 12345.d0, marker = -7.d0
    real(c_double), parameter :: error_max = 10 * epsilon(1.d0)

    real(c_double), target :: alpha = 2.d0, beta = -1.d0
    real(c_double), target :: fitness
    real(c_double) :: hA(m,k) = reshape([1.d0, -2.d0, 3.d0, 0.5d0, 1.d0, -1.d0, &
        2.d0, 0.d0, 4.d0, -3.d0, 1.d0, 2.d0], [m,k])
    real(c_double) :: hB(k,n) = reshape([2.d0, 1.d0, 0.d0, -1.d0, 1.d0, 3.d0, -2.d0, 0.5d0], [k,n])
    real(c_double) :: hC(m,n) = reshape([1.d0, 2.d0, 3.d0, 4.d0, 5.d0, 6.d0], [m,n])
    real(c_double) :: hD(m,n), expected(m,n)
    real(c_double), pointer :: dA(:,:) => null(), dB(:,:) => null()
    real(c_double), pointer :: dC(:,:) => null(), dD(:,:) => null()
    type(c_ptr) :: handle

    write(*,"(a)",advance="no") "-- Running test 'dgemm_ex_fitness_query' (Fortran 2008 interfaces) - "

    expected = alpha * matmul(hA, hB) + beta * hC

    call rocblasCheck(rocblas_create_handle(handle))
    call rocblasCheck(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host))
    call hipCheck(hipMalloc(dA, source=hA))
    call hipCheck(hipMalloc(dB, source=hB))
    call hipCheck(hipMalloc(dC, source=hC))
    hD = marker
    call hipCheck(hipMalloc(dD, source=hD))

    ! 1. fitness present: the library stores its address and sets it to -huge
    fitness = sentinel
    call rocblasCheck(rocblas_set_solution_fitness_query(handle, fitness))
    if (fitness /= -huge(fitness)) then
        write(*,*) "FAILED! fitness after setting the query = ", fitness, " (expected -huge)"
        call exit(1)
    end if
    call gemm_ex()
    call hipCheck(hipMemcpy(hD, dD, hipMemcpyDeviceToHost))
    if (fitness /= -huge(fitness)) then
        ! a Tensile solution was queried: nothing was launched
        if (any(hD /= marker)) then
            write(*,*) "FAILED! D written during the fitness query (fitness = ", fitness, "): ", hD
            call exit(1)
        end if
    else if (.not. (all(hD == marker) .or. &
             maxval(abs(hD - expected)) <= error_max * maxval(abs(expected)))) then
        write(*,*) "FAILED! D during the fitness query is neither untouched nor the product: ", hD
        call exit(1)
    end if

    ! 2. fitness absent: NULL turns the query off; fitness is no longer written
    fitness = sentinel
    call rocblasCheck(rocblas_set_solution_fitness_query(handle))
    hD = marker
    call hipCheck(hipMemcpy(dD, hD, hipMemcpyHostToDevice))
    call gemm_ex()
    call hipCheck(hipMemcpy(hD, dD, hipMemcpyDeviceToHost))
    if (fitness /= sentinel) then
        write(*,*) "FAILED! fitness written after the query was turned off: ", fitness
        call exit(1)
    end if
    if (maxval(abs(hD - expected)) > error_max * maxval(abs(expected))) then
        write(*,*) "FAILED! D = ", hD, " expected ", expected
        call exit(1)
    end if

    call hipCheck(hipFree(dA))
    call hipCheck(hipFree(dB))
    call hipCheck(hipFree(dC))
    call hipCheck(hipFree(dD))
    call rocblasCheck(rocblas_destroy_handle(handle))
    call hipCheck(hipDeviceReset())

    write(*,*) "PASSED!"

contains

    ! D := alpha*A*B + beta*C, all f64_r, through the generic name
    subroutine gemm_ex()
        call rocblasCheck(rocblas_gemm_ex(handle, rocblas_operation_none, rocblas_operation_none, &
            m, n, k, c_loc(alpha), c_loc(dA), rocblas_datatype_f64_r, m, &
            c_loc(dB), rocblas_datatype_f64_r, k, c_loc(beta), &
            c_loc(dC), rocblas_datatype_f64_r, m, &
            c_loc(dD), rocblas_datatype_f64_r, m, &
            rocblas_datatype_f64_r, rocblas_gemm_algo_standard, 0, 0))
        call hipCheck(hipDeviceSynchronize())
    end subroutine

end program rocblas_dgemm_ex_fitness_query_test
