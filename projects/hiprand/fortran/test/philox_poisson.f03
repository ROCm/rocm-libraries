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


program hiprand_philox_poisson_test

    use iso_c_binding
    use hip
    use hiprand

    implicit none

    integer(c_size_t), parameter :: N = 65536
    integer(c_size_t), parameter :: Nbytes = N * 4
    integer(c_int64_t), parameter :: seed = 12345_c_int64_t
    real(c_double), parameter :: lambda = 10.0_c_double
    real(c_double), parameter :: delta = 0.5_c_double

    type(c_ptr) :: gen = c_null_ptr

    integer(c_int), allocatable, target, dimension(:) :: hx
    type(c_ptr) :: dx = c_null_ptr
    real(c_double) :: sample_mean

    write(*,"(a)",advance="no") "-- Running test 'PHILOX_POISSON' (Fortran 2003 interfaces) - "

    call hiprandCheck(hiprandCreateGenerator(gen, HIPRAND_RNG_PSEUDO_PHILOX4_32_10))
    call hiprandCheck(hiprandSetPseudoRandomGeneratorSeed(gen, seed))

    allocate(hx(N))
    call hipCheck(hipMalloc(dx, Nbytes))

    ! Poisson(lambda) samples are unsigned 32-bit integers with mean lambda.
    call hiprandCheck(hiprandGeneratePoisson(gen, dx, N, lambda))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))

    sample_mean = real(sum(hx), c_double) / real(N, c_double)
    if (abs(sample_mean - lambda) > delta) then
        write(*,*) "FAILED! mean out of tolerance: ", sample_mean
        call exit(1)
    end if

    call hipCheck(hipFree(dx))
    deallocate(hx)
    call hiprandCheck(hiprandDestroyGenerator(gen))

    write(*,*) "PASSED!"

end program hiprand_philox_poisson_test
