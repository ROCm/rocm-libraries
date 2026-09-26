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


program rocrand_sobol32_uniform_double_test

    use iso_c_binding
    use hip
    use rocrand

    implicit none

    integer(c_size_t), parameter :: N = 65536
    integer(c_size_t), parameter :: Nbytes = N * 8
    integer(c_int), parameter :: dimensions = 1
    real(c_double), parameter :: expected_mean = 0.5_c_double, delta = 0.1_c_double

    type(c_ptr) :: gen = c_null_ptr

    real(c_double), allocatable, target, dimension(:) :: hx
    type(c_ptr) :: dx = c_null_ptr
    real(c_double) :: sample_mean

    write(*,"(a)",advance="no") "-- Running test 'SOBOL32_UNIFORM_DOUBLE' (Fortran 2003 interfaces) - "

    ! Sobol32 is a quasi-random (low-discrepancy) generator: it takes a
    ! dimension count rather than a seed.
    call rocrandCheck(rocrand_create_generator(gen, ROCRAND_RNG_QUASI_SOBOL32))
    call rocrandCheck(rocrand_set_quasi_random_generator_dimensions(gen, dimensions))

    allocate(hx(N))
    call hipCheck(hipMalloc(dx, Nbytes))

    ! Uniform (0,1] samples have mean 1/2.
    call rocrandCheck(rocrand_generate_uniform_double(gen, dx, N))
    call hipCheck(hipDeviceSynchronize())
    call hipCheck(hipMemcpy(c_loc(hx(1)), dx, Nbytes, hipMemcpyDeviceToHost))

    sample_mean = sum(hx) / real(N, kind(sample_mean))
    if (abs(sample_mean - expected_mean) > delta) then
        write(*,*) "FAILED! mean out of tolerance: ", sample_mean
        call exit(1)
    end if

    call hipCheck(hipFree(dx))
    deallocate(hx)
    call rocrandCheck(rocrand_destroy_generator(gen))

    write(*,*) "PASSED!"

end program rocrand_sobol32_uniform_double_test
