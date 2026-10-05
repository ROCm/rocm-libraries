!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

module test_hiprand
    use fruit
    use fruit_helpers
    use hip
    use hiprand

    implicit none

contains

    !> Initialise test suite - no-op.
    subroutine setup
    end subroutine setup

    !> Clean up test suite - no-op.
    subroutine teardown
    end subroutine teardown

    !> Test generator functions
    subroutine test_init()
        type(c_ptr) :: gen
        integer(kind =8), parameter :: seed = 5, offset = 10
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateSeeds(gen))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandSetPseudoRandomGeneratorSeed(gen, seed))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandSetGeneratorOffset(gen, offset))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_init

    !> Test hiprandGenerate.
    subroutine test_hiprandGenerate()
        type(c_ptr) :: gen
        integer(kind =4), target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerate(gen, d_x, output_size))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerate

    !> Test hiprandGenerate with host generator.
    subroutine test_hiprandGenerate_host()
        type(c_ptr) :: gen
        integer(kind =4), target, dimension(128) :: h_x
        integer(c_size_t), parameter :: output_size = 128
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGeneratorHost(gen, &
        HIPRAND_RNG_PSEUDO_PHILOX4_32_10))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerate(gen, c_loc(h_x), output_size))
        call assert_equals(hipSuccess, hipDeviceSynchronize())
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerate_host

    !> Test hiprandGenerateLongLong.
    subroutine test_hiprandGenerateLongLong()
        type(c_ptr) :: gen
        integer(kind =8), target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_QUASI_SOBOL64))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateLongLong(gen, d_x, output_size))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateLongLong

    !> Test hiprandGenerateUniform.
    subroutine test_hiprandGenerateUniform()
        type(c_ptr) :: gen
        real, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        real, parameter :: mean = 0.5, delta = 0.1
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateUniform(gen, d_x, output_size))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! Uniform output lies in (0, 1]; this also fails on NaN.
        call assert_equals(.true., all(h_x > 0 .and. h_x <= 1), 'sample outside (0, 1]')
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateUniform

    !> Test hiprandGenerateUniformDouble.
    subroutine test_hiprandGenerateUniformDouble()
        type(c_ptr) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: mean = 0.5d0, delta = 0.1d0
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateUniformDouble(gen, d_x, &
        output_size))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! Uniform output lies in (0, 1]; this also fails on NaN.
        call assert_equals(.true., all(h_x > 0 .and. h_x <= 1), 'sample outside (0, 1]')
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateUniformDouble

    !> Test hiprandGenerateNormal.
    subroutine test_hiprandGenerateNormal()
        type(c_ptr) :: gen
        real, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        ! Not the standard normal, so a binding that drops mean or stddev cannot pass.
        real, parameter :: mean = 2.0, stddev = 0.5, delta = 0.2
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateNormal(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! FRUIT's range assertions below pass on NaN.
        call assert_equals(.true., all(h_x == h_x), 'sample contains NaN')
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(sqrt(sum((h_x - mean) ** 2) / output_size), stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateNormal

    !> Test hiprandGenerateNormalDouble.
    subroutine test_hiprandGenerateNormalDouble()
        type(c_ptr) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        ! Not the standard normal, so a binding that drops mean or stddev cannot pass.
        double precision, parameter :: mean = 2.0d0, stddev = 0.5d0, delta = 0.2d0
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateNormalDouble(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! FRUIT's range assertions below pass on NaN.
        call assert_equals(.true., all(h_x == h_x), 'sample contains NaN')
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(sqrt(sum((h_x - mean) ** 2) / output_size), stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateNormalDouble

    !> Test hiprandGenerateLogNormal.
    subroutine test_hiprandGenerateLogNormal()
        type(c_ptr) :: gen
        real, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        real, parameter :: mean = 1.6, stddev = 0.25, delta = 0.2
        real :: m, s
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateLogNormal(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! If x is log-normal with parameters (mean, stddev) then log(x) is
        ! normal with those moments. Log-normal samples are strictly
        ! positive; check it first, as a NaN from log() would slip through
        ! the range assertions below.
        call assert_equals(.true., all(h_x > 0), 'log-normal sample was not positive')
        m = sum(log(h_x)) / output_size
        s = sqrt(sum((log(h_x) - m) ** 2) / output_size)
        call assert_equals(m, mean, delta)
        call assert_equals(s, stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateLogNormal

    !> Test hiprandGenerateLogNormalDouble.
    subroutine test_hiprandGenerateLogNormalDouble()
        type(c_ptr) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: mean = 1.6d0, stddev = 0.25d0, delta = 0.2d0
        double precision :: m, s
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateLogNormalDouble(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! If x is log-normal with parameters (mean, stddev) then log(x) is
        ! normal with those moments. Log-normal samples are strictly
        ! positive; check it first, as a NaN from log() would slip through
        ! the range assertions below.
        call assert_equals(.true., all(h_x > 0), 'log-normal sample was not positive')
        m = sum(log(h_x)) / output_size
        s = sqrt(sum((log(h_x) - m) ** 2) / output_size)
        call assert_equals(m, mean, delta)
        call assert_equals(s, stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateLogNormalDouble

    !> Test hiprandGeneratePoisson.
    subroutine test_hiprandGeneratePoisson()
        type(c_ptr) :: gen, distribution
        integer(kind =4), target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: lambda = 20.0d0
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreatePoissonDistribution(lambda, &
        distribution))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGeneratePoisson(gen, d_x, &
        output_size, lambda))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyDistribution( &
        distribution))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGeneratePoisson

    !> Call each test.
    subroutine hiprand_basket()
    character(len=*) :: suite_name
    parameter(suite_name='test_hiprand')

    call run_fruit_test_case(test_init,'test_init',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerate,'test_hiprandGenerate',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerate_host,'test_hiprandGenerate_host',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerateLongLong,'test_hiprandGenerateLongLong',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerateUniform,'test_hiprandGenerateUniform',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerateUniformDouble, &
        'test_hiprandGenerateUniformDouble',setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerateNormal,'test_hiprandGenerateNormal',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerateNormalDouble, &
        'test_hiprandGenerateNormalDouble',setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerateLogNormal,'test_hiprandGenerateLogNormal',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGenerateLogNormalDouble, &
        'test_hiprandGenerateLogNormalDouble',setup,teardown,suite_name)

    call run_fruit_test_case(test_hiprandGeneratePoisson,'test_hiprandGeneratePoisson',&
        setup,teardown,suite_name)

    end subroutine hiprand_basket

end module test_hiprand
