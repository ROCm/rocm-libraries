!! Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
!!
!! Permission is hereby granted, free of charge, to any person obtaining a copy
!! of this software and associated documentation files (the "Software"), to deal
!! in the Software without restriction, including without limitation the rights
!! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
!! copies of the Software, and to permit persons to whom the Software is
!! furnished to do so, subject to the following conditions:
!!
!! The above copyright notice and this permission notice shall be included in
!! all copies or substantial portions of the Software.
!!
!! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
!! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
!! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
!! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
!! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
!! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
!! THE SOFTWARE.

module test_hiprand
    use fruit
    use fruit_helpers
    use hipfor
    use hiprand_m

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
        integer(kind =8) :: gen
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
        integer(kind =8) :: gen
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
        integer(kind =8) :: gen
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
        integer(kind =8) :: gen
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
        integer(kind =8) :: gen
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
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateUniform

    !> Test hiprandGenerateUniformDouble.
    subroutine test_hiprandGenerateUniformDouble()
        integer(kind =8) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: mean = 0.5, delta = 0.1
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateUniformDouble(gen, d_x, &
        output_size))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateUniformDouble

    !> Test hiprandGenerateNormal.
    subroutine test_hiprandGenerateNormal()
        integer(kind =8) :: gen
        real, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        real, parameter :: mean = 0.0, stddev = 1.0, delta = 0.2
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateNormal(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(sqrt(sum((h_x - mean) ** 2) / output_size), stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateNormal

    !> Test hiprandGenerateNormalDouble.
    subroutine test_hiprandGenerateNormalDouble()
        integer(kind =8) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: mean = 0.0, stddev = 1.0, delta = 0.2
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateNormalDouble(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(sqrt(sum((h_x - mean) ** 2) / output_size), stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateNormalDouble

    !> Test hiprandGenerateLogNormal.
    subroutine test_hiprandGenerateLogNormal()
        integer(kind =8) :: gen
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
        m = sum(h_x) / output_size
        s = sqrt(sum((h_x - mean) ** 2) / output_size)
        call assert_equals(log(m * m / sqrt(s + m * m)), mean, delta)
        call assert_equals(sqrt(log(1.0 + s / (m * m))), stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateLogNormal

    !> Test hiprandGenerateLogNormalDouble.
    subroutine test_hiprandGenerateLogNormalDouble()
        integer(kind =8) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: mean = 1.6, stddev = 0.25, delta = 0.2
        double precision :: m, s
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandCreateGenerator(gen, &
        HIPRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandGenerateLogNormalDouble(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        m = sum(h_x) / output_size
        s = sqrt(sum((h_x - mean) ** 2) / output_size)
        call assert_equals(log(m * m / sqrt(s + m * m)), mean, delta)
        call assert_equals(sqrt(log(1.0 + s / (m * m))), stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(HIPRAND_STATUS_SUCCESS, hiprandDestroyGenerator(gen))
    end subroutine test_hiprandGenerateLogNormalDouble

    !> Test hiprandGeneratePoisson.
    subroutine test_hiprandGeneratePoisson()
        integer(kind =8) :: gen, distribution
        integer(kind =4), target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: lambda = 20.0
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
