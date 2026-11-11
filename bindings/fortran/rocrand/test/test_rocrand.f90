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

module test_rocrand
    use fruit
    use fruit_helpers
    use hip
    use rocrand

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
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_initialize_generator(gen))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_set_seed(gen, seed))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_set_offset(gen, offset))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_init

    !> Test rocrand_generate.
    subroutine test_rocrand_generate()
        type(c_ptr) :: gen
        integer(kind =4), target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate(gen, d_x, output_size))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate

    !> Test rocrand_generate using a host generator.
    subroutine test_rocrand_generate_host()
        type(c_ptr) :: gen
        integer(kind =4), target, dimension(128) :: h_x
        integer(c_size_t), parameter :: output_size = 128
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator_host(gen, &
        ROCRAND_RNG_PSEUDO_MRG31K3P))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate(gen, c_loc(h_x), output_size))
        call assert_equals(hipSuccess, hipDeviceSynchronize())
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate_host

    !> Test rocrand_generate_uniform.
    subroutine test_rocrand_generate_uniform()
        type(c_ptr) :: gen
        real, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        real, parameter :: mean = 0.5, delta = 0.1
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate_uniform(gen, d_x, output_size))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! Uniform output lies in (0, 1]; this also fails on NaN.
        call assert_equals(.true., all(h_x > 0 .and. h_x <= 1), 'sample outside (0, 1]')
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate_uniform

    !> Test rocrand_generate_uniform_double.
    subroutine test_rocrand_generate_uniform_double()
        type(c_ptr) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: mean = 0.5d0, delta = 0.1d0
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate_uniform_double(gen, d_x, &
        output_size))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! Uniform output lies in (0, 1]; this also fails on NaN.
        call assert_equals(.true., all(h_x > 0 .and. h_x <= 1), 'sample outside (0, 1]')
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate_uniform_double

    !> Test rocrand_generate_normal.
    subroutine test_rocrand_generate_normal()
        type(c_ptr) :: gen
        real, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        ! Not the standard normal, so a binding that drops mean or stddev cannot pass.
        real, parameter :: mean = 2.0, stddev = 0.5, delta = 0.2
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate_normal(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! FRUIT's range assertions below pass on NaN.
        call assert_equals(.true., all(h_x == h_x), 'sample contains NaN')
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(sqrt(sum((h_x - mean) ** 2) / output_size), stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate_normal

    !> Test rocrand_generate_normal_double.
    subroutine test_rocrand_generate_normal_double()
        type(c_ptr) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        ! Not the standard normal, so a binding that drops mean or stddev cannot pass.
        double precision, parameter :: mean = 2.0d0, stddev = 0.5d0, delta = 0.2d0
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate_normal_double(gen, d_x, &
        output_size, mean, stddev))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        ! FRUIT's range assertions below pass on NaN.
        call assert_equals(.true., all(h_x == h_x), 'sample contains NaN')
        call assert_equals((sum(h_x) / output_size), mean, delta)
        call assert_equals(sqrt(sum((h_x - mean) ** 2) / output_size), stddev, delta)
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate_normal_double

    !> Test rocrand_generate_log_normal.
    subroutine test_rocrand_generate_log_normal()
        type(c_ptr) :: gen
        real, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        real, parameter :: mean = 1.6, stddev = 0.25, delta = 0.2
        real :: m, s
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate_log_normal(gen, d_x, &
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
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate_log_normal

    !> Test rocrand_generate_log_normal_double.
    subroutine test_rocrand_generate_log_normal_double()
        type(c_ptr) :: gen
        double precision, target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: mean = 1.6d0, stddev = 0.25d0, delta = 0.2d0
        double precision :: m, s
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate_log_normal_double(gen, d_x, &
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
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate_log_normal_double

    !> Test rocrand_generate_poisson.
    subroutine test_rocrand_generate_poisson()
        type(c_ptr) :: gen, distribution
        integer(kind =4), target, dimension(128) :: h_x
        type(c_ptr) :: d_x
        integer(c_size_t), parameter :: output_size = 128
        double precision, parameter :: lambda = 20.0d0
        call assert_equals(hipSuccess, hipMalloc(d_x, output_size * sizeof(h_x(1))))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_generator(gen, &
        ROCRAND_RNG_PSEUDO_DEFAULT))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_create_poisson_distribution(lambda, &
        distribution))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_generate_poisson(gen, d_x, &
        output_size, lambda))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_discrete_distribution( &
        distribution))
        call assert_equals(hipSuccess, hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), &
        hipMemcpyDeviceToHost))
        call assert_equals(hipSuccess, hipFree(d_x))
        call assert_equals(ROCRAND_STATUS_SUCCESS, rocrand_destroy_generator(gen))
    end subroutine test_rocrand_generate_poisson

    !> Call each test.
    subroutine rocrand_basket()
    character(len=*) :: suite_name
    parameter(suite_name='test_rocrand')

    call run_fruit_test_case(test_init,'test_init',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate,'test_rocrand_generate',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate_host,'test_rocrand_generate_host',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate_uniform,'test_rocrand_generate_uniform',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate_uniform_double, &
        'test_rocrand_generate_uniform_double',setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate_normal,'test_rocrand_generate_normal',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate_normal_double, &
        'test_rocrand_generate_normal_double',setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate_log_normal,'test_rocrand_generate_log_normal',&
        setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate_log_normal_double, &
        'test_rocrand_generate_log_normal_double',setup,teardown,suite_name)

    call run_fruit_test_case(test_rocrand_generate_poisson,'test_rocrand_generate_poisson',&
        setup,teardown,suite_name)

    end subroutine rocrand_basket

end module test_rocrand
