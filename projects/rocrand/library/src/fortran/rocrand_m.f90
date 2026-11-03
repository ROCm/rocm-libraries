!! Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

! rocRAND Fortran API is deprecated. hipfort should be used instead.
module rocrand_m
    use hipfor

    integer, public :: ROCRAND_RNG_PSEUDO_DEFAULT = 400
    integer, public :: ROCRAND_RNG_PSEUDO_XORWOW = 401
    integer, public :: ROCRAND_RNG_PSEUDO_MRG32K3A = 402
    integer, public :: ROCRAND_RNG_PSEUDO_MTGP32 = 403
    integer, public :: ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = 404
    integer, public :: ROCRAND_RNG_PSEUDO_MRG31K3P = 405
    integer, public :: ROCRAND_RNG_PSEUDO_LFSR113 = 406
    integer, public :: ROCRAND_RNG_PSEUDO_MT19937 = 407
    integer, public :: ROCRAND_RNG_PSEUDO_THREEFRY2_32_20 = 408
    integer, public :: ROCRAND_RNG_PSEUDO_THREEFRY2_64_20 = 409
    integer, public :: ROCRAND_RNG_PSEUDO_THREEFRY4_32_20 = 410
    integer, public :: ROCRAND_RNG_PSEUDO_THREEFRY4_64_20 = 411
    integer, public :: ROCRAND_RNG_QUASI_DEFAULT = 500
    integer, public :: ROCRAND_RNG_QUASI_SOBOL32 = 501
    integer, public :: ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502
    integer, public :: ROCRAND_RNG_QUASI_SOBOL64 = 504
    integer, public :: ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 505

    integer, public :: ROCRAND_STATUS_SUCCESS = 0
    integer, public :: ROCRAND_STATUS_VERSION_MISMATCH  = 100
    integer, public :: ROCRAND_STATUS_NOT_CREATED  = 101
    integer, public :: ROCRAND_STATUS_ALLOCATION_FAILED  = 102
    integer, public :: ROCRAND_STATUS_TYPE_ERROR = 103
    integer, public :: ROCRAND_STATUS_OUT_OF_RANGE  = 104
    integer, public :: ROCRAND_STATUS_LENGTH_NOT_MULTIPLE  = 105
    integer, public :: ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED  = 106
    integer, public :: ROCRAND_STATUS_LAUNCH_FAILURE  = 107
    integer, public :: ROCRAND_STATUS_INTERNAL_ERROR  = 108

    interface
        function rocrand_create_generator(generator, rng_type) &
        bind(C, name="rocrand_create_generator")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_create_generator
            integer(c_size_t) :: generator
            integer(c_int), value :: rng_type
        end function

        function rocrand_create_generator_host(generator, rng_type) &
        bind(C, name="rocrand_create_generator_host")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_create_generator_host
            integer(c_size_t) :: generator
            integer(c_int), value :: rng_type
        end function

        function rocrand_destroy_generator(generator) &
        bind(C, name="rocrand_destroy_generator")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_destroy_generator
            integer(c_size_t), value :: generator
        end function

        function rocrand_generate(generator, output_data, n) &
        bind(C, name="rocrand_generate")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_generate
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
        end function

        function rocrand_generate_uniform(generator, output_data, n) &
        bind(C, name="rocrand_generate_uniform")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_generate_uniform
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
        end function

        function rocrand_generate_uniform_double(generator, output_data, n) &
        bind(C, name="rocrand_generate_uniform_double")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_generate_uniform_double
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
        end function

        function rocrand_generate_normal(generator, output_data, n, mean, &
        stddev) bind(C, name="rocrand_generate_normal")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_generate_normal
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_float), value :: mean
            real(c_float), value :: stddev
        end function

        function rocrand_generate_normal_double(generator, output_data, n, &
        mean, stddev) bind(C, name="rocrand_generate_normal_double")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_generate_normal_double
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_double), value :: mean
            real(c_double), value :: stddev
        end function

        function rocrand_generate_log_normal(generator, output_data, n, mean, &
        stddev) bind(C, name="rocrand_generate_log_normal")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_generate_log_normal
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_float), value :: mean
            real(c_float), value :: stddev
        end function

        function rocrand_generate_log_normal_double(generator, output_data, n, &
        mean, stddev) bind(C, name="rocrand_generate_log_normal_double")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_generate_log_normal_double
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_double), value :: mean
            real(c_double), value :: stddev
        end function

        function rocrand_generate_poisson(generator, output_data, n, lambda) &
        bind(C, name="rocrand_generate_poisson")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_generate_poisson
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_double), value :: lambda
        end function

        function rocrand_initialize_generator(generator) &
        bind(C, name="rocrand_initialize_generator")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_initialize_generator
            integer(c_size_t), value :: generator
        end function

        function rocrand_set_stream(generator, stream) &
        bind(C, name="rocrand_set_stream")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_set_stream
            integer(c_size_t), value :: generator
            integer(c_size_t), value :: stream
        end function

        function rocrand_set_seed(generator, seed) &
        bind(C, name="rocrand_set_seed")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_set_seed
            integer(c_size_t), value :: generator
            integer(kind =8), value :: seed
        end function

        function rocrand_set_offset(generator, offset) &
        bind(C, name="rocrand_set_offset")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_set_offset
            integer(c_size_t), value :: generator
            integer(kind =8), value :: offset
        end function

        function rocrand_set_quasi_random_generator_dimensions(generator, &
        dimensions) bind(C, name="rocrand_set_quasi_random_generator_dimensions")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_set_quasi_random_generator_dimensions
            integer(c_size_t), value :: generator
            integer(c_int), value :: dimensions
        end function

        function rocrand_get_version(version) &
        bind(C, name="rocrand_get_version")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_get_version
            integer(c_int) :: version
        end function

        function rocrand_create_poisson_distribution(lambda, &
        discrete_distribution) bind(C, name="rocrand_create_poisson_distribution")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_create_poisson_distribution
            real(c_double), value :: lambda
            integer(c_size_t) :: discrete_distribution
        end function

        function rocrand_create_discrete_distribution(probabilities, size, offset, &
        discrete_distribution) bind(C, name="rocrand_create_discrete_distribution")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_create_discrete_distribution
            real(c_double), intent(in) :: probabilities
            integer(c_int), value :: size
            integer(c_int), value :: offset
            integer(c_size_t), value :: discrete_distribution
        end function

        function rocrand_destroy_discrete_distribution(discrete_distribution) &
        bind(C, name="rocrand_destroy_discrete_distribution")
            use iso_c_binding
            implicit none
            integer(c_int) :: rocrand_destroy_discrete_distribution
            integer(c_size_t), value :: discrete_distribution
        end function
    end interface
end module rocrand_m
