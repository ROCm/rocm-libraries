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

! hipRAND Fortran API is deprecated. hipfort should be used instead.
module hiprand_m
    use hipfor

    integer, public :: HIPRAND_RNG_PSEUDO_DEFAULT = 400
    integer, public :: HIPRAND_RNG_PSEUDO_XORWOW = 401
    integer, public :: HIPRAND_RNG_PSEUDO_MRG32K3A = 402
    integer, public :: HIPRAND_RNG_PSEUDO_MTGP32 = 403
    integer, public :: HIPRAND_RNG_PSEUDO_MT19937 = 404
    integer, public :: HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 405
    integer, public :: HIPRAND_RNG_QUASI_DEFAULT = 500
    integer, public :: HIPRAND_RNG_QUASI_SOBOL32 = 501
    integer, public :: HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502
    integer, public :: HIPRAND_RNG_QUASI_SOBOL64 = 503
    integer, public :: HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 504

    integer, public :: HIPRAND_STATUS_SUCCESS = 0
    integer, public :: HIPRAND_STATUS_VERSION_MISMATCH  = 100
    integer, public :: HIPRAND_STATUS_NOT_INITIALIZED  = 101
    integer, public :: HIPRAND_STATUS_ALLOCATION_FAILED  = 102
    integer, public :: HIPRAND_STATUS_TYPE_ERROR = 103
    integer, public :: HIPRAND_STATUS_OUT_OF_RANGE  = 104
    integer, public :: HIPRAND_STATUS_LENGTH_NOT_MULTIPLE  = 105
    integer, public :: HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED  = 106
    integer, public :: HIPRAND_STATUS_LAUNCH_FAILURE  = 201
    integer, public :: HIPRAND_STATUS_PREEXISTING_FAILURE  = 202
    integer, public :: HIPRAND_STATUS_INITIALIZATION_FAILED  = 203
    integer, public :: HIPRAND_STATUS_ARCH_MISMATCH  = 204
    integer, public :: HIPRAND_STATUS_INTERNAL_ERROR  = 999
    integer, public :: HIPRAND_STATUS_NOT_IMPLEMENTED  = 1000

    interface
        function hiprandCreateGenerator(generator, rng_type) &
        bind(C, name="hiprandCreateGenerator")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandCreateGenerator
            integer(c_size_t) :: generator
            integer(c_int), value :: rng_type
        end function

        function hiprandCreateGeneratorHost(generator, rng_type) &
        bind(C, name="hiprandCreateGeneratorHost")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandCreateGeneratorHost
            integer(c_size_t) :: generator
            integer(c_int), value :: rng_type
        end function

        function hiprandDestroyGenerator(generator) &
        bind(C, name="hiprandDestroyGenerator")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandDestroyGenerator
            integer(c_size_t), value :: generator
        end function

        function hiprandGenerate(generator, output_data, n) &
        bind(C, name="hiprandGenerate")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerate
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
        end function

        function hiprandGenerateLongLong(generator, output_data, n) &
        bind(C, name="hiprandGenerateLongLong")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerateLongLong
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
        end function

        function hiprandGenerateUniform(generator, output_data, n) &
        bind(C, name="hiprandGenerateUniform")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerateUniform
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
        end function

        function hiprandGenerateUniformDouble(generator, output_data, n) &
        bind(C, name="hiprandGenerateUniformDouble")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerateUniformDouble
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
        end function

        function hiprandGenerateNormal(generator, output_data, n, mean, &
        stddev) bind(C, name="hiprandGenerateNormal")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerateNormal
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_float), value :: mean
            real(c_float), value :: stddev
        end function

        function hiprandGenerateNormalDouble(generator, output_data, n, &
        mean, stddev) bind(C, name="hiprandGenerateNormalDouble")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerateNormalDouble
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_double), value :: mean
            real(c_double), value :: stddev
        end function

        function hiprandGenerateLogNormal(generator, output_data, n, mean, &
        stddev) bind(C, name="hiprandGenerateLogNormal")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerateLogNormal
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_float), value :: mean
            real(c_float), value :: stddev
        end function

        function hiprandGenerateLogNormalDouble(generator, output_data, n, &
        mean, stddev) bind(C, name="hiprandGenerateLogNormalDouble")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerateLogNormalDouble
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_double), value :: mean
            real(c_double), value :: stddev
        end function

        function hiprandGeneratePoisson(generator, output_data, n, lambda) &
        bind(C, name="hiprandGeneratePoisson")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGeneratePoisson
            integer(c_size_t), value :: generator
            type(c_ptr), value :: output_data
            integer(c_size_t), value :: n
            real(c_double), value :: lambda
        end function

        function hiprandGenerateSeeds(generator) &
        bind(C, name="hiprandGenerateSeeds")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGenerateSeeds
            integer(c_size_t), value :: generator
        end function

        function hiprandSetStream(generator, stream) &
        bind(C, name="hiprandSetStream")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandSetStream
            integer(c_size_t), value :: generator
            integer(c_size_t), value :: stream
        end function

        function hiprandSetPseudoRandomGeneratorSeed(generator, seed) &
        bind(C, name="hiprandSetPseudoRandomGeneratorSeed")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandSetPseudoRandomGeneratorSeed
            integer(c_size_t), value :: generator
            integer(kind =8), value :: seed
        end function

        function hiprandSetGeneratorOffset(generator, offset) &
        bind(C, name="hiprandSetGeneratorOffset")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandSetGeneratorOffset
            integer(c_size_t), value :: generator
            integer(kind =8), value :: offset
        end function

        function hiprandSetQuasiRandomGeneratorDimensions(generator, &
        dimensions) bind(C, name="hiprandSetQuasiRandomGeneratorDimensions")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandSetQuasiRandomGeneratorDimensions
            integer(c_size_t), value :: generator
            integer(c_int), value :: dimensions
        end function

        function hiprandGetVersion(version) &
        bind(C, name="hiprandGetVersion")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandGetVersion
            integer(c_int) :: version
        end function

        function hiprandCreatePoissonDistribution(lambda, &
        discrete_distribution) bind(C, name="hiprandCreatePoissonDistribution")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandCreatePoissonDistribution
            real(c_double), value :: lambda
            integer(c_size_t) :: discrete_distribution
        end function

        function hiprandDestroyDistribution(discrete_distribution) &
        bind(C, name="hiprandDestroyDistribution")
            use iso_c_binding
            implicit none
            integer(c_int) :: hiprandDestroyDistribution
            integer(c_size_t), value :: discrete_distribution
        end function
    end interface
end module hiprand_m
