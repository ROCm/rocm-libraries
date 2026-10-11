!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
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

module rocrand
  use, intrinsic :: iso_c_binding
  implicit none

  ! rocrand_status
  enum, bind(c)
    enumerator :: ROCRAND_STATUS_SUCCESS = 0
    enumerator :: ROCRAND_STATUS_VERSION_MISMATCH = 100
    enumerator :: ROCRAND_STATUS_NOT_CREATED = 101
    enumerator :: ROCRAND_STATUS_ALLOCATION_FAILED = 102
    enumerator :: ROCRAND_STATUS_TYPE_ERROR = 103
    enumerator :: ROCRAND_STATUS_OUT_OF_RANGE = 104
    enumerator :: ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = 105
    enumerator :: ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
    enumerator :: ROCRAND_STATUS_LAUNCH_FAILURE = 107
    enumerator :: ROCRAND_STATUS_INTERNAL_ERROR = 108
  end enum

  ! rocrand_rng_type
  enum, bind(c)
    enumerator :: ROCRAND_RNG_PSEUDO_DEFAULT = 400
    enumerator :: ROCRAND_RNG_PSEUDO_XORWOW = 401
    enumerator :: ROCRAND_RNG_PSEUDO_MRG32K3A = 402
    enumerator :: ROCRAND_RNG_PSEUDO_MTGP32 = 403
    enumerator :: ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = 404
    enumerator :: ROCRAND_RNG_PSEUDO_MRG31K3P = 405
    enumerator :: ROCRAND_RNG_PSEUDO_LFSR113 = 406
    enumerator :: ROCRAND_RNG_PSEUDO_MT19937 = 407
    enumerator :: ROCRAND_RNG_PSEUDO_THREEFRY2_32_20 = 408
    enumerator :: ROCRAND_RNG_PSEUDO_THREEFRY2_64_20 = 409
    enumerator :: ROCRAND_RNG_PSEUDO_THREEFRY4_32_20 = 410
    enumerator :: ROCRAND_RNG_PSEUDO_THREEFRY4_64_20 = 411
    enumerator :: ROCRAND_RNG_QUASI_DEFAULT = 500
    enumerator :: ROCRAND_RNG_QUASI_SOBOL32 = 501
    enumerator :: ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502
    enumerator :: ROCRAND_RNG_QUASI_SOBOL64 = 504
    enumerator :: ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 505
  end enum

  ! rocrand_ordering
  enum, bind(c)
    enumerator :: ROCRAND_ORDERING_PSEUDO_BEST = 100
    enumerator :: ROCRAND_ORDERING_PSEUDO_DEFAULT = 101
    enumerator :: ROCRAND_ORDERING_PSEUDO_SEEDED = 102
    enumerator :: ROCRAND_ORDERING_PSEUDO_LEGACY = 103
    enumerator :: ROCRAND_ORDERING_PSEUDO_DYNAMIC = 104
    enumerator :: ROCRAND_ORDERING_QUASI_DEFAULT = 201
  end enum

  ! rocrand_direction_vector_set
  enum, bind(c)
    enumerator :: ROCRAND_DIRECTION_VECTORS_32_JOEKUO6 = 101
    enumerator :: ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102
    enumerator :: ROCRAND_DIRECTION_VECTORS_64_JOEKUO6 = 103
    enumerator :: ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
  end enum

  integer(c_int), parameter :: ROCRAND_VERSION = 500200
  integer(c_int), parameter :: ROCRAND_DEFAULT_MAX_BLOCK_SIZE = 256


  type, bind(c) :: uint4
    integer(c_int) :: x
    integer(c_int) :: y
    integer(c_int) :: z
    integer(c_int) :: w
  end type uint4

  type, bind(c) :: rocrand_discrete_distribution_st
    integer(c_int) :: size !< Number of entries in the probability table
    integer(c_int) :: offset !< The distribution can be offset
    type(c_ptr) :: alias !< Alias table
    type(c_ptr) :: probability !< Probability data for the alias table
    type(c_ptr) :: cdf !< Cumulative distribution function
  end type rocrand_discrete_distribution_st


  interface

    !---------------------------------------------
    ! rocrand_create_generator
    !---------------------------------------------
    function rocrand_create_generator(generator, rng_type) &
       result(create_generator) &
       bind(C, name="rocrand_create_generator")
       import :: c_ptr, ROCRAND_RNG_PSEUDO_DEFAULT, ROCRAND_STATUS_SUCCESS
       type(c_ptr) :: generator
       integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)), value :: rng_type
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: create_generator
    end function rocrand_create_generator

    !---------------------------------------------
    ! rocrand_create_generator_host
    !---------------------------------------------
    function rocrand_create_generator_host(generator, rng_type) &
       result(create_generator_host) &
       bind(C, name="rocrand_create_generator_host")
       import :: c_ptr, ROCRAND_RNG_PSEUDO_DEFAULT, ROCRAND_STATUS_SUCCESS
       type(c_ptr) :: generator
       integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)), value :: rng_type
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: create_generator_host
    end function rocrand_create_generator_host

    !---------------------------------------------
    ! rocrand_create_generator_host_blocking
    !---------------------------------------------
    function rocrand_create_generator_host_blocking(generator, rng_type) &
       result(create_generator_host_blocking) &
       bind(C, name="rocrand_create_generator_host_blocking")
       import :: c_ptr, ROCRAND_RNG_PSEUDO_DEFAULT, ROCRAND_STATUS_SUCCESS
       type(c_ptr) :: generator
       integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)), value :: rng_type
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: create_generator_host_blocking
    end function rocrand_create_generator_host_blocking

    !---------------------------------------------
    ! rocrand_destroy_generator
    !---------------------------------------------
    function rocrand_destroy_generator(generator) &
       result(destroy_generator) &
       bind(C, name="rocrand_destroy_generator")
       import :: c_ptr, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: destroy_generator
    end function rocrand_destroy_generator

    !---------------------------------------------
    ! rocrand_generate_char
    !---------------------------------------------
    function rocrand_generate_char(generator, output_data, n) &
       result(generate_char) &
       bind(C, name="rocrand_generate_char")
       import :: c_ptr, c_size_t, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       type(c_ptr), value :: output_data
       integer(c_size_t), value :: n
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_char
    end function rocrand_generate_char

    !---------------------------------------------
    ! rocrand_generate_short
    !---------------------------------------------
    function rocrand_generate_short(generator, output_data, n) &
       result(generate_short) &
       bind(C, name="rocrand_generate_short")
       import :: c_ptr, c_size_t, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       type(c_ptr), value :: output_data
       integer(c_size_t), value :: n
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_short
    end function rocrand_generate_short

    !---------------------------------------------
    ! rocrand_generate_uniform_half
    !---------------------------------------------
    function rocrand_generate_uniform_half(generator, output_data, n) &
       result(generate_uniform_half) &
       bind(C, name="rocrand_generate_uniform_half")
       import :: c_ptr, c_size_t, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       type(c_ptr), value :: output_data
       integer(c_size_t), value :: n
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_uniform_half
    end function rocrand_generate_uniform_half

    !---------------------------------------------
    ! rocrand_generate_normal_half
    !---------------------------------------------
    function rocrand_generate_normal_half(generator, output_data, n, mean, stddev) &
       result(generate_normal_half) &
       bind(C, name="rocrand_generate_normal_half")
       import :: c_ptr, c_size_t, c_short, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       type(c_ptr), value :: output_data
       integer(c_size_t), value :: n
       integer(c_short), value :: mean
       integer(c_short), value :: stddev
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_normal_half
    end function rocrand_generate_normal_half

    !---------------------------------------------
    ! rocrand_generate_log_normal_half
    !---------------------------------------------
    function rocrand_generate_log_normal_half(generator, output_data, n, mean, stddev) &
       result(generate_log_normal_half) &
       bind(C, name="rocrand_generate_log_normal_half")
       import :: c_ptr, c_size_t, c_short, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       type(c_ptr), value :: output_data
       integer(c_size_t), value :: n
       integer(c_short), value :: mean
       integer(c_short), value :: stddev
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_log_normal_half
    end function rocrand_generate_log_normal_half

    !---------------------------------------------
    ! rocrand_initialize_generator
    !---------------------------------------------
    function rocrand_initialize_generator(generator) &
       result(initialize_generator) &
       bind(C, name="rocrand_initialize_generator")
       import :: c_ptr, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: initialize_generator
    end function rocrand_initialize_generator

    !---------------------------------------------
    ! rocrand_set_stream
    !---------------------------------------------
    function rocrand_set_stream(generator, stream) &
       result(set_stream) &
       bind(C, name="rocrand_set_stream")
       import :: c_ptr, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       type(c_ptr), value :: stream
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_stream
    end function rocrand_set_stream

    !---------------------------------------------
    ! rocrand_set_seed
    !---------------------------------------------
    function rocrand_set_seed(generator, seed) &
       result(set_seed) &
       bind(C, name="rocrand_set_seed")
       import :: c_ptr, c_int64_t, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       integer(c_int64_t), value :: seed
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_seed
    end function rocrand_set_seed

    !---------------------------------------------
    ! rocrand_set_seed_uint4
    !---------------------------------------------
    function rocrand_set_seed_uint4(generator, seed) &
       result(set_seed_uint4) &
       bind(C, name="rocrand_set_seed_uint4")
       import :: c_ptr, uint4, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       type(uint4), value :: seed
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_seed_uint4
    end function rocrand_set_seed_uint4

    !---------------------------------------------
    ! rocrand_set_offset
    !---------------------------------------------
    function rocrand_set_offset(generator, offset) &
       result(set_offset) &
       bind(C, name="rocrand_set_offset")
       import :: c_ptr, c_int64_t, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       integer(c_int64_t), value :: offset
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_offset
    end function rocrand_set_offset

    !---------------------------------------------
    ! rocrand_set_ordering
    !---------------------------------------------
    function rocrand_set_ordering(generator, order) &
       result(set_ordering) &
       bind(C, name="rocrand_set_ordering")
       import :: c_ptr, ROCRAND_ORDERING_PSEUDO_BEST, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       integer(kind(ROCRAND_ORDERING_PSEUDO_BEST)), value :: order
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_ordering
    end function rocrand_set_ordering

    !---------------------------------------------
    ! rocrand_set_quasi_random_generator_dimensions
    !---------------------------------------------
    function rocrand_set_quasi_random_generator_dimensions(generator, dimensions) &
       result(set_quasi_random_generator_dimensions) &
       bind(C, name="rocrand_set_quasi_random_generator_dimensions")
       import :: c_ptr, c_int, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: generator
       integer(c_int), value :: dimensions
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_quasi_random_generator_dimensions
    end function rocrand_set_quasi_random_generator_dimensions

    !---------------------------------------------
    ! rocrand_get_version
    !---------------------------------------------
    function rocrand_get_version(version) &
       result(get_version) &
       bind(C, name="rocrand_get_version")
       import :: c_int, ROCRAND_STATUS_SUCCESS
       integer(c_int) :: version
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: get_version
    end function rocrand_get_version

    !---------------------------------------------
    ! rocrand_create_poisson_distribution
    !---------------------------------------------
    function rocrand_create_poisson_distribution(lambda, discrete_distribution) &
       result(create_poisson_distribution) &
       bind(C, name="rocrand_create_poisson_distribution")
       import :: c_double, c_ptr, ROCRAND_STATUS_SUCCESS
       real(c_double), value :: lambda
       type(c_ptr) :: discrete_distribution
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: create_poisson_distribution
    end function rocrand_create_poisson_distribution

    !---------------------------------------------
    ! rocrand_destroy_discrete_distribution
    !---------------------------------------------
    function rocrand_destroy_discrete_distribution(discrete_distribution) &
       result(destroy_discrete_distribution) &
       bind(C, name="rocrand_destroy_discrete_distribution")
       import :: c_ptr, ROCRAND_STATUS_SUCCESS
       type(c_ptr), value :: discrete_distribution
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: destroy_discrete_distribution
    end function rocrand_destroy_discrete_distribution

    !---------------------------------------------
    ! rocrand_get_direction_vectors32
    !---------------------------------------------
    function rocrand_get_direction_vectors32(vectors, set) &
       result(get_direction_vectors32) &
       bind(C, name="rocrand_get_direction_vectors32")
       import :: c_ptr, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6, ROCRAND_STATUS_SUCCESS
       type(c_ptr) :: vectors
       integer(kind(ROCRAND_DIRECTION_VECTORS_32_JOEKUO6)), value :: set
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: get_direction_vectors32
    end function rocrand_get_direction_vectors32

    !---------------------------------------------
    ! rocrand_get_direction_vectors64
    !---------------------------------------------
    function rocrand_get_direction_vectors64(vectors, set) &
       result(get_direction_vectors64) &
       bind(C, name="rocrand_get_direction_vectors64")
       import :: c_ptr, ROCRAND_DIRECTION_VECTORS_32_JOEKUO6, ROCRAND_STATUS_SUCCESS
       type(c_ptr) :: vectors
       integer(kind(ROCRAND_DIRECTION_VECTORS_32_JOEKUO6)), value :: set
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: get_direction_vectors64
    end function rocrand_get_direction_vectors64

    !---------------------------------------------
    ! rocrand_get_scramble_constants32
    !---------------------------------------------
    function rocrand_get_scramble_constants32(constants) &
       result(get_scramble_constants32) &
       bind(C, name="rocrand_get_scramble_constants32")
       import :: c_ptr, ROCRAND_STATUS_SUCCESS
       type(c_ptr) :: constants
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: get_scramble_constants32
    end function rocrand_get_scramble_constants32

    !---------------------------------------------
    ! rocrand_get_scramble_constants64
    !---------------------------------------------
    function rocrand_get_scramble_constants64(constants) &
       result(get_scramble_constants64) &
       bind(C, name="rocrand_get_scramble_constants64")
       import :: c_ptr, ROCRAND_STATUS_SUCCESS
       type(c_ptr) :: constants
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: get_scramble_constants64
    end function rocrand_get_scramble_constants64

  end interface

  interface rocrand_generate
    function rocrand_generate_(generator,output_data,n) bind(c, name="rocrand_generate")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_rank_0,&
      rocrand_generate_rank_1
#endif
#endif
  end interface

  interface rocrand_generate_long_long
    function rocrand_generate_long_long_(generator,output_data,n) &
        bind(c, name="rocrand_generate_long_long")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_long_long_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_long_long_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_long_long_rank_0,&
      rocrand_generate_long_long_rank_1
#endif
#endif
  end interface

  interface rocrand_generate_uniform
    function rocrand_generate_uniform_(generator,output_data,n) &
        bind(c, name="rocrand_generate_uniform")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_uniform_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_uniform_rank_0,&
      rocrand_generate_uniform_rank_1
#endif
#endif
  end interface

  interface rocrand_generate_uniform_double
    function rocrand_generate_uniform_double_(generator,output_data,n) &
        bind(c, name="rocrand_generate_uniform_double")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_double_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_uniform_double_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_uniform_double_rank_0,&
      rocrand_generate_uniform_double_rank_1
#endif
#endif
  end interface

  interface rocrand_generate_normal
    function rocrand_generate_normal_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_normal")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_float),value :: mean
      real(c_float),value :: stddev
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_normal_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_normal_rank_0,&
      rocrand_generate_normal_rank_1
#endif
#endif
  end interface

  interface rocrand_generate_normal_double
    function rocrand_generate_normal_double_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_normal_double")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_double_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: mean
      real(c_double),value :: stddev
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_normal_double_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_normal_double_rank_0,&
      rocrand_generate_normal_double_rank_1
#endif
#endif
  end interface

  interface rocrand_generate_log_normal
    function rocrand_generate_log_normal_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_log_normal")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_float),value :: mean
      real(c_float),value :: stddev
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_log_normal_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_log_normal_rank_0,&
      rocrand_generate_log_normal_rank_1
#endif
#endif
  end interface

  interface rocrand_generate_log_normal_double
    function rocrand_generate_log_normal_double_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_log_normal_double")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_double_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: mean
      real(c_double),value :: stddev
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_log_normal_double_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_log_normal_double_rank_0,&
      rocrand_generate_log_normal_double_rank_1
#endif
#endif
  end interface

  interface rocrand_generate_poisson
    function rocrand_generate_poisson_(generator,output_data,n,lambda) &
        bind(c, name="rocrand_generate_poisson")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_poisson_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: lambda
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_generate_poisson_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_generate_poisson_rank_0,&
      rocrand_generate_poisson_rank_1
#endif
#endif
  end interface

  interface rocrand_create_discrete_distribution
    function rocrand_create_discrete_distribution_(probabilities,mySize,offset, &
        discrete_distribution) &
        bind(c, name="rocrand_create_discrete_distribution")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_discrete_distribution_
      type(c_ptr),value :: probabilities
      integer(c_int),value :: mySize
      integer(c_int),value :: offset
      type(c_ptr) :: discrete_distribution
    end function

#ifdef USE_ASSUMED_RANK
    module procedure rocrand_create_discrete_distribution_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      rocrand_create_discrete_distribution_rank_0,&
      rocrand_create_discrete_distribution_rank_1
#endif
#endif
  end interface


  contains

    subroutine rocrandCheck(status)
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: status
      if (status /= ROCRAND_STATUS_SUCCESS) then
        write (*, *) "ROCRAND ERROR: code = ", status
        stop 1
      end if
    end subroutine rocrandCheck

#if defined(USE_ASSUMED_SHAPE) || defined(USE_ASSUMED_RANK)

#ifdef USE_ASSUMED_RANK
    function rocrand_generate_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_assumed_rank
      type(c_ptr) :: generator
      integer(c_int),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_assumed_rank = rocrand_generate_(generator,c_loc(output_data),n)
    end function

#else
    function rocrand_generate_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_rank_0
      type(c_ptr) :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_rank_0 = rocrand_generate_(generator,c_loc(output_data),n)
    end function

    function rocrand_generate_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_rank_1
      type(c_ptr) :: generator
      integer(c_int),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_rank_1 = rocrand_generate_(generator,c_loc(output_data),n)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_generate_long_long_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_long_long_assumed_rank
      type(c_ptr) :: generator
      integer(c_int64_t),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_long_long_assumed_rank = rocrand_generate_long_long_(generator, &
        c_loc(output_data),n)
    end function

#else
    function rocrand_generate_long_long_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_long_long_rank_0
      type(c_ptr) :: generator
      integer(c_int64_t),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_long_long_rank_0 = rocrand_generate_long_long_(generator, &
        c_loc(output_data),n)
    end function

    function rocrand_generate_long_long_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_long_long_rank_1
      type(c_ptr) :: generator
      integer(c_int64_t),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_long_long_rank_1 = rocrand_generate_long_long_(generator, &
        c_loc(output_data),n)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_generate_uniform_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_assumed_rank
      type(c_ptr) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_assumed_rank = rocrand_generate_uniform_(generator, &
        c_loc(output_data),n)
    end function

#else
    function rocrand_generate_uniform_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_rank_0
      type(c_ptr) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_rank_0 = rocrand_generate_uniform_(generator,c_loc(output_data),n)
    end function

    function rocrand_generate_uniform_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_rank_1
      type(c_ptr) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_rank_1 = rocrand_generate_uniform_(generator,c_loc(output_data),n)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_generate_uniform_double_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_double_assumed_rank
      type(c_ptr) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_double_assumed_rank = rocrand_generate_uniform_double_(generator, &
        c_loc(output_data),n)
    end function

#else
    function rocrand_generate_uniform_double_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_double_rank_0
      type(c_ptr) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_double_rank_0 = rocrand_generate_uniform_double_(generator, &
        c_loc(output_data),n)
    end function

    function rocrand_generate_uniform_double_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_double_rank_1
      type(c_ptr) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_double_rank_1 = rocrand_generate_uniform_double_(generator, &
        c_loc(output_data),n)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_generate_normal_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_assumed_rank
      type(c_ptr) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_normal_assumed_rank = rocrand_generate_normal_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function rocrand_generate_normal_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_rank_0
      type(c_ptr) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_normal_rank_0 = rocrand_generate_normal_(generator,c_loc(output_data),n, &
        mean,stddev)
    end function

    function rocrand_generate_normal_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_rank_1
      type(c_ptr) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_normal_rank_1 = rocrand_generate_normal_(generator,c_loc(output_data),n, &
        mean,stddev)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_generate_normal_double_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_double_assumed_rank
      type(c_ptr) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_normal_double_assumed_rank = rocrand_generate_normal_double_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function rocrand_generate_normal_double_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_double_rank_0
      type(c_ptr) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_normal_double_rank_0 = rocrand_generate_normal_double_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

    function rocrand_generate_normal_double_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_double_rank_1
      type(c_ptr) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_normal_double_rank_1 = rocrand_generate_normal_double_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_generate_log_normal_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_assumed_rank
      type(c_ptr) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_log_normal_assumed_rank = rocrand_generate_log_normal_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function rocrand_generate_log_normal_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_rank_0
      type(c_ptr) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_log_normal_rank_0 = rocrand_generate_log_normal_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

    function rocrand_generate_log_normal_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_rank_1
      type(c_ptr) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_log_normal_rank_1 = rocrand_generate_log_normal_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_generate_log_normal_double_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_double_assumed_rank
      type(c_ptr) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_log_normal_double_assumed_rank = rocrand_generate_log_normal_double_( &
        generator,c_loc(output_data),n,mean,stddev)
    end function

#else
    function rocrand_generate_log_normal_double_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_double_rank_0
      type(c_ptr) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_log_normal_double_rank_0 = rocrand_generate_log_normal_double_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

    function rocrand_generate_log_normal_double_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_double_rank_1
      type(c_ptr) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_log_normal_double_rank_1 = rocrand_generate_log_normal_double_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_generate_poisson_assumed_rank(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_poisson_assumed_rank
      type(c_ptr) :: generator
      integer(c_int),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      rocrand_generate_poisson_assumed_rank = rocrand_generate_poisson_(generator, &
        c_loc(output_data),n,lambda)
    end function

#else
    function rocrand_generate_poisson_rank_0(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_poisson_rank_0
      type(c_ptr) :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      rocrand_generate_poisson_rank_0 = rocrand_generate_poisson_(generator,c_loc(output_data),n, &
        lambda)
    end function

    function rocrand_generate_poisson_rank_1(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_poisson_rank_1
      type(c_ptr) :: generator
      integer(c_int),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      rocrand_generate_poisson_rank_1 = rocrand_generate_poisson_(generator,c_loc(output_data),n, &
        lambda)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function rocrand_create_discrete_distribution_assumed_rank(probabilities,mySize,offset, &
        discrete_distribution)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_discrete_distribution_assumed_rank
      real(c_double),target,contiguous,dimension(..) :: probabilities
      integer(c_int) :: mySize
      integer(c_int) :: offset
      type(c_ptr) :: discrete_distribution
      !
      rocrand_create_discrete_distribution_assumed_rank = rocrand_create_discrete_distribution_( &
        c_loc(probabilities),mySize,offset,discrete_distribution)
    end function

#else
    function rocrand_create_discrete_distribution_rank_0(probabilities,mySize,offset, &
        discrete_distribution)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_discrete_distribution_rank_0
      real(c_double),target :: probabilities
      integer(c_int) :: mySize
      integer(c_int) :: offset
      type(c_ptr) :: discrete_distribution
      !
      rocrand_create_discrete_distribution_rank_0 = rocrand_create_discrete_distribution_(c_loc( &
        probabilities),mySize,offset,discrete_distribution)
    end function

    function rocrand_create_discrete_distribution_rank_1(probabilities,mySize,offset, &
        discrete_distribution)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_discrete_distribution_rank_1
      real(c_double),target,dimension(:) :: probabilities
      integer(c_int) :: mySize
      integer(c_int) :: offset
      type(c_ptr) :: discrete_distribution
      !
      rocrand_create_discrete_distribution_rank_1 = rocrand_create_discrete_distribution_(c_loc( &
        probabilities),mySize,offset,discrete_distribution)
    end function

#endif
#endif
end module rocrand
