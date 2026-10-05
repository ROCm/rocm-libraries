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
    integer(c_int) :: size
    integer(c_int) :: offset
    type(c_ptr) :: alias
    type(c_ptr) :: probability
    type(c_ptr) :: cdf
  end type rocrand_discrete_distribution_st


  interface rocrand_create_generator
    function rocrand_create_generator_(generator,rng_type) bind(c, name="rocrand_create_generator")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_create_generator_
      type(c_ptr) :: generator
      integer(c_int),value :: rng_type
    end function
  end interface

  interface rocrand_create_generator_host
    function rocrand_create_generator_host_(generator,rng_type) &
        bind(c, name="rocrand_create_generator_host")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_create_generator_host_
      type(c_ptr) :: generator
      integer(c_int),value :: rng_type
    end function
  end interface

  interface rocrand_create_generator_host_blocking
    function rocrand_create_generator_host_blocking_(generator,rng_type) &
        bind(c, name="rocrand_create_generator_host_blocking")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_create_generator_host_blocking_
      type(c_ptr) :: generator
      integer(c_int),value :: rng_type
    end function
  end interface

  interface rocrand_destroy_generator
    function rocrand_destroy_generator_(generator) bind(c, name="rocrand_destroy_generator")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_destroy_generator_
      type(c_ptr),value :: generator
    end function
  end interface

  interface rocrand_generate
    function rocrand_generate_(generator,output_data,n) bind(c, name="rocrand_generate")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_
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
      integer(c_int) :: rocrand_generate_long_long_
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

  interface rocrand_generate_char
    function rocrand_generate_char_(generator,output_data,n) bind(c, name="rocrand_generate_char")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_char_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function
  end interface

  interface rocrand_generate_short
    function rocrand_generate_short_(generator,output_data,n) bind(c, name="rocrand_generate_short")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_short_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function
  end interface

  interface rocrand_generate_uniform
    function rocrand_generate_uniform_(generator,output_data,n) &
        bind(c, name="rocrand_generate_uniform")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_uniform_
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
      integer(c_int) :: rocrand_generate_uniform_double_
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

  interface rocrand_generate_uniform_half
    function rocrand_generate_uniform_half_(generator,output_data,n) &
        bind(c, name="rocrand_generate_uniform_half")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_uniform_half_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function
  end interface

  interface rocrand_generate_normal
    function rocrand_generate_normal_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_normal")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_normal_
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
      integer(c_int) :: rocrand_generate_normal_double_
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

  interface rocrand_generate_normal_half
    function rocrand_generate_normal_half_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_normal_half")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_normal_half_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      integer(c_short),value :: mean
      integer(c_short),value :: stddev
    end function
  end interface

  interface rocrand_generate_log_normal
    function rocrand_generate_log_normal_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_log_normal")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_log_normal_
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
      integer(c_int) :: rocrand_generate_log_normal_double_
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

  interface rocrand_generate_log_normal_half
    function rocrand_generate_log_normal_half_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_log_normal_half")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_log_normal_half_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      integer(c_short),value :: mean
      integer(c_short),value :: stddev
    end function
  end interface

  interface rocrand_generate_poisson
    function rocrand_generate_poisson_(generator,output_data,n,lambda) &
        bind(c, name="rocrand_generate_poisson")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_generate_poisson_
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

  interface rocrand_initialize_generator
    function rocrand_initialize_generator_(generator) bind(c, name="rocrand_initialize_generator")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_initialize_generator_
      type(c_ptr),value :: generator
    end function
  end interface

  interface rocrand_set_stream
    function rocrand_set_stream_(generator,stream) bind(c, name="rocrand_set_stream")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_set_stream_
      type(c_ptr),value :: generator
      type(c_ptr),value :: stream
    end function
  end interface

  interface rocrand_set_seed
    function rocrand_set_seed_(generator,seed) bind(c, name="rocrand_set_seed")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_set_seed_
      type(c_ptr),value :: generator
      integer(c_int64_t),value :: seed
    end function
  end interface

  interface rocrand_set_seed_uint4
    function rocrand_set_seed_uint4_(generator,seed) bind(c, name="rocrand_set_seed_uint4")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_set_seed_uint4_
      type(c_ptr),value :: generator
      type(uint4),value :: seed
    end function
  end interface

  interface rocrand_set_offset
    function rocrand_set_offset_(generator,offset) bind(c, name="rocrand_set_offset")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_set_offset_
      type(c_ptr),value :: generator
      integer(c_int64_t),value :: offset
    end function
  end interface

  interface rocrand_set_ordering
    function rocrand_set_ordering_(generator,order) bind(c, name="rocrand_set_ordering")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_set_ordering_
      type(c_ptr),value :: generator
      integer(c_int),value :: order
    end function
  end interface

  interface rocrand_set_quasi_random_generator_dimensions
    function rocrand_set_quasi_random_generator_dimensions_(generator,dimensions) &
        bind(c, name="rocrand_set_quasi_random_generator_dimensions")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_set_quasi_random_generator_dimensions_
      type(c_ptr),value :: generator
      integer(c_int),value :: dimensions
    end function
  end interface

  interface rocrand_get_version
    function rocrand_get_version_(version) bind(c, name="rocrand_get_version")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_get_version_
      type(c_ptr),value :: version
    end function

    module procedure rocrand_get_version_typed
  end interface

  interface rocrand_create_poisson_distribution
    function rocrand_create_poisson_distribution_(lambda,discrete_distribution) &
        bind(c, name="rocrand_create_poisson_distribution")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_create_poisson_distribution_
      real(c_double),value :: lambda
      type(c_ptr) :: discrete_distribution
    end function
  end interface

  interface rocrand_create_discrete_distribution
    function rocrand_create_discrete_distribution_(probabilities,mySize,offset, &
        discrete_distribution) &
        bind(c, name="rocrand_create_discrete_distribution")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_create_discrete_distribution_
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

  interface rocrand_destroy_discrete_distribution
    function rocrand_destroy_discrete_distribution_(discrete_distribution) &
        bind(c, name="rocrand_destroy_discrete_distribution")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_destroy_discrete_distribution_
      type(c_ptr),value :: discrete_distribution
    end function
  end interface

  interface rocrand_get_direction_vectors32
    function rocrand_get_direction_vectors32_(vectors,set) &
        bind(c, name="rocrand_get_direction_vectors32")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_get_direction_vectors32_
      type(c_ptr) :: vectors
      integer(c_int),value :: set
    end function
  end interface

  interface rocrand_get_direction_vectors64
    function rocrand_get_direction_vectors64_(vectors,set) &
        bind(c, name="rocrand_get_direction_vectors64")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_get_direction_vectors64_
      type(c_ptr) :: vectors
      integer(c_int),value :: set
    end function
  end interface

  interface rocrand_get_scramble_constants32
    function rocrand_get_scramble_constants32_(constants) &
        bind(c, name="rocrand_get_scramble_constants32")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_get_scramble_constants32_
      type(c_ptr) :: constants
    end function
  end interface

  interface rocrand_get_scramble_constants64
    function rocrand_get_scramble_constants64_(constants) &
        bind(c, name="rocrand_get_scramble_constants64")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: rocrand_get_scramble_constants64_
      type(c_ptr) :: constants
    end function
  end interface


  contains

    subroutine rocrandCheck(status)
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: status
      character(len=:), allocatable :: name
      if (status /= ROCRAND_STATUS_SUCCESS) then
        select case (status)
        case (ROCRAND_STATUS_VERSION_MISMATCH)
          name = "ROCRAND_STATUS_VERSION_MISMATCH"
        case (ROCRAND_STATUS_NOT_CREATED)
          name = "ROCRAND_STATUS_NOT_CREATED"
        case (ROCRAND_STATUS_ALLOCATION_FAILED)
          name = "ROCRAND_STATUS_ALLOCATION_FAILED"
        case (ROCRAND_STATUS_TYPE_ERROR)
          name = "ROCRAND_STATUS_TYPE_ERROR"
        case (ROCRAND_STATUS_OUT_OF_RANGE)
          name = "ROCRAND_STATUS_OUT_OF_RANGE"
        case (ROCRAND_STATUS_LENGTH_NOT_MULTIPLE)
          name = "ROCRAND_STATUS_LENGTH_NOT_MULTIPLE"
        case (ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED)
          name = "ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED"
        case (ROCRAND_STATUS_LAUNCH_FAILURE)
          name = "ROCRAND_STATUS_LAUNCH_FAILURE"
        case (ROCRAND_STATUS_INTERNAL_ERROR)
          name = "ROCRAND_STATUS_INTERNAL_ERROR"
        case default
          name = "unknown status"
        end select
        write (*, "(a,i0,a)") "ROCRAND ERROR: " // name // " (code ", status, ")"
        stop 1
      end if
    end subroutine rocrandCheck
    function rocrand_get_version_typed(version)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_get_version_typed
      integer(c_int),target :: version
      !
      rocrand_get_version_typed = rocrand_get_version_(c_loc(version))
    end function


#if defined(USE_ASSUMED_SHAPE) || defined(USE_ASSUMED_RANK)

#ifdef USE_ASSUMED_RANK
    function rocrand_generate_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_assumed_rank
      type(c_ptr), value :: generator
      integer(c_int),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_assumed_rank = rocrand_generate_(generator,c_loc(output_data),n)
    end function

#else
    function rocrand_generate_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_rank_0
      type(c_ptr), value :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_rank_0 = rocrand_generate_(generator,c_loc(output_data),n)
    end function

    function rocrand_generate_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_rank_1
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
      integer(c_int64_t),target,dimension(..) :: output_data
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
      real(c_float),target,dimension(..) :: output_data
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
      type(c_ptr), value :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_rank_0 = rocrand_generate_uniform_(generator,c_loc(output_data),n)
    end function

    function rocrand_generate_uniform_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_rank_1
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
      real(c_double),target,dimension(..) :: output_data
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
      real(c_float),target,dimension(..) :: output_data
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
      real(c_double),target,dimension(..) :: output_data
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
      real(c_float),target,dimension(..) :: output_data
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
      real(c_double),target,dimension(..) :: output_data
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
      integer(c_int),target,dimension(..) :: output_data
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
      type(c_ptr), value :: generator
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
      type(c_ptr), value :: generator
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
      real(c_double),target,dimension(..) :: probabilities
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
