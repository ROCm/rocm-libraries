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

module hiprand
  use, intrinsic :: iso_c_binding
  implicit none

  ! hiprandStatus
  enum, bind(c)
    enumerator :: HIPRAND_STATUS_SUCCESS = 0
    enumerator :: HIPRAND_STATUS_VERSION_MISMATCH = 100
    enumerator :: HIPRAND_STATUS_NOT_INITIALIZED = 101
    enumerator :: HIPRAND_STATUS_ALLOCATION_FAILED = 102
    enumerator :: HIPRAND_STATUS_TYPE_ERROR = 103
    enumerator :: HIPRAND_STATUS_OUT_OF_RANGE = 104
    enumerator :: HIPRAND_STATUS_LENGTH_NOT_MULTIPLE = 105
    enumerator :: HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
    enumerator :: HIPRAND_STATUS_LAUNCH_FAILURE = 201
    enumerator :: HIPRAND_STATUS_PREEXISTING_FAILURE = 202
    enumerator :: HIPRAND_STATUS_INITIALIZATION_FAILED = 203
    enumerator :: HIPRAND_STATUS_ARCH_MISMATCH = 204
    enumerator :: HIPRAND_STATUS_INTERNAL_ERROR = 999
    enumerator :: HIPRAND_STATUS_NOT_IMPLEMENTED = 1000
  end enum

  ! hiprandRngType
  enum, bind(c)
    enumerator :: HIPRAND_RNG_TEST = 0
    enumerator :: HIPRAND_RNG_PSEUDO_DEFAULT = 400
    enumerator :: HIPRAND_RNG_PSEUDO_XORWOW = 401
    enumerator :: HIPRAND_RNG_PSEUDO_MRG32K3A = 402
    enumerator :: HIPRAND_RNG_PSEUDO_MTGP32 = 403
    enumerator :: HIPRAND_RNG_PSEUDO_MT19937 = 404
    enumerator :: HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 405
    enumerator :: HIPRAND_RNG_QUASI_DEFAULT = 500
    enumerator :: HIPRAND_RNG_QUASI_SOBOL32 = 501
    enumerator :: HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502
    enumerator :: HIPRAND_RNG_QUASI_SOBOL64 = 503
    enumerator :: HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 504
  end enum

  ! hiprandOrdering
  enum, bind(c)
    enumerator :: HIPRAND_ORDERING_PSEUDO_BEST = 100
    enumerator :: HIPRAND_ORDERING_PSEUDO_DEFAULT = 101
    enumerator :: HIPRAND_ORDERING_PSEUDO_SEEDED = 102
    enumerator :: HIPRAND_ORDERING_PSEUDO_LEGACY = 103
    enumerator :: HIPRAND_ORDERING_PSEUDO_DYNAMIC = 104
    enumerator :: HIPRAND_ORDERING_QUASI_DEFAULT = 201
  end enum

  ! hiprandDirectionVectorSet
  enum, bind(c)
    enumerator :: HIPRAND_DIRECTION_VECTORS_32_JOEKUO6 = 101
    enumerator :: HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102
    enumerator :: HIPRAND_DIRECTION_VECTORS_64_JOEKUO6 = 103
    enumerator :: HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104
  end enum

  integer(c_int), parameter :: HIPRAND_VERSION = 300400
  integer(c_int), parameter :: HIPRAND_DEFAULT_MAX_BLOCK_SIZE = 256
  integer(c_int), parameter :: HIPRAND_DEFAULT_MIN_WARPS_PER_EU = 1


  interface hiprandCreateGenerator
    function hiprandCreateGenerator_(generator,rng_type) bind(c, name="hiprandCreateGenerator")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandCreateGenerator_
      type(c_ptr) :: generator
      integer(c_int),value :: rng_type
    end function
  end interface

  interface hiprandCreateGeneratorHost
    function hiprandCreateGeneratorHost_(generator,rng_type) &
        bind(c, name="hiprandCreateGeneratorHost")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandCreateGeneratorHost_
      type(c_ptr) :: generator
      integer(c_int),value :: rng_type
    end function
  end interface

  interface hiprandDestroyGenerator
    function hiprandDestroyGenerator_(generator) bind(c, name="hiprandDestroyGenerator")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandDestroyGenerator_
      type(c_ptr),value :: generator
    end function
  end interface

  interface hiprandGenerate
    function hiprandGenerate_(generator,output_data,n) bind(c, name="hiprandGenerate")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerate_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGenerate_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGenerate_rank_0,&
      hiprandGenerate_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateChar
    function hiprandGenerateChar_(generator,output_data,n) bind(c, name="hiprandGenerateChar")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateChar_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function
  end interface

  interface hiprandGenerateShort
    function hiprandGenerateShort_(generator,output_data,n) bind(c, name="hiprandGenerateShort")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateShort_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function
  end interface

  interface hiprandGenerateLongLong
    function hiprandGenerateLongLong_(generator,output_data,n) &
        bind(c, name="hiprandGenerateLongLong")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateLongLong_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGenerateLongLong_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGenerateLongLong_rank_0,&
      hiprandGenerateLongLong_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateUniform
    function hiprandGenerateUniform_(generator,output_data,n) bind(c, name="hiprandGenerateUniform")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateUniform_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGenerateUniform_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGenerateUniform_rank_0,&
      hiprandGenerateUniform_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateUniformDouble
    function hiprandGenerateUniformDouble_(generator,output_data,n) &
        bind(c, name="hiprandGenerateUniformDouble")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateUniformDouble_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGenerateUniformDouble_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGenerateUniformDouble_rank_0,&
      hiprandGenerateUniformDouble_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateUniformHalf
    function hiprandGenerateUniformHalf_(generator,output_data,n) &
        bind(c, name="hiprandGenerateUniformHalf")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateUniformHalf_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function
  end interface

  interface hiprandGenerateNormal
    function hiprandGenerateNormal_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateNormal")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateNormal_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_float),value :: mean
      real(c_float),value :: stddev
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGenerateNormal_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGenerateNormal_rank_0,&
      hiprandGenerateNormal_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateNormalDouble
    function hiprandGenerateNormalDouble_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateNormalDouble")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateNormalDouble_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: mean
      real(c_double),value :: stddev
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGenerateNormalDouble_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGenerateNormalDouble_rank_0,&
      hiprandGenerateNormalDouble_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateNormalHalf
    function hiprandGenerateNormalHalf_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateNormalHalf")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateNormalHalf_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      integer(c_short),value :: mean
      integer(c_short),value :: stddev
    end function
  end interface

  interface hiprandGenerateLogNormal
    function hiprandGenerateLogNormal_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateLogNormal")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateLogNormal_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_float),value :: mean
      real(c_float),value :: stddev
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGenerateLogNormal_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGenerateLogNormal_rank_0,&
      hiprandGenerateLogNormal_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateLogNormalDouble
    function hiprandGenerateLogNormalDouble_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateLogNormalDouble")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateLogNormalDouble_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: mean
      real(c_double),value :: stddev
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGenerateLogNormalDouble_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGenerateLogNormalDouble_rank_0,&
      hiprandGenerateLogNormalDouble_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateLogNormalHalf
    function hiprandGenerateLogNormalHalf_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateLogNormalHalf")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateLogNormalHalf_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      integer(c_short),value :: mean
      integer(c_short),value :: stddev
    end function
  end interface

  interface hiprandGeneratePoisson
    function hiprandGeneratePoisson_(generator,output_data,n,lambda) &
        bind(c, name="hiprandGeneratePoisson")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGeneratePoisson_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: lambda
    end function

#ifdef USE_ASSUMED_RANK
    module procedure hiprandGeneratePoisson_assumed_rank
#else
#ifdef USE_ASSUMED_SHAPE
    module procedure &
      hiprandGeneratePoisson_rank_0,&
      hiprandGeneratePoisson_rank_1
#endif
#endif
  end interface

  interface hiprandGenerateSeeds
    function hiprandGenerateSeeds_(generator) bind(c, name="hiprandGenerateSeeds")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGenerateSeeds_
      type(c_ptr),value :: generator
    end function
  end interface

  interface hiprandSetStream
    function hiprandSetStream_(generator,stream) bind(c, name="hiprandSetStream")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandSetStream_
      type(c_ptr),value :: generator
      type(c_ptr),value :: stream
    end function
  end interface

  interface hiprandSetPseudoRandomGeneratorSeed
    function hiprandSetPseudoRandomGeneratorSeed_(generator,seed) &
        bind(c, name="hiprandSetPseudoRandomGeneratorSeed")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandSetPseudoRandomGeneratorSeed_
      type(c_ptr),value :: generator
      integer(c_int64_t),value :: seed
    end function
  end interface

  interface hiprandSetGeneratorOffset
    function hiprandSetGeneratorOffset_(generator,offset) bind(c, name="hiprandSetGeneratorOffset")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandSetGeneratorOffset_
      type(c_ptr),value :: generator
      integer(c_int64_t),value :: offset
    end function
  end interface

  interface hiprandSetGeneratorOrdering
    function hiprandSetGeneratorOrdering_(generator,order) &
        bind(c, name="hiprandSetGeneratorOrdering")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandSetGeneratorOrdering_
      type(c_ptr),value :: generator
      integer(c_int),value :: order
    end function
  end interface

  interface hiprandSetQuasiRandomGeneratorDimensions
    function hiprandSetQuasiRandomGeneratorDimensions_(generator,dimensions) &
        bind(c, name="hiprandSetQuasiRandomGeneratorDimensions")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandSetQuasiRandomGeneratorDimensions_
      type(c_ptr),value :: generator
      integer(c_int),value :: dimensions
    end function
  end interface

  interface hiprandGetVersion
    function hiprandGetVersion_(version) bind(c, name="hiprandGetVersion")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGetVersion_
      type(c_ptr),value :: version
    end function

    module procedure hiprandGetVersion_typed
  end interface

  interface hiprandCreatePoissonDistribution
    function hiprandCreatePoissonDistribution_(lambda,discrete_distribution) &
        bind(c, name="hiprandCreatePoissonDistribution")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandCreatePoissonDistribution_
      real(c_double),value :: lambda
      type(c_ptr) :: discrete_distribution
    end function
  end interface

  interface hiprandDestroyDistribution
    function hiprandDestroyDistribution_(discrete_distribution) &
        bind(c, name="hiprandDestroyDistribution")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandDestroyDistribution_
      type(c_ptr),value :: discrete_distribution
    end function
  end interface

  interface hiprandGetDirectionVectors32
    function hiprandGetDirectionVectors32_(vectors,set) bind(c, name="hiprandGetDirectionVectors32")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGetDirectionVectors32_
      type(c_ptr) :: vectors
      integer(c_int),value :: set
    end function
  end interface

  interface hiprandGetDirectionVectors64
    function hiprandGetDirectionVectors64_(vectors,set) bind(c, name="hiprandGetDirectionVectors64")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGetDirectionVectors64_
      type(c_ptr) :: vectors
      integer(c_int),value :: set
    end function
  end interface

  interface hiprandGetScrambleConstants32
    function hiprandGetScrambleConstants32_(constants) bind(c, name="hiprandGetScrambleConstants32")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGetScrambleConstants32_
      type(c_ptr) :: constants
    end function
  end interface

  interface hiprandGetScrambleConstants64
    function hiprandGetScrambleConstants64_(constants) bind(c, name="hiprandGetScrambleConstants64")
      use iso_c_binding
      import
      implicit none
      integer(c_int) :: hiprandGetScrambleConstants64_
      type(c_ptr) :: constants
    end function
  end interface


  contains

    subroutine hiprandCheck(status)
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: status
      character(len=:), allocatable :: name
      if (status /= HIPRAND_STATUS_SUCCESS) then
        select case (status)
        case (HIPRAND_STATUS_VERSION_MISMATCH)
          name = "HIPRAND_STATUS_VERSION_MISMATCH"
        case (HIPRAND_STATUS_NOT_INITIALIZED)
          name = "HIPRAND_STATUS_NOT_INITIALIZED"
        case (HIPRAND_STATUS_ALLOCATION_FAILED)
          name = "HIPRAND_STATUS_ALLOCATION_FAILED"
        case (HIPRAND_STATUS_TYPE_ERROR)
          name = "HIPRAND_STATUS_TYPE_ERROR"
        case (HIPRAND_STATUS_OUT_OF_RANGE)
          name = "HIPRAND_STATUS_OUT_OF_RANGE"
        case (HIPRAND_STATUS_LENGTH_NOT_MULTIPLE)
          name = "HIPRAND_STATUS_LENGTH_NOT_MULTIPLE"
        case (HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED)
          name = "HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED"
        case (HIPRAND_STATUS_LAUNCH_FAILURE)
          name = "HIPRAND_STATUS_LAUNCH_FAILURE"
        case (HIPRAND_STATUS_PREEXISTING_FAILURE)
          name = "HIPRAND_STATUS_PREEXISTING_FAILURE"
        case (HIPRAND_STATUS_INITIALIZATION_FAILED)
          name = "HIPRAND_STATUS_INITIALIZATION_FAILED"
        case (HIPRAND_STATUS_ARCH_MISMATCH)
          name = "HIPRAND_STATUS_ARCH_MISMATCH"
        case (HIPRAND_STATUS_INTERNAL_ERROR)
          name = "HIPRAND_STATUS_INTERNAL_ERROR"
        case (HIPRAND_STATUS_NOT_IMPLEMENTED)
          name = "HIPRAND_STATUS_NOT_IMPLEMENTED"
        case default
          name = "unknown status"
        end select
        write (*, "(a,i0,a)") "HIPRAND ERROR: " // name // " (code ", status, ")"
        stop 1
      end if
    end subroutine hiprandCheck
    function hiprandGetVersion_typed(version)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGetVersion_typed
      integer(c_int),target :: version
      !
      hiprandGetVersion_typed = hiprandGetVersion_(c_loc(version))
    end function


#if defined(USE_ASSUMED_SHAPE) || defined(USE_ASSUMED_RANK)

#ifdef USE_ASSUMED_RANK
    function hiprandGenerate_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerate_assumed_rank
      type(c_ptr), value :: generator
      integer(c_int),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerate_assumed_rank = hiprandGenerate_(generator,c_loc(output_data),n)
    end function

#else
    function hiprandGenerate_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerate_rank_0
      type(c_ptr), value :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerate_rank_0 = hiprandGenerate_(generator,c_loc(output_data),n)
    end function

    function hiprandGenerate_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerate_rank_1
      type(c_ptr), value :: generator
      integer(c_int),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerate_rank_1 = hiprandGenerate_(generator,c_loc(output_data),n)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hiprandGenerateLongLong_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLongLong_assumed_rank
      type(c_ptr), value :: generator
      integer(c_int64_t),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateLongLong_assumed_rank = hiprandGenerateLongLong_(generator, &
        c_loc(output_data),n)
    end function

#else
    function hiprandGenerateLongLong_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLongLong_rank_0
      type(c_ptr), value :: generator
      integer(c_int64_t),target :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateLongLong_rank_0 = hiprandGenerateLongLong_(generator,c_loc(output_data),n)
    end function

    function hiprandGenerateLongLong_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLongLong_rank_1
      type(c_ptr), value :: generator
      integer(c_int64_t),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateLongLong_rank_1 = hiprandGenerateLongLong_(generator,c_loc(output_data),n)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hiprandGenerateUniform_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniform_assumed_rank
      type(c_ptr), value :: generator
      real(c_float),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniform_assumed_rank = hiprandGenerateUniform_(generator,c_loc(output_data),n)
    end function

#else
    function hiprandGenerateUniform_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniform_rank_0
      type(c_ptr), value :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniform_rank_0 = hiprandGenerateUniform_(generator,c_loc(output_data),n)
    end function

    function hiprandGenerateUniform_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniform_rank_1
      type(c_ptr), value :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniform_rank_1 = hiprandGenerateUniform_(generator,c_loc(output_data),n)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hiprandGenerateUniformDouble_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniformDouble_assumed_rank
      type(c_ptr), value :: generator
      real(c_double),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniformDouble_assumed_rank = hiprandGenerateUniformDouble_(generator, &
        c_loc(output_data),n)
    end function

#else
    function hiprandGenerateUniformDouble_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniformDouble_rank_0
      type(c_ptr), value :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniformDouble_rank_0 = hiprandGenerateUniformDouble_(generator, &
        c_loc(output_data),n)
    end function

    function hiprandGenerateUniformDouble_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniformDouble_rank_1
      type(c_ptr), value :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniformDouble_rank_1 = hiprandGenerateUniformDouble_(generator, &
        c_loc(output_data),n)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hiprandGenerateNormal_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormal_assumed_rank
      type(c_ptr), value :: generator
      real(c_float),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateNormal_assumed_rank = hiprandGenerateNormal_(generator,c_loc(output_data),n, &
        mean,stddev)
    end function

#else
    function hiprandGenerateNormal_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormal_rank_0
      type(c_ptr), value :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateNormal_rank_0 = hiprandGenerateNormal_(generator,c_loc(output_data),n,mean, &
        stddev)
    end function

    function hiprandGenerateNormal_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormal_rank_1
      type(c_ptr), value :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateNormal_rank_1 = hiprandGenerateNormal_(generator,c_loc(output_data),n,mean, &
        stddev)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hiprandGenerateNormalDouble_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormalDouble_assumed_rank
      type(c_ptr), value :: generator
      real(c_double),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateNormalDouble_assumed_rank = hiprandGenerateNormalDouble_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function hiprandGenerateNormalDouble_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormalDouble_rank_0
      type(c_ptr), value :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateNormalDouble_rank_0 = hiprandGenerateNormalDouble_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

    function hiprandGenerateNormalDouble_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormalDouble_rank_1
      type(c_ptr), value :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateNormalDouble_rank_1 = hiprandGenerateNormalDouble_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hiprandGenerateLogNormal_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormal_assumed_rank
      type(c_ptr), value :: generator
      real(c_float),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateLogNormal_assumed_rank = hiprandGenerateLogNormal_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function hiprandGenerateLogNormal_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormal_rank_0
      type(c_ptr), value :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateLogNormal_rank_0 = hiprandGenerateLogNormal_(generator,c_loc(output_data),n, &
        mean,stddev)
    end function

    function hiprandGenerateLogNormal_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormal_rank_1
      type(c_ptr), value :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateLogNormal_rank_1 = hiprandGenerateLogNormal_(generator,c_loc(output_data),n, &
        mean,stddev)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hiprandGenerateLogNormalDouble_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormalDouble_assumed_rank
      type(c_ptr), value :: generator
      real(c_double),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateLogNormalDouble_assumed_rank = hiprandGenerateLogNormalDouble_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function hiprandGenerateLogNormalDouble_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormalDouble_rank_0
      type(c_ptr), value :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateLogNormalDouble_rank_0 = hiprandGenerateLogNormalDouble_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

    function hiprandGenerateLogNormalDouble_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormalDouble_rank_1
      type(c_ptr), value :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateLogNormalDouble_rank_1 = hiprandGenerateLogNormalDouble_(generator, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
#ifdef USE_ASSUMED_RANK
    function hiprandGeneratePoisson_assumed_rank(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGeneratePoisson_assumed_rank
      type(c_ptr), value :: generator
      integer(c_int),target,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      hiprandGeneratePoisson_assumed_rank = hiprandGeneratePoisson_(generator,c_loc(output_data), &
        n,lambda)
    end function

#else
    function hiprandGeneratePoisson_rank_0(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGeneratePoisson_rank_0
      type(c_ptr), value :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      hiprandGeneratePoisson_rank_0 = hiprandGeneratePoisson_(generator,c_loc(output_data),n,lambda)
    end function

    function hiprandGeneratePoisson_rank_1(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGeneratePoisson_rank_1
      type(c_ptr), value :: generator
      integer(c_int),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      hiprandGeneratePoisson_rank_1 = hiprandGeneratePoisson_(generator,c_loc(output_data),n,lambda)
    end function

#endif
#endif
end module hiprand
