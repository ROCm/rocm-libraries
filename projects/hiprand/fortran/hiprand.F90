!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ==============================================================================
! hipfort: FORTRAN Interfaces for GPU kernels
! ==============================================================================
! Copyright (c) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.
! [MITx11 License]
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
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
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


  type, bind(c) :: hiprandDiscreteDistribution_t
    type(c_ptr) :: ptr
  end type hiprandDiscreteDistribution_t

  type, bind(c) :: hiprandGenerator_t
    type(c_ptr) :: ptr
  end type hiprandGenerator_t

  interface

    !---------------------------------------------
    ! hiprandGetDirectionVectors32
    !---------------------------------------------
    !>  \brief Retrieves the Sobol 32 direction vector array specified by \p set.
    !>
    !>  \param vectors - Pointer to the Sobol 32 direction vector array.
    !>  \param set - Specifies which hipRAND vector set for quasirandom generators to retrieve.
    !>
    !>  \return
    !>  - HIPRAND_STATUS_OUT_OF_RANGE if \p set is invalid
    !>  - HIPRAND_STATUS_SUCCESS if \p vectors was set successfully
    function hiprandGetDirectionVectors32(vectors, set) &
       result(GetDirectionVectors32) &
       bind(C, name="hiprandGetDirectionVectors32")
       import :: c_ptr, HIPRAND_DIRECTION_VECTORS_32_JOEKUO6, HIPRAND_STATUS_SUCCESS
       type(c_ptr) :: vectors
       integer(kind(HIPRAND_DIRECTION_VECTORS_32_JOEKUO6)), value :: set
       integer(kind(HIPRAND_STATUS_SUCCESS)) :: GetDirectionVectors32
    end function hiprandGetDirectionVectors32

    !---------------------------------------------
    ! hiprandGetDirectionVectors64
    !---------------------------------------------
    !>  \brief Retrieves the Sobol 64 direction vector array specified by \p set.
    !>
    !>  \param vectors - Pointer to the Sobol 64 direction vector array.
    !>  \param set - Specifies which hipRAND vector set for quasirandom generators to retrieve.
    !>
    !>  \return
    !>  - HIPRAND_STATUS_OUT_OF_RANGE if \p set is invalid
    !>  - HIPRAND_STATUS_SUCCESS if \p vectors was set successfully
    function hiprandGetDirectionVectors64(vectors, set) &
       result(GetDirectionVectors64) &
       bind(C, name="hiprandGetDirectionVectors64")
       import :: c_ptr, HIPRAND_DIRECTION_VECTORS_32_JOEKUO6, HIPRAND_STATUS_SUCCESS
       type(c_ptr) :: vectors
       integer(kind(HIPRAND_DIRECTION_VECTORS_32_JOEKUO6)), value :: set
       integer(kind(HIPRAND_STATUS_SUCCESS)) :: GetDirectionVectors64
    end function hiprandGetDirectionVectors64

    !---------------------------------------------
    ! hiprandGetScrambleConstants32
    !---------------------------------------------
    !>  \brief Retrieves the scramble constants for 32-bit scrambled Sobol generation.
    !>
    !>  \param constants - Pointer to the constants pointer.
    !>
    !>  \return
    !>  - HIPRAND_STATUS_SUCCESS if the pointer was set successfully
    function hiprandGetScrambleConstants32(constants) &
       result(GetScrambleConstants32) &
       bind(C, name="hiprandGetScrambleConstants32")
       import :: c_ptr, HIPRAND_STATUS_SUCCESS
       type(c_ptr) :: constants
       integer(kind(HIPRAND_STATUS_SUCCESS)) :: GetScrambleConstants32
    end function hiprandGetScrambleConstants32

    !---------------------------------------------
    ! hiprandGetScrambleConstants64
    !---------------------------------------------
    !>  \brief Retrieves the scramble constants for 64-bit scrambled Sobol generation.
    !>
    !>  \param constants - Pointer to the constants pointer.
    !>
    !>  \return
    !>  - HIPRAND_STATUS_SUCCESS if the pointer was set successfully
    function hiprandGetScrambleConstants64(constants) &
       result(GetScrambleConstants64) &
       bind(C, name="hiprandGetScrambleConstants64")
       import :: c_ptr, HIPRAND_STATUS_SUCCESS
       type(c_ptr) :: constants
       integer(kind(HIPRAND_STATUS_SUCCESS)) :: GetScrambleConstants64
    end function hiprandGetScrambleConstants64

  end interface

  !>  \brief Creates a new random number generator.
  !>
  !>  Creates a new random number generator of type \p rng_type,
  !>  and returns it in \p generator. That generator will use
  !>  GPU to create random numbers.
  !>
  !>  Values for \p rng_type are:
  !>  - HIPRAND_RNG_PSEUDO_DEFAULT
  !>  - HIPRAND_RNG_PSEUDO_XORWOW
  !>  - HIPRAND_RNG_PSEUDO_MRG32K3A
  !>  - HIPRAND_RNG_PSEUDO_MTGP32
  !>  - HIPRAND_RNG_PSEUDO_MT19937
  !>  - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
  !>  - HIPRAND_RNG_QUASI_DEFAULT
  !>  - HIPRAND_RNG_QUASI_SOBOL32
  !>  - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
  !>  - HIPRAND_RNG_QUASI_SOBOL64
  !>  - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
  !>
  !>  \param generator - Pointer to generator
  !>  \param rng_type - Type of random number generator to create
  !>
  !>  \return
  !>  - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed
  !>  - HIPRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU
  !>  - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
  !>    dynamically linked library version
  !>  - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid
  !>  - HIPRAND_STATUS_NOT_IMPLEMENTED if generator of type \p rng_type is not implemented yet
  !>  - HIPRAND_STATUS_SUCCESS if generator was created successfully
  interface hiprandCreateGenerator
    function hiprandCreateGenerator_(generator,rng_type) bind(c, name="hiprandCreateGenerator")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandCreateGenerator_
      type(c_ptr) :: generator
      integer(kind(HIPRAND_RNG_TEST)),value :: rng_type
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandCreateGenerator_typed
#endif
  end interface

  !>  \brief Creates a new random number generator on host.
  !>
  !>  Creates a new host random number generator of type \p rng_type
  !>  and returns it in \p generator. Created generator will use
  !>  host CPU to generate random numbers.
  !>
  !>  Values for \p rng_type are:
  !>  - HIPRAND_RNG_PSEUDO_DEFAULT
  !>  - HIPRAND_RNG_PSEUDO_XORWOW
  !>  - HIPRAND_RNG_PSEUDO_MRG32K3A
  !>  - HIPRAND_RNG_PSEUDO_MTGP32
  !>  - HIPRAND_RNG_PSEUDO_MT19937
  !>  - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
  !>  - HIPRAND_RNG_QUASI_DEFAULT
  !>  - HIPRAND_RNG_QUASI_SOBOL32
  !>  - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
  !>  - HIPRAND_RNG_QUASI_SOBOL64
  !>  - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
  !>
  !>  \param generator - Pointer to generator
  !>  \param rng_type - Type of random number generator to create
  !>
  !>  \return
  !>  - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed
  !>  - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
  !>    dynamically linked library version
  !>  - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid
  !>  - HIPRAND_STATUS_NOT_IMPLEMENTED if host generator of type \p rng_type is not implemented yet
  !>  - HIPRAND_STATUS_SUCCESS if generator was created successfully
  interface hiprandCreateGeneratorHost
    function hiprandCreateGeneratorHost_(generator,rng_type) &
        bind(c, name="hiprandCreateGeneratorHost")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandCreateGeneratorHost_
      type(c_ptr) :: generator
      integer(kind(HIPRAND_RNG_TEST)),value :: rng_type
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandCreateGeneratorHost_typed
#endif
  end interface

  !>  \brief Destroys random number generator.
  !>
  !>  Destroys random number generator and frees related memory.
  !>
  !>  \param generator - Generator to be destroyed
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_SUCCESS if generator was destroyed successfully
  interface hiprandDestroyGenerator
    function hiprandDestroyGenerator_(generator) bind(c, name="hiprandDestroyGenerator")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandDestroyGenerator_
      type(c_ptr),value :: generator
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandDestroyGenerator_typed
#endif
  end interface

  !>  \brief Generates uniformly distributed 32-bit unsigned integers.
  !>
  !>  Generates \p n uniformly distributed 32-bit unsigned integers and
  !>  saves them to \p output_data.
  !>
  !>  Generated numbers are between \p 0 and \p 2^32, including \p 0 and
  !>  excluding \p 2^32.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of 32-bit unsigned integers to generate
  !>
  !>  Note: \p generator must be not be of type \p HIPRAND_RNG_QUASI_SOBOL64
  !>  or \p HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64.
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerate
    function hiprandGenerate_(generator,output_data,n) bind(c, name="hiprandGenerate")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerate_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGenerate_assumed_rank,&
      hiprandGenerate_typed
#else
    module procedure &
      hiprandGenerate_rank_0,&
      hiprandGenerate_rank_1,&
      hiprandGenerate_typed
#endif
#endif
  end interface

  !>  \brief Generates uniformly distributed 8-bit unsigned integers.
  !>
  !>  Generates \p n uniformly distributed 8-bit unsigned integers and
  !>  saves them to \p output_data.
  !>
  !>  Generated numbers are between \p 0 and \p 2^8, including \p 0 and
  !>  excluding \p 2^8.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of 8-bit unsigned integers to generate
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateChar
    function hiprandGenerateChar_(generator,output_data,n) bind(c, name="hiprandGenerateChar")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateChar_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandGenerateChar_typed
#endif
  end interface

  !>  \brief Generates uniformly distributed 16-bit unsigned integers.
  !>
  !>  Generates \p n uniformly distributed 16-bit unsigned integers and
  !>  saves them to \p output_data.
  !>
  !>  Generated numbers are between \p 0 and \p 2^16, including \p 0 and
  !>  excluding \p 2^16.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of 16-bit unsigned integers to generate
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateShort
    function hiprandGenerateShort_(generator,output_data,n) bind(c, name="hiprandGenerateShort")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateShort_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandGenerateShort_typed
#endif
  end interface

  !>  \brief Generates uniformly distributed 64-bit unsigned integers.
  !>
  !>  Generates \p n uniformly distributed 64-bit unsigned integers and
  !>  saves them to \p output_data.
  !>
  !>  Generated numbers are between \p 0 and \p 2^64, including \p 0 and
  !>  excluding \p 2^64.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of 64-bit unsigned integers to generate
  !>
  !>  Note: \p generator must be of type \p HIPRAND_RNG_QUASI_SOBOL64
  !>  or \p HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64.
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateLongLong
    function hiprandGenerateLongLong_(generator,output_data,n) &
        bind(c, name="hiprandGenerateLongLong")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLongLong_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGenerateLongLong_assumed_rank,&
      hiprandGenerateLongLong_typed
#else
    module procedure &
      hiprandGenerateLongLong_rank_0,&
      hiprandGenerateLongLong_rank_1,&
      hiprandGenerateLongLong_typed
#endif
#endif
  end interface

  !>  \brief Generates uniformly distributed floats.
  !>
  !>  Generates \p n uniformly distributed 32-bit floating-point values
  !>  and saves them to \p output_data.
  !>
  !>  Generated numbers are between \p 0.0f and \p 1.0f, excluding \p 0.0f and
  !>  including \p 1.0f.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of floats to generate
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateUniform
    function hiprandGenerateUniform_(generator,output_data,n) bind(c, name="hiprandGenerateUniform")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniform_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGenerateUniform_assumed_rank,&
      hiprandGenerateUniform_typed
#else
    module procedure &
      hiprandGenerateUniform_rank_0,&
      hiprandGenerateUniform_rank_1,&
      hiprandGenerateUniform_typed
#endif
#endif
  end interface

  !>  \brief Generates uniformly distributed double-precision floating-point values.
  !>
  !>  Generates \p n uniformly distributed 64-bit double-precision floating-point
  !>  values and saves them to \p output_data.
  !>
  !>  Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
  !>  including \p 1.0.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of floats to generate
  !>
  !>  Note: When \p generator is of type: \p HIPRAND_RNG_PSEUDO_MRG32K3A,
  !>  \p HIPRAND_RNG_PSEUDO_MTGP32, \p HIPRAND_RNG_QUASI_SOBOL32, or
  !>  \p HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 then the returned \p double
  !>  values are generated from only 32 random bits
  !>  each (one unsigned int value per one generated \p double).
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateUniformDouble
    function hiprandGenerateUniformDouble_(generator,output_data,n) &
        bind(c, name="hiprandGenerateUniformDouble")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniformDouble_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGenerateUniformDouble_assumed_rank,&
      hiprandGenerateUniformDouble_typed
#else
    module procedure &
      hiprandGenerateUniformDouble_rank_0,&
      hiprandGenerateUniformDouble_rank_1,&
      hiprandGenerateUniformDouble_typed
#endif
#endif
  end interface

  !>  \brief Generates uniformly distributed half-precision floating-point values.
  !>
  !>  Generates \p n uniformly distributed 16-bit half-precision floating-point
  !>  values and saves them to \p output_data.
  !>
  !>  Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
  !>  including \p 1.0.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of halfs to generate
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateUniformHalf
    function hiprandGenerateUniformHalf_(generator,output_data,n) &
        bind(c, name="hiprandGenerateUniformHalf")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniformHalf_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandGenerateUniformHalf_typed
#endif
  end interface

  !>  \brief Generates normally distributed floats.
  !>
  !>  Generates \p n normally distributed 32-bit floating-point
  !>  values and saves them to \p output_data.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of floats to generate
  !>  \param mean - Mean value of normal distribution
  !>  \param stddev - Standard deviation value of normal distribution
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
  !>  aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateNormal
    function hiprandGenerateNormal_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateNormal")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormal_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_float),value :: mean
      real(c_float),value :: stddev
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGenerateNormal_assumed_rank,&
      hiprandGenerateNormal_typed
#else
    module procedure &
      hiprandGenerateNormal_rank_0,&
      hiprandGenerateNormal_rank_1,&
      hiprandGenerateNormal_typed
#endif
#endif
  end interface

  !>  \brief Generates normally distributed doubles.
  !>
  !>  Generates \p n normally distributed 64-bit double-precision floating-point
  !>  numbers and saves them to \p output_data.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of doubles to generate
  !>  \param mean - Mean value of normal distribution
  !>  \param stddev - Standard deviation value of normal distribution
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
  !>  aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateNormalDouble
    function hiprandGenerateNormalDouble_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateNormalDouble")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormalDouble_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: mean
      real(c_double),value :: stddev
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGenerateNormalDouble_assumed_rank,&
      hiprandGenerateNormalDouble_typed
#else
    module procedure &
      hiprandGenerateNormalDouble_rank_0,&
      hiprandGenerateNormalDouble_rank_1,&
      hiprandGenerateNormalDouble_typed
#endif
#endif
  end interface

  !>  \brief Generates normally distributed halfs.
  !>
  !>  Generates \p n normally distributed 16-bit half-precision floating-point
  !>  numbers and saves them to \p output_data.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of halfs to generate
  !>  \param mean - Mean value of normal distribution
  !>  \param stddev - Standard deviation value of normal distribution
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
  !>  aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateNormalHalf
    function hiprandGenerateNormalHalf_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateNormalHalf")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormalHalf_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      integer(c_short),value :: mean
      integer(c_short),value :: stddev
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandGenerateNormalHalf_typed
#endif
  end interface

  !>  \brief Generates log-normally distributed floats.
  !>
  !>  Generates \p n log-normally distributed 32-bit floating-point values
  !>  and saves them to \p output_data.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of floats to generate
  !>  \param mean - Mean value of log normal distribution
  !>  \param stddev - Standard deviation value of log normal distribution
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
  !>  aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateLogNormal
    function hiprandGenerateLogNormal_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateLogNormal")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormal_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_float),value :: mean
      real(c_float),value :: stddev
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGenerateLogNormal_assumed_rank,&
      hiprandGenerateLogNormal_typed
#else
    module procedure &
      hiprandGenerateLogNormal_rank_0,&
      hiprandGenerateLogNormal_rank_1,&
      hiprandGenerateLogNormal_typed
#endif
#endif
  end interface

  !>  \brief Generates log-normally distributed doubles.
  !>
  !>  Generates \p n log-normally distributed 64-bit double-precision floating-point
  !>  values and saves them to \p output_data.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of doubles to generate
  !>  \param mean - Mean value of log normal distribution
  !>  \param stddev - Standard deviation value of log normal distribution
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
  !>  aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateLogNormalDouble
    function hiprandGenerateLogNormalDouble_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateLogNormalDouble")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormalDouble_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: mean
      real(c_double),value :: stddev
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGenerateLogNormalDouble_assumed_rank,&
      hiprandGenerateLogNormalDouble_typed
#else
    module procedure &
      hiprandGenerateLogNormalDouble_rank_0,&
      hiprandGenerateLogNormalDouble_rank_1,&
      hiprandGenerateLogNormalDouble_typed
#endif
#endif
  end interface

  !>  \brief Generates log-normally distributed halfs.
  !>
  !>  Generates \p n log-normally distributed 16-bit half-precision floating-point
  !>  values and saves them to \p output_data.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of halfs to generate
  !>  \param mean - Mean value of log normal distribution
  !>  \param stddev - Standard deviation value of log normal distribution
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
  !>  aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGenerateLogNormalHalf
    function hiprandGenerateLogNormalHalf_(generator,output_data,n,mean,stddev) &
        bind(c, name="hiprandGenerateLogNormalHalf")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormalHalf_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      integer(c_short),value :: mean
      integer(c_short),value :: stddev
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandGenerateLogNormalHalf_typed
#endif
  end interface

  !>  \brief Generates Poisson-distributed 32-bit unsigned integers.
  !>
  !>  Generates \p n Poisson-distributed 32-bit unsigned integers and
  !>  saves them to \p output_data.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of 32-bit unsigned integers to generate
  !>  \param lambda - lambda for the Poisson distribution
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel
  !>  - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive
  !>  - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface hiprandGeneratePoisson
    function hiprandGeneratePoisson_(generator,output_data,n,lambda) &
        bind(c, name="hiprandGeneratePoisson")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGeneratePoisson_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: lambda
    end function

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      hiprandGeneratePoisson_assumed_rank,&
      hiprandGeneratePoisson_typed
#else
    module procedure &
      hiprandGeneratePoisson_rank_0,&
      hiprandGeneratePoisson_rank_1,&
      hiprandGeneratePoisson_typed
#endif
#endif
  end interface

  !>  \brief Initializes the generator's state on GPU or host.
  !>
  !>  Initializes the generator's state on GPU or host.
  !>
  !>  If hiprandGenerateSeeds() was not called for a generator, it will be
  !>  automatically called by functions which generates random numbers like
  !>  hiprandGenerate(), hiprandGenerateUniform(), hiprandGenerateNormal() etc.
  !>
  !>  \param generator - Generator to initialize
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was never created
  !>  - HIPRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
  !>    a previous kernel launch
  !>  - HIPRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason
  !>  - HIPRAND_STATUS_SUCCESS if the seeds were generated successfully
  interface hiprandGenerateSeeds
    function hiprandGenerateSeeds_(generator) bind(c, name="hiprandGenerateSeeds")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateSeeds_
      type(c_ptr),value :: generator
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandGenerateSeeds_typed
#endif
  end interface

  !>  \brief Sets the current stream for kernel launches.
  !>
  !>  Sets the current stream for all kernel launches of the generator.
  !>  All functions will use this stream.
  !>
  !>  \param generator - Generator to modify
  !>  \param stream - Stream to use or NULL for default stream
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_SUCCESS if stream was set successfully
  interface hiprandSetStream
    function hiprandSetStream_(generator,stream) bind(c, name="hiprandSetStream")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandSetStream_
      type(c_ptr),value :: generator
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandSetStream_typed
#endif
  end interface

  !>  \brief Sets the seed of a pseudo-random number generator.
  !>
  !>  Sets the seed of the pseudo-random number generator.
  !>
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's offset.
  !>
  !>  \param generator - Pseudo-random number generator
  !>  \param seed - New seed value
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_TYPE_ERROR if the generator is a quasi random number generator
  !>  - HIPRAND_STATUS_SUCCESS if seed was set successfully
  interface hiprandSetPseudoRandomGeneratorSeed
    function hiprandSetPseudoRandomGeneratorSeed_(generator,seed) &
        bind(c, name="hiprandSetPseudoRandomGeneratorSeed")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandSetPseudoRandomGeneratorSeed_
      type(c_ptr),value :: generator
      integer(c_int64_t),value :: seed
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandSetPseudoRandomGeneratorSeed_typed
#endif
  end interface

  !>  \brief Sets the offset of a random number generator.
  !>
  !>  Sets the absolute offset of the random number generator.
  !>
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's seed.
  !>
  !>  Absolute offset cannot be set if generator's type is
  !>  HIPRAND_RNG_PSEUDO_MTGP32 or HIPRAND_RNG_PSEUDO_MT19937.
  !>
  !>  \param generator - Random number generator
  !>  \param offset - New absolute offset
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_SUCCESS if offset was successfully set
  !>  - HIPRAND_STATUS_TYPE_ERROR if generator's type is HIPRAND_RNG_PSEUDO_MTGP32
  !>  or HIPRAND_RNG_PSEUDO_MT19937
  interface hiprandSetGeneratorOffset
    function hiprandSetGeneratorOffset_(generator,offset) bind(c, name="hiprandSetGeneratorOffset")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandSetGeneratorOffset_
      type(c_ptr),value :: generator
      integer(c_int64_t),value :: offset
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandSetGeneratorOffset_typed
#endif
  end interface

  !>  \brief Sets the ordering of a random number generator.
  !>
  !>  Sets the ordering of the results of a random number generator.
  !>
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's seed.
  !>
  !>  \param generator - Random number generator
  !>  \param order - New ordering of results
  !>
  !>  The ordering choices for pseudorandom sequences are
  !>  HIPRAND_ORDERING_PSEUDO_DEFAULT and
  !>  HIPRAND_ORDERING_PSEUDO_LEGACY.
  !>  The default ordering is HIPRAND_ORDERING_PSEUDO_DEFAULT, which is equal to
  !>  HIPRAND_ORDERING_PSEUDO_LEGACY for now.
  !>
  !>  For quasirandom sequences there is only one ordering, HIPRAND_ORDERING_QUASI_DEFAULT.
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized
  !>  - HIPRAND_STATUS_OUT_OF_RANGE if the ordering is not valid
  !>  - HIPRAND_STATUS_SUCCESS if the ordering was successfully set
  !>  - HIPRAND_STATUS_TYPE_ERROR if generator's type is not valid
  interface hiprandSetGeneratorOrdering
    function hiprandSetGeneratorOrdering_(generator,order) &
        bind(c, name="hiprandSetGeneratorOrdering")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandSetGeneratorOrdering_
      type(c_ptr),value :: generator
      integer(kind(HIPRAND_ORDERING_PSEUDO_BEST)),value :: order
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandSetGeneratorOrdering_typed
#endif
  end interface

  !>  \brief Set the number of dimensions of a quasi-random number generator.
  !>
  !>  Set the number of dimensions of a quasi-random number generator.
  !>  Supported values of \p dimensions are 1 to 20000.
  !>
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's offset.
  !>
  !>  \param generator - Quasi-random number generator
  !>  \param dimensions - Number of dimensions
  !>
  !>  \return
  !>  - HIPRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - HIPRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator
  !>  - HIPRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range
  !>  - HIPRAND_STATUS_SUCCESS if the number of dimensions was set successfully
  interface hiprandSetQuasiRandomGeneratorDimensions
    function hiprandSetQuasiRandomGeneratorDimensions_(generator,dimensions) &
        bind(c, name="hiprandSetQuasiRandomGeneratorDimensions")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandSetQuasiRandomGeneratorDimensions_
      type(c_ptr),value :: generator
      integer(c_int),value :: dimensions
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandSetQuasiRandomGeneratorDimensions_typed
#endif
  end interface

  !>  \brief Returns the version number of the cuRAND or rocRAND library.
  !>
  !>  Returns in \p version the version number of the underlying cuRAND or
  !>  rocRAND library.
  !>
  !>  \param version - Version of the library
  !>
  !>  \return
  !>  - HIPRAND_STATUS_OUT_OF_RANGE if \p version is NULL
  !>  - HIPRAND_STATUS_SUCCESS if the version number was successfully returned
  interface hiprandGetVersion
    function hiprandGetVersion_(version) bind(c, name="hiprandGetVersion")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGetVersion_
      integer(c_int) :: version
    end function

    function hiprandGetVersion_dptr(version) bind(c, name="hiprandGetVersion")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGetVersion_dptr
      type(c_ptr),value :: version
    end function
  end interface

  !>  \brief Construct the histogram for a Poisson distribution.
  !>
  !>  Construct the histogram for the Poisson distribution with lambda \p lambda.
  !>
  !>  \param lambda - lambda for the Poisson distribution
  !>  \param discrete_distribution - pointer to the histogram in device memory
  !>
  !>  \return
  !>  - HIPRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated
  !>  - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null
  !>  - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive
  !>  - HIPRAND_STATUS_SUCCESS if the histogram was constructed successfully
  interface hiprandCreatePoissonDistribution
    function hiprandCreatePoissonDistribution_(lambda,discrete_distribution) &
        bind(c, name="hiprandCreatePoissonDistribution")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandCreatePoissonDistribution_
      real(c_double),value :: lambda
      type(c_ptr) :: discrete_distribution
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandCreatePoissonDistribution_typed
#endif
  end interface

  !>  \brief Destroy the histogram array for a discrete distribution.
  !>
  !>  Destroy the histogram array for a discrete distribution created by
  !>  hiprandCreatePoissonDistribution.
  !>
  !>  \param discrete_distribution - pointer to the histogram in device memory
  !>
  !>  \return
  !>  - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null
  !>  - HIPRAND_STATUS_SUCCESS if the histogram was destroyed successfully
  interface hiprandDestroyDistribution
    function hiprandDestroyDistribution_(discrete_distribution) &
        bind(c, name="hiprandDestroyDistribution")
      use iso_c_binding
      import
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandDestroyDistribution_
      type(c_ptr),value :: discrete_distribution
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure hiprandDestroyDistribution_typed
#endif
  end interface


  contains

    subroutine hiprandCheck(status)
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: status
      if (status /= HIPRAND_STATUS_SUCCESS) then
        write (*, *) "HIPRAND ERROR: code = ", status
        stop 1
      end if
    end subroutine hiprandCheck

#ifdef USE_FPOINTER_INTERFACES

    function hiprandCreateGenerator_typed(generator, rng_type) result(CreateGenerator)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t) :: generator
      integer(kind(HIPRAND_RNG_TEST)), value :: rng_type
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: CreateGenerator
      CreateGenerator = hiprandCreateGenerator_(generator%ptr, rng_type)
    end function hiprandCreateGenerator_typed

    function hiprandCreateGeneratorHost_typed(generator, rng_type) result(CreateGeneratorHost)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t) :: generator
      integer(kind(HIPRAND_RNG_TEST)), value :: rng_type
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: CreateGeneratorHost
      CreateGeneratorHost = hiprandCreateGeneratorHost_(generator%ptr, rng_type)
    end function hiprandCreateGeneratorHost_typed

    function hiprandDestroyGenerator_typed(generator) result(DestroyGenerator)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: DestroyGenerator
      DestroyGenerator = hiprandDestroyGenerator_(generator%ptr)
    end function hiprandDestroyGenerator_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGenerate_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerate_assumed_rank
      type(hiprandGenerator_t) :: generator
      integer(c_int),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerate_assumed_rank = hiprandGenerate_(generator%ptr,c_loc(output_data),n)
    end function

#else
    function hiprandGenerate_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerate_rank_0
      type(hiprandGenerator_t) :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerate_rank_0 = hiprandGenerate_(generator%ptr,c_loc(output_data),n)
    end function

    function hiprandGenerate_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerate_rank_1
      type(hiprandGenerator_t) :: generator
      integer(c_int),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerate_rank_1 = hiprandGenerate_(generator%ptr,c_loc(output_data),n)
    end function

#endif
    function hiprandGenerate_typed(generator, output_data, n) result(Generate)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: Generate
      Generate = hiprandGenerate_(generator%ptr, output_data, n)
    end function hiprandGenerate_typed

    function hiprandGenerateChar_typed(generator, output_data, n) result(GenerateChar)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateChar
      GenerateChar = hiprandGenerateChar_(generator%ptr, output_data, n)
    end function hiprandGenerateChar_typed

    function hiprandGenerateShort_typed(generator, output_data, n) result(GenerateShort)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateShort
      GenerateShort = hiprandGenerateShort_(generator%ptr, output_data, n)
    end function hiprandGenerateShort_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGenerateLongLong_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLongLong_assumed_rank
      type(hiprandGenerator_t) :: generator
      integer(c_int64_t),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateLongLong_assumed_rank = hiprandGenerateLongLong_(generator%ptr, &
        c_loc(output_data),n)
    end function

#else
    function hiprandGenerateLongLong_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLongLong_rank_0
      type(hiprandGenerator_t) :: generator
      integer(c_int64_t),target :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateLongLong_rank_0 = hiprandGenerateLongLong_(generator%ptr,c_loc(output_data),n)
    end function

    function hiprandGenerateLongLong_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLongLong_rank_1
      type(hiprandGenerator_t) :: generator
      integer(c_int64_t),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateLongLong_rank_1 = hiprandGenerateLongLong_(generator%ptr,c_loc(output_data),n)
    end function

#endif
    function hiprandGenerateLongLong_typed(generator, output_data, n) result(GenerateLongLong)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateLongLong
      GenerateLongLong = hiprandGenerateLongLong_(generator%ptr, output_data, n)
    end function hiprandGenerateLongLong_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGenerateUniform_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniform_assumed_rank
      type(hiprandGenerator_t) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniform_assumed_rank = hiprandGenerateUniform_(generator%ptr, &
        c_loc(output_data),n)
    end function

#else
    function hiprandGenerateUniform_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniform_rank_0
      type(hiprandGenerator_t) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniform_rank_0 = hiprandGenerateUniform_(generator%ptr,c_loc(output_data),n)
    end function

    function hiprandGenerateUniform_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniform_rank_1
      type(hiprandGenerator_t) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniform_rank_1 = hiprandGenerateUniform_(generator%ptr,c_loc(output_data),n)
    end function

#endif
    function hiprandGenerateUniform_typed(generator, output_data, n) result(GenerateUniform)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateUniform
      GenerateUniform = hiprandGenerateUniform_(generator%ptr, output_data, n)
    end function hiprandGenerateUniform_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGenerateUniformDouble_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniformDouble_assumed_rank
      type(hiprandGenerator_t) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniformDouble_assumed_rank = hiprandGenerateUniformDouble_(generator%ptr, &
        c_loc(output_data),n)
    end function

#else
    function hiprandGenerateUniformDouble_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniformDouble_rank_0
      type(hiprandGenerator_t) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniformDouble_rank_0 = hiprandGenerateUniformDouble_(generator%ptr, &
        c_loc(output_data),n)
    end function

    function hiprandGenerateUniformDouble_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateUniformDouble_rank_1
      type(hiprandGenerator_t) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      hiprandGenerateUniformDouble_rank_1 = hiprandGenerateUniformDouble_(generator%ptr, &
        c_loc(output_data),n)
    end function

#endif
    function hiprandGenerateUniformDouble_typed(generator, output_data, &
        n) result(GenerateUniformDouble)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateUniformDouble
      GenerateUniformDouble = hiprandGenerateUniformDouble_(generator%ptr, output_data, n)
    end function hiprandGenerateUniformDouble_typed

    function hiprandGenerateUniformHalf_typed(generator, output_data, n) result(GenerateUniformHalf)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateUniformHalf
      GenerateUniformHalf = hiprandGenerateUniformHalf_(generator%ptr, output_data, n)
    end function hiprandGenerateUniformHalf_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGenerateNormal_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormal_assumed_rank
      type(hiprandGenerator_t) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateNormal_assumed_rank = hiprandGenerateNormal_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function hiprandGenerateNormal_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormal_rank_0
      type(hiprandGenerator_t) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateNormal_rank_0 = hiprandGenerateNormal_(generator%ptr,c_loc(output_data),n, &
        mean,stddev)
    end function

    function hiprandGenerateNormal_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormal_rank_1
      type(hiprandGenerator_t) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateNormal_rank_1 = hiprandGenerateNormal_(generator%ptr,c_loc(output_data),n, &
        mean,stddev)
    end function

#endif
    function hiprandGenerateNormal_typed(generator, output_data, n, mean, &
        stddev) result(GenerateNormal)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_float), value :: mean
      real(c_float), value :: stddev
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateNormal
      GenerateNormal = hiprandGenerateNormal_(generator%ptr, output_data, n, mean, stddev)
    end function hiprandGenerateNormal_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGenerateNormalDouble_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormalDouble_assumed_rank
      type(hiprandGenerator_t) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateNormalDouble_assumed_rank = hiprandGenerateNormalDouble_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function hiprandGenerateNormalDouble_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormalDouble_rank_0
      type(hiprandGenerator_t) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateNormalDouble_rank_0 = hiprandGenerateNormalDouble_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

    function hiprandGenerateNormalDouble_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateNormalDouble_rank_1
      type(hiprandGenerator_t) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateNormalDouble_rank_1 = hiprandGenerateNormalDouble_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
    function hiprandGenerateNormalDouble_typed(generator, output_data, n, mean, &
        stddev) result(GenerateNormalDouble)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_double), value :: mean
      real(c_double), value :: stddev
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateNormalDouble
      GenerateNormalDouble = hiprandGenerateNormalDouble_(generator%ptr, output_data, n, mean, &
        stddev)
    end function hiprandGenerateNormalDouble_typed

    function hiprandGenerateNormalHalf_typed(generator, output_data, n, mean, &
        stddev) result(GenerateNormalHalf)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(c_short), value :: mean
      integer(c_short), value :: stddev
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateNormalHalf
      GenerateNormalHalf = hiprandGenerateNormalHalf_(generator%ptr, output_data, n, mean, stddev)
    end function hiprandGenerateNormalHalf_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGenerateLogNormal_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormal_assumed_rank
      type(hiprandGenerator_t) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateLogNormal_assumed_rank = hiprandGenerateLogNormal_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function hiprandGenerateLogNormal_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormal_rank_0
      type(hiprandGenerator_t) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateLogNormal_rank_0 = hiprandGenerateLogNormal_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

    function hiprandGenerateLogNormal_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormal_rank_1
      type(hiprandGenerator_t) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      hiprandGenerateLogNormal_rank_1 = hiprandGenerateLogNormal_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
    function hiprandGenerateLogNormal_typed(generator, output_data, n, mean, &
        stddev) result(GenerateLogNormal)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_float), value :: mean
      real(c_float), value :: stddev
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateLogNormal
      GenerateLogNormal = hiprandGenerateLogNormal_(generator%ptr, output_data, n, mean, stddev)
    end function hiprandGenerateLogNormal_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGenerateLogNormalDouble_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormalDouble_assumed_rank
      type(hiprandGenerator_t) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateLogNormalDouble_assumed_rank = hiprandGenerateLogNormalDouble_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function hiprandGenerateLogNormalDouble_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormalDouble_rank_0
      type(hiprandGenerator_t) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateLogNormalDouble_rank_0 = hiprandGenerateLogNormalDouble_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

    function hiprandGenerateLogNormalDouble_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGenerateLogNormalDouble_rank_1
      type(hiprandGenerator_t) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      hiprandGenerateLogNormalDouble_rank_1 = hiprandGenerateLogNormalDouble_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
    function hiprandGenerateLogNormalDouble_typed(generator, output_data, n, mean, &
        stddev) result(GenerateLogNormalDouble)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_double), value :: mean
      real(c_double), value :: stddev
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateLogNormalDouble
      GenerateLogNormalDouble = hiprandGenerateLogNormalDouble_(generator%ptr, output_data, n, &
        mean, stddev)
    end function hiprandGenerateLogNormalDouble_typed

    function hiprandGenerateLogNormalHalf_typed(generator, output_data, n, mean, &
        stddev) result(GenerateLogNormalHalf)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(c_short), value :: mean
      integer(c_short), value :: stddev
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateLogNormalHalf
      GenerateLogNormalHalf = hiprandGenerateLogNormalHalf_(generator%ptr, output_data, n, mean, &
        stddev)
    end function hiprandGenerateLogNormalHalf_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function hiprandGeneratePoisson_assumed_rank(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGeneratePoisson_assumed_rank
      type(hiprandGenerator_t) :: generator
      integer(c_int),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      hiprandGeneratePoisson_assumed_rank = hiprandGeneratePoisson_(generator%ptr, &
        c_loc(output_data),n,lambda)
    end function

#else
    function hiprandGeneratePoisson_rank_0(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGeneratePoisson_rank_0
      type(hiprandGenerator_t) :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      hiprandGeneratePoisson_rank_0 = hiprandGeneratePoisson_(generator%ptr,c_loc(output_data),n, &
        lambda)
    end function

    function hiprandGeneratePoisson_rank_1(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: hiprandGeneratePoisson_rank_1
      type(hiprandGenerator_t) :: generator
      integer(c_int),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      hiprandGeneratePoisson_rank_1 = hiprandGeneratePoisson_(generator%ptr,c_loc(output_data),n, &
        lambda)
    end function

#endif
    function hiprandGeneratePoisson_typed(generator, output_data, n, lambda) result(GeneratePoisson)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_double), value :: lambda
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GeneratePoisson
      GeneratePoisson = hiprandGeneratePoisson_(generator%ptr, output_data, n, lambda)
    end function hiprandGeneratePoisson_typed

    function hiprandGenerateSeeds_typed(generator) result(GenerateSeeds)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: GenerateSeeds
      GenerateSeeds = hiprandGenerateSeeds_(generator%ptr)
    end function hiprandGenerateSeeds_typed

    function hiprandSetStream_typed(generator, stream) result(SetStream)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      type(c_ptr), value :: stream
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: SetStream
      SetStream = hiprandSetStream_(generator%ptr, stream)
    end function hiprandSetStream_typed

    function hiprandSetPseudoRandomGeneratorSeed_typed(generator, &
        seed) result(SetPseudoRandomGeneratorSeed)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      integer(c_int64_t), value :: seed
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: SetPseudoRandomGeneratorSeed
      SetPseudoRandomGeneratorSeed = hiprandSetPseudoRandomGeneratorSeed_(generator%ptr, seed)
    end function hiprandSetPseudoRandomGeneratorSeed_typed

    function hiprandSetGeneratorOffset_typed(generator, offset) result(SetGeneratorOffset)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      integer(c_int64_t), value :: offset
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: SetGeneratorOffset
      SetGeneratorOffset = hiprandSetGeneratorOffset_(generator%ptr, offset)
    end function hiprandSetGeneratorOffset_typed

    function hiprandSetGeneratorOrdering_typed(generator, order) result(SetGeneratorOrdering)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      integer(kind(HIPRAND_ORDERING_PSEUDO_BEST)), value :: order
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: SetGeneratorOrdering
      SetGeneratorOrdering = hiprandSetGeneratorOrdering_(generator%ptr, order)
    end function hiprandSetGeneratorOrdering_typed

    function hiprandSetQuasiRandomGeneratorDimensions_typed(generator, &
        dimensions) result(SetQuasiRandomGeneratorDimensions)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandGenerator_t), value :: generator
      integer(c_int), value :: dimensions
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: SetQuasiRandomGeneratorDimensions
      SetQuasiRandomGeneratorDimensions = hiprandSetQuasiRandomGeneratorDimensions_(generator%ptr, &
        dimensions)
    end function hiprandSetQuasiRandomGeneratorDimensions_typed

    function hiprandCreatePoissonDistribution_typed(lambda, &
        discrete_distribution) result(CreatePoissonDistribution)
      use, intrinsic :: iso_c_binding
      implicit none
      real(c_double), value :: lambda
      type(hiprandDiscreteDistribution_t) :: discrete_distribution
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: CreatePoissonDistribution
      CreatePoissonDistribution = hiprandCreatePoissonDistribution_(lambda, &
        discrete_distribution%ptr)
    end function hiprandCreatePoissonDistribution_typed

    function hiprandDestroyDistribution_typed(discrete_distribution) result(DestroyDistribution)
      use, intrinsic :: iso_c_binding
      implicit none
      type(hiprandDiscreteDistribution_t), value :: discrete_distribution
      integer(kind(HIPRAND_STATUS_SUCCESS)) :: DestroyDistribution
      DestroyDistribution = hiprandDestroyDistribution_(discrete_distribution%ptr)
    end function hiprandDestroyDistribution_typed

#endif
end module hiprand
