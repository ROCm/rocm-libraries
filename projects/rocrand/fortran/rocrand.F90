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

  integer(c_int), parameter :: ROCRAND_VERSION = 400500
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


  type, bind(c) :: rocrand_generator_t
    type(c_ptr) :: ptr
  end type rocrand_generator_t

  interface

    !---------------------------------------------
    ! rocrand_create_poisson_distribution
    !---------------------------------------------
    !>  \brief Construct the histogram for a Poisson distribution.
    !>
    !>  Construct the histogram for the Poisson distribution with lambda \p lambda.
    !>
    !>  \param lambda - lambda for the Poisson distribution
    !>  \param discrete_distribution - pointer to the histogram in device memory
    !>
    !>  \return
    !>  - ROCRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated
    !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null
    !>  - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive
    !>  - ROCRAND_STATUS_SUCCESS if the histogram was constructed successfully
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
    !>  \brief Destroy the histogram array for a discrete distribution.
    !>
    !>  Destroy the histogram array for a discrete distribution created by
    !>  rocrand_create_poisson_distribution.
    !>
    !>  \param discrete_distribution - pointer to the histogram in device memory
    !>
    !>  \return
    !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null
    !>  - ROCRAND_STATUS_SUCCESS if the histogram was destroyed successfully
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
    !>  \brief Get the vector for 32-bit (scrambled-)sobol generation.
    !>
    !>  \param vectors - location where to write the vector pointer to
    !>
    !>  \param set - which direction vector set to use
    !>
    !>  \return
    !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p set was invalid for this method
    !>  - ROCRAND_STATUS_SUCCESS if the pointer was set succesfully
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
    !>  \brief Get the vector for 64-bit (scrambled-)sobol generation.
    !>
    !>  \param vectors - location where to write the vector pointer to
    !>
    !>  \param set - which direction vector set to use
    !>
    !>  \return
    !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p set was invalid for this method
    !>  - ROCRAND_STATUS_SUCCESS if the pointer was set succesfully
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
    !>  \brief Get the scramble constants for 32-bit scrambled sobol generation.
    !>
    !>  \param constants - location where to write the constants pointer to
    !>
    !>  \return
    !>  - ROCRAND_STATUS_SUCCESS if the pointer was set succesfully
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
    !>  \brief Get the scramble constants for 64-bit scrambled sobol generation.
    !>
    !>  \param constants - location where to write the constants pointer to
    !>
    !>  \return
    !>  - ROCRAND_STATUS_SUCCESS if the pointer was set succesfully
    function rocrand_get_scramble_constants64(constants) &
       result(get_scramble_constants64) &
       bind(C, name="rocrand_get_scramble_constants64")
       import :: c_ptr, ROCRAND_STATUS_SUCCESS
       type(c_ptr) :: constants
       integer(kind(ROCRAND_STATUS_SUCCESS)) :: get_scramble_constants64
    end function rocrand_get_scramble_constants64

  end interface

  !>  \brief Creates a new random number generator.
  !>
  !>  Creates a new pseudo random number generator of type \p rng_type
  !>  and returns it in \p generator.
  !>
  !>  Values for \p rng_type are:
  !>  - ROCRAND_RNG_PSEUDO_XORWOW
  !>  - ROCRAND_RNG_PSEUDO_MRG31K3P
  !>  - ROCRAND_RNG_PSEUDO_MRG32K3A
  !>  - ROCRAND_RNG_PSEUDO_MTGP32
  !>  - ROCRAND_RNG_PSEUDO_PHILOX4_32_10
  !>  - ROCRAND_RNG_PSEUDO_LFSR113
  !>  - ROCRAND_RNG_PSEUDO_THREEFRY2_32_20
  !>  - ROCRAND_RNG_PSEUDO_THREEFRY2_64_20
  !>  - ROCRAND_RNG_PSEUDO_THREEFRY4_32_20
  !>  - ROCRAND_RNG_PSEUDO_THREEFRY4_64_20
  !>  - ROCRAND_RNG_QUASI_SOBOL32
  !>  - ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32
  !>  - ROCRAND_RNG_QUASI_SOBOL64
  !>  - ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64
  !>
  !>  \param generator - Pointer to generator
  !>  \param rng_type - Type of generator to create
  !>
  !>  \return
  !>  - ROCRAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated
  !>  - ROCRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
  !>    dynamically linked library version
  !>  - ROCRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid
  !>  - ROCRAND_STATUS_SUCCESS if generator was created successfully
  interface rocrand_create_generator
    function rocrand_create_generator_(generator,rng_type) bind(c, name="rocrand_create_generator")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_generator_
      type(c_ptr) :: generator
      integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)),value :: rng_type
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_create_generator_typed
#endif
  end interface

  !>  \brief Creates a new host random number generator.
  !>
  !>  Creates a new pseudo random number generator of type \p rng_type
  !>  and returns it in \p generator. This generator is executed on the host rather than
  !>  on a device, and it is enqueued on the stream associated with the generator.
  !>
  !>  All generators are supported.
  !>
  !>  \param generator - Pointer to generator
  !>  \param rng_type - Type of generator to create
  !>
  !>  \return
  !>  - ROCRAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated
  !>  - ROCRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
  !>    dynamically linked library version
  !>  - ROCRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid
  !>  - ROCRAND_STATUS_SUCCESS if generator was created successfully
  interface rocrand_create_generator_host
    function rocrand_create_generator_host_(generator,rng_type) &
        bind(c, name="rocrand_create_generator_host")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_generator_host_
      type(c_ptr) :: generator
      integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)),value :: rng_type
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_create_generator_host_typed
#endif
  end interface

  !>  \brief Creates a new host random number generator, similar to `rocrand_create_generator_host`.
  !>    The exception is that, instead of enqueuing the host function in the stream,
  !>    execution happens synchronously with respect to the calling thread and the stream is
  !>    ignored.
  interface rocrand_create_generator_host_blocking
    function rocrand_create_generator_host_blocking_(generator,rng_type) &
        bind(c, name="rocrand_create_generator_host_blocking")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_generator_host_blocking_
      type(c_ptr) :: generator
      integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)),value :: rng_type
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_create_generator_host_blocking_typed
#endif
  end interface

  !>  \brief Destroys random number generator.
  !>
  !>  Destroys random number generator and frees related memory.
  !>
  !>  \param generator - Generator to be destroyed
  !>
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_SUCCESS if generator was destroyed successfully
  interface rocrand_destroy_generator
    function rocrand_destroy_generator_(generator) bind(c, name="rocrand_destroy_generator")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_destroy_generator_
      type(c_ptr),value :: generator
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_destroy_generator_typed
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
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_assumed_rank,&
      rocrand_generate_typed
#else
    module procedure &
      rocrand_generate_rank_0,&
      rocrand_generate_rank_1,&
      rocrand_generate_typed
#endif
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
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_TYPE_ERROR if the generator can't natively generate 64-bit random numbers
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_long_long_assumed_rank,&
      rocrand_generate_long_long_typed
#else
    module procedure &
      rocrand_generate_long_long_rank_0,&
      rocrand_generate_long_long_rank_1,&
      rocrand_generate_long_long_typed
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface rocrand_generate_char
    function rocrand_generate_char_(generator,output_data,n) bind(c, name="rocrand_generate_char")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_char_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_generate_char_typed
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface rocrand_generate_short
    function rocrand_generate_short_(generator,output_data,n) bind(c, name="rocrand_generate_short")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_short_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_generate_short_typed
#endif
  end interface

  !>  \brief Generates uniformly distributed \p float values.
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_uniform_assumed_rank,&
      rocrand_generate_uniform_typed
#else
    module procedure &
      rocrand_generate_uniform_rank_0,&
      rocrand_generate_uniform_rank_1,&
      rocrand_generate_uniform_typed
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
  !>  \param n - Number of doubles to generate
  !>
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_uniform_double_assumed_rank,&
      rocrand_generate_uniform_double_typed
#else
    module procedure &
      rocrand_generate_uniform_double_rank_0,&
      rocrand_generate_uniform_double_rank_1,&
      rocrand_generate_uniform_double_typed
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface rocrand_generate_uniform_half
    function rocrand_generate_uniform_half_(generator,output_data,n) &
        bind(c, name="rocrand_generate_uniform_half")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_half_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_generate_uniform_half_typed
#endif
  end interface

  !>  \brief Generates normally distributed \p float values.
  !>
  !>  Generates \p n normally distributed distributed 32-bit floating-point
  !>  values and saves them to \p output_data.
  !>
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of floats to generate
  !>  \param mean - Mean value of normal distribution
  !>  \param stddev - Standard deviation value of normal distribution
  !>
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_normal_assumed_rank,&
      rocrand_generate_normal_typed
#else
    module procedure &
      rocrand_generate_normal_rank_0,&
      rocrand_generate_normal_rank_1,&
      rocrand_generate_normal_typed
#endif
#endif
  end interface

  !>  \brief Generates normally distributed \p double values.
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_normal_double_assumed_rank,&
      rocrand_generate_normal_double_typed
#else
    module procedure &
      rocrand_generate_normal_double_rank_0,&
      rocrand_generate_normal_double_rank_1,&
      rocrand_generate_normal_double_typed
#endif
#endif
  end interface

  !>  \brief Generates normally distributed \p half values.
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface rocrand_generate_normal_half
    function rocrand_generate_normal_half_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_normal_half")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_half_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      integer(c_short),value :: mean
      integer(c_short),value :: stddev
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_generate_normal_half_typed
#endif
  end interface

  !>  \brief Generates log-normally distributed \p float values.
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_log_normal_assumed_rank,&
      rocrand_generate_log_normal_typed
#else
    module procedure &
      rocrand_generate_log_normal_rank_0,&
      rocrand_generate_log_normal_rank_1,&
      rocrand_generate_log_normal_typed
#endif
#endif
  end interface

  !>  \brief Generates log-normally distributed \p double values.
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_log_normal_double_assumed_rank,&
      rocrand_generate_log_normal_double_typed
#else
    module procedure &
      rocrand_generate_log_normal_double_rank_0,&
      rocrand_generate_log_normal_double_rank_1,&
      rocrand_generate_log_normal_double_typed
#endif
#endif
  end interface

  !>  \brief Generates log-normally distributed \p half values.
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
  interface rocrand_generate_log_normal_half
    function rocrand_generate_log_normal_half_(generator,output_data,n,mean,stddev) &
        bind(c, name="rocrand_generate_log_normal_half")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_half_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      integer(c_short),value :: mean
      integer(c_short),value :: stddev
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_generate_log_normal_half_typed
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure &
      rocrand_generate_poisson_assumed_rank,&
      rocrand_generate_poisson_typed
#else
    module procedure &
      rocrand_generate_poisson_rank_0,&
      rocrand_generate_poisson_rank_1,&
      rocrand_generate_poisson_typed
#endif
#endif
  end interface

  !>  \brief Initializes the generator's state on GPU or host.
  !>
  !>  Initializes the generator's state on GPU or host. User it not
  !>  required to call this function before using a generator.
  !>
  !>  If rocrand_initialize() was not called for a generator, it will be
  !>  automatically called by functions which generates random numbers like
  !>  rocrand_generate(), rocrand_generate_uniform() etc.
  !>
  !>  \param generator - Generator to initialize
  !>
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed
  !>  - ROCRAND_STATUS_SUCCESS if the seeds were generated successfully
  interface rocrand_initialize_generator
    function rocrand_initialize_generator_(generator) bind(c, name="rocrand_initialize_generator")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_initialize_generator_
      type(c_ptr),value :: generator
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_initialize_generator_typed
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_SUCCESS if stream was set successfully
  interface rocrand_set_stream
    function rocrand_set_stream_(generator,stream) bind(c, name="rocrand_set_stream")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_stream_
      type(c_ptr),value :: generator
      type(c_ptr),value :: stream
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_set_stream_typed
#endif
  end interface

  !>  \brief Sets the seed of a pseudo-random number generator.
  !>
  !>  Sets the seed of the pseudo-random number generator.
  !>
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's offset.
  !>
  !>  For an MRG32K3a or MRG31K3p generator the seed value can't be zero. If \p seed is
  !>  equal to zero and generator's type is ROCRAND_RNG_PSEUDO_MRG32K3A or
  !>  ROCRAND_RNG_PSEUDO_MRG31K3P,
  !>  value \p 12345 is used as seed instead.
  !>
  !>  For a LFSR113 generator seed values must be larger than 1, 7, 15,
  !>  127. The \p seed upper and lower 32 bits used as first and
  !>  second seed value. If those values smaller than 2 and/or 8, those
  !>  are increased with 1 and/or 7.
  !>
  !>  \param generator - Pseudo-random number generator
  !>  \param seed - New seed value
  !>
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_TYPE_ERROR if the generator is a quasi-random number generator
  !>  - ROCRAND_STATUS_SUCCESS if seed was set successfully
  interface rocrand_set_seed
    function rocrand_set_seed_(generator,seed) bind(c, name="rocrand_set_seed")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_seed_
      type(c_ptr),value :: generator
      integer(c_int64_t),value :: seed
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_set_seed_typed
#endif
  end interface

  !>  \brief Sets the seeds of a pseudo-random number generator.
  !>
  !>  Sets the seed of the pseudo-random number generator. Currently only for LFSR113
  !>
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's offset.
  !>
  !>  Only usable for LFSR113.
  !>
  !>  For a LFSR113 generator seed values must be bigger than 1, 7, 15,
  !>  127. If those values smaller, than the requested minimum values [2, 8, 16, 128], then
  !>  it will be increased with the minimum values minus 1 [1, 7, 15, 127].
  !>
  !>  \param generator - Pseudo-random number generator
  !>  \param seed - New seed value
  !>
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_TYPE_ERROR if the generator is a quasi-random number generator
  !>  - ROCRAND_STATUS_SUCCESS if seed was set successfully
  interface rocrand_set_seed_uint4
    function rocrand_set_seed_uint4_(generator,seed) bind(c, name="rocrand_set_seed_uint4")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_seed_uint4_
      type(c_ptr),value :: generator
      type(uint4),value :: seed
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_set_seed_uint4_typed
#endif
  end interface

  !>  \brief Sets the offset of a random number generator.
  !>
  !>  Sets the absolute offset of the random number generator.
  !>
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's seed.
  !>
  !>  Absolute offset cannot be set if generator's type is ROCRAND_RNG_PSEUDO_MTGP32 or
  !>  ROCRAND_RNG_PSEUDO_LFSR113.
  !>
  !>  \param generator - Random number generator
  !>  \param offset - New absolute offset
  !>
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_SUCCESS if offset was successfully set
  !>  - ROCRAND_STATUS_TYPE_ERROR if generator's type is ROCRAND_RNG_PSEUDO_MTGP32 or
  !>  ROCRAND_RNG_PSEUDO_LFSR113
  interface rocrand_set_offset
    function rocrand_set_offset_(generator,offset) bind(c, name="rocrand_set_offset")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_offset_
      type(c_ptr),value :: generator
      integer(c_int64_t),value :: offset
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_set_offset_typed
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
  !>  The ordering choices for pseudorandom sequences are the following.
  !>  Note that not all generators support all orderings. For details, see
  !>  the Programmer's Guide in the documentation.
  !>  - ROCRAND_ORDERING_PSEUDO_DEFAULT
  !>  - ROCRAND_ORDERING_PSEUDO_LEGACY
  !>  - ROCRAND_ORDERING_PSEUDO_BEST
  !>  - ROCRAND_ORDERING_PSEUDO_SEEDED
  !>  - ROCRAND_ORDERING_PSEUDO_DYNAMIC
  !>
  !>  For quasirandom sequences there is only one ordering, ROCRAND_ORDERING_QUASI_DEFAULT.
  !>
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if the ordering is not valid
  !>  - ROCRAND_STATUS_SUCCESS if the ordering was successfully set
  !>  - ROCRAND_STATUS_TYPE_ERROR if generator's type is not valid
  interface rocrand_set_ordering
    function rocrand_set_ordering_(generator,order) bind(c, name="rocrand_set_ordering")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_ordering_
      type(c_ptr),value :: generator
      integer(kind(ROCRAND_ORDERING_PSEUDO_BEST)),value :: order
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_set_ordering_typed
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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created
  !>  - ROCRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range
  !>  - ROCRAND_STATUS_SUCCESS if the number of dimensions was set successfully
  interface rocrand_set_quasi_random_generator_dimensions
    function rocrand_set_quasi_random_generator_dimensions_(generator,dimensions) &
        bind(c, name="rocrand_set_quasi_random_generator_dimensions")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_quasi_random_generator_dimensions_
      type(c_ptr),value :: generator
      integer(c_int),value :: dimensions
    end function

#ifdef USE_FPOINTER_INTERFACES
    module procedure rocrand_set_quasi_random_generator_dimensions_typed
#endif
  end interface

  !>  \brief Returns the version number of the library.
  !>
  !>  Returns in \p version the version number of the dynamically linked
  !>  rocRAND library.
  !>
  !>  \param version - Version of the library
  !>
  !>  \return
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p version is NULL
  !>  - ROCRAND_STATUS_SUCCESS if the version number was successfully returned
  interface rocrand_get_version
    function rocrand_get_version_(version) bind(c, name="rocrand_get_version")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_get_version_
      integer(c_int) :: version
    end function

    function rocrand_get_version_dptr(version) bind(c, name="rocrand_get_version")
      use iso_c_binding
      import
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_get_version_dptr
      type(c_ptr),value :: version
    end function
  end interface

  !>  \brief Construct the histogram for a custom discrete distribution.
  !>
  !>  Construct the histogram for the discrete distribution of \p size
  !>  32-bit unsigned integers from the range [\p offset, \p offset + \p size)
  !>  using \p probabilities as probabilities.
  !>
  !>  \param probabilities - probabilities of the the distribution in host memory
  !>  \param mySize - size of \p probabilities
  !>  \param offset - offset of values
  !>  \param discrete_distribution - pointer to the histogram in device memory
  !>
  !>  \return
  !>  - ROCRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p size was zero
  !>  - ROCRAND_STATUS_SUCCESS if the histogram was constructed successfully
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

#ifdef USE_FPOINTER_INTERFACES
#ifdef USE_ASSUMED_RANK_INTERFACES
    module procedure rocrand_create_discrete_distribution_assumed_rank
#else
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

#ifdef USE_FPOINTER_INTERFACES

    function rocrand_create_generator_typed(generator, rng_type) result(create_generator)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t) :: generator
      integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)), value :: rng_type
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: create_generator
      create_generator = rocrand_create_generator_(generator%ptr, rng_type)
    end function rocrand_create_generator_typed

    function rocrand_create_generator_host_typed(generator, rng_type) result(create_generator_host)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t) :: generator
      integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)), value :: rng_type
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: create_generator_host
      create_generator_host = rocrand_create_generator_host_(generator%ptr, rng_type)
    end function rocrand_create_generator_host_typed

    function rocrand_create_generator_host_blocking_typed(generator, &
        rng_type) result(create_generator_host_blocking)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t) :: generator
      integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)), value :: rng_type
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: create_generator_host_blocking
      create_generator_host_blocking = rocrand_create_generator_host_blocking_(generator%ptr, &
        rng_type)
    end function rocrand_create_generator_host_blocking_typed

    function rocrand_destroy_generator_typed(generator) result(destroy_generator)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: destroy_generator
      destroy_generator = rocrand_destroy_generator_(generator%ptr)
    end function rocrand_destroy_generator_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_assumed_rank
      type(rocrand_generator_t) :: generator
      integer(c_int),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_assumed_rank = rocrand_generate_(generator%ptr,c_loc(output_data),n)
    end function

#else
    function rocrand_generate_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_rank_0
      type(rocrand_generator_t) :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_rank_0 = rocrand_generate_(generator%ptr,c_loc(output_data),n)
    end function

    function rocrand_generate_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_rank_1
      type(rocrand_generator_t) :: generator
      integer(c_int),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_rank_1 = rocrand_generate_(generator%ptr,c_loc(output_data),n)
    end function

#endif
    function rocrand_generate_typed(generator, output_data, n) result(generate)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate
      generate = rocrand_generate_(generator%ptr, output_data, n)
    end function rocrand_generate_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_long_long_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_long_long_assumed_rank
      type(rocrand_generator_t) :: generator
      integer(c_int64_t),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_long_long_assumed_rank = rocrand_generate_long_long_(generator%ptr, &
        c_loc(output_data),n)
    end function

#else
    function rocrand_generate_long_long_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_long_long_rank_0
      type(rocrand_generator_t) :: generator
      integer(c_int64_t),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_long_long_rank_0 = rocrand_generate_long_long_(generator%ptr, &
        c_loc(output_data),n)
    end function

    function rocrand_generate_long_long_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_long_long_rank_1
      type(rocrand_generator_t) :: generator
      integer(c_int64_t),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_long_long_rank_1 = rocrand_generate_long_long_(generator%ptr, &
        c_loc(output_data),n)
    end function

#endif
    function rocrand_generate_long_long_typed(generator, output_data, n) result(generate_long_long)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_long_long
      generate_long_long = rocrand_generate_long_long_(generator%ptr, output_data, n)
    end function rocrand_generate_long_long_typed

    function rocrand_generate_char_typed(generator, output_data, n) result(generate_char)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_char
      generate_char = rocrand_generate_char_(generator%ptr, output_data, n)
    end function rocrand_generate_char_typed

    function rocrand_generate_short_typed(generator, output_data, n) result(generate_short)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_short
      generate_short = rocrand_generate_short_(generator%ptr, output_data, n)
    end function rocrand_generate_short_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_uniform_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_assumed_rank
      type(rocrand_generator_t) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_assumed_rank = rocrand_generate_uniform_(generator%ptr, &
        c_loc(output_data),n)
    end function

#else
    function rocrand_generate_uniform_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_rank_0
      type(rocrand_generator_t) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_rank_0 = rocrand_generate_uniform_(generator%ptr, &
        c_loc(output_data),n)
    end function

    function rocrand_generate_uniform_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_rank_1
      type(rocrand_generator_t) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_rank_1 = rocrand_generate_uniform_(generator%ptr, &
        c_loc(output_data),n)
    end function

#endif
    function rocrand_generate_uniform_typed(generator, output_data, n) result(generate_uniform)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_uniform
      generate_uniform = rocrand_generate_uniform_(generator%ptr, output_data, n)
    end function rocrand_generate_uniform_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_uniform_double_assumed_rank(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_double_assumed_rank
      type(rocrand_generator_t) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_double_assumed_rank = rocrand_generate_uniform_double_( &
        generator%ptr,c_loc(output_data),n)
    end function

#else
    function rocrand_generate_uniform_double_rank_0(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_double_rank_0
      type(rocrand_generator_t) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_double_rank_0 = rocrand_generate_uniform_double_(generator%ptr, &
        c_loc(output_data),n)
    end function

    function rocrand_generate_uniform_double_rank_1(generator,output_data,n)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_double_rank_1
      type(rocrand_generator_t) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      !
      rocrand_generate_uniform_double_rank_1 = rocrand_generate_uniform_double_(generator%ptr, &
        c_loc(output_data),n)
    end function

#endif
    function rocrand_generate_uniform_double_typed(generator, output_data, &
        n) result(generate_uniform_double)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_uniform_double
      generate_uniform_double = rocrand_generate_uniform_double_(generator%ptr, output_data, n)
    end function rocrand_generate_uniform_double_typed

    function rocrand_generate_uniform_half_typed(generator, output_data, &
        n) result(generate_uniform_half)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_uniform_half
      generate_uniform_half = rocrand_generate_uniform_half_(generator%ptr, output_data, n)
    end function rocrand_generate_uniform_half_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_normal_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_assumed_rank
      type(rocrand_generator_t) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_normal_assumed_rank = rocrand_generate_normal_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function rocrand_generate_normal_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_rank_0
      type(rocrand_generator_t) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_normal_rank_0 = rocrand_generate_normal_(generator%ptr,c_loc(output_data), &
        n,mean,stddev)
    end function

    function rocrand_generate_normal_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_rank_1
      type(rocrand_generator_t) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_normal_rank_1 = rocrand_generate_normal_(generator%ptr,c_loc(output_data), &
        n,mean,stddev)
    end function

#endif
    function rocrand_generate_normal_typed(generator, output_data, n, mean, &
        stddev) result(generate_normal)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_float), value :: mean
      real(c_float), value :: stddev
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_normal
      generate_normal = rocrand_generate_normal_(generator%ptr, output_data, n, mean, stddev)
    end function rocrand_generate_normal_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_normal_double_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_double_assumed_rank
      type(rocrand_generator_t) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_normal_double_assumed_rank = rocrand_generate_normal_double_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function rocrand_generate_normal_double_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_double_rank_0
      type(rocrand_generator_t) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_normal_double_rank_0 = rocrand_generate_normal_double_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

    function rocrand_generate_normal_double_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_double_rank_1
      type(rocrand_generator_t) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_normal_double_rank_1 = rocrand_generate_normal_double_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
    function rocrand_generate_normal_double_typed(generator, output_data, n, mean, &
        stddev) result(generate_normal_double)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_double), value :: mean
      real(c_double), value :: stddev
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_normal_double
      generate_normal_double = rocrand_generate_normal_double_(generator%ptr, output_data, n, &
        mean, stddev)
    end function rocrand_generate_normal_double_typed

    function rocrand_generate_normal_half_typed(generator, output_data, n, mean, &
        stddev) result(generate_normal_half)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(c_short), value :: mean
      integer(c_short), value :: stddev
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_normal_half
      generate_normal_half = rocrand_generate_normal_half_(generator%ptr, output_data, n, mean, &
        stddev)
    end function rocrand_generate_normal_half_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_log_normal_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_assumed_rank
      type(rocrand_generator_t) :: generator
      real(c_float),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_log_normal_assumed_rank = rocrand_generate_log_normal_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#else
    function rocrand_generate_log_normal_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_rank_0
      type(rocrand_generator_t) :: generator
      real(c_float),target :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_log_normal_rank_0 = rocrand_generate_log_normal_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

    function rocrand_generate_log_normal_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_rank_1
      type(rocrand_generator_t) :: generator
      real(c_float),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_float) :: mean
      real(c_float) :: stddev
      !
      rocrand_generate_log_normal_rank_1 = rocrand_generate_log_normal_(generator%ptr, &
        c_loc(output_data),n,mean,stddev)
    end function

#endif
    function rocrand_generate_log_normal_typed(generator, output_data, n, mean, &
        stddev) result(generate_log_normal)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_float), value :: mean
      real(c_float), value :: stddev
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_log_normal
      generate_log_normal = rocrand_generate_log_normal_(generator%ptr, output_data, n, mean, &
        stddev)
    end function rocrand_generate_log_normal_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_log_normal_double_assumed_rank(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_double_assumed_rank
      type(rocrand_generator_t) :: generator
      real(c_double),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_log_normal_double_assumed_rank = rocrand_generate_log_normal_double_( &
        generator%ptr,c_loc(output_data),n,mean,stddev)
    end function

#else
    function rocrand_generate_log_normal_double_rank_0(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_double_rank_0
      type(rocrand_generator_t) :: generator
      real(c_double),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_log_normal_double_rank_0 = rocrand_generate_log_normal_double_( &
        generator%ptr,c_loc(output_data),n,mean,stddev)
    end function

    function rocrand_generate_log_normal_double_rank_1(generator,output_data,n,mean,stddev)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_double_rank_1
      type(rocrand_generator_t) :: generator
      real(c_double),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: mean
      real(c_double) :: stddev
      !
      rocrand_generate_log_normal_double_rank_1 = rocrand_generate_log_normal_double_( &
        generator%ptr,c_loc(output_data),n,mean,stddev)
    end function

#endif
    function rocrand_generate_log_normal_double_typed(generator, output_data, n, mean, &
        stddev) result(generate_log_normal_double)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_double), value :: mean
      real(c_double), value :: stddev
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_log_normal_double
      generate_log_normal_double = rocrand_generate_log_normal_double_(generator%ptr, output_data, &
        n, mean, stddev)
    end function rocrand_generate_log_normal_double_typed

    function rocrand_generate_log_normal_half_typed(generator, output_data, n, mean, &
        stddev) result(generate_log_normal_half)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      integer(c_short), value :: mean
      integer(c_short), value :: stddev
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_log_normal_half
      generate_log_normal_half = rocrand_generate_log_normal_half_(generator%ptr, output_data, n, &
        mean, stddev)
    end function rocrand_generate_log_normal_half_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
    function rocrand_generate_poisson_assumed_rank(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_poisson_assumed_rank
      type(rocrand_generator_t) :: generator
      integer(c_int),target,contiguous,dimension(..) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      rocrand_generate_poisson_assumed_rank = rocrand_generate_poisson_(generator%ptr, &
        c_loc(output_data),n,lambda)
    end function

#else
    function rocrand_generate_poisson_rank_0(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_poisson_rank_0
      type(rocrand_generator_t) :: generator
      integer(c_int),target :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      rocrand_generate_poisson_rank_0 = rocrand_generate_poisson_(generator%ptr, &
        c_loc(output_data),n,lambda)
    end function

    function rocrand_generate_poisson_rank_1(generator,output_data,n,lambda)
      use iso_c_binding
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_poisson_rank_1
      type(rocrand_generator_t) :: generator
      integer(c_int),target,dimension(:) :: output_data
      integer(c_size_t) :: n
      real(c_double) :: lambda
      !
      rocrand_generate_poisson_rank_1 = rocrand_generate_poisson_(generator%ptr, &
        c_loc(output_data),n,lambda)
    end function

#endif
    function rocrand_generate_poisson_typed(generator, output_data, n, &
        lambda) result(generate_poisson)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: output_data
      integer(c_size_t), value :: n
      real(c_double), value :: lambda
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: generate_poisson
      generate_poisson = rocrand_generate_poisson_(generator%ptr, output_data, n, lambda)
    end function rocrand_generate_poisson_typed

    function rocrand_initialize_generator_typed(generator) result(initialize_generator)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: initialize_generator
      initialize_generator = rocrand_initialize_generator_(generator%ptr)
    end function rocrand_initialize_generator_typed

    function rocrand_set_stream_typed(generator, stream) result(set_stream)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(c_ptr), value :: stream
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_stream
      set_stream = rocrand_set_stream_(generator%ptr, stream)
    end function rocrand_set_stream_typed

    function rocrand_set_seed_typed(generator, seed) result(set_seed)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      integer(c_int64_t), value :: seed
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_seed
      set_seed = rocrand_set_seed_(generator%ptr, seed)
    end function rocrand_set_seed_typed

    function rocrand_set_seed_uint4_typed(generator, seed) result(set_seed_uint4)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      type(uint4), value :: seed
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_seed_uint4
      set_seed_uint4 = rocrand_set_seed_uint4_(generator%ptr, seed)
    end function rocrand_set_seed_uint4_typed

    function rocrand_set_offset_typed(generator, offset) result(set_offset)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      integer(c_int64_t), value :: offset
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_offset
      set_offset = rocrand_set_offset_(generator%ptr, offset)
    end function rocrand_set_offset_typed

    function rocrand_set_ordering_typed(generator, order) result(set_ordering)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      integer(kind(ROCRAND_ORDERING_PSEUDO_BEST)), value :: order
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_ordering
      set_ordering = rocrand_set_ordering_(generator%ptr, order)
    end function rocrand_set_ordering_typed

    function rocrand_set_quasi_random_generator_dimensions_typed(generator, &
        dimensions) result(set_quasi_random_generator_dimensions)
      use, intrinsic :: iso_c_binding
      implicit none
      type(rocrand_generator_t), value :: generator
      integer(c_int), value :: dimensions
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: set_quasi_random_generator_dimensions
      set_quasi_random_generator_dimensions = rocrand_set_quasi_random_generator_dimensions_( &
        generator%ptr, dimensions)
    end function rocrand_set_quasi_random_generator_dimensions_typed

#ifdef USE_ASSUMED_RANK_INTERFACES
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
