!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! ==============================================================================
! hipfort: FORTRAN Interfaces for GPU kernels
! ==============================================================================
! Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
          
           
module hipfort_rocrand
  use hipfort_rocrand_enums
  implicit none

 
  !>  \brief Creates a new random number generator.
  !> 
  !>  Creates a new pseudo random number generator of type \p rng_type
  !>  and returns it in \p generator.
  !> 
  !>  Values for \p rng_type are:
  !>  - ROCRAND_RNG_PSEUDO_XORWOW
  !>  - ROCRAND_RNG_PSEUDO_MRG32K3A
  !>  - ROCRAND_RNG_PSEUDO_MTGP32
  !>  - ROCRAND_RNG_PSEUDO_PHILOX4_32_10
  !>  - ROCRAND_RNG_QUASI_SOBOL32
  !> 
  !>  \param generator - Pointer to generator
  !>  \param rng_type - Type of generator to create
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated \n
  !>  - ROCRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
  !>    dynamically linked library version \n
  !>  - ROCRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
  !>  - ROCRAND_STATUS_SUCCESS if generator was created successfully \n
  !>
  interface rocrand_create_generator
    function rocrand_create_generator_(generator,rng_type) bind(c, name="rocrand_create_generator")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_generator_
      type(c_ptr) :: generator
      integer(kind(ROCRAND_RNG_PSEUDO_DEFAULT)),value :: rng_type
    end function

  end interface
  !>  \brief Destroys random number generator.
  !> 
  !>  Destroys random number generator and frees related memory.
  !> 
  !>  \param generator - Generator to be destroyed
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_SUCCESS if generator was destroyed successfully \n
  interface rocrand_destroy_generator
    function rocrand_destroy_generator_(generator) bind(c, name="rocrand_destroy_generator")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_destroy_generator_
      type(c_ptr),value :: generator
    end function

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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate
    function rocrand_generate_(generator,output_data,n) bind(c, name="rocrand_generate")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_char
    function rocrand_generate_char_(generator,output_data,n) bind(c, name="rocrand_generate_char")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_char_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_short
    function rocrand_generate_short_(generator,output_data,n) bind(c, name="rocrand_generate_short")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_short_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

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
  !>  \param n - Number of <tt>float</tt>s to generate
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_uniform
    function rocrand_generate_uniform_(generator,output_data,n) bind(c, name="rocrand_generate_uniform")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

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
  !>  \param n - Number of <tt>double</tt>s to generate
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_uniform_double
    function rocrand_generate_uniform_double_(generator,output_data,n) bind(c, name="rocrand_generate_uniform_double")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_uniform_double_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
    end function

  end interface
  !>  \brief Generates normally distributed \p float values.
  !> 
  !>  Generates \p n normally distributed distributed 32-bit floating-point
  !>  values and saves them to \p output_data.
  !> 
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of <tt>float</tt>s to generate
  !>  \param mean - Mean value of normal distribution
  !>  \param stddev - Standard deviation value of normal distribution
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_normal
    function rocrand_generate_normal_(generator,output_data,n,mean,stddev) bind(c, name="rocrand_generate_normal")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_float),value :: mean
      real(c_float),value :: stddev
    end function

  end interface
  !>  \brief Generates normally distributed \p double values.
  !> 
  !>  Generates \p n normally distributed 64-bit double-precision floating-point
  !>  numbers and saves them to \p output_data.
  !> 
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of <tt>double</tt>s to generate
  !>  \param mean - Mean value of normal distribution
  !>  \param stddev - Standard deviation value of normal distribution
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_normal_double
    function rocrand_generate_normal_double_(generator,output_data,n,mean,stddev) bind(c, name="rocrand_generate_normal_double")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_normal_double_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: mean
      real(c_double),value :: stddev
    end function

  end interface
  !>  \brief Generates log-normally distributed \p float values.
  !> 
  !>  Generates \p n log-normally distributed 32-bit floating-point values
  !>  and saves them to \p output_data.
  !> 
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of <tt>float</tt>s to generate
  !>  \param mean - Mean value of log normal distribution
  !>  \param stddev - Standard deviation value of log normal distribution
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_log_normal
    function rocrand_generate_log_normal_(generator,output_data,n,mean,stddev) bind(c, name="rocrand_generate_log_normal")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_float),value :: mean
      real(c_float),value :: stddev
    end function

  end interface
  !>  \brief Generates log-normally distributed \p double values.
  !> 
  !>  Generates \p n log-normally distributed 64-bit double-precision floating-point
  !>  values and saves them to \p output_data.
  !> 
  !>  \param generator - Generator to use
  !>  \param output_data - Pointer to memory to store generated numbers
  !>  \param n - Number of <tt>double</tt>s to generate
  !>  \param mean - Mean value of log normal distribution
  !>  \param stddev - Standard deviation value of log normal distribution
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_log_normal_double
    function rocrand_generate_log_normal_double_(generator,output_data,n,mean,stddev) bind(c, name="rocrand_generate_log_normal_double")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_log_normal_double_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: mean
      real(c_double),value :: stddev
    end function

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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
  !>  - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
  !>  of used quasi-random generator \n
  !>  - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
  interface rocrand_generate_poisson
    function rocrand_generate_poisson_(generator,output_data,n,lambda) bind(c, name="rocrand_generate_poisson")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_generate_poisson_
      type(c_ptr),value :: generator
      type(c_ptr),value :: output_data
      integer(c_size_t),value :: n
      real(c_double),value :: lambda
    end function

  end interface
  !>  \brief Initializes the generator's state on GPU or host.
  !> 
  !>  Initializes the generator's state on GPU or host. User it not
  !>  required to call this function before using a generator.
  !> 
  !>  If rocrand_initialize() was not called for a generator, it will be
  !>  automatically called by functions which generates random numbers like
  !>  rocrand_generate(), rocrang_generate_uniform() etc.
  !> 
  !>  \param generator - Generator to initialize
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
  !>  - ROCRAND_STATUS_SUCCESS if the seeds were generated successfully \n
  interface rocrand_initialize_generator
    function rocrand_initialize_generator_(generator) bind(c, name="rocrand_initialize_generator")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_initialize_generator_
      type(c_ptr),value :: generator
    end function

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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_SUCCESS if stream was set successfully \n
  interface rocrand_set_stream
    function rocrand_set_stream_(generator,stream) bind(c, name="rocrand_set_stream")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_stream_
      type(c_ptr),value :: generator
      type(c_ptr),value :: stream
    end function

  end interface
  !>  \brief Sets the seed of a pseudo-random number generator.
  !> 
  !>  Sets the seed of the pseudo-random number generator.
  !> 
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's offset.
  !> 
  !>  For a MRG32K3a generator seed value can't be zero. If \p seed is
  !>  equal zero and generator's type is ROCRAND_RNG_PSEUDO_MRG32K3A,
  !>  value \p 12345 is used as a seed instead.
  !> 
  !>  \param generator - Pseudo-random number generator
  !>  \param seed - New seed value
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_TYPE_ERROR if the generator is a quasi-random number generator \n
  !>  - ROCRAND_STATUS_SUCCESS if seed was set successfully \n
  interface rocrand_set_seed
    function rocrand_set_seed_(generator,seed) bind(c, name="rocrand_set_seed")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_seed_
      type(c_ptr),value :: generator
      integer(c_long_long),value :: seed
    end function

  end interface
  !>  \brief Sets the offset of a random number generator.
  !> 
  !>  Sets the absolute offset of the random number generator.
  !> 
  !>  - This operation resets the generator's internal state.
  !>  - This operation does not change the generator's seed.
  !> 
  !>  Absolute offset cannot be set if generator's type is ROCRAND_RNG_PSEUDO_MTGP32.
  !> 
  !>  \param generator - Random number generator
  !>  \param offset - New absolute offset
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_SUCCESS if offset was successfully set \n
  !>  - ROCRAND_STATUS_TYPE_ERROR if generator's type is ROCRAND_RNG_PSEUDO_MTGP32
  interface rocrand_set_offset
    function rocrand_set_offset_(generator,offset) bind(c, name="rocrand_set_offset")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_offset_
      type(c_ptr),value :: generator
      integer(c_long_long),value :: offset
    end function

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
  !>  - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
  !>  - ROCRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator \n
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range \n
  !>  - ROCRAND_STATUS_SUCCESS if the number of dimensions was set successfully \n
  interface rocrand_set_quasi_random_generator_dimensions
    function rocrand_set_quasi_random_generator_dimensions_(generator,dimensions) bind(c, name="rocrand_set_quasi_random_generator_dimensions")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_set_quasi_random_generator_dimensions_
      type(c_ptr),value :: generator
      integer(c_int),value :: dimensions
    end function

  end interface
  !>  \brief Returns the version number of the library.
  !> 
  !>  Returns in \p version the version number of the dynamically linked
  !>  rocRAND library.
  !> 
  !>  \param version - Version of the library
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p version is NULL \n
  !>  - ROCRAND_STATUS_SUCCESS if the version number was successfully returned \n
  interface rocrand_get_version
    function rocrand_get_version_(version) bind(c, name="rocrand_get_version")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_get_version_
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
  !>  - ROCRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
  !>  - ROCRAND_STATUS_SUCCESS if the histogram was ructed successfully \n
  interface rocrand_create_poisson_distribution
    function rocrand_create_poisson_distribution_(lambda,discrete_distribution) bind(c, name="rocrand_create_poisson_distribution")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_poisson_distribution_
      real(c_double),value :: lambda
      type(c_ptr) :: discrete_distribution
    end function

  end interface
  !>  \brief Construct the histogram for a custom discrete distribution.
  !> 
  !>  Construct the histogram for the discrete distribution of \p size
  !>  32-bit unsigned integers from the range [\p offset, \p offset + \p size)
  !>  using \p probabilities as probabilities.
  !> 
  !>  \param probabilities - probabilities of the the distribution in host memory
  !>  \param size - size of \p probabilities
  !>  \param offset - offset of values
  !>  \param discrete_distribution - pointer to the histogram in device memory
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p size was zero \n
  !>  - ROCRAND_STATUS_SUCCESS if the histogram was ructed successfully \n
  interface rocrand_create_discrete_distribution
    function rocrand_create_discrete_distribution_(probabilities,mySize,offset,discrete_distribution) bind(c, name="rocrand_create_discrete_distribution")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_create_discrete_distribution_
      type(c_ptr),value :: probabilities
      integer(c_int),value :: mySize
      integer(c_int),value :: offset
      type(c_ptr) :: discrete_distribution
    end function

  end interface
  !>  \brief Destroy the histogram array for a discrete distribution.
  !> 
  !>  Destroy the histogram array for a discrete distribution created by
  !>  rocrand_create_poisson_distribution.
  !> 
  !>  \param discrete_distribution - pointer to the histogram in device memory
  !> 
  !>  \return
  !>  - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null \n
  !>  - ROCRAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
  interface rocrand_destroy_discrete_distribution
    function rocrand_destroy_discrete_distribution_(discrete_distribution) bind(c, name="rocrand_destroy_discrete_distribution")
      use iso_c_binding
      use hipfort_rocrand_enums
      implicit none
      integer(kind(ROCRAND_STATUS_SUCCESS)) :: rocrand_destroy_discrete_distribution_
      type(c_ptr),value :: discrete_distribution
    end function

  end interface

#ifdef USE_FPOINTER_INTERFACES

  
#endif
end module hipfort_rocrand