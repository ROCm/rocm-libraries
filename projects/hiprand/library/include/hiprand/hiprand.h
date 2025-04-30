// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPRAND_H_
#define HIPRAND_H_

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

/** \addtogroup hiprandhost
 *
 *  @{
 */

/// \cond HIPRAND_DOCS_MACRO
#ifndef HIPRANDAPI
    #if defined(_WIN32) && !defined(HIPRAND_STATIC_BUILD)
        #ifdef __clang__
            // Not all versions of clang recognizes __declspec(dllimport) and __declspec(dllexport)
            #define HIPRAND_DETAIL_HAS_DECLSPEC_ATTRIBUTE(attr) __has_declspec_attribute(attr)
        #else
            // We assume that the non-clang compilers on Windows do recognize
            #define HIPRAND_DETAIL_HAS_DECLSPEC_ATTRIBUTE(attr) 1
        #endif

        #if defined(hiprand_EXPORTS) && HIPRAND_DETAIL_HAS_DECLSPEC_ATTRIBUTE(dllexport)
            /* We are building this library */
            #define HIPRANDAPI __declspec(dllexport)
        #elif HIPRAND_DETAIL_HAS_DECLSPEC_ATTRIBUTE(dllimport)
            /* We are using this library */
            #define HIPRANDAPI __declspec(dllimport)
        #else
            #define HIPRANDAPI
        #endif

        #undef HIPRAND_DETAIL_HAS_DECLSPEC_ATTRIBUTE
    #else
        #define HIPRANDAPI
    #endif
#endif
/// \endcond

#include "hiprand/hiprand_version.h"
#ifdef HIPRAND_DOXYGEN // Only for the documentation
/// \def HIPRAND_VERSION
/// \brief hipRAND library version
///
/// Version number may not be visible in the documentation.
///
/// <tt>HIPRAND_VERSION % 100</tt> is the patch level,
/// <tt>HIPRAND_VERSION / 100 % 1000 </tt>is the minor version,
/// <tt>HIPRAND_VERSION / 100000</tt> is the major version.
///
/// For example, if \p HIPRAND_VERSION is \p 100500, then
/// the major version is \p 1, the minor version is \p 5, and
/// the patch level is \p 0.
#define HIPRAND_VERSION
#endif

#if defined(__HIP_PLATFORM_AMD__)
    #include "hiprand/hiprand_rocm.h"
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #include "hiprand/hiprand_nvcc.h"
#endif

/// \cond HIPRAND_DOCS_TYPEDEFS
/// \brief hipRAND random number generator (opaque)
typedef hiprandGenerator_st * hiprandGenerator_t;
/// \endcond

/// \cond HIPRAND_DOCS_TYPEDEFS
/// \brief hipRAND discrete distribution
typedef hiprandDiscreteDistribution_st * hiprandDiscreteDistribution_t;
/// \endcond

/// \cond HIPRAND_DOCS_TYPEDEFS
/// hipRAND half type (derived from HIP)
typedef __half half;
/// \endcond

#define HIPRAND_DEFAULT_MAX_BLOCK_SIZE 256
#define HIPRAND_DEFAULT_MIN_WARPS_PER_EU 1

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \brief hipRAND function call status type
 */
typedef enum hiprandStatus {
    HIPRAND_STATUS_SUCCESS = 0, ///< Success
    HIPRAND_STATUS_VERSION_MISMATCH = 100, ///< Header file and linked library version do not match
    HIPRAND_STATUS_NOT_INITIALIZED = 101, ///< Generator not created
    HIPRAND_STATUS_ALLOCATION_FAILED = 102, ///< Memory allocation failed
    HIPRAND_STATUS_TYPE_ERROR = 103, ///< Generator type is wrong
    HIPRAND_STATUS_OUT_OF_RANGE = 104, ///< Argument out of range
    HIPRAND_STATUS_LENGTH_NOT_MULTIPLE = 105, ///< Requested size is not a multiple of quasirandom generator's dimension,
                                              ///< or requested size is not even (see hiprandGenerateNormal()),
                                              ///< or pointer is misaligned (see hiprandGenerateNormal())
    HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106, ///< GPU does not have double precision
    HIPRAND_STATUS_LAUNCH_FAILURE = 201, ///< Kernel launch failure
    HIPRAND_STATUS_PREEXISTING_FAILURE = 202, ///< Preexisting failure on library entry
    HIPRAND_STATUS_INITIALIZATION_FAILED = 203, ///< Initialization of HIP failed
    HIPRAND_STATUS_ARCH_MISMATCH = 204, ///< Architecture mismatch, GPU does not support requested feature
    HIPRAND_STATUS_INTERNAL_ERROR = 999, ///< Internal library error
    HIPRAND_STATUS_NOT_IMPLEMENTED = 1000 ///< Feature not implemented yet
} hiprandStatus_t;

/**
 * \brief hipRAND generator type
 */
typedef enum hiprandRngType {
    /// \cond
    HIPRAND_RNG_TEST = 0,
    /// \endcond
    HIPRAND_RNG_PSEUDO_DEFAULT = 400, ///< Default pseudorandom generator
    HIPRAND_RNG_PSEUDO_XORWOW = 401, ///< XORWOW pseudorandom generator
    HIPRAND_RNG_PSEUDO_MRG32K3A = 402, ///< MRG32k3a pseudorandom generator
    HIPRAND_RNG_PSEUDO_MTGP32 = 403, ///< Mersenne Twister MTGP32 pseudorandom generator
    HIPRAND_RNG_PSEUDO_MT19937 = 404, ///< Mersenne Twister 19937
    HIPRAND_RNG_PSEUDO_PHILOX4_32_10 = 405, ///< PHILOX_4x32 (10 rounds) pseudorandom generator
    HIPRAND_RNG_QUASI_DEFAULT = 500, ///< Default quasirandom generator
    HIPRAND_RNG_QUASI_SOBOL32 = 501, ///< Sobol32 quasirandom generator
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 502,  ///< Scrambled Sobol32 quasirandom generator
    HIPRAND_RNG_QUASI_SOBOL64 = 503, ///< Sobol64 quasirandom generator
    HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 504  ///< Scrambled Sobol64 quasirandom generator
} hiprandRngType_t;

/**
 * \brief hipRAND generator ordering
 */
typedef enum hiprandOrdering
{
    HIPRAND_ORDERING_PSEUDO_BEST = 100, ///< Best ordering for pseudorandom results
    HIPRAND_ORDERING_PSEUDO_DEFAULT
    = 101, ///< Specific default thread sequence for pseudorandom results, same as HIPRAND_ORDERING_PSEUDO_BEST
    HIPRAND_ORDERING_PSEUDO_SEEDED
    = 102, ///< Specific seeding pattern for fast lower quality pseudorandom results
    HIPRAND_ORDERING_PSEUDO_LEGACY
    = 103, ///< Specific legacy sequence for pseudorandom results. This remains the same across releases, but not across backends.
    HIPRAND_ORDERING_PSEUDO_DYNAMIC
    = 104, ///< Specific ordering adjusted to the device it is being executed on, provides the best performance
    HIPRAND_ORDERING_QUASI_DEFAULT
    = 201 ///< Specific n-dimensional ordering for quasirandom results
} hiprandOrdering_t;

/**
 * \brief hipRAND vector set for quasirandom generators.
 */
typedef enum hiprandDirectionVectorSet
{
    HIPRAND_DIRECTION_VECTORS_32_JOEKUO6           = 101,
    HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6 = 102,
    HIPRAND_DIRECTION_VECTORS_64_JOEKUO6           = 103,
    HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6 = 104,
} hiprandDirectionVectorSet_t;

// Host API functions

/**
 * \brief Creates a new random number generator.
 *
 * Creates a new random number generator of type \p rng_type,
 * and returns it in \p generator. That generator will use
 * GPU to create random numbers.
 *
 * Values for \p rng_type are:
 * - HIPRAND_RNG_PSEUDO_DEFAULT
 * - HIPRAND_RNG_PSEUDO_XORWOW
 * - HIPRAND_RNG_PSEUDO_MRG32K3A
 * - HIPRAND_RNG_PSEUDO_MTGP32
 * - HIPRAND_RNG_PSEUDO_MT19937
 * - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
 * - HIPRAND_RNG_QUASI_DEFAULT
 * - HIPRAND_RNG_QUASI_SOBOL32
 * - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
 * - HIPRAND_RNG_QUASI_SOBOL64
 * - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
 *
 * \param generator Pointer to generator
 * \param rng_type Type of random number generator to create
 *
 * \return
 * - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed \n
 * - HIPRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
 * - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
 *   dynamically linked library version \n
 * - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
 * - HIPRAND_STATUS_NOT_IMPLEMENTED if generator of type \p rng_type is not implemented yet \n
 * - HIPRAND_STATUS_SUCCESS if generator was created successfully \n
 *
 */
hiprandStatus_t HIPRANDAPI
hiprandCreateGenerator(hiprandGenerator_t * generator, hiprandRngType_t rng_type);

/**
 * \brief Creates a new random number generator on host.
 *
 * Creates a new host random number generator of type \p rng_type
 * and returns it in \p generator. Created generator will use
 * host CPU to generate random numbers.
 *
 * Values for \p rng_type are:
 * - HIPRAND_RNG_PSEUDO_DEFAULT
 * - HIPRAND_RNG_PSEUDO_XORWOW
 * - HIPRAND_RNG_PSEUDO_MRG32K3A
 * - HIPRAND_RNG_PSEUDO_MTGP32
 * - HIPRAND_RNG_PSEUDO_MT19937
 * - HIPRAND_RNG_PSEUDO_PHILOX4_32_10
 * - HIPRAND_RNG_QUASI_DEFAULT
 * - HIPRAND_RNG_QUASI_SOBOL32
 * - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32
 * - HIPRAND_RNG_QUASI_SOBOL64
 * - HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64
 *
 * \param generator Pointer to generator
 * \param rng_type Type of random number generator to create
 *
 * \return
 * - HIPRAND_STATUS_ALLOCATION_FAILED, if memory allocation failed \n
 * - HIPRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
 *   dynamically linked library version \n
 * - HIPRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
 * - HIPRAND_STATUS_NOT_IMPLEMENTED if host generator of type \p rng_type is not implemented yet \n
 * - HIPRAND_STATUS_SUCCESS if generator was created successfully \n
 */
hiprandStatus_t HIPRANDAPI
hiprandCreateGeneratorHost(hiprandGenerator_t * generator, hiprandRngType_t rng_type);

/**
 * \brief Destroys random number generator.
 *
 * Destroys random number generator and frees related memory.
 *
 * \param generator Generator to be destroyed
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_SUCCESS if generator was destroyed successfully \n
 */
hiprandStatus_t HIPRANDAPI
hiprandDestroyGenerator(hiprandGenerator_t generator);

/**
 * \brief Generates uniformly distributed 32-bit unsigned integers.
 *
 * Generates \p n uniformly distributed 32-bit unsigned integers and
 * saves them to \p output_data.
 *
 * Generated numbers are between \p 0 and \p 2^32, including \p 0 and
 * excluding \p 2^32.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of 32-bit unsigned integers to generate
 *
 * Note: \p generator must be not be of type \p HIPRAND_RNG_QUASI_SOBOL64
 * or \p HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64.
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerate(hiprandGenerator_t generator,
                unsigned int * output_data, size_t n);

/**
 * \brief Generates uniformly distributed 8-bit unsigned integers.
 *
 * Generates \p n uniformly distributed 8-bit unsigned integers and
 * saves them to \p output_data.
 *
 * Generated numbers are between \p 0 and \p 2^8, including \p 0 and
 * excluding \p 2^8.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of 8-bit unsigned integers to generate
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateChar(hiprandGenerator_t generator,
                    unsigned char * output_data, size_t n);

/**
 * \brief Generates uniformly distributed 16-bit unsigned integers.
 *
 * Generates \p n uniformly distributed 16-bit unsigned integers and
 * saves them to \p output_data.
 *
 * Generated numbers are between \p 0 and \p 2^16, including \p 0 and
 * excluding \p 2^16.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of 16-bit unsigned integers to generate
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateShort(hiprandGenerator_t generator,
                     unsigned short * output_data, size_t n);

/**
 * \brief Generates uniformly distributed 64-bit unsigned integers.
 *
 * Generates \p n uniformly distributed 64-bit unsigned integers and
 * saves them to \p output_data.
 *
 * Generated numbers are between \p 0 and \p 2^64, including \p 0 and
 * excluding \p 2^64.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of 64-bit unsigned integers to generate
 *
 * Note: \p generator must be of type \p HIPRAND_RNG_QUASI_SOBOL64
 * or \p HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64.
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI hiprandGenerateLongLong(hiprandGenerator_t  generator,
                                                   unsigned long long* output_data,
                                                   size_t              n);

/**
 * \brief Generates uniformly distributed floats.
 *
 * Generates \p n uniformly distributed 32-bit floating-point values
 * and saves them to \p output_data.
 *
 * Generated numbers are between \p 0.0f and \p 1.0f, excluding \p 0.0f and
 * including \p 1.0f.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of floats to generate
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateUniform(hiprandGenerator_t generator,
                       float * output_data, size_t n);

/**
 * \brief Generates uniformly distributed double-precision floating-point values.
 *
 * Generates \p n uniformly distributed 64-bit double-precision floating-point
 * values and saves them to \p output_data.
 *
 * Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
 * including \p 1.0.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of floats to generate
 *
 * Note: When \p generator is of type: \p HIPRAND_RNG_PSEUDO_MRG32K3A,
 * \p HIPRAND_RNG_PSEUDO_MTGP32, \p HIPRAND_RNG_QUASI_SOBOL32, or
 * \p HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32 then the returned \p double
 * values are generated from only 32 random bits
 * each (one <tt>unsigned int</tt> value per one generated \p double).
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateUniformDouble(hiprandGenerator_t generator,
                             double * output_data, size_t n);

/**
* \brief Generates uniformly distributed half-precision floating-point values.
*
* Generates \p n uniformly distributed 16-bit half-precision floating-point
* values and saves them to \p output_data.
*
* Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
* including \p 1.0.
*
* \param generator Generator to use
* \param output_data Pointer to memory to store generated numbers
* \param n Number of halfs to generate
*
* \return
* - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
* - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
* - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
* of used quasi-random generator \n
* - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
*/
hiprandStatus_t HIPRANDAPI
hiprandGenerateUniformHalf(hiprandGenerator_t generator,
                           half * output_data, size_t n);

/**
 * \brief Generates normally distributed floats.
 *
 * Generates \p n normally distributed 32-bit floating-point
 * values and saves them to \p output_data.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of floats to generate
 * \param mean Mean value of normal distribution
 * \param stddev Standard deviation value of normal distribution
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
 * aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateNormal(hiprandGenerator_t generator,
                      float * output_data, size_t n,
                      float mean, float stddev);

/**
 * \brief Generates normally distributed doubles.
 *
 * Generates \p n normally distributed 64-bit double-precision floating-point
 * numbers and saves them to \p output_data.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of doubles to generate
 * \param mean Mean value of normal distribution
 * \param stddev Standard deviation value of normal distribution
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
 * aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateNormalDouble(hiprandGenerator_t generator,
                            double * output_data, size_t n,
                            double mean, double stddev);

/**
 * \brief Generates normally distributed halfs.
 *
 * Generates \p n normally distributed 16-bit half-precision floating-point
 * numbers and saves them to \p output_data.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of halfs to generate
 * \param mean Mean value of normal distribution
 * \param stddev Standard deviation value of normal distribution
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
 * aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateNormalHalf(hiprandGenerator_t generator,
                          half * output_data, size_t n,
                          half mean, half stddev);

/**
 * \brief Generates log-normally distributed floats.
 *
 * Generates \p n log-normally distributed 32-bit floating-point values
 * and saves them to \p output_data.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of floats to generate
 * \param mean Mean value of log normal distribution
 * \param stddev Standard deviation value of log normal distribution
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
 * aligned to \p sizeof(float2) bytes, or \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateLogNormal(hiprandGenerator_t generator,
                         float * output_data, size_t n,
                         float mean, float stddev);

/**
 * \brief Generates log-normally distributed doubles.
 *
 * Generates \p n log-normally distributed 64-bit double-precision floating-point
 * values and saves them to \p output_data.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of doubles to generate
 * \param mean Mean value of log normal distribution
 * \param stddev Standard deviation value of log normal distribution
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
 * aligned to \p sizeof(double2) bytes, or \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateLogNormalDouble(hiprandGenerator_t generator,
                               double * output_data, size_t n,
                               double mean, double stddev);

/**
* \brief Generates log-normally distributed halfs.
*
* Generates \p n log-normally distributed 16-bit half-precision floating-point
* values and saves them to \p output_data.
*
* \param generator Generator to use
* \param output_data Pointer to memory to store generated numbers
* \param n Number of halfs to generate
* \param mean Mean value of log normal distribution
* \param stddev Standard deviation value of log normal distribution
*
* \return
* - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
* - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
* - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not even, \p output_data is not
* aligned to \p sizeof(half2) bytes, or \p n is not a multiple of the dimension
* of used quasi-random generator \n
* - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
*/
hiprandStatus_t HIPRANDAPI
hiprandGenerateLogNormalHalf(hiprandGenerator_t generator,
                             half * output_data, size_t n,
                             half mean, half stddev);

/**
 * \brief Generates Poisson-distributed 32-bit unsigned integers.
 *
 * Generates \p n Poisson-distributed 32-bit unsigned integers and
 * saves them to \p output_data.
 *
 * \param generator Generator to use
 * \param output_data Pointer to memory to store generated numbers
 * \param n Number of 32-bit unsigned integers to generate
 * \param lambda lambda for the Poisson distribution
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
 * - HIPRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - HIPRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGeneratePoisson(hiprandGenerator_t generator,
                       unsigned int * output_data, size_t n,
                       double lambda);

/**
 * \brief Initializes the generator's state on GPU or host.
 *
 * Initializes the generator's state on GPU or host.
 *
 * If hiprandGenerateSeeds() was not called for a generator, it will be
 * automatically called by functions which generates random numbers like
 * hiprandGenerate(), hiprandGenerateUniform(), hiprandGenerateNormal() etc.
 *
 * \param generator Generator to initialize
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was never created \n
 * - HIPRAND_STATUS_PREEXISTING_FAILURE if there was an existing error from
 *   a previous kernel launch \n
 * - HIPRAND_STATUS_LAUNCH_FAILURE if the kernel launch failed for any reason \n
 * - HIPRAND_STATUS_SUCCESS if the seeds were generated successfully \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGenerateSeeds(hiprandGenerator_t generator);

/**
 * \brief Sets the current stream for kernel launches.
 *
 * Sets the current stream for all kernel launches of the generator.
 * All functions will use this stream.
 *
 * \param generator Generator to modify
 * \param stream Stream to use or NULL for default stream
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_SUCCESS if stream was set successfully \n
 */
hiprandStatus_t HIPRANDAPI
hiprandSetStream(hiprandGenerator_t generator, hipStream_t stream);

/**
 * \brief Sets the seed of a pseudo-random number generator.
 *
 * Sets the seed of the pseudo-random number generator.
 *
 * - This operation resets the generator's internal state.
 * - This operation does not change the generator's offset.
 *
 * \param generator Pseudo-random number generator
 * \param seed New seed value
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_TYPE_ERROR if the generator is a quasi random number generator \n
 * - HIPRAND_STATUS_SUCCESS if seed was set successfully \n
 */
hiprandStatus_t HIPRANDAPI
hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t generator, unsigned long long seed);

/**
 * \brief Sets the offset of a random number generator.
 *
 * Sets the absolute offset of the random number generator.
 *
 * - This operation resets the generator's internal state.
 * - This operation does not change the generator's seed.
 *
 * Absolute offset cannot be set if generator's type is
 * HIPRAND_RNG_PSEUDO_MTGP32 or HIPRAND_RNG_PSEUDO_MT19937.
 *
 * \param generator Random number generator
 * \param offset New absolute offset
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_SUCCESS if offset was successfully set \n
 * - HIPRAND_STATUS_TYPE_ERROR if generator's type is HIPRAND_RNG_PSEUDO_MTGP32
 * or HIPRAND_RNG_PSEUDO_MT19937 \n
 */
hiprandStatus_t HIPRANDAPI
hiprandSetGeneratorOffset(hiprandGenerator_t generator, unsigned long long offset);

/**
 * \brief Sets the ordering of a random number generator.
 *
 * Sets the ordering of the results of a random number generator.
 *
 * - This operation resets the generator's internal state.
 * - This operation does not change the generator's seed.
 *
 * \param generator Random number generator
 * \param order New ordering of results
 *
 * The ordering choices for pseudorandom sequences are
 * HIPRAND_ORDERING_PSEUDO_DEFAULT and
 * HIPRAND_ORDERING_PSEUDO_LEGACY.
 * The default ordering is HIPRAND_ORDERING_PSEUDO_DEFAULT, which is equal to
 * HIPRAND_ORDERING_PSEUDO_LEGACY for now.
 *
 * For quasirandom sequences there is only one ordering, HIPRAND_ORDERING_QUASI_DEFAULT.
 *
 * \return
 * - HIPRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - HIPRAND_STATUS_OUT_OF_RANGE if the ordering is not valid \n
 * - HIPRAND_STATUS_SUCCESS if the ordering was successfully set \n
 * - HIPRAND_STATUS_TYPE_ERROR if generator's type is not valid
 */
hiprandStatus_t HIPRANDAPI hiprandSetGeneratorOrdering(hiprandGenerator_t generator,
                                                       hiprandOrdering_t  order);

/**
 * \brief Set the number of dimensions of a quasi-random number generator.
 *
 * Set the number of dimensions of a quasi-random number generator.
 * Supported values of \p dimensions are 1 to 20000.
 *
 * - This operation resets the generator's internal state.
 * - This operation does not change the generator's offset.
 *
 * \param generator Quasi-random number generator
 * \param dimensions Number of dimensions
 *
 * \return
 * - HIPRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - HIPRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator \n
 * - HIPRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range \n
 * - HIPRAND_STATUS_SUCCESS if the number of dimensions was set successfully \n
 */
hiprandStatus_t HIPRANDAPI
hiprandSetQuasiRandomGeneratorDimensions(hiprandGenerator_t generator, unsigned int dimensions);

/**
 * \brief Returns the version number of the cuRAND or rocRAND library.
 *
 * Returns in \p version the version number of the underlying cuRAND or
 * rocRAND library.
 *
 * \param version Version of the library
 *
 * \return
 * - HIPRAND_STATUS_OUT_OF_RANGE if \p version is NULL \n
 * - HIPRAND_STATUS_SUCCESS if the version number was successfully returned \n
 */
hiprandStatus_t HIPRANDAPI
hiprandGetVersion(int * version);

/**
 * \brief Construct the histogram for a Poisson distribution.
 *
 * Construct the histogram for the Poisson distribution with lambda \p lambda.
 *
 * \param lambda lambda for the Poisson distribution
 * \param discrete_distribution pointer to the histogram in device memory
 *
 * \return
 * - HIPRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
 * - HIPRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
 * - HIPRAND_STATUS_SUCCESS if the histogram was constructed successfully \n
 */
hiprandStatus_t HIPRANDAPI
hiprandCreatePoissonDistribution(double lambda, hiprandDiscreteDistribution_t * discrete_distribution);

/**
 * \brief Destroy the histogram array for a discrete distribution.
 *
 * Destroy the histogram array for a discrete distribution created by
 * hiprandCreatePoissonDistribution.
 *
 * \param discrete_distribution pointer to the histogram in device memory
 *
 * \return
 * - HIPRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null \n
 * - HIPRAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
 */
hiprandStatus_t HIPRANDAPI
hiprandDestroyDistribution(hiprandDiscreteDistribution_t discrete_distribution);

hiprandStatus_t HIPRANDAPI hiprandGetDirectionVectors32(hiprandDirectionVectors32_t** vectors,
                                                        hiprandDirectionVectorSet_t   set);

hiprandStatus_t HIPRANDAPI hiprandGetDirectionVectors64(hiprandDirectionVectors64_t** vectors,
                                                        hiprandDirectionVectorSet_t   set);

hiprandStatus_t HIPRANDAPI hiprandGetScrambleConstants32(const unsigned int** constants);

hiprandStatus_t HIPRANDAPI hiprandGetScrambleConstants64(const unsigned long long** constants);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

/** @} */ // end of group hiprandhost

#endif // HIPRAND_H_
