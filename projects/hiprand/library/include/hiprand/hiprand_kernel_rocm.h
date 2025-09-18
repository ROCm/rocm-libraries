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

#ifndef HIPRAND_KERNEL_ROCM_H_
#define HIPRAND_KERNEL_ROCM_H_

/// \addtogroup hipranddevice
/// @{

#ifndef QUALIFIERS
    #define QUALIFIERS __forceinline__ __device__
#endif // QUALIFIERS

#include <hip/hip_runtime.h>
#include <type_traits>

/// \cond
#define FQUALIFIERS QUALIFIERS
/// \endcond
#include <rocrand/rocrand_kernel.h>

#include "hiprand/hiprand.h"
#include "hiprand/hiprand_rocm.h"

/// \cond
#define DEFINE_HIPRAND_STATE(hiprand_name, rocrand_name) \
    struct hiprand_name : public rocrand_name            \
    {                                                    \
        typedef rocrand_name base;                       \
    };                                                   \
    typedef struct hiprand_name hiprand_name##_t;

DEFINE_HIPRAND_STATE(hiprandState, rocrand_state_xorwow)
DEFINE_HIPRAND_STATE(hiprandStateXORWOW, rocrand_state_xorwow)
DEFINE_HIPRAND_STATE(hiprandStatePhilox4_32_10, rocrand_state_philox4x32_10)
DEFINE_HIPRAND_STATE(hiprandStateMRG32k3a, rocrand_state_mrg32k3a)
DEFINE_HIPRAND_STATE(hiprandStateMtgp32, rocrand_state_mtgp32)
DEFINE_HIPRAND_STATE(hiprandStateSobol32, rocrand_state_sobol32)
DEFINE_HIPRAND_STATE(hiprandStateScrambledSobol32, rocrand_state_scrambled_sobol32)
DEFINE_HIPRAND_STATE(hiprandStateSobol64, rocrand_state_sobol64)
DEFINE_HIPRAND_STATE(hiprandStateScrambledSobol64, rocrand_state_scrambled_sobol64)

#undef DEFINE_HIPRAND_STATE

typedef rocrand_discrete_distribution hiprandDiscreteDistribution_t;
/// \endcond

/// \brief Opaque MTGP32 PRNG state parameters in the device format
typedef mtgp32_params      mtgp32_kernel_params_t;

#ifdef HIPRAND_DOXYGEN
/// \brief Opaque MTGP32 PRNG state parameters in the host format
typedef struct mtgp32_params_fast mtgp32_params_fast_t;
#endif

/// \brief Deprecated alias of mtgp32_params_fast_t
/// \deprecated please use mtgp32_params_fast_t directly.
typedef mtgp32_fast_params mtgp32_fast_param_t;

/// \cond
namespace detail
{

template<typename T, typename... R>
struct is_any_of : std::false_type
{};

template<typename T, typename F>
struct is_any_of<T, F> : std::is_same<T, F>
{};

template<typename T, typename F, typename... R>
struct is_any_of<T, F, R...>
    : std::integral_constant<bool, std::is_same<T, F>::value || is_any_of<T, R...>::value>
{};

} // end namespace detail
/// \endcond

/// \cond
inline hiprandStatus_t to_hiprand_status(rocrand_status status)
{
    switch(status)
    {
        case ROCRAND_STATUS_SUCCESS: return HIPRAND_STATUS_SUCCESS;
        case ROCRAND_STATUS_NOT_CREATED: return HIPRAND_STATUS_NOT_INITIALIZED;
        case ROCRAND_STATUS_VERSION_MISMATCH: return HIPRAND_STATUS_VERSION_MISMATCH;
        case ROCRAND_STATUS_ALLOCATION_FAILED: return HIPRAND_STATUS_ALLOCATION_FAILED;
        case ROCRAND_STATUS_TYPE_ERROR: return HIPRAND_STATUS_TYPE_ERROR;
        case ROCRAND_STATUS_OUT_OF_RANGE: return HIPRAND_STATUS_OUT_OF_RANGE;
        case ROCRAND_STATUS_LENGTH_NOT_MULTIPLE: return HIPRAND_STATUS_LENGTH_NOT_MULTIPLE;
        case ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
        case ROCRAND_STATUS_LAUNCH_FAILURE: return HIPRAND_STATUS_LAUNCH_FAILURE;
        // case ROCRAND_STATUS_PREEXISTING_FAILURE:
        //     return HIPRAND_STATUS_PREEXISTING_FAILURE;
        // case ROCRAND_STATUS_INITIALIZATION_FAILED:
        //     return HIPRAND_STATUS_INITIALIZATION_FAILED;
        // case ROCRAND_STATUS_ARCH_MISMATCH:
        //     return HIPRAND_STATUS_ARCH_MISMATCH;
        case ROCRAND_STATUS_INTERNAL_ERROR: return HIPRAND_STATUS_INTERNAL_ERROR;
        default: return HIPRAND_STATUS_INTERNAL_ERROR;
    }
}

template<typename StateType>
QUALIFIERS void check_state_type()
{
    static_assert(detail::is_any_of<StateType,
                                    hiprandState_t,
                                    hiprandStateXORWOW_t,
                                    hiprandStatePhilox4_32_10_t,
                                    hiprandStateMRG32k3a_t,
                                    hiprandStateMtgp32_t,
                                    hiprandStateSobol32_t,
                                    hiprandStateScrambledSobol32_t,
                                    hiprandStateSobol64_t,
                                    hiprandStateScrambledSobol64_t>::value,
                  "StateType is not a hipRAND generator state");
}
/// \endcond

/**
 * \brief Copy MTGP32 state to another state using block of threads
 *
 * Copies a MTGP32 state \p src to \p dest using a block of threads
 * efficiently. Example usage would be:
 *
 * \code
 * __global__
 * void generate_kernel(hiprandStateMtgp32_t * states, unsigned int * output, const size_t size)
 * {
 *      const unsigned int state_id = hipBlockIdx_x;
 *      unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
 *      unsigned int stride = hipGridDim_x * hipBlockDim_x;
 *
 *      __shared__ GeneratorState state;
 *      hiprand_mtgp32_block_copy(&states[state_id], &state);
 *
 *      while(index < size)
 *      {
 *          output[index] = rocrand(&state);
 *          index += stride;
 *      }
 *
 *      hiprand_mtgp32_block_copy(&state, &states[state_id]);
 * }
 * \endcode
 *
 * \param src Pointer to a state to copy from
 * \param dest Pointer to a state to copy to
 */
QUALIFIERS void hiprand_mtgp32_block_copy(hiprandStateMtgp32_t* src, hiprandStateMtgp32_t* dest)
{
    rocrand_mtgp32_block_copy(src, dest);
}

/**
 * \brief Changes parameters of a MTGP32 state.
 *
 * \param state Pointer to a MTGP32 state
 * \param params Pointer to new parameters
 */
QUALIFIERS void hiprand_mtgp32_set_params(hiprandStateMtgp32_t*   state,
                                          mtgp32_kernel_params_t* params)
{
    rocrand_mtgp32_set_params(state, params);
}

/// \brief Initializes a PRNG state.
///
/// \tparam StateType Pseudorandom number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t or ::hiprandStateMRG32k3a_t
///
/// \param seed Pseudorandom number generator's seed
/// \param subsequence Number of subsequence to skipahead
/// \param offset Absolute subsequence offset, i.e. how many states from
/// current subsequence should be skipped
/// \param state Pointer to a state to initialize
///
/// \note MTGP32 must be initialized from the host using hiprandMakeMTGP32KernelState().
template<class StateType>
QUALIFIERS void hiprand_init(const unsigned long long seed,
                             const unsigned long long subsequence,
                             const unsigned long long offset,
                             StateType*               state)
{
    check_state_type<StateType>();
    static_assert(!std::is_same<StateType, hiprandStateMtgp32_t>::value,
                  "hiprandStateMtgp32_t does not have hiprand_init function, "
                  "check hiprandMakeMTGP32KernelState() host function");
    static_assert(!detail::is_any_of<StateType,
                                     hiprandStateSobol32_t,
                                     hiprandStateScrambledSobol32_t,
                                     hiprandStateSobol64_t,
                                     hiprandStateScrambledSobol64_t>::value,
                  "Quasirandom generators use different hiprand_init() function");
    rocrand_init(seed, subsequence, offset, state);
}

/// \brief Initializes a Sobol32 state.
///
/// \param direction_vectors Pointer to array of 32 <tt>unsigned int</tt>s that
/// represent the direction numbers.
/// \param offset Absolute subsequence offset, i.e. how many states should be skipped.
/// \param state Pointer to a state to initialize.
QUALIFIERS void hiprand_init(hiprandDirectionVectors32_t direction_vectors,
                             unsigned int                offset,
                             hiprandStateSobol32_t*      state)
{
    rocrand_init(direction_vectors, offset, state);
}

/// \brief Initializes a ScrambledSobol32 state.
///
/// \param direction_vectors Pointer to array of 32 <tt>unsigned int</tt>s that
/// represent the direction numbers.
/// \param scramble_constant Constant used for scrambling the sequence.
/// \param offset Absolute subsequence offset, i.e. how many states should be skipped.
/// \param state Pointer to a state to initialize.
QUALIFIERS void hiprand_init(hiprandDirectionVectors32_t     direction_vectors,
                             unsigned int                    scramble_constant,
                             unsigned int                    offset,
                             hiprandStateScrambledSobol32_t* state)
{
    rocrand_init(direction_vectors, scramble_constant, offset, state);
}

/// \brief Initializes a Sobol64 state.
///
/// \param direction_vectors Pointer to array of 64 <tt>unsigned long long int</tt>s that
/// represent the direction numbers.
/// \param offset Absolute subsequence offset, i.e. how many states should be skipped.
/// \param state Pointer to a state to initialize.
QUALIFIERS void hiprand_init(hiprandDirectionVectors64_t direction_vectors,
                             unsigned int                offset,
                             hiprandStateSobol64_t*      state)
{
    rocrand_init(direction_vectors, offset, state);
}

/// \brief Initializes a ScrambledSobol64 state.
///
/// \param direction_vectors Pointer to array of 64 <tt>unsigned long long int</tt>s that
/// represent the direction numbers.
/// \param scramble_constant Constant used for scrambling the sequence.
/// \param offset Absolute subsequence offset, i.e. how many states should be skipped.
/// \param state Pointer to a state to initialize.
QUALIFIERS void hiprand_init(hiprandDirectionVectors64_t     direction_vectors,
                             unsigned long long int          scramble_constant,
                             unsigned int                    offset,
                             hiprandStateScrambledSobol64_t* state)
{
    rocrand_init(direction_vectors, scramble_constant, offset, state);
}

/// \brief Updates RNG state skipping \p n states ahead.
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t, ::hiprandStateMRG32k3a_t,
/// ::hiprandStateSobol32_t, ::hiprandStateScrambledSobol32_t,
/// ::hiprandStateSobol64_t or ::hiprandStateScrambledSobol64_t.
///
/// \param n Number of states to skipahead
/// \param state Pointer to a state to modify
template<class StateType>
QUALIFIERS void skipahead(unsigned long long n, StateType* state)
{
    check_state_type<StateType>();
    static_assert(!std::is_same<StateType, hiprandStateMtgp32_t>::value,
                  "hiprandStateMtgp32_t does not have skipahead function");
    typedef typename StateType::base base_type;
    skipahead(n, static_cast<base_type*>(state));
}

/// \brief Updates PRNG state skipping \p n sequences ahead.
///
/// | PRNG          | Sequence size [Number of elements] |
/// |---------------|------------------------------------|
/// | XORWOW        | \f$ 2^{67} \f$                     |
/// | Philox        | \f$ 4 \times 2^{64} \f$            |
/// | MRG32k3a      | \f$ 2^{67} \f$                     |
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t,
/// or ::hiprandStateMRG32k3a_t
///
/// \param n Number of subsequences to skipahead
/// \param state Pointer to a state to update
template<class StateType>
QUALIFIERS void skipahead_sequence(unsigned long long n, StateType* state)
{
    check_state_type<StateType>();
    static_assert(!detail::is_any_of<StateType,
                                     hiprandStateMtgp32_t,
                                     hiprandStateSobol32_t,
                                     hiprandStateScrambledSobol32_t,
                                     hiprandStateSobol64_t,
                                     hiprandStateScrambledSobol64_t>::value,
                  "StateType does not have skipahead_sequence function");
    typedef typename StateType::base base_type;
    skipahead_subsequence(n, static_cast<base_type*>(state));
}

/// \brief Updates PRNG state skipping \p n subsequences ahead.
///
/// | PRNG         | Subsequence size [Number of elements] |
/// |--------------|---------------------------------------|
/// | XORWOW       | \f$ 2^{67} \f$                        |
/// | Philox       | \f$ 4 \times 2^{64} \f$               |
/// | MRG32k3a     | \f$ 2^{127} \f$                       |
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t
/// or ::hiprandStateMRG32k3a_t
///
/// \param n Number of subsequences to skipahead
/// \param state Pointer to a state to update
template<class StateType>
QUALIFIERS void skipahead_subsequence(unsigned long long n, StateType* state)
{
    check_state_type<StateType>();
    static_assert(!detail::is_any_of<StateType,
                                     hiprandStateMtgp32_t,
                                     hiprandStateSobol32_t,
                                     hiprandStateScrambledSobol32_t,
                                     hiprandStateSobol64_t,
                                     hiprandStateScrambledSobol64_t>::value,
                  "StateType does not have skipahead_subsequence function");
    typedef typename StateType::base base_type;
    skipahead_subsequence(n, static_cast<base_type*>(state));
}

/// \brief Generates uniformly distributed random <tt>unsigned int</tt>
/// from [0; 2^32 - 1] range.
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t, ::hiprandStateMRG32k3a_t,
/// ::hiprandStateMtgp32_t, ::hiprandStateSobol32_t, ::hiprandStateScrambledSobol32_t.
///
/// \param state Pointer to a RNG state to use
/// \return Uniformly distributed random 32-bit <tt>unsigned int</tt>
template<class StateType>
QUALIFIERS unsigned int hiprand(StateType* state)
{
    check_state_type<StateType>();
    return rocrand(state);
}

/// \brief Generates four uniformly distributed random <tt>unsigned int</tt>s
/// from [0; 2^32 - 1] range.
///
/// \param state Pointer to a Philox state to use
/// \return Four uniformly distributed random 32-bit <tt>unsigned int</tt>s as \p uint4
QUALIFIERS uint4 hiprand4(hiprandStatePhilox4_32_10_t* state)
{
    return rocrand4(state);
}

/// \brief Generates uniformly distributed random <tt>unsigned long long int</tt>
/// from [0; 2^64 - 1] range.
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of the following types:
/// ::hiprandStateSobol64_t or ::hiprandStateScrambledSobol64_t.
///
/// \param state Pointer to a RNG state to use
/// \return Uniformly distributed random 64-bit <tt>unsigned long long int</tt>
template<class StateType>
QUALIFIERS unsigned long long int hiprand_long_long(StateType* state)
{
    check_state_type<StateType>();
    return rocrand(state);
}

/// \brief Generates uniformly distributed random <tt>float</tt> value
/// from (0; 1] range.
///
/// \tparam StateType Random number generator state type.
///
/// \param state Pointer to a RNG state to use
/// \return Uniformly distributed random <tt>float</tt> value
template<class StateType>
QUALIFIERS float hiprand_uniform(StateType* state)
{
    check_state_type<StateType>();
    return rocrand_uniform(state);
}

/// \brief Generates four uniformly distributed random <tt>float</tt> value
/// from (0; 1] range.
///
/// \param state Pointer to a Philox state to use
/// \return Four uniformly distributed random <tt>float</tt> values as \p float4
QUALIFIERS float4 hiprand_uniform4(hiprandStatePhilox4_32_10_t* state)
{
    return rocrand_uniform4(state);
}

/// \brief Generates uniformly distributed random <tt>double</tt> value from (0; 1] range
///
/// \tparam StateType Random number generator state type.
///
/// \param state Pointer to a RNG state to use
/// \return Uniformly distributed random <tt>double</tt> value
///
/// \note When \p state is of type: ::hiprandStateMRG32k3a_t, ::hiprandStateMtgp32_t,
/// ::hiprandStateSobol32_t or ::hiprandStateScrambledSobol32_t then the returned
/// \p double value is generated using only 32 random bits (one <tt>unsigned int</tt> value).
/// In case of the Sobol types, this is done to guarantee the quasi-random properties.
template<class StateType>
QUALIFIERS double hiprand_uniform_double(StateType* state)
{
    check_state_type<StateType>();
    return rocrand_uniform_double(state);
}

/// \brief Generates two uniformly distributed random <tt>double</tt> values
/// from (0; 1] range.
///
/// \param state Pointer to a Philox state to use
/// \return Two uniformly distributed random <tt>double</tt> values as \p double2
QUALIFIERS double2 hiprand_uniform2_double(hiprandStatePhilox4_32_10_t* state)
{
    return rocrand_uniform_double2(state);
}

/// \brief Generates four uniformly distributed random <tt>double</tt> values
/// from (0; 1] range.
///
/// \param state Pointer to a Philox state to use
/// \return Four uniformly distributed random <tt>double</tt> values as \p double4
QUALIFIERS double4 hiprand_uniform4_double(hiprandStatePhilox4_32_10_t* state)
{
    return rocrand_uniform_double4(state);
}

/// \brief Generates normally distributed random <tt>float</tt> value
///
/// Mean value of normal distribution is equal to 0.0, and standard deviation
/// equals 1.0.
///
/// \tparam StateType Random number generator state type.

/// \param state Pointer to a RNG state to use
/// \return Normally distributed random <tt>float</tt> value
template<class StateType>
QUALIFIERS float hiprand_normal(StateType* state)
{
    check_state_type<StateType>();
    return rocrand_normal(state);
}

/// \brief Generates two normally distributed random <tt>float</tt> values
///
/// Mean value of normal distribution is equal to 0.0, and standard deviation
/// equals 1.0.
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t,
/// or ::hiprandStateMRG32k3a_t
///
/// \param state Pointer to a RNG state to use
/// \return Two normally distributed random <tt>float</tt> values as \p float2
template<class StateType>
QUALIFIERS float2 hiprand_normal2(StateType* state)
{
    check_state_type<StateType>();
    static_assert(detail::is_any_of<StateType,
                                    hiprandState_t,
                                    hiprandStateXORWOW_t,
                                    hiprandStatePhilox4_32_10_t,
                                    hiprandStateMRG32k3a_t>::value,
                  "Used StateType is not supported");
    return rocrand_normal2(state);
}

/// \brief Generates four normally distributed random <tt>float</tt> values
///
/// Mean value of normal distribution is equal to 0.0, and standard deviation
/// equals 1.0.
///
/// \param state Pointer to a Philox state to use
/// \return Four normally distributed random <tt>float</tt> values as \p float4
QUALIFIERS float4 hiprand_normal4(hiprandStatePhilox4_32_10_t* state)
{
    return rocrand_normal4(state);
}

/// \brief Generates normally distributed random <tt>double</tt> value
///
/// Mean value of normal distribution is equal to 0.0, and standard deviation
/// equals 1.0.
///
/// \tparam StateType Random number generator state type.
///
/// \param state Pointer to a RNG state to use
/// \return Normally distributed random <tt>double</tt> value
template<class StateType>
QUALIFIERS double hiprand_normal_double(StateType* state)
{
    check_state_type<StateType>();
    return rocrand_normal_double(state);
}

/// \brief Generates two normally distributed random <tt>double</tt> values
///
/// Mean value of normal distribution is equal to 0.0, and standard deviation
/// equals 1.0.
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t,
/// or ::hiprandStateMRG32k3a_t
///
/// \param state Pointer to a RNG state to use
/// \return Two normally distributed random <tt>double</tt> values as \p double2
template<class StateType>
QUALIFIERS double2 hiprand_normal2_double(StateType* state)
{
    check_state_type<StateType>();
    static_assert(detail::is_any_of<StateType,
                                    hiprandState_t,
                                    hiprandStateXORWOW_t,
                                    hiprandStatePhilox4_32_10_t,
                                    hiprandStateMRG32k3a_t>::value,
                  "Used StateType is not supported");
    return rocrand_normal_double2(state);
}

/// \brief Generates four normally distributed random <tt>double</tt> values
///
/// Mean value of normal distribution is equal to 0.0, and standard deviation
/// equals 1.0.
///
/// \param state Pointer to a Philox state to use
/// \return Four normally distributed random <tt>double</tt> values as \p double4
QUALIFIERS double4 hiprand_normal4_double(hiprandStatePhilox4_32_10_t* state)
{
    return rocrand_normal_double4(state);
}

/// \brief Generates log-normally distributed random <tt>float</tt> value
///
/// \tparam StateType Random number generator state type.
///
/// \param state Pointer to a RNG state to use
/// \param mean Mean value of log-normal distribution
/// \param stddev Standard deviation value of log-normal distribution
/// \return Log-normally distributed random <tt>float</tt> value
template<class StateType>
QUALIFIERS float hiprand_log_normal(StateType* state, float mean, float stddev)
{
    check_state_type<StateType>();
    return rocrand_log_normal(state, mean, stddev);
}

/// \brief Generates two log-normally distributed random <tt>float</tt> values
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t,
/// or ::hiprandStateMRG32k3a_t
///
/// \param state Pointer to a RNG state to use
/// \param mean Mean value of log-normal distribution
/// \param stddev Standard deviation value of log-normal distribution
/// \return Two log-normally distributed random <tt>float</tt> values as \p float2
template<class StateType>
QUALIFIERS float2 hiprand_log_normal2(StateType* state, float mean, float stddev)
{
    check_state_type<StateType>();
    static_assert(detail::is_any_of<StateType,
                                    hiprandState_t,
                                    hiprandStateXORWOW_t,
                                    hiprandStatePhilox4_32_10_t,
                                    hiprandStateMRG32k3a_t>::value,
                  "Used StateType is not supported");
    return rocrand_log_normal2(state, mean, stddev);
}

/// \brief Generates four log-normally distributed random <tt>float</tt> values
///
/// \param state Pointer to a Philox state to use
/// \param mean Mean value of log-normal distribution
/// \param stddev Standard deviation value of log-normal distribution
/// \return Four log-normally distributed random <tt>float</tt> values as \p float4
QUALIFIERS float4 hiprand_log_normal4(hiprandStatePhilox4_32_10_t* state, float mean, float stddev)
{
    return rocrand_log_normal4(state, mean, stddev);
}

/// \brief Generates log-normally distributed random <tt>double</tt> value
///
/// \tparam StateType Random number generator state type.
///
/// \param state Pointer to a RNG state to use
/// \param mean Mean value of log-normal distribution
/// \param stddev Standard deviation value of log-normal distribution
/// \return Log-normally distributed random <tt>double</tt> value
template<class StateType>
QUALIFIERS double hiprand_log_normal_double(StateType* state, double mean, double stddev)
{
    check_state_type<StateType>();
    return rocrand_log_normal_double(state, mean, stddev);
}

/// \brief Generates two log-normally distributed random <tt>double</tt> values
///
/// \tparam StateType Random number generator state type.
/// \p StateType type must be one of following types:
/// ::hiprandStateXORWOW_t, ::hiprandStatePhilox4_32_10_t,
/// ::hiprandStateMRG32k3a_t or hiprandStateMtgp32_t.
///
/// \param state Pointer to a RNG state to use
/// \param mean Mean value of log-normal distribution
/// \param stddev Standard deviation value of log-normal distribution
/// \return Two log-normally distributed random <tt>double</tt> values as \p double2
template<class StateType>
QUALIFIERS double2 hiprand_log_normal2_double(StateType* state, double mean, double stddev)
{
    check_state_type<StateType>();
    static_assert(detail::is_any_of<StateType,
                                    hiprandState_t,
                                    hiprandStateXORWOW_t,
                                    hiprandStatePhilox4_32_10_t,
                                    hiprandStateMRG32k3a_t>::value,
                  "Used StateType is not supported");
    return rocrand_log_normal_double2(state, mean, stddev);
}

/// \brief Generates four log-normally distributed random <tt>double</tt> values
///
/// \param state Pointer to a Philox state to use
/// \param mean Mean value of log-normal distribution
/// \param stddev Standard deviation value of log-normal distribution
/// \return Four log-normally distributed random <tt>double</tt> values as \p double4
QUALIFIERS double4 hiprand_log_normal4_double(hiprandStatePhilox4_32_10_t* state,
                                              double                       mean,
                                              double                       stddev)
{
    return rocrand_log_normal_double4(state, mean, stddev);
}

/// \brief Generates Poisson-distributed random <tt>unsigned int</tt> value
///
/// \tparam StateType Random number generator state type.
///
/// \param state Pointer to a RNG state to use
/// \param lambda Lambda (mean) parameter of Poisson distribution
/// \return Poisson-distributed random <tt>unsigned int</tt> value
template<class StateType>
QUALIFIERS unsigned int hiprand_poisson(StateType* state, double lambda)
{
    check_state_type<StateType>();
    return rocrand_poisson(state, lambda);
}

/// \brief Generates four Poisson-distributed random <tt>unsigned int</tt> values
///
/// \param state Pointer to a Philox state to use
/// \param lambda Lambda (mean) parameter of Poisson distribution
/// \return Four Poisson-distributed random <tt>unsigned int</tt> values as \p uint4
QUALIFIERS uint4 hiprand_poisson4(hiprandStatePhilox4_32_10_t* state, double lambda)
{
    return rocrand_poisson4(state, lambda);
}

/// \brief Generates random <tt>unsigned int</tt> value according to
/// given discrete distribution
///
/// \tparam StateType Random number generator state type.
///
/// \param state Pointer to a RNG state to use
/// \param discrete_distribution Discrete distribution
/// \return Random <tt>unsigned int</tt> value
///
/// See also: hiprandCreatePoissonDistribution()
template<class StateType>
QUALIFIERS unsigned int hiprand_discrete(StateType*                    state,
                                 hiprandDiscreteDistribution_t discrete_distribution)
{
    check_state_type<StateType>();
    return rocrand_discrete(state, discrete_distribution);
}

/// \brief Generates four random <tt>unsigned int</tt> values according to
/// given discrete distribution
///
/// \param state Pointer to a Philox state to use
/// \param discrete_distribution Discrete distribution
/// \return Four random <tt>unsigned int</tt> values as \p uint4
///
/// See also: hiprandCreatePoissonDistribution()
QUALIFIERS uint4 hiprand_discrete4(hiprandStatePhilox4_32_10_t*  state,
                                   hiprandDiscreteDistribution_t discrete_distribution)
{
    return rocrand_discrete4(state, discrete_distribution);
}

/// @} // end of group hipranddevice

#endif // HIPRAND_KERNEL_ROCM_H_
