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

#ifndef HIPRAND_KERNEL_NVCC_H_
#define HIPRAND_KERNEL_NVCC_H_

#ifndef QUALIFIERS
#define QUALIFIERS __forceinline__ __device__
#endif // QUALIFIERS

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <type_traits>

#include <curand_kernel.h>

#define DEFINE_HIPRAND_STATE(hiprand_name, curand_name) \
    struct hiprand_name : public curand_name \
    { \
        typedef curand_name base; \
    }; \
    typedef struct hiprand_name hiprand_name ## _t;

DEFINE_HIPRAND_STATE(hiprandState, curandState)
DEFINE_HIPRAND_STATE(hiprandStateXORWOW, curandStateXORWOW)
DEFINE_HIPRAND_STATE(hiprandStatePhilox4_32_10, curandStatePhilox4_32_10)
DEFINE_HIPRAND_STATE(hiprandStateMRG32k3a, curandStateMRG32k3a)
DEFINE_HIPRAND_STATE(hiprandStateMtgp32, curandStateMtgp32)
DEFINE_HIPRAND_STATE(hiprandStateSobol32, curandStateSobol32)
DEFINE_HIPRAND_STATE(hiprandStateScrambledSobol32, curandStateScrambledSobol32)
DEFINE_HIPRAND_STATE(hiprandStateSobol64, curandStateSobol64)
DEFINE_HIPRAND_STATE(hiprandStateScrambledSobol64, curandStateScrambledSobol64)

#undef DEFINE_HIPRAND_STATE

typedef curandDiscreteDistribution_t hiprandDiscreteDistribution_t;

/// \cond
namespace detail {

template<typename T, typename... R>
struct is_any_of : std::false_type { };

template<typename T, typename F>
struct is_any_of<T, F> : std::is_same<T, F> { };

template<typename T, typename F, typename... R>
struct is_any_of<T, F, R...>
    : std::integral_constant<
        bool,
        std::is_same<T, F>::value || is_any_of<T, R...>::value
      >
{ };

} // end namespace detail
/// \endcond

inline
hiprandStatus_t to_hiprand_status(curandStatus_t status)
{
    switch(status)
    {
        case CURAND_STATUS_SUCCESS:
            return HIPRAND_STATUS_SUCCESS;
        case CURAND_STATUS_NOT_INITIALIZED:
            return HIPRAND_STATUS_NOT_INITIALIZED;
        case CURAND_STATUS_VERSION_MISMATCH:
            return HIPRAND_STATUS_VERSION_MISMATCH;
        case CURAND_STATUS_ALLOCATION_FAILED:
            return HIPRAND_STATUS_ALLOCATION_FAILED;
        case CURAND_STATUS_TYPE_ERROR:
            return HIPRAND_STATUS_TYPE_ERROR;
        case CURAND_STATUS_OUT_OF_RANGE:
            return HIPRAND_STATUS_OUT_OF_RANGE;
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return HIPRAND_STATUS_LENGTH_NOT_MULTIPLE;
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
        case CURAND_STATUS_LAUNCH_FAILURE:
            return HIPRAND_STATUS_LAUNCH_FAILURE;
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return HIPRAND_STATUS_PREEXISTING_FAILURE;
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return HIPRAND_STATUS_INITIALIZATION_FAILED;
        case CURAND_STATUS_ARCH_MISMATCH:
            return HIPRAND_STATUS_ARCH_MISMATCH;
        case CURAND_STATUS_INTERNAL_ERROR:
            return HIPRAND_STATUS_INTERNAL_ERROR;
        default:
            return HIPRAND_STATUS_INTERNAL_ERROR;
    }
}

template<typename StateType>
QUALIFIERS
void check_state_type()
{
    static_assert(
        detail::is_any_of<
            StateType,
            hiprandState_t,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t,
            hiprandStateMtgp32_t,
            hiprandStateSobol32_t,
            hiprandStateScrambledSobol32_t,
            hiprandStateSobol64_t,
            hiprandStateScrambledSobol64_t
        >::value,
        "StateType is not a hipRAND generator state"
    );
}

QUALIFIERS
void hiprand_mtgp32_block_copy(hiprandStateMtgp32_t * src,
                               hiprandStateMtgp32_t * dest)
{
    #if defined(__HIP_DEVICE_COMPILE__)
    const unsigned int thread_id = threadIdx.x;
    for (int i = thread_id; i < MTGP32_STATE_SIZE; i += blockDim.x)
        dest->s[i] = src->s[i];

    if (thread_id == 0)
    {
        dest->offset = src->offset;
        dest->pIdx = src->pIdx;
    }
    __syncthreads();
    #else
    *dest = *src;
    #endif
}

QUALIFIERS
void hiprand_mtgp32_set_params(hiprandStateMtgp32_t * state,
                               mtgp32_kernel_params_t * params)
{
    typedef typename hiprandStateMtgp32_t::base base_type;
    static_cast<base_type*>(state)->k = params;
}

template<class StateType>
QUALIFIERS
void hiprand_init(const unsigned long long seed,
                  const unsigned long long subsequence,
                  const unsigned long long offset,
                  StateType * state)
{
    static_assert(
        !std::is_same<
            StateType,
            hiprandStateMtgp32_t
        >::value,
        "hiprandStateMtgp32_t does not have hiprand_init function, "
        "check hiprandMakeMTGP32KernelState() host function"
    );
    static_assert(
        !detail::is_any_of<
            StateType,
            hiprandStateSobol32_t,
            hiprandStateScrambledSobol32_t,
            hiprandStateSobol64_t,
            hiprandStateScrambledSobol64_t
        >::value,
        "Quasirandom generators use different hiprand_init() function"
    );
    check_state_type<StateType>();
    curand_init(seed, subsequence, offset, state);
}

QUALIFIERS void hiprand_init(hiprandDirectionVectors32_t direction_vectors,
                             unsigned int                offset,
                             hiprandStateSobol32_t*      state)
{
    curand_init(direction_vectors, offset, state);
}

QUALIFIERS void hiprand_init(hiprandDirectionVectors32_t     direction_vectors,
                             unsigned int                    scramble_constant,
                             unsigned int                    offset,
                             hiprandStateScrambledSobol32_t* state)
{
    curand_init(direction_vectors, scramble_constant, offset, state);
}

QUALIFIERS void hiprand_init(hiprandDirectionVectors64_t direction_vectors,
                             unsigned int                offset,
                             hiprandStateSobol64_t*      state)
{
    curand_init(direction_vectors, offset, state);
}

QUALIFIERS void hiprand_init(hiprandDirectionVectors64_t     direction_vectors,
                             unsigned long long int          scramble_constant,
                             unsigned int                    offset,
                             hiprandStateScrambledSobol64_t* state)
{
    curand_init(direction_vectors, scramble_constant, offset, state);
}

template<class StateType>
QUALIFIERS
    typename std::enable_if<!std::is_same<StateType, curandStateSobol32>::value
                            && !std::is_same<StateType, curandStateScrambledSobol32>::value
                            && !std::is_same<StateType, curandStateSobol64>::value
                            && !std::is_same<StateType, curandStateScrambledSobol64>::value>::type
    skipahead(unsigned long long n, StateType* state)
{
    check_state_type<StateType>();
    static_assert(
        !std::is_same<
            StateType,
            hiprandStateMtgp32_t
        >::value,
        "hiprandStateMtgp32_t does not have skipahead function"
    );
    typedef typename StateType::base base_type;
    skipahead(n, static_cast<base_type*>(state));
}

template<class StateType>
QUALIFIERS
void skipahead_sequence(unsigned long long n, StateType * state)
{
    check_state_type<StateType>();
    static_assert(
        !detail::is_any_of<
            StateType,
            hiprandStateMtgp32_t,
            hiprandStateSobol32_t,
            hiprandStateScrambledSobol32_t,
            hiprandStateSobol64_t,
            hiprandStateScrambledSobol64_t
        >::value,
        "StateType does not have skipahead_sequence function"
    );
    typedef typename StateType::base base_type;
    skipahead_sequence(n, static_cast<base_type*>(state));
}

template<class StateType>
QUALIFIERS
void skipahead_subsequence(unsigned long long n, StateType * state)
{
    // skipahead_subsequence(unsigned long long n, curandStateMRG32k3a_t *state)
    // is defined in curand_kernel.h
    check_state_type<StateType>();
    static_assert(
        !detail::is_any_of<
            StateType,
            hiprandStateMtgp32_t,
            hiprandStateSobol32_t,
            hiprandStateScrambledSobol32_t,
            hiprandStateSobol64_t,
            hiprandStateScrambledSobol64_t
        >::value,
        "StateType does not have skipahead_subsequence function"
    );
    typedef typename StateType::base base_type;
    skipahead_sequence(n, static_cast<base_type*>(state));
}

template<class StateType>
QUALIFIERS
unsigned int hiprand(StateType * state)
{
    check_state_type<StateType>();
    return curand(state);
}

QUALIFIERS
uint4 hiprand4(hiprandStatePhilox4_32_10_t * state)
{
    return curand4(state);
}

template<class StateType>
QUALIFIERS unsigned long long int hiprand_long_long(StateType* state)
{
    check_state_type<StateType>();
    return curand(state);
}

template<class StateType>
QUALIFIERS
float hiprand_uniform(StateType * state)
{
    check_state_type<StateType>();
    return curand_uniform(state);
}

QUALIFIERS
float4 hiprand_uniform4(hiprandStatePhilox4_32_10_t * state)
{
    return curand_uniform4(state);
}

template<class StateType>
QUALIFIERS
double hiprand_uniform_double(StateType * state)
{
    check_state_type<StateType>();
    return curand_uniform_double(state);
}

QUALIFIERS
double2 hiprand_uniform2_double(hiprandStatePhilox4_32_10_t * state)
{
    return curand_uniform2_double(state);
}

QUALIFIERS
double4 hiprand_uniform4_double(hiprandStatePhilox4_32_10_t * state)
{
    return curand_uniform4_double(state);
}

template<class StateType>
QUALIFIERS
float hiprand_normal(StateType * state)
{
    check_state_type<StateType>();
    return curand_normal(state);
}

template<class StateType>
QUALIFIERS
float2 hiprand_normal2(StateType * state)
{
    check_state_type<StateType>();
    static_assert(
        detail::is_any_of<
            StateType,
            hiprandState_t,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    return curand_normal2(state);
}

QUALIFIERS
float4 hiprand_normal4(hiprandStatePhilox4_32_10_t * state)
{
    return curand_normal4(state);
}

template<class StateType>
QUALIFIERS
double hiprand_normal_double(StateType * state)
{
    check_state_type<StateType>();
    return curand_normal_double(state);
}

template<class StateType>
QUALIFIERS
double2 hiprand_normal2_double(StateType * state)
{
    check_state_type<StateType>();
    static_assert(
        detail::is_any_of<
            StateType,
            hiprandState_t,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    return curand_normal2_double(state);
}

QUALIFIERS
double4 hiprand_normal4_double(hiprandStatePhilox4_32_10_t * state)
{
    return curand_normal4_double(state);
}

template<class StateType>
QUALIFIERS
float hiprand_log_normal(StateType * state,
                         float mean, float stddev)
{
    check_state_type<StateType>();
    return curand_log_normal(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
float2 hiprand_log_normal2(StateType * state,
                          float mean, float stddev)
{
    check_state_type<StateType>();
    static_assert(
        detail::is_any_of<
            StateType,
            hiprandState_t,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    return curand_log_normal2(state, mean, stddev);
}

QUALIFIERS
float4 hiprand_log_normal4(hiprandStatePhilox4_32_10_t * state,
                           float mean, float stddev)
{
    return curand_log_normal4(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
double hiprand_log_normal_double(StateType * state,
                                 double mean, double stddev)
{
    check_state_type<StateType>();
    return curand_log_normal_double(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
double2 hiprand_log_normal2_double(StateType * state,
                                  double mean, double stddev)
{
    check_state_type<StateType>();
    static_assert(
        detail::is_any_of<
            StateType,
            hiprandState_t,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    return curand_log_normal2_double(state, mean, stddev);
}

QUALIFIERS
double4 hiprand_log_normal4_double(hiprandStatePhilox4_32_10_t * state,
                                   double mean, double stddev)
{
    return curand_log_normal4_double(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
unsigned int hiprand_poisson(StateType * state, double lambda)
{
    check_state_type<StateType>();
    return curand_poisson(state, lambda);
}

QUALIFIERS
uint4 hiprand_poisson4(hiprandStatePhilox4_32_10_t * state, double lambda)
{
    return curand_poisson4(state, lambda);
}

template<class StateType>
QUALIFIERS
unsigned int hiprand_discrete(StateType * state, hiprandDiscreteDistribution_t discrete_distribution)
{
    check_state_type<StateType>();
    return curand_discrete(state, discrete_distribution);
}

QUALIFIERS
uint4 hiprand_discrete4(hiprandStatePhilox4_32_10_t * state, hiprandDiscreteDistribution_t discrete_distribution)
{
    return curand_discrete4(state, discrete_distribution);
}

#endif // HIPRAND_KERNEL_H_
