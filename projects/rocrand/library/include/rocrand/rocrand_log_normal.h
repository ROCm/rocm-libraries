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

#ifndef ROCRAND_LOG_NORMAL_H_
#define ROCRAND_LOG_NORMAL_H_

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

#include "rocrand/rocrand_lfsr113.h"
#include "rocrand/rocrand_mrg31k3p.h"
#include "rocrand/rocrand_mrg32k3a.h"
#include "rocrand/rocrand_mtgp32.h"
#include "rocrand/rocrand_philox4x32_10.h"
#include "rocrand/rocrand_scrambled_sobol32.h"
#include "rocrand/rocrand_scrambled_sobol64.h"
#include "rocrand/rocrand_sobol32.h"
#include "rocrand/rocrand_sobol64.h"
#include "rocrand/rocrand_threefry2x32_20.h"
#include "rocrand/rocrand_threefry2x64_20.h"
#include "rocrand/rocrand_threefry4x32_20.h"
#include "rocrand/rocrand_threefry4x64_20.h"
#include "rocrand/rocrand_xorwow.h"

#include "rocrand/rocrand_normal.h"

#include <hip/hip_runtime.h>

#include <math.h>

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using Philox
 * generator in \p state, and increments position of the generator by one.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, returns first of them, and saves
 * the second to be returned on the next call.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ float
    rocrand_log_normal(rocrand_state_philox4x32_10* state, float mean, float stddev)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_philox4x32_10> bm_helper;

    if(bm_helper::has_float(state))
    {
        return expf(mean + (stddev * bm_helper::get_float(state)));
    }

    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    bm_helper::save_float(state, r.y);
    return expf(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using Philox
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_philox4x32_10* state, float mean, float stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    return float2 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y))
    };
}

/**
 * \brief Returns four log-normally distributed \p float values.
 *
 * Generates and returns four log-normally distributed \p float values using Philox
 * generator in \p state, and increments position of the generator by four.
 * The function uses the Box-Muller transform method to generate four normally distributed
 * values, transforms them to log-normally distributed values, and returns them.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Four log-normally distributed \p float value as \p float4
 */
__forceinline__ __device__ __host__
float4 rocrand_log_normal4(rocrand_state_philox4x32_10* state, float mean, float stddev)
{
    float4 r = rocrand_device::detail::normal_distribution4(rocrand4(state));
    return float4 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y)),
        expf(mean + (stddev * r.z)),
        expf(mean + (stddev * r.w))
    };
}

/**
 * \brief Returns a log-normally distributed \p double values.
 *
 * Generates and returns a log-normally distributed \p double value using Philox
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * \p double values, transforms them to log-normally distributed \p double values, returns
 * first of them, and saves the second to be returned on the next call.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ double
    rocrand_log_normal_double(rocrand_state_philox4x32_10* state, double mean, double stddev)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_philox4x32_10> bm_helper;

    if(bm_helper::has_double(state))
    {
        return exp(mean + (stddev * bm_helper::get_double(state)));
    }
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    bm_helper::save_double(state, r.y);
    return exp(mean + r.x * stddev);
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using Philox
 * generator in \p state, and increments position of the generator by four.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_philox4x32_10* state, double mean, double stddev)
{
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    return double2 {
        exp(mean + (stddev * r.x)),
        exp(mean + (stddev * r.y))
    };
}

/**
 * \brief Returns four log-normally distributed \p double values.
 *
 * Generates and returns four log-normally distributed \p double values using Philox
 * generator in \p state, and increments position of the generator by eight.
 * The function uses the Box-Muller transform method to generate four normally distributed
 * values, transforms them to log-normally distributed values, and returns them.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Four log-normally distributed \p double values as \p double4
 */
__forceinline__ __device__ __host__
double4 rocrand_log_normal_double4(rocrand_state_philox4x32_10* state, double mean, double stddev)
{
    double2 r1, r2;
    r1 = rocrand_log_normal_double2(state, mean, stddev);
    r2 = rocrand_log_normal_double2(state, mean, stddev);
    return double4 {
        r1.x, r1.y, r2.x, r2.y
    };
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using MRG31k3p
 * generator in \p state, and increments position of the generator by one.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, returns first of them,
 * and saves the second to be returned on the next call.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ float
    rocrand_log_normal(rocrand_state_mrg31k3p* state, float mean, float stddev)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg31k3p> bm_helper;

    if(bm_helper::has_float(state))
    {
        return expf(mean + (stddev * bm_helper::get_float(state)));
    }

    auto state1 = state->next();
    auto state2 = state->next();

    float2 r
        = rocrand_device::detail::mrg_normal_distribution2<rocrand_state_mrg31k3p>(state1, state2);
    bm_helper::save_float(state, r.y);
    return expf(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using MRG31k3p
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_mrg31k3p* state, float mean, float stddev)
{
    auto state1 = state->next();
    auto state2 = state->next();

    float2 r
        = rocrand_device::detail::mrg_normal_distribution2<rocrand_state_mrg31k3p>(state1, state2);
    return float2{expf(mean + (stddev * r.x)), expf(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using MRG31k3p
 * generator in \p state, and increments position of the generator by one.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * \p double values, transforms them to log-normally distributed \p double values, returns
 * first of them, and saves the second to be returned on the next call.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ double
    rocrand_log_normal_double(rocrand_state_mrg31k3p* state, double mean, double stddev)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg31k3p> bm_helper;

    if(bm_helper::has_double(state))
    {
        return exp(mean + (stddev * bm_helper::get_double(state)));
    }

    auto state1 = state->next();
    auto state2 = state->next();

    double2 r
        = rocrand_device::detail::mrg_normal_distribution_double2<rocrand_state_mrg31k3p>(state1,
                                                                                          state2);
    bm_helper::save_double(state, r.y);
    return exp(mean + r.x * stddev);
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using MRG31k3p
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_mrg31k3p* state, double mean, double stddev)
{
    auto state1 = state->next();
    auto state2 = state->next();

    double2 r
        = rocrand_device::detail::mrg_normal_distribution_double2<rocrand_state_mrg31k3p>(state1,
                                                                                          state2);
    return double2{exp(mean + (stddev * r.x)), exp(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using MRG32k3a
 * generator in \p state, and increments position of the generator by one.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, returns first of them,
 * and saves the second to be returned on the next call.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ float
    rocrand_log_normal(rocrand_state_mrg32k3a* state, float mean, float stddev)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg32k3a> bm_helper;

    if(bm_helper::has_float(state))
    {
        return expf(mean + (stddev * bm_helper::get_float(state)));
    }

    auto state1 = state->next();
    auto state2 = state->next();

    float2 r
        = rocrand_device::detail::mrg_normal_distribution2<rocrand_state_mrg32k3a>(state1, state2);
    bm_helper::save_float(state, r.y);
    return expf(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using MRG32k3a
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_mrg32k3a* state, float mean, float stddev)
{
    auto state1 = state->next();
    auto state2 = state->next();

    float2 r
        = rocrand_device::detail::mrg_normal_distribution2<rocrand_state_mrg32k3a>(state1, state2);
    return float2 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y))
    };
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using MRG32k3a
 * generator in \p state, and increments position of the generator by one.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * \p double values, transforms them to log-normally distributed \p double values, returns
 * first of them, and saves the second to be returned on the next call.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ double
    rocrand_log_normal_double(rocrand_state_mrg32k3a* state, double mean, double stddev)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg32k3a> bm_helper;

    if(bm_helper::has_double(state))
    {
        return exp(mean + (stddev * bm_helper::get_double(state)));
    }

    auto state1 = state->next();
    auto state2 = state->next();

    double2 r
        = rocrand_device::detail::mrg_normal_distribution_double2<rocrand_state_mrg32k3a>(state1,
                                                                                          state2);
    bm_helper::save_double(state, r.y);
    return exp(mean + r.x * stddev);
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using MRG32k3a
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_mrg32k3a* state, double mean, double stddev)
{
    auto state1 = state->next();
    auto state2 = state->next();

    double2 r
        = rocrand_device::detail::mrg_normal_distribution_double2<rocrand_state_mrg32k3a>(state1,
                                                                                          state2);
    return double2 {
        exp(mean + (stddev * r.x)),
        exp(mean + (stddev * r.y))
    };
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using XORWOW
 * generator in \p state, and increments position of the generator by one.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, returns first of them,
 * and saves the second to be returned on the next call.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ float
    rocrand_log_normal(rocrand_state_xorwow* state, float mean, float stddev)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_xorwow> bm_helper;

    if(bm_helper::has_float(state))
    {
        return expf(mean + (stddev * bm_helper::get_float(state)));
    }

    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    bm_helper::save_float(state, r.y);
    return expf(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using XORWOW
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_xorwow* state, float mean, float stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    return float2 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y))
    };
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using XORWOW
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * \p double values, transforms them to log-normally distributed \p double values, returns
 * first of them, and saves the second to be returned on the next call.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ double
    rocrand_log_normal_double(rocrand_state_xorwow* state, double mean, double stddev)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_xorwow> bm_helper;

    if(bm_helper::has_double(state))
    {
        return exp(mean + (stddev * bm_helper::get_double(state)));
    }

    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4 { state1, state2, state3, state4 }
    );
    bm_helper::save_double(state, r.y);
    return exp(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using XORWOW
 * generator in \p state, and increments position of the generator by four.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_xorwow* state, double mean, double stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4 { state1, state2, state3, state4 }
    );
    return double2 {
        exp(mean + (stddev * r.x)),
        exp(mean + (stddev * r.y))
    };
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using MTGP32
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__
float rocrand_log_normal(rocrand_state_mtgp32* state, float mean, float stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using MTGP32
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__
float2 rocrand_log_normal2(rocrand_state_mtgp32* state, float mean, float stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    return float2{expf(mean + (stddev * r.x)), expf(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using MTGP32
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__
double rocrand_log_normal_double(rocrand_state_mtgp32* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using MTGP32
 * generator in \p state, and increments position of the generator by four.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__
double2 rocrand_log_normal_double2(rocrand_state_mtgp32* state, double mean, double stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4{state1, state2, state3, state4});
    return double2{exp(mean + (stddev * r.x)), exp(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using SOBOL32
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_sobol32* state, float mean, float stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using SOBOL32
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_sobol32* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using SCRAMBLED_SOBOL32
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_scrambled_sobol32* state, float mean, float stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using SCRAMBLED_SOBOL32
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_scrambled_sobol32* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using SOBOL64
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_sobol64* state, float mean, float stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using SOBOL64
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_sobol64* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using SCRAMBLED_SOBOL64
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_scrambled_sobol64* state, float mean, float stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using SCRAMBLED_SOBOL64
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_scrambled_sobol64* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using LFSR113
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_lfsr113* state, float mean, float stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using LFSR113
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_lfsr113* state, float mean, float stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    return float2{expf(mean + (stddev * r.x)), expf(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using LFSR113
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_lfsr113* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using LFSR113
 * generator in \p state, and increments position of the generator by four.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_lfsr113* state, double mean, double stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4{state1, state2, state3, state4});
    return double2{exp(mean + (stddev * r.x)), exp(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using Threefry
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_threefry2x32_20* state, double mean, double stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using Threefry
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_threefry2x32_20* state, float mean, float stddev)
{
    float2 r = rocrand_device::detail::normal_distribution2(rocrand2(state));
    return float2{expf(mean + (stddev * r.x)), expf(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using Threefry
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_threefry2x32_20* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using Threefry
 * generator in \p state, and increments position of the generator by four.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_threefry2x32_20* state, double mean, double stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4{state1, state2, state3, state4});
    return double2{exp(mean + (stddev * r.x)), exp(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using Threefry
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_threefry2x64_20* state, double mean, double stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using Threefry
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_threefry2x64_20* state, float mean, float stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    return float2{expf(mean + (stddev * r.x)), expf(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using Threefry
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_threefry2x64_20* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using Threefry
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_threefry2x64_20* state, double mean, double stddev)
{
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand2(state));
    return double2{exp(mean + (stddev * r.x)), exp(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using Threefry
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_threefry4x32_20* state, double mean, double stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using Threefry
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_threefry4x32_20* state, float mean, float stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    return float2{expf(mean + (stddev * r.x)), expf(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using Threefry
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_threefry4x32_20* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using Threefry
 * generator in \p state, and increments position of the generator by four.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_threefry4x32_20* state, double mean, double stddev)
{
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    return double2{exp(mean + (stddev * r.x)), exp(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p float value.
 *
 * Generates and returns a log-normally distributed \p float value using Threefry
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_log_normal(rocrand_state_threefry4x64_20* state, double mean, double stddev)
{
    float r = rocrand_device::detail::normal_distribution(rocrand(state));
    return expf(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p float values.
 *
 * Generates and returns two log-normally distributed \p float values using Threefry
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_log_normal2(rocrand_state_threefry4x64_20* state, float mean, float stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    return float2{expf(mean + (stddev * r.x)), expf(mean + (stddev * r.y))};
}

/**
 * \brief Returns a log-normally distributed \p double value.
 *
 * Generates and returns a log-normally distributed \p double value using Threefry
 * generator in \p state, and increments position of the generator by one.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Log-normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_log_normal_double(rocrand_state_threefry4x64_20* state, double mean, double stddev)
{
    double r = rocrand_device::detail::normal_distribution_double(rocrand(state));
    return exp(mean + (stddev * r));
}

/**
 * \brief Returns two log-normally distributed \p double values.
 *
 * Generates and returns two log-normally distributed \p double values using Threefry
 * generator in \p state, and increments position of the generator by two.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, transforms them to log-normally distributed values, and returns both.
 *
 * \param state  Pointer to a state to use
 * \param mean   Mean of the related log-normal distribution
 * \param stddev Standard deviation of the related log-normal distribution
 *
 * \return Two log-normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_log_normal_double2(rocrand_state_threefry4x64_20* state, double mean, double stddev)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    double2 r = rocrand_device::detail::normal_distribution_double2(ulonglong2{state1, state2});
    return double2{exp(mean + (stddev * r.x)), exp(mean + (stddev * r.y))};
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_LOG_NORMAL_H_
