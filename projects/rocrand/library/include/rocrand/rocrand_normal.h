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

#ifndef ROCRAND_NORMAL_H_
#define ROCRAND_NORMAL_H_

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

#include "rocrand/rocrand_uniform.h"

#include <hip/hip_runtime.h>

#include <math.h>

namespace rocrand_device {
namespace detail {

__forceinline__ __device__ __host__ float2 box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = ROCRAND_2POW32_INV + (x * ROCRAND_2POW32_INV);
    float v = ROCRAND_2POW32_INV_2PI + (y * ROCRAND_2POW32_INV_2PI);
    float s = sqrtf(-2.0f * logf(u));
    #ifdef __HIP_DEVICE_COMPILE__
        __sincosf(v, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sinf(v) * s;
        result.y = cosf(v) * s;
    #endif
    return result;
}

__forceinline__ __device__ __host__ float2 box_muller(unsigned long long v)
{
    unsigned int x = static_cast<unsigned int>(v);
    unsigned int y = static_cast<unsigned int>(v >> 32);

    return box_muller(x, y);
}

__forceinline__ __device__ __host__ double2 box_muller_double(uint4 v)
{
    double2 result;
    unsigned long long int v1 = (unsigned long long int)v.x ^
        ((unsigned long long int)v.y << (53 - 32));
    double u = ROCRAND_2POW53_INV_DOUBLE + (v1 * ROCRAND_2POW53_INV_DOUBLE);
    unsigned long long int v2 = (unsigned long long int)v.z ^
        ((unsigned long long int)v.w << (53 - 32));
    double w = (ROCRAND_2POW53_INV_DOUBLE * 2.0) +
        (v2 * (ROCRAND_2POW53_INV_DOUBLE * 2.0));
    double s = sqrt(-2.0 * log(u));
    #ifdef __HIP_DEVICE_COMPILE__
        sincospi(w, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sin(w * ROCRAND_PI_DOUBLE) * s;
        result.y = cos(w * ROCRAND_PI_DOUBLE) * s;
    #endif
    return result;
}

__forceinline__ __device__ __host__ double2 box_muller_double(ulonglong2 v)
{
    unsigned int x = static_cast<unsigned int>(v.x);
    unsigned int y = static_cast<unsigned int>(v.x >> 32);
    unsigned int z = static_cast<unsigned int>(v.y);
    unsigned int w = static_cast<unsigned int>(v.y >> 32);

    return box_muller_double(make_uint4(x, y, z, w));
}

__forceinline__ __device__ __host__ __half2 box_muller_half(unsigned short x, unsigned short y)
{
    #if defined(ROCRAND_HALF_MATH_SUPPORTED)
    __half u = __float2half(ROCRAND_2POW16_INV + (x * ROCRAND_2POW16_INV));
    __half v = __float2half(ROCRAND_2POW16_INV_2PI + (y * ROCRAND_2POW16_INV_2PI));
    __half s = hsqrt(__hmul(__float2half(-2.0f), hlog(u)));
    return __half2 {
        __hmul(hsin(v), s),
        __hmul(hcos(v), s)
    };
    #else
    float2 r;
    float u = ROCRAND_2POW16_INV + (x * ROCRAND_2POW16_INV);
    float v = ROCRAND_2POW16_INV_2PI + (y * ROCRAND_2POW16_INV_2PI);
    float s = sqrtf(-2.0f * logf(u));
    #ifdef __HIP_DEVICE_COMPILE__
        __sincosf(v, &r.x, &r.y);
        r.x *= s;
        r.y *= s;
    #else
        r.x = sinf(v) * s;
        r.y = cosf(v) * s;
    #endif
    return __half2 {
        __float2half(r.x),
        __float2half(r.y)
    };
    #endif
}

template<typename state_type>
__forceinline__ __device__ __host__ float2 mrg_box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float  u = rocrand_device::detail::mrg_uniform_distribution<state_type>(x);
    float  v = rocrand_device::detail::mrg_uniform_distribution<state_type>(y) * ROCRAND_2PI;
    float s = sqrtf(-2.0f * logf(u));
    #ifdef __HIP_DEVICE_COMPILE__
        __sincosf(v, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sinf(v) * s;
        result.y = cosf(v) * s;
    #endif
    return result;
}

template<typename state_type>
__forceinline__ __device__ __host__ double2 mrg_box_muller_double(unsigned int x, unsigned int y)
{
    double2 result;
    double  u = rocrand_device::detail::mrg_uniform_distribution<state_type>(x);
    double  v = rocrand_device::detail::mrg_uniform_distribution<state_type>(y) * 2.0;
    double s = sqrt(-2.0 * log(u));
    #ifdef __HIP_DEVICE_COMPILE__
        sincospi(v, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sin(v * ROCRAND_PI_DOUBLE) * s;
        result.y = cos(v * ROCRAND_PI_DOUBLE) * s;
    #endif
    return result;
}

__forceinline__ __device__ __host__ float roc_f_erfinv(float x)
{
    float tt1, tt2, lnx, sgn;
    sgn = (x < 0.0f) ? -1.0f : 1.0f;

    x = (1.0f - x) * (1.0f + x);
    lnx = logf(x);

    #ifdef __HIP_DEVICE_COMPILE__
    if (isnan(lnx))
    #else
    if (std::isnan(lnx))
    #endif
        return 1.0f;
    #ifdef __HIP_DEVICE_COMPILE__
    else if (isinf(lnx))
    #else
    else if (std::isinf(lnx))
    #endif
        return 0.0f;

    tt1 = 2.0f / (ROCRAND_PI * 0.147f) + 0.5f * lnx;
    tt2 = 1.0f / (0.147f) * lnx;

    return(sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
}

__forceinline__ __device__ __host__ double roc_d_erfinv(double x)
{
    double tt1, tt2, lnx, sgn;
    sgn = (x < 0.0) ? -1.0 : 1.0;

    x = (1.0 - x) * (1.0 + x);
    lnx = log(x);

    #ifdef __HIP_DEVICE_COMPILE__
    if (isnan(lnx))
    #else
    if (std::isnan(lnx))
    #endif
        return 1.0;
    #ifdef __HIP_DEVICE_COMPILE__
    else if (isinf(lnx))
    #else
    else if (std::isinf(lnx))
    #endif
        return 0.0;

    tt1 = 2.0 / (ROCRAND_PI_DOUBLE * 0.147) + 0.5 * lnx;
    tt2 = 1.0 / (0.147) * lnx;

    return(sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2)));
}

__forceinline__ __device__ __host__ float normal_distribution(unsigned int x)
{
    float p = ::rocrand_device::detail::uniform_distribution(x);
    float v = ROCRAND_SQRT2 * ::rocrand_device::detail::roc_f_erfinv(2.0f * p - 1.0f);
    return v;
}

__forceinline__ __device__ __host__ float normal_distribution(unsigned long long int x)
{
    float p = ::rocrand_device::detail::uniform_distribution(x);
    float v = ROCRAND_SQRT2 * ::rocrand_device::detail::roc_f_erfinv(2.0f * p - 1.0f);
    return v;
}

__forceinline__ __device__ __host__ float2 normal_distribution2(unsigned int v1, unsigned int v2)
{
    return ::rocrand_device::detail::box_muller(v1, v2);
}

__forceinline__ __device__ __host__ float2 normal_distribution2(uint2 v)
{
    return ::rocrand_device::detail::box_muller(v.x, v.y);
}

__forceinline__ __device__ __host__ float2 normal_distribution2(unsigned long long v)
{
    return ::rocrand_device::detail::box_muller(v);
}

__forceinline__ __device__ __host__ float4 normal_distribution4(uint4 v)
{
    float2 r1 = ::rocrand_device::detail::box_muller(v.x, v.y);
    float2 r2 = ::rocrand_device::detail::box_muller(v.z, v.w);
    return float4{
        r1.x,
        r1.y,
        r2.x,
        r2.y
    };
}

__forceinline__ __device__ __host__ float4 normal_distribution4(longlong2 v)
{
    float2 r1 = ::rocrand_device::detail::box_muller(v.x);
    float2 r2 = ::rocrand_device::detail::box_muller(v.y);
    return float4{r1.x, r1.y, r2.x, r2.y};
}

__forceinline__ __device__ __host__ float4 normal_distribution4(unsigned long long v1,
                                                                unsigned long long v2)
{
    float2 r1 = ::rocrand_device::detail::box_muller(v1);
    float2 r2 = ::rocrand_device::detail::box_muller(v2);
    return float4{r1.x, r1.y, r2.x, r2.y};
}

__forceinline__ __device__ __host__ double normal_distribution_double(unsigned int x)
{
    double p = ::rocrand_device::detail::uniform_distribution_double(x);
    double v = ROCRAND_SQRT2 * ::rocrand_device::detail::roc_d_erfinv(2.0 * p - 1.0);
    return v;
}

__forceinline__ __device__ __host__ double normal_distribution_double(unsigned long long int x)
{
    double p = ::rocrand_device::detail::uniform_distribution_double(x);
    double v = ROCRAND_SQRT2 * ::rocrand_device::detail::roc_d_erfinv(2.0 * p - 1.0);
    return v;
}

__forceinline__ __device__ __host__ double2 normal_distribution_double2(uint4 v)
{
    return ::rocrand_device::detail::box_muller_double(v);
}

__forceinline__ __device__ __host__ double2 normal_distribution_double2(ulonglong2 v)
{
    return ::rocrand_device::detail::box_muller_double(v);
}

__forceinline__ __device__ __host__ __half2 normal_distribution_half2(unsigned int v)
{
    return ::rocrand_device::detail::box_muller_half(
        static_cast<unsigned short>(v),
        static_cast<unsigned short>(v >> 16)
    );
}

__forceinline__ __device__ __host__ __half2 normal_distribution_half2(unsigned long long v)
{
    return ::rocrand_device::detail::box_muller_half(static_cast<unsigned short>(v),
                                                     static_cast<unsigned short>(v >> 32));
}

template<typename state_type>
__forceinline__ __device__ __host__ float2 mrg_normal_distribution2(unsigned int v1,
                                                                    unsigned int v2)
{
    return ::rocrand_device::detail::mrg_box_muller<state_type>(v1, v2);
}

template<typename state_type>
__forceinline__ __device__ __host__ double2 mrg_normal_distribution_double2(unsigned int v1,
                                                                            unsigned int v2)
{
    return ::rocrand_device::detail::mrg_box_muller_double<state_type>(v1, v2);
}

template<typename state_type>
__forceinline__ __device__ __host__ __half2 mrg_normal_distribution_half2(unsigned int v)
{
    v = rocrand_device::detail::mrg_uniform_distribution_uint<state_type>(v);
    return ::rocrand_device::detail::box_muller_half(
        static_cast<unsigned short>(v),
        static_cast<unsigned short>(v >> 16)
    );
}

} // end namespace detail
} // end namespace rocrand_device

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using Philox
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ float rocrand_normal(rocrand_state_philox4x32_10* state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_philox4x32_10> bm_helper;

    if(bm_helper::has_float(state))
    {
        return bm_helper::get_float(state);
    }

    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    bm_helper::save_float(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using Philox
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_philox4x32_10* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    return rocrand_device::detail::normal_distribution2(state1, state2);
}

/**
 * \brief Returns four normally distributed \p float values.
 *
 * Generates and returns four normally distributed \p float values using Philox
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate four normally
 * distributed values, and returns them.
 *
 * \param state Pointer to a state to use
 *
 * \return Four normally distributed \p float value as \p float4
 */
__forceinline__ __device__ __host__
float4 rocrand_normal4(rocrand_state_philox4x32_10* state)
{
    return rocrand_device::detail::normal_distribution4(rocrand4(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using Philox
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ double rocrand_normal_double(rocrand_state_philox4x32_10* state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_philox4x32_10> bm_helper;

    if(bm_helper::has_double(state))
    {
        return bm_helper::get_double(state);
    }
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    bm_helper::save_double(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using Philox
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double values as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_philox4x32_10* state)
{
    return rocrand_device::detail::normal_distribution_double2(rocrand4(state));
}

/**
 * \brief Returns four normally distributed \p double values.
 *
 * Generates and returns four normally distributed \p double values using Philox
 * generator in \p state, and increments position of the generator by eight.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate four normally
 * distributed values, and returns them.
 *
 * \param state Pointer to a state to use
 *
 * \return Four normally distributed \p double values as \p double4
 */
__forceinline__ __device__ __host__
double4 rocrand_normal_double4(rocrand_state_philox4x32_10* state)
{
    double2 r1, r2;
    r1 = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    r2 = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    return double4 {
        r1.x, r1.y, r2.x, r2.y
    };
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using MRG31k3p
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ float rocrand_normal(rocrand_state_mrg31k3p* state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg31k3p> bm_helper;

    if(bm_helper::has_float(state))
    {
        return bm_helper::get_float(state);
    }

    auto state1 = state->next();
    auto state2 = state->next();

    float2 r
        = rocrand_device::detail::mrg_normal_distribution2<rocrand_state_mrg31k3p>(state1, state2);
    bm_helper::save_float(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using MRG31k3p
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_mrg31k3p* state)
{
    auto state1 = state->next();
    auto state2 = state->next();

    return rocrand_device::detail::mrg_normal_distribution2<rocrand_state_mrg31k3p>(state1, state2);
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using MRG31k3p
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ double rocrand_normal_double(rocrand_state_mrg31k3p* state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg31k3p> bm_helper;

    if(bm_helper::has_double(state))
    {
        return bm_helper::get_double(state);
    }

    auto state1 = state->next();
    auto state2 = state->next();

    double2 r
        = rocrand_device::detail::mrg_normal_distribution_double2<rocrand_state_mrg31k3p>(state1,
                                                                                          state2);
    bm_helper::save_double(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using MRG31k3p
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_mrg31k3p* state)
{
    auto state1 = state->next();
    auto state2 = state->next();

    return rocrand_device::detail::mrg_normal_distribution_double2<rocrand_state_mrg31k3p>(state1,
                                                                                           state2);
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using MRG32k3a
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ float rocrand_normal(rocrand_state_mrg32k3a* state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg32k3a> bm_helper;

    if(bm_helper::has_float(state))
    {
        return bm_helper::get_float(state);
    }

    auto state1 = state->next();
    auto state2 = state->next();

    float2 r
        = rocrand_device::detail::mrg_normal_distribution2<rocrand_state_mrg32k3a>(state1, state2);
    bm_helper::save_float(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using MRG32k3a
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_mrg32k3a* state)
{
    auto state1 = state->next();
    auto state2 = state->next();

    return rocrand_device::detail::mrg_normal_distribution2<rocrand_state_mrg32k3a>(state1, state2);
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using MRG32k3a
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ double rocrand_normal_double(rocrand_state_mrg32k3a* state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg32k3a> bm_helper;

    if(bm_helper::has_double(state))
    {
        return bm_helper::get_double(state);
    }

    auto state1 = state->next();
    auto state2 = state->next();

    double2 r
        = rocrand_device::detail::mrg_normal_distribution_double2<rocrand_state_mrg32k3a>(state1,
                                                                                          state2);
    bm_helper::save_double(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using MRG32k3a
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_mrg32k3a* state)
{
    auto state1 = state->next();
    auto state2 = state->next();

    return rocrand_device::detail::mrg_normal_distribution_double2<rocrand_state_mrg32k3a>(state1,
                                                                                           state2);
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using XORWOW
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ float rocrand_normal(rocrand_state_xorwow* state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_xorwow> bm_helper;

    if(bm_helper::has_float(state))
    {
        return bm_helper::get_float(state);
    }
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    float2 r = rocrand_device::detail::normal_distribution2(state1, state2);
    bm_helper::save_float(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using XORWOW
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float values as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_xorwow* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    return rocrand_device::detail::normal_distribution2(state1, state2);
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using XORWOW
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
__forceinline__ __device__ __host__ double rocrand_normal_double(rocrand_state_xorwow* state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_xorwow> bm_helper;

    if(bm_helper::has_double(state))
    {
        return bm_helper::get_double(state);
    }

    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4 { state1, state2, state3, state4 }
    );
    bm_helper::save_double(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using XORWOW
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_xorwow* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    return rocrand_device::detail::normal_distribution_double2(
        uint4 { state1, state2, state3, state4 }
    );
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using MTGP32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__
float rocrand_normal(rocrand_state_mtgp32* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using MTGP32
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float values as \p float2
 */
__forceinline__ __device__
float2 rocrand_normal2(rocrand_state_mtgp32* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    return rocrand_device::detail::normal_distribution2(state1, state2);
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using MTGP32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__
double rocrand_normal_double(rocrand_state_mtgp32* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using MTGP32
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__
double2 rocrand_normal_double2(rocrand_state_mtgp32* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    return rocrand_device::detail::normal_distribution_double2(
        uint4{state1, state2, state3, state4});
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using SOBOL32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_sobol32* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using SOBOL32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_sobol32* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using SCRAMBLED_SOBOL32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_scrambled_sobol32* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using SCRAMBLED_SOBOL32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_scrambled_sobol32* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using SOBOL64
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_sobol64* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using SOBOL64
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_sobol64* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using SCRAMBLED_SOBOL64
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_scrambled_sobol64* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using SCRAMBLED_SOBOL64
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_scrambled_sobol64* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using LFSR113
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_lfsr113* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using LFSR113
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_lfsr113* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    return rocrand_device::detail::normal_distribution2(state1, state2);
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using LFSR113
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_lfsr113* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using LFSR113
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_lfsr113* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);
    auto state3 = rocrand(state);
    auto state4 = rocrand(state);

    return rocrand_device::detail::normal_distribution_double2(
        uint4{state1, state2, state3, state4});
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p float value using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_threefry2x32_20* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using ThreeFry
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_threefry2x32_20* state)
{
    return rocrand_device::detail::normal_distribution2(rocrand2(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_threefry2x32_20* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using ThreeFry
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_threefry2x32_20* state)
{
    auto state1 = rocrand2(state);
    auto state2 = rocrand2(state);

    return rocrand_device::detail::normal_distribution_double2(
        uint4{state1.x, state1.y, state2.x, state2.y});
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p float value using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_threefry2x64_20* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_threefry2x64_20* state)
{
    return rocrand_device::detail::normal_distribution2(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_threefry2x64_20* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using ThreeFry
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_threefry2x64_20* state)
{
    return rocrand_device::detail::normal_distribution_double2(rocrand2(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p float value using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_threefry4x32_20* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using ThreeFry
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_threefry4x32_20* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    return rocrand_device::detail::normal_distribution2(state1, state2);
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_threefry4x32_20* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using ThreeFry
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_threefry4x32_20* state)
{
    return rocrand_device::detail::normal_distribution_double2(rocrand4(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p float value using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
__forceinline__ __device__ __host__
float rocrand_normal(rocrand_state_threefry4x64_20* state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using ThreeFry
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
__forceinline__ __device__ __host__
float2 rocrand_normal2(rocrand_state_threefry4x64_20* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    return rocrand_device::detail::normal_distribution2(state1, state2);
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using ThreeFry
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
__forceinline__ __device__ __host__
double rocrand_normal_double(rocrand_state_threefry4x64_20* state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using ThreeFry
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
__forceinline__ __device__ __host__
double2 rocrand_normal_double2(rocrand_state_threefry4x64_20* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    return rocrand_device::detail::normal_distribution_double2(ulonglong2{state1, state2});
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_NORMAL_H_
