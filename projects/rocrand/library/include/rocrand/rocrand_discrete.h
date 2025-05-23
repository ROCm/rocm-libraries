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

#ifndef ROCRAND_DISCRETE_H_
#define ROCRAND_DISCRETE_H_

#include "rocrand/rocrand_common.h"
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

#include <hip/hip_runtime.h>

#include <math.h>

// On certain architectures such as NAVI2 and NAVI3, double arithmetic is significantly slower.
// In such cases we want to prefer the CDF method over the alias method.
// This macro is undefined at the end of the file.
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX10__) || defined(__GFX11__))
    #define ROCRAND_PREFER_CDF_OVER_ALIAS
#endif

// Alias method
//
// Walker, A. J.
// An Efficient Method for Generating Discrete Random Variables with General Distributions, 1977
//
// Vose M. D.
// A Linear Algorithm For Generating Random Numbers With a Given Distribution, 1991

namespace rocrand_device {
namespace detail {

__forceinline__ __device__ __host__ unsigned int
    discrete_alias(const double       x,
                   const unsigned int size,
                   const unsigned int offset,
                   const unsigned int* __restrict__ alias,
                   const double* __restrict__ probability)
{
    // Calculate value using Alias table

    // x is [0, 1)
    const double       nx  = size * x;
    const double       fnx = floor(nx);
    const double       y   = nx - fnx;
    const unsigned int i   = static_cast<unsigned int>(fnx);
    return offset + (y < probability[i] ? i : alias[i]);
}

__forceinline__ __device__ __host__ unsigned int
    discrete_alias(const double x, const rocrand_discrete_distribution_st& dis)
{
    return discrete_alias(x, dis.size, dis.offset, dis.alias, dis.probability);
}

__forceinline__ __device__ __host__ unsigned int
    discrete_alias(const unsigned int r, const rocrand_discrete_distribution_st& dis)
{
    constexpr double inv_double_32 = ROCRAND_2POW32_INV_DOUBLE;
    const double x = r * inv_double_32;
    return discrete_alias(x, dis);
}

// To prevent ambiguity compile error when compiler is facing the type "unsigned long"!!!
__forceinline__ __device__ __host__ unsigned int
    discrete_alias(const unsigned long r, const rocrand_discrete_distribution_st& dis)
{
    constexpr double inv_double_32 = ROCRAND_2POW32_INV_DOUBLE;
    const double x = r * inv_double_32;
    return discrete_alias(x, dis);
}

__forceinline__ __device__ __host__ unsigned int
    discrete_alias(const unsigned long long int r, const rocrand_discrete_distribution_st& dis)
{
    constexpr double inv_double_64 = ROCRAND_2POW64_INV_DOUBLE;
    const double x = r * inv_double_64;
    return discrete_alias(x, dis);
}

__forceinline__ __device__ __host__ unsigned int discrete_cdf(const double       x,
                                                              const unsigned int size,
                                                              const unsigned int offset,
                                                              const double* __restrict__ cdf)
{
    // Calculate value using binary search in CDF

    unsigned int min = 0;
    unsigned int max = size - 1;
    do
    {
        const unsigned int center = (min + max) / 2;
        const double       p      = cdf[center];
        if(x > p)
        {
            min = center + 1;
        }
        else
        {
            max = center;
        }
    }
    while(min != max);

    return offset + min;
}

__forceinline__ __device__ __host__ unsigned int
    discrete_cdf(const double x, const rocrand_discrete_distribution_st& dis)
{
    return discrete_cdf(x, dis.size, dis.offset, dis.cdf);
}

__forceinline__ __device__ __host__ unsigned int
    discrete_cdf(const unsigned int r, const rocrand_discrete_distribution_st& dis)
{
    constexpr double inv_double_32 = ROCRAND_2POW32_INV_DOUBLE;
    const double x = r * inv_double_32;
    return discrete_cdf(x, dis);
}

// To prevent ambiguity compile error when compiler is facing the type "unsigned long"!!!
__forceinline__ __device__ __host__ unsigned int
    discrete_cdf(const unsigned long r, const rocrand_discrete_distribution_st& dis)
{
    constexpr double inv_double_32 = ROCRAND_2POW32_INV_DOUBLE;
    const double x = r * inv_double_32;
    return discrete_cdf(x, dis);
}

__forceinline__ __device__ __host__ unsigned int
    discrete_cdf(const unsigned long long int r, const rocrand_discrete_distribution_st& dis)
{
    constexpr double inv_double_64 = ROCRAND_2POW64_INV_DOUBLE;
    const double x = r * inv_double_64;
    return discrete_cdf(x, dis);
}

} // end namespace detail
} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using Philox generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_philox4x32_10*        state,
                              const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns four discrete distributed <tt>unsigned int</tt> values.
 *
 * Returns four <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using Philox generator in \p state, and increments
 * the position of the generator by four.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return Four <tt>unsigned int</tt> values distributed according to \p discrete_distribution as \p uint4
 */
__forceinline__ __device__ __host__
uint4 rocrand_discrete4(rocrand_state_philox4x32_10*        state,
                        const rocrand_discrete_distribution discrete_distribution)
{
    const uint4 u4 = rocrand4(state);
    return uint4 {
        rocrand_device::detail::discrete_alias(u4.x, *discrete_distribution),
        rocrand_device::detail::discrete_alias(u4.y, *discrete_distribution),
        rocrand_device::detail::discrete_alias(u4.z, *discrete_distribution),
        rocrand_device::detail::discrete_alias(u4.w, *discrete_distribution)
    };
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using MRG31k3p generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_mrg31k3p*             state,
                              const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using MRG32k3a generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_mrg32k3a*             state,
                              const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using XORWOW generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_xorwow*               state,
                              const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using MTGP32 generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__
unsigned int rocrand_discrete(rocrand_state_mtgp32*               state,
                              const rocrand_discrete_distribution discrete_distribution)
{
#ifdef ROCRAND_PREFER_CDF_OVER_ALIAS
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
#else
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
#endif
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using SOBOL32 generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_sobol32*              state,
                              const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to discrete distribution
 * \p discrete_distribution using SCRAMBLED_SOBOL32 generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_scrambled_sobol32*    state,
                              const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned long long int</tt> value.
 *
 * Returns a <tt>unsigned long long int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using SOBOL64 generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned long long int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_sobol64*              state,
                              const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned long long int</tt> value.
 *
 * Returns a <tt>unsigned long long int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using SCRAMBLED_SOBOL64 generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned long long int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_scrambled_sobol64*    state,
                              const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using LFSR113 generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_lfsr113*              state,
                              const rocrand_discrete_distribution discrete_distribution)
{
#ifdef ROCRAND_PREFER_CDF_OVER_ALIAS
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
#else
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
#endif
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using ThreeFry generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_threefry2x32_20*      state,
                              const rocrand_discrete_distribution discrete_distribution)
{
#ifdef ROCRAND_PREFER_CDF_OVER_ALIAS
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
#else
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
#endif
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using ThreeFry generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_threefry2x64_20*      state,
                              const rocrand_discrete_distribution discrete_distribution)
{
#ifdef ROCRAND_PREFER_CDF_OVER_ALIAS
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
#else
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
#endif
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using ThreeFry generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_threefry4x32_20*      state,
                              const rocrand_discrete_distribution discrete_distribution)
{
#ifdef ROCRAND_PREFER_CDF_OVER_ALIAS
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
#else
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
#endif
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using ThreeFry generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state Pointer to a state to use
 * \param discrete_distribution Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
__forceinline__ __device__ __host__
unsigned int rocrand_discrete(rocrand_state_threefry4x64_20*      state,
                              const rocrand_discrete_distribution discrete_distribution)
{
#ifdef ROCRAND_PREFER_CDF_OVER_ALIAS
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
#else
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
#endif
}

/** @} */ // end of group rocranddevice

// Undefine the macro that may be defined at the top of the file!
#if defined(ROCRAND_PREFER_CDF_OVER_ALIAS)
    #undef ROCRAND_PREFER_CDF_OVER_ALIAS
#endif

#endif // ROCRAND_DISCRETE_H_
