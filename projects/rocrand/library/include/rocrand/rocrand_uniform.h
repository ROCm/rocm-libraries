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

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

#ifndef ROCRAND_UNIFORM_H_
#define ROCRAND_UNIFORM_H_

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

#include "rocrand/rocrand_common.h"

#include <hip/hip_runtime.h>

namespace rocrand_device {
namespace detail {

struct two_uints
{
    unsigned int x;
    unsigned int y;
};

union two_uints_to_ulong
{
    two_uints uint2_value;
    unsigned long long int ulong_value;
};

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f.
__forceinline__ __device__ __host__ float uniform_distribution(unsigned int v)
{
    return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV);
}

// For unsigned integer between 0 and ULLONG_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f.
__forceinline__ __device__ __host__ float uniform_distribution(unsigned long long int v)
{
    return ROCRAND_2POW32_INV + (v >> 32) * ROCRAND_2POW32_INV;
}

__forceinline__ __device__ __host__ float4 uniform_distribution4(uint4 v)
{
   return float4 {
       ROCRAND_2POW32_INV + (v.x * ROCRAND_2POW32_INV),
       ROCRAND_2POW32_INV + (v.y * ROCRAND_2POW32_INV),
       ROCRAND_2POW32_INV + (v.z * ROCRAND_2POW32_INV),
       ROCRAND_2POW32_INV + (v.w * ROCRAND_2POW32_INV)
   };
}

__forceinline__ __device__ __host__ float4 uniform_distribution4(ulonglong4 v)
{
    return float4{ROCRAND_2POW64_INV + (v.x * ROCRAND_2POW64_INV),
                  ROCRAND_2POW64_INV + (v.y * ROCRAND_2POW64_INV),
                  ROCRAND_2POW64_INV + (v.z * ROCRAND_2POW64_INV),
                  ROCRAND_2POW64_INV + (v.w * ROCRAND_2POW64_INV)};
}

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0 and 1.0, excluding 0.0 and including 1.0.
__forceinline__ __device__ __host__ double uniform_distribution_double(unsigned int v)
{
    return ROCRAND_2POW32_INV_DOUBLE + (v * ROCRAND_2POW32_INV_DOUBLE);
}

__forceinline__ __device__ __host__ double uniform_distribution_double(unsigned int v1,
                                                                       unsigned int v2)
{
    two_uints_to_ulong v;
    v.uint2_value.x = v1;
    v.uint2_value.y = (v2 >> 11);
    return ROCRAND_2POW53_INV_DOUBLE + (v.ulong_value * ROCRAND_2POW53_INV_DOUBLE);
}

__forceinline__ __device__ __host__ double uniform_distribution_double(unsigned long long int v)
{
    return ROCRAND_2POW53_INV_DOUBLE + (
        // 2^53 is the biggest int that can be stored in double, such
        // that it and all smaller integers can be stored in double
        (v >> 11) * ROCRAND_2POW53_INV_DOUBLE
    );
}

__forceinline__ __device__ __host__ double2 uniform_distribution_double2(uint4 v)
{
    return double2 {
        uniform_distribution_double(v.x, v.y),
        uniform_distribution_double(v.z, v.w)
    };
}

__forceinline__ __device__ __host__ double4 uniform_distribution_double4(uint4 v1, uint4 v2)
{
    return double4 {
        uniform_distribution_double(v1.x, v1.y),
        uniform_distribution_double(v1.z, v1.w),
        uniform_distribution_double(v2.x, v2.y),
        uniform_distribution_double(v2.z, v2.w)
    };
}

__forceinline__ __device__ __host__ double2 uniform_distribution_double2(ulonglong2 v)
{
    return double2{uniform_distribution_double(v.x), uniform_distribution_double(v.y)};
}

__forceinline__ __device__ __host__ double2 uniform_distribution_double2(ulonglong4 v)
{
    return double2{uniform_distribution_double(v.x), uniform_distribution_double(v.y)};
}

__forceinline__ __device__ __host__ double4 uniform_distribution_double4(ulonglong4 v)
{
    return double4{uniform_distribution_double(v.x),
                   uniform_distribution_double(v.z),
                   uniform_distribution_double(v.x),
                   uniform_distribution_double(v.z)};
}

__forceinline__ __device__ __host__ __half uniform_distribution_half(unsigned short v)
{
    return __float2half(ROCRAND_2POW16_INV + (v * ROCRAND_2POW16_INV));
}

// For an unsigned integer produced by an MRG-based engine, returns a value
// in range [0, UINT32_MAX].
template<typename state_type>
__forceinline__ __device__ __host__ unsigned int mrg_uniform_distribution_uint(unsigned int v)
    = delete;

template<>
__forceinline__ __device__ __host__ unsigned int
    mrg_uniform_distribution_uint<rocrand_state_mrg31k3p>(unsigned int v)
{
    return static_cast<unsigned int>((v - 1) * ROCRAND_MRG31K3P_UINT32_NORM);
}

template<>
__forceinline__ __device__ __host__ unsigned int
    mrg_uniform_distribution_uint<rocrand_state_mrg32k3a>(unsigned int v)
{
    return static_cast<unsigned int>((v - 1) * ROCRAND_MRG32K3A_UINT_NORM);
}

// For an unsigned integer produced by an MRG-based engine, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f.
template<typename state_type>
__forceinline__ __device__ __host__ float mrg_uniform_distribution(unsigned int v) = delete;

template<>
__forceinline__ __device__ __host__ float
    mrg_uniform_distribution<rocrand_state_mrg31k3p>(unsigned int v)
{
    double ret = static_cast<double>(v) * ROCRAND_MRG31K3P_NORM_DOUBLE;
    return static_cast<float>(ret);
}

template<>
__forceinline__ __device__ __host__ float
    mrg_uniform_distribution<rocrand_state_mrg32k3a>(unsigned int v)
{
    double ret = static_cast<double>(v) * ROCRAND_MRG32K3A_NORM_DOUBLE;
    return static_cast<float>(ret);
}

// For an unsigned integer produced by an MRG generator, returns value between
// 0.0 and 1.0, excluding 0.0 and including 1.0.
template<typename state_type>
__forceinline__ __device__ __host__ double mrg_uniform_distribution_double(unsigned int v) = delete;

template<>
__forceinline__ __device__ __host__ double
    mrg_uniform_distribution_double<rocrand_state_mrg31k3p>(unsigned int v)
{
    double ret = static_cast<double>(v) * ROCRAND_MRG31K3P_NORM_DOUBLE;
    return ret;
}

template<>
__forceinline__ __device__ __host__ double
    mrg_uniform_distribution_double<rocrand_state_mrg32k3a>(unsigned int v)
{
    double ret = static_cast<double>(v) * ROCRAND_MRG32K3A_NORM_DOUBLE;
    return ret;
}

} // end namespace detail
} // end namespace rocrand_device

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using Philox generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_philox4x32_10* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns two uniformly distributed random <tt>float</tt> values
 * from (0; 1] range.
 *
 * Generates and returns two uniformly distributed \p float values from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using Philox generator in \p state, and
 * increments position of the generator by two.
 *
 * \param state Pointer to a state to use
 *
 * \return Two uniformly distributed \p float values from (0; 1] range as \p float2.
 */
__forceinline__ __device__ __host__
float2 rocrand_uniform2(rocrand_state_philox4x32_10* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    return float2 {
        rocrand_device::detail::uniform_distribution(state1),
        rocrand_device::detail::uniform_distribution(state2)
    };
}

/**
 * \brief Returns four uniformly distributed random <tt>float</tt> values
 * from (0; 1] range.
 *
 * Generates and returns four uniformly distributed \p float values from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using Philox generator in \p state, and
 * increments position of the generator by four.
 *
 * \param state Pointer to a state to use
 *
 * \return Four uniformly distributed \p float values from (0; 1] range as \p float4.
 */
__forceinline__ __device__ __host__
float4 rocrand_uniform4(rocrand_state_philox4x32_10* state)
{
    return rocrand_device::detail::uniform_distribution4(rocrand4(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using Philox generator in \p state, and
 * increments position of the generator by two.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_philox4x32_10* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    return rocrand_device::detail::uniform_distribution_double(state1, state2);
}

/**
 * \brief Returns two uniformly distributed random <tt>double</tt> values
 * from (0; 1] range.
 *
 * Generates and returns two uniformly distributed \p double values from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using Philox generator in \p state, and
 * increments position of the generator by four.
 *
 * \param state Pointer to a state to use
 *
 * \return Two uniformly distributed \p double values from (0; 1] range as \p double2.
 */
__forceinline__ __device__ __host__
double2 rocrand_uniform_double2(rocrand_state_philox4x32_10* state)
{
    return rocrand_device::detail::uniform_distribution_double2(rocrand4(state));
}

/**
 * \brief Returns four uniformly distributed random <tt>double</tt> values
 * from (0; 1] range.
 *
 * Generates and returns four uniformly distributed \p double values from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using Philox generator in \p state, and
 * increments position of the generator by eight.
 *
 * \param state Pointer to a state to use
 *
 * \return Four uniformly distributed \p double values from (0; 1] range as \p double4.
 */
__forceinline__ __device__ __host__
double4 rocrand_uniform_double4(rocrand_state_philox4x32_10* state)
{
    return rocrand_device::detail::uniform_distribution_double4(rocrand4(state), rocrand4(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using MRG31K3P generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_mrg31k3p* state)
{
    return rocrand_device::detail::mrg_uniform_distribution<rocrand_state_mrg31k3p>(state->next());
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using MRG31K3P generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_mrg31k3p* state)
{
    return rocrand_device::detail::mrg_uniform_distribution_double<rocrand_state_mrg31k3p>(
        state->next());
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using MRG32K3A generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_mrg32k3a* state)
{
    return rocrand_device::detail::mrg_uniform_distribution<rocrand_state_mrg32k3a>(state->next());
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using MRG32K3A generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_mrg32k3a* state)
{
    return rocrand_device::detail::mrg_uniform_distribution_double<rocrand_state_mrg32k3a>(
        state->next());
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using XORWOW generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_xorwow* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using MRG32K3A generator in \p state, and
 * increments position of the generator by two.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_xorwow* state)
{
    auto state1 = rocrand(state);
    auto state2 = rocrand(state);

    return rocrand_device::detail::uniform_distribution_double(state1, state2);
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using MTGP32 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__
float rocrand_uniform(rocrand_state_mtgp32* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using MTGP32 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__
double rocrand_uniform_double(rocrand_state_mtgp32* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using SOBOL32 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_sobol32* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using SOBOL32 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_sobol32* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using SCRAMBLED_SOBOL32 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_scrambled_sobol32* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using SCRAMBLED_SOBOL32 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_scrambled_sobol32* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using SOBOL64 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_sobol64* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using SOBOL64 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_sobol64* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using SCRAMBLED_SOBOL64 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_scrambled_sobol64* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using SCRAMBLED_SOBOL64 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_scrambled_sobol64* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0f, including \p 1.0f) using LFSR113 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_lfsr113* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using LFSR113 generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_lfsr113* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using ThreeFry generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_threefry2x32_20* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using ThreeFry generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_threefry2x32_20* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using ThreeFry generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_threefry2x64_20* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using ThreeFry generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_threefry2x64_20* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using ThreeFry generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_threefry4x32_20* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using ThreeFry generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_threefry4x32_20* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>float</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p float value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using ThreeFry generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * \return Uniformly distributed \p float value from (0; 1] range.
 */
__forceinline__ __device__ __host__
float rocrand_uniform(rocrand_state_threefry4x64_20* state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Returns a uniformly distributed random <tt>double</tt> value
 * from (0; 1] range.
 *
 * Generates and returns a uniformly distributed \p double value from (0; 1] range
 * (excluding \p 0.0, including \p 1.0) using ThreeFry generator in \p state, and
 * increments position of the generator by one.
 *
 * \param state Pointer to a state to use
 *
 * Note: In this implementation returned \p double value is generated
 * from only 32 random bits (one <tt>unsigned int</tt> value).
 *
 * \return Uniformly distributed \p double value from (0; 1] range.
 */
__forceinline__ __device__ __host__
double rocrand_uniform_double(rocrand_state_threefry4x64_20* state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_UNIFORM_H_
