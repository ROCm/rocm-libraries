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

#ifndef ROCRAND_SOBOL32_H_
#define ROCRAND_SOBOL32_H_

#include <hip/hip_runtime.h>

namespace rocrand_device {

template<bool UseSharedVectors>
struct sobol32_state
{
    unsigned int d;
    unsigned int i;
    unsigned int vectors[32];

    __forceinline__ __device__ __host__ sobol32_state() : d(), i(), vectors() {}

    __forceinline__ __device__ __host__ sobol32_state(const unsigned int  d,
                                                      const unsigned int  i,
                                                      const unsigned int* vectors)
        : d(d), i(i)
    {
        for(int k = 0; k < 32; k++)
        {
            this->vectors[k] = vectors[k];
        }
    }
};

template<>
struct sobol32_state<true>
{
    unsigned int d;
    unsigned int i;
    const unsigned int * vectors;

    __forceinline__ __device__ __host__ sobol32_state() : d(), i(), vectors() {}

    __forceinline__ __device__ __host__ sobol32_state(const unsigned int  d,
                                                      const unsigned int  i,
                                                      const unsigned int* vectors)
        : d(d), i(i), vectors(vectors)
    {}
};

template<bool UseSharedVectors>
class sobol32_engine
{
public:

    typedef struct sobol32_state<UseSharedVectors> sobol32_state;

    __forceinline__ __device__ __host__ sobol32_engine() {}

    __forceinline__ __device__ __host__ sobol32_engine(const unsigned int* vectors,
                                                       const unsigned int  offset)
        : m_state(0, 0, vectors)
    {
        discard_state(offset);
    }

    /// Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard(unsigned int offset)
    {
        discard_state(offset);
    }

    __forceinline__ __device__ __host__ void discard()
    {
        discard_state();
    }

    /// Advances the internal state by stride times, where stride is power of 2
    __forceinline__ __device__ __host__ void discard_stride(unsigned int stride)
    {
        discard_state_power2(stride);
    }

    __forceinline__ __device__ __host__ unsigned int operator()()
    {
        return this->next();
    }

    __forceinline__ __device__ __host__ unsigned int next()
    {
        unsigned int p = m_state.d;
        discard_state();
        return p;
    }

    __forceinline__ __device__ __host__ unsigned int current() const
    {
        return m_state.d;
    }

    __forceinline__ __device__ __host__ static constexpr bool uses_shared_vectors()
    {
        return UseSharedVectors;
    }

protected:
    // Advances the internal state by offset times.
    __forceinline__ __device__ __host__ void discard_state(unsigned int offset)
    {
        m_state.i += offset;
        const unsigned int g = m_state.i ^ (m_state.i >> 1);
        m_state.d = 0;
        for(int i = 0; i < 32; i++)
        {
            m_state.d ^= (g & (1U << i) ? m_state.vectors[i] : 0);
        }
    }

    // Advances the internal state to the next state
    __forceinline__ __device__ __host__ void discard_state()
    {
        m_state.d ^= m_state.vectors[rightmost_zero_bit(m_state.i)];
        m_state.i++;
    }

    __forceinline__ __device__ __host__ void discard_state_power2(unsigned int stride)
    {
        // Leap frog
        //
        // T Bradley, J Toit, M Giles, R Tong, P Woodhams
        // Parallelisation Techniques for Random Number Generators
        // GPU Computing Gems, 2011
        //
        // For power of 2 jumps only 2 bits in Gray code change values
        // All bits lower than log2(stride) flip 2, 4... times, i.e.
        // do not change their values.

        // log2(stride) bit
        m_state.d ^= m_state.vectors[rightmost_zero_bit(~stride) - 1];
        // the rightmost zero bit of i, not including the lower log2(stride) bits
        m_state.d ^= m_state.vectors[rightmost_zero_bit(m_state.i | (stride - 1))];
        m_state.i += stride;
    }

    // Returns the index of the rightmost zero bit in the binary expansion of
    // x (Gray code of the current element's index)
    __forceinline__ __device__ __host__ unsigned int rightmost_zero_bit(unsigned int x)
    {
        #if defined(__HIP_DEVICE_COMPILE__)
        unsigned int z = __ffs(~x);
        return z ? z - 1 : 0;
        #else
        if(x == 0)
            return 0;
        unsigned int y = x;
        unsigned int z = 1;
        while(y & 1)
        {
            y >>= 1;
            z++;
        }
        return z - 1;
        #endif
    }

protected:
    // State
    sobol32_state m_state;

}; // sobol32_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::sobol32_engine<false> rocrand_state_sobol32;
/// \endcond

/**
 * \brief Initialize SOBOL32 state.
 *
 * Initializes the SOBOL32 generator \p state with the given
 * direction \p vectors and \p offset.
 *
 * \param vectors Direction vectors
 * \param offset Absolute offset into sequence
 * \param state Pointer to state to initialize
 */
__forceinline__ __device__ __host__
void rocrand_init(const unsigned int*    vectors,
                  const unsigned int     offset,
                  rocrand_state_sobol32* state)
{
    *state = rocrand_state_sobol32(vectors, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned int</tt> value
 * from [0; 2^32 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned int</tt>
 * value from [0; 2^32 - 1] range using Sobol32 generator in \p state.
 * State is incremented by one position.
 *
 * \param state Pointer to a state to use
 *
 * \return Quasirandom value (32-bit) as an <tt>unsigned int</tt>
 */
__forceinline__ __device__ __host__
unsigned int rocrand(rocrand_state_sobol32* state)
{
    return state->next();
}

/**
 * \brief Updates SOBOL32 state to skip ahead by \p offset elements.
 *
 * Updates the SOBOL32 state in \p state to skip ahead by \p offset elements.
 *
 * \param offset Number of elements to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead(unsigned long long offset, rocrand_state_sobol32* state)
{
    return state->discard(offset);
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_SOBOL32_H_
