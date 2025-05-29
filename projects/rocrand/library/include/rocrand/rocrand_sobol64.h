// Copyright (c) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_SOBOL64_H_
#define ROCRAND_SOBOL64_H_

#include <hip/hip_runtime.h>

namespace rocrand_device {

template<bool UseSharedVectors>
struct sobol64_state
{
    unsigned long long int d;
    unsigned long long int i;
    unsigned long long int vectors[64];

    __forceinline__ __device__ __host__ sobol64_state() : d(), i(), vectors() {}

    __forceinline__ __device__ __host__ sobol64_state(const unsigned long long int  d,
                                                      const unsigned long long int  i,
                                                      const unsigned long long int* vectors)
        : d(d), i(i)
    {
        for(int k = 0; k < 64; k++)
        {
            this->vectors[k] = vectors[k];
        }
    }
};

template<>
struct sobol64_state<true>
{
    unsigned long long int d;
    unsigned long long int i;
    const unsigned long long int * vectors;

    __forceinline__ __device__ __host__ sobol64_state() : d(), i(), vectors() {}

    __forceinline__ __device__ __host__ sobol64_state(const unsigned long long int  d,
                                                      const unsigned long long int  i,
                                                      const unsigned long long int* vectors)
        : d(d), i(i), vectors(vectors)
    {}
};

template<bool UseSharedVectors>
class sobol64_engine
{
public:

    typedef struct sobol64_state<UseSharedVectors> sobol64_state;

    __forceinline__ __device__ __host__ sobol64_engine() {}

    __forceinline__ __device__ __host__ sobol64_engine(const unsigned long long int* vectors,
                                                       const unsigned long long int  offset)
        : m_state(0, 0, vectors)
    {
        discard_state(offset);
    }

    /// Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard(unsigned long long int offset)
    {
        discard_state(offset);
    }

    __forceinline__ __device__ __host__ void discard()
    {
        discard_state();
    }

    /// Advances the internal state by stride times, where stride is power of 2
    __forceinline__ __device__ __host__ void discard_stride(unsigned long long int stride)
    {
        discard_state_power2(stride);
    }

    __forceinline__ __device__ __host__ unsigned long long int operator()()
    {
        return this->next();
    }

    __forceinline__ __device__ __host__ unsigned long long int next()
    {
        unsigned long long int p = m_state.d;
        discard_state();
        return p;
    }

    __forceinline__ __device__ __host__ unsigned long long int current() const
    {
        return m_state.d;
    }

    __forceinline__ __device__ __host__ static constexpr bool uses_shared_vectors()
    {
        return UseSharedVectors;
    }

protected:
    // Advances the internal state by offset times.
    __forceinline__ __device__ __host__ void discard_state(unsigned long long int offset)
    {
        m_state.i += offset;
        const unsigned long long int g = m_state.i ^ (m_state.i >> 1ull);
        m_state.d = 0;
        for(int i = 0; i < 64; i++)
        {
            m_state.d ^= (g & (1ull << i) ? m_state.vectors[i] : 0ull);
        }
    }

    // Advances the internal state to the next state
    __forceinline__ __device__ __host__ void discard_state()
    {
        m_state.d ^= m_state.vectors[rightmost_zero_bit(m_state.i)];
        m_state.i++;
    }

    __forceinline__ __device__ __host__ void discard_state_power2(unsigned long long int stride)
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
    // NOTE changing unsigned long long int to unit64_t will cause compile failure on device
    __forceinline__ __device__ __host__ unsigned int rightmost_zero_bit(unsigned long long int x)
    {
        #if defined(__HIP_DEVICE_COMPILE__)
        unsigned int z = __ffsll(~x);
        return z ? z - 1 : 0;
        #else
        if(x == 0)
            return 0;
        unsigned long long int y = x;
        unsigned long long int z = 1;
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
    sobol64_state m_state;

}; // sobol64_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::sobol64_engine<false> rocrand_state_sobol64;
/// \endcond

/**
 * \brief Initialize sobol64 state.
 *
 * Initializes the sobol64 generator \p state with the given
 * direction \p vectors and \p offset.
 *
 * \param vectors Direction vectors
 * \param offset Absolute offset into sequence
 * \param state Pointer to state to initialize
 */
__forceinline__ __device__ __host__
void rocrand_init(const unsigned long long int* vectors,
                  const unsigned int            offset,
                  rocrand_state_sobol64*        state)
{
    *state = rocrand_state_sobol64(vectors, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned long long int</tt> value
 * from [0; 2^64 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned long long int</tt>
 * value from [0; 2^64 - 1] range using sobol64 generator in \p state.
 * State is incremented by one position.
 *
 * \param state Pointer to a state to use
 *
 * \return Quasirandom value (64-bit) as an <tt>unsigned long long int</tt>
 */
__forceinline__ __device__ __host__
unsigned long long int rocrand(rocrand_state_sobol64* state)
{
    return state->next();
}

/**
 * \brief Updates sobol64 state to skip ahead by \p offset elements.
 *
 * Updates the sobol64 state in \p state to skip ahead by \p offset elements.
 *
 * \param offset Number of elements to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead(unsigned long long int offset, rocrand_state_sobol64* state)
{
    return state->discard(offset);
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_sobol64_H_
