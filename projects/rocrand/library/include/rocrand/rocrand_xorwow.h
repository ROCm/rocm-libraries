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

#ifndef ROCRAND_XORWOW_H_
#define ROCRAND_XORWOW_H_

#include "rocrand/rocrand_common.h"
#include "rocrand/rocrand_xorwow_precomputed.h"

#include <hip/hip_runtime.h>

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */
 /**
 * \def ROCRAND_XORWOW_DEFAULT_SEED
 * \brief Default seed for XORWOW PRNG.
 */
 #define ROCRAND_XORWOW_DEFAULT_SEED 0ULL
 /** @} */ // end of group rocranddevice

namespace rocrand_device {
namespace detail {

__forceinline__ __device__ __host__ void copy_vec(unsigned int* dst, const unsigned int* src)
{
    for (int i = 0; i < XORWOW_N; i++)
    {
        dst[i] = src[i];
    }
}

__forceinline__ __device__ __host__ void mul_mat_vec_inplace(const unsigned int* m, unsigned int* v)
{
    unsigned int r[XORWOW_N] = { 0 };
    for (int ij = 0; ij < XORWOW_N * XORWOW_M; ij++)
    {
        const int i = ij / XORWOW_M;
        const int j = ij % XORWOW_M;
        const unsigned int b = (v[i] & (1U << j)) ? 0xffffffff : 0x0;
        for (int k = 0; k < XORWOW_N; k++)
        {
            r[k] ^= b & m[i * XORWOW_M * XORWOW_N + j * XORWOW_N + k];
        }
    }
    copy_vec(v, r);
}

} // end detail namespace

class xorwow_engine
{
public:
    struct xorwow_state
    {
        // Weyl sequence value
        unsigned int d;

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        // The Box–Muller transform requires two inputs to convert uniformly
        // distributed real values [0; 1] to normally distributed real values
        // (with mean = 0, and stddev = 1). Often user wants only one
        // normally distributed number, to save performance and random
        // numbers the 2nd value is saved for future requests.
        unsigned int boxmuller_float_state; // is there a float in boxmuller_float
        unsigned int boxmuller_double_state; // is there a double in boxmuller_double
        float boxmuller_float; // normally distributed float
        double boxmuller_double; // normally distributed double
    #endif

        // Xorshift values (160 bits)
        unsigned int x[5];
    };

    __forceinline__ __device__ __host__ xorwow_engine()
        : xorwow_engine(ROCRAND_XORWOW_DEFAULT_SEED, 0, 0)
    {}

    /// Initializes the internal state of the PRNG using
    /// seed value \p seed, goes to \p subsequence -th subsequence,
    /// and skips \p offset random numbers.
    ///
    /// A subsequence is 2^67 numbers long.
    __forceinline__ __device__ __host__ xorwow_engine(const unsigned long long seed,
                                                      const unsigned long long subsequence,
                                                      const unsigned long long offset)
    {
        m_state.x[0] = 123456789U;
        m_state.x[1] = 362436069U;
        m_state.x[2] = 521288629U;
        m_state.x[3] = 88675123U;
        m_state.x[4] = 5783321U;

        m_state.d = 6615241U;

        // Constants are arbitrary prime numbers
        const unsigned int s0 = static_cast<unsigned int>(seed) ^ 0x2c7f967fU;
        const unsigned int s1 = static_cast<unsigned int>(seed >> 32) ^ 0xa03697cbU;
        const unsigned int t0 = 1228688033U * s0;
        const unsigned int t1 = 2073658381U * s1;
        m_state.x[0] += t0;
        m_state.x[1] ^= t0;
        m_state.x[2] += t1;
        m_state.x[3] ^= t1;
        m_state.x[4] += t0;
        m_state.d += t1 + t0;

        discard_subsequence(subsequence);
        discard(offset);

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        m_state.boxmuller_float_state = 0;
        m_state.boxmuller_double_state = 0;
    #endif
    }

    /// Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard(unsigned long long offset)
    {
        #ifdef __HIP_DEVICE_COMPILE__
        jump(offset, d_xorwow_jump_matrices);
        #else
        jump(offset, h_xorwow_jump_matrices);
        #endif

        // Apply n steps to Weyl sequence value as well
        m_state.d += static_cast<unsigned int>(offset) * 362437;
    }

    /// Advances the internal state to skip \p subsequence subsequences.
    /// A subsequence is 2^67 numbers long.
    __forceinline__ __device__ __host__ void discard_subsequence(unsigned long long subsequence)
    {
        // Discard n * 2^67 samples
        #ifdef __HIP_DEVICE_COMPILE__
        jump(subsequence, d_xorwow_sequence_jump_matrices);
        #else
        jump(subsequence, h_xorwow_sequence_jump_matrices);
        #endif

        // d has the same value because 2^67 is divisible by 2^32 (d is 32-bit)
    }

    __forceinline__ __device__ __host__ unsigned int operator()()
    {
        return next();
    }

    __forceinline__ __device__ __host__ unsigned int next()
    {
        const unsigned int t = m_state.x[0] ^ (m_state.x[0] >> 2);
        m_state.x[0] = m_state.x[1];
        m_state.x[1] = m_state.x[2];
        m_state.x[2] = m_state.x[3];
        m_state.x[3] = m_state.x[4];
        m_state.x[4] = (m_state.x[4] ^ (m_state.x[4] << 4)) ^ (t ^ (t << 1));

        m_state.d += 362437;

        return m_state.d + m_state.x[4];
    }

protected:
    __forceinline__ __device__ __host__ void
        jump(unsigned long long v,
             const unsigned int jump_matrices[XORWOW_JUMP_MATRICES][XORWOW_SIZE])
    {
        // x~(n + v) = (A^v mod m)x~n mod m
        // The matrix (A^v mod m) can be precomputed for selected values of v.
        //
        // For XORWOW_JUMP_LOG2 = 2
        // xorwow_jump_matrices contains precomputed matrices:
        //   A^1, A^4, A^16...
        //
        // For XORWOW_JUMP_LOG2 = 2 and XORWOW_SEQUENCE_JUMP_LOG2 = 67
        // xorwow_sequence_jump_matrices contains precomputed matrices:
        //   A^(1 * 2^67), A^(4 * 2^67), A^(16 * 2^67)...
        //
        // Intermediate powers can be calculated as multiplication of the powers above.

        unsigned int mi = 0;
        while (v > 0)
        {
            const unsigned int is = static_cast<unsigned int>(v) & ((1 << XORWOW_JUMP_LOG2) - 1);
            for (unsigned int i = 0; i < is; i++)
            {
                detail::mul_mat_vec_inplace(jump_matrices[mi], m_state.x);
            }
            mi++;
            v >>= XORWOW_JUMP_LOG2;
        }
    }

protected:
    // State
    xorwow_state m_state;

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
    friend struct detail::engine_boxmuller_helper<xorwow_engine>;
    #endif

}; // xorwow_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::xorwow_engine rocrand_state_xorwow;
/// \endcond

/**
 * \brief Initialize XORWOW state.
 *
 * Initializes the XORWOW generator \p state with the given
 * \p seed, \p subsequence, and \p offset.
 *
 * \param seed Value to use as a seed
 * \param subsequence Subsequence to start at
 * \param offset Absolute offset into subsequence
 * \param state Pointer to state to initialize
 */
__forceinline__ __device__ __host__
void rocrand_init(const unsigned long long seed,
                  const unsigned long long subsequence,
                  const unsigned long long offset,
                  rocrand_state_xorwow*    state)
{
    *state = rocrand_state_xorwow(seed, subsequence, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned int</tt> value
 * from [0; 2^32 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned int</tt>
 * value from [0; 2^32 - 1] range using XORWOW generator in \p state.
 * State is incremented by one position.
 *
 * \param state Pointer to a state to use
 *
 * \return Pseudorandom value (32-bit) as an <tt>unsigned int</tt>
 */
__forceinline__ __device__ __host__
unsigned int rocrand(rocrand_state_xorwow* state)
{
    return state->next();
}

/**
 * \brief Updates XORWOW state to skip ahead by \p offset elements.
 *
 * Updates the XORWOW state in \p state to skip ahead by \p offset elements.
 *
 * \param offset Number of elements to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead(unsigned long long offset, rocrand_state_xorwow* state)
{
    return state->discard(offset);
}

/**
 * \brief Updates XORWOW state to skip ahead by \p subsequence subsequences.
 *
 * Updates the XORWOW \p state to skip ahead by \p subsequence subsequences.
 * Each subsequence is 2^67 numbers long.
 *
 * \param subsequence Number of subsequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_subsequence(unsigned long long subsequence, rocrand_state_xorwow* state)
{
    return state->discard_subsequence(subsequence);
}

/**
 * \brief Updates XORWOW state to skip ahead by \p sequence sequences.
 *
 * Updates the XORWOW \p state skipping \p sequence sequences ahead.
 * For XORWOW each sequence is 2^67 numbers long (equal to the size of a subsequence).
 *
 * \param sequence Number of sequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_sequence(unsigned long long sequence, rocrand_state_xorwow* state)
{
    return state->discard_subsequence(sequence);
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_XORWOW_H_
