// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_LFSR113_H_
#define ROCRAND_LFSR113_H_

#include "rocrand/rocrand_lfsr113_precomputed.h"

#include <hip/hip_runtime.h>

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */
/// \def ROCRAND_LFSR113_DEFAULT_SEED_X
/// \brief Default X seed for LFSR113 PRNG.
#define ROCRAND_LFSR113_DEFAULT_SEED_X 2
/// \def ROCRAND_LFSR113_DEFAULT_SEED_Y
/// \brief Default Y seed for LFSR113 PRNG.
#define ROCRAND_LFSR113_DEFAULT_SEED_Y 8
/// \def ROCRAND_LFSR113_DEFAULT_SEED_Z
/// \brief Default Z seed for LFSR113 PRNG.
#define ROCRAND_LFSR113_DEFAULT_SEED_Z 16
/// \def ROCRAND_LFSR113_DEFAULT_SEED_W
/// \brief Default W seed for LFSR113 PRNG.
#define ROCRAND_LFSR113_DEFAULT_SEED_W 128
/** @} */ // end of group rocranddevice

namespace rocrand_device
{
namespace detail
{

__forceinline__ __device__ __host__ void mul_mat_vec_inplace(const unsigned int* m, uint4* z)
{
    unsigned int v[4]         = {z->x, z->y, z->z, z->w};
    unsigned int r[LFSR113_N] = {0};
    for(int ij = 0; ij < LFSR113_N * LFSR113_M; ij++)
    {
        const int          i = ij / LFSR113_M;
        const int          j = ij % LFSR113_M;
        const unsigned int b = (v[i] & (1U << j)) ? 0xffffffff : 0x0;
        for(int k = 0; k < LFSR113_N; k++)
        {
            r[k] ^= b & m[i * LFSR113_M * LFSR113_N + j * LFSR113_N + k];
        }
    }
    // Copy result into z
    z->x = r[0];
    z->y = r[1];
    z->z = r[2];
    z->w = r[3];
}
} // end namespace detail

class lfsr113_engine
{
public:
    struct lfsr113_state
    {
        uint4 z;
        uint4 subsequence;
    };

    /// Initializes the internal state of the PRNG using
    /// seed value \p seed, goes to \p subsequence -th subsequence,
    /// and skips \p offset random numbers.
    ///
    /// A subsequence is 2^55 numbers long.
    __forceinline__ __device__ __host__ lfsr113_engine(const uint4 seed
                                                       = {ROCRAND_LFSR113_DEFAULT_SEED_X,
                                                          ROCRAND_LFSR113_DEFAULT_SEED_Y,
                                                          ROCRAND_LFSR113_DEFAULT_SEED_Z,
                                                          ROCRAND_LFSR113_DEFAULT_SEED_W},
                                                       const unsigned int       subsequence = 0,
                                                       const unsigned long long offset      = 0)
    {
        this->seed(seed, subsequence, offset);
    }

    /// Reinitializes the internal state of the PRNG using new
    /// seed value \p seed_value, skips \p subsequence subsequences
    /// and skips \p offset random numbers.
    ///
    /// A subsequence is 2^55 numbers long.
    __forceinline__ __device__ __host__ void seed(uint4                    seed_value,
                                                  const unsigned long long subsequence,
                                                  const unsigned long long offset = 0)
    {
        m_state.subsequence = seed_value;

        reset_start_subsequence();
        discard_subsequence(subsequence);
        discard(offset);
    }

    /// Advances the internal state to skip one number.
    __forceinline__ __device__ __host__ void discard()
    {
        discard_state();
    }

    /// Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard(unsigned long long offset)
    {
#ifdef __HIP_DEVICE_COMPILE__
        jump(offset, d_lfsr113_jump_matrices);
#else
        jump(offset, h_lfsr113_jump_matrices);
#endif
    }

    /// Advances the internal state to skip \p subsequence subsequences.
    /// A subsequence is 2^55 numbers long.
    __forceinline__ __device__ __host__ void discard_subsequence(unsigned int subsequence)
    {
// Discard n * 2^55 samples
#ifdef __HIP_DEVICE_COMPILE__
        jump(subsequence, d_lfsr113_sequence_jump_matrices);
#else
        jump(subsequence, h_lfsr113_sequence_jump_matrices);
#endif
    }

    __forceinline__ __device__ __host__ unsigned int operator()()
    {
        return next();
    }

    __forceinline__ __device__ __host__ unsigned int next()
    {
        unsigned int b;

        b           = (((m_state.z.x << 6) ^ m_state.z.x) >> 13);
        m_state.z.x = (((m_state.z.x & 4294967294U) << 18) ^ b);

        b           = (((m_state.z.y << 2) ^ m_state.z.y) >> 27);
        m_state.z.y = (((m_state.z.y & 4294967288U) << 2) ^ b);

        b           = (((m_state.z.z << 13) ^ m_state.z.z) >> 21);
        m_state.z.z = (((m_state.z.z & 4294967280U) << 7) ^ b);

        b           = (((m_state.z.w << 3) ^ m_state.z.w) >> 12);
        m_state.z.w = (((m_state.z.w & 4294967168U) << 13) ^ b);

        return (m_state.z.x ^ m_state.z.y ^ m_state.z.z ^ m_state.z.w);
    }

protected:
    /// Resets the state to the start of the current subsequence.
    __forceinline__ __device__ __host__ void reset_start_subsequence()
    {
        m_state.z.x = m_state.subsequence.x;
        m_state.z.y = m_state.subsequence.y;
        m_state.z.z = m_state.subsequence.z;
        m_state.z.w = m_state.subsequence.w;
    }

    // Advances the internal state to the next state.
    __forceinline__ __device__ __host__ void discard_state()
    {
        this->next();
    }

    __forceinline__ __device__ __host__ void
        jump(unsigned long long v,
             const unsigned int (&jump_matrices)[LFSR113_JUMP_MATRICES][LFSR113_SIZE])
    {
        // x~(n + v) = (A^v mod m)x~n mod m
        // The matrix (A^v mod m) can be precomputed for selected values of v.
        //
        // For LFSR113_JUMP_LOG2 = 2
        // lfsr113_jump_matrices contains precomputed matrices:
        //   A^1, A^4, A^16...
        //
        // For LFSR113_JUMP_LOG2 = 2 and LFSR113_SEQUENCE_JUMP_LOG2 = 55
        // lfsr113_sequence_jump_matrices contains precomputed matrices:
        //   A^(1 * 2^55), A^(4 * 2^55), A^(16 * 2^55)...
        //
        // Intermediate powers can be calculated as multiplication of the powers above.

        unsigned int mi = 0;
        while(v > 0)
        {
            const unsigned int is = static_cast<unsigned int>(v) & ((1 << LFSR113_JUMP_LOG2) - 1);
            for(unsigned int i = 0; i < is; i++)
            {
                detail::mul_mat_vec_inplace(jump_matrices[mi], &m_state.z);
            }
            mi++;
            v >>= LFSR113_JUMP_LOG2;
        }
    }

protected:
    lfsr113_state m_state;

}; // lfsr113_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::lfsr113_engine rocrand_state_lfsr113;
/// \endcond

/**
 * \brief Initializes LFSR113 state.
 *
 * Initializes the LFSR113 generator \p state with the given
 * \p seed, \p subsequence, and \p offset.
 *
 * \param seed Value to use as a seed
 * \param subsequence Subsequence to start at
 * \param state Pointer to state to initialize
 */
__forceinline__ __device__ __host__
void rocrand_init(const uint4 seed, const unsigned int subsequence, rocrand_state_lfsr113* state)
{
    *state = rocrand_state_lfsr113(seed, subsequence);
}

/**
 * \brief Initializes LFSR113 state.
 *
 * Initializes the LFSR113 generator \p state with the given
 * \p seed, \p subsequence, and \p offset.
 *
 * \param seed Value to use as a seed
 * \param subsequence Subsequence to start at
 * \param offset Absolute offset into subsequence
 * \param state Pointer to state to initialize
 */
__forceinline__ __device__ __host__
void rocrand_init(const uint4              seed,
                  const unsigned int       subsequence,
                  const unsigned long long offset,
                  rocrand_state_lfsr113*   state)
{
    *state = rocrand_state_lfsr113(seed, subsequence, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned int</tt> value
 * from [0; 2^32 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned int</tt>
 * value from [0; 2^32 - 1] range using LFSR113 generator in \p state.
 * State is incremented by one position.
 *
 * \param state Pointer to a state to use
 *
 * \return Pseudorandom value (32-bit) as an <tt>unsigned int</tt>
 */
__forceinline__ __device__ __host__
unsigned int rocrand(rocrand_state_lfsr113* state)
{
    return state->next();
}

/**
 * \brief Updates LFSR113 state to skip ahead by \p offset elements.
 *
 * Updates the LFSR113 state in \p state to skip ahead by \p offset elements.
 *
 * \param offset Number of elements to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead(unsigned long long offset, rocrand_state_lfsr113* state)
{
    return state->discard(offset);
}

/**
 * \brief Updates LFSR113 state to skip ahead by \p subsequence subsequences.
 *
 * Updates the LFSR113 \p state to skip ahead by \p subsequence subsequences.
 * Each subsequence is 2^55 numbers long.
 *
 * \param subsequence Number of subsequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_subsequence(unsigned int subsequence, rocrand_state_lfsr113* state)
{
    return state->discard_subsequence(subsequence);
}

/**
 * \brief Updates LFSR113 state to skip ahead by \p sequence sequences.
 *
 * Updates the LFSR113 \p state to skip ahead by \p sequence sequences.
 * For LFSR113 each sequence is 2^55 numbers long (equal to the size of a subsequence).
 *
 * \param sequence Number of sequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_sequence(unsigned int sequence, rocrand_state_lfsr113* state)
{
    return state->discard_subsequence(sequence);
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_LFSR113_H_
