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

#ifndef ROCRAND_MRG31K3P_H_
#define ROCRAND_MRG31K3P_H_

#include "rocrand/rocrand_common.h"
#include "rocrand/rocrand_mrg31k3p_precomputed.h"

#include <hip/hip_runtime.h>

#define ROCRAND_MRG31K3P_M1 2147483647U // 2 ^ 31 - 1
#define ROCRAND_MRG31K3P_M2 2147462579U // 2 ^ 31 - 21069
#define ROCRAND_MRG31K3P_MASK12 511U // 2 ^ 9 - 1
#define ROCRAND_MRG31K3P_MASK13 16777215U // 2 ^ 24 - 1
#define ROCRAND_MRG31K3P_MASK21 65535U // 2 ^ 16 - 1
#define ROCRAND_MRG31K3P_NORM_DOUBLE (4.656612875245796923e-10) // 1 / ROCRAND_MRG31K3P_M1
#define ROCRAND_MRG31K3P_UINT32_NORM \
    (2.000000001396983862) // UINT32_MAX / (ROCRAND_MRG31K3P_M1 - 1)

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */
/**
 * \def ROCRAND_MRG31K3P_DEFAULT_SEED
 * \brief Default seed for MRG31K3P PRNG.
 */
#define ROCRAND_MRG31K3P_DEFAULT_SEED 12345ULL
/** @} */ // end of group rocranddevice

namespace rocrand_device
{

class mrg31k3p_engine
{
public:
    struct mrg31k3p_state
    {
        unsigned int x1[3];
        unsigned int x2[3];

#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        // The Box–Muller transform requires two inputs to convert uniformly
        // distributed real values [0; 1] to normally distributed real values
        // (with mean = 0, and stddev = 1). Often user wants only one
        // normally distributed number, to save performance and random
        // numbers the 2nd value is saved for future requests.
        unsigned int boxmuller_float_state; // is there a float in boxmuller_float
        unsigned int boxmuller_double_state; // is there a double in boxmuller_double
        float        boxmuller_float; // normally distributed float
        double       boxmuller_double; // normally distributed double
#endif
    };

    __forceinline__ __device__ __host__ mrg31k3p_engine()
    {
        this->seed(ROCRAND_MRG31K3P_DEFAULT_SEED, 0, 0);
    }

    /// Initializes the internal state of the PRNG using
    /// seed value \p seed, goes to \p subsequence -th subsequence,
    /// and skips \p offset random numbers.
    ///
    /// New seed value should not be zero. If \p seed_value is equal
    /// zero, value \p ROCRAND_MRG31K3P_DEFAULT_SEED is used instead.
    ///
    /// A subsequence is 2^72 numbers long.
    __forceinline__ __device__ __host__ mrg31k3p_engine(const unsigned long long seed,
                                                        const unsigned long long subsequence,
                                                        const unsigned long long offset)
    {
        this->seed(seed, subsequence, offset);
    }

    /// Reinitializes the internal state of the PRNG using new
    /// seed value \p seed_value, skips \p subsequence subsequences
    /// and \p offset random numbers.
    ///
    /// New seed value should not be zero. If \p seed_value is equal
    /// zero, value \p ROCRAND_MRG31K3P_DEFAULT_SEED is used instead.
    ///
    /// A subsequence is 2^72 numbers long.
    __forceinline__ __device__ __host__ void seed(unsigned long long       seed_value,
                                                  const unsigned long long subsequence,
                                                  const unsigned long long offset)
    {
        if(seed_value == 0)
        {
            seed_value = ROCRAND_MRG31K3P_DEFAULT_SEED;
        }
        unsigned int x = static_cast<unsigned int>(seed_value ^ 0x55555555U);
        unsigned int y = static_cast<unsigned int>((seed_value >> 32) ^ 0xAAAAAAAAU);
        m_state.x1[0]  = mod_mul_m1(x, seed_value);
        m_state.x1[1]  = mod_mul_m1(y, seed_value);
        m_state.x1[2]  = mod_mul_m1(x, seed_value);
        m_state.x2[0]  = mod_mul_m2(y, seed_value);
        m_state.x2[1]  = mod_mul_m2(x, seed_value);
        m_state.x2[2]  = mod_mul_m2(y, seed_value);
        this->restart(subsequence, offset);
    }

    /// Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard(unsigned long long offset)
    {
        this->discard_impl(offset);
    }

    /// Advances the internal state to skip \p subsequence subsequences.
    /// A subsequence is 2^72 numbers long.
    __forceinline__ __device__ __host__ void discard_subsequence(unsigned long long subsequence)
    {
        this->discard_subsequence_impl(subsequence);
    }

    /// Advances the internal state to skip \p sequence sequences.
    /// A sequence is 2^134 numbers long.
    __forceinline__ __device__ __host__ void discard_sequence(unsigned long long sequence)
    {
        this->discard_sequence_impl(sequence);
    }

    __forceinline__ __device__ __host__ void restart(const unsigned long long subsequence,
                                                     const unsigned long long offset)
    {
#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        m_state.boxmuller_float_state  = 0;
        m_state.boxmuller_double_state = 0;
#endif
        this->discard_subsequence_impl(subsequence);
        this->discard_impl(offset);
    }

    __forceinline__ __device__ __host__ unsigned int operator()()
    {
        return this->next();
    }

    // Returned value is in range [1, ROCRAND_MRG31K3P_M1].
    __forceinline__ __device__ __host__ unsigned int next()
    {
        // First component
        unsigned int tmp
            = (((m_state.x1[1] & ROCRAND_MRG31K3P_MASK12) << 22) + (m_state.x1[1] >> 9))
              + (((m_state.x1[2] & ROCRAND_MRG31K3P_MASK13) << 7) + (m_state.x1[2] >> 24));
        tmp -= (tmp >= ROCRAND_MRG31K3P_M1) ? ROCRAND_MRG31K3P_M1 : 0;
        tmp += m_state.x1[2];
        tmp -= (tmp >= ROCRAND_MRG31K3P_M1) ? ROCRAND_MRG31K3P_M1 : 0;
        m_state.x1[2] = m_state.x1[1];
        m_state.x1[1] = m_state.x1[0];
        m_state.x1[0] = tmp;

        // Second component
        tmp = (((m_state.x2[0] & ROCRAND_MRG31K3P_MASK21) << 15) + 21069 * (m_state.x2[0] >> 16));
        tmp -= (tmp >= ROCRAND_MRG31K3P_M2) ? ROCRAND_MRG31K3P_M2 : 0;
        tmp += ((m_state.x2[2] & ROCRAND_MRG31K3P_MASK21) << 15);
        tmp -= (tmp >= ROCRAND_MRG31K3P_M2) ? ROCRAND_MRG31K3P_M2 : 0;
        tmp += 21069 * (m_state.x2[2] >> 16);
        tmp -= (tmp >= ROCRAND_MRG31K3P_M2) ? ROCRAND_MRG31K3P_M2 : 0;
        tmp += m_state.x2[2];
        tmp -= (tmp >= ROCRAND_MRG31K3P_M2) ? ROCRAND_MRG31K3P_M2 : 0;
        m_state.x2[2] = m_state.x2[1];
        m_state.x2[1] = m_state.x2[0];
        m_state.x2[0] = tmp;

        // Combination
        return m_state.x1[0] - m_state.x2[0]
               + (m_state.x1[0] <= m_state.x2[0] ? ROCRAND_MRG31K3P_M1 : 0);
    }

protected:
    // Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard_impl(unsigned long long offset)
    {
        discard_state(offset);
    }

    // Advances the internal state to skip \p subsequence subsequences.
    __forceinline__ __device__ __host__ void
        discard_subsequence_impl(unsigned long long subsequence)
    {
        int i = 0;

        while(subsequence > 0)
        {
            if(subsequence & 1)
            {
#if defined(__HIP_DEVICE_COMPILE__)
                mod_mat_vec_m1(d_mrg31k3p_A1P72 + i, m_state.x1);
                mod_mat_vec_m2(d_mrg31k3p_A2P72 + i, m_state.x2);
#else
                mod_mat_vec_m1(h_mrg31k3p_A1P72 + i, m_state.x1);
                mod_mat_vec_m2(h_mrg31k3p_A2P72 + i, m_state.x2);
#endif
            }
            subsequence >>= 1;
            i += 9;
        }
    }

    // Advances the internal state to skip \p sequences.
    __forceinline__ __device__ __host__ void discard_sequence_impl(unsigned long long sequence)
    {
        int i = 0;

        while(sequence > 0)
        {
            if(sequence & 1)
            {
#if defined(__HIP_DEVICE_COMPILE__)
                mod_mat_vec_m1(d_mrg31k3p_A1P134 + i, m_state.x1);
                mod_mat_vec_m2(d_mrg31k3p_A2P134 + i, m_state.x2);
#else
                mod_mat_vec_m1(h_mrg31k3p_A1P134 + i, m_state.x1);
                mod_mat_vec_m2(h_mrg31k3p_A2P134 + i, m_state.x2);
#endif
            }
            sequence >>= 1;
            i += 9;
        }
    }

    // Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard_state(unsigned long long offset)
    {
        int i = 0;

        while(offset > 0)
        {
            if(offset & 1)
            {
#if defined(__HIP_DEVICE_COMPILE__)
                mod_mat_vec_m1(d_mrg31k3p_A1 + i, m_state.x1);
                mod_mat_vec_m2(d_mrg31k3p_A2 + i, m_state.x2);
#else
                mod_mat_vec_m1(h_mrg31k3p_A1 + i, m_state.x1);
                mod_mat_vec_m2(h_mrg31k3p_A2 + i, m_state.x2);
#endif
            }
            offset >>= 1;
            i += 9;
        }
    }

    // Advances the internal state to the next state.
    __forceinline__ __device__ __host__ void discard_state()
    {
        discard_state(1);
    }

private:
    __forceinline__ __device__ __host__ static void mod_mat_vec_m1(const unsigned int* A,
                                                                   unsigned int*       s)
    {
        unsigned long long x[3] = {s[0], s[1], s[2]};

        s[0] = mod_m1(mod_m1(A[0] * x[0]) + mod_m1(A[1] * x[1]) + mod_m1(A[2] * x[2]));

        s[1] = mod_m1(mod_m1(A[3] * x[0]) + mod_m1(A[4] * x[1]) + mod_m1(A[5] * x[2]));

        s[2] = mod_m1(mod_m1(A[6] * x[0]) + mod_m1(A[7] * x[1]) + mod_m1(A[8] * x[2]));
    }

    __forceinline__ __device__ __host__ static void mod_mat_vec_m2(const unsigned int* A,
                                                                   unsigned int*       s)
    {
        unsigned long long x[3] = {s[0], s[1], s[2]};

        s[0] = mod_m2(mod_m2(A[0] * x[0]) + mod_m2(A[1] * x[1]) + mod_m2(A[2] * x[2]));

        s[1] = mod_m2(mod_m2(A[3] * x[0]) + mod_m2(A[4] * x[1]) + mod_m2(A[5] * x[2]));

        s[2] = mod_m2(mod_m2(A[6] * x[0]) + mod_m2(A[7] * x[1]) + mod_m2(A[8] * x[2]));
    }

    __forceinline__ __device__ __host__ static unsigned long long mod_mul_m1(unsigned int       i,
                                                                             unsigned long long j)
    {
        return mod_m1(i * j);
    }

    __forceinline__ __device__ __host__ static unsigned long long mod_m1(unsigned long long p)
    {
        return p % ROCRAND_MRG31K3P_M1;
    }

    __forceinline__ __device__ __host__ static unsigned long long mod_mul_m2(unsigned int       i,
                                                                             unsigned long long j)
    {
        return mod_m2(i * j);
    }

    __forceinline__ __device__ __host__ static unsigned long long mod_m2(unsigned long long p)
    {
        return p % ROCRAND_MRG31K3P_M2;
    }

protected:
    // State
    mrg31k3p_state m_state;

#ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
    friend struct detail::engine_boxmuller_helper<mrg31k3p_engine>;
#endif
}; // mrg31k3p_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::mrg31k3p_engine rocrand_state_mrg31k3p;
/// \endcond

/**
 * \brief Initializes MRG31K3P state.
 *
 * Initializes the MRG31K3P generator \p state with the given
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
                  rocrand_state_mrg31k3p*  state)
{
    *state = rocrand_state_mrg31k3p(seed, subsequence, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned int</tt> value
 * from [0; 2^32 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned int</tt>
 * value from [0; 2^32 - 1] range using MRG31K3P generator in \p state.
 * State is incremented by one position.
 *
 * \param state Pointer to a state to use
 *
 * \return Pseudorandom value (32-bit) as an <tt>unsigned int</tt>
 */
__forceinline__ __device__ __host__
unsigned int rocrand(rocrand_state_mrg31k3p* state)
{
    // next() in [1, ROCRAND_MRG31K3P_M1]
    return static_cast<unsigned int>((state->next() - 1) * ROCRAND_MRG31K3P_UINT32_NORM);
}

/**
 * \brief Updates MRG31K3P state to skip ahead by \p offset elements.
 *
 * Updates the MRG31K3P state in \p state to skip ahead by \p offset elements.
 *
 * \param offset Number of elements to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead(unsigned long long offset, rocrand_state_mrg31k3p* state)
{
    return state->discard(offset);
}

/**
 * \brief Updates MRG31K3P state to skip ahead by \p subsequence subsequences.
 *
 * Updates the MRG31K3P state in \p state to skip ahead by \p subsequence subsequences.
 * Each subsequence is 2^72 numbers long.
 *
 * \param subsequence Number of subsequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_subsequence(unsigned long long subsequence, rocrand_state_mrg31k3p* state)
{
    return state->discard_subsequence(subsequence);
}

/**
 * \brief Updates MRG31K3P state to skip ahead by \p sequence sequences.
 *
 * Updates the MRG31K3P state in \p state to skip ahead by \p sequence sequences.
 * Each sequence is 2^134 numbers long.
 *
 * \param sequence Number of sequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_sequence(unsigned long long sequence, rocrand_state_mrg31k3p* state)
{
    return state->discard_sequence(sequence);
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_MRG31K3P_H_
