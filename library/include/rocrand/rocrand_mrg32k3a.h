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

#ifndef ROCRAND_MRG32K3A_H_
#define ROCRAND_MRG32K3A_H_

#include "rocrand/rocrand_common.h"
#include "rocrand/rocrand_mrg32k3a_precomputed.h"

#include <hip/hip_runtime.h>

#define ROCRAND_MRG32K3A_POW32 4294967296U
#define ROCRAND_MRG32K3A_M1 4294967087U
#define ROCRAND_MRG32K3A_M1C 209U
#define ROCRAND_MRG32K3A_M2 4294944443U
#define ROCRAND_MRG32K3A_M2C 22853U
#define ROCRAND_MRG32K3A_A12 1403580U
#define ROCRAND_MRG32K3A_A13 (4294967087U - 810728U)
#define ROCRAND_MRG32K3A_A13N 810728U
#define ROCRAND_MRG32K3A_A21 527612U
#define ROCRAND_MRG32K3A_A23 (4294944443U - 1370589U)
#define ROCRAND_MRG32K3A_A23N 1370589U
#define ROCRAND_MRG32K3A_NORM_DOUBLE (2.3283065498378288e-10) // 1/ROCRAND_MRG32K3A_M1
#define ROCRAND_MRG32K3A_UINT_NORM \
    (1.000000048661607) // (ROCRAND_MRG32K3A_POW32 - 1)/(ROCRAND_MRG32K3A_M1 - 1)

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */
 /**
 * \def ROCRAND_MRG32K3A_DEFAULT_SEED
 * \brief Default seed for MRG32K3A PRNG.
 */
 #define ROCRAND_MRG32K3A_DEFAULT_SEED 12345ULL
 /** @} */ // end of group rocranddevice

namespace rocrand_device {

class mrg32k3a_engine
{
public:
    struct mrg32k3a_state
    {
        unsigned int g1[3];
        unsigned int g2[3];

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
    };

    __forceinline__ __device__ __host__ mrg32k3a_engine()
    {
        this->seed(ROCRAND_MRG32K3A_DEFAULT_SEED, 0, 0);
    }

    /// Initializes the internal state of the PRNG using
    /// seed value \p seed, goes to \p subsequence -th subsequence,
    /// and skips \p offset random numbers.
    ///
    /// New seed value should not be zero. If \p seed_value is equal
    /// zero, value \p ROCRAND_MRG32K3A_DEFAULT_SEED is used instead.
    ///
    /// A subsequence is 2^76 numbers long.
    __forceinline__ __device__ __host__ mrg32k3a_engine(const unsigned long long seed,
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
    /// zero, value \p ROCRAND_MRG32K3A_DEFAULT_SEED is used instead.
    ///
    /// A subsequence is 2^76 numbers long.
    __forceinline__ __device__ __host__ void seed(unsigned long long       seed_value,
                                                  const unsigned long long subsequence,
                                                  const unsigned long long offset)
    {
        if(seed_value == 0)
        {
            seed_value = ROCRAND_MRG32K3A_DEFAULT_SEED;
        }
        unsigned int x = (unsigned int) seed_value ^ 0x55555555U;
        unsigned int y = (unsigned int) ((seed_value >> 32) ^ 0xAAAAAAAAU);
        m_state.g1[0] = mod_mul_m1(x, seed_value);
        m_state.g1[1] = mod_mul_m1(y, seed_value);
        m_state.g1[2] = mod_mul_m1(x, seed_value);
        m_state.g2[0] = mod_mul_m2(y, seed_value);
        m_state.g2[1] = mod_mul_m2(x, seed_value);
        m_state.g2[2] = mod_mul_m2(y, seed_value);
        this->restart(subsequence, offset);
    }

    /// Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard(unsigned long long offset)
    {
        this->discard_impl(offset);
    }

    /// Advances the internal state to skip \p subsequence subsequences.
    /// A subsequence is 2^76 numbers long.
    __forceinline__ __device__ __host__ void discard_subsequence(unsigned long long subsequence)
    {
        this->discard_subsequence_impl(subsequence);
    }

    /// Advances the internal state to skip \p sequence sequences.
    /// A sequence is 2^127 numbers long.
    __forceinline__ __device__ __host__ void discard_sequence(unsigned long long sequence)
    {
        this->discard_sequence_impl(sequence);
    }

    __forceinline__ __device__ __host__ void restart(const unsigned long long subsequence,
                                                     const unsigned long long offset)
    {
    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        m_state.boxmuller_float_state = 0;
        m_state.boxmuller_double_state = 0;
    #endif
        this->discard_subsequence_impl(subsequence);
        this->discard_impl(offset);
    }

    __forceinline__ __device__ __host__ unsigned int operator()()
    {
        return this->next();
    }

    // Returned value is in range [1, ROCRAND_MRG32K3A_M1],
    // where ROCRAND_MRG32K3A_M1 < UINT_MAX
    __forceinline__ __device__ __host__
    unsigned int next()
    {
        const unsigned int p1 = mod_m1(detail::mad_u64_u32(
            ROCRAND_MRG32K3A_A12,
            m_state.g1[1],
            detail::mul_u64_u32(ROCRAND_MRG32K3A_A13N, (ROCRAND_MRG32K3A_M1 - m_state.g1[0]))));

        m_state.g1[0] = m_state.g1[1];
        m_state.g1[1] = m_state.g1[2];
        m_state.g1[2] = p1;

        const unsigned int p2 = mod_m2(detail::mad_u64_u32(
            ROCRAND_MRG32K3A_A21,
            m_state.g2[2],
            detail::mul_u64_u32(ROCRAND_MRG32K3A_A23N, (ROCRAND_MRG32K3A_M2 - m_state.g2[0]))));

        m_state.g2[0] = m_state.g2[1];
        m_state.g2[1] = m_state.g2[2];
        m_state.g2[2] = p2;

        return (p1 - p2) + (p1 <= p2 ? ROCRAND_MRG32K3A_M1 : 0);
    }

protected:
    // Advances the internal state to skip \p offset numbers.
    // DOES NOT CALCULATE NEW ULONGLONG
    __forceinline__ __device__ __host__ void discard_impl(unsigned long long offset)
    {
        discard_state(offset);
    }

    // DOES NOT CALCULATE NEW ULONGLONG
    __forceinline__ __device__ __host__ void
        discard_subsequence_impl(unsigned long long subsequence)
    {
        int i = 0;

        while(subsequence > 0) {
            if (subsequence & 1) {
                #if defined(__HIP_DEVICE_COMPILE__)
                mod_mat_vec_m1(d_A1P76 + i, m_state.g1);
                mod_mat_vec_m2(d_A2P76 + i, m_state.g2);
                #else
                mod_mat_vec_m1(h_A1P76 + i, m_state.g1);
                mod_mat_vec_m2(h_A2P76 + i, m_state.g2);
                #endif
            }
            subsequence >>= 1;
            i += 9;
        }
    }

    // DOES NOT CALCULATE NEW ULONGLONG
    __forceinline__ __device__ __host__ void discard_sequence_impl(unsigned long long sequence)
    {
        int i = 0;

        while(sequence > 0) {
            if (sequence & 1) {
                #if defined(__HIP_DEVICE_COMPILE__)
                mod_mat_vec_m1(d_A1P127 + i, m_state.g1);
                mod_mat_vec_m2(d_A2P127 + i, m_state.g2);
                #else
                mod_mat_vec_m1(h_A1P127 + i, m_state.g1);
                mod_mat_vec_m2(h_A2P127 + i, m_state.g2);
                #endif
            }
            sequence >>= 1;
            i += 9;
        }
    }

    // Advances the internal state by offset times.
    // DOES NOT CALCULATE NEW ULONGLONG
    __forceinline__ __device__ __host__ void discard_state(unsigned long long offset)
    {
        int i = 0;

        while(offset > 0) {
            if (offset & 1) {
                #if defined(__HIP_DEVICE_COMPILE__)
                mod_mat_vec_m1(d_A1 + i, m_state.g1);
                mod_mat_vec_m2(d_A2 + i, m_state.g2);
                #else
                mod_mat_vec_m1(h_A1 + i, m_state.g1);
                mod_mat_vec_m2(h_A2 + i, m_state.g2);
                #endif
            }
            offset >>= 1;
            i += 9;
        }
    }

    // Advances the internal state to the next state
    // DOES NOT CALCULATE NEW ULONGLONG
    __forceinline__ __device__ __host__ void discard_state()
    {
        discard_state(1);
    }

private:
    __forceinline__ __device__ __host__
    static void mod_mat_vec_m1(const unsigned int* A, unsigned int* s)
    {
        unsigned long long x[3] = {s[0], s[1], s[2]};

        s[0] = mod_m1(mod_m1(A[0] * x[0]) + mod_m1(A[1] * x[1]) + mod_m1(A[2] * x[2]));

        s[1] = mod_m1(mod_m1(A[3] * x[0]) + mod_m1(A[4] * x[1]) + mod_m1(A[5] * x[2]));

        s[2] = mod_m1(mod_m1(A[6] * x[0]) + mod_m1(A[7] * x[1]) + mod_m1(A[8] * x[2]));
    }

    __forceinline__ __device__ __host__
    static void mod_mat_vec_m2(const unsigned int* A, unsigned int* s)
    {
        unsigned long long x[3] = {s[0], s[1], s[2]};

        s[0] = mod_m2(mod_m2(A[0] * x[0]) + mod_m2(A[1] * x[1]) + mod_m2(A[2] * x[2]));

        s[1] = mod_m2(mod_m2(A[3] * x[0]) + mod_m2(A[4] * x[1]) + mod_m2(A[5] * x[2]));

        s[2] = mod_m2(mod_m2(A[6] * x[0]) + mod_m2(A[7] * x[1]) + mod_m2(A[8] * x[2]));
    }

    __forceinline__ __device__ __host__ static unsigned long long mod_mul_m1(unsigned int       i,
                                                                             unsigned long long j)
    {
        long long hi, lo, temp1, temp2;

        hi = i / 131072;
        lo = i - (hi * 131072);
        temp1 = mod_m1(hi * j) * 131072;
        temp2 = mod_m1(lo * j);
        lo = mod_m1(temp1 + temp2);

        if (lo < 0)
            lo += ROCRAND_MRG32K3A_M1;
        return lo;
    }

    __forceinline__ __device__ __host__
    static unsigned long long mod_m1(unsigned long long p)
    {
        p = detail::mad_u64_u32(ROCRAND_MRG32K3A_M1C,
                                static_cast<unsigned int>(p >> 32),
                                static_cast<unsigned int>(p));
        if(p >= ROCRAND_MRG32K3A_M1)
            p -= ROCRAND_MRG32K3A_M1;

        return p;
    }

    __forceinline__ __device__ __host__
    static unsigned long long mod_mul_m2(unsigned int i, unsigned long long j)
    {
        long long hi, lo, temp1, temp2;

        hi = i / 131072;
        lo = i - (hi * 131072);
        temp1 = mod_m2(hi * j) * 131072;
        temp2 = mod_m2(lo * j);
        lo = mod_m2(temp1 + temp2);

        if (lo < 0)
            lo += ROCRAND_MRG32K3A_M2;
        return lo;
    }

    __forceinline__ __device__ __host__
    static unsigned long long mod_m2(unsigned long long p)
    {
        p = detail::mad_u64_u32(ROCRAND_MRG32K3A_M2C,
                                static_cast<unsigned int>(p >> 32),
                                static_cast<unsigned int>(p));
        p = detail::mad_u64_u32(ROCRAND_MRG32K3A_M2C,
                                static_cast<unsigned int>(p >> 32),
                                static_cast<unsigned int>(p));
        if(p >= ROCRAND_MRG32K3A_M2)
            p -= ROCRAND_MRG32K3A_M2;

        return p;
    }

protected:
    // State
    mrg32k3a_state m_state;

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
    friend struct detail::engine_boxmuller_helper<mrg32k3a_engine>;
    #endif

}; // mrg32k3a_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::mrg32k3a_engine rocrand_state_mrg32k3a;
/// \endcond

/**
 * \brief Initializes MRG32K3A state.
 *
 * Initializes the MRG32K3A generator \p state with the given
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
                  rocrand_state_mrg32k3a*  state)
{
    *state = rocrand_state_mrg32k3a(seed, subsequence, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned int</tt> value
 * from [0; 2^32 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned int</tt>
 * value from [0; 2^32 - 1] range using MRG32K3A generator in \p state.
 * State is incremented by one position.
 *
 * \param state Pointer to a state to use
 *
 * \return Pseudorandom value (32-bit) as an <tt>unsigned int</tt>
 */
__forceinline__ __device__ __host__
unsigned int rocrand(rocrand_state_mrg32k3a* state)
{
    // next() in [1, ROCRAND_MRG32K3A_M1]
    return static_cast<unsigned int>((state->next() - 1) * ROCRAND_MRG32K3A_UINT_NORM);
}

/**
 * \brief Updates MRG32K3A state to skip ahead by \p offset elements.
 *
 * Updates the MRG32K3A state in \p state to skip ahead by \p offset elements.
 *
 * \param offset Number of elements to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead(unsigned long long offset, rocrand_state_mrg32k3a* state)
{
    return state->discard(offset);
}

/**
 * \brief Updates MRG32K3A state to skip ahead by \p subsequence subsequences.
 *
 * Updates the MRG32K3A state in \p state to skip ahead by \p subsequence subsequences.
 * Each subsequence is 2^76 numbers long.
 *
 * \param subsequence Number of subsequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_subsequence(unsigned long long subsequence, rocrand_state_mrg32k3a* state)
{
    return state->discard_subsequence(subsequence);
}

/**
 * \brief Updates MRG32K3A state to skip ahead by \p sequence sequences.
 *
 * Updates the MRG32K3A state in \p state to skip ahead by \p sequence sequences.
 * Each sequence is 2^127 numbers long.
 *
 * \param sequence Number of sequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_sequence(unsigned long long sequence, rocrand_state_mrg32k3a* state)
{
    return state->discard_sequence(sequence);
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_MRG32K3A_H_
