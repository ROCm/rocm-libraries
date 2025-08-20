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

/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ROCRAND_PHILOX4X32_10_H_
#define ROCRAND_PHILOX4X32_10_H_

#include "rocrand/rocrand_common.h"

#include <hip/hip_runtime.h>

// Constants from Random123
// See https://www.deshawresearch.com/resources_random123.html
#define ROCRAND_PHILOX_M4x32_0 0xD2511F53U
#define ROCRAND_PHILOX_M4x32_1 0xCD9E8D57U
#define ROCRAND_PHILOX_W32_0 0x9E3779B9U
#define ROCRAND_PHILOX_W32_1 0xBB67AE85U

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */
 /**
 * \def ROCRAND_PHILOX4x32_DEFAULT_SEED
 * \brief Default seed for PHILOX4x32 PRNG.
 */
#define ROCRAND_PHILOX4x32_DEFAULT_SEED 0xdeadbeefdeadbeefULL
/** @} */ // end of group rocranddevice

namespace rocrand_device
{

class philox4x32_10_engine
{
public:
    struct philox4x32_10_state
    {
        uint4 counter;
        uint4 result;
        uint2 key;
        unsigned int substate;

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

    __forceinline__ __device__ __host__ philox4x32_10_engine()
    {
        this->seed(ROCRAND_PHILOX4x32_DEFAULT_SEED, 0, 0);
    }

    /// Initializes the internal state of the PRNG using
    /// seed value \p seed, goes to \p subsequence -th subsequence,
    /// and skips \p offset random numbers.
    ///
    /// A subsequence consists of 2 ^ 66 random numbers.
    __forceinline__ __device__ __host__ philox4x32_10_engine(const unsigned long long seed,
                                                             const unsigned long long subsequence,
                                                             const unsigned long long offset)
    {
        this->seed(seed, subsequence, offset);
    }

    /// Reinitializes the internal state of the PRNG using new
    /// seed value \p seed_value, skips \p subsequence subsequences
    /// and \p offset random numbers.
    ///
    /// A subsequence consists of 2 ^ 66 random numbers.
    __forceinline__ __device__ __host__ void seed(unsigned long long       seed_value,
                                                  const unsigned long long subsequence,
                                                  const unsigned long long offset)
    {
        m_state.key.x = static_cast<unsigned int>(seed_value);
        m_state.key.y = static_cast<unsigned int>(seed_value >> 32);
        this->restart(subsequence, offset);
    }

    /// Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard(unsigned long long offset)
    {
        this->discard_impl(offset);
        this->m_state.result = this->ten_rounds(m_state.counter, m_state.key);
    }

    /// Advances the internal state to skip \p subsequence subsequences,
    /// a subsequence consisting of 2 ^ 66 random numbers.
    /// In other words, this function is equivalent to calling \p discard
    /// 2 ^ 66 times without using the return value, but is much faster.
    __forceinline__ __device__ __host__ void discard_subsequence(unsigned long long subsequence)
    {
        this->discard_subsequence_impl(subsequence);
        m_state.result = this->ten_rounds(m_state.counter, m_state.key);
    }

    __forceinline__ __device__ __host__ void restart(const unsigned long long subsequence,
                                                     const unsigned long long offset)
    {
        m_state.counter = {0, 0, 0, 0};
        m_state.result  = {0, 0, 0, 0};
        m_state.substate = 0;
    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
        m_state.boxmuller_float_state = 0;
        m_state.boxmuller_double_state = 0;
    #endif
        this->discard_subsequence_impl(subsequence);
        this->discard_impl(offset);
        m_state.result = this->ten_rounds(m_state.counter, m_state.key);
    }

    __forceinline__ __device__ __host__ unsigned int operator()()
    {
        return this->next();
    }

    __forceinline__ __device__ __host__ unsigned int next()
    {
    #if defined(__HIP_PLATFORM_AMD__)
        unsigned int ret = ROCRAND_HIPVEC_ACCESS(m_state.result)[m_state.substate];
    #else
        unsigned int ret = (&m_state.result.x)[m_state.substate];
    #endif

        m_state.substate++;
        if(m_state.substate == 4)
        {
            m_state.substate = 0;
            this->discard_state();
            m_state.result = this->ten_rounds(m_state.counter, m_state.key);
        }
        return ret;
    }

    __forceinline__ __device__ __host__ uint4 next4()
    {
        uint4 ret = m_state.result;
        this->discard_state();
        m_state.result = this->ten_rounds(m_state.counter, m_state.key);
        return this->interleave(ret, m_state.result);
    }

protected:
    // Advances the internal state to skip \p offset numbers.
    // DOES NOT CALCULATE NEW 4 UINTs (m_state.result)
    __forceinline__ __device__ __host__ void discard_impl(unsigned long long offset)
    {
        // Adjust offset for subset
        m_state.substate += offset & 3;
        unsigned long long counter_offset = offset / 4;
        counter_offset += m_state.substate < 4 ? 0 : 1;
        m_state.substate += m_state.substate < 4 ? 0 : -4;
        // Discard states
        this->discard_state(counter_offset);
    }

    // DOES NOT CALCULATE NEW 4 UINTs (m_state.result)
    __forceinline__ __device__ __host__ void
        discard_subsequence_impl(unsigned long long subsequence)
    {
        unsigned int lo = static_cast<unsigned int>(subsequence);
        unsigned int hi = static_cast<unsigned int>(subsequence >> 32);

        unsigned int temp = m_state.counter.z;
        m_state.counter.z += lo;
        m_state.counter.w += hi + (m_state.counter.z < temp ? 1 : 0);
    }

    // Advances the internal state by offset times.
    // DOES NOT CALCULATE NEW 4 UINTs (m_state.result)
    __forceinline__ __device__ __host__ void discard_state(unsigned long long offset)
    {
        unsigned int lo = static_cast<unsigned int>(offset);
        unsigned int hi = static_cast<unsigned int>(offset >> 32);

        uint4 temp = m_state.counter;
        m_state.counter.x += lo;
        m_state.counter.y += hi + (m_state.counter.x < temp.x ? 1 : 0);
        m_state.counter.z += (m_state.counter.y < temp.y ? 1 : 0);
        m_state.counter.w += (m_state.counter.z < temp.z ? 1 : 0);
    }

    // Advances the internal state to the next state
    // DOES NOT CALCULATE NEW 4 UINTs (m_state.result)
    __forceinline__ __device__ __host__ void discard_state()
    {
        m_state.counter = this->bump_counter(m_state.counter);
    }

    __forceinline__ __device__ __host__ static uint4 bump_counter(uint4 counter)
    {
        counter.x++;
        unsigned int add      = counter.x == 0 ? 1 : 0;
        counter.y += add; add = counter.y == 0 ? add : 0;
        counter.z += add; add = counter.z == 0 ? add : 0;
        counter.w += add;
        return counter;
    }

    __forceinline__ __device__ __host__ uint4 interleave(const uint4 prev, const uint4 next) const
    {
        switch(m_state.substate)
        {
            case 0:
                return prev;
            case 1:
                return uint4{ prev.y, prev.z, prev.w, next.x };
            case 2:
                return uint4{ prev.z, prev.w, next.x, next.y };
            case 3:
                return uint4{ prev.w, next.x, next.y, next.z };
        }
        __builtin_unreachable();
    }

    // 10 Philox4x32 rounds
    __forceinline__ __device__ __host__ uint4 ten_rounds(uint4 counter, uint2 key)
    {
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 1
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 2
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 3
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 4
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 5
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 6
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 7
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 8
        counter = this->single_round(counter, key); key = this->bumpkey(key); // 9
        return this->single_round(counter, key);                        // 10
    }

private:
    // Single Philox4x32 round
    __forceinline__ __device__ __host__ static uint4 single_round(uint4 counter, uint2 key)
    {
        // Source: Random123
        unsigned long long mul0 = detail::mul_u64_u32(ROCRAND_PHILOX_M4x32_0, counter.x);
        unsigned int       hi0  = static_cast<unsigned int>(mul0 >> 32);
        unsigned int       lo0  = static_cast<unsigned int>(mul0);
        unsigned long long mul1 = detail::mul_u64_u32(ROCRAND_PHILOX_M4x32_1, counter.z);
        unsigned int       hi1  = static_cast<unsigned int>(mul1 >> 32);
        unsigned int       lo1  = static_cast<unsigned int>(mul1);
        return uint4{hi1 ^ counter.y ^ key.x, lo1, hi0 ^ counter.w ^ key.y, lo0};
    }

    __forceinline__ __device__ __host__ static uint2 bumpkey(uint2 key)
    {
        key.x += ROCRAND_PHILOX_W32_0;
        key.y += ROCRAND_PHILOX_W32_1;
        return key;
    }

protected:
    // State
    philox4x32_10_state m_state;

    #ifndef ROCRAND_DETAIL_BM_NOT_IN_STATE
    friend struct detail::engine_boxmuller_helper<philox4x32_10_engine>;
    #endif

}; // philox4x32_10_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::philox4x32_10_engine rocrand_state_philox4x32_10;
/// \endcond

/**
 * \brief Initializes Philox state.
 *
 * Initializes the Philox generator \p state with the given
 * \p seed, \p subsequence, and \p offset.
 *
 * \param seed Value to use as a seed
 * \param subsequence Subsequence to start at
 * \param offset Absolute offset into subsequence
 * \param state Pointer to state to initialize
 */
__forceinline__ __device__ __host__
void rocrand_init(const unsigned long long     seed,
                  const unsigned long long     subsequence,
                  const unsigned long long     offset,
                  rocrand_state_philox4x32_10* state)
{
    *state = rocrand_state_philox4x32_10(seed, subsequence, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned int</tt> value
 * from [0; 2^32 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned int</tt>
 * value from [0; 2^32 - 1] range using Philox generator in \p state.
 * State is incremented by one position.
 *
 * \param state Pointer to a state to use
 *
 * \return Pseudorandom value (32-bit) as an <tt>unsigned int</tt>
 */
__forceinline__ __device__ __host__
unsigned int rocrand(rocrand_state_philox4x32_10* state)
{
    return state->next();
}

/**
 * \brief Returns four uniformly distributed random <tt>unsigned int</tt> values
 * from [0; 2^32 - 1] range.
 *
 * Generates and returns four uniformly distributed random <tt>unsigned int</tt>
 * values from [0; 2^32 - 1] range using Philox generator in \p state.
 * State is incremented by four positions.
 *
 * \param state Pointer to a state to use
 *
 * \return Four pseudorandom values (32-bit) as an <tt>uint4</tt>
 */
__forceinline__ __device__ __host__
uint4 rocrand4(rocrand_state_philox4x32_10* state)
{
    return state->next4();
}

/**
 * \brief Updates Philox state to skip ahead by \p offset elements.
 *
 * Updates the Philox generator state in \p state to skip ahead by \p offset elements.
 *
 * \param offset Number of elements to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead(unsigned long long offset, rocrand_state_philox4x32_10* state)
{
    return state->discard(offset);
}

/**
 * \brief Updates Philox state to skip ahead by \p subsequence subsequences.
 *
 * Updates the Philox generator state in \p state to skip ahead by \p subsequence subsequences.
 * Each subsequence is 4 * 2^64 numbers long.
 *
 * \param subsequence Number of subsequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_subsequence(unsigned long long subsequence, rocrand_state_philox4x32_10* state)
{
    return state->discard_subsequence(subsequence);
}

/**
 * \brief Updates Philox state to skip ahead by \p sequence sequences.
 *
 * Updates the Philox generator state in \p state skipping \p sequence sequences ahead.
 * For Philox each sequence is 4 * 2^64 numbers long (equal to the size of a subsequence).
 *
 * \param sequence Number of sequences to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead_sequence(unsigned long long sequence, rocrand_state_philox4x32_10* state)
{
    return state->discard_subsequence(sequence);
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_PHILOX4X32_10_H_
