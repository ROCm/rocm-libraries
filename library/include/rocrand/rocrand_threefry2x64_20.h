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

#ifndef ROCRAND_THREEFRY2X64_20_H_
#define ROCRAND_THREEFRY2X64_20_H_

#include "rocrand/rocrand_threefry2_impl.h"

#include <hip/hip_runtime.h>

namespace rocrand_device
{

class threefry2x64_20_engine : public threefry_engine2_base<ulonglong2, unsigned long long, 20>
{
public:
    ROCRAND_DEPRECATED("Alias \"threefry2x64_20_state\" is deprecated. Use \"state_type\" instead.")
    typedef threefry_engine2_base<ulonglong2, unsigned long long, 20>::threefry_state_2
        threefry2x64_20_state;

    /// Initializes the internal state of the PRNG using
    /// seed value \p seed, goes to \p subsequence -th subsequence,
    /// and skips \p offset random numbers.
    ///
    /// A subsequence consists of 2 ^ 65 random numbers.
    __forceinline__ __device__ __host__ threefry2x64_20_engine(const unsigned long long seed = 0,
                                                               const unsigned long long subsequence
                                                               = 0,
                                                               const unsigned long long offset = 0)
    {
        this->seed(seed, subsequence, offset);
    }

    /// Reinitializes the internal state of the PRNG using new
    /// seed value \p seed_value, skips \p subsequence subsequences
    /// and \p offset random numbers.
    ///
    /// A subsequence consists of 2 ^ 65 random numbers.
    __forceinline__ __device__ __host__ void seed(const unsigned long long seed        = 0,
                                                  const unsigned long long subsequence = 0,
                                                  const unsigned long long offset      = 0)
    {
        m_state.counter  = {0ULL, 0ULL};
        m_state.key      = {seed, seed >> 32};
        m_state.result   = {0ULL, 0ULL};
        m_state.substate = 0;

        this->discard_subsequence_impl(subsequence);
        this->discard(offset);
        m_state.result = this->threefry_rounds(m_state.counter, m_state.key);
    }

}; // threefry2x64_20_engine class

} // end namespace rocrand_device

typedef rocrand_device::threefry2x64_20_engine rocrand_state_threefry2x64_20;

/**
 * \brief Initializes Threefry state.
 *
 * Initializes the Threefry generator \p state with the given
 * \p seed, \p subsequence, and \p offset.
 *
 * \param seed Value to use as a seed
 * \param subsequence Subsequence to start at
 * \param offset Absolute offset into subsequence
 * \param state Pointer to state to initialize
 */
__forceinline__ __device__ __host__
void rocrand_init(const unsigned long long       seed,
                  const unsigned long long       subsequence,
                  const unsigned long long       offset,
                  rocrand_state_threefry2x64_20* state)
{
    *state = rocrand_state_threefry2x64_20(seed, subsequence, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned long long</tt> value
 * from [0; 2^64 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned long long</tt>
 * value from [0; 2^64 - 1] range using Threefry generator in \p state.
 * State is incremented by one position.
 *
 * Threefry2x64 has a period of 2 ^ 128 numbers.
 *
 * \param state Pointer to a state to use
 *
 * \return Pseudorandom value (64-bit) as an <tt>unsigned long long</tt>
 */
__forceinline__ __device__ __host__
unsigned long long rocrand(rocrand_state_threefry2x64_20* state)
{
    return state->next();
}

/**
 * \brief Returns two uniformly distributed random <tt>unsigned long long</tt> values
 * from [0; 2^64 - 1] range.
 *
 * Generates and returns two uniformly distributed random <tt>unsigned long long</tt>
 * values from [0; 2^64 - 1] range using Threefry generator in \p state.
 * State is incremented by two positions.
 *
 * \param state Pointer to a state to use
 *
 * \return Two pseudorandom values (64-bit) as an <tt>ulonglong2</tt>
 */
__forceinline__ __device__ __host__
ulonglong2 rocrand2(rocrand_state_threefry2x64_20* state)
{
    return state->next2();
}

#endif // ROCRAND_THREEFRY2X64_20_H_
