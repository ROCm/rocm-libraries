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

#ifndef ROCRAND_SCRAMBLED_SOBOL64_H_
#define ROCRAND_SCRAMBLED_SOBOL64_H_

#include "rocrand/rocrand_sobol64.h"

#include <hip/hip_runtime.h>

namespace rocrand_device
{

template<bool UseSharedVectors>
class scrambled_sobol64_engine
{
public:
    __forceinline__ __device__ __host__ scrambled_sobol64_engine() : scramble_constant() {}

    __forceinline__ __device__ __host__
        scrambled_sobol64_engine(const unsigned long long int* vectors,
                                 const unsigned long long int  scramble_constant,
                                 const unsigned int            offset)
        : m_engine(vectors, 0), scramble_constant(scramble_constant)
    {
        discard(offset);
    }

    /// Advances the internal state to skip \p offset numbers.
    __forceinline__ __device__ __host__ void discard(unsigned long long int offset)
    {
        m_engine.discard(offset);
    }

    __forceinline__ __device__ __host__ void discard()
    {
        m_engine.discard();
    }

    /// Advances the internal state by stride times, where stride is power of 2
    __forceinline__ __device__ __host__ void discard_stride(unsigned long long int stride)
    {
        m_engine.discard_stride(stride);
    }

    __forceinline__ __device__ __host__ unsigned long long int operator()()
    {
        return this->next();
    }

    __forceinline__ __device__ __host__ unsigned long long int next()
    {
        unsigned long long int p = m_engine.next();
        return p ^ scramble_constant;
    }

    __forceinline__ __device__ __host__ unsigned long long int current()
    {
        unsigned long long int p = m_engine.current();
        return p ^ scramble_constant;
    }

    __forceinline__ __device__ __host__ static constexpr bool uses_shared_vectors()
    {
        return UseSharedVectors;
    }

protected:
    // Underlying sobol64 engine
    sobol64_engine<UseSharedVectors> m_engine;
    // scrambling constant
    unsigned long long int scramble_constant;

}; // scrambled_sobol64_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::scrambled_sobol64_engine<false> rocrand_state_scrambled_sobol64;
/// \endcond

/**
 * \brief Initialize scrambled_sobol64 state.
 *
 * Initializes the scrambled_sobol64 generator \p state with the given
 * direction \p vectors and \p offset.
 *
 * \param vectors Direction vectors
 * \param scramble_constant Constant used for scrambling the sequence
 * \param offset Absolute offset into sequence
 * \param state Pointer to state to initialize
 */
__forceinline__ __device__ __host__
void rocrand_init(const unsigned long long int*    vectors,
                  const unsigned long long int     scramble_constant,
                  const unsigned int               offset,
                  rocrand_state_scrambled_sobol64* state)
{
    *state = rocrand_state_scrambled_sobol64(vectors, scramble_constant, offset);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned long long int</tt> value
 * from [0; 2^64 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned long long int</tt>
 * value from [0; 2^64 - 1] range using scrambled_sobol64 generator in \p state.
 * State is incremented by one position.
 *
 * \param state Pointer to a state to use
 *
 * \return Quasirandom value (64-bit) as an <tt>unsigned long long int</tt>
 */
__forceinline__ __device__ __host__
unsigned long long int rocrand(rocrand_state_scrambled_sobol64* state)
{
    return state->next();
}

/**
 * \brief Updates scrambled_sobol64 state to skip ahead by \p offset elements.
 *
 * Updates the scrambled_sobol64 state in \p state to skip ahead by \p offset elements.
 *
 * \param offset Number of elements to skip
 * \param state Pointer to state to update
 */
__forceinline__ __device__ __host__
void skipahead(unsigned long long offset, rocrand_state_scrambled_sobol64* state)
{
    return state->discard(offset);
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_SCRAMBLED_SOBOL64_H_
