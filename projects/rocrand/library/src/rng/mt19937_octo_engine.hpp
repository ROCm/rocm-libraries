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

// Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  3. The names of its contributors may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROCRAND_RNG_MT19937_OCTO_ENGINE_H_
#define ROCRAND_RNG_MT19937_OCTO_ENGINE_H_

#include <rocrand/rocrand_mt19937_precomputed.h>

#include <hip/hip_runtime.h>

namespace rocrand_impl::host
{
namespace mt19937_constants
{
// MT19937 constants.

/// Number of elements in the state vector.
inline constexpr unsigned int n = 624;
/// Exponent of Mersenne prime.
inline constexpr unsigned int mexp = 19937;
/// The next value of element \p i depends on <tt>(i + 1) % n</tt> and <tt>(i + m) % n</tt>.
inline constexpr unsigned int m = 397;
/// Vector constant used in masking operation to represent multiplication by matrix A.
inline constexpr unsigned int matrix_a = 0x9908B0DFU;
/// Mask to select the most significant <tt>w - r</tt> bits from a \p w bits word, where
/// \p w is the number of bits per generated word (32) and \p r is an algorithm constant.
inline constexpr unsigned int upper_mask = 0x80000000U;
/// Mask to select the least significant \p r bits from a \p w bits word, where
/// \p w is the number of bits per generated word (32) and \p r is an algorithm constant.
inline constexpr unsigned int lower_mask = 0x7FFFFFFFU;
} // namespace mt19937_constants

struct mt19937_octo_state
{
    /// Tuples of \p items_per_thread follow a regular pattern.
    static constexpr inline unsigned int items_per_thread = 7U;

    /// Thread 0 has element   0, thread 1 has element 113, thread 2 has element 170,
    /// thread 3 had element 283, thread 4 has element 340, thread 5 has element 397,
    /// thread 6 has element 510, thread 7 has element 567.
    /// Thread i for i in [0, 7) has the following elements (ipt = items_per_thread):
    /// [  1 + ipt * i,   1 + ipt * (i + 1)), [398 + ipt * i, 398 + ipt * (i + 1)), [171 + ipt * i, 171 + ipt * (i + 1)),
    /// [568 + ipt * i, 568 + ipt * (i + 1)), [341 + ipt * i, 341 + ipt * (i + 1)), [114 + ipt * i, 114 + ipt * (i + 1)),
    /// [511 + ipt * i, 511 + ipt * (i + 1)), [284 + ipt * i, 284 + ipt * (i + 1)), [ 57 + ipt * i,  57 + ipt * (i + 1)),
    /// [454 + ipt * i, 454 + ipt * (i + 1)), [227 + ipt * i, 227 + ipt * (i + 1))
    ///
    /// which are 1 + 11 * 7 = 78 values per thread.
    unsigned int mt[1U + items_per_thread * 11U];
};

template<unsigned int>
struct mt19937_octo_engine_accessor;

struct mt19937_octo_engine
{
    template<unsigned int>
    friend struct mt19937_octo_engine_accessor;

    static constexpr inline unsigned int items_per_thread = mt19937_octo_state::items_per_thread;

    /// Number of threads that cooperate to run one generator. Value is fixed in implementation.
    static constexpr inline unsigned int threads_per_generator = 8;

    /// Constants to map the indices to \p mt19937_octo_state.mt_extra indices.
    /// For example, \p i000_0 is the index of \p 0 owned by thread 0.

    static constexpr inline unsigned int i000_0 = 0;
    static constexpr inline unsigned int i113_1 = 0;
    static constexpr inline unsigned int i170_2 = 0;
    static constexpr inline unsigned int i283_3 = 0;
    static constexpr inline unsigned int i340_4 = 0;
    static constexpr inline unsigned int i397_5 = 0;
    static constexpr inline unsigned int i510_6 = 0;
    static constexpr inline unsigned int i567_7 = 0;

    /// Constants used to map the indices to \p mt19937_octo_state.mt indices.
    /// For example, \p i001 is the index of <tt>1 + tid * ipt</tt>.

    static constexpr inline unsigned int i001 = 1 + items_per_thread * 0;
    static constexpr inline unsigned int i057 = 1 + items_per_thread * 1;
    static constexpr inline unsigned int i114 = 1 + items_per_thread * 2;
    static constexpr inline unsigned int i171 = 1 + items_per_thread * 3;
    static constexpr inline unsigned int i227 = 1 + items_per_thread * 4;
    static constexpr inline unsigned int i284 = 1 + items_per_thread * 5;
    static constexpr inline unsigned int i341 = 1 + items_per_thread * 6;
    static constexpr inline unsigned int i398 = 1 + items_per_thread * 7;
    static constexpr inline unsigned int i454 = 1 + items_per_thread * 8;
    static constexpr inline unsigned int i511 = 1 + items_per_thread * 9;
    static constexpr inline unsigned int i568 = 1 + items_per_thread * 10;

    /// Initialize the octo engine from the engine it shares with seven other threads.
    __forceinline__ __device__ __host__
    void gather(const unsigned int engine[mt19937_constants::n], dim3 thread_idx)
    {
        constexpr unsigned int off_cnt = 11;
        /// Used to map the \p mt19937_octo_state.mt indices to \p mt19937_state.mt indices.
        constexpr unsigned int offsets[off_cnt]
            = {1, 57, 114, 171, 227, 284, 341, 398, 454, 511, 568};

        const unsigned int tid = thread_idx.x & 7U;

        // initialize the elements that follow a regular pattern
        for(unsigned int i = 0; i < off_cnt; i++)
        {
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                const unsigned int index = offsets[i] + items_per_thread * tid + j;
                // + 1 for the special value
                m_state.mt[1U + items_per_thread * i + j] = engine[index];
            }
        }

        // initialize the elements that do not follow a regular pattern
        constexpr unsigned int dest_idx[threads_per_generator]
            = {i000_0, i113_1, i170_2, i283_3, i340_4, i397_5, i510_6, i567_7};
        constexpr unsigned int src_idx[threads_per_generator]
            = {0, 113, 170, 283, 340, 397, 510, 567};
        m_state.mt[dest_idx[tid]] = engine[src_idx[tid]];
    }

    /// Returns \p val from thread <tt>tid mod 8</tt>.
    static __forceinline__ __device__ unsigned int shuffle(unsigned int val, unsigned int tid)
    {
        return __shfl(val, tid, 8);
    }

    /// For thread i, returns \p val from thread <tt>(i + 1) mod 8</tt>
    static __forceinline__ __device__ unsigned int shuffle_down(unsigned int val)
    {
        return __shfl_down(val, 1, 8);
    }

    /// For thread i, returns \p val from thread <tt>(i - 1) mod 8</tt>
    static __forceinline__ __device__ unsigned int shuffle_up(unsigned int val)
    {
        return __shfl_up(val, 1, 8);
    }
    /// Calculates value of index \p i using values <tt>i</tt>, <tt>(i + 1) % n</tt>, and <tt>(i + m) % n</tt>.
    static __forceinline__ __device__ __host__
    unsigned int comp(unsigned int mt_i, unsigned int mt_i_1, unsigned int mt_i_m)
    {
        const unsigned int y
            = (mt_i & mt19937_constants::upper_mask) | (mt_i_1 & mt19937_constants::lower_mask);
        unsigned int mag = ((y & 1) ? -1 : 0) & mt19937_constants::matrix_a;
        return mt_i_m ^ (y >> 1) ^ mag;
    }

    /// Thread \p tid computes      [i + ipt * tid,     i + ipt * (tid + 1)).
    /// This computation depends on [i + ipt * tid + 1, i + ipt * (tid + 1) + 1) and
    ///                             [i + ipt * tid + m, i + ipt * (tid + 1) + m).
    /// \p idx_i is the local address of <tt>i</tt>: <tt>i + ipt * tid</tt>.
    /// \p idx_m is the local address of <tt>m</tt>: <tt>i + ipt * tid + m</tt>.
    /// \p last_dep_tid_7 is the value of <tt>i + ipt * (tid + 1)</tt>, which is
    /// required as it is the only value not owned by thread <tt>pid</tt>.
    __forceinline__ __device__ void comp_vector(unsigned int tid,
                                                unsigned int idx_i,
                                                unsigned int idx_m,
                                                unsigned int last_dep_tid_7)
    {
        // communicate the dependency for the last value
        unsigned int last_dep = shuffle_down(m_state.mt[idx_i]);
        // thread 7 needs a special value that does not fit the pattern
        last_dep = tid == 7 ? last_dep_tid_7 : last_dep;

        unsigned int j;
        for(j = 0; j < items_per_thread - 1; j++)
        {
            // compute (i + ipt * tid + j)': needs (i + ipt * tid + 1 + j) % n and (i + ipt * tid + m + j) % n
            m_state.mt[idx_i + j]
                = comp(m_state.mt[idx_i + j], m_state.mt[idx_i + j + 1], m_state.mt[idx_m + j]);
        }
        // compute the last value using the communicated dependency
        m_state.mt[idx_i + j] = comp(m_state.mt[idx_i + j], last_dep, m_state.mt[idx_m + j]);
    }

    __host__
    static void comp_vector(unsigned int idx_i,
                            unsigned int idx_m,
                            unsigned int last_dep_tid_7,
                            mt19937_octo_engine (&thread_engines)[8])
    {
        static constexpr unsigned int numberOfLanes = 8;
        // communicate the dependency for the last value
        unsigned int last_deps[numberOfLanes];
        for(unsigned int i = 0; i < numberOfLanes; ++i)
        {
            last_deps[i] = thread_engines[(i + 1) % numberOfLanes].m_state.mt[idx_i];
        }

        for(unsigned int i = 0; i < numberOfLanes; ++i)
        {
            // thread 7 needs a special value that does not fit the pattern
            unsigned int last_dep = i == 7 ? last_dep_tid_7 : last_deps[i];

            unsigned int j;
            for(j = 0; j < items_per_thread - 1; j++)
            {
                // compute (i + ipt * i + j)': needs (i + ipt * i + 1 + j) % n and (i + ipt * i + m + j) % n
                thread_engines[i].m_state.mt[idx_i + j]
                    = comp(thread_engines[i].m_state.mt[idx_i + j],
                           thread_engines[i].m_state.mt[idx_i + j + 1],
                           thread_engines[i].m_state.mt[idx_m + j]);
            }
            // compute the last value using the communicated dependency
            thread_engines[i].m_state.mt[idx_i + j] = comp(thread_engines[i].m_state.mt[idx_i + j],
                                                           last_dep,
                                                           thread_engines[i].m_state.mt[idx_m + j]);
        }
    }

    /// Eights threads collaborate in computing the n next values.
    __forceinline__ __device__ void gen_next_n()
    {
        const unsigned int tid = threadIdx.x & 7U;

        // compute eleven vectors that follow a regular pattern and compute
        // eight special values for a total of n new elements.
        // ' indicates new value

        // compute   0': needs   1 and 397
        const unsigned int v397 = shuffle(m_state.mt[i397_5], 5);
        if(tid == 0)
        {
            m_state.mt[i000_0] = comp(m_state.mt[i000_0], m_state.mt[i001], v397);
        }

        // compute [  1 + i * ipt,   1 + ipt * (i + 1))' = [  1,  56]':
        // needs [  1,  57] and [398, 453]
        const unsigned int v057 = shuffle(m_state.mt[i057], 0);
        comp_vector(tid, i001, i398, v057);

        // compute [ 57 + i * ipt,  57 + ipt * (i + 1))' = [ 57, 112]':
        // needs [ 57, 113] and [454, 509]
        const unsigned int v113 = shuffle(m_state.mt[i113_1], 1);
        comp_vector(tid, i057, i454, v113);

        // compute 113': needs 114 and 510
        const unsigned int v114 = shuffle(m_state.mt[i114], 0);
        const unsigned int v510 = shuffle(m_state.mt[i510_6], 6);
        if(tid == 1)
        {
            m_state.mt[i113_1] = comp(m_state.mt[i113_1], v114, v510);
        }

        // compute [114 + i * ipt, 114 + ipt * (i + 1))' = [114, 169]':
        // needs [114, 170] and [511, 566]
        const unsigned int v170 = shuffle(m_state.mt[i170_2], 2);
        comp_vector(tid, i114, i511, v170);

        // compute 170': needs 171 and 567
        const unsigned int v171 = shuffle(m_state.mt[i171], 0);
        const unsigned int v567 = shuffle(m_state.mt[i567_7], 7);
        if(tid == 2)
        {
            m_state.mt[i170_2] = comp(m_state.mt[i170_2], v171, v567);
        }

        // compute [171 + i * ipt, 171 + ipt * (i + 1))' = [171, 226]':
        // needs [171, 227] and [568, 623]
        const unsigned int v227 = shuffle(m_state.mt[i227], 0);
        comp_vector(tid, i171, i568, v227);

        // compute [227 + i * ipt, 227 + ipt * (i + 1))' = [227, 282]':
        // needs [227, 283] and [  0,  55]'
        const unsigned int v283 = shuffle(m_state.mt[i283_3], 3);
        // comp_vector(tid, i227, s_000, v283);
        // written out below, since value 0 does not fit the regular pattern
        {
            unsigned int last_dep = shuffle_down(m_state.mt[i227]);
            last_dep              = tid == 7 ? v283 : last_dep;

            // communicate the dependency for the first value
            unsigned int first_dep = shuffle_up(m_state.mt[i001 + items_per_thread - 1]);
            first_dep              = tid == 0 ? m_state.mt[i000_0] : first_dep;

            // extract the first and last iterations from the loop
            unsigned int j       = 0;
            m_state.mt[i227 + j] = comp(m_state.mt[i227 + j], m_state.mt[i227 + j + 1], first_dep);
            for(j = 1; j < items_per_thread - 1; j++)
            {
                m_state.mt[i227 + j] = comp(m_state.mt[i227 + j],
                                            m_state.mt[i227 + j + 1],
                                            m_state.mt[i001 + j - 1]);
            }
            m_state.mt[i227 + j] = comp(m_state.mt[i227 + j], last_dep, m_state.mt[i001 + j - 1]);
        }

        // compute 283': needs 284 and  56'
        const unsigned int v284 = shuffle(m_state.mt[i284], 0);
        const unsigned int v056 = shuffle(m_state.mt[i001 + 6], 7); // 1 + 7 * 7 + 6 = 56
        if(tid == 3)
        {
            m_state.mt[i283_3] = comp(m_state.mt[i283_3], v284, v056);
        }

        // compute [284 + i * ipt, 284 + ipt * (i + 1))' = [284, 339]':
        // needs [284, 340] and [ 57, 112]'
        const unsigned int v340 = shuffle(m_state.mt[i340_4], 4);
        comp_vector(tid, i284, i057, v340);

        // compute 340': needs 341 and 113'
        const unsigned int v113_ = shuffle(m_state.mt[i113_1], 1);
        const unsigned int v341  = shuffle(m_state.mt[i341], 0);
        if(tid == 4)
        {
            m_state.mt[i340_4] = comp(m_state.mt[i340_4], v341, v113_);
        }

        // compute [341 + i * ipt, 341 + ipt * (i + 1))' = [341, 396]':
        // needs [341, 397] and [114, 169]'
        const unsigned int v397_ = shuffle(m_state.mt[i397_5], 5);
        comp_vector(tid, i341, i114, v397_);

        // compute 397': needs 398 and 170'
        const unsigned int v398  = shuffle(m_state.mt[i398], 0);
        const unsigned int v170_ = shuffle(m_state.mt[i170_2], 2);
        if(tid == 5)
        {
            m_state.mt[i397_5] = comp(m_state.mt[i397_5], v398, v170_);
        }

        // compute [398 + i * ipt, 398 + ipt * (i + 1))' = [398, 453]':
        // needs [398, 454] and [171, 226]'
        const unsigned int v454 = shuffle(m_state.mt[i454], 0);
        comp_vector(tid, i398, i171, v454);

        // compute [454 + i * ipt, 454 + ipt * (i + 1))' = [454, 509]':
        // needs [454, 510] and [227, 282]'
        const unsigned int v510_ = shuffle(m_state.mt[i510_6], 6);
        comp_vector(tid, i454, i227, v510_);

        // compute 510': needs 511 and 283'
        const unsigned int v511  = shuffle(m_state.mt[i511], 0);
        const unsigned int v283_ = shuffle(m_state.mt[i283_3], 3);
        if(tid == 6)
        {
            m_state.mt[i510_6] = comp(m_state.mt[i510_6], v511, v283_);
        }

        // compute [511 + i * ipt, 511 + ipt * (i + 1))' = [511, 566]':
        // needs [511, 567] and [284, 339]'
        const unsigned int v567_ = shuffle(m_state.mt[i567_7], 7);
        comp_vector(tid, i511, i284, v567_);

        // compute 567': needs 568 and 340'
        const unsigned int v568 = shuffle(m_state.mt[i568], 0);
        const unsigned int i340 = shuffle(m_state.mt[i340_4], 4);
        if(tid == 7)
        {
            m_state.mt[i567_7] = comp(m_state.mt[i567_7], v568, i340);
        }

        // compute [568 + i * ipt, 568 + ipt * (i + 1))' = [568, 623]':
        // needs [568, 623], [0, 0]', and [341, 396]'
        const unsigned int v000 = shuffle(m_state.mt[i000_0], 0);
        comp_vector(tid, i568, i341, v000);
    }

    static void gen_next_n(mt19937_octo_engine (&thread_engines)[8])
    {
        // compute eleven vectors that follow a regular pattern and compute
        // eight special values for a total of n new elements.
        // ' indicates new value

        // compute   0': needs   1 and 397
        const unsigned int v397 = thread_engines[5].m_state.mt[i397_5];
        thread_engines[0].m_state.mt[i000_0]
            = comp(thread_engines[0].m_state.mt[i000_0], thread_engines[0].m_state.mt[i001], v397);

        // compute [  1 + i * ipt,   1 + ipt * (i + 1))' = [  1,  56]':
        // needs [  1,  57] and [398, 453]
        const unsigned int v057 = thread_engines[0].m_state.mt[i057];
        comp_vector(i001, i398, v057, thread_engines);

        // compute [ 57 + i * ipt,  57 + ipt * (i + 1))' = [ 57, 112]':
        // needs [ 57, 113] and [454, 509]
        const unsigned int v113 = thread_engines[1].m_state.mt[i113_1];
        comp_vector(i057, i454, v113, thread_engines);

        // compute 113': needs 114 and 510
        const unsigned int v114 = thread_engines[0].m_state.mt[i114];
        const unsigned int v510 = thread_engines[6].m_state.mt[i510_6];
        thread_engines[1].m_state.mt[i113_1]
            = comp(thread_engines[1].m_state.mt[i113_1], v114, v510);

        // compute [114 + i * ipt, 114 + ipt * (i + 1))' = [114, 169]':
        // needs [114, 170] and [511, 566]
        const unsigned int v170 = thread_engines[2].m_state.mt[i170_2];
        comp_vector(i114, i511, v170, thread_engines);

        // compute 170': needs 171 and 567
        const unsigned int v171 = thread_engines[0].m_state.mt[i171];
        const unsigned int v567 = thread_engines[7].m_state.mt[i567_7];
        thread_engines[2].m_state.mt[i170_2]
            = comp(thread_engines[2].m_state.mt[i170_2], v171, v567);

        // compute [171 + i * ipt, 171 + ipt * (i + 1))' = [171, 226]':
        // needs [171, 227] and [568, 623]
        const unsigned int v227 = thread_engines[0].m_state.mt[i227];
        comp_vector(i171, i568, v227, thread_engines);

        // compute [227 + i * ipt, 227 + ipt * (i + 1))' = [227, 282]':
        // needs [227, 283] and [  0,  55]'
        const unsigned int v283 = thread_engines[3].m_state.mt[i283_3];
        // comp_vector(tid, i227, s_000, v283);
        // written out below, since value 0 does not fit the regular pattern
        {
            unsigned int last_deps[8];
            unsigned int first_deps[8];

            for(int i = 0; i < 8; ++i)
            {
                last_deps[i]  = thread_engines[(i + 1) % 8].m_state.mt[i227];
                first_deps[i] = thread_engines[(i - 1) % 8].m_state.mt[i001 + items_per_thread - 1];
            }

            for(int i = 0; i < 8; ++i)
            {
                // communicate the dependency for the first and last value
                unsigned int last_dep = i == 7 ? v283 : last_deps[i];
                unsigned int first_dep
                    = i == 0 ? thread_engines[0].m_state.mt[i000_0] : first_deps[i];

                // extract the first and last iterations from the loop
                unsigned int j = 0;
                thread_engines[i].m_state.mt[i227 + j]
                    = comp(thread_engines[i].m_state.mt[i227 + j],
                           thread_engines[i].m_state.mt[i227 + j + 1],
                           first_dep);
                for(j = 1; j < items_per_thread - 1; j++)
                {
                    thread_engines[i].m_state.mt[i227 + j]
                        = comp(thread_engines[i].m_state.mt[i227 + j],
                               thread_engines[i].m_state.mt[i227 + j + 1],
                               thread_engines[i].m_state.mt[i001 + j - 1]);
                }
                thread_engines[i].m_state.mt[i227 + j]
                    = comp(thread_engines[i].m_state.mt[i227 + j],
                           last_dep,
                           thread_engines[i].m_state.mt[i001 + j - 1]);
            }
        }

        // compute 283': needs 284 and  56'
        const unsigned int v284 = thread_engines[0].m_state.mt[i284];
        const unsigned int v056 = thread_engines[7].m_state.mt[i001 + 6]; // 1 + 7 * 7 + 6 = 56
        thread_engines[3].m_state.mt[i283_3]
            = comp(thread_engines[3].m_state.mt[i283_3], v284, v056);

        // compute [284 + i * ipt, 284 + ipt * (i + 1))' = [284, 339]':
        // needs [284, 340] and [ 57, 112]'
        const unsigned int v340 = thread_engines[4].m_state.mt[i340_4];
        comp_vector(i284, i057, v340, thread_engines);

        // compute 340': needs 341 and 113'
        const unsigned int v113_ = thread_engines[1].m_state.mt[i113_1];
        const unsigned int v341  = thread_engines[0].m_state.mt[i341];
        thread_engines[4].m_state.mt[i340_4]
            = comp(thread_engines[4].m_state.mt[i340_4], v341, v113_);

        // compute [341 + i * ipt, 341 + ipt * (i + 1))' = [341, 396]':
        // needs [341, 397] and [114, 169]'
        const unsigned int v397_ = thread_engines[5].m_state.mt[i397_5];
        comp_vector(i341, i114, v397_, thread_engines);

        // compute 397': needs 398 and 170'
        const unsigned int v398  = thread_engines[0].m_state.mt[i398];
        const unsigned int v170_ = thread_engines[2].m_state.mt[i170_2];
        thread_engines[5].m_state.mt[i397_5]
            = comp(thread_engines[5].m_state.mt[i397_5], v398, v170_);

        // compute [398 + i * ipt, 398 + ipt * (i + 1))' = [398, 453]':
        // needs [398, 454] and [171, 226]'
        const unsigned int v454 = thread_engines[0].m_state.mt[i454];
        comp_vector(i398, i171, v454, thread_engines);

        // compute [454 + i * ipt, 454 + ipt * (i + 1))' = [454, 509]':
        // needs [454, 510] and [227, 282]'
        const unsigned int v510_ = thread_engines[6].m_state.mt[i510_6];
        comp_vector(i454, i227, v510_, thread_engines);

        // compute 510': needs 511 and 283'
        const unsigned int v511  = thread_engines[0].m_state.mt[i511];
        const unsigned int v283_ = thread_engines[3].m_state.mt[i283_3];
        thread_engines[6].m_state.mt[i510_6]
            = comp(thread_engines[6].m_state.mt[i510_6], v511, v283_);

        // compute [511 + i * ipt, 511 + ipt * (i + 1))' = [511, 566]':
        // needs [511, 567] and [284, 339]'
        const unsigned int v567_ = thread_engines[7].m_state.mt[i567_7];
        comp_vector(i511, i284, v567_, thread_engines);

        // compute 567': needs 568 and 340'
        const unsigned int v568 = thread_engines[0].m_state.mt[i568];
        const unsigned int i340 = thread_engines[4].m_state.mt[i340_4];
        thread_engines[7].m_state.mt[i567_7]
            = comp(thread_engines[7].m_state.mt[i567_7], v568, i340);

        // compute [568 + i * ipt, 568 + ipt * (i + 1))' = [568, 623]':
        // needs [568, 623], [0, 0]', and [341, 396]'
        const unsigned int v000 = thread_engines[0].m_state.mt[i000_0];
        comp_vector(i568, i341, v000, thread_engines);
    }

    /// Return \p i state value without tempering
    __forceinline__ __device__ __host__
    unsigned int get(unsigned int i) const
    {
        return m_state.mt[i];
    }

    /// Perform tempering on y
    static __forceinline__ __device__ __host__
    unsigned int temper(unsigned int y)
    {
        constexpr unsigned int TEMPERING_MASK_B = 0x9D2C5680U;
        constexpr unsigned int TEMPERING_MASK_C = 0xEFC60000U;

        y ^= y >> 11; /* tempering shift u */
        y ^= (y << 7) /* tempering shift s */ & TEMPERING_MASK_B;
        y ^= (y << 15) /* tempering shift t */ & TEMPERING_MASK_C;
        y ^= y >> 18; /* tempering shift l */

        return y;
    }

private:
    mt19937_octo_state m_state;
};

template<unsigned int stride>
struct mt19937_octo_engine_accessor
{
    __forceinline__ __device__ __host__ explicit mt19937_octo_engine_accessor(unsigned int* _engines)
        : engines(_engines)
    {}

    /// Load one value \p i of the octo engine \p engine_id from global memory with coalesced
    /// access
    __forceinline__ __device__ __host__
    unsigned int load_value(unsigned int engine_id, unsigned int i) const
    {
        return engines[i * stride + engine_id];
    }

    /// Load the octo engine from global memory with coalesced access
    __forceinline__ __device__ __host__
    mt19937_octo_engine load(unsigned int engine_id) const
    {
        mt19937_octo_engine engine;
#pragma unroll
        for(unsigned int i = 0; i < mt19937_constants::n / threads_per_generator; i++)
        {
            engine.m_state.mt[i] = engines[i * stride + engine_id];
        }
        return engine;
    }

    /// Save the octo engine to global memory with coalesced access
    __forceinline__ __device__ __host__
    void save(unsigned int engine_id, const mt19937_octo_engine& engine) const
    {
#pragma unroll
        for(unsigned int i = 0; i < mt19937_constants::n / threads_per_generator; i++)
        {
            engines[i * stride + engine_id] = engine.m_state.mt[i];
        }
    }

private:
    static constexpr inline unsigned int threads_per_generator
        = mt19937_octo_engine::threads_per_generator;

    unsigned int* engines;
};

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_MT19937_OCTO_ENGINE_H_
