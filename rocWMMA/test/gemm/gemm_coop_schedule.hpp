/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GEMM_COOP_SCHEDULE_HPP
#define GEMM_COOP_SCHEDULE_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    namespace CooperativeGemm
    {
        namespace Schedule
        {
            // Collaborative waves in the same workgroup row only.
            // E.g. Wg = (128, 2) = 2x2 waves
            // (0, 0)   (0, 1)  => Share Schedule: i0 = (0, 0), i1 = (0, 1), count = 2
            // (1, 0)   (1, 1)  => Share Schedule: i0 = (1, 0), i2 = (1, 1), count = 2
            template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
            struct SameRowFwd
            {
                using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;

                constexpr static inline auto waveIndex()
                {
                    return get<1>(WaveSpace::localWaveCoord());
                }
                constexpr static inline uint32_t waveCount()
                {
                    return get<1>(WaveSpace::workgroupDim());
                }
            };

            // Collaborative waves in the same workgroup col only.
            // E.g. Wg = (128, 2) = 2x2 waves
            // (0, 0)   (0, 1)
            // (1, 0)   (1, 1)
            //   ||       ||
            //   ||       \/
            //   \/    Share Schedule: i0 = (0, 1), i1 = (1, 1), count = 2
            // Share Schedule: i0 = (0, 1), i1 = (1, 0), count = 2
            template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
            struct SameColFwd
            {
                using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;

                constexpr static inline auto waveIndex()
                {
                    return get<0>(WaveSpace::localWaveCoord());
                }
                constexpr static inline uint32_t waveCount()
                {
                    return get<0>(WaveSpace::workgroupDim());
                }
            };

            // All waves are collaborative.
            // Scheduling order is analogous to row major priority.
            // E.g. Wg = (128, 2) = 2x2 waves
            // (0, 0)   (0, 1)   Share Schedule: i0 = (0, 0), i1 = (0, 1),
            // (1, 0)   (1, 1)                   i2 = (1, 0), i3 = (1, 1), count = 4
            template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
            struct AllRowMajor
            {
                using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;
                constexpr static inline auto waveIndex()
                {
                    auto localWaveCoord = WaveSpace::localWaveCoord();
                    return get<0>(localWaveCoord) * get<1>(WaveSpace::workgroupDim())
                           + get<1>(localWaveCoord);
                }
                constexpr static inline uint32_t waveCount()
                {
                    auto wgDim = WaveSpace::workgroupDim();
                    return get<0>(wgDim) * get<1>(wgDim);
                }
            };

            // All waves are collaborative.
            // Scheduling order is analogous to col major priority.
            // E.g. Wg = (128, 2) = 2x2 waves
            // (0, 0)   (0, 1)   Share Schedule: i0 = (0, 0), i2 = (0, 1),
            // (1, 0)   (1, 1)                   i1 = (1, 0), i3 = (1, 1), count = 4
            template <uint32_t TBlockX = 0, uint32_t TBlockY = 0>
            struct AllColMajor
            {
                using WaveSpace = detail::WaveSpace<TBlockX, TBlockY>;

                constexpr static inline auto waveIndex()
                {
                    auto localWaveCoord = WaveSpace::localWaveCoord();
                    return get<1>(localWaveCoord) * get<0>(WaveSpace::workgroupDim())
                           + get<0>(localWaveCoord);
                }
                constexpr static inline uint32_t waveCount()
                {
                    auto wgDim = WaveSpace::workgroupDim();
                    return get<0>(wgDim) * get<1>(wgDim);
                }
            };

            template <class Schedule>
            struct WaveCountIsConstexpr;

            // Schedule with non-zero TBlockX/Y values has constexpr waveCount();
            template <template <uint32_t, uint32_t> class Schedule,
                      uint32_t TBlockX,
                      uint32_t TBlockY>
            struct WaveCountIsConstexpr<Schedule<TBlockX, TBlockY>> : public std::true_type
            {
            };

            // Schedule with TBlockX/Y = (0,0) values does not have constexpr waveCount();
            template <template <uint32_t, uint32_t> class Schedule>
            struct WaveCountIsConstexpr<Schedule<0u, 0u>> : public std::false_type
            {
            };

        } // namespace Schedule

    } // namespace CooperativeGemm

} // namespace rocwmma

#endif // GEMM_COOP_SCHEDULE_HPP
