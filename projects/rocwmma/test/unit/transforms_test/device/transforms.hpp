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

#ifndef ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP
#define ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP

#include "hip_device.hpp"
#include "transforms.hpp"
#include <rocwmma/rocwmma.hpp>

namespace rocwmma
{
    namespace detail
    {
        ROCWMMA_DEVICE static inline auto threadId()
        {
            return (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x)
                   + threadIdx.x;
        }
    }
    template <typename DataT, uint32_t VW, uint32_t BlockDim>
    struct AosVec;

    template <typename DataT, uint32_t VW, uint32_t BlockDim>
    struct SoaVec;

    // AosVec
    template <typename DataT>
    struct AosVec<DataT, 16, 16>
    {
        static constexpr uint32_t VW       = 16;
        static constexpr uint32_t BlockDim = 16;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) * VW + waveOffset;

            return VecType{start,
                           start + 1,
                           start + 2,
                           start + 3,
                           start + 4,
                           start + 5,
                           start + 6,
                           start + 7,
                           start + 8,
                           start + 9,
                           start + 10,
                           start + 11,
                           start + 12,
                           start + 13,
                           start + 14,
                           start + 15};
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 16, 32>
    {
        static constexpr uint32_t VW       = 16;
        static constexpr uint32_t BlockDim = 32;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) * VW + waveOffset;

            return VecType{start,
                           start + 1,
                           start + 2,
                           start + 3,
                           start + 4,
                           start + 5,
                           start + 6,
                           start + 7,
                           start + 8,
                           start + 9,
                           start + 10,
                           start + 11,
                           start + 12,
                           start + 13,
                           start + 14,
                           start + 15};
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 16, 64>
    {
        static constexpr uint32_t VW        = 16;
        static constexpr uint32_t BlockDim  = 64;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{start,
                               start + 1,
                               start + 2,
                               start + 3,
                               start + 4,
                               start + 5,
                               start + 6,
                               start + 7,
                               start + 8,
                               start + 9,
                               start + 10,
                               start + 11,
                               start + 12,
                               start + 13,
                               start + 14,
                               start + 15};
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{start,
                               start + 1,
                               start + 2,
                               start + 3,
                               start + 4,
                               start + 5,
                               start + 6,
                               start + 7,
                               start + 8,
                               start + 9,
                               start + 10,
                               start + 11,
                               start + 12,
                               start + 13,
                               start + 14,
                               start + 15,
                               start + WAVE_SIZE,
                               start + WAVE_SIZE + 1,
                               start + WAVE_SIZE + 2,
                               start + WAVE_SIZE + 3,
                               start + WAVE_SIZE + 4,
                               start + WAVE_SIZE + 5,
                               start + WAVE_SIZE + 6,
                               start + WAVE_SIZE + 7,
                               start + WAVE_SIZE + 8,
                               start + WAVE_SIZE + 9,
                               start + WAVE_SIZE + 10,
                               start + WAVE_SIZE + 11,
                               start + WAVE_SIZE + 12,
                               start + WAVE_SIZE + 13,
                               start + WAVE_SIZE + 14,
                               start + WAVE_SIZE + 15};
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 16, 128>
    {
        static constexpr uint32_t VW        = 16;
        static constexpr uint32_t BlockDim  = 128;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{start,
                               start + 1,
                               start + 2,
                               start + 3,
                               start + 4,
                               start + 5,
                               start + 6,
                               start + 7,
                               start + 8,
                               start + 9,
                               start + 10,
                               start + 11,
                               start + 12,
                               start + 13,
                               start + 14,
                               start + 15,
                               start + WAVE_SIZE,
                               start + WAVE_SIZE + 1,
                               start + WAVE_SIZE + 2,
                               start + WAVE_SIZE + 3,
                               start + WAVE_SIZE + 4,
                               start + WAVE_SIZE + 5,
                               start + WAVE_SIZE + 6,
                               start + WAVE_SIZE + 7,
                               start + WAVE_SIZE + 8,
                               start + WAVE_SIZE + 9,
                               start + WAVE_SIZE + 10,
                               start + WAVE_SIZE + 11,
                               start + WAVE_SIZE + 12,
                               start + WAVE_SIZE + 13,
                               start + WAVE_SIZE + 14,
                               start + WAVE_SIZE + 15};
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{start,
                               start + 1,
                               start + 2,
                               start + 3,
                               start + 4,
                               start + 5,
                               start + 6,
                               start + 7,
                               start + 8,
                               start + 9,
                               start + 10,
                               start + 11,
                               start + 12,
                               start + 13,
                               start + 14,
                               start + 15,
                               start + WAVE_SIZE * 1,
                               start + WAVE_SIZE * 1 + 1,
                               start + WAVE_SIZE * 1 + 2,
                               start + WAVE_SIZE * 1 + 3,
                               start + WAVE_SIZE * 1 + 4,
                               start + WAVE_SIZE * 1 + 5,
                               start + WAVE_SIZE * 1 + 6,
                               start + WAVE_SIZE * 1 + 7,
                               start + WAVE_SIZE * 1 + 8,
                               start + WAVE_SIZE * 1 + 9,
                               start + WAVE_SIZE * 1 + 10,
                               start + WAVE_SIZE * 1 + 11,
                               start + WAVE_SIZE * 1 + 12,
                               start + WAVE_SIZE * 1 + 13,
                               start + WAVE_SIZE * 1 + 14,
                               start + WAVE_SIZE * 1 + 15,
                               start + WAVE_SIZE * 2,
                               start + WAVE_SIZE * 2 + 1,
                               start + WAVE_SIZE * 2 + 2,
                               start + WAVE_SIZE * 2 + 3,
                               start + WAVE_SIZE * 2 + 4,
                               start + WAVE_SIZE * 2 + 5,
                               start + WAVE_SIZE * 2 + 6,
                               start + WAVE_SIZE * 2 + 7,
                               start + WAVE_SIZE * 2 + 8,
                               start + WAVE_SIZE * 2 + 9,
                               start + WAVE_SIZE * 2 + 10,
                               start + WAVE_SIZE * 2 + 11,
                               start + WAVE_SIZE * 2 + 12,
                               start + WAVE_SIZE * 2 + 13,
                               start + WAVE_SIZE * 2 + 14,
                               start + WAVE_SIZE * 2 + 15,
                               start + WAVE_SIZE * 3,
                               start + WAVE_SIZE * 3 + 1,
                               start + WAVE_SIZE * 3 + 2,
                               start + WAVE_SIZE * 3 + 3,
                               start + WAVE_SIZE * 3 + 4,
                               start + WAVE_SIZE * 3 + 5,
                               start + WAVE_SIZE * 3 + 6,
                               start + WAVE_SIZE * 3 + 7,
                               start + WAVE_SIZE * 3 + 8,
                               start + WAVE_SIZE * 3 + 9,
                               start + WAVE_SIZE * 3 + 10,
                               start + WAVE_SIZE * 3 + 11,
                               start + WAVE_SIZE * 3 + 12,
                               start + WAVE_SIZE * 3 + 13,
                               start + WAVE_SIZE * 3 + 14,
                               start + WAVE_SIZE * 3 + 15};
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 16, 256>
    {
        static constexpr uint32_t VW        = 16;
        static constexpr uint32_t BlockDim  = 256;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{start,
                               start + 1,
                               start + 2,
                               start + 3,
                               start + 4,
                               start + 5,
                               start + 6,
                               start + 7,
                               start + 8,
                               start + 9,
                               start + 10,
                               start + 11,
                               start + 12,
                               start + 13,
                               start + 14,
                               start + 15,
                               start + WAVE_SIZE * 1,
                               start + WAVE_SIZE * 1 + 1,
                               start + WAVE_SIZE * 1 + 2,
                               start + WAVE_SIZE * 1 + 3,
                               start + WAVE_SIZE * 1 + 4,
                               start + WAVE_SIZE * 1 + 5,
                               start + WAVE_SIZE * 1 + 6,
                               start + WAVE_SIZE * 1 + 7,
                               start + WAVE_SIZE * 1 + 8,
                               start + WAVE_SIZE * 1 + 9,
                               start + WAVE_SIZE * 1 + 10,
                               start + WAVE_SIZE * 1 + 11,
                               start + WAVE_SIZE * 1 + 12,
                               start + WAVE_SIZE * 1 + 13,
                               start + WAVE_SIZE * 1 + 14,
                               start + WAVE_SIZE * 1 + 15,
                               start + WAVE_SIZE * 2,
                               start + WAVE_SIZE * 2 + 1,
                               start + WAVE_SIZE * 2 + 2,
                               start + WAVE_SIZE * 2 + 3,
                               start + WAVE_SIZE * 2 + 4,
                               start + WAVE_SIZE * 2 + 5,
                               start + WAVE_SIZE * 2 + 6,
                               start + WAVE_SIZE * 2 + 7,
                               start + WAVE_SIZE * 2 + 8,
                               start + WAVE_SIZE * 2 + 9,
                               start + WAVE_SIZE * 2 + 10,
                               start + WAVE_SIZE * 2 + 11,
                               start + WAVE_SIZE * 2 + 12,
                               start + WAVE_SIZE * 2 + 13,
                               start + WAVE_SIZE * 2 + 14,
                               start + WAVE_SIZE * 2 + 15,
                               start + WAVE_SIZE * 3,
                               start + WAVE_SIZE * 3 + 1,
                               start + WAVE_SIZE * 3 + 2,
                               start + WAVE_SIZE * 3 + 3,
                               start + WAVE_SIZE * 3 + 4,
                               start + WAVE_SIZE * 3 + 5,
                               start + WAVE_SIZE * 3 + 6,
                               start + WAVE_SIZE * 3 + 7,
                               start + WAVE_SIZE * 3 + 8,
                               start + WAVE_SIZE * 3 + 9,
                               start + WAVE_SIZE * 3 + 10,
                               start + WAVE_SIZE * 3 + 11,
                               start + WAVE_SIZE * 3 + 12,
                               start + WAVE_SIZE * 3 + 13,
                               start + WAVE_SIZE * 3 + 14,
                               start + WAVE_SIZE * 3 + 15};
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{start,
                               start + 1,
                               start + 2,
                               start + 3,
                               start + 4,
                               start + 5,
                               start + 6,
                               start + 7,
                               start + 8,
                               start + 9,
                               start + 10,
                               start + 11,
                               start + 12,
                               start + 13,
                               start + 14,
                               start + 15,
                               start + WAVE_SIZE * 1,
                               start + WAVE_SIZE * 1 + 1,
                               start + WAVE_SIZE * 1 + 2,
                               start + WAVE_SIZE * 1 + 3,
                               start + WAVE_SIZE * 1 + 4,
                               start + WAVE_SIZE * 1 + 5,
                               start + WAVE_SIZE * 1 + 6,
                               start + WAVE_SIZE * 1 + 7,
                               start + WAVE_SIZE * 1 + 8,
                               start + WAVE_SIZE * 1 + 9,
                               start + WAVE_SIZE * 1 + 10,
                               start + WAVE_SIZE * 1 + 11,
                               start + WAVE_SIZE * 1 + 12,
                               start + WAVE_SIZE * 1 + 13,
                               start + WAVE_SIZE * 1 + 14,
                               start + WAVE_SIZE * 1 + 15,
                               start + WAVE_SIZE * 2,
                               start + WAVE_SIZE * 2 + 1,
                               start + WAVE_SIZE * 2 + 2,
                               start + WAVE_SIZE * 2 + 3,
                               start + WAVE_SIZE * 2 + 4,
                               start + WAVE_SIZE * 2 + 5,
                               start + WAVE_SIZE * 2 + 6,
                               start + WAVE_SIZE * 2 + 7,
                               start + WAVE_SIZE * 2 + 8,
                               start + WAVE_SIZE * 2 + 9,
                               start + WAVE_SIZE * 2 + 10,
                               start + WAVE_SIZE * 2 + 11,
                               start + WAVE_SIZE * 2 + 12,
                               start + WAVE_SIZE * 2 + 13,
                               start + WAVE_SIZE * 2 + 14,
                               start + WAVE_SIZE * 2 + 15,
                               start + WAVE_SIZE * 3,
                               start + WAVE_SIZE * 3 + 1,
                               start + WAVE_SIZE * 3 + 2,
                               start + WAVE_SIZE * 3 + 3,
                               start + WAVE_SIZE * 3 + 4,
                               start + WAVE_SIZE * 3 + 5,
                               start + WAVE_SIZE * 3 + 6,
                               start + WAVE_SIZE * 3 + 7,
                               start + WAVE_SIZE * 3 + 8,
                               start + WAVE_SIZE * 3 + 9,
                               start + WAVE_SIZE * 3 + 10,
                               start + WAVE_SIZE * 3 + 11,
                               start + WAVE_SIZE * 3 + 12,
                               start + WAVE_SIZE * 3 + 13,
                               start + WAVE_SIZE * 3 + 14,
                               start + WAVE_SIZE * 3 + 15,
                               start + WAVE_SIZE * 4,
                               start + WAVE_SIZE * 4 + 1,
                               start + WAVE_SIZE * 4 + 2,
                               start + WAVE_SIZE * 4 + 3,
                               start + WAVE_SIZE * 4 + 4,
                               start + WAVE_SIZE * 4 + 5,
                               start + WAVE_SIZE * 4 + 6,
                               start + WAVE_SIZE * 4 + 7,
                               start + WAVE_SIZE * 4 + 8,
                               start + WAVE_SIZE * 4 + 9,
                               start + WAVE_SIZE * 4 + 10,
                               start + WAVE_SIZE * 4 + 11,
                               start + WAVE_SIZE * 4 + 12,
                               start + WAVE_SIZE * 4 + 13,
                               start + WAVE_SIZE * 4 + 14,
                               start + WAVE_SIZE * 4 + 15,
                               start + WAVE_SIZE * 5,
                               start + WAVE_SIZE * 5 + 1,
                               start + WAVE_SIZE * 5 + 2,
                               start + WAVE_SIZE * 5 + 3,
                               start + WAVE_SIZE * 5 + 4,
                               start + WAVE_SIZE * 5 + 5,
                               start + WAVE_SIZE * 5 + 6,
                               start + WAVE_SIZE * 5 + 7,
                               start + WAVE_SIZE * 5 + 8,
                               start + WAVE_SIZE * 5 + 9,
                               start + WAVE_SIZE * 5 + 10,
                               start + WAVE_SIZE * 5 + 11,
                               start + WAVE_SIZE * 5 + 12,
                               start + WAVE_SIZE * 5 + 13,
                               start + WAVE_SIZE * 5 + 14,
                               start + WAVE_SIZE * 5 + 15,
                               start + WAVE_SIZE * 6,
                               start + WAVE_SIZE * 6 + 1,
                               start + WAVE_SIZE * 6 + 2,
                               start + WAVE_SIZE * 6 + 3,
                               start + WAVE_SIZE * 6 + 4,
                               start + WAVE_SIZE * 6 + 5,
                               start + WAVE_SIZE * 6 + 6,
                               start + WAVE_SIZE * 6 + 7,
                               start + WAVE_SIZE * 6 + 8,
                               start + WAVE_SIZE * 6 + 9,
                               start + WAVE_SIZE * 6 + 10,
                               start + WAVE_SIZE * 6 + 11,
                               start + WAVE_SIZE * 6 + 12,
                               start + WAVE_SIZE * 6 + 13,
                               start + WAVE_SIZE * 6 + 14,
                               start + WAVE_SIZE * 6 + 15,
                               start + WAVE_SIZE * 7,
                               start + WAVE_SIZE * 7 + 1,
                               start + WAVE_SIZE * 7 + 2,
                               start + WAVE_SIZE * 7 + 3,
                               start + WAVE_SIZE * 7 + 4,
                               start + WAVE_SIZE * 7 + 5,
                               start + WAVE_SIZE * 7 + 6,
                               start + WAVE_SIZE * 7 + 7,
                               start + WAVE_SIZE * 7 + 8,
                               start + WAVE_SIZE * 7 + 9,
                               start + WAVE_SIZE * 7 + 10,
                               start + WAVE_SIZE * 7 + 11,
                               start + WAVE_SIZE * 7 + 12,
                               start + WAVE_SIZE * 7 + 13,
                               start + WAVE_SIZE * 7 + 14,
                               start + WAVE_SIZE * 7 + 15};
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 16>
    {
        static constexpr uint32_t VW       = 8;
        static constexpr uint32_t BlockDim = 16;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) * VW + waveOffset;

            return VecType{
                start,
                start + 1,
                start + 2,
                start + 3,
                start + 4,
                start + 5,
                start + 6,
                start + 7,
            };
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 32>
    {
        static constexpr uint32_t VW       = 8;
        static constexpr uint32_t BlockDim = 32;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) * VW + waveOffset;

            return VecType{
                start,
                start + 1,
                start + 2,
                start + 3,
                start + 4,
                start + 5,
                start + 6,
                start + 7,
            };
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 64>
    {
        static constexpr uint32_t VW        = 8;
        static constexpr uint32_t BlockDim  = 64;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + WAVE_SIZE,
                    start + WAVE_SIZE + 1,
                    start + WAVE_SIZE + 2,
                    start + WAVE_SIZE + 3,
                    start + WAVE_SIZE + 4,
                    start + WAVE_SIZE + 5,
                    start + WAVE_SIZE + 6,
                    start + WAVE_SIZE + 7,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 128>
    {
        static constexpr uint32_t VW        = 8;
        static constexpr uint32_t BlockDim  = 128;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + WAVE_SIZE,
                    start + WAVE_SIZE + 1,
                    start + WAVE_SIZE + 2,
                    start + WAVE_SIZE + 3,
                    start + WAVE_SIZE + 4,
                    start + WAVE_SIZE + 5,
                    start + WAVE_SIZE + 6,
                    start + WAVE_SIZE + 7,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 1 + 2,
                    start + WAVE_SIZE * 1 + 3,
                    start + WAVE_SIZE * 1 + 4,
                    start + WAVE_SIZE * 1 + 5,
                    start + WAVE_SIZE * 1 + 6,
                    start + WAVE_SIZE * 1 + 7,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 2 + 2,
                    start + WAVE_SIZE * 2 + 3,
                    start + WAVE_SIZE * 2 + 4,
                    start + WAVE_SIZE * 2 + 5,
                    start + WAVE_SIZE * 2 + 6,
                    start + WAVE_SIZE * 2 + 7,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                    start + WAVE_SIZE * 3 + 2,
                    start + WAVE_SIZE * 3 + 3,
                    start + WAVE_SIZE * 3 + 4,
                    start + WAVE_SIZE * 3 + 5,
                    start + WAVE_SIZE * 3 + 6,
                    start + WAVE_SIZE * 3 + 7,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 8, 256>
    {
        static constexpr uint32_t VW        = 8;
        static constexpr uint32_t BlockDim  = 256;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 1 + 2,
                    start + WAVE_SIZE * 1 + 3,
                    start + WAVE_SIZE * 1 + 4,
                    start + WAVE_SIZE * 1 + 5,
                    start + WAVE_SIZE * 1 + 6,
                    start + WAVE_SIZE * 1 + 7,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 2 + 2,
                    start + WAVE_SIZE * 2 + 3,
                    start + WAVE_SIZE * 2 + 4,
                    start + WAVE_SIZE * 2 + 5,
                    start + WAVE_SIZE * 2 + 6,
                    start + WAVE_SIZE * 2 + 7,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                    start + WAVE_SIZE * 3 + 2,
                    start + WAVE_SIZE * 3 + 3,
                    start + WAVE_SIZE * 3 + 4,
                    start + WAVE_SIZE * 3 + 5,
                    start + WAVE_SIZE * 3 + 6,
                    start + WAVE_SIZE * 3 + 7,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + 4,
                    start + 5,
                    start + 6,
                    start + 7,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 1 + 2,
                    start + WAVE_SIZE * 1 + 3,
                    start + WAVE_SIZE * 1 + 4,
                    start + WAVE_SIZE * 1 + 5,
                    start + WAVE_SIZE * 1 + 6,
                    start + WAVE_SIZE * 1 + 7,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 2 + 2,
                    start + WAVE_SIZE * 2 + 3,
                    start + WAVE_SIZE * 2 + 4,
                    start + WAVE_SIZE * 2 + 5,
                    start + WAVE_SIZE * 2 + 6,
                    start + WAVE_SIZE * 2 + 7,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                    start + WAVE_SIZE * 3 + 2,
                    start + WAVE_SIZE * 3 + 3,
                    start + WAVE_SIZE * 3 + 4,
                    start + WAVE_SIZE * 3 + 5,
                    start + WAVE_SIZE * 3 + 6,
                    start + WAVE_SIZE * 3 + 7,
                    start + WAVE_SIZE * 4,
                    start + WAVE_SIZE * 4 + 1,
                    start + WAVE_SIZE * 4 + 2,
                    start + WAVE_SIZE * 4 + 3,
                    start + WAVE_SIZE * 4 + 4,
                    start + WAVE_SIZE * 4 + 5,
                    start + WAVE_SIZE * 4 + 6,
                    start + WAVE_SIZE * 4 + 7,
                    start + WAVE_SIZE * 5,
                    start + WAVE_SIZE * 5 + 1,
                    start + WAVE_SIZE * 5 + 2,
                    start + WAVE_SIZE * 5 + 3,
                    start + WAVE_SIZE * 5 + 4,
                    start + WAVE_SIZE * 5 + 5,
                    start + WAVE_SIZE * 5 + 6,
                    start + WAVE_SIZE * 5 + 7,
                    start + WAVE_SIZE * 6,
                    start + WAVE_SIZE * 6 + 1,
                    start + WAVE_SIZE * 6 + 2,
                    start + WAVE_SIZE * 6 + 3,
                    start + WAVE_SIZE * 6 + 4,
                    start + WAVE_SIZE * 6 + 5,
                    start + WAVE_SIZE * 6 + 6,
                    start + WAVE_SIZE * 6 + 7,
                    start + WAVE_SIZE * 7,
                    start + WAVE_SIZE * 7 + 1,
                    start + WAVE_SIZE * 7 + 2,
                    start + WAVE_SIZE * 7 + 3,
                    start + WAVE_SIZE * 7 + 4,
                    start + WAVE_SIZE * 7 + 5,
                    start + WAVE_SIZE * 7 + 6,
                    start + WAVE_SIZE * 7 + 7,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 16>
    {
        static constexpr uint32_t VW       = 4;
        static constexpr uint32_t BlockDim = 16;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) * VW + waveOffset;
            return VecType{
                start,
                start + 1,
                start + 2,
                start + 3,
            };
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 32>
    {
        static constexpr uint32_t VW       = 4;
        static constexpr uint32_t BlockDim = 32;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) * VW + waveOffset;

            return VecType{
                start,
                start + 1,
                start + 2,
                start + 3,
            };
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 64>
    {
        static constexpr uint32_t VW        = 4;
        static constexpr uint32_t BlockDim  = 64;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 1 + 2,
                    start + WAVE_SIZE * 1 + 3,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 128>
    {
        static constexpr uint32_t VW        = 4;
        static constexpr uint32_t BlockDim  = 128;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 1 + 2,
                    start + WAVE_SIZE * 1 + 3,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 1 + 2,
                    start + WAVE_SIZE * 1 + 3,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 2 + 2,
                    start + WAVE_SIZE * 2 + 3,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                    start + WAVE_SIZE * 3 + 2,
                    start + WAVE_SIZE * 3 + 3,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 4, 256>
    {
        static constexpr uint32_t VW        = 4;
        static constexpr uint32_t BlockDim  = 256;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 1 + 2,
                    start + WAVE_SIZE * 1 + 3,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 2 + 2,
                    start + WAVE_SIZE * 2 + 3,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                    start + WAVE_SIZE * 3 + 2,
                    start + WAVE_SIZE * 3 + 3,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + 2,
                    start + 3,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 1 + 2,
                    start + WAVE_SIZE * 1 + 3,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 2 + 2,
                    start + WAVE_SIZE * 2 + 3,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                    start + WAVE_SIZE * 3 + 2,
                    start + WAVE_SIZE * 3 + 3,
                    start + WAVE_SIZE * 4,
                    start + WAVE_SIZE * 4 + 1,
                    start + WAVE_SIZE * 4 + 2,
                    start + WAVE_SIZE * 4 + 3,
                    start + WAVE_SIZE * 5,
                    start + WAVE_SIZE * 5 + 1,
                    start + WAVE_SIZE * 5 + 2,
                    start + WAVE_SIZE * 5 + 3,
                    start + WAVE_SIZE * 6,
                    start + WAVE_SIZE * 6 + 1,
                    start + WAVE_SIZE * 6 + 2,
                    start + WAVE_SIZE * 6 + 3,
                    start + WAVE_SIZE * 7,
                    start + WAVE_SIZE * 7 + 1,
                    start + WAVE_SIZE * 7 + 2,
                    start + WAVE_SIZE * 7 + 3,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 2, 16>
    {
        static constexpr uint32_t VW       = 2;
        static constexpr uint32_t BlockDim = 16;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) * VW + waveOffset;

            return VecType{
                start,
                start + 1,
            };
        }
    };

    template <typename DataT>
    struct AosVec<DataT, 2, 32>
    {
        static constexpr uint32_t VW       = 2;
        static constexpr uint32_t BlockDim = 32;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) * VW + waveOffset;

            return VecType{
                start,
                start + 1,
            };
        }
    };
    template <typename DataT>
    struct AosVec<DataT, 2, 64>
    {
        static constexpr uint32_t VW        = 2;
        static constexpr uint32_t BlockDim  = 64;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };
    template <typename DataT>
    struct AosVec<DataT, 2, 128>
    {
        static constexpr uint32_t VW        = 2;
        static constexpr uint32_t BlockDim  = 128;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };
    template <typename DataT>
    struct AosVec<DataT, 2, 256>
    {
        static constexpr uint32_t VW        = 2;
        static constexpr uint32_t BlockDim  = 256;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = (threadId * VW / WAVE_SIZE) * BlockDim;
            auto const start      = threadId % (WAVE_SIZE / VW) * VW + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    start + 1,
                    start + WAVE_SIZE * 1,
                    start + WAVE_SIZE * 1 + 1,
                    start + WAVE_SIZE * 2,
                    start + WAVE_SIZE * 2 + 1,
                    start + WAVE_SIZE * 3,
                    start + WAVE_SIZE * 3 + 1,
                    start + WAVE_SIZE * 4,
                    start + WAVE_SIZE * 4 + 1,
                    start + WAVE_SIZE * 5,
                    start + WAVE_SIZE * 5 + 1,
                    start + WAVE_SIZE * 6,
                    start + WAVE_SIZE * 6 + 1,
                    start + WAVE_SIZE * 7,
                    start + WAVE_SIZE * 7 + 1,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    // SoaVec
    template <typename DataT>
    struct SoaVec<DataT, 16, 16>
    {
        static constexpr uint32_t VW       = 16;
        static constexpr uint32_t BlockDim = 16;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) % BlockDim + waveOffset;

            return VecType{start,
                           BlockDim + start,
                           BlockDim * 2 + start,
                           BlockDim * 3 + start,
                           BlockDim * 4 + start,
                           BlockDim * 5 + start,
                           BlockDim * 6 + start,
                           BlockDim * 7 + start,
                           BlockDim * 8 + start,
                           BlockDim * 9 + start,
                           BlockDim * 10 + start,
                           BlockDim * 11 + start,
                           BlockDim * 12 + start,
                           BlockDim * 13 + start,
                           BlockDim * 14 + start,
                           BlockDim * 15 + start};
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 16, 32>
    {
        static constexpr uint32_t VW       = 16;
        static constexpr uint32_t BlockDim = 32;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) % BlockDim + waveOffset;

            return VecType{start,
                           BlockDim + start,
                           BlockDim * 2 + start,
                           BlockDim * 3 + start,
                           BlockDim * 4 + start,
                           BlockDim * 5 + start,
                           BlockDim * 6 + start,
                           BlockDim * 7 + start,
                           BlockDim * 8 + start,
                           BlockDim * 9 + start,
                           BlockDim * 10 + start,
                           BlockDim * 11 + start,
                           BlockDim * 12 + start,
                           BlockDim * 13 + start,
                           BlockDim * 14 + start,
                           BlockDim * 15 + start};
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 16, 64>
    {
        static constexpr uint32_t VW        = 16;
        static constexpr uint32_t BlockDim  = 64;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{start,
                               BlockDim + start,
                               BlockDim * 2 + start,
                               BlockDim * 3 + start,
                               BlockDim * 4 + start,
                               BlockDim * 5 + start,
                               BlockDim * 6 + start,
                               BlockDim * 7 + start,
                               BlockDim * 8 + start,
                               BlockDim * 9 + start,
                               BlockDim * 10 + start,
                               BlockDim * 11 + start,
                               BlockDim * 12 + start,
                               BlockDim * 13 + start,
                               BlockDim * 14 + start,
                               BlockDim * 15 + start};
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{start,
                               BlockDim + start,
                               BlockDim * 2 + start,
                               BlockDim * 3 + start,
                               BlockDim * 4 + start,
                               BlockDim * 5 + start,
                               BlockDim * 6 + start,
                               BlockDim * 7 + start,
                               BlockDim * 8 + start,
                               BlockDim * 9 + start,
                               BlockDim * 10 + start,
                               BlockDim * 11 + start,
                               BlockDim * 12 + start,
                               BlockDim * 13 + start,
                               BlockDim * 14 + start,
                               BlockDim * 15 + start,
                               WAVE_SIZE + start,
                               WAVE_SIZE + BlockDim + start,
                               WAVE_SIZE + BlockDim * 2 + start,
                               WAVE_SIZE + BlockDim * 3 + start,
                               WAVE_SIZE + BlockDim * 4 + start,
                               WAVE_SIZE + BlockDim * 5 + start,
                               WAVE_SIZE + BlockDim * 6 + start,
                               WAVE_SIZE + BlockDim * 7 + start,
                               WAVE_SIZE + BlockDim * 8 + start,
                               WAVE_SIZE + BlockDim * 9 + start,
                               WAVE_SIZE + BlockDim * 10 + start,
                               WAVE_SIZE + BlockDim * 11 + start,
                               WAVE_SIZE + BlockDim * 12 + start,
                               WAVE_SIZE + BlockDim * 13 + start,
                               WAVE_SIZE + BlockDim * 14 + start,
                               WAVE_SIZE + BlockDim * 15 + start};
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 16, 128>
    {
        static constexpr uint32_t VW        = 16;
        static constexpr uint32_t BlockDim  = 128;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{start,
                               BlockDim + start,
                               BlockDim * 2 + start,
                               BlockDim * 3 + start,
                               BlockDim * 4 + start,
                               BlockDim * 5 + start,
                               BlockDim * 6 + start,
                               BlockDim * 7 + start,
                               BlockDim * 8 + start,
                               BlockDim * 9 + start,
                               BlockDim * 10 + start,
                               BlockDim * 11 + start,
                               BlockDim * 12 + start,
                               BlockDim * 13 + start,
                               BlockDim * 14 + start,
                               BlockDim * 15 + start,
                               WAVE_SIZE + start,
                               WAVE_SIZE + BlockDim + start,
                               WAVE_SIZE + BlockDim * 2 + start,
                               WAVE_SIZE + BlockDim * 3 + start,
                               WAVE_SIZE + BlockDim * 4 + start,
                               WAVE_SIZE + BlockDim * 5 + start,
                               WAVE_SIZE + BlockDim * 6 + start,
                               WAVE_SIZE + BlockDim * 7 + start,
                               WAVE_SIZE + BlockDim * 8 + start,
                               WAVE_SIZE + BlockDim * 9 + start,
                               WAVE_SIZE + BlockDim * 10 + start,
                               WAVE_SIZE + BlockDim * 11 + start,
                               WAVE_SIZE + BlockDim * 12 + start,
                               WAVE_SIZE + BlockDim * 13 + start,
                               WAVE_SIZE + BlockDim * 14 + start,
                               WAVE_SIZE + BlockDim * 15 + start};
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{start,
                               BlockDim + start,
                               BlockDim * 2 + start,
                               BlockDim * 3 + start,
                               BlockDim * 4 + start,
                               BlockDim * 5 + start,
                               BlockDim * 6 + start,
                               BlockDim * 7 + start,
                               BlockDim * 8 + start,
                               BlockDim * 9 + start,
                               BlockDim * 10 + start,
                               BlockDim * 11 + start,
                               BlockDim * 12 + start,
                               BlockDim * 13 + start,
                               BlockDim * 14 + start,
                               BlockDim * 15 + start,
                               WAVE_SIZE * 1 + start,
                               WAVE_SIZE * 1 + BlockDim + start,
                               WAVE_SIZE * 1 + BlockDim * 2 + start,
                               WAVE_SIZE * 1 + BlockDim * 3 + start,
                               WAVE_SIZE * 1 + BlockDim * 4 + start,
                               WAVE_SIZE * 1 + BlockDim * 5 + start,
                               WAVE_SIZE * 1 + BlockDim * 6 + start,
                               WAVE_SIZE * 1 + BlockDim * 7 + start,
                               WAVE_SIZE * 1 + BlockDim * 8 + start,
                               WAVE_SIZE * 1 + BlockDim * 9 + start,
                               WAVE_SIZE * 1 + BlockDim * 10 + start,
                               WAVE_SIZE * 1 + BlockDim * 11 + start,
                               WAVE_SIZE * 1 + BlockDim * 12 + start,
                               WAVE_SIZE * 1 + BlockDim * 13 + start,
                               WAVE_SIZE * 1 + BlockDim * 14 + start,
                               WAVE_SIZE * 1 + BlockDim * 15 + start,
                               WAVE_SIZE * 2 + start,
                               WAVE_SIZE * 2 + BlockDim + start,
                               WAVE_SIZE * 2 + BlockDim * 2 + start,
                               WAVE_SIZE * 2 + BlockDim * 3 + start,
                               WAVE_SIZE * 2 + BlockDim * 4 + start,
                               WAVE_SIZE * 2 + BlockDim * 5 + start,
                               WAVE_SIZE * 2 + BlockDim * 6 + start,
                               WAVE_SIZE * 2 + BlockDim * 7 + start,
                               WAVE_SIZE * 2 + BlockDim * 8 + start,
                               WAVE_SIZE * 2 + BlockDim * 9 + start,
                               WAVE_SIZE * 2 + BlockDim * 10 + start,
                               WAVE_SIZE * 2 + BlockDim * 11 + start,
                               WAVE_SIZE * 2 + BlockDim * 12 + start,
                               WAVE_SIZE * 2 + BlockDim * 13 + start,
                               WAVE_SIZE * 2 + BlockDim * 14 + start,
                               WAVE_SIZE * 2 + BlockDim * 15 + start,
                               WAVE_SIZE * 3 + start,
                               WAVE_SIZE * 3 + BlockDim + start,
                               WAVE_SIZE * 3 + BlockDim * 2 + start,
                               WAVE_SIZE * 3 + BlockDim * 3 + start,
                               WAVE_SIZE * 3 + BlockDim * 4 + start,
                               WAVE_SIZE * 3 + BlockDim * 5 + start,
                               WAVE_SIZE * 3 + BlockDim * 6 + start,
                               WAVE_SIZE * 3 + BlockDim * 7 + start,
                               WAVE_SIZE * 3 + BlockDim * 8 + start,
                               WAVE_SIZE * 3 + BlockDim * 9 + start,
                               WAVE_SIZE * 3 + BlockDim * 10 + start,
                               WAVE_SIZE * 3 + BlockDim * 11 + start,
                               WAVE_SIZE * 3 + BlockDim * 12 + start,
                               WAVE_SIZE * 3 + BlockDim * 13 + start,
                               WAVE_SIZE * 3 + BlockDim * 14 + start,
                               WAVE_SIZE * 3 + BlockDim * 15 + start};
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 16, 256>
    {
        static constexpr uint32_t VW        = 16;
        static constexpr uint32_t BlockDim  = 256;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{start,
                               BlockDim + start,
                               BlockDim * 2 + start,
                               BlockDim * 3 + start,
                               BlockDim * 4 + start,
                               BlockDim * 5 + start,
                               BlockDim * 6 + start,
                               BlockDim * 7 + start,
                               BlockDim * 8 + start,
                               BlockDim * 9 + start,
                               BlockDim * 10 + start,
                               BlockDim * 11 + start,
                               BlockDim * 12 + start,
                               BlockDim * 13 + start,
                               BlockDim * 14 + start,
                               BlockDim * 15 + start,
                               WAVE_SIZE * 1 + start,
                               WAVE_SIZE * 1 + BlockDim + start,
                               WAVE_SIZE * 1 + BlockDim * 2 + start,
                               WAVE_SIZE * 1 + BlockDim * 3 + start,
                               WAVE_SIZE * 1 + BlockDim * 4 + start,
                               WAVE_SIZE * 1 + BlockDim * 5 + start,
                               WAVE_SIZE * 1 + BlockDim * 6 + start,
                               WAVE_SIZE * 1 + BlockDim * 7 + start,
                               WAVE_SIZE * 1 + BlockDim * 8 + start,
                               WAVE_SIZE * 1 + BlockDim * 9 + start,
                               WAVE_SIZE * 1 + BlockDim * 10 + start,
                               WAVE_SIZE * 1 + BlockDim * 11 + start,
                               WAVE_SIZE * 1 + BlockDim * 12 + start,
                               WAVE_SIZE * 1 + BlockDim * 13 + start,
                               WAVE_SIZE * 1 + BlockDim * 14 + start,
                               WAVE_SIZE * 1 + BlockDim * 15 + start,
                               WAVE_SIZE * 2 + start,
                               WAVE_SIZE * 2 + BlockDim + start,
                               WAVE_SIZE * 2 + BlockDim * 2 + start,
                               WAVE_SIZE * 2 + BlockDim * 3 + start,
                               WAVE_SIZE * 2 + BlockDim * 4 + start,
                               WAVE_SIZE * 2 + BlockDim * 5 + start,
                               WAVE_SIZE * 2 + BlockDim * 6 + start,
                               WAVE_SIZE * 2 + BlockDim * 7 + start,
                               WAVE_SIZE * 2 + BlockDim * 8 + start,
                               WAVE_SIZE * 2 + BlockDim * 9 + start,
                               WAVE_SIZE * 2 + BlockDim * 10 + start,
                               WAVE_SIZE * 2 + BlockDim * 11 + start,
                               WAVE_SIZE * 2 + BlockDim * 12 + start,
                               WAVE_SIZE * 2 + BlockDim * 13 + start,
                               WAVE_SIZE * 2 + BlockDim * 14 + start,
                               WAVE_SIZE * 2 + BlockDim * 15 + start,
                               WAVE_SIZE * 3 + start,
                               WAVE_SIZE * 3 + BlockDim + start,
                               WAVE_SIZE * 3 + BlockDim * 2 + start,
                               WAVE_SIZE * 3 + BlockDim * 3 + start,
                               WAVE_SIZE * 3 + BlockDim * 4 + start,
                               WAVE_SIZE * 3 + BlockDim * 5 + start,
                               WAVE_SIZE * 3 + BlockDim * 6 + start,
                               WAVE_SIZE * 3 + BlockDim * 7 + start,
                               WAVE_SIZE * 3 + BlockDim * 8 + start,
                               WAVE_SIZE * 3 + BlockDim * 9 + start,
                               WAVE_SIZE * 3 + BlockDim * 10 + start,
                               WAVE_SIZE * 3 + BlockDim * 11 + start,
                               WAVE_SIZE * 3 + BlockDim * 12 + start,
                               WAVE_SIZE * 3 + BlockDim * 13 + start,
                               WAVE_SIZE * 3 + BlockDim * 14 + start,
                               WAVE_SIZE * 3 + BlockDim * 15 + start};
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{start,
                               BlockDim + start,
                               BlockDim * 2 + start,
                               BlockDim * 3 + start,
                               BlockDim * 4 + start,
                               BlockDim * 5 + start,
                               BlockDim * 6 + start,
                               BlockDim * 7 + start,
                               BlockDim * 8 + start,
                               BlockDim * 9 + start,
                               BlockDim * 10 + start,
                               BlockDim * 11 + start,
                               BlockDim * 12 + start,
                               BlockDim * 13 + start,
                               BlockDim * 14 + start,
                               BlockDim * 15 + start,
                               WAVE_SIZE * 1 + start,
                               WAVE_SIZE * 1 + BlockDim + start,
                               WAVE_SIZE * 1 + BlockDim * 2 + start,
                               WAVE_SIZE * 1 + BlockDim * 3 + start,
                               WAVE_SIZE * 1 + BlockDim * 4 + start,
                               WAVE_SIZE * 1 + BlockDim * 5 + start,
                               WAVE_SIZE * 1 + BlockDim * 6 + start,
                               WAVE_SIZE * 1 + BlockDim * 7 + start,
                               WAVE_SIZE * 1 + BlockDim * 8 + start,
                               WAVE_SIZE * 1 + BlockDim * 9 + start,
                               WAVE_SIZE * 1 + BlockDim * 10 + start,
                               WAVE_SIZE * 1 + BlockDim * 11 + start,
                               WAVE_SIZE * 1 + BlockDim * 12 + start,
                               WAVE_SIZE * 1 + BlockDim * 13 + start,
                               WAVE_SIZE * 1 + BlockDim * 14 + start,
                               WAVE_SIZE * 1 + BlockDim * 15 + start,
                               WAVE_SIZE * 2 + start,
                               WAVE_SIZE * 2 + BlockDim + start,
                               WAVE_SIZE * 2 + BlockDim * 2 + start,
                               WAVE_SIZE * 2 + BlockDim * 3 + start,
                               WAVE_SIZE * 2 + BlockDim * 4 + start,
                               WAVE_SIZE * 2 + BlockDim * 5 + start,
                               WAVE_SIZE * 2 + BlockDim * 6 + start,
                               WAVE_SIZE * 2 + BlockDim * 7 + start,
                               WAVE_SIZE * 2 + BlockDim * 8 + start,
                               WAVE_SIZE * 2 + BlockDim * 9 + start,
                               WAVE_SIZE * 2 + BlockDim * 10 + start,
                               WAVE_SIZE * 2 + BlockDim * 11 + start,
                               WAVE_SIZE * 2 + BlockDim * 12 + start,
                               WAVE_SIZE * 2 + BlockDim * 13 + start,
                               WAVE_SIZE * 2 + BlockDim * 14 + start,
                               WAVE_SIZE * 2 + BlockDim * 15 + start,
                               WAVE_SIZE * 3 + start,
                               WAVE_SIZE * 3 + BlockDim + start,
                               WAVE_SIZE * 3 + BlockDim * 2 + start,
                               WAVE_SIZE * 3 + BlockDim * 3 + start,
                               WAVE_SIZE * 3 + BlockDim * 4 + start,
                               WAVE_SIZE * 3 + BlockDim * 5 + start,
                               WAVE_SIZE * 3 + BlockDim * 6 + start,
                               WAVE_SIZE * 3 + BlockDim * 7 + start,
                               WAVE_SIZE * 3 + BlockDim * 8 + start,
                               WAVE_SIZE * 3 + BlockDim * 9 + start,
                               WAVE_SIZE * 3 + BlockDim * 10 + start,
                               WAVE_SIZE * 3 + BlockDim * 11 + start,
                               WAVE_SIZE * 3 + BlockDim * 12 + start,
                               WAVE_SIZE * 3 + BlockDim * 13 + start,
                               WAVE_SIZE * 3 + BlockDim * 14 + start,
                               WAVE_SIZE * 3 + BlockDim * 15 + start,
                               WAVE_SIZE * 4 + start,
                               WAVE_SIZE * 4 + BlockDim + start,
                               WAVE_SIZE * 4 + BlockDim * 2 + start,
                               WAVE_SIZE * 4 + BlockDim * 3 + start,
                               WAVE_SIZE * 4 + BlockDim * 4 + start,
                               WAVE_SIZE * 4 + BlockDim * 5 + start,
                               WAVE_SIZE * 4 + BlockDim * 6 + start,
                               WAVE_SIZE * 4 + BlockDim * 7 + start,
                               WAVE_SIZE * 4 + BlockDim * 8 + start,
                               WAVE_SIZE * 4 + BlockDim * 9 + start,
                               WAVE_SIZE * 4 + BlockDim * 10 + start,
                               WAVE_SIZE * 4 + BlockDim * 11 + start,
                               WAVE_SIZE * 4 + BlockDim * 12 + start,
                               WAVE_SIZE * 4 + BlockDim * 13 + start,
                               WAVE_SIZE * 4 + BlockDim * 14 + start,
                               WAVE_SIZE * 4 + BlockDim * 15 + start,
                               WAVE_SIZE * 5 + start,
                               WAVE_SIZE * 5 + BlockDim + start,
                               WAVE_SIZE * 5 + BlockDim * 2 + start,
                               WAVE_SIZE * 5 + BlockDim * 3 + start,
                               WAVE_SIZE * 5 + BlockDim * 4 + start,
                               WAVE_SIZE * 5 + BlockDim * 5 + start,
                               WAVE_SIZE * 5 + BlockDim * 6 + start,
                               WAVE_SIZE * 5 + BlockDim * 7 + start,
                               WAVE_SIZE * 5 + BlockDim * 8 + start,
                               WAVE_SIZE * 5 + BlockDim * 9 + start,
                               WAVE_SIZE * 5 + BlockDim * 10 + start,
                               WAVE_SIZE * 5 + BlockDim * 11 + start,
                               WAVE_SIZE * 5 + BlockDim * 12 + start,
                               WAVE_SIZE * 5 + BlockDim * 13 + start,
                               WAVE_SIZE * 5 + BlockDim * 14 + start,
                               WAVE_SIZE * 5 + BlockDim * 15 + start,
                               WAVE_SIZE * 6 + start,
                               WAVE_SIZE * 6 + BlockDim + start,
                               WAVE_SIZE * 6 + BlockDim * 2 + start,
                               WAVE_SIZE * 6 + BlockDim * 3 + start,
                               WAVE_SIZE * 6 + BlockDim * 4 + start,
                               WAVE_SIZE * 6 + BlockDim * 5 + start,
                               WAVE_SIZE * 6 + BlockDim * 6 + start,
                               WAVE_SIZE * 6 + BlockDim * 7 + start,
                               WAVE_SIZE * 6 + BlockDim * 8 + start,
                               WAVE_SIZE * 6 + BlockDim * 9 + start,
                               WAVE_SIZE * 6 + BlockDim * 10 + start,
                               WAVE_SIZE * 6 + BlockDim * 11 + start,
                               WAVE_SIZE * 6 + BlockDim * 12 + start,
                               WAVE_SIZE * 6 + BlockDim * 13 + start,
                               WAVE_SIZE * 6 + BlockDim * 14 + start,
                               WAVE_SIZE * 6 + BlockDim * 15 + start,
                               WAVE_SIZE * 7 + start,
                               WAVE_SIZE * 7 + BlockDim + start,
                               WAVE_SIZE * 7 + BlockDim * 2 + start,
                               WAVE_SIZE * 7 + BlockDim * 3 + start,
                               WAVE_SIZE * 7 + BlockDim * 4 + start,
                               WAVE_SIZE * 7 + BlockDim * 5 + start,
                               WAVE_SIZE * 7 + BlockDim * 6 + start,
                               WAVE_SIZE * 7 + BlockDim * 7 + start,
                               WAVE_SIZE * 7 + BlockDim * 8 + start,
                               WAVE_SIZE * 7 + BlockDim * 9 + start,
                               WAVE_SIZE * 7 + BlockDim * 10 + start,
                               WAVE_SIZE * 7 + BlockDim * 11 + start,
                               WAVE_SIZE * 7 + BlockDim * 12 + start,
                               WAVE_SIZE * 7 + BlockDim * 13 + start,
                               WAVE_SIZE * 7 + BlockDim * 14 + start,
                               WAVE_SIZE * 7 + BlockDim * 15 + start};
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 16>
    {
        static constexpr uint32_t VW       = 8;
        static constexpr uint32_t BlockDim = 16;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) % BlockDim + waveOffset;

            return VecType{
                start,
                BlockDim + start,
                BlockDim * 2 + start,
                BlockDim * 3 + start,
                BlockDim * 4 + start,
                BlockDim * 5 + start,
                BlockDim * 6 + start,
                BlockDim * 7 + start,
            };
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 32>
    {
        static constexpr uint32_t VW       = 8;
        static constexpr uint32_t BlockDim = 32;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) % BlockDim + waveOffset;

            return VecType{
                start,
                BlockDim + start,
                BlockDim * 2 + start,
                BlockDim * 3 + start,
                BlockDim * 4 + start,
                BlockDim * 5 + start,
                BlockDim * 6 + start,
                BlockDim * 7 + start,
            };
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 64>
    {
        static constexpr uint32_t VW        = 8;
        static constexpr uint32_t BlockDim  = 64;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    BlockDim * 4 + start,
                    BlockDim * 5 + start,
                    BlockDim * 6 + start,
                    BlockDim * 7 + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    BlockDim * 4 + start,
                    BlockDim * 5 + start,
                    BlockDim * 6 + start,
                    BlockDim * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                    WAVE_SIZE + BlockDim * 4 + start,
                    WAVE_SIZE + BlockDim * 5 + start,
                    WAVE_SIZE + BlockDim * 6 + start,
                    WAVE_SIZE + BlockDim * 7 + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 128>
    {
        static constexpr uint32_t VW        = 8;
        static constexpr uint32_t BlockDim  = 128;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    BlockDim * 4 + start,
                    BlockDim * 5 + start,
                    BlockDim * 6 + start,
                    BlockDim * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                    WAVE_SIZE + BlockDim * 4 + start,
                    WAVE_SIZE + BlockDim * 5 + start,
                    WAVE_SIZE + BlockDim * 6 + start,
                    WAVE_SIZE + BlockDim * 7 + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    BlockDim * 4 + start,
                    BlockDim * 5 + start,
                    BlockDim * 6 + start,
                    BlockDim * 7 + start,
                    WAVE_SIZE * 1 + start,
                    WAVE_SIZE * 1 + BlockDim + start,
                    WAVE_SIZE * 1 + BlockDim * 2 + start,
                    WAVE_SIZE * 1 + BlockDim * 3 + start,
                    WAVE_SIZE * 1 + BlockDim * 4 + start,
                    WAVE_SIZE * 1 + BlockDim * 5 + start,
                    WAVE_SIZE * 1 + BlockDim * 6 + start,
                    WAVE_SIZE * 1 + BlockDim * 7 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 2 + BlockDim * 2 + start,
                    WAVE_SIZE * 2 + BlockDim * 3 + start,
                    WAVE_SIZE * 2 + BlockDim * 4 + start,
                    WAVE_SIZE * 2 + BlockDim * 5 + start,
                    WAVE_SIZE * 2 + BlockDim * 6 + start,
                    WAVE_SIZE * 2 + BlockDim * 7 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                    WAVE_SIZE * 3 + BlockDim * 2 + start,
                    WAVE_SIZE * 3 + BlockDim * 3 + start,
                    WAVE_SIZE * 3 + BlockDim * 4 + start,
                    WAVE_SIZE * 3 + BlockDim * 5 + start,
                    WAVE_SIZE * 3 + BlockDim * 6 + start,
                    WAVE_SIZE * 3 + BlockDim * 7 + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 8, 256>
    {
        static constexpr uint32_t VW        = 8;
        static constexpr uint32_t BlockDim  = 256;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    BlockDim * 4 + start,
                    BlockDim * 5 + start,
                    BlockDim * 6 + start,
                    BlockDim * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                    WAVE_SIZE + BlockDim * 4 + start,
                    WAVE_SIZE + BlockDim * 5 + start,
                    WAVE_SIZE + BlockDim * 6 + start,
                    WAVE_SIZE + BlockDim * 7 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 2 + BlockDim * 2 + start,
                    WAVE_SIZE * 2 + BlockDim * 3 + start,
                    WAVE_SIZE * 2 + BlockDim * 4 + start,
                    WAVE_SIZE * 2 + BlockDim * 5 + start,
                    WAVE_SIZE * 2 + BlockDim * 6 + start,
                    WAVE_SIZE * 2 + BlockDim * 7 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                    WAVE_SIZE * 3 + BlockDim * 2 + start,
                    WAVE_SIZE * 3 + BlockDim * 3 + start,
                    WAVE_SIZE * 3 + BlockDim * 4 + start,
                    WAVE_SIZE * 3 + BlockDim * 5 + start,
                    WAVE_SIZE * 3 + BlockDim * 6 + start,
                    WAVE_SIZE * 3 + BlockDim * 7 + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    BlockDim * 4 + start,
                    BlockDim * 5 + start,
                    BlockDim * 6 + start,
                    BlockDim * 7 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                    WAVE_SIZE + BlockDim * 4 + start,
                    WAVE_SIZE + BlockDim * 5 + start,
                    WAVE_SIZE + BlockDim * 6 + start,
                    WAVE_SIZE + BlockDim * 7 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 2 + BlockDim * 2 + start,
                    WAVE_SIZE * 2 + BlockDim * 3 + start,
                    WAVE_SIZE * 2 + BlockDim * 4 + start,
                    WAVE_SIZE * 2 + BlockDim * 5 + start,
                    WAVE_SIZE * 2 + BlockDim * 6 + start,
                    WAVE_SIZE * 2 + BlockDim * 7 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                    WAVE_SIZE * 3 + BlockDim * 2 + start,
                    WAVE_SIZE * 3 + BlockDim * 3 + start,
                    WAVE_SIZE * 3 + BlockDim * 4 + start,
                    WAVE_SIZE * 3 + BlockDim * 5 + start,
                    WAVE_SIZE * 3 + BlockDim * 6 + start,
                    WAVE_SIZE * 3 + BlockDim * 7 + start,
                    WAVE_SIZE * 4 + start,
                    WAVE_SIZE * 4 + BlockDim + start,
                    WAVE_SIZE * 4 + BlockDim * 2 + start,
                    WAVE_SIZE * 4 + BlockDim * 3 + start,
                    WAVE_SIZE * 4 + BlockDim * 4 + start,
                    WAVE_SIZE * 4 + BlockDim * 5 + start,
                    WAVE_SIZE * 4 + BlockDim * 6 + start,
                    WAVE_SIZE * 4 + BlockDim * 7 + start,
                    WAVE_SIZE * 5 + start,
                    WAVE_SIZE * 5 + BlockDim + start,
                    WAVE_SIZE * 5 + BlockDim * 2 + start,
                    WAVE_SIZE * 5 + BlockDim * 3 + start,
                    WAVE_SIZE * 5 + BlockDim * 4 + start,
                    WAVE_SIZE * 5 + BlockDim * 5 + start,
                    WAVE_SIZE * 5 + BlockDim * 6 + start,
                    WAVE_SIZE * 5 + BlockDim * 7 + start,
                    WAVE_SIZE * 6 + start,
                    WAVE_SIZE * 6 + BlockDim + start,
                    WAVE_SIZE * 6 + BlockDim * 2 + start,
                    WAVE_SIZE * 6 + BlockDim * 3 + start,
                    WAVE_SIZE * 6 + BlockDim * 4 + start,
                    WAVE_SIZE * 6 + BlockDim * 5 + start,
                    WAVE_SIZE * 6 + BlockDim * 6 + start,
                    WAVE_SIZE * 6 + BlockDim * 7 + start,
                    WAVE_SIZE * 7 + start,
                    WAVE_SIZE * 7 + BlockDim + start,
                    WAVE_SIZE * 7 + BlockDim * 2 + start,
                    WAVE_SIZE * 7 + BlockDim * 3 + start,
                    WAVE_SIZE * 7 + BlockDim * 4 + start,
                    WAVE_SIZE * 7 + BlockDim * 5 + start,
                    WAVE_SIZE * 7 + BlockDim * 6 + start,
                    WAVE_SIZE * 7 + BlockDim * 7 + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 16>
    {
        static constexpr uint32_t VW       = 4;
        static constexpr uint32_t BlockDim = 16;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) % BlockDim + waveOffset;

            return VecType{
                start,
                BlockDim + start,
                BlockDim * 2 + start,
                BlockDim * 3 + start,
            };
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 32>
    {
        static constexpr uint32_t VW       = 4;
        static constexpr uint32_t BlockDim = 32;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) % BlockDim + waveOffset;

            return VecType{
                start,
                BlockDim + start,
                BlockDim * 2 + start,
                BlockDim * 3 + start,
            };
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 64>
    {
        static constexpr uint32_t VW        = 4;
        static constexpr uint32_t BlockDim  = 64;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 128>
    {
        static constexpr uint32_t VW        = 4;
        static constexpr uint32_t BlockDim  = 128;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 2 + BlockDim * 2 + start,
                    WAVE_SIZE * 2 + BlockDim * 3 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                    WAVE_SIZE * 3 + BlockDim * 2 + start,
                    WAVE_SIZE * 3 + BlockDim * 3 + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 4, 256>
    {
        static constexpr uint32_t VW        = 4;
        static constexpr uint32_t BlockDim  = 256;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 2 + BlockDim * 2 + start,
                    WAVE_SIZE * 2 + BlockDim * 3 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                    WAVE_SIZE * 3 + BlockDim * 2 + start,
                    WAVE_SIZE * 3 + BlockDim * 3 + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    BlockDim * 2 + start,
                    BlockDim * 3 + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE + BlockDim * 2 + start,
                    WAVE_SIZE + BlockDim * 3 + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 2 + BlockDim * 2 + start,
                    WAVE_SIZE * 2 + BlockDim * 3 + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                    WAVE_SIZE * 3 + BlockDim * 2 + start,
                    WAVE_SIZE * 3 + BlockDim * 3 + start,
                    WAVE_SIZE * 4 + start,
                    WAVE_SIZE * 4 + BlockDim + start,
                    WAVE_SIZE * 4 + BlockDim * 2 + start,
                    WAVE_SIZE * 4 + BlockDim * 3 + start,
                    WAVE_SIZE * 5 + start,
                    WAVE_SIZE * 5 + BlockDim + start,
                    WAVE_SIZE * 5 + BlockDim * 2 + start,
                    WAVE_SIZE * 5 + BlockDim * 3 + start,
                    WAVE_SIZE * 6 + start,
                    WAVE_SIZE * 6 + BlockDim + start,
                    WAVE_SIZE * 6 + BlockDim * 2 + start,
                    WAVE_SIZE * 6 + BlockDim * 3 + start,
                    WAVE_SIZE * 7 + start,
                    WAVE_SIZE * 7 + BlockDim + start,
                    WAVE_SIZE * 7 + BlockDim * 2 + start,
                    WAVE_SIZE * 7 + BlockDim * 3 + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 2, 16>
    {
        static constexpr uint32_t VW       = 2;
        static constexpr uint32_t BlockDim = 16;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) % BlockDim + waveOffset;

            return VecType{
                start,
                BlockDim + start,
            };
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 2, 32>
    {
        static constexpr uint32_t VW       = 2;
        static constexpr uint32_t BlockDim = 32;
        using VecType                      = VecT<DataT, VW>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / BlockDim * VW * BlockDim;
            auto const start      = (threadId % BlockDim) % BlockDim + waveOffset;

            return VecType{
                start,
                BlockDim + start,
            };
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 2, 64>
    {
        static constexpr uint32_t VW        = 2;
        static constexpr uint32_t BlockDim  = 64;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 2, 128>
    {
        static constexpr uint32_t VW        = 2;
        static constexpr uint32_t BlockDim  = 128;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT>
    struct SoaVec<DataT, 2, 256>
    {
        static constexpr uint32_t VW        = 2;
        static constexpr uint32_t BlockDim  = 256;
        static constexpr uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        static constexpr uint32_t VecSize   = VW * BlockDim / WAVE_SIZE;
        using VecType                       = VecT<DataT, VecSize>;

        ROCWMMA_DEVICE constexpr static inline auto genData()
        {
            auto const threadId   = (uint8_t)detail::threadId();
            auto const waveOffset = threadId / WAVE_SIZE * VW * BlockDim;
            auto const start      = (threadId % WAVE_SIZE) % BlockDim + waveOffset;

            if constexpr(ROCWMMA_WAVE64_MODE)
            {

                return VecType{
                    start,
                    BlockDim + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                };
            }
            else if constexpr(ROCWMMA_WAVE32_MODE)
            {
                return VecType{
                    start,
                    BlockDim + start,
                    WAVE_SIZE + start,
                    WAVE_SIZE + BlockDim + start,
                    WAVE_SIZE * 2 + start,
                    WAVE_SIZE * 2 + BlockDim + start,
                    WAVE_SIZE * 3 + start,
                    WAVE_SIZE * 3 + BlockDim + start,
                    WAVE_SIZE * 4 + start,
                    WAVE_SIZE * 4 + BlockDim + start,
                    WAVE_SIZE * 5 + start,
                    WAVE_SIZE * 5 + BlockDim + start,
                    WAVE_SIZE * 6 + start,
                    WAVE_SIZE * 6 + BlockDim + start,
                    WAVE_SIZE * 7 + start,
                    WAVE_SIZE * 7 + BlockDim + start,
                };
            }
            else
            {
                // This host code should not be called since it is marked as ROCWMMA_DEVICE
                // This code snippet exists since hipcc complains about the mismatched function
                return VecType();
            }
        }
    };

    template <typename DataT, uint32_t VW, uint32_t BlockDim>
    ROCWMMA_DEVICE static inline bool aos_soa_b32()
    {
        bool err = false;

        const uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        auto           v         = AosVec<DataT, VW, BlockDim>::genData();

        __syncthreads();

        auto soa = TransformsImpl::Ops::AosToSoa<BlockDim, VW>::exec(v);

        auto cmp_v = SoaVec<DataT, VW, BlockDim>::genData();
        err |= soa != cmp_v;

        return err;
    }

    template <typename DataT, uint32_t VW, uint32_t BlockDim>
    ROCWMMA_DEVICE static inline bool soa_aos_b32()
    {
        bool err = false;

        const uint32_t WAVE_SIZE = Constants::AMDGCN_WAVE_SIZE;
        auto           v         = SoaVec<DataT, VW, BlockDim>::genData();
        __syncthreads();

        auto aos = TransformsImpl::Ops::SoaToAos<BlockDim, VW>::exec(v);

        auto cmp_v = AosVec<DataT, VW, BlockDim>::genData();
        err |= aos != cmp_v;

        return err;
    }

    template <typename DataT, uint32_t VW, uint32_t BlockDim>
    ROCWMMA_KERNEL void aossoaTest(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        // Add tests here
        err = err ? err : aos_soa_b32<DataT, VW, BlockDim>();

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

    template <typename DataT, uint32_t VW, uint32_t BlockDim>
    ROCWMMA_KERNEL void soaaosTest(uint32_t     m,
                                   uint32_t     n,
                                   DataT const* in,
                                   DataT*       out,
                                   uint32_t     ld,
                                   DataT        param1,
                                   DataT        param2)
    {
        __shared__ int32_t result;
        result = 0;
        synchronize_workgroup();

        bool err = false;

        // Add tests here
        err = err ? err : soa_aos_b32<DataT, VW, BlockDim>();

        // Reduce error count
        atomicAdd(&result, (int32_t)err);

        // Wait for all threads
        synchronize_workgroup();

        // Just need one thread to update output
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0
           && blockIdx.y == 0 && blockIdx.z == 0)
        {
            out[0] = static_cast<DataT>(result == 0 ? SUCCESS_VALUE : ERROR_VALUE);
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_TRANSFORMS_TEST_HPP
