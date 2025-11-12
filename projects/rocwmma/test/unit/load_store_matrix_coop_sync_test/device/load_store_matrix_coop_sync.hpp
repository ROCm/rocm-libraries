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

#ifndef ROCWMMA_DEVICE_LOAD_STORE_MATRIX_COOP_SYNC_HPP
#define ROCWMMA_DEVICE_LOAD_STORE_MATRIX_COOP_SYNC_HPP

#include <rocwmma/internal/mapping_util.hpp>
#include <rocwmma/rocwmma.hpp>

#include "unit_test_traits.hpp"

namespace rocwmma
{

    template <typename Scheduler>
    struct SchedulerTraits;

    template <uint32_t _TBlockX, uint32_t _TBlockY, template <uint32_t, uint32_t> class Scheduler>
    struct SchedulerTraits<Scheduler<_TBlockX, _TBlockY>>
    {
        static constexpr uint32_t TBlockX = _TBlockX;
        static constexpr uint32_t TBlockY = _TBlockY;
    };

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename Scheduler,
              typename Enabler = void>
    struct CooperativePredicates
    {
        static constexpr uint32_t BlockDim = is_same_v<MatrixT, matrix_a> ? BlockM : BlockN;
        static constexpr uint32_t BlockK   = is_same_v<MatrixT, matrix_a> ? BlockN : BlockM;

        static constexpr uint32_t TotalWaves = Scheduler::waveCount();
        static constexpr uint32_t WaveSize   = Constants::AMDGCN_WAVE_SIZE;

        // Minimum vector elements to get full registers
        static constexpr uint32_t MinElementsPerReg
            = max(1u, (uint32_t)(sizeof(float32_t) / sizeof(DataT)));

        // Cooperative check: must have a minimum of 1 full reg per wave,
        // otherwise we get comile errors
        static constexpr uint32_t Elements = BlockDim * BlockK;
        static constexpr uint32_t FullRegs = Elements / WaveSize / MinElementsPerReg;
        static constexpr bool CoopCheck = (FullRegs >= TotalWaves) && (FullRegs % TotalWaves == 0u);

        // Interleaved KPerthread must be > 0, otherwise we get compile errors
        static constexpr uint32_t MmaDim         = detail::MmaDimSelector<BlockDim, DataT>::Result;
        static constexpr uint32_t KPerThread_Int = BlockK * MmaDim / (WaveSize * TotalWaves);
        static constexpr bool     KPTCheck       = KPerThread_Int > 0u;

        static constexpr bool enable()
        {
            return CoopCheck && KPTCheck;
        }
    };

    template <typename MatrixT,
              uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename Scheduler>
    struct CooperativePredicates<
        MatrixT,
        BlockM,
        BlockN,
        DataT,
        Scheduler,
        enable_if_t<SchedulerTraits<Scheduler>::TBlockX % Constants::AMDGCN_WAVE_SIZE != 0u>>
    {
        // Don't run tests for TBlockX that are non-multiple of wave size
        static constexpr bool enable()
        {
            return false;
        }
    };

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayout,
              typename Scheduler>
    __global__ void
        LoadStoreMatrixCoopSyncA(uint32_t m, uint32_t n, DataT const* in, DataT* out, uint32_t ld)
    {
        if constexpr(FragSize_guard<BlockM,
                                    BlockN,
                                    DataT,
                                    DataLayout,
                                    Constants::AMDGCN_WAVE_SIZE,
                                    Constants::AMDGCN_CURRENT_ARCH_ID>::enable()
                     && CooperativePredicates<matrix_a, BlockM, BlockN, DataT, Scheduler>::enable())
        {
            // Mapping:
            // Incoming -> Matrix A (ColNT)
            // BlockM -> BlockM
            // <Dummy> -> BlockN
            // BlockN -> BlockK
            auto frag = fragment<matrix_a, BlockM, BlockN, BlockN, DataT, DataLayout, Scheduler>();

            using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;

            auto workgroupDim      = Mapping::workgroupDim();
            auto waveCoord         = Mapping::waveCoord();
            auto currentBlockCoord = Mapping::blockCoord();

            // Start at the first block in WG coverage
            auto startBlockCoord = currentBlockCoord - waveCoord;

            // Do cooperative loads for all blocks covered by WG
            for(int i = 0; i < get<0>(workgroupDim); i++)
            {
                for(int j = 0; j < get<1>(workgroupDim); j++)
                {
                    // Map, load and store.
                    auto  blockCoord = startBlockCoord + make_coord2d(i, j);
                    auto* read       = Mapping::dataCoord(in, Mapping::matrixCoord(blockCoord), ld);
                    auto* write = Mapping::dataCoord(out, Mapping::matrixCoord(blockCoord), ld);
                    load_matrix_sync(frag, read, ld);
                    store_matrix_sync(write, frag, ld);
                }
            }
        }
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              typename DataT,
              typename DataLayout,
              typename Scheduler>
    __global__ void
        LoadStoreMatrixCoopSyncB(uint32_t m, uint32_t n, DataT const* in, DataT* out, uint32_t ld)
    {
        if constexpr(FragSize_guard<BlockM,
                                    BlockN,
                                    DataT,
                                    DataLayout,
                                    Constants::AMDGCN_WAVE_SIZE,
                                    Constants::AMDGCN_CURRENT_ARCH_ID>::enable()
                     && CooperativePredicates<matrix_b, BlockM, BlockN, DataT, Scheduler>::enable())
        {
            // Mapping:
            // Incoming -> Matrix B (RowNT)
            // <Dummy> -> BlockM
            // BlockN -> BlockN
            // BlockM -> BlockK
            auto frag = fragment<matrix_b, BlockM, BlockN, BlockM, DataT, DataLayout, Scheduler>();

            using Mapping = MappingUtil<BlockM, BlockN, DataT, DataLayout>;

            auto workgroupDim      = Mapping::workgroupDim();
            auto waveCoord         = Mapping::waveCoord();
            auto currentBlockCoord = Mapping::blockCoord();

            // Start at the first block in WG coverage
            auto startBlockCoord = currentBlockCoord - waveCoord;

            // Do cooperative loads for all blocks covered by WG
            for(int i = 0; i < get<0>(workgroupDim); i++)
            {
                for(int j = 0; j < get<1>(workgroupDim); j++)
                {
                    // Map, load and store.
                    auto  blockCoord = startBlockCoord + make_coord2d(i, j);
                    auto* read       = Mapping::dataCoord(in, Mapping::matrixCoord(blockCoord), ld);
                    auto* write = Mapping::dataCoord(out, Mapping::matrixCoord(blockCoord), ld);
                    load_matrix_sync(frag, read, ld);
                    store_matrix_sync(write, frag, ld);
                }
            }
        }
    }

} // namespace rocwmma

#endif // ROCWMMA_DEVICE_LOAD_STORE_MATRIX_COOP_SYNC_HPP
