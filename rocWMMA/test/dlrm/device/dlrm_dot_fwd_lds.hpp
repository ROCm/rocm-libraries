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

#ifndef DLRM_DOT_FWD_LDS_HPP
#define DLRM_DOT_FWD_LDS_HPP

#include "./common.hpp"
#include "./lds_mapping_util.hpp"

namespace rocwmma
{

    template <typename DataT, uint TILE_DIM, typename LdsMapping>
    __global__ void __launch_bounds__(128, 1) dlrmDotFwdLds(const DataT* __restrict input,
                                                            DataT* __restrict output,
                                                            float32_t* acc,
                                                            uint       m,
                                                            uint       k,
                                                            uint       b,
                                                            uint       inputBatchOffset,
                                                            uint       outputBatchOffset,
                                                            uint       accBatchOffset)
    {
        using MappingA   = MappingUtil<TILE_DIM, TILE_DIM, DataT, row_major>;
        using MappingB   = MappingUtil<TILE_DIM, TILE_DIM, DataT, col_major>;
        using MappingC   = MappingUtil<TILE_DIM, TILE_DIM, DataT, row_major>;
        using MappingAcc = MappingUtil<TILE_DIM, TILE_DIM, float32_t, row_major>;

        using FragA   = fragment<matrix_a, TILE_DIM, TILE_DIM, TILE_DIM, DataT, row_major>;
        using FragB   = fragment<matrix_b, TILE_DIM, TILE_DIM, TILE_DIM, DataT, col_major>;
        using FragAcc = fragment<accumulator, TILE_DIM, TILE_DIM, TILE_DIM, float32_t>;

        // Will store to LDS as though it were a register file.
        // Rows = register count
        // Cols = unpacked register elements = 64
        // Row major to minimize bank conflicts
        using MappingLds = LdsMappingUtil<TILE_DIM,
                                          TILE_DIM,
                                          TILE_DIM,
                                          DataT,
                                          row_major,
                                          col_major,
                                          row_major,
                                          LdsMapping,
                                          1,
                                          1>;

        // Copy bottom MLP to output
        // Threads with a global index < k are responsible for copying MLP data
        auto globalThreadCoord = blockIdx.x * blockDim.x + threadIdx.x;
        auto count             = k / blockDim.x;
        count                  = (count > 1) ? count : 1;
        if(blockIdx.x == 0 && blockIdx.y == 0)
        {
            for(int i = 0; i < count; i++)
            {
                if(i * blockDim.x + globalThreadCoord < k)
                {
                    output[outputBatchOffset * blockIdx.z + i * blockDim.x + globalThreadCoord]
                        = input[inputBatchOffset * blockIdx.z + i * blockDim.x + globalThreadCoord];
                }
            }
        }

        // Target output block
        auto matrixCoordC = MappingC::matrixCoord();

        if(get<0>(matrixCoordC) < m && get<1>(matrixCoordC) < m)
        {
            // Initialize accumulator
            auto fragAcc = FragAcc();
            fill_fragment(fragAcc, static_cast<float32_t>(0));

            // Setup starting global addresses
            auto* inputWithOffset = input + inputBatchOffset * blockIdx.z;
            auto* addrA
                = MappingA::dataCoord(inputWithOffset, make_coord2d(get<0>(matrixCoordC), 0), k);
            auto* addrB
                = MappingB::dataCoord(inputWithOffset, make_coord2d(0, get<1>(matrixCoordC)), k);

            /// Setup LDS addressing and start writing pre-fetch to LDS
            HIP_DYNAMIC_SHARED(void*, localMemPtr);
            auto* ldsPtrLo = reinterpret_cast<DataT*>(localMemPtr);
            auto* ldsPtrHi = ldsPtrLo + MappingLds::ldsWidth() * MappingLds::ldsHeight();

            // Prefetch the first block from global memory
            MappingLds::prefetchGlobalA(ldsPtrLo, addrA, k);
            MappingLds::prefetchGlobalB(ldsPtrLo, addrB, k);

            // Wait for A / B write LDS
            synchronize_workgroup();

            // Setup address increments.
            // A steps BlockK through m x k
            // B steps BlockK through k x m
            auto fragA = FragA();
            auto fragB = FragB();

            auto incrA = MappingA::dataOffset(make_coord2d(0, TILE_DIM), k);
            auto incrB = MappingB::dataOffset(make_coord2d(TILE_DIM, 0), k);

            auto endA = addrA + incrA * (k / TILE_DIM);

            addrA += incrA;
            addrB += incrB;
            while(addrA != endA)
            {
                // Cache lds blocks to register
                MappingLds::prefetchLocalA(fragA, ldsPtrLo, 0);
                MappingLds::prefetchLocalB(fragB, ldsPtrLo, 0);

                // Start pulling in the next block
                MappingLds::prefetchGlobalA(ldsPtrHi, addrA, k);
                MappingLds::prefetchGlobalB(ldsPtrHi, addrB, k);

                // Mma for current block
                mma_sync(fragAcc, fragA, fragB, fragAcc);

                // Wait for A / B read LDS
                synchronize_workgroup();

                addrA += incrA;
                addrB += incrB;

                auto* tmp = ldsPtrLo;
                ldsPtrLo  = ldsPtrHi;
                ldsPtrHi  = tmp;
            }

            // Mma for the last block
            MappingLds::prefetchLocalA(fragA, ldsPtrLo, 0);
            MappingLds::prefetchLocalB(fragB, ldsPtrLo, 0);
            mma_sync(fragAcc, fragA, fragB, fragAcc);

            // Wait for final mma before writing to LDS
            synchronize_workgroup();

            // Store acc frag to lds for recasting
            auto* ldsPtrAcc     = reinterpret_cast<float32_t*>(localMemPtr);
            auto* ldsWavePtrAcc = ldsPtrAcc + MappingLds::waveOffsetA();
            store_matrix_sync(ldsWavePtrAcc, fragAcc, TILE_DIM, mem_row_major);

            // Wait for LDS write before accessing
            synchronize_workgroup();

            // Copy lower triangular from lds to output
            auto fragColIdx   = threadIdx.x % TILE_DIM;
            auto globalColIdx = get<1>(matrixCoordC) + fragColIdx;
            auto rowsPerStep  = Constants::AMDGCN_WAVE_SIZE / TILE_DIM;

            count = (TILE_DIM * TILE_DIM) >> Log2<Constants::AMDGCN_WAVE_SIZE>::value;
            for(int i = 0; i < count; i++)
            {
                auto fragRowIdx = i * rowsPerStep
                                  + ((threadIdx.x & (Constants::AMDGCN_WAVE_SIZE - 1)) / TILE_DIM);
                auto globalRowIdx = get<0>(matrixCoordC) + fragRowIdx;
                if(globalRowIdx > globalColIdx)
                {
                    auto outputOffset = k + ((globalRowIdx * (globalRowIdx - 1)) >> 1);
                    output[outputBatchOffset * blockIdx.z + outputOffset + globalColIdx]
                        = static_cast<DataT>(ldsWavePtrAcc[fragRowIdx * TILE_DIM + fragColIdx]);
                }
            }
        }
    }
} // namespace rocwmma

#endif // DLRM_DOT_FWD_LDS_HPP
