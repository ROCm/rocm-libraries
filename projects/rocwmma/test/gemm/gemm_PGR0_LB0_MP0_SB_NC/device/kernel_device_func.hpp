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

#ifndef ROCWMMA_GEMM_TEST_DEVICE_FUNC
#define ROCWMMA_GEMM_TEST_DEVICE_FUNC

// Silence warnings for calls on unsupported architectures.
// Unsupported architectures will generate no-ops and test
// will be avoided at runtime anyway.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "kernel_predicates.hpp"
#include <rocwmma/rocwmma.hpp>
#pragma GCC diagnostic pop

namespace rocwmma
{
    ///
    /// This class of kernel is a naive kernel whereas
    /// each wave is responsible for calculating a macro tile area of
    /// a single block: BlockM x BlockN
    ///
    /// Kernel behaviour is described by:
    /// PGR0 = Prefetch Global Read = 0, no prefetch
    /// LB0 = Lds Blocks = 0, no Lds usage
    /// MP0 = Mfma Priority = 0, no setprio
    /// SB = Single-block
    /// NC = Non-cooperative
    ///

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename InputT,
              typename OutputT,
              typename ComputeT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutC,
              typename LayoutD,
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t WaveSize,
              uint32_t ArchId>
    __global__ void __launch_bounds__(256) gemm_PGR0_LB0_MP0_SB_NC(uint32_t       m,
                                                                   uint32_t       n,
                                                                   uint32_t       k,
                                                                   InputT const*  a,
                                                                   InputT const*  b,
                                                                   OutputT const* c,
                                                                   OutputT*       d,
                                                                   uint32_t       lda,
                                                                   uint32_t       ldb,
                                                                   uint32_t       ldc,
                                                                   uint32_t       ldd,
                                                                   ComputeT       alpha,
                                                                   ComputeT       beta)
    {
        if constexpr(gemm_PGR0_LB0_MP0_SB_NC_guard<BlockM,
                                                   BlockN,
                                                   BlockK,
                                                   InputT,
                                                   OutputT,
                                                   ComputeT,
                                                   TBlockX,
                                                   TBlockY,
                                                   WaveSize,
                                                   ArchId>::enableBuild())
        {
            using FragA   = fragment<matrix_a, BlockM, BlockN, BlockK, InputT, LayoutA>;
            using FragB   = fragment<matrix_b, BlockM, BlockN, BlockK, InputT, LayoutB>;
            using FragC   = fragment<accumulator, BlockM, BlockN, BlockK, OutputT, LayoutC>;
            using FragAcc = fragment<accumulator, BlockM, BlockN, BlockK, ComputeT, LayoutD>;

            using MappingA = MappingUtil<BlockM, BlockK, InputT, LayoutA>;
            using MappingB = MappingUtil<BlockK, BlockN, InputT, LayoutB>;
            using MappingC = MappingUtil<BlockM, BlockN, OutputT, LayoutC>;
            using MappingD = MappingUtil<BlockM, BlockN, OutputT, LayoutD>;

            // Target C / D block on 2D grid
            auto matrixCoordC = MappingC::matrixCoord();

            if(get<0>(matrixCoordC) + BlockM > m || get<1>(matrixCoordC) + BlockN > n)
            {
                return;
            }

            if(BlockK > k)
            {
                return;
            }

            // Initialize accumulator
            auto fragAcc = FragAcc();
            fill_fragment(fragAcc, static_cast<ComputeT>(0));

            // Setup starting addresses
            // Offset A to col 0
            // Offset B to row 0
            auto* addrA = MappingA::dataCoord(a, MappingC::matrixCoordN(0), lda);
            auto* addrB = MappingB::dataCoord(b, MappingC::matrixCoordM(0), ldb);

            // Setup address increments.
            // A steps BlockK through m x k
            // B steps BlockK through k x n
            auto incrA = MappingA::dataOffset(make_coord2d(0u, BlockK), lda);
            auto incrB = MappingB::dataOffset(make_coord2d(BlockK, 0u), ldb);
            auto count = k / BlockK;

            // Accumulate A * B
            for(int i = 0; i < count; i++)
            {
                // Keeping the workgroup in sync here is not necessary for correctness.
                // HOWEVER, if we keep waves in sync chances are good we may
                // benefit from cache hits on re-used data from A and B global loads.
                synchronize_workgroup();

                auto fragA = FragA();
                auto fragB = FragB();

                // Load and multiply
                load_matrix_sync(fragA, addrA, lda);
                load_matrix_sync(fragB, addrB, ldb);
                mma_sync(fragAcc, fragA, fragB, fragAcc);

                addrA += incrA;
                addrB += incrB;
            }

            auto fragC = FragC();

            // Setup address and load C
            auto* addrC = MappingC::dataCoord(c, matrixCoordC, ldc);
            load_matrix_sync(fragC, addrC, ldc);

            // D = alpha * accumAB + beta * C
#pragma unroll
            for(int i = 0; i < fragC.num_elements; ++i)
            {
                fragC.x[i] = OutputT(alpha * ComputeT(fragAcc.x[i]) + beta * ComputeT(fragC.x[i]));
            }

            // Output addresss
            auto* addrD = MappingD::dataCoord(d, matrixCoordC, ldd);

            // Store the output
            store_matrix_sync(addrD, fragC, ldd);
        }
    }
} // namespace rocwmma

#endif // ROCWMMA_GEMM_TEST_DEVICE_FUNC
