/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/ContractionSolution.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <random>

#include <iomanip>
#include <regex>

namespace Tensile
{
    namespace streamk
    {
        namespace math
        {
            /**
             * Performs `(n + d - 1) / d`, but is robust against the case where
             * `(n + d - 1)` would overflow.
             */
            template <typename N, typename D>
            __device__ __host__ inline constexpr N safe_ceil_div(N n, D d)
            {
                // Static cast to undo integral promotion.
                return static_cast<N>(n / d + (n % d != 0 ? 1 : 0));
            }
        } // namespace math

        constexpr size_t num_iters_per_cta(
            size_t BLK_M, size_t BLK_N, size_t BLK_K, size_t m, size_t n, size_t k, int g)
        {
            return math::safe_ceil_div(math::safe_ceil_div(m, BLK_M) * math::safe_ceil_div(n, BLK_N)
                                           * math::safe_ceil_div(k, BLK_K),
                                       g);
        }

        constexpr size_t number_of_output_tiles(size_t BLK_M, size_t BLK_N, size_t m, size_t n)
        {
            size_t m_tiles = math::safe_ceil_div(m, BLK_M);
            size_t n_tiles = math::safe_ceil_div(n, BLK_N);
            return m_tiles * n_tiles;
        }

        constexpr size_t num_fixup_peers(size_t BLK_K, size_t k, size_t iters_per_cta)
        {
            return math::safe_ceil_div(math::safe_ceil_div(k, BLK_K), iters_per_cta);
        }

        std::tuple<double, size_t, size_t> predicted_runtime(size_t BLK_M,
                                                             size_t BLK_N,
                                                             size_t BLK_K,
                                                             size_t m,
                                                             size_t n,
                                                             size_t k,
                                                             int    g,
                                                             double a,
                                                             double b,
                                                             double c,
                                                             double d)
        {
            size_t iters_per_cta = num_iters_per_cta(BLK_M, BLK_N, BLK_K, m, n, k, g);
            size_t fixup_peers   = num_fixup_peers(BLK_K, k, iters_per_cta);

            return {a + (b * (fixup_peers > 1)) + (c * iters_per_cta) + (d * (fixup_peers - 1)),
                    iters_per_cta,
                    fixup_peers};
        }

        int best_predicted_grid_size(size_t BLK_M,
                                     size_t BLK_N,
                                     size_t BLK_K,
                                     size_t m,
                                     size_t n,
                                     size_t k,
                                     int    grid_start,
                                     int    grid_end)
        {
            static const bool debug = Debug::Instance().printStreamKGridInfo();

            // Fixed overhead alpha (a), fixed-size cost incurred by
            // each work-group, e.g. the grid launch latency, the initial
            // compulsary cache misses, the cost of writing the final output tile
            // to C.
            double a = 5.04 + 8.30;

            // Beta (b) incorporates conditional costs of outputting temporary partial
            // sums for scenarios where the number of output tiles does not quantize
            // perfectly across the number of processors.
            double b = 5.47;

            // c represents instruction and stall workload of each MAC-iteration.
            double c = 4.17;

            // Delta (d) is the cost of reading and accumulating the partial sums from
            // other work-groups covering the same tile.
            double d = 18.59;

            // std::vector<double> runtimes;
            std::pair<int, double> min_grid_runtime;
            min_grid_runtime.second = std::numeric_limits<double>::max();
            int g                   = grid_start;

            // Predict the number of CTAs to use between 1 and 304
            for(; g <= grid_end; ++g)
            {
                auto [runtime, iters_per_cta, fixup_peers]
                    = predicted_runtime(BLK_M, BLK_N, BLK_K, m, n, k, g, a, b, c, d);

                if(debug)
                {
                    std::cout << "grid size: " << g << ", runtime: " << runtime
                              << ", iters_per_cta: " << iters_per_cta
                              << ", fixup_peers: " << fixup_peers << ", m: " << m << ", n: " << n
                              << ", k: " << k << ", a: " << a << ", b: " << b << ", c: " << c
                              << ", d: " << d << std::endl;
                }

                if(min_grid_runtime.second > runtime)
                {
                    min_grid_runtime.first  = g;
                    min_grid_runtime.second = runtime;
                }
            }

            if(debug)
            {
                std::cout << "Number of Output Tiles: "
                          << number_of_output_tiles(BLK_M, BLK_N, m, n) << std::endl;
                std::cout << "Minimum runtime: " << min_grid_runtime.second
                          << " @ grid size: " << min_grid_runtime.first << std::endl;
            }

            return min_grid_runtime.first;
        }
    } // namespace streamk

    PerfModel perf;

    int32_t ContractionSolution::staggerUIter(ContractionSolution::Problem const& problem,
                                              ContractionSolution::Inputs const&  inputs,
                                              Hardware const&                     hardware) const
    {
        uint32_t sizeL = problem.boundSize(0);

        // how many stride-sized clicks to stagger start offset
        unsigned int staggerUIter = sizeMapping.staggerU;

        // /DepthU/GSU
        int unrollLoopIters = sizeL / sizeMapping.depthU / sizeMapping.globalSplitU;

        unsigned int shifted = 1 << sizeMapping.staggerStrideShift;

        while(staggerUIter > 1)
        {
            if(unrollLoopIters >= (staggerUIter * shifted))
                break;

            staggerUIter /= 2; // step down to smaller stagger
        }

        if(staggerUIter >= 1)
            staggerUIter -= 1;

        return staggerUIter;
    }

    // Return magic number.  If magicShift is 0, compute and return it.
    uint32_t ContractionSolution::magicNumberAlg1(uint32_t x, uint32_t* magicShift) const
    {
        uint64_t magicNum;
        *magicShift = 33;
        magicNum    = (1L << *magicShift) / x + 1;
        if((magicNum >> 32) != 0)
        {
            *magicShift = 31;
            magicNum    = (1L << *magicShift) / x + 1;
        }

        assert(magicNum >> 32 == 0); // ensure magic number fits

        return static_cast<uint32_t>(magicNum);
    }

    uint32_t ContractionSolution::magicNumberAlg2(uint32_t d, uint32_t* magicShift) const
    {
        struct mu
        {
            unsigned M; // Magic number,
            int      a; // "add" indicator,
            int      s;
        }; // and shift amount.

        struct mu magu;
        if(d == 0)
        {
            // Make dividend of 0 return 0
            magu.M = 0;
            magu.a = 0;
            magu.s = 0;
        }
        else
        {
            // Must have 1 <= d <= 2**32-1.
            int      p;
            unsigned nc, delta, q1, r1, q2, r2;
            magu.a = 0; // Initialize "add" indicator.
            nc     = -1 - (-d) % d; // Unsigned arithmetic here.
            p      = 31; // Init. p.
            q1     = 0x80000000 / nc; // Init. q1 = 2**p/nc.
            r1     = 0x80000000 - q1 * nc; // Init. r1 = rem(2**p, nc).
            q2     = 0x7FFFFFFF / d; // Init. q2 = (2**p - 1)/d.
            r2     = 0x7FFFFFFF - q2 * d; // Init. r2 = rem(2**p - 1, d).
            do
            {
                p = p + 1;
                if(r1 >= nc - r1)
                {
                    q1 = 2 * q1 + 1; // Update q1.
                    r1 = 2 * r1 - nc;
                } // Update r1.
                else
                {
                    q1 = 2 * q1;
                    r1 = 2 * r1;
                }
                if(r2 + 1 >= d - r2)
                {
                    if(q2 >= 0x7FFFFFFF)
                        magu.a = 1;
                    q2 = 2 * q2 + 1; // Update q2.
                    r2 = 2 * r2 + 1 - d;
                } // Update r2.
                else
                {
                    if(q2 >= 0x80000000)
                        magu.a = 1;
                    q2 = 2 * q2;
                    r2 = 2 * r2 + 1;
                }
                delta = d - 1 - r2;
            } while(p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));

            magu.M = q2 + 1; // Magic number
            magu.s = p - 32; // and shift amount to return
        }

        *magicShift         = magu.s;
        const uint32_t abit = 0x80000000;
        if(magu.a)
            *magicShift |= abit;

        // std::cout << " d=" << d << " M=" << magu.M << " a=" << magu.a << " s=" <<
        // magu.s << "\n";

        return magu.M;
    }

    uint32_t
        ContractionSolution::magicNumber(int magicDivAlg, uint32_t x, uint32_t* magicShift) const
    {
        if(magicDivAlg == 1)
            return magicNumberAlg1(x, magicShift);
        else if(magicDivAlg == 2)
            return magicNumberAlg2(x, magicShift);
        else
            throw std::runtime_error("bad magicDivAlg");
    }

    uint32_t ContractionSolution::smallMagicNumber(uint32_t x) const
    {
        uint64_t  magicNum;
        const int smallMagicShift = 31;
        magicNum                  = (1L << smallMagicShift) / x + 1;
        assert(magicNum >> 32 == 0); // ensure magic number fits
        return static_cast<uint32_t>(magicNum);
    }

    std::vector<size_t> generatePackedIndicesA(ContractionSolution::Problem const& problem,
                                               size_t                              packBatchDims)
    {
        std::vector<size_t> packedIndices;

        // TODO -move packedIndices calc to problem decode.
        for(auto idx = 0; idx < problem.a().dimensions(); idx++)
        {
            bool isSum = problem.boundIndices().end()
                         != std::find_if(problem.boundIndices().begin(),
                                         problem.boundIndices().end(),
                                         [idx](const ContractionProblem::BoundIndex& bi) {
                                             return bi.a == idx;
                                         });

            bool nonPackableBatch = false;
            // TODO - base this check on if the batch is SetConstStrideA=0 - if so,
            // don't pack
            if(!(packBatchDims & 0x1))
            {
                nonPackableBatch = problem.batchIndices().end()
                                   != std::find_if(problem.batchIndices().begin(),
                                                   problem.batchIndices().end(),
                                                   [idx](const ContractionProblem::BatchIndex& bi) {
                                                       return bi.a == idx;
                                                   });
            }

            if(!isSum && !nonPackableBatch)
                packedIndices.push_back(idx);
        }

        return packedIndices;
    }

    std::vector<size_t> generatePackedIndicesB(ContractionSolution::Problem const& problem,
                                               size_t                              packBatchDims)
    {
        std::vector<size_t> packedIndices;

        // Pack in all non-summation indices, except don't need magic number for the
        // last one
        for(auto idx = 0; idx < problem.b().dimensions(); idx++)
        {
            bool isSum = problem.boundIndices().end()
                         != std::find_if(problem.boundIndices().begin(),
                                         problem.boundIndices().end(),
                                         [idx](const ContractionProblem::BoundIndex& bi) {
                                             return bi.b == idx;
                                         });

            bool nonPackableBatch = false;
            // TODO - base this check on if the batch is SetConstStrideB=0 - if so,
            // don't pack
            if(!(packBatchDims & 0x2))
            {
                nonPackableBatch = problem.batchIndices().end()
                                   != std::find_if(problem.batchIndices().begin(),
                                                   problem.batchIndices().end(),
                                                   [idx](const ContractionProblem::BatchIndex& bi) {
                                                       return bi.b == idx;
                                                   });
            }

            if(!isSum && !nonPackableBatch)
                packedIndices.push_back(idx);
        }

        return packedIndices;
    }

    template <typename TypedInputs, bool T_Debug>
    KernelInvocation
        ContractionSolution::generateSingleCall(ContractionSolution::Problem const& problem,
                                                TypedInputs const&                  inputs,
                                                Hardware const&                     hardware) const
    {
        //TENSILE_ASSERT_EXC(sizeMapping.workGroupMapping >= 0);

        TensorDescriptor const& a = problem.a();
        TensorDescriptor const& b = problem.b();
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(1024, 128);

        rv.kernelName = kernelName;

        rv.workGroupSize.x = sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y
                             * sizeMapping.workGroupSize.z;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        rv.numWorkGroups.x = 1;
        rv.numWorkGroups.y = 1;

        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
        {
            rv.numWorkGroups.x *= problem.freeSizeA(i);
        }
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
        {
            rv.numWorkGroups.y *= problem.freeSizeB(i);
        }

        rv.numWorkGroups.z = 1;
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
        {
            if(sizeMapping.packBatchDims & 0x1)
                rv.numWorkGroups.x *= problem.batchSize(i);
            if(sizeMapping.packBatchDims & 0x2)
                rv.numWorkGroups.y *= problem.batchSize(i);
            if(!sizeMapping.packBatchDims)
                rv.numWorkGroups.z *= problem.batchSize(i);
        }

        if(problem.transposeC01())
            std::swap(rv.numWorkGroups.x, rv.numWorkGroups.y);

        rv.numWorkGroups.x = CeilDivide(rv.numWorkGroups.x, sizeMapping.macroTile.x);
        rv.numWorkGroups.y = CeilDivide(rv.numWorkGroups.y, sizeMapping.macroTile.y);

        uint32_t problemNumGroupTiles0 = rv.numWorkGroups.x;
        uint32_t problemNumGroupTiles1 = rv.numWorkGroups.y;
        // used only when persistent kernel along batch
        uint32_t problemNumGroupTiles2 = rv.numWorkGroups.z;

        rv.numWorkGroups.y *= sizeMapping.globalSplitU;

        size_t cuCount   = 0;
        size_t skGrid    = 0;
        auto   tiles     = problem.getNumTiles(sizeMapping);
        int    fullTiles = 0;
        if(sizeMapping.streamK != 0 || sizeMapping.persistentKernel != 0)
        {
            AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
            assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);
            cuCount = pAMDGPU->computeUnitCount;
            if(sizeMapping.streamK != 0)
            {
                skGrid             = getSKGrid(problem, hardware, tiles);
                rv.numWorkGroups.x = skGrid;
                rv.numWorkGroups.y = 1;
                rv.numWorkGroups.z = 1;
                fullTiles          = pAMDGPU->skFullTiles;
                if(fullTiles == -1)
                {
                    fullTiles = predictNumFullTiles(problem, hardware, tiles, skGrid);
                }
            }
        }

        if(sizeMapping.persistentKernel != 0)
        {
            size_t finalPKValue  = sizeMapping.persistentKernel;
            size_t problemGroups = rv.numWorkGroups.x * rv.numWorkGroups.y;
            if(sizeMapping.persistentKernelAlongBatch)
            {
                problemGroups *= rv.numWorkGroups.z;
                rv.numWorkGroups.z = 1;
            }

            if(finalPKValue == -1)
            {
                // 1. Get the largest pk value (ex.3)
                //    which can make the PK.G (ex.3*120=360) <= problemGroups (ex.433)
                // 2. Scale by 5/8 (can try 0.5~1, to control the tiles-per-workgroup = 1~2)
                finalPKValue = 5 * (problemGroups / cuCount) / 8;
                finalPKValue = std::max(finalPKValue, (size_t)1);
                //std::cout << "final persistent kernel value: " << finalPKValue << std::endl;
            }

            size_t persistentGroups = cuCount * finalPKValue;
            rv.numWorkGroups.x      = std::min(persistentGroups, problemGroups);
            rv.numWorkGroups.y      = 1;
        }

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(debugKernel)
        {
            rv.args.appendUnbound<unsigned int*>("debugBuffer");
        }

        rv.sharedMemBytes = 0;

        size_t startStrideCD = problemType.useInitialStridesCD ? 0 : 1;
        size_t startStrideAB = problemType.useInitialStridesAB ? 0 : 1;

        auto problemSizes = problem.problemSizes();
        TENSILE_ASSERT_EXC(problemSizes.size() > 0);

        if(!isSourceKernel())
        {
            uint64_t tensor2dSizeC = c.totalAllocatedElements();
            uint64_t tensor2dSizeA = (sizeMapping.packBatchDims & 0x1)
                                         ? a.totalAllocatedElements()
                                         : problem.allocatedElementsNonBatchA();
            uint64_t tensor2dSizeB = (sizeMapping.packBatchDims & 0x2)
                                         ? b.totalAllocatedElements()
                                         : problem.allocatedElementsNonBatchB();

            if(sizeMapping.preloadKernargs)
            {
                TENSILE_ASSERT_EXC(problemType.stridedBatched);
                TENSILE_ASSERT_EXC(problemType.useBeta);
                TENSILE_ASSERT_EXC(startStrideAB == 1);

                rv.args.append<uint64_t>("tensor2dSizeA", tensor2dSizeA);
                rv.args.append<uint64_t>("tensor2dSizeB", tensor2dSizeB);

                auto aptr = inputs.a + a.offset();
                auto bptr = inputs.b + b.offset();
                rv.args.append<typename TypedInputs::AType const*>("a", aptr);
                rv.args.append<typename TypedInputs::BType const*>("b", bptr);

                rv.args.append<typename TypedInputs::AlphaType>("alpha", inputs.alpha);
                if(std::is_same<typename TypedInputs::AlphaType, Half>::value && !isSourceKernel())
                    rv.args.append<typename TypedInputs::AlphaType>("alpha_2", inputs.alpha);

                for(size_t i = startStrideAB; i < a.dimensions(); i++)
                    rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideA", i), a.strides()[i]);

                for(size_t i = startStrideAB; i < b.dimensions(); i++)
                    rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideB", i), b.strides()[i]);

                rv.args.append<uint32_t>("sizeSum", problemSizes.back());
            }

            // rv.args.append<uint64_t>("tensor2dSizeC", tensor2dSizeC);

            if(!sizeMapping.preloadKernargs)
            {
                rv.args.append<uint64_t>("tensor2dSizeA", tensor2dSizeA);
                rv.args.append<uint64_t>("tensor2dSizeB", tensor2dSizeB);
            }
        }

        if(sizeMapping.globalAccumulation && sizeMapping.streamK == 0)
        {
            rv.args.append<void const*>("ws_d", inputs.ws);
            rv.args.append<void const*>("ws_c", inputs.ws);
        }
        else if(problemType.stridedBatched)
        {
            auto dptr = inputs.d + d.offset();
            auto cptr = inputs.c + c.offset();
            rv.args.append<typename TypedInputs::DType const*>("d", dptr);
            rv.args.append<typename TypedInputs::CType const*>("c", cptr);
        }
        else
        {
            rv.args.append<typename TypedInputs::DType const* const*>("batchD", inputs.batchD);
            rv.args.append<typename TypedInputs::CType const* const*>("batchC", inputs.batchC);
        }

        if(problemType.stridedBatched)
        {
            if(!sizeMapping.preloadKernargs)
            {
                auto aptr = inputs.a + a.offset();
                auto bptr = inputs.b + b.offset();
                rv.args.append<typename TypedInputs::AType const*>("a", aptr);
                rv.args.append<typename TypedInputs::BType const*>("b", bptr);
            }
        }
        else
        {
            rv.args.append<typename TypedInputs::AType const* const*>("batchA", inputs.batchA);
            rv.args.append<typename TypedInputs::BType const* const*>("batchB", inputs.batchB);
        }

        if(sizeMapping.streamK > 0 && sizeMapping.streamKAtomic == 0)
        {
            // StreamK workspace + flags
            rv.args.append<void const*>("ws", inputs.ws);
            void*  ws          = inputs.ws;
            size_t flagsOffset = partialTileSize(skGrid);
            void*  flags       = (void*)(static_cast<char*>(ws) + flagsOffset);
            rv.args.append<void*>("Flags", flags);
        }

        if(!problemType.stridedBatched)
        {
            rv.args.append<uint64_t>("offsetD", d.offset());
            rv.args.append<uint64_t>("offsetC", c.offset());
            rv.args.append<uint64_t>("offsetA", a.offset());
            rv.args.append<uint64_t>("offsetB", b.offset());
        }

        if(!sizeMapping.preloadKernargs)
        {
            rv.args.append<typename TypedInputs::AlphaType>("alpha", inputs.alpha);
            if(std::is_same<typename TypedInputs::AlphaType, Half>::value && !isSourceKernel())
                rv.args.append<typename TypedInputs::AlphaType>("alpha_2", inputs.alpha);
        }

        if(problemType.useBeta)
        {
            rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);
            if(std::is_same<typename TypedInputs::BetaType, Half>::value && !isSourceKernel())
                rv.args.append<typename TypedInputs::BetaType>("beta_2", inputs.beta);
        }

        if(sizeMapping.globalAccumulation && sizeMapping.streamK == 0)
        {
            size_t wsStride = startStrideCD ? d.sizes()[0] : 1;
            for(size_t i = startStrideCD; i < d.dimensions(); i++)
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW_D", i), wsStride);
                wsStride *= d.sizes()[i];
            }

            wsStride = startStrideCD ? d.sizes()[0] : 1;
            for(size_t i = startStrideCD; i < c.dimensions(); i++)
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW_C", i), wsStride);
                wsStride *= d.sizes()[i];
            }
        }
        else
        {
            for(size_t i = startStrideCD; i < d.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideD", i), d.strides()[i]);

            for(size_t i = startStrideCD; i < c.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideC", i), c.strides()[i]);
        }

        if(!sizeMapping.preloadKernargs)
        {
            for(size_t i = startStrideAB; i < a.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideA", i), a.strides()[i]);

            for(size_t i = startStrideAB; i < b.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideB", i), b.strides()[i]);
        }

        {
            int maxIdx = problemSizes.size();
            if(sizeMapping.preloadKernargs)
                maxIdx--;
            for(int idx = 0; idx < maxIdx; idx++)
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("size_", idx), problemSizes[idx]);
            }
        }

        if(sizeMapping.packSummationDims)
            // boundIndices are ordered with unroll last.
            // Magic numbers for all but first are needed to unpack other dims.
            for(auto si = 1; si < problem.boundIndices().size(); si++)
            {
                auto numIter  = problem.boundSize(si);
                bool isUnroll = si == problem.boundIndices().size() - 1;
                if(isUnroll)
                {
                    numIter = numIter / sizeMapping.depthU / sizeMapping.globalSplitU
                              * sizeMapping.depthU;
                }
                uint32_t magicShift;
                rv.args.append<uint32_t>(
                    concatenate_if<T_Debug>("magicNumberNumIter_", si),
                    magicNumber(sizeMapping.magicDivAlg, numIter, &magicShift));
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicShiftNumIter_", si),
                                         magicShift);

                if(isUnroll and sizeMapping.globalSplitU > 1)
                {
                    // compute magic number for gsu remainder iterations:
                    // Kernel will select whether to use above or remainder portion based on work-group assignment
                    rv.args.append<uint32_t>(
                        concatenate_if<T_Debug>("magicNumberNumIter_GsuRemainder"),
                        magicNumber(
                            sizeMapping.magicDivAlg, numIter + sizeMapping.depthU, &magicShift));
                    rv.args.append<uint32_t>(
                        concatenate_if<T_Debug>("magicShiftNumIter_GsuRemainder"), magicShift);
                }
            }

        if(problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesA(problem, sizeMapping.packBatchDims);

            // Pack in all non-summation indices, except don't need magic number for the
            // last one
            for(auto pi = packedIndices.begin(); pi != packedIndices.end() - 1; pi++)
            {
                auto     idx  = *pi;
                auto     size = a.sizes()[idx];
                uint32_t magicShift;
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicNumberSizeA_", idx),
                                         magicNumber(sizeMapping.magicDivAlg, size, &magicShift));
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicShiftSizeA_", idx),
                                         magicShift);
            }
        }
        if(problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesB(problem, sizeMapping.packBatchDims);

            // Pack in all non-summation indices, except don't need magic number for the
            // last one
            for(auto pi = packedIndices.begin(); pi != packedIndices.end() - 1; pi++)
            {
                auto     idx  = *pi;
                auto     size = b.sizes()[idx];
                uint32_t magicShift;
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicNumberSizeB_", idx),
                                         magicNumber(sizeMapping.magicDivAlg, size, &magicShift));
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicShiftSizeB_", idx),
                                         magicShift);
            }
        }

        for(auto si : problem.boundIndices())
        {
            if(si.aZeroPad.valid())
            {
                rv.args.append<int32_t>(concatenate_if<T_Debug>("padStartA_", si.a),
                                        si.aZeroPad.padStart);
                rv.args.append<int32_t>(concatenate_if<T_Debug>("padEndA_", si.a),
                                        si.aZeroPad.padEnd);
            }
            if(si.bZeroPad.valid())
            {
                rv.args.append<int32_t>(concatenate_if<T_Debug>("padStartB_", si.b),
                                        si.bZeroPad.padStart);
                rv.args.append<int32_t>(concatenate_if<T_Debug>("padEndB_", si.b),
                                        si.bZeroPad.padEnd);
            }
        }

        if(sizeMapping.staggerU)
            rv.args.append<int32_t>("staggerUIter", staggerUIter(problem, inputs, hardware));

        rv.args.append<uint32_t>("problemNumGroupTiles0", problemNumGroupTiles0);
        rv.args.append<uint32_t>("problemNumGroupTiles1", problemNumGroupTiles1);

        if(!isSourceKernel())
        {
            uint32_t numFullBlocks            = problemNumGroupTiles1;
            uint32_t wgmRemainder1            = 0;
            uint32_t magicNumberWgmRemainder1 = 0;

            // conditional args, aligned with KernelWriterAssembly.py
            if(sizeMapping.persistentKernel != 0 || sizeMapping.streamK != 0)
            {
                uint32_t magicShift;
                rv.args.append<uint32_t>("magicNumberProblemNumGroupTiles0",
                                         magicNumber(2, problemNumGroupTiles0, &magicShift));
                rv.args.append<uint32_t>("magicShiftProblemNumGroupTiles0", magicShift);
            }

            if(sizeMapping.persistentKernel != 0)
            {
                rv.args.append<uint32_t>("gridNumWorkGroups0", rv.numWorkGroups.x);
            }

            if(sizeMapping.persistentKernelAlongBatch)
            {
                uint32_t numGroupTiles0x1 = problemNumGroupTiles0 * problemNumGroupTiles1;
                uint32_t magicShift;

                rv.args.append<uint32_t>("problemNumGroupTiles2", problemNumGroupTiles2);
                rv.args.append<uint32_t>("magicNumberProblemNumGroupTiles0By1",
                                         magicNumber(2, numGroupTiles0x1, &magicShift));
                rv.args.append<uint32_t>("magicShiftProblemNumGroupTiles0By1", magicShift);
            }

            if(sizeMapping.streamK != 0)
            {
                auto     itersPerTile = problem.getItersPerTile(sizeMapping);
                auto     totalIters   = tiles * itersPerTile;
                uint32_t magicNumberItersPerTile;
                uint32_t magicShiftItersPerTile;
                magicNumberItersPerTile = magicNumber(2, itersPerTile, &magicShiftItersPerTile);

                rv.args.append<uint32_t>("itersPerTile", itersPerTile);
                rv.args.append<uint32_t>("magicNumberItersPerTile", magicNumberItersPerTile);
                rv.args.append<uint32_t>("magicShiftItersPerTile", magicShiftItersPerTile);

                uint32_t numGroupTiles0x1 = problemNumGroupTiles0 * problemNumGroupTiles1;
                uint32_t magicNumProblemNumGroupTiles0By1;
                uint32_t magicShiftProblemNumGroupTiles0By1;
                magicNumProblemNumGroupTiles0By1
                    = magicNumber(2, numGroupTiles0x1, &magicShiftProblemNumGroupTiles0By1);
                rv.args.append<uint32_t>("magicNumProblemNumGroupTiles0By1",
                                         magicNumProblemNumGroupTiles0By1);
                rv.args.append<uint32_t>("magicShiftProblemNumGroupTiles0By1",
                                         magicShiftProblemNumGroupTiles0By1);

                rv.args.append<uint32_t>("totalIters", totalIters);
                if(sizeMapping.streamK == 1) // Basic SK
                {
                    uint32_t itersPerWave = CeilDivide(totalIters, rv.numWorkGroups.x);
                    rv.args.append<uint32_t>("SKItersPerWG", itersPerWave);
                }
                else if(sizeMapping.streamK >= 2) // Two-tile SK
                {
                    // skTiles is number of Stream-K tiles to complete
                    // Two-tile algorithm causes each WG to run an even number of Stream-K iterations,
                    // followed by an even number of data-parllel tiles.
                    // If total tiles is evenly divisble by grid size,
                    // then no Stream-K tiles are needed, all data-parallel
                    uint32_t skTiles = skGrid;
                    if(tiles % skGrid != 0)
                    {
                        // Number of SK tiles determined by performance prediction
                        // Either runs remainder tiles, or remainder + 1 full grid of tiles
                        // Can be overridden by setting TENSILE_STREAMK_FULL_TILES
                        skTiles = std::min(skGrid * fullTiles + tiles % skGrid, tiles);
                    }

                    uint32_t skItersPerWG = skTiles * itersPerTile / skGrid;
                    uint32_t skExtraIters = skTiles * itersPerTile % (skGrid);

                    rv.args.append<uint32_t>("SKItersPerWG", skItersPerWG);
                    rv.args.append<uint32_t>("skGrid", skGrid);
                    rv.args.append<uint32_t>("skTiles", skTiles);
                    rv.args.append<uint32_t>("skExtraIters", skExtraIters);
                }
            }

            if(sizeMapping.workGroupMapping != 0)
            {
                if(sizeMapping.workGroupMapping >= 0)
                {
                    numFullBlocks = problemNumGroupTiles1 / sizeMapping.workGroupMapping;
                    wgmRemainder1 = problemNumGroupTiles1 % sizeMapping.workGroupMapping;
                }
                else
                {
                    numFullBlocks = problemNumGroupTiles0 / (-sizeMapping.workGroupMapping);
                    wgmRemainder1 = problemNumGroupTiles0 % (-sizeMapping.workGroupMapping);
                }
                if(wgmRemainder1 == 0)
                    wgmRemainder1 = sizeMapping.workGroupMapping;
                magicNumberWgmRemainder1 = smallMagicNumber(wgmRemainder1);
            }

            if(std::abs(sizeMapping.workGroupMapping) > 1)
            {
                rv.args.append<uint32_t>("numFullBlocks", numFullBlocks);
                rv.args.append<uint32_t>("wgmRemainder1", wgmRemainder1);
                rv.args.append<uint32_t>("magicNumberWgmRemainder1", magicNumberWgmRemainder1);
            }
        }

        if(problemType.stochasticRounding)
        {
            // generate seed from random generator
            std::random_device                      rd;
            std::mt19937                            gen(rd());
            std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
            uint32_t                                seed = distribution(gen);
            //rv.args.append<uint32_t>("RNDSeed", 0x0);
            rv.args.append<uint32_t>("RNDSeed", seed);
        }

        if(!isSourceKernel())
        {
            rv.args.append<uint32_t>("pad", 0);
        }

        rv.codeObjectFile = codeObjectFilename.load();

        return rv;
    }

    bool ContractionSolution::isSourceKernel() const
    {
        return sizeMapping.sourceKernel;
    }

    template <typename TypedInputs, bool T_Debug>
    KernelInvocation ContractionSolution::generateStreamKInitCall(Problem const&     problem,
                                                                  TypedInputs const& inputs,
                                                                  Hardware const&    hardware) const
    {
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        rv.kernelName = streamKInitKernelName(problem, inputs, hardware);

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        auto   tiles  = problem.getNumTiles(sizeMapping);
        size_t skGrid = getSKGrid(problem, hardware, tiles);
        size_t wiZ    = 1;
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
            wiZ *= problem.batchSize(i);
        size_t flagCount = skGrid * wiZ;

        rv.numWorkGroups.x = CeilDivide(flagCount, rv.workGroupSize.x);
        rv.numWorkGroups.y = 1;
        rv.numWorkGroups.z = 1;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        void*  ws          = inputs.ws;
        size_t flagsOffset = partialTileSize(skGrid);
        void*  flags       = (void*)(static_cast<char*>(ws) + flagsOffset);
        rv.args.append<void*>("Flags", flags);

        rv.args.append<uint32_t>("flagCount", flagCount);

        //Pass along code object dependency
        // TODO check this
        rv.codeObjectFile = codeObjectFilename.load();

        return rv;
    }

    template <typename TypedInputs>
    std::string ContractionSolution::streamKInitKernelName(Problem const&     problem,
                                                           TypedInputs const& inputs,
                                                           Hardware const&    hardware) const
    {
        std::string name = "WSFlags";
        return name;
    }

    template <typename TypedInputs, bool T_Debug>
    KernelInvocation ContractionSolution::generateBetaOnlyCall(Problem const&     problem,
                                                               TypedInputs const& inputs,
                                                               Hardware const&    hardware) const
    {
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        rv.kernelName = betaOnlyKernelName(problem, inputs, hardware);

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        size_t wiX = 1;
        size_t wiY = 1;
        size_t wiZ = 1;
        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            wiX *= problem.freeSizeA(i);
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            wiY *= problem.freeSizeB(i);
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
            wiZ *= problem.batchSize(i);

        rv.numWorkGroups.x = CeilDivide(wiX * wiY * wiZ, rv.workGroupSize.x);
        rv.numWorkGroups.y = 1;
        rv.numWorkGroups.z = 1;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(sizeMapping.globalAccumulation)
            rv.args.append<void*>("WS", inputs.ws);
        else if(problemType.stridedBatched)
        {
            auto dptr = inputs.d;
            dptr += d.offset();
            rv.args.append<typename TypedInputs::DType*>("D", dptr);
        }
        else
            rv.args.append<typename TypedInputs::DType const* const*>("batchD", inputs.batchD);

        if(problemType.stridedBatched)
        {
            auto cptr = inputs.c;
            cptr += c.offset();
            rv.args.append<typename TypedInputs::CType const*>("C", cptr);
        }
        else
            rv.args.append<typename TypedInputs::CType const* const*>("batchC", inputs.batchC);

        if(!problemType.stridedBatched)
        {
            rv.args.append<uint64_t>("offsetD", d.offset());
            rv.args.append<uint64_t>("offsetC", c.offset());
        }

        if(sizeMapping.globalAccumulation)
        {
            size_t stride = d.sizes()[0];
            for(size_t i = 1; i < d.dimensions(); i++)
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW", i),
                                         d.sizes()[i] == 1 ? 0 : stride);
                stride *= d.sizes()[i];
            }
        }
        else
        {
            for(size_t i = 1; i < d.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideD", i),
                                         d.sizes()[i] == 1 ? 0 : d.strides()[i]);
        }

        for(size_t i = 1; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideC", i),
                                     c.sizes()[i] == 1 ? 0 : c.strides()[i]);

        int idx = 0;
        for(auto size : problem.d().sizes())
        {
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("size_", idx), size);
            idx++;
        }

        rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);

        //Pass along code object dependency
        rv.codeObjectFile = codeObjectFilename.load();

        return rv;
    }

    template <typename TypedInputs>
    std::string ContractionSolution::betaOnlyKernelName(Problem const&     problem,
                                                        TypedInputs const& inputs,
                                                        Hardware const&    hardware) const
    {
        std::string name = concatenate(
            "C", problem.cNames(), "_", TypeInfo<typename TypedInputs::DType>::Abbrev());

        if(!problemType.stridedBatched)
        {
            name += "_GB";
        }

        if(sizeMapping.globalAccumulation)
        {
            name += "_GA";
        }

        return name;
    }

    template <typename TypedInputs, bool T_Debug>
    KernelInvocation ContractionSolution::generateOutputConversionCall(
        Problem const& problem, TypedInputs const& inputs, Hardware const& hardware) const
    {
        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);
        size_t cuCount = pAMDGPU->computeUnitCount;

        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        size_t wiX = 1;
        size_t wiY = 1;
        size_t wiZ = 1;
        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            wiX *= problem.freeSizeA(i);
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            wiY *= problem.freeSizeB(i);
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
            wiZ *= problem.batchSize(i);

        const unsigned int numThreadsPerCU = 256;
        unsigned int       gsu             = static_cast<int>(sizeMapping.globalSplitU);
        if(sizeMapping.globalAccumulation == 1)
            // globalAccumulation = 1 case, ignore globalSplitU and use 1
            gsu = 1;
        // wider global load for postGSU
        // only for compute type = Float
        bool supportedTypeForVWopt
            = problem.alphaType() == DataType::Float || problem.alphaType() == DataType::Double;
        size_t total = wiX * wiY * wiZ;
        int    vw    = 1;
        size_t threshVW2
            = cuCount * numThreadsPerCU * 2; // should be more than number of physical threads * vw
        if(supportedTypeForVWopt && total > threshVW2 && problem.freeSizeA(0) % 2 == 0)
            vw = 2;

        // parallel reduction width calculation
        // only for compute type = Float
        bool supportedTypeForReductionOpt
            = problem.alphaType() == DataType::Float || problem.alphaType() == DataType::Double;
        size_t threshReduction
            = cuCount
              * numThreadsPerCU; // should be less than number of physical threads / reduction
        const unsigned int maxReductionConst = 4;
        const unsigned int minGSUperReduction
            = 32; // Minimum GSU=128 for Reduction=4, GSU=64 for Reduction2
        unsigned int maxReduction = std::min(maxReductionConst, gsu / minGSUperReduction);
        unsigned int reduction    = 1;
        if(supportedTypeForReductionOpt && (gsu & (gsu - 1)) == 0 && maxReduction > 1)
        {
            // apply reduction only if float compute type and gsu is power of 2 (for small array only)
            reduction = maxReduction;
            while(reduction > 1)
            {
                size_t totalThread = total * reduction / vw;
                if(gsu % reduction == 0 && totalThread <= threshReduction)
                    // found an applicable reduction
                    break;
                // for next loop
                reduction /= 2;
            }
            if(reduction == 2 && problem.alphaType() == DataType::Float)
                // so far, reduction=2 does not perform well
                reduction = 1;
        }

        // GSU loop unroll opt
        // only for compute type = Float, Double
        bool supportedTypeForUnrollOpt
            = problem.alphaType() == DataType::Float || problem.alphaType() == DataType::Double;
        int gsuUnrollUnit = 16 * reduction; // must match Tensile generator code
        if(supportedTypeForUnrollOpt == false || sizeMapping.globalAccumulation == 1)
            gsuUnrollUnit = 1;

        rv.kernelName = outputConversionKernelName(
            problem, inputs, gsu, vw, reduction, gsuUnrollUnit, hardware);

        rv.numWorkGroups.x = CeilDivide(wiX * wiY * wiZ * reduction, rv.workGroupSize.x * vw);
        rv.numWorkGroups.y = 1;
        rv.numWorkGroups.z = 1;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(problemType.stridedBatched)
        {
            auto dptr = inputs.d;
            dptr += d.offset();
            rv.args.append<typename TypedInputs::DType*>("D", dptr);
        }
        else
            rv.args.append<typename TypedInputs::DType const* const*>("batchD", inputs.batchD);

        rv.args.append<void*>("WS", inputs.ws);

        if(problemType.stridedBatched)
        {
            auto cptr = inputs.c;
            cptr += c.offset();
            rv.args.append<typename TypedInputs::CType const*>("C", cptr);
        }
        else
            rv.args.append<typename TypedInputs::CType const* const*>("batchC", inputs.batchC);

        if(!problemType.stridedBatched)
        {
            rv.args.append<uint64_t>("offsetD", d.offset());
            rv.args.append<uint64_t>("offsetC", c.offset());
        }

        if(sizeMapping.globalAccumulation == 2)
            rv.args.append<typename TypedInputs::AlphaType>("alpha", inputs.alpha);
        else
            rv.args.append<typename TypedInputs::AlphaType>("alpha", 1.0f);

        if(sizeMapping.globalAccumulation == 2 and problemType.useBeta)
            rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);
        else
            rv.args.append<typename TypedInputs::BetaType>("beta", 0.0f);

        for(size_t i = 1; i < d.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideD", i), d.strides()[i]);

        uint32_t wsStride = d.sizes()[0];
        for(size_t i = 1; i < d.dimensions(); i++)
        {
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW", i), wsStride);
            wsStride *= d.sizes()[i];
        }

        for(size_t i = 1; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideC", i), c.strides()[i]);

        int idx = 0;
        for(auto size : problem.d().sizes())
        {
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("size_", idx), size);
            idx++;
        }

        rv.args.append<uint32_t>("gsu", gsu);

        if(problemType.stochasticRounding)
        {
            // generate seed from random generator
            std::random_device                      rd;
            std::mt19937                            gen(rd());
            std::uniform_int_distribution<uint32_t> distribution(0, 0xFFFFFFFF);
            uint32_t                                seed = distribution(gen);
            //rv.args.append<uint32_t>("RNDSeed", 0x0);
            rv.args.append<uint32_t>("RNDSeed", seed);
        }

        //@TODO determine if this is needed, may not end up in the same code object file
        rv.codeObjectFile = codeObjectFilename.load();

        return rv;
    }

    bool ContractionSolution::canSolve(Problem const& problem, Hardware const& hardware) const
    {
        static const bool debug = Debug::Instance().printPredicateEvaluation();

        Task task(hardware, problem, *this);
        if(debug)
        {
            hardwarePredicate->debugEval(hardware, std::cout);
            problemPredicate->debugEval(problem, std::cout);
            taskPredicate->debugEval(task, std::cout);
        }
        return (*taskPredicate)(task) && (*problemPredicate)(problem)
               && (*hardwarePredicate)(hardware);
    }

    bool ContractionSolution::matchesProblemType(Problem const&  problem,
                                                 Hardware const& hardware) const
    {
        bool matchesHardware = (*hardwarePredicate)(hardware);
        bool matchesType
            = (problemType.aType == problem.a().dataType())
              && (problemType.bType == problem.b().dataType())
              && (problemType.cType == problem.c().dataType())
              && (problemType.dType == problem.d().dataType())
              && (problemType.highPrecisionAccumulate == problem.highPrecisionAccumulate())
              && (problemType.stridedBatched == problem.stridedBatched())
              && (problemType.fp16AltImpl == problem.fp16AltImpl())
              && (problemType.fp16AltImplRound == problem.fp16AltImplRound());
        return matchesHardware && matchesType;
    }

    template <typename TypedInputs>
    std::string ContractionSolution::outputConversionKernelName(Problem const&     problem,
                                                                TypedInputs const& inputs,
                                                                int                gsu,
                                                                int                vw,
                                                                int                reduction,
                                                                int                gsuUnrollUnit,
                                                                Hardware const&    hardware) const
    {
        std::string name = concatenate(
            "C", problem.cNames(), "_", TypeInfo<typename TypedInputs::DType>::Abbrev());

        if(!problemType.stridedBatched)
        {
            name += "_GB";
        }

        name += "_PostGSU";

        // add extra string for gsu
        // This part must match tensile code generation (in KernelWriterConversion.py)
        // add mod related str (only for gsuUnrollUnit > 1)
        if(gsuUnrollUnit > 1)
        {
            size_t gsuMod = gsu % gsuUnrollUnit;
            if(gsuMod == 0)
                gsuMod = gsuUnrollUnit;
            std::string modStr = "";
            if(gsu > gsuUnrollUnit)
            {
                modStr = "_mod" + std::to_string(gsuUnrollUnit);
            }
            name += std::to_string(gsuMod) + modStr;
        }
        if(vw >= 2)
            name += "_VW" + std::to_string(vw);

        if(reduction >= 2)
            name += "_R" + std::to_string(reduction);

        return name;
    }

    template <typename TypedInputs>
    std::vector<KernelInvocation> ContractionSolution::solveTyped(Problem const&     problem,
                                                                  TypedInputs const& inputs,
                                                                  Hardware const&    hardware) const
    {
        bool debug         = Debug::Instance().printKernelArguments() || this->kernelArgsLog;
        bool debugSkipInit = Debug::Instance().skipInitKernelLaunch();

        int boundSize = 1;
        for(size_t i = 0; i < problem.boundIndices().size(); i++)
            boundSize *= problem.boundSize(i);

        // Check for nullptrs if alpha is non-zero.
        if(((inputs.alpha != static_cast<typename TypedInputs::AlphaType>(0)) && (boundSize != 0))
           && ((problem.stridedBatched() && (inputs.a == nullptr || inputs.b == nullptr))
               || (!problem.stridedBatched()
                   && (inputs.batchA == nullptr || inputs.batchB == nullptr))))
        {
            std::string matrixID = inputs.a == nullptr ? "A" : "B";
            std::string msg      = std::string("Unsupported nullptr for ") + matrixID
                              + std::string(" when (Alpha !=0) && (K != 0)\n");
            throw std::runtime_error(msg.c_str());
        }

        // Check if alpha matches problem definition
        if(problem.alphaRestriction() != ScalarValue::Any
           && problem.alphaRestriction() != toScalarValueEnum(inputs.alpha))
        {
            std::stringstream inputValue;
            inputValue << inputs.alpha;
            std::string msg = std::string("Alpha value ") + inputValue.str()
                              + std::string(" doesn't match that set in problem: ")
                              + ToString(problem.alphaRestriction());
            throw std::runtime_error(msg.c_str());
        }

        // Check if beta matches problem definition
        if(problem.betaRestriction() != ScalarValue::Any
           && problem.betaRestriction() != toScalarValueEnum(inputs.beta))
        {
            std::stringstream inputValue;
            inputValue << inputs.beta;
            std::string msg = std::string("Beta value ") + inputValue.str()
                              + std::string(" doesn't match that set in problem: ")
                              + ToString(problem.betaRestriction());
            throw std::runtime_error(msg.c_str());
        }

        if(problem.cEqualsD() && inputs.c != inputs.d)
            throw std::runtime_error(
                "ContractionProblem has cEqualsD set, but pointers for c and d are not equal");

        std::vector<KernelInvocation> rv;

        if(sizeMapping.streamK > 0 && sizeMapping.streamKAtomic == 0 && !debugSkipInit)
        {
            auto   tiles  = problem.getNumTiles(sizeMapping);
            size_t skGrid = getSKGrid(problem, hardware, tiles);
            // If problem can run full data-parallel, flags are not needed since there is no fixup step
            // Skipping init kernel when flags are not needed saves significant time for small problems
            if(tiles % skGrid != 0)
            {
                if(debug)
                    rv.push_back(
                        generateStreamKInitCall<TypedInputs, true>(problem, inputs, hardware));
                else
                    rv.push_back(
                        generateStreamKInitCall<TypedInputs, false>(problem, inputs, hardware));
            }
        }

        if((sizeMapping.streamK > 0 && sizeMapping.streamKAtomic == 1)
           || (sizeMapping.globalSplitU > 1 && sizeMapping.globalAccumulation != 2))
        {
            if(debug)
                rv.push_back(generateBetaOnlyCall<TypedInputs, true>(problem, inputs, hardware));
            else
                rv.push_back(generateBetaOnlyCall<TypedInputs, false>(problem, inputs, hardware));
        }

        if(debug)
            rv.push_back(generateSingleCall<TypedInputs, true>(problem, inputs, hardware));
        else
            rv.push_back(generateSingleCall<TypedInputs, false>(problem, inputs, hardware));

        if(sizeMapping.globalAccumulation && !sizeMapping.streamK)
        {
            // TODO Streamk May need conversion call for HPA??
            if(debug)
                rv.push_back(
                    generateOutputConversionCall<TypedInputs, true>(problem, inputs, hardware));
            else
                rv.push_back(
                    generateOutputConversionCall<TypedInputs, false>(problem, inputs, hardware));
        }

        return rv;
    }

    bool ContractionSolution::getMatrixInstructionFromKernelName(
        vector4<std::uint32_t>& matrixInst) const
    {

        std::string regexp_string("_MI(\\d+)x(\\d+)x(\\d+)x(\\d+)_");
        std::regex  miRegex(regexp_string);
        std::smatch matches;

        std::string kName = this->KernelName();

        bool matched
            = std::regex_search(kName, matches, miRegex)
              && (matches.size()
                  == 5 /* 1 (index 0) for entire match and 4 for the sub-experssion matches */);

        if(matched)
        {
            // the first sub_match element (index 0) corresponds to the entire match
            matrixInst.x = atoi(matches[1].str().c_str());
            matrixInst.y = atoi(matches[2].str().c_str());
            matrixInst.z = atoi(matches[3].str().c_str());
            matrixInst.w = atoi(matches[4].str().c_str());
        }

        return matched;
    }

    bool ContractionSolution::getGSUAlgorithmFromKernelName(std::string& gsuAlg) const
    {
        std::string regexp_string("_GSUA([S|M])B_");
        std::regex  miRegex(regexp_string);
        std::smatch matches;

        std::string kName = this->KernelName();

        bool matched
            = std::regex_search(kName, matches, miRegex)
              && (matches.size()
                  == 2 /* 1 (index 0) for entire match and 1 for the sub-experssion matches */);

        if(matched)
        {
            // the first sub_match element (index 0) corresponds to the entire match
            gsuAlg = (matches[1].compare("S") == 0) ? "SingleBuffer" : "MultipleBuffer";
        }

        return matched;
    }

    std::vector<KernelInvocation>
        ContractionSolution::solve(ContractionSolution::Problem const& problem,
                                   ContractionSolution::Inputs const&  inputs,
                                   Hardware const&                     hardware) const
    {
        if(Debug::Instance().printWinningKernelName())
            std::cout << "Running kernel: " << this->KernelName() << std::endl;

        if(Debug::Instance().printKernelCommonParams())
        {
            std::cout << "Kernel name: " << this->KernelName() << std::endl;
            std::cout << "Kernel parameters: " << std::endl;

            vector4<std::uint32_t> matrixInst;
            if(this->getMatrixInstructionFromKernelName(matrixInst))
                std::cout << std::right << std::setw(30) << "MatrixInstruction: " << matrixInst
                          << std::endl;

            std::string GSUAlg;
            if(this->getGSUAlgorithmFromKernelName(GSUAlg))
                std::cout << std::right << std::setw(30) << "GSUAlgorithm: " << GSUAlg << std::endl;

            std::cout << this->sizeMapping << std::endl;
        }
        // retreive alpha/beta type set via setAlpha/BetaType()
        auto alphaType = problem.alphaType();
        auto betaType  = problem.betaType();

        // TODO: Some gtests are passing the "problem" without actually defining the
        // alpha/beta type (alphaType and betaType remain None).
        // Until we fix those gtests, we need to keep this condition to adjust the missing
        // alpha/beta data types.
        if(alphaType == DataType::None)
        {
            alphaType
                = problemType.aType == DataType::BFloat16 ? DataType::Float : problemType.dType;
        }
        if(betaType == DataType::None)
        {
            betaType = alphaType;
        }

        auto contractionInputsTypeId = ContractionInputs::TypeId(problemType.aType,
                                                                 problemType.bType,
                                                                 problemType.cType,
                                                                 problemType.dType,
                                                                 alphaType,
                                                                 betaType);

        switch(contractionInputsTypeId)
        {
        case ContractionInputs_S_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_S_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_D_D_D::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_D_D_D const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_C_C_C::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_C_C_C const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_Z_Z_Z::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_Z_Z_Z const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#ifdef TENSILE_USE_HALF
        case ContractionInputs_H_H_H::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_H_H_H const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_H_H_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_H_H_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_H_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_H_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#endif // TENSILE_USE_HALF
        case ContractionInputs_I8x4_I32_I32::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_I8x4_I32_I32 const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_I32_I32_I32::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_I32_I32_I32 const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_I8_I32_I32::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_I8_I32_I32 const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#ifdef TENSILE_USE_BF16
        case ContractionInputs_B_B_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B_B_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_B_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#endif // TENSILE_USE_BF16
#ifdef TENSILE_USE_FP8_BF8
        case ContractionInputs_F8_F8_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_F8_F8_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_F8_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_F8_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_B8_B8_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B8_B8_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_B8_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B8_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        // hybrid cases: F8B8SS, B8F8SS
        case ContractionInputs_F8B8_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_F8B8_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_B8F8_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B8F8_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        // hybrid cases with To = B8
        case ContractionInputs_F8B8_B8_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_F8B8_B8_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_B8F8_B8_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B8F8_B8_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
            // hybrid with F16 output
        case ContractionInputs_F8_H_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_F8_H_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_B8_H_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B8_H_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_F8B8_H_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_F8B8_H_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_B8F8_H_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B8F8_H_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#endif // TENSILE_USE_FP8_BF8

        default:;
        }
        throw std::runtime_error("Data type not implemented.");
    }

    ContractionSolution::StaticPerformanceModel
        ContractionSolution::staticPerformanceModel(double M,
                                                    double N,
                                                    double K,
                                                    double NumBatches,
                                                    double MT0,
                                                    double MT1,
                                                    double NumCUs,
                                                    double TotalGranularity,
                                                    int    GlobalSplitU) const
    {
        StaticPerformanceModel spm;

        int beta      = (int)problemType.useBeta;
        int betaReads = 0, betaWrites = 0;
        if(GlobalSplitU == 1)
        {
            if(beta != 0.0)
                betaReads = 1.0;
        }
        else
        {
            if(beta == 0)
                betaWrites = 1; // zero output
            else if(beta != 1.0) // if 1.0, just atomic update output
            {
                // if not 1.0, read, scale, write, then atomic update in kernel
                betaReads  = 1; // initial read for scale
                betaWrites = 1; // writeback after scale
            }
        }

        auto aInfo = DataTypeInfo::Get(problemType.aType);
        auto bInfo = DataTypeInfo::Get(problemType.bType);
        auto cInfo = DataTypeInfo::Get(problemType.cType);
        auto dInfo = DataTypeInfo::Get(problemType.dType);

        spm.memReadBytesA = (NumBatches * M * N * K) / MT1 * aInfo.elementSize;
        spm.memReadBytesB = (NumBatches * M * N * K) / MT0 * bInfo.elementSize;
        spm.memReadBytesC = (NumBatches * M * N) * betaReads * cInfo.elementSize;

        if(GlobalSplitU == 1)
            spm.memWriteBytesD = (NumBatches * M * N) * (1 + betaWrites) * dInfo.elementSize;
        else
        {
            bool   hardwareAtomic   = false; // TODO-model
            double atomicOperations = hardwareAtomic ? 2 : 3; // read-mod-write or cas  //TODO-model
            double atomicCollisions = 1.0; // TODO-could be based on K, GSU
            spm.memWriteBytesD      = (NumBatches * M * N)
                                 * (betaWrites + atomicOperations * atomicCollisions)
                                 * dInfo.elementSize;
        }
        spm.memReadBytes   = spm.memReadBytesA + spm.memReadBytesB + spm.memReadBytesC;
        spm.memGlobalReads = spm.memReadBytesA / aInfo.elementSize
                             + spm.memReadBytesB / bInfo.elementSize
                             + spm.memReadBytesC / cInfo.elementSize;
        spm.memGlobalWrites = spm.memWriteBytesD / dInfo.elementSize;

        return spm;
    }

    size_t ContractionSolution::requiredWorkspaceSize(Problem const&  problem,
                                                      Hardware const& hardware) const
    {
        size_t size = 0;

        if(sizeMapping.streamK > 0 && sizeMapping.streamKAtomic == 0)
        {
            auto   tiles  = problem.getNumTiles(sizeMapping);
            size_t skGrid = getSKGrid(problem, hardware, tiles);
            // Get space required for partial tiles
            size += partialTileSize(skGrid);
            // Add space for flags
            // Flags for partial tiles - dword per flag for fast addressing and comparisons
            // If tiles is evenly divided by grid size flags are not needed (DP mode)
            if(tiles % skGrid != 0)
                size += skGrid * 4;
            // size *= batches; // TODO need tile and flag per batch
        }
        else
            size += problem.d().totalLogicalElements() * sizeMapping.workspaceSizePerElemC;

        return size;
    }

    float lerp(float x, float x0, float x1, float y0, float y1)
    {
        return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
    }

    int ContractionSolution::predictNumFullTiles(Problem const&  problem,
                                                 Hardware const& hardware,
                                                 size_t          tiles,
                                                 size_t          skGrid) const
    {
        int itersPerTile = problem.getItersPerTile(sizeMapping);
        int k            = itersPerTile * sizeMapping.depthU;
        // TODO Test if prediction should be normalized to iterations (DepthU) or full size

        auto remainder = tiles % skGrid;
        if(remainder == 0)
            return 0;

        // Temporary check for correct hardware until prediction curves are made for additional devices
        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);
        size_t cuCount = pAMDGPU->computeUnitCount;
        if(cuCount != 304)
            return 0;

        // TODO Generalize code to allow prediction models for additional architectures
        const std::vector<std::tuple<size_t, size_t>> predictionCurve = {{512, 304},
                                                                         {768, 201},
                                                                         {1024, 75},
                                                                         {1536, 40},
                                                                         {2048, 30},
                                                                         {3072, 21},
                                                                         {4096, 14},
                                                                         {5120, 11},
                                                                         {6144, 9},
                                                                         {7168, 8},
                                                                         {8192, 5},
                                                                         {16384, 4}};

        if(k < std::get<0>(predictionCurve[0]))
            return 1;

        for(int i = 1; i < predictionCurve.size(); ++i)
        {
            if(k < std::get<0>(predictionCurve[i]))
            {
                float threshold = lerp(k,
                                       std::get<0>(predictionCurve[i - 1]),
                                       std::get<0>(predictionCurve[i]),
                                       std::get<1>(predictionCurve[i - 1]),
                                       std::get<1>(predictionCurve[i]));
                return remainder >= threshold ? 0 : 1;
            }
        }

        return remainder >= std::get<1>(predictionCurve.back()) ? 0 : 1;
    }

    size_t ContractionSolution::getSKGrid(Problem const&  problem,
                                          Hardware const& hardware,
                                          size_t          tiles) const
    {
        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);

        assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);
        size_t cuCount = pAMDGPU->computeUnitCount;

        // User-specified grid size for Stream-K kernel.
        if(pAMDGPU->skFixedGrid > 0)
        {
            return pAMDGPU->skFixedGrid;
        }

        // Dynamically pick the minimum between the cuCount or number of tiles.
        else if(pAMDGPU->skDynamicGrid == 1)
        {
            return min(cuCount, tiles);
        }

        // Dynamically pick the minimum between the cuCount or number of tiles,
        // and scale down really large sizes to use fewer CUs for power/energy savings.
        else if(pAMDGPU->skDynamicGrid == 2)
        {
            size_t skGrid = cuCount;
            if(tiles > skGrid)
            {
                for(size_t i = 1; i <= 32; i *= 2)
                {
                    size_t tilesPerCU  = CeilDivide(i * tiles, cuCount);
                    size_t reducedGrid = CeilDivide(i * tiles, tilesPerCU);
                    float  utilization = ((float)reducedGrid) / ((float)cuCount);
                    if(utilization > 0.75f)
                    {
                        if(utilization < 1.0f)
                            skGrid = reducedGrid;
                        break;
                    }
                }
            }

            return min(skGrid, tiles);
        }

        // Dynamically predict the best grid-size by weighing the cost of the fix-up
        // step and the cost of processing MAC-loop instructions. When the cost of fix-up
        // is the bottleneck, use smaller grid size.
        // Architecture dependent.
        else if(pAMDGPU->skDynamicGrid == 3)
        {
            size_t x = 1;
            size_t y = 1;
            size_t z = 1;
            for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            {
                x *= problem.freeSizeA(i);
            }
            for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            {
                y *= problem.freeSizeB(i);
            }
            // TODO Batch dimension
            for(size_t i = 0; i < problem.boundIndices().size(); ++i)
            {
                z *= problem.boundSize(i);
            }

            return streamk::best_predicted_grid_size(sizeMapping.macroTile.x,
                                                     sizeMapping.macroTile.y,
                                                     sizeMapping.depthU,
                                                     x,
                                                     y,
                                                     z,
                                                     1,
                                                     cuCount);
        }

        // Limit the CUs Stream-K is launched on either max or the specified,
        // whichever is minimum.
        else if(pAMDGPU->skMaxCUs > 0)
        {
            return min(cuCount, static_cast<size_t>(pAMDGPU->skMaxCUs));
        }

        // Multiply the cuCount with a constant factor (c), and launch
        // c * cuCount number of workgroups for Stream-K.
        else if(pAMDGPU->skGridMultiplier > 1)
        {
            return cuCount * pAMDGPU->skGridMultiplier;
        }

        // If no option is specified, launch exactly cuCount worth of workgroups.
        else
        {
            return cuCount;
        }
    }

    size_t ContractionSolution::partialTileSize(size_t skGrid) const
    {
        size_t size = 0;

        size_t tileSize
            = sizeMapping.macroTile.x * sizeMapping.macroTile.y * sizeMapping.workspaceSizePerElemC;
        size += tileSize * skGrid; // Partials tile per WG
        // TODO batches
        // TODO round up for alignment?

        return size;
    }

    float ContractionSolution::computeGranularity(float x)
    {
        return x / ceil(x);
    }

    ContractionSolution::Granularities ContractionSolution::computeGranularities(
        Hardware const& hardware, double M, double N, double K, double NumBatches) const
    {
        ContractionSolution::Granularities granularities;

        double MT0 = sizeMapping.macroTile.x;
        double MT1 = sizeMapping.macroTile.y;

        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        assert(pAMDGPU);

        double NumCUs        = pAMDGPU->computeUnitCount;
        double wavefrontSize = pAMDGPU->wavefrontSize;
        double simdPerCu     = pAMDGPU->simdPerCu;

        double GlobalSplitU = sizeMapping.globalSplitU;
        double LocalSplitU  = sizeMapping.workGroupSize.z;

        granularities.MT0 = MT0;
        granularities.MT1 = MT1;
        granularities.GSU = GlobalSplitU;
        granularities.LSU = LocalSplitU;
        granularities.CUs = NumCUs;

        granularities.numTiles0 = M / MT0;
        granularities.numTiles1 = N / MT1;

        granularities.tile0Granularity = computeGranularity(granularities.numTiles0);
        granularities.tile1Granularity = computeGranularity(granularities.numTiles1);

        granularities.tilesPerCu
            = (NumBatches * ceil(granularities.numTiles0) * ceil(granularities.numTiles1))
              / (NumCUs / GlobalSplitU / LocalSplitU);

        granularities.totalTiles    = ceil(granularities.numTiles0) * ceil(granularities.numTiles1);
        granularities.natTilesPerCu = NumBatches * granularities.totalTiles / NumCUs;
        granularities.suTilesPerCu  = (granularities.totalTiles * GlobalSplitU) / NumCUs;
        granularities.suCuGranularity = computeGranularity(granularities.suTilesPerCu);

        granularities.waveGranularity = std::min(
            1.00,
            static_cast<double>(floor(granularities.tilesPerCu + 1.0) * sizeMapping.workGroupSize.x
                                * sizeMapping.workGroupSize.y * sizeMapping.workGroupSize.z)
                / pAMDGPU->wavefrontSize / pAMDGPU->simdPerCu);

        granularities.waves
            = ceil((sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y) / wavefrontSize);

        granularities.suWavesPerSimdx2
            = (granularities.suTilesPerCu * granularities.waves) / (2 * simdPerCu);
        granularities.suWaveGranularity
            = granularities.suWavesPerSimdx2 * ceil(granularities.suWavesPerSimdx2);

        double nat_tiles_per_cu
            = NumBatches * ceil(granularities.numTiles0) * ceil(granularities.numTiles1) / NumCUs;
        granularities.natCuGranularity = ceil(nat_tiles_per_cu) * ceil(nat_tiles_per_cu) / NumCUs;

        granularities.cuGranularity = computeGranularity(granularities.tilesPerCu);

        granularities.totalGranularity
            = granularities.tile0Granularity * granularities.tile1Granularity
              * granularities.cuGranularity * granularities.waveGranularity;

        granularities.totalTileAwareGranularity
            = granularities.tile0Granularity * granularities.tile1Granularity
              * granularities.suCuGranularity * granularities.suWaveGranularity;

        return granularities;
    }

    ContractionSolution::ProjectedPerformance
        ContractionSolution::projectedPerformance(Problem const&  problem,
                                                  Hardware const& hardware) const
    {
        ProjectedPerformance pp;

        double M = 1.0, N = 1.0;
        if(problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesA(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                M *= problem.a().sizes()[*pi];
        }
        else
            M = problem.freeSizeA(0);

        if(problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesB(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                N *= problem.b().sizes()[*pi];
        }
        else
            N = problem.freeSizeB(0);

        double NumBatches = 1;
        if(sizeMapping.packBatchDims == 0)
        {
            for(size_t i = 0; i < problem.batchIndices().size(); i++)
                NumBatches *= problem.batchSize(i);
        }
        double K = problem.boundSize(0); // TODO - fix for multiple summations

        pp.granularities = ContractionSolution::computeGranularities(hardware, M, N, K, NumBatches);

        auto it = ideals.begin();

        int    closestKMeasure     = std::numeric_limits<int>::max();
        double closestKPerformance = 0.0;

        while(it != ideals.end())
        {
            int myK       = it->first;
            int myMeasure = std::abs(myK - K);
            if(myMeasure < closestKMeasure)
            {
                closestKMeasure     = myMeasure;
                closestKPerformance = it->second;
            }
            it++;
        }

        double MT0    = pp.granularities.MT0;
        double MT1    = pp.granularities.MT1;
        double NumCUs = pp.granularities.CUs;

        double GlobalSplitU         = pp.granularities.GSU;
        double IdealGranularityPerf = closestKPerformance;

        pp.staticModel = staticPerformanceModel(
            M, N, K, NumBatches, MT0, MT1, NumCUs, pp.granularities.totalGranularity, GlobalSplitU);

        pp.speedGFlops = IdealGranularityPerf * pp.granularities.totalGranularity;
        pp.CUs         = NumCUs;

        return pp;
    }

    ContractionSolution::TAMetricProblemScore ContractionSolution::computeProblemScore(
        Hardware const& hardware, double M, double N, double K, double NumBatches) const
    {
        ContractionSolution::TAMetricProblemScore pp;
        pp.granularites = ContractionSolution::computeGranularities(hardware, M, N, K, NumBatches);

        pp.M = M;
        pp.N = N;
        pp.K = K;

        double slope     = linearModel.slope;
        double intercept = linearModel.intercept;
        double perf_max  = linearModel.max;

        double sum_value        = K;
        double sum_perf0        = sum_value / (intercept + (slope * sum_value));
        pp.summationPerformance = 1000.0 * sum_perf0 / perf_max;

        return pp;
    }

    double ContractionSolution::computeTileAwareMetric(
        ContractionSolution::TAMetricProblemScore pp,
        ContractionSolution::TAMetricProblemScore ppReference) const
    {
        double tile0GranularityDim = abs(log(ppReference.granularites.tile0Granularity)
                                         - log(pp.granularites.tile0Granularity));
        double metric              = tile0GranularityDim;

        double tile1GranularityDim = abs(log(ppReference.granularites.tile1Granularity)
                                         - log(pp.granularites.tile1Granularity));
        metric += tile1GranularityDim;

        double natCuGranularityDim = abs(log(ppReference.granularites.natCuGranularity)
                                         - log(pp.granularites.natCuGranularity));
        metric += natCuGranularityDim;

        double suCuGranularityDim = abs(log(ppReference.granularites.suCuGranularity)
                                        - log(pp.granularites.suCuGranularity));
        metric += suCuGranularityDim;

        double suWaveGranularityDim = abs(log(ppReference.granularites.suWaveGranularity)
                                          - log(pp.granularites.suWaveGranularity));
        metric += suWaveGranularityDim;

        double natTilesPerCuDim
            = abs(log(ppReference.granularites.natTilesPerCu) - log(pp.granularites.natTilesPerCu));
        metric += natTilesPerCuDim;

        double suTilesPerCuDim
            = abs(log(ppReference.granularites.suTilesPerCu) - log(pp.granularites.suTilesPerCu));
        metric += suTilesPerCuDim;

        double summationPerformanceDim
            = abs(ppReference.summationPerformance - pp.summationPerformance);
        metric += summationPerformanceDim;

        return metric;
    }

    double ContractionSolution::computeTAMScore(Problem const&  problem,
                                                Hardware const& hardware,
                                                double          model_M,
                                                double          model_N,
                                                double          model_K,
                                                double          model_NumBatches) const
    {
        double M = 1.0, N = 1.0;
        if(problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesA(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                M *= problem.a().sizes()[*pi];
        }
        else
            M = problem.freeSizeA(0);

        if(problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesB(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                N *= problem.b().sizes()[*pi];
        }
        else
            N = problem.freeSizeB(0);

        double NumBatches = 1;
        if(sizeMapping.packBatchDims == 0)
        {
            for(size_t i = 0; i < problem.batchIndices().size(); i++)
                NumBatches *= problem.batchSize(i);
        }
        double K = problem.boundSize(0); // TODO - fix for multiple summations

        ContractionSolution::TAMetricProblemScore pp
            = computeProblemScore(hardware, M, N, K, NumBatches);

        ContractionSolution::TAMetricProblemScore ppReference
            = computeProblemScore(hardware, model_M, model_N, model_K, model_NumBatches);

        double distance = computeTileAwareMetric(pp, ppReference);

        return distance;
    }

    std::ostream& operator<<(std::ostream&                                      stream,
                             ContractionSolution::StaticPerformanceModel const& spm)
    {
        return stream << " memReadBytesA=" << spm.memReadBytesA
                      << " memReadBytesB=" << spm.memReadBytesB
                      << " memReadBytesC=" << spm.memReadBytesC
                      << " memWriteBytesD=" << spm.memWriteBytesD;
    }

    std::ostream& operator<<(std::ostream&                                    stream,
                             ContractionSolution::ProjectedPerformance const& pp)
    {
        return stream << " numTiles0=" << pp.granularities.numTiles0
                      << " numTiles1=" << pp.granularities.numTiles1
                      << " tilesPerCu=" << pp.granularities.tilesPerCu

                      << " totalGranularity=" << pp.granularities.totalGranularity
                      << " tile0Granularity=" << pp.granularities.tile0Granularity
                      << " tile1Granularity=" << pp.granularities.tile1Granularity
                      << " cuGranularity=" << pp.granularities.cuGranularity
                      << " waveGranularity=" << pp.granularities.waveGranularity

                      << " speedGFlops=" << pp.speedGFlops

                      << " staticModel=[ " << pp.staticModel << " ]";
    }

    std::ostream& operator<<(std::ostream& stream, BufferLoadCheckPacket const& st)
    {
        return stream << " shiftPtrElemA=" << st.shiftPtrElemA
                      << " shiftPtrElemB=" << st.shiftPtrElemB << " depthUorMT0=" << st.depthUorMT0
                      << " depthUorMT1=" << st.depthUorMT1;
    }

    std::ostream& operator<<(std::ostream& stream, const SizeMapping& sizeMapping)
    {
        std::ios_base::fmtflags flags(stream.flags());

        stream << std::right << std::setw(30) << "workGroupSize: " << sizeMapping.workGroupSize
               << std::endl
               << std::setw(30) << "threadTile: " << sizeMapping.threadTile << std::endl
               << std::setw(30) << "macroTile: " << sizeMapping.macroTile << std::endl

               << std::setw(30) << "staggerU: " << sizeMapping.staggerU << std::endl
               << std::setw(30) << "depthU: " << sizeMapping.depthU << std::endl
               << std::setw(30) << "globalSplitU: " << sizeMapping.globalSplitU << std::endl
               << std::setw(30) << "staggerStrideShift: " << sizeMapping.staggerStrideShift
               << std::endl
               << std::setw(30) << "workGroupMapping: " << sizeMapping.workGroupMapping << std::endl

               << std::setw(30) << "packBatchDims: " << sizeMapping.packBatchDims << std::endl
               << std::setw(30) << "packSummationDims: " << sizeMapping.packSummationDims
               << std::endl
               << std::setw(30) << "magicDivAlg: " << sizeMapping.magicDivAlg << std::endl
               << std::setw(30) << "streamK: " << sizeMapping.streamK << std::endl
               << std::setw(30) << "streamKAtomic: " << sizeMapping.streamKAtomic << std::endl
               << std::setw(30) << "persistentKernel: " << sizeMapping.persistentKernel << std::endl
               << std::setw(30)
               << "persistentKernelAlongBatch: " << sizeMapping.persistentKernelAlongBatch
               << std::endl

               << std::setw(30) << "sourceKernel: " << sizeMapping.sourceKernel << std::endl
               << std::setw(30) << "globalAccumulation: " << sizeMapping.globalAccumulation
               << std::endl
               << std::setw(30) << "workspaceSizePerElemC: " << sizeMapping.workspaceSizePerElemC
               << std::endl;

        stream.flags(flags);

        return stream;
    }
} // namespace Tensile
