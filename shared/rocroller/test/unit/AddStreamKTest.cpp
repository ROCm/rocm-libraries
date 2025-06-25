/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;

namespace AddStreamKTest
{

    void computeReference(std::function<void(uint, uint, uint, uint)> f,
                          uint                                        numTileM,
                          uint                                        numTileN,
                          uint                                        numTileK,
                          uint                                        numWGs,
                          uint                                        numTilesSK,
                          uint                                        numTilesDP)
    {
        std::map<std::tuple<uint, uint, uint>, uint> coverage;

        auto numSKTilesPerWG = (numTilesSK + numWGs - 1) / numWGs;
        auto numDPTilesPerWG = numTilesDP / numWGs;

        for(uint wg = 0; wg < numWGs; wg++)
        {
            uint forTileIdx, forKIdx;

            forKIdx = 0;
            for(forTileIdx = 0; (forTileIdx < numSKTilesPerWG)
                                && ((numSKTilesPerWG * wg + forTileIdx) < numTilesSK);
                forTileIdx += forKIdx)
            {
                uint tile = numSKTilesPerWG * wg + forTileIdx;

                auto nextNonAccum = ((tile) / numTileK + 1) * numTileK;
                auto lastTile     = std::min(nextNonAccum, numTilesSK);
                for(forKIdx = 0; (numSKTilesPerWG * wg + forTileIdx + forKIdx < lastTile)
                                 && (forTileIdx + forKIdx < numSKTilesPerWG);
                    forKIdx += 1)
                {
                    uint tile = numSKTilesPerWG * wg + forTileIdx + forKIdx;

                    uint m = (tile / numTileK) / numTileN;
                    uint n = (tile / numTileK) % numTileN;
                    uint k = tile % numTileK;

                    coverage[{m, n, k}]++;
                    f(m, n, k, wg);
                    //referenceResult[m * numTileN * numTileK + n * numTileK + k] = wg;
                }
            }

            for(forTileIdx = 0; forTileIdx < numDPTilesPerWG; forTileIdx += forKIdx)
            {
                for(forKIdx = 0; forKIdx < numTileK; forKIdx += 1)
                {
                    uint tile = numTilesSK + numDPTilesPerWG * wg + forTileIdx + forKIdx;

                    uint m = (tile / numTileK) / numTileN;
                    uint n = (tile / numTileK) % numTileN;
                    uint k = tile % numTileK;

                    coverage[{m, n, k}]++;
                    f(m, n, k, wg);
                    //referenceResult[m * numTileN * numTileK + n * numTileK + k] = wg;
                }
            }
        }

        for(uint m = 0; m < numTileM; ++m)
            for(uint n = 0; n < numTileN; ++n)
                for(uint k = 0; k < numTileK; ++k)
                    ASSERT_EQ((coverage[{m, n, k}]), 1);
    }

    struct GPU_AddStreamKTest : public GPUContextFixtureParam<bool>
    {
    protected:
        void SetUp() override
        {
            m_kernelOptions->assertWaitCntState = false;

            GPUContextFixtureParam<bool>::SetUp();
        }
    };

    //
    // Coordinate graph looks like:
    //
    //                                                     ForLoop
    //                                                        |
    //                                                        v
    //     MacroTileNumber(M)    MacroTileNumber(N)    MacroTilenumber(K)
    //            \                     |                     /
    //             \                    v                    /
    //              \--------------> Flatten <--------------/
    //                                  |
    //                                  v
    //                                 User
    //
    // Note the M/N tile numbers are dangling.  The AddStreamK pass
    // will find them and flatten them along with the K tile number.
    //
    // Control graph looks like:
    //
    // - Kernel
    //   - WaitZero
    //   - ForLoop
    //     - Assign(VGPR, WG-number)
    //     - StoreVGPR
    //     - WaitZero
    //
    // Launch with:
    // - workgroup count equal to number of CUs on GPU
    // - a single thread per workgroup
    //
    // Result will be: an M * N * K length array representing the
    // flattened tile-space.  The {m, n, k} entry in the array will be
    // the number of the WG that processed it.
    TEST_P(GPU_AddStreamKTest, GPU_BasicStreamKStore)
    {
        bool twoTile = std::get<1>(GetParam());

        rocRoller::KernelGraph::KernelGraph kgraph;

        uint numTileM = 57;
        uint numTileN = 57;
        uint numTileK = 57;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        uint numWGs = deviceProperties.multiProcessorCount;

        auto k = m_context->kernel();

        k->addArgument(
            {"result", {DataType::Int64, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(1);
        k->setWorkitemCount(
            {Expression::literal(numWGs), Expression::literal(1u), Expression::literal(1u)});
        k->setWorkgroupSize({1, 1, 1});

        // global result
        auto kernel = kgraph.control.addElement(Kernel());
        auto [forKCoord, forKOp]
            = rangeFor(kgraph, Expression::literal(numTileK), rocRoller::KLOOP);

        auto user = kgraph.coordinates.addElement(User({}, "result"));

        auto tileM = kgraph.coordinates.addElement(
            MacroTileNumber(0, Expression::literal(numTileM), nullptr));
        auto tileN = kgraph.coordinates.addElement(
            MacroTileNumber(1, Expression::literal(numTileN), nullptr));
        auto tileK = kgraph.coordinates.addElement(
            MacroTileNumber(-1, Expression::literal(numTileK), nullptr));

        kgraph.coordinates.addElement(PassThrough(), {forKCoord}, {tileK});
        kgraph.coordinates.addElement(Flatten(), {tileM, tileN, tileK}, {user});

        // result
        auto dstVGPR = kgraph.coordinates.addElement(VGPR());
        auto wgDim   = kgraph.coordinates.addElement(Workgroup(0));
        auto wgExpr  = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{wgDim, Register::Type::Vector, DataType::UInt32});
        auto assignWGNumber = kgraph.control.addElement(Assign{Register::Type::Vector, wgExpr});
        kgraph.mapper.connect(assignWGNumber, dstVGPR, NaryArgument::DEST);
        // auto assignWGNumber = kgraph.control.addElement(NOP{});

        kgraph.coordinates.addElement(PassThrough(), {user}, {dstVGPR});

        auto storeOp = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeOp, user);
        kgraph.mapper.connect<VGPR>(storeOp, dstVGPR);

        auto preWaitOp  = kgraph.control.addElement(WaitZero());
        auto loopWaitOp = kgraph.control.addElement(WaitZero());

        kgraph.control.addElement(Body(), {kernel}, {preWaitOp});
        kgraph.control.addElement(Sequence(), {preWaitOp}, {forKOp});
        kgraph.control.addElement(Body(), {forKOp}, {assignWGNumber});
        kgraph.control.addElement(Sequence(), {assignWGNumber}, {storeOp});
        kgraph.control.addElement(Sequence(), {storeOp}, {loopWaitOp});

        auto addStreamK = std::make_shared<AddStreamK>(std::vector<int>{0, 1},
                                                       rocRoller::KLOOP,
                                                       rocRoller::KLOOP,
                                                       twoTile,
                                                       Expression::literal(numWGs),
                                                       nullptr,
                                                       m_context);
        kgraph          = kgraph.transform(addStreamK);
        auto kg2        = std::make_shared<rocRoller::KernelGraph::KernelGraph>(kgraph);
        k->setKernelGraphMeta(kg2);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        m_context->schedule(rocRoller::KernelGraph::generate(*kg2, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(isLocalDevice())
        {
            auto deviceResult = make_shared_device<int>(numTileM * numTileN * numTileK);

            uint numTilesSK, numTilesDP;
            if(twoTile)
            {

                numTilesSK = numTileK * ((numTileM * numTileN) % numWGs + numWGs);
                numTilesDP = numTileM * numTileN * numTileK - numTilesSK;
            }
            else
            {
                numTilesSK = numTileM * numTileN * numTileK;
                numTilesDP = 0;
            }

            KernelArguments kargs(false);
            kargs.append("result", deviceResult.get());

            auto assemblyArgs = m_context->kernel()->arguments();
            for(int i = 1; i < assemblyArgs.size(); i++)
                kargs.append(assemblyArgs[i].name, evaluate(assemblyArgs[i].expression));

            KernelInvocation kinv;
            kinv.workitemCount = {numWGs, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<int> hostResult(numTileM * numTileN * numTileK);
            ASSERT_THAT(hipMemcpy(hostResult.data(),
                                  deviceResult.get(),
                                  hostResult.size() * sizeof(int),
                                  hipMemcpyDefault),
                        HasHipSuccess(0));

            // Compute reference...
            std::vector<int> referenceResult(numTileM * numTileN * numTileK);
            computeReference(
                [&](uint m, uint n, uint k, uint wg) {
                    referenceResult[m * numTileN * numTileK + n * numTileK + k] = wg;
                },
                numTileM,
                numTileN,
                numTileK,
                numWGs,
                numTilesSK,
                numTilesDP);

            EXPECT_EQ(hostResult, referenceResult);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    //
    // Coordinate graph (for the load) looks like:
    //
    //                                 User
    //                                  |
    //                                  v
    //              /---------------- Tile -----------------\
    //             /                    |                    \
    //            v                     v                     v
    //     MacroTileNumber(M)    MacroTileNumber(N)    MacroTilenumber(K)
    //                                                        |
    //                                                        v
    //                                                     ForLoop
    //
    // Note the M/N tile numbers are dangling.  The AddStreamK pass
    // will find them and flatten them along with the K tile number.
    //
    // Control graph looks like:
    //
    // - Kernel
    //   - a = Assign(VGPR, 0)
    //   - ForLoop
    //     - x = LoadVGPR
    //     - Assign(VGPR, a + x)
    //   - StoreVGPR(a)
    //
    // Launch with:
    // - workgroup count equal to number of CUs on GPU
    // - a single thread per workgroup
    //
    // Input is an M * N * K length array of floating point values.
    //
    // Result will be: a numWG length array.  The n'th WG will sum
    // the input values in it's local tile-space and store the result
    // in the n'th entry of the output array.
    TEST_P(GPU_AddStreamKTest, GPU_BasicStreamKLoad)
    {
        bool twoTile = std::get<1>(GetParam());
        if(twoTile)
        {
            // TODO: Re-work this to accumulate all K so that twoTile works.
            GTEST_SKIP();
        }

        rocRoller::KernelGraph::KernelGraph kgraph;

        uint numTileM = 10;
        uint numTileN = 12;
        uint numTileK = 512;

        hipDeviceProp_t deviceProperties;
        ASSERT_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        uint numWGs = deviceProperties.multiProcessorCount;

        auto k = m_context->kernel();

        k->addArgument(
            {"in", {DataType::Float, PointerType::PointerGlobal}, DataDirection::ReadOnly});
        k->addArgument(
            {"out", {DataType::Float, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(1);
        k->setWorkitemCount(
            {Expression::literal(numWGs), Expression::literal(1u), Expression::literal(1u)});
        k->setWorkgroupSize({1, 1, 1});

        // global result
        auto kernel = kgraph.control.addElement(Kernel());
        auto [forKCoord, forKOp]
            = rangeFor(kgraph, Expression::literal(numTileK), rocRoller::KLOOP);

        auto in  = kgraph.coordinates.addElement(User({}, "in"));
        auto out = kgraph.coordinates.addElement(User({}, "out"));

        auto tileM = kgraph.coordinates.addElement(
            MacroTileNumber(0, Expression::literal(numTileM), nullptr));
        auto tileN = kgraph.coordinates.addElement(
            MacroTileNumber(1, Expression::literal(numTileN), nullptr));
        auto tileK = kgraph.coordinates.addElement(
            MacroTileNumber(-1, Expression::literal(numTileK), nullptr));

        kgraph.coordinates.addElement(PassThrough(), {tileK}, {forKCoord});
        kgraph.coordinates.addElement(Tile(), {in}, {tileM, tileN, tileK});

        auto wg = kgraph.coordinates.addElement(Workgroup(0));
        kgraph.coordinates.addElement(PassThrough(), {wg}, {out});

        auto DF = [](int tag) {
            return std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{tag, Register::Type::Vector, DataType::Float});
        };

        // result
        auto aVGPR = kgraph.coordinates.addElement(VGPR());
        auto xVGPR = kgraph.coordinates.addElement(VGPR());

        auto initOp
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(0.f)});
        kgraph.mapper.connect(initOp, aVGPR, NaryArgument::DEST);

        auto addOp
            = kgraph.control.addElement(Assign{Register::Type::Vector, DF(aVGPR) + DF(xVGPR)});
        kgraph.mapper.connect(addOp, aVGPR, NaryArgument::DEST);

        auto loadOp = kgraph.control.addElement(LoadVGPR(DataType::Float));
        kgraph.mapper.connect<User>(loadOp, in);
        kgraph.mapper.connect<VGPR>(loadOp, xVGPR);

        auto storeOp = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeOp, out);
        kgraph.mapper.connect<VGPR>(storeOp, aVGPR);

        auto preWaitOp  = kgraph.control.addElement(WaitZero());
        auto loopWaitOp = kgraph.control.addElement(WaitZero());

        kgraph.control.addElement(Body(), {kernel}, {preWaitOp});
        kgraph.control.addElement(Sequence(), {preWaitOp}, {initOp});
        kgraph.control.addElement(Sequence(), {initOp}, {forKOp});
        kgraph.control.addElement(Body(), {forKOp}, {loadOp});
        kgraph.control.addElement(Sequence(), {loadOp}, {loopWaitOp});
        kgraph.control.addElement(Sequence(), {loopWaitOp}, {addOp});
        kgraph.control.addElement(Sequence(), {forKOp}, {storeOp});

        auto addStreamK = std::make_shared<AddStreamK>(std::vector<int>{0, 1},
                                                       rocRoller::KLOOP,
                                                       rocRoller::KLOOP,
                                                       false,
                                                       Expression::literal(numWGs),
                                                       nullptr,
                                                       m_context);
        kgraph          = kgraph.transform(addStreamK);
        auto kg2        = std::make_shared<rocRoller::KernelGraph::KernelGraph>(kgraph);
        k->setKernelGraphMeta(kg2);

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        m_context->schedule(rocRoller::KernelGraph::generate(*kg2, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(isLocalDevice())
        {
            RandomGenerator random(17629u);
            auto            hostX = random.vector<float>(numTileM * numTileN * numTileK, -1.0, 1.0);
            auto            deviceX = make_shared_device(hostX);
            auto            deviceA = make_shared_device<float>(numWGs);

            KernelArguments kargs(false);
            kargs.append("in", deviceX.get());
            kargs.append("out", deviceA.get());

            auto assemblyArgs = m_context->kernel()->arguments();
            for(int i = 2; i < assemblyArgs.size(); i++)
                kargs.append(assemblyArgs[i].name, evaluate(assemblyArgs[i].expression));

            KernelInvocation kinv;
            kinv.workitemCount = {numWGs, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<float> hostA(numWGs);
            ASSERT_THAT(
                hipMemcpy(
                    hostA.data(), deviceA.get(), hostA.size() * sizeof(float), hipMemcpyDefault),
                HasHipSuccess(0));

            // Compute reference...
            std::vector<float> referenceA(numWGs);
            {
                auto totalTiles = numTileM * numTileN * numTileK;
                auto tilesPerWG = (totalTiles + numWGs - 1) / numWGs;

                for(uint wg = 0; wg < numWGs; wg++)
                {
                    uint forTileIdx, forKIdx;

                    forKIdx = 0;
                    for(forTileIdx = 0;
                        (forTileIdx < tilesPerWG) && ((tilesPerWG * wg + forTileIdx) < totalTiles);
                        forTileIdx += forKIdx)
                    {
                        uint tile;

                        tile = tilesPerWG * wg + forTileIdx;

                        auto startMN = tile / numTileK;
                        for(forKIdx = 0;
                            (((tilesPerWG * wg + forTileIdx + forKIdx) / numTileK) == startMN)
                            && (tilesPerWG * wg + forTileIdx + forKIdx < totalTiles)
                            && (forTileIdx + forKIdx < tilesPerWG);
                            forKIdx += 1)
                        {
                            tile = tilesPerWG * wg + forTileIdx + forKIdx;

                            uint m, n, k;
                            m = (tile / numTileK) / numTileN;
                            n = (tile / numTileK) % numTileN;
                            k = tile % numTileK;

                            referenceA[wg] += hostX[m * numTileN * numTileK + n * numTileK + k];
                        }
                    }
                }
            }

            auto tol = AcceptableError{epsilon<double>(), "Should be exact."};
            auto res = compare(hostA, referenceA, tol);
            EXPECT_TRUE(res.ok) << res.message();
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    INSTANTIATE_TEST_SUITE_P(GPU_AddStreamKTest,
                             GPU_AddStreamKTest,
                             ::testing::Combine(currentGPUISA(), ::testing::Values(true, false)));

}
