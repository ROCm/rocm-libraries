
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include "GPUContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;

namespace AddStreamKTest
{

    class AddStreamKTestGPU : public CurrentGPUContextFixture
    {
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
    TEST_F(AddStreamKTestGPU, GPU_BasicStreamKStore)
    {

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

        auto user = kgraph.coordinates.addElement(User("result"));

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
                                                       Expression::literal(numWGs),
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

            KernelArguments kargs(false);
            kargs.append("result", deviceResult.get());
            kargs.append("numWGs", numWGs);
            kargs.append("numTiles0", numTileM);
            kargs.append("numTiles1", numTileN);
            kargs.append("numTilesAcc", numTileK);
            kargs.append("numTilesPerWG", (numTileM * numTileN * numTileK + numWGs - 1) / numWGs);

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

                            referenceResult[m * numTileN * numTileK + n * numTileK + k] = wg;
                        }
                    }
                }
            }

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
    // Result will be: a numCU length array.  The n'th WG will sum
    // the input values in it's local tile-space and store the result
    // in the n'th entry of the output array.
    TEST_F(AddStreamKTestGPU, GPU_BasicStreamKLoad)
    {
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

        auto in  = kgraph.coordinates.addElement(User("in"));
        auto out = kgraph.coordinates.addElement(User("out"));

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
                                                       Expression::literal(numWGs),
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
            kargs.append("numWGs", numWGs);
            kargs.append("numTiles0", numTileM);
            kargs.append("numTiles1", numTileN);
            kargs.append("numTilesAcc", numTileK);
            kargs.append("numTilesPerWG", (numTileM * numTileN * numTileK + numWGs - 1) / numWGs);

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

            double rnorm = relativeNorm(hostA, referenceA);
            ASSERT_LT(rnorm, 1.e-12);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

}
