// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "CustomMatchers.hpp"
#include "TestContext.hpp"

#include <common/CommonGraphs.hpp>
#include <common/Utilities.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelOptions.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;

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
        for(forTileIdx = 0;
            (forTileIdx < numSKTilesPerWG) && ((numSKTilesPerWG * wg + forTileIdx) < numTilesSK);
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
            }
        }
    }

    for(uint m = 0; m < numTileM; ++m)
        for(uint n = 0; n < numTileN; ++n)
            for(uint k = 0; k < numTileK; ++k)
                REQUIRE(coverage[{m, n, k}] == 1);
}

void applyGraphTransforms(rocRoller::KernelGraph::KernelGraph& kgraph,
                          rocRoller::CommandParametersPtr      params,
                          rocRoller::ContextPtr                context,
                          rocRoller::Expression::ExpressionPtr numWGsExpr)
{
    using namespace rocRoller::KernelGraph;
    std::vector<GraphTransformPtr> transforms;
    transforms.push_back(std::make_shared<IdentifyParallelDimensions>());
    transforms.push_back(std::make_shared<OrderMemory>(false));
    transforms.push_back(std::make_shared<UpdateParameters>(params));
    transforms.push_back(std::make_shared<AddLDS>(params, context));
    transforms.push_back(std::make_shared<LowerLinear>(context));
    transforms.push_back(std::make_shared<LowerTile>(params, context));
    transforms.push_back(std::make_shared<LowerTensorContraction>(params, context));
    transforms.push_back(std::make_shared<Simplify>());
    transforms.push_back(std::make_shared<FuseExpressions>());
    transforms.push_back(std::make_shared<AddStreamK>(
        context, params, rocRoller::XLOOP, rocRoller::KLOOP, numWGsExpr));
    transforms.push_back(std::make_shared<ConnectWorkgroups>(context));

    for(auto& t : transforms)
        kgraph = kgraph.transform(t);
}

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
TEST_CASE("AddStreamK BasicStreamKStore", "[streamk][kernel-graph][gpu]")
{
    bool twoTile = GENERATE(false, true);

    SECTION(twoTile ? "TwoTile mode" : "Standard mode")
    {
        rocRoller::KernelOptions kernelOpts;
        kernelOpts->assertWaitCntState = false;

        auto context = TestContext::ForTestDevice(kernelOpts, twoTile ? "TwoTile" : "Standard");

        rocRoller::KernelGraph::KernelGraph kgraph;

        uint numTileM = 57;
        uint numTileN = 57;
        uint numTileK = 57;

        hipDeviceProp_t deviceProperties;
        REQUIRE_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        uint numWGs = deviceProperties.multiProcessorCount;

        auto k = context->kernel();

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

        CommandParametersPtr params           = std::make_shared<CommandParameters>();
        params->loopOverOutputTilesDimensions = {0, 1};
        params->streamK = twoTile ? StreamKMode::TwoTile : StreamKMode::Standard;

        auto addStreamK = std::make_shared<AddStreamK>(
            context.get(), params, rocRoller::KLOOP, rocRoller::KLOOP, Expression::literal(numWGs));

        kgraph = kgraph.transform(addStreamK);
        if(context->kernelOptions()->removeSetCoordinate)
            kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        auto kg2 = std::make_shared<rocRoller::KernelGraph::KernelGraph>(kgraph);
        k->setKernelGraphMeta(kg2);

        context->schedule(k->preamble());
        context->schedule(k->prolog());

        context->schedule(rocRoller::KernelGraph::generate(*kg2, context->kernel()));

        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());

        if(context->hipDeviceIndex() >= 0)
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

            auto assemblyArgs = context->kernel()->arguments();
            for(int i = 1; i < assemblyArgs.size(); i++)
                kargs.append(assemblyArgs[i].getName(), evaluate(assemblyArgs[i].getExpression()));

            KernelInvocation kinv;
            kinv.workitemCount = {numWGs, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<int> hostResult(numTileM * numTileN * numTileK);
            REQUIRE_THAT(hipMemcpy(hostResult.data(),
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

            CHECK(hostResult == referenceResult);
        }
        else
        {
            std::vector<char> assembledKernel = context->instructions()->assemble();
            CHECK(assembledKernel.size() > 0);
        }
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
TEST_CASE("AddStreamK BasicStreamKLoad", "[streamk][kernel-graph][gpu]")
{
    // TODO: Re-work this to accumulate all K so that twoTile works.
    bool twoTile = GENERATE(false);

    SECTION(twoTile ? "TwoTile mode" : "Standard mode")
    {
        rocRoller::KernelOptions kernelOpts;
        kernelOpts->assertWaitCntState = false;

        auto context = TestContext::ForTestDevice(kernelOpts, twoTile ? "TwoTile" : "Standard");
        rocRoller::KernelGraph::KernelGraph kgraph;

        uint numTileM = 10;
        uint numTileN = 12;
        uint numTileK = 512;

        hipDeviceProp_t deviceProperties;
        REQUIRE_THAT(hipGetDeviceProperties(&deviceProperties, 0), HasHipSuccess(0));
        uint numWGs = deviceProperties.multiProcessorCount;

        auto k = context->kernel();

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

        CommandParametersPtr params           = std::make_shared<CommandParameters>();
        params->loopOverOutputTilesDimensions = {0, 1};
        params->streamK                       = StreamKMode::Standard;

        auto addStreamK = std::make_shared<AddStreamK>(
            context.get(), params, rocRoller::KLOOP, rocRoller::KLOOP, Expression::literal(numWGs));

        kgraph = kgraph.transform(addStreamK);
        if(context->kernelOptions()->removeSetCoordinate)
            kgraph = kgraph.transform(std::make_shared<RemoveSetCoordinate>());

        auto kg2 = std::make_shared<rocRoller::KernelGraph::KernelGraph>(kgraph);
        k->setKernelGraphMeta(kg2);

        context->schedule(k->preamble());
        context->schedule(k->prolog());

        context->schedule(rocRoller::KernelGraph::generate(*kg2, context->kernel()));

        context->schedule(k->postamble());
        context->schedule(k->amdgpu_metadata());

        if(context->hipDeviceIndex() >= 0)
        {
            RandomGenerator random(17629u);
            auto            hostX = random.vector<float>(numTileM * numTileN * numTileK, -1.0, 1.0);
            auto            deviceX = make_shared_device(hostX);
            auto            deviceA = make_shared_device<float>(numWGs);

            KernelArguments kargs(false);
            kargs.append("in", deviceX.get());
            kargs.append("out", deviceA.get());

            auto assemblyArgs = context->kernel()->arguments();
            for(int i = 2; i < assemblyArgs.size(); i++)
                kargs.append(assemblyArgs[i].getName(), evaluate(assemblyArgs[i].getExpression()));

            KernelInvocation kinv;
            kinv.workitemCount = {numWGs, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<float> hostA(numWGs);
            REQUIRE_THAT(
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
            CHECK(res.ok);
            if(!res.ok)
            {
                INFO(res.message());
            }
        }
        else
        {
            std::vector<char> assembledKernel = context->instructions()->assemble();
            CHECK(assembledKernel.size() > 0);
        }
    }
}

TEST_CASE("AddStreamK with unroll K", "[streamk][kernel-graph]")
{
    using namespace rocRoller;
    using namespace KernelGraph;
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    auto context = TestContext::ForTestDevice();
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto unrollK = GENERATE(1u, 2u, 4u);
    auto mode = GENERATE(StreamKMode::Standard, StreamKMode::TwoTile, StreamKMode::TwoTileDPFirst);

    example.setTileSize(128, 256, 8);
    example.setMFMA(32, 32, 2, 1);
    example.setUseLDS(false, false, false);
    example.setPrefetch(false, 0, 0, false);
    example.setUnroll(unrollK);
    example.setStreamK(mode);

    auto numWGs     = example.getFlattenedWorkgroupSize();
    auto numWGsExpr = std::make_shared<Expression::Expression>(numWGs);

    auto kgraph = example.getKernelGraph();
    auto params = example.getCommandParameters();

    applyGraphTransforms(kgraph, params, context.get(), numWGsExpr);

    auto loopPredicate = [&](int tag) {
        auto maybeForLoop = kgraph.control.get<ForLoopOp>(tag);
        if(maybeForLoop)
            return maybeForLoop->loopName == rocRoller::KLOOP;
        return false;
    };

    // In two-tile mode, we should have two K loops: one for the SK
    // tiles and one for the DP tiles.
    auto expectForKLoops = mode == StreamKMode::Standard ? 1 : 2;

    SECTION("Pre unroll")
    {
        auto forLoopTags = kgraph.control.findElements(loopPredicate).to<std::vector>();

        CHECK(forLoopTags.size() == expectForKLoops);
        for(auto& tag : forLoopTags)
        {
            auto [lhs, rhs] = getForLoopIncrement(kgraph, tag);
            auto increment  = getUnsignedInt(evaluate(rhs));
            CHECK(increment == 1);
        }
    }

    kgraph = kgraph.transform(std::make_shared<UnrollLoops>(params, context.get()));

    SECTION("Post unroll")
    {
        auto forLoopTags = kgraph.control.findElements(loopPredicate).to<std::vector>();

        for(auto& tag : forLoopTags)
        {
            auto [lhs, rhs] = getForLoopIncrement(kgraph, tag);
            auto increment  = getUnsignedInt(evaluate(rhs));
            CHECK(increment == unrollK);
        }
    }
}

TEST_CASE("AddStreamK scratch policy usage", "[streamk][kernel-graph][scratch]")
{
    using namespace rocRoller;
    using namespace rocRoller::Operations;
    using namespace KernelGraph;
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    auto context = TestContext::ForDefaultTarget();
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);
    auto mode    = StreamKMode::Standard;

    example.setTileSize(128, 256, 8);
    example.setMFMA(32, 32, 2, 1);
    example.setUseLDS(false, false, false);
    example.setPrefetch(false, 0, 0, false);
    example.setStreamK(mode);

    auto numWGs     = example.getFlattenedWorkgroupSize();
    auto numWGsExpr = std::make_shared<Expression::Expression>(numWGs);

    auto kgraph = example.getKernelGraph();
    auto params = example.getCommandParameters();

    applyGraphTransforms(kgraph, params, context.get(), numWGsExpr);

    SECTION("Tile data uses None policy")
    {
        // After AddStreamK, the None policy should have non-zero scratch allocation
        // (used for tile data exchange between workgroups)
        auto amountNone = context->getScratchAmount(ScratchPolicy::None);
        auto valueNone  = Expression::evaluate(amountNone);
        // Tile data scratch should be allocated
        CHECK(getUnsignedInt(valueNone) > 0);
    }

    SECTION("Flags use ZeroedBeforeAndAfter policy")
    {
        // After AddStreamK, the ZeroedBeforeAndAfter policy should have non-zero scratch
        // allocation (used for synchronization flags that must be zeroed before/after kernel)
        auto amountZeroed = context->getScratchAmount(ScratchPolicy::ZeroedBeforeAndAfter);
        auto valueZeroed  = Expression::evaluate(amountZeroed);
        // Flags scratch should be allocated for sync purposes
        CHECK(getUnsignedInt(valueZeroed) > 0);
    }

    SECTION("Different policies have different allocations")
    {
        auto amountNone   = context->getScratchAmount(ScratchPolicy::None);
        auto amountZeroed = context->getScratchAmount(ScratchPolicy::ZeroedBeforeAndAfter);

        auto valueNone   = Expression::evaluate(amountNone);
        auto valueZeroed = Expression::evaluate(amountZeroed);

        // Both should have allocations, but they should be independent
        // (flags are typically smaller - one uint32 per WG, while tile data is larger)
        CHECK(getUnsignedInt(valueNone) > 0);
        CHECK(getUnsignedInt(valueZeroed) > 0);
        // Tile data should typically be larger than flags
        CHECK(getUnsignedInt(valueNone) > getUnsignedInt(valueZeroed));
    }

    SECTION("Only has one scratch coordinate for tile data")
    {

        auto findScratchNone = [&](int tag) {
            auto maybeUser = kgraph.coordinates.get<User>(tag);
            if(!maybeUser)
                return false;
            return maybeUser->argumentName == getScratchName(ScratchPolicy::None);
        };

        auto scratchNoneCoordinates
            = kgraph.coordinates.findElements(findScratchNone).to<std::vector>();
        CHECK(scratchNoneCoordinates.size() == 1);
    }

    SECTION("Load, store, and reset flags scratch space correctly")
    {
        auto findScratchZeroed = [&](int tag) {
            auto maybeUser = kgraph.coordinates.get<User>(tag);
            if(!maybeUser)
                return false;
            return maybeUser->argumentName == getScratchName(ScratchPolicy::ZeroedBeforeAndAfter);
        };

        auto scratchZeroedCoordinates
            = kgraph.coordinates.findElements(findScratchZeroed).to<std::vector>();
        CHECK(scratchZeroedCoordinates.size() == 2);

        // Verify the reset flags coordinate connects to the original flags coordinate via a duplicate edge
        auto resetFlagsCoordinate = -1, originalFlagsCoordinate = -1;
        for(const auto& tag : scratchZeroedCoordinates)
        {
            // Duplicate edge connects the reset flags coordinate to the original flags coordinate
            auto isDuplicate = isEdge<Duplicate>;
            auto outDuplicates
                = kgraph.coordinates.getOutputNodeIndices(tag, isDuplicate).to<std::vector>();

            CHECK((outDuplicates.size() == 1 || outDuplicates.empty()));
            if(outDuplicates.size() == 1)
            {
                resetFlagsCoordinate    = tag;
                originalFlagsCoordinate = outDuplicates[0];
            }
        }
        CHECK(resetFlagsCoordinate != -1);
        CHECK(originalFlagsCoordinate != -1);

        // Duplicate coordinate should have a higher tag than the original flags coordinate
        CHECK(resetFlagsCoordinate > originalFlagsCoordinate);

        // Verify the graph of flags scratch coordinates matches AddStreamK implementation
        auto isPassThroughEdge = isEdge<PassThrough>;
        auto isJoinEdge        = isEdge<Join>;
        auto isSplitEdge       = isEdge<Split>;
        auto maybeNextWorkgroupTag
            = kgraph.coordinates.getOutputNodeIndices(originalFlagsCoordinate, isPassThroughEdge)
                  .to<std::vector>();
        CHECK(maybeNextWorkgroupTag.size() == 1);
        auto nextWorkgroupTag = maybeNextWorkgroupTag[0];
        auto maybeSplit = kgraph.coordinates.getOutputNodeIndices(nextWorkgroupTag, isSplitEdge)
                              .to<std::vector>();
        CHECK(maybeSplit.size() == 3);
        auto maybeResetNextWorkgroupTag0
            = kgraph.coordinates.getOutputNodeIndices(maybeSplit[0], isJoinEdge).to<std::vector>();
        auto maybeResetNextWorkgroupTag1
            = kgraph.coordinates.getOutputNodeIndices(maybeSplit[1], isJoinEdge).to<std::vector>();
        auto maybeResetNextWorkgroupTag2
            = kgraph.coordinates.getOutputNodeIndices(maybeSplit[2], isJoinEdge).to<std::vector>();
        CHECK(maybeResetNextWorkgroupTag0.size() == 1);
        CHECK(maybeResetNextWorkgroupTag1.size() == 1);
        CHECK(maybeResetNextWorkgroupTag2.size() == 1);
        CHECK(maybeResetNextWorkgroupTag0[0] == maybeResetNextWorkgroupTag1[0]);
        CHECK(maybeResetNextWorkgroupTag1[0] == maybeResetNextWorkgroupTag2[0]);
        auto maybeResetFlagsCoordinate
            = kgraph.coordinates
                  .getOutputNodeIndices(maybeResetNextWorkgroupTag0[0], isPassThroughEdge)
                  .to<std::vector>();
        CHECK(maybeResetFlagsCoordinate.size() == 1);
        CHECK(maybeResetFlagsCoordinate[0] == resetFlagsCoordinate);

        // Verify there are two store flags operations
        auto storeFlagsOps
            = kgraph.control.findElements(kgraph.control.isElemType<StoreSGPR>()).to<std::vector>();
        CHECK(storeFlagsOps.size() == 2);

        // Verify one StoreSGPR is stores the flags and the other resets the flags
        auto storeFlagsTag = -1, resetFlagsTag = -1;
        for(auto const tag : storeFlagsOps)
        {
            auto flagCoordinateTag = kgraph.mapper.get<User>(tag);
            if(flagCoordinateTag == resetFlagsCoordinate)
            {
                resetFlagsTag = tag;
            }
            else
            {
                storeFlagsTag = tag;
            }
        }
        CHECK(storeFlagsTag != -1);
        CHECK(resetFlagsTag != -1);

        // Verify the reset flags happens after the store flags
        auto order
            = kgraph.control.compareNodes(rocRoller::UpdateCache, storeFlagsTag, resetFlagsTag);
        CHECK(order == NodeOrdering::LeftFirst);
    }
}

TEST_CASE("AddStreamK control graph structure", "[streamk][kernel-graph][control-graph]")
{
    using namespace rocRoller;
    using namespace KernelGraph;
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    auto context = TestContext::ForDefaultTarget();
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto mode = GENERATE(StreamKMode::Standard, StreamKMode::TwoTile, StreamKMode::TwoTileDPFirst);

    example.setTileSize(128, 256, 8);
    example.setMFMA(32, 32, 2, 1);
    example.setUseLDS(false, false, false);
    example.setPrefetch(false, 0, 0, false);
    example.setStreamK(mode);

    auto numWGs     = example.getFlattenedWorkgroupSize();
    auto numWGsExpr = std::make_shared<Expression::Expression>(numWGs);

    auto kgraph = example.getKernelGraph();
    auto params = example.getCommandParameters();

    applyGraphTransforms(kgraph, params, context.get(), numWGsExpr);

    auto findForLoops = [&](std::string const& name) {
        return kgraph.control
            .findElements([&](int tag) {
                auto maybeForLoop = kgraph.control.get<ForLoopOp>(tag);
                return maybeForLoop && maybeForLoop->loopName == name;
            })
            .to<std::vector>();
    };

    auto findConditionals = [&](std::string const& name) {
        return kgraph.control
            .findElements([&](int tag) {
                auto maybeCond = kgraph.control.get<ConditionalOp>(tag);
                return maybeCond && maybeCond->conditionName == name;
            })
            .to<std::vector>();
    };

    auto findDoWhileLoops = [&](std::string const& name) {
        return kgraph.control
            .findElements([&](int tag) {
                auto maybeDoWhile = kgraph.control.get<DoWhileOp>(tag);
                return maybeDoWhile && maybeDoWhile->loopName == name;
            })
            .to<std::vector>();
    };

    bool isTwoTile = mode == StreamKMode::TwoTile || mode == StreamKMode::TwoTileDPFirst;

    SECTION("Send, receive, spin, post-accumulation are in SK section")
    {
        auto skLoops   = findForLoops("SKStreamTileLoop");
        auto sendConds = findConditionals("Send Tile");
        auto recvConds = findConditionals("Receive Tile");
        auto spinLoops = findDoWhileLoops("Global sync spin loop");
        auto postAccum = findConditionals("Post-accumulation Condition");

        REQUIRE(skLoops.size() == 1);
        REQUIRE(sendConds.size() == 1);
        REQUIRE(recvConds.size() == 1);
        REQUIRE(spinLoops.size() == 1);
        REQUIRE(postAccum.size() == 1);

        // Send, receive, spin, and post-accumulation are all inside the body
        // of the SK tile loop (chained via Body -> Sequence within the loop).
        CHECK(kgraph.control.compareNodes(rocRoller::UpdateCache, skLoops[0], sendConds[0])
              == NodeOrdering::RightInBodyOfLeft);
        CHECK(kgraph.control.compareNodes(rocRoller::UpdateCache, skLoops[0], recvConds[0])
              == NodeOrdering::RightInBodyOfLeft);
        auto spinContainers = kgraph.control.nodesContaining(spinLoops[0]).to<std::set>();
        CHECK(spinContainers.contains(skLoops[0]));
        CHECK(spinContainers.contains(recvConds[0]));
        CHECK(kgraph.control.compareNodes(rocRoller::UpdateCache, skLoops[0], postAccum[0])
              == NodeOrdering::RightInBodyOfLeft);

        // Within the body, they are ordered: send -> receive -> post-accumulation
        CHECK(kgraph.control.compareNodes(rocRoller::UpdateCache, sendConds[0], recvConds[0])
              == NodeOrdering::LeftFirst);
        CHECK(kgraph.control.compareNodes(rocRoller::UpdateCache, recvConds[0], postAccum[0])
              == NodeOrdering::LeftFirst);
    }

    if(isTwoTile)
    {
        SECTION("TwoTile: SK and DP tile loop ordering matches mode")
        {
            auto skLoops = findForLoops("SKStreamTileLoop");
            auto dpLoops = findForLoops("DPStreamTileLoop");

            REQUIRE(skLoops.size() == 1);
            REQUIRE(dpLoops.size() == 1);

            auto order
                = kgraph.control.compareNodes(rocRoller::UpdateCache, skLoops[0], dpLoops[0]);

            if(mode == StreamKMode::TwoTile)
            {
                // SK first, then DP
                CHECK(order == NodeOrdering::LeftFirst);
            }
            else
            {
                // TwoTileDPFirst: DP first, then SK
                CHECK(order == NodeOrdering::RightFirst);
            }
        }
    }
}

TEST_CASE("AddStreamK coordinate graph structure", "[streamk][kernel-graph][coordinate-graph]")
{
    using namespace rocRoller;
    using namespace KernelGraph;
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    auto context = TestContext::ForDefaultTarget();
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    auto mode = GENERATE(StreamKMode::Standard, StreamKMode::TwoTile, StreamKMode::TwoTileDPFirst);

    example.setTileSize(128, 256, 8);
    example.setMFMA(32, 32, 2, 1);
    example.setUseLDS(false, false, false);
    example.setPrefetch(false, 0, 0, false);
    example.setStreamK(mode);

    auto numWGs     = example.getFlattenedWorkgroupSize();
    auto numWGsExpr = std::make_shared<Expression::Expression>(numWGs);

    auto kgraph = example.getKernelGraph();
    auto params = example.getCommandParameters();

    applyGraphTransforms(kgraph, params, context.get(), numWGsExpr);

    bool isTwoTile = mode == StreamKMode::TwoTile || mode == StreamKMode::TwoTileDPFirst;

    auto isSplitEdge       = isEdge<Split>;
    auto isTileEdge        = isEdge<Tile>;
    auto isPassThroughEdge = isEdge<PassThrough>;
    auto isDataFlowEdge    = isEdge<DataFlow>;

    auto findForLoops = [&](std::string const& name) {
        return kgraph.control
            .findElements([&](int tag) {
                auto maybeForLoop = kgraph.control.get<ForLoopOp>(tag);
                return maybeForLoop && maybeForLoop->loopName == name;
            })
            .to<std::vector>();
    };

    // Get the forward ForLoop coordinate from a tile loop's control node.
    // conditionalFor connects via NaryArgument::DEST to the increment coordinate
    // (a Linear node). DataFlow edges lead from that to the forward/reverse
    // ForLoop coordinates. The forward one has a Split input.
    auto getForwardForLoopCoord = [&](int tileLoopControlTag) -> std::optional<int> {
        auto incrCoord = kgraph.mapper.get(tileLoopControlTag, NaryArgument::DEST);
        if(incrCoord == -1)
            return std::nullopt;

        auto dataFlowOutputs
            = kgraph.coordinates.getOutputNodeIndices(incrCoord, isDataFlowEdge).to<std::vector>();

        for(auto coord : dataFlowOutputs)
        {
            auto splitInputs
                = kgraph.coordinates.getInputNodeIndices(coord, isSplitEdge).to<std::vector>();
            if(!splitInputs.empty())
                return coord;
        }
        return std::nullopt;
    };

    // Trace from a tile loop to its tileSpace node:
    //   tileLoop -> ForLoop coord -> Split -> localTileSpace -> Tile -> tileSpace
    auto getTileSpaceNode = [&](int tileLoopControlTag) -> std::optional<int> {
        auto forLoopCoord = getForwardForLoopCoord(tileLoopControlTag);
        if(!forLoopCoord)
            return std::nullopt;

        auto splitParents
            = kgraph.coordinates.getInputNodeIndices(*forLoopCoord, isSplitEdge).to<std::vector>();
        if(splitParents.empty())
            return std::nullopt;

        auto tileParents
            = kgraph.coordinates.getInputNodeIndices(splitParents[0], isTileEdge).to<std::vector>();
        if(tileParents.empty())
            return std::nullopt;

        return tileParents[0];
    };

    // Find the Sunder edge upstream of a given node (if any).
    auto findUpstreamSunderEdge = [&](int nodeTag) -> std::optional<int> {
        for(auto edgeTag : kgraph.coordinates.getNeighbours<Graph::Direction::Upstream>(nodeTag))
        {
            if(kgraph.coordinates.get<Sunder>(edgeTag).has_value())
                return edgeTag;
        }
        return std::nullopt;
    };

    if(!isTwoTile)
    {
        SECTION("Standard: tileSpace connected to tileSpaceSK via PassThrough, not Sunder")
        {
            auto skLoops = findForLoops("SKStreamTileLoop");
            REQUIRE(skLoops.size() == 1);

            auto tileSpaceSK = getTileSpaceNode(skLoops[0]);
            REQUIRE(tileSpaceSK.has_value());

            // tileSpaceSK should have a PassThrough edge upstream to tileSpace
            auto ptParents = kgraph.coordinates.getInputNodeIndices(*tileSpaceSK, isPassThroughEdge)
                                 .to<std::vector>();
            CHECK(ptParents.size() == 1);

            // No Sunder edge upstream in Standard mode
            CHECK(!findUpstreamSunderEdge(*tileSpaceSK).has_value());
        }
    }

    if(isTwoTile)
    {
        SECTION("TwoTile: Sunder output ordering matches mode")
        {
            // Trace from SK tile loop to tileSpaceSK, then find its Sunder edge
            auto skLoops = findForLoops("SKStreamTileLoop");
            REQUIRE(skLoops.size() == 1);

            auto tileSpaceSK = getTileSpaceNode(skLoops[0]);
            REQUIRE(tileSpaceSK.has_value());

            auto sunderEdgeTag = findUpstreamSunderEdge(*tileSpaceSK);
            REQUIRE(sunderEdgeTag.has_value());

            auto loc = kgraph.coordinates.getLocation(*sunderEdgeTag);
            // Forward Sunder: 1 input (tileSpace), 3 outputs (SK, DP, selector)
            CHECK(loc.incoming.size() == 1);
            REQUIRE(loc.outgoing.size() == 3);

            auto first  = loc.outgoing[0];
            auto second = loc.outgoing[1];

            // For TwoTile (SK first): outputs are [tileSpaceSK, tileSpaceDP, selector]
            // For TwoTileDPFirst:     outputs are [tileSpaceDP, tileSpaceSK, selector]
            if(mode == StreamKMode::TwoTile)
                CHECK(*tileSpaceSK == first);
            else
                CHECK(*tileSpaceSK == second);
        }

        SECTION("TwoTile: SK and DP have separate tile ForLoop dims")
        {
            auto skLoops = findForLoops("SKStreamTileLoop");
            auto dpLoops = findForLoops("DPStreamTileLoop");
            REQUIRE(skLoops.size() == 1);
            REQUIRE(dpLoops.size() == 1);

            auto skCoord = getForwardForLoopCoord(skLoops[0]);
            auto dpCoord = getForwardForLoopCoord(dpLoops[0]);
            REQUIRE(skCoord.has_value());
            REQUIRE(dpCoord.has_value());

            // SK and DP tile loops should map to different ForLoop coordinates
            CHECK(*skCoord != *dpCoord);

            // Each should come from a different Split (different local tile space)
            auto skSplitParents
                = kgraph.coordinates.getInputNodeIndices(*skCoord, isSplitEdge).to<std::vector>();
            auto dpSplitParents
                = kgraph.coordinates.getInputNodeIndices(*dpCoord, isSplitEdge).to<std::vector>();
            REQUIRE(skSplitParents.size() == 1);
            REQUIRE(dpSplitParents.size() == 1);

            CHECK(skSplitParents[0] != dpSplitParents[0]);
        }
    }
}
