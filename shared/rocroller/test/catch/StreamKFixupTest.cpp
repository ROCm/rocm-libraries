// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestContext.hpp"

#include <common/CommonGraphs.hpp>
#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

TEST_CASE("StreamK multiple fix-ups", "[streamk][kernel-graph]")
{
    using namespace rocRoller;
    using namespace KernelGraph;
    using namespace ControlGraph;
    using namespace CoordinateGraph;

    auto context = TestContext::ForTestDevice();
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    example.setTileSize(128, 256, 8);
    example.setMFMA(32, 32, 2, 1);
    example.setUseLDS(false, false, false);
    example.setPrefetch(false, 0, 0, false);

    auto numWGs     = example.getFlattenedWorkgroupSize();
    auto numWGsExpr = std::make_shared<Expression::Expression>(numWGs);

    // Verify the control graph matches the pseudocode
    // if ReceiveTile
    //  for ReceiveTileLoop
    //      Assert(WG + forReceiveTileLoopCoord + 1 < numWGs)
    //      DoWhile
    //          LoadSGPR(flag[WG + forReceiveTileLoopCoord + 1])
    //      for XLoop and YLoop
    //          partialResult = LoadTiled(scratchAccTile[WG + forReceiveTileLoopCoord + 1])
    //          fullyAccTile = Assign(localPartiallyAccTile)
    //          fullyAccTile = Assign Add(fullyAccTile, partialResult)
    //          localAccTile = Assign(fullyAccTile)
    //          WaitZero()
    auto verifyStreamKReceiveTileLoop = [](rocRoller::KernelGraph::KernelGraph const& kgraph) {
        auto forLoopTags        = kgraph.control.getNodes<ForLoopOp>().to<std::vector>();
        auto receiveTileLoopTag = -1;

        // verify there is only 1 fixup for-loop
        for(const auto tag : forLoopTags)
        {
            auto forLoopOp = kgraph.control.get<ForLoopOp>(tag);
            if(forLoopOp->loopName == rocRoller::RECEIVE)
            {
                CHECK(receiveTileLoopTag == -1);
                receiveTileLoopTag = tag;
            }
        }
        CHECK(receiveTileLoopTag != -1);

        // verify the for-loop is under the receive tile condition
        auto receiveTileCondTag = -1;
        auto parent = only(kgraph.control.getInputNodeIndices<Body>(receiveTileLoopTag));
        CHECK(parent.has_value());
        receiveTileCondTag = parent.value();
        CHECK(kgraph.control.get<ConditionalOp>(receiveTileCondTag).has_value());

        // verify there is onlyl 1 for-XLoop and for-YLoop under the fixup for-loop
        auto isForXLoopPredicate = [&](int tag) -> bool {
            auto maybeForLoop = kgraph.control.get<ForLoopOp>(tag);
            if(maybeForLoop.has_value())
                return maybeForLoop->loopName == rocRoller::XLOOP;
            return false;
        };
        auto isForYLoopPredicate = [&](int tag) -> bool {
            auto maybeForLoop = kgraph.control.get<ForLoopOp>(tag);
            if(maybeForLoop.has_value())
                return maybeForLoop->loopName == rocRoller::YLOOP;
            return false;
        };
        auto fixupXLoop = only(kgraph.control.findNodes(receiveTileLoopTag, isForXLoopPredicate));
        auto fixupYLoop = only(kgraph.control.findNodes(receiveTileLoopTag, isForYLoopPredicate));
        CHECK(fixupXLoop.has_value());
        CHECK(fixupYLoop.has_value());

        // verify this sequence chain: LoadTiled -> Assign -> AssignAdd -> Assign
        auto loadPartialResult
            = only(kgraph.control.getOutputNodeIndices<Body>(fixupYLoop.value()));
        CHECK(loadPartialResult.has_value());
        CHECK(kgraph.control.get<LoadTiled>(loadPartialResult.value()).has_value());
        auto assignAccTile
            = only(kgraph.control.getOutputNodeIndices<Sequence>(loadPartialResult.value()));
        CHECK(assignAccTile.has_value());
        CHECK(kgraph.control.get<Assign>(assignAccTile.value()).has_value());
        auto assignAdd = only(kgraph.control.getOutputNodeIndices<Sequence>(assignAccTile.value()));
        CHECK(assignAdd.has_value());
        CHECK(kgraph.control.get<Assign>(assignAdd.value()).has_value());
        auto assignResultTile
            = only(kgraph.control.getOutputNodeIndices<Sequence>(assignAdd.value()));
        CHECK(assignResultTile.has_value());
        CHECK(kgraph.control.get<Assign>(assignResultTile.value()).has_value());

        // verify the mapper connection and expression
        auto getTileTag = [&](int assignTag) -> int {
            auto op = kgraph.control.getNode<Assign>(assignTag);
            return std::get<Expression::DataFlowTag>(*op.expression).tag;
        };
        auto partialResultTileTag = kgraph.mapper.get<MacroTile>(loadPartialResult.value());
        auto accTileTag           = getTileTag(assignAccTile.value());
        auto assignAccTileDestTag = getDEST(kgraph, assignAccTile.value());
        auto [assignAddLHSTag, assignAddLHSExpr]
            = getBinaryLHS<Expression::Add>(kgraph, assignAdd.value());
        auto [assignAddRHSTag, assignAddRHSExpr]
            = getBinaryRHS<Expression::Add>(kgraph, assignAdd.value());
        auto assignAddDest        = getDEST(kgraph, assignAdd.value());
        auto resultTileTag        = getTileTag(assignResultTile.value());
        auto assignResultTileDest = getDEST(kgraph, assignResultTile.value());

        CHECK(assignResultTileDest == accTileTag);
        CHECK(assignAccTileDestTag == resultTileTag);
        CHECK(assignAddDest == resultTileTag);

        auto isLHS = partialResultTileTag == assignAddLHSTag;
        auto isRHS = partialResultTileTag == assignAddRHSTag;
        CHECK((isLHS || isRHS));
        if(isLHS)
        {
            CHECK(assignAddRHSTag == resultTileTag);
        }
        if(isRHS)
        {
            CHECK(assignAddLHSTag == resultTileTag);
        }
    };

    auto applyGraphTransforms
        = [&numWGsExpr, &context](CommandParametersPtr                 params,
                                  rocRoller::KernelGraph::KernelGraph& kgraph) {
              std::vector<GraphTransformPtr> transforms;
              transforms.push_back(std::make_shared<IdentifyParallelDimensions>());
              transforms.push_back(std::make_shared<OrderMemory>(false));
              transforms.push_back(std::make_shared<UpdateParameters>(params));
              transforms.push_back(std::make_shared<AddLDS>(params, context.get()));
              transforms.push_back(std::make_shared<LowerLinear>(context.get()));
              transforms.push_back(std::make_shared<LowerTile>(params, context.get()));
              transforms.push_back(std::make_shared<LowerTensorContraction>(params, context.get()));
              transforms.push_back(std::make_shared<Simplify>());
              transforms.push_back(std::make_shared<FuseExpressions>());
              transforms.push_back(std::make_shared<AddStreamK>(
                  context.get(), params, rocRoller::XLOOP, rocRoller::KLOOP, numWGsExpr));

              for(auto& t : transforms)
                  kgraph = kgraph.transform(t);
          };

    SECTION("Standard StreamK Multiple Fixups")
    {
        example.setStreamK(StreamKMode::Standard);
        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        applyGraphTransforms(params, kgraph);
        verifyStreamKReceiveTileLoop(kgraph);
    }

    SECTION("TwoTile StreamK Multiple Fixups")
    {
        example.setStreamK(StreamKMode::TwoTile);
        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        applyGraphTransforms(params, kgraph);
        verifyStreamKReceiveTileLoop(kgraph);
    }

    SECTION("TwoTileDPFirst StreamK Multiple Fixups")
    {
        example.setStreamK(StreamKMode::TwoTileDPFirst);
        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        applyGraphTransforms(params, kgraph);
        verifyStreamKReceiveTileLoop(kgraph);
    }
}
