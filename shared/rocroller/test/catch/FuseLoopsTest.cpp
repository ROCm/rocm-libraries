// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <omp.h>

#include "TestContext.hpp"

#include <common/CommonGraphs.hpp>
#include <common/Utilities.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseLoops_detail.hpp>
#include <rocRoller/Utilities/Utils.hpp>

using namespace rocRoller;
using namespace Catch::Matchers;

namespace FuseLoopsTest
{
    namespace
    {
        // Helper struct to hold loop counts by type
        struct LoopCounts
        {
            int total      = 0;
            int kLoops     = 0;
            int kLoopTails = 0;
            int xLoops     = 0;
            int yLoops     = 0;
        };

        // Helper function to count loops by type
        LoopCounts CountLoopsByType(rocRoller::KernelGraph::KernelGraph const& graph)
        {
            using namespace rocRoller::KernelGraph::ControlGraph;

            LoopCounts counts;
            auto       allLoops = graph.control.getNodes<ForLoopOp>().to<std::vector>();
            counts.total        = allLoops.size();

            for(auto loopTag : allLoops)
            {
                auto forLoop = graph.control.get<ForLoopOp>(loopTag);
                if(forLoop->loopName == rocRoller::KLOOP)
                    counts.kLoops++;
                else if(forLoop->loopName == rocRoller::KLOOPTAIL)
                    counts.kLoopTails++;
                else if(forLoop->loopName == rocRoller::XLOOP)
                    counts.xLoops++;
                else if(forLoop->loopName == rocRoller::YLOOP)
                    counts.yLoops++;
            }

            return counts;
        }

        // Helper function to populate loop body information
        void PopulateLoopInfo(
            rocRoller::KernelGraph::KernelGraph const& graph,
            std::unordered_map<int, rocRoller::KernelGraph::FuseLoopsDetail::LoopBodyInfo>&
                loopInfo)
        {
            using namespace rocRoller::KernelGraph;
            using namespace rocRoller::KernelGraph::ControlGraph;

            auto allLoops = graph.control.getNodes<ForLoopOp>().to<std::vector>();
            for(auto loopTag : allLoops)
                FuseLoopsDetail::PopulateChildLoops<Body>(graph, loopTag, loopInfo);
            FuseLoopsDetail::PopulateParentLoops(graph, loopInfo);
        }

        // Helper function to verify loop structure validity
        void VerifyLoopStructure(rocRoller::KernelGraph::KernelGraph const& graph, int loopTag)
        {
            using namespace rocRoller::KernelGraph::ControlGraph;

            auto forLoop = graph.control.get<ForLoopOp>(loopTag);
            auto bodies  = graph.control.getOutputNodeIndices<Body>(loopTag).to<std::vector>();
            auto increments
                = graph.control.getOutputNodeIndices<ForLoopIncrement>(loopTag).to<std::vector>();
            auto inits = graph.control.getOutputNodeIndices<Initialize>(loopTag).to<std::vector>();

            REQUIRE(!bodies.empty()); // All loops must have bodies
            CHECK(increments.size() == 1); // Exactly 1 increment per loop
            CHECK(inits.size() == 1); // Exactly 1 initialization per loop
            REQUIRE(forLoop.has_value()); // ForLoop must be valid
            REQUIRE(forLoop->condition != nullptr); // Loop condition must exist
        }
    } // anonymous namespace

    TEST_CASE("FuseLoops transformation works.", "[kernel-graph][graph-transforms]")
    {
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;

        auto context = TestContext::ForDefaultTarget();

        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        int macK  = 16;
        int waveK = 2;

        example.setTileSize(256, 64, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, true, false);

        example.setPrefetch(true, 2, 2, false);

        auto graph  = example.getKernelGraph();
        auto params = example.getCommandParameters();

        graph = transform<IdentifyParallelDimensions>(graph);
        graph = transform<OrderMemory>(graph, true);
        graph = transform<UpdateParameters>(graph, params);
        graph = transform<AddLDS>(graph, params, context.get());
        graph = transform<LowerLinear>(graph, context.get());
        graph = transform<LowerTile>(graph, params, context.get());
        graph = transform<LowerTensorContraction>(graph, params, context.get());
        graph = transform<Simplify>(graph);
        graph = transform<ConstantPropagation>(graph);
        graph = transform<FuseExpressions>(graph);
        graph = transform<ConnectWorkgroups>(graph, context.get());
        graph = transform<UnrollLoops>(graph, params, context.get());

        {
            auto counts = CountLoopsByType(graph);
            CHECK(counts.xLoops == 3); // 1 main compute + 2 epilogue loops
            CHECK(counts.yLoops == 12); // 3 XLoops * 4 unrolled YLoops each
            CHECK(counts.kLoops == 4); // 4 YLoops with K-dimension * 1 KLoop each
            CHECK(counts.kLoopTails == 4); // 1 tail loop per KLoop
        }

        SECTION("Call IdentifyFusibleLoops directly.")
        {
            std::unordered_map<int, FuseLoopsDetail::LoopBodyInfo> loopInfo;
            PopulateLoopInfo(graph, loopInfo);

            int                                                  loopsWithFusibleChildren = 0;
            std::vector<std::pair<int, std::unordered_set<int>>> fusibleSets;

            for(auto const& [loopTag, info] : loopInfo)
            {
                auto fusibleLoops = FuseLoopsDetail::IdentifyFusibleLoops(graph, loopTag, loopInfo);
                if(fusibleLoops.has_value())
                {
                    loopsWithFusibleChildren++;
                    fusibleSets.push_back({loopTag, *fusibleLoops});

                    // Each fusible set must have exactly the children of the parent
                    CHECK(fusibleLoops->size() == info.childLoops.size());

                    // All fusible loops must be in the child loops set
                    for(auto fusibleLoop : *fusibleLoops)
                        CHECK(info.childLoops.contains(fusibleLoop));

                    // Verify all fusible loops have identical length and increment
                    Expression::ExpressionPtr firstLoopSize;
                    Expression::ExpressionPtr firstIncrement;
                    for(auto forLoop : *fusibleLoops)
                    {
                        auto forLoopDim = getSize(std::get<Dimension>(graph.coordinates.getElement(
                            graph.mapper.get(forLoop, NaryArgument::DEST))));
                        auto [_, increment] = getForLoopIncrement(graph, forLoop);

                        if(!firstLoopSize)
                        {
                            firstLoopSize  = forLoopDim;
                            firstIncrement = increment;
                        }
                        else
                        {
                            REQUIRE(identical(forLoopDim, firstLoopSize));
                            REQUIRE(identical(increment, firstIncrement));
                        }
                    }
                }
            }

            CHECK(loopsWithFusibleChildren == 3); // 3 XLoops each have fusible children
        }

        SECTION("Test GetChildLoop helper function.")
        {
            std::unordered_map<int, FuseLoopsDetail::LoopBodyInfo> loopInfo;
            auto allLoops = graph.control.getNodes<ForLoopOp>().to<std::vector>();

            for(auto loopTag : allLoops)
                FuseLoopsDetail::PopulateChildLoops<Body>(graph, loopTag, loopInfo);

            int loopsChecked     = 0;
            int childrenVerified = 0;

            // Verify GetChildLoop finds exactly the expected children
            for(auto const& [loopTag, info] : loopInfo)
            {
                if(!info.childLoops.empty())
                {
                    loopsChecked++;
                    auto bodies
                        = graph.control.getOutputNodeIndices<Body>(loopTag).to<std::vector>();

                    // Count how many body edges lead to child loops
                    int expectedChildrenFromBodies = 0;
                    for(auto bodyNode : bodies)
                    {
                        std::unordered_set<int> visited;
                        auto foundChild = FuseLoopsDetail::GetChildLoop(graph, bodyNode, visited);

                        if(foundChild.has_value())
                        {
                            // Every found child MUST be in our childLoops set
                            REQUIRE(info.childLoops.contains(*foundChild));
                            expectedChildrenFromBodies++;
                            childrenVerified++;
                        }
                    }

                    // The number of children found should match the size of childLoops
                    CHECK(expectedChildrenFromBodies == static_cast<int>(info.childLoops.size()));
                }
            }

            CHECK(loopsChecked == 7); // 3 XLoops + 4 YLoops (with K children)
            CHECK(childrenVerified == 16); // (3 XLoops * 4 YLoops) + (4 YLoops * 1 KLoop)
        }

        SECTION("Test PopulateChildLoops with Sequence edges.")
        {
            std::unordered_map<int, FuseLoopsDetail::LoopBodyInfo> loopInfo;

            // First populate with Body edges
            auto allLoops = graph.control.getNodes<ForLoopOp>().to<std::vector>();
            for(auto loopTag : allLoops)
                FuseLoopsDetail::PopulateChildLoops<Body>(graph, loopTag, loopInfo);

            auto initialLoopCount = loopInfo.size();

            // Track which KLOOPs we process and what we find
            int kLoopsProcessed       = 0;
            int sequenceChildrenFound = 0;

            // Now populate with Sequence edges (used for tail loops)
            for(auto loopTag : allLoops)
            {
                auto forLoop = graph.control.get<ForLoopOp>(loopTag);
                // Only check K loops for sequence-connected children (tail loops)
                if(forLoop->loopName.starts_with(rocRoller::KLOOP))
                {
                    kLoopsProcessed++;
                    auto childrenBeforeSequence = loopInfo[loopTag].childLoops.size();

                    FuseLoopsDetail::PopulateChildLoops<Sequence>(graph, loopTag, loopInfo);
                    auto childrenAfterSequence = loopInfo[loopTag].childLoops.size();

                    if(childrenAfterSequence > childrenBeforeSequence)
                    {
                        auto newChildren = childrenAfterSequence - childrenBeforeSequence;
                        sequenceChildrenFound += newChildren;
                    }
                }
            }

            CHECK(initialLoopCount == 23); // Total loops from UnrollLoops
            CHECK(loopInfo.size() == 23); // All loops have LoopBodyInfo entries

            CHECK(kLoopsProcessed == 8); // 4 KLoops + 4 KLoopTails processed

            CHECK(sequenceChildrenFound == 4); // 4 KLoops each have 1 KLoopTail child
        }

        SECTION("Test FuseLoops detail function directly.")
        {
            auto                                                   graphCopy = graph;
            std::unordered_map<int, FuseLoopsDetail::LoopBodyInfo> loopInfo;
            PopulateLoopInfo(graphCopy, loopInfo);

            // Find a loop with fusible children
            std::optional<int>                     parentWithFusibleChildren;
            std::optional<std::unordered_set<int>> fusibleSet;

            for(auto const& [loopTag, info] : loopInfo)
            {
                auto fusible = FuseLoopsDetail::IdentifyFusibleLoops(graphCopy, loopTag, loopInfo);
                if(fusible.has_value() && fusible->size() >= 2)
                {
                    parentWithFusibleChildren = loopTag;
                    fusibleSet                = fusible;
                    break;
                }
            }

            REQUIRE(parentWithFusibleChildren.has_value());
            REQUIRE(fusibleSet.has_value());

            auto numLoopsBefore = graphCopy.control.getNodes<ForLoopOp>().to<std::vector>().size();
            auto numChildrenToFuse = fusibleSet->size();

            CHECK(numLoopsBefore == 23); // Total loops before any fusion
            CHECK(numChildrenToFuse == 4); // 4 YLoops per XLoop are fusible

            std::unordered_set<int> loopsToBeRemoved = *fusibleSet;
            auto                    fusedLoopTag     = *fusibleSet->begin();
            loopsToBeRemoved.erase(fusedLoopTag); // This one will be kept

            FuseLoopsDetail::FuseLoops(
                graphCopy, *parentWithFusibleChildren, loopInfo, *fusibleSet);

            auto loopsAfter    = graphCopy.control.getNodes<ForLoopOp>().to<std::vector>();
            auto numLoopsAfter = loopsAfter.size();

            auto expectedLoopsAfter = numLoopsBefore - (numChildrenToFuse - 1);
            CHECK(numLoopsAfter == expectedLoopsAfter); // 23 - 3 = 20 (kept 1, removed 3)

            // Verify the fused loop still exists
            REQUIRE(graphCopy.control.exists(fusedLoopTag));

            // Verify all other loops in fusibleSet were removed
            for(auto removedLoop : loopsToBeRemoved)
                CHECK(!graphCopy.control.exists(removedLoop));

            // Verify the fused loop has bodies from all fused loops
            auto bodies
                = graphCopy.control.getOutputNodeIndices<Body>(fusedLoopTag).to<std::vector>();
            REQUIRE(!bodies.empty());

            // Verify the fused loop still has proper structure
            auto increments = graphCopy.control.getOutputNodeIndices<ForLoopIncrement>(fusedLoopTag)
                                  .to<std::vector>();
            CHECK(increments.size() == 1); // Each loop has exactly 1 increment

            auto inits = graphCopy.control.getOutputNodeIndices<Initialize>(fusedLoopTag)
                             .to<std::vector>();
            CHECK(inits.size() == 1); // Each loop has exactly 1 initialization
        }

        SECTION("Test IdentifyAndFuseLoops orchestrator function.")
        {
            auto                                                   graphCopy = graph;
            std::unordered_map<int, FuseLoopsDetail::LoopBodyInfo> loopInfo;
            PopulateLoopInfo(graphCopy, loopInfo);

            // Count fusible loop sets before fusion
            int fusibleSetsCount  = 0;
            int totalFusibleLoops = 0;

            for(auto const& [loopTag, info] : loopInfo)
            {
                auto fusible = FuseLoopsDetail::IdentifyFusibleLoops(graphCopy, loopTag, loopInfo);
                if(fusible.has_value() && fusible->size() >= 2)
                {
                    fusibleSetsCount++;
                    totalFusibleLoops += fusible->size();
                }
            }

            CHECK(fusibleSetsCount == 3); // 3 XLoops with fusible children
            CHECK(totalFusibleLoops == 12); // 3 XLoops * 4 YLoops each

            auto loopsBefore    = graphCopy.control.getNodes<ForLoopOp>().to<std::vector>();
            auto numLoopsBefore = loopsBefore.size();
            CHECK(numLoopsBefore == 23); // Starting loop count

            FuseLoopsDetail::IdentifyAndFuseLoops(graphCopy, loopInfo);

            auto loopsAfter    = graphCopy.control.getNodes<ForLoopOp>().to<std::vector>();
            auto numLoopsAfter = loopsAfter.size();

            CHECK(numLoopsAfter == 11); // 3 X + 3 fused Y + 1 fused K + 4 KTail
            CHECK(numLoopsBefore - numLoopsAfter == 12); // Removed 9 YLoops + 3 KLoops

            for(auto loopTag : loopsAfter)
            {
                REQUIRE(graphCopy.control.exists(loopTag));

                auto bodies
                    = graphCopy.control.getOutputNodeIndices<Body>(loopTag).to<std::vector>();
                auto increments = graphCopy.control.getOutputNodeIndices<ForLoopIncrement>(loopTag)
                                      .to<std::vector>();
                auto inits
                    = graphCopy.control.getOutputNodeIndices<Initialize>(loopTag).to<std::vector>();

                REQUIRE(!bodies.empty()); // All loops must have bodies
                CHECK(increments.size() == 1); // Exactly 1 increment per loop
                CHECK(inits.size() == 1); // Exactly 1 initialization per loop
            }
        }

        SECTION("Apply graph transform.")
        {
            auto countsBefore = CountLoopsByType(graph);
            CHECK(countsBefore.kLoops == 4); // 4 YLoops with K-dimension
            CHECK(countsBefore.kLoopTails == 4); // 1 tail per KLoop
            CHECK(countsBefore.xLoops == 3); // 1 compute + 2 epilogue
            CHECK(countsBefore.yLoops == 12); // 3 XLoops * 4 unrolled YLoops

            graph = transform<FuseLoops>(graph);

            auto countsAfter = CountLoopsByType(graph);
            CHECK(countsAfter.kLoops == 1); // 4 KLoops fused into 1
            CHECK(countsAfter.kLoopTails == 1); // 4 KLoopTails fused into 1
            CHECK(countsAfter.xLoops == 3); // XLoops not fused
            CHECK(countsAfter.yLoops == 3); // 12 YLoops fused into 3 (one per XLoop)

            auto loopsAfterFusion = graph.control.getNodes<ForLoopOp>().to<std::vector>();
            for(auto loopTag : loopsAfterFusion)
                VerifyLoopStructure(graph, loopTag);

            // Verify the KLoop has 8 bodies (fused from 4 separate KLoops, each unrolled 2x)
            for(auto loopTag : loopsAfterFusion)
            {
                auto forLoop = graph.control.get<ForLoopOp>(loopTag);
                if(forLoop->loopName == rocRoller::KLOOP)
                {
                    auto bodies
                        = graph.control.getOutputNodeIndices<Body>(loopTag).to<std::vector>();
                    CHECK(bodies.size() == 8); // 4 fused KLoops * 2 unrolled bodies each
                }
                else if(forLoop->loopName == rocRoller::KLOOPTAIL)
                {
                    auto bodies
                        = graph.control.getOutputNodeIndices<Body>(loopTag).to<std::vector>();
                    CHECK(bodies.size() == 4); // 4 fused KLoopTails * 1 body each
                }
            }
        }
    }
}
