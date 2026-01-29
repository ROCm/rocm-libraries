// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Constraints.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/AddLDSBarriers.hpp>

using namespace rocRoller;
using namespace rocRoller::Expression;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace Catch::Matchers;

namespace KG = rocRoller::KernelGraph;

namespace
{
    /**
     * @brief Helper to get the VerifyLDSBarriers constraint from AddLDSBarriers transform.
     */
    std::vector<KG::GraphConstraint> getVerifyLDSBarriersConstraints()
    {
        KG::AddLDSBarriers transform;
        return transform.postConstraints();
    }
}

// =============================================================================
// Basic LDS barrier tests (no loops)
// =============================================================================

TEST_CASE("VerifyLDSBarriers - No LDS operations passes", "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile   = graph.coordinates.addElement(MacroTile());
    auto kernel = graph.control.addElement(Kernel());
    auto assign = graph.control.addElement(
        Assign{Register::Type::Vector, dataFlowTag(tile, Register::Type::Vector, DataType::Float)});

    graph.control.addElement(Body(), {kernel}, {assign});
    graph.mapper.connect(assign, tile, NaryArgument::DEST);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - LDS write and read without barrier fails",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel   = graph.control.addElement(Kernel());
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("Missing LDS barrier"));
}

TEST_CASE("VerifyLDSBarriers - LDS write and read with LDS-connected barrier passes",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel   = graph.control.addElement(Kernel());
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier  = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier});
    graph.control.addElement(Sequence(), {barrier}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<LDS>(barrier, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - Barrier not connected to LDS is not considered",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile      = graph.coordinates.addElement(MacroTile());
    auto lds       = graph.coordinates.addElement(LDS());
    auto otherTile = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier  = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier});
    graph.control.addElement(Sequence(), {barrier}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    // Barrier connected to MacroTile, NOT LDS
    graph.mapper.connect<MacroTile>(barrier, otherTile);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("Missing LDS barrier"));
}

TEST_CASE("VerifyLDSBarriers - Barrier connected to different LDS coordinate fails",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds1 = graph.coordinates.addElement(LDS());
    auto lds2 = graph.coordinates.addElement(LDS()); // Different LDS coordinate

    auto kernel   = graph.control.addElement(Kernel());
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier  = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier});
    graph.control.addElement(Sequence(), {barrier}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds1);
    // Barrier connected to lds2, NOT lds1
    graph.mapper.connect<LDS>(barrier, lds2);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds1);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    // The barrier is connected to LDS but to a DIFFERENT LDS coordinate,
    // so it should fail.
    CHECK_FALSE(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - Multiple LDS coordinates require barriers for each",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile1 = graph.coordinates.addElement(MacroTile());
    auto tile2 = graph.coordinates.addElement(MacroTile());
    auto lds1  = graph.coordinates.addElement(LDS());
    auto lds2  = graph.coordinates.addElement(LDS());

    auto kernel    = graph.control.addElement(Kernel());
    auto storeLDS1 = graph.control.addElement(StoreLDSTile());
    auto storeLDS2 = graph.control.addElement(StoreLDSTile());
    auto barrier1  = graph.control.addElement(Barrier());
    auto barrier2  = graph.control.addElement(Barrier());
    auto loadLDS1  = graph.control.addElement(LoadLDSTile());
    auto loadLDS2  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {storeLDS1});
    graph.control.addElement(Sequence(), {storeLDS1}, {storeLDS2});
    graph.control.addElement(Sequence(), {storeLDS2}, {barrier1});
    graph.control.addElement(Sequence(), {barrier1}, {loadLDS1});
    graph.control.addElement(Sequence(), {loadLDS1}, {barrier2});
    graph.control.addElement(Sequence(), {barrier2}, {loadLDS2});

    graph.mapper.connect<MacroTile>(storeLDS1, tile1);
    graph.mapper.connect<LDS>(storeLDS1, lds1);
    graph.mapper.connect<MacroTile>(storeLDS2, tile2);
    graph.mapper.connect<LDS>(storeLDS2, lds2);
    graph.mapper.connect<LDS>(barrier1, lds1);
    graph.mapper.connect<MacroTile>(loadLDS1, tile1);
    graph.mapper.connect<LDS>(loadLDS1, lds1);
    graph.mapper.connect<LDS>(barrier2, lds2);
    graph.mapper.connect<MacroTile>(loadLDS2, tile2);
    graph.mapper.connect<LDS>(loadLDS2, lds2);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    // Both LDS coordinates need barriers, but barrier only connected to lds1.
    // The implementation checks if barrier is connected to any LDS,
    // so this should pass because there's an LDS-connected barrier between all ops.
    CHECK(result.satisfied);
}

// =============================================================================
// ForLoop tests
// =============================================================================

TEST_CASE("VerifyLDSBarriers - ForLoop without barrier fails", "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile     = graph.coordinates.addElement(MacroTile());
    auto lds      = graph.coordinates.addElement(LDS());
    auto forCoord = graph.coordinates.addElement(ForLoop());

    auto kernel   = graph.control.addElement(Kernel());
    auto counter  = dataFlowTag(forCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoop  = graph.control.addElement(ForLoopOp{counter < literal(10), "test_loop"});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {forLoop});
    graph.control.addElement(Body(), {forLoop}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("Missing LDS barrier"));
}

TEST_CASE("VerifyLDSBarriers - ForLoop with single barrier between ops fails for loop-carried",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile     = graph.coordinates.addElement(MacroTile());
    auto lds      = graph.coordinates.addElement(LDS());
    auto forCoord = graph.coordinates.addElement(ForLoop());

    auto kernel   = graph.control.addElement(Kernel());
    auto counter  = dataFlowTag(forCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoop  = graph.control.addElement(ForLoopOp{counter < literal(10), "test_loop"});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier  = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {forLoop});
    graph.control.addElement(Body(), {forLoop}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier});
    graph.control.addElement(Sequence(), {barrier}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<LDS>(barrier, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    // Only one barrier between store and load.
    // This handles forward dependency, but not loop-carried dependency
    // (from iteration N's load to iteration N+1's store).
    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("loop-carried"));
}

TEST_CASE("VerifyLDSBarriers - ForLoop with barrier after load handles loop-carried",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile     = graph.coordinates.addElement(MacroTile());
    auto lds      = graph.coordinates.addElement(LDS());
    auto forCoord = graph.coordinates.addElement(ForLoop());

    auto kernel   = graph.control.addElement(Kernel());
    auto counter  = dataFlowTag(forCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoop  = graph.control.addElement(ForLoopOp{counter < literal(10), "test_loop"});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier1 = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());
    auto barrier2 = graph.control.addElement(Barrier());

    graph.control.addElement(Body(), {kernel}, {forLoop});
    graph.control.addElement(Body(), {forLoop}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier1});
    graph.control.addElement(Sequence(), {barrier1}, {loadLDS});
    graph.control.addElement(Sequence(), {loadLDS}, {barrier2});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<LDS>(barrier1, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);
    graph.mapper.connect<LDS>(barrier2, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    // barrier1: between store and load (forward dependency)
    // barrier2: after load (handles loop-carried dependency)
    CHECK(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - ForLoop with barrier before store handles loop-carried",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile     = graph.coordinates.addElement(MacroTile());
    auto lds      = graph.coordinates.addElement(LDS());
    auto forCoord = graph.coordinates.addElement(ForLoop());

    auto kernel   = graph.control.addElement(Kernel());
    auto counter  = dataFlowTag(forCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoop  = graph.control.addElement(ForLoopOp{counter < literal(10), "test_loop"});
    auto barrier1 = graph.control.addElement(Barrier());
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier2 = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {forLoop});
    graph.control.addElement(Body(), {forLoop}, {barrier1});
    graph.control.addElement(Sequence(), {barrier1}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier2});
    graph.control.addElement(Sequence(), {barrier2}, {loadLDS});

    graph.mapper.connect<LDS>(barrier1, lds);
    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<LDS>(barrier2, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    // barrier1: before store (handles loop-carried dependency)
    // barrier2: between store and load (forward dependency)
    CHECK(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - ForLoop barrier not connected to LDS fails",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile      = graph.coordinates.addElement(MacroTile());
    auto lds       = graph.coordinates.addElement(LDS());
    auto forCoord  = graph.coordinates.addElement(ForLoop());
    auto otherTile = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto counter  = dataFlowTag(forCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoop  = graph.control.addElement(ForLoopOp{counter < literal(10), "test_loop"});
    auto barrier1 = graph.control.addElement(Barrier());
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier2 = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {forLoop});
    graph.control.addElement(Body(), {forLoop}, {barrier1});
    graph.control.addElement(Sequence(), {barrier1}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier2});
    graph.control.addElement(Sequence(), {barrier2}, {loadLDS});

    // barrier1 connected to MacroTile instead of LDS
    graph.mapper.connect<MacroTile>(barrier1, otherTile);
    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    // barrier2 connected to MacroTile instead of LDS
    graph.mapper.connect<MacroTile>(barrier2, otherTile);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("Missing LDS barrier"));
}

// =============================================================================
// DoWhile tests
// =============================================================================

TEST_CASE("VerifyLDSBarriers - DoWhile without barrier fails", "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile     = graph.coordinates.addElement(MacroTile());
    auto lds      = graph.coordinates.addElement(LDS());
    auto condTile = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto condExpr = dataFlowTag(condTile, Register::Type::Vector, DataType::Float) > literal(0.0f);
    auto doWhile  = graph.control.addElement(DoWhileOp{condExpr});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {doWhile});
    graph.control.addElement(Body(), {doWhile}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("Missing LDS barrier"));
}

TEST_CASE("VerifyLDSBarriers - DoWhile with single barrier between ops fails for loop-carried",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile     = graph.coordinates.addElement(MacroTile());
    auto lds      = graph.coordinates.addElement(LDS());
    auto condTile = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto condExpr = dataFlowTag(condTile, Register::Type::Vector, DataType::Float) > literal(0.0f);
    auto doWhile  = graph.control.addElement(DoWhileOp{condExpr});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier  = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {doWhile});
    graph.control.addElement(Body(), {doWhile}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier});
    graph.control.addElement(Sequence(), {barrier}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<LDS>(barrier, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("loop-carried"));
}

TEST_CASE("VerifyLDSBarriers - DoWhile with two barriers passes", "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile     = graph.coordinates.addElement(MacroTile());
    auto lds      = graph.coordinates.addElement(LDS());
    auto condTile = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto condExpr = dataFlowTag(condTile, Register::Type::Vector, DataType::Float) > literal(0.0f);
    auto doWhile  = graph.control.addElement(DoWhileOp{condExpr});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier1 = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());
    auto barrier2 = graph.control.addElement(Barrier());

    graph.control.addElement(Body(), {kernel}, {doWhile});
    graph.control.addElement(Body(), {doWhile}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier1});
    graph.control.addElement(Sequence(), {barrier1}, {loadLDS});
    graph.control.addElement(Sequence(), {loadLDS}, {barrier2});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<LDS>(barrier1, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);
    graph.mapper.connect<LDS>(barrier2, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - DoWhile only LDS-connected barriers are valid",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile      = graph.coordinates.addElement(MacroTile());
    auto lds       = graph.coordinates.addElement(LDS());
    auto condTile  = graph.coordinates.addElement(MacroTile());
    auto otherTile = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto condExpr = dataFlowTag(condTile, Register::Type::Vector, DataType::Float) > literal(0.0f);
    auto doWhile  = graph.control.addElement(DoWhileOp{condExpr});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier1 = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());
    auto barrier2 = graph.control.addElement(Barrier());

    graph.control.addElement(Body(), {kernel}, {doWhile});
    graph.control.addElement(Body(), {doWhile}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier1});
    graph.control.addElement(Sequence(), {barrier1}, {loadLDS});
    graph.control.addElement(Sequence(), {loadLDS}, {barrier2});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<LDS>(barrier1, lds); // LDS-connected
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);
    // barrier2 NOT connected to LDS, connected to MacroTile instead
    graph.mapper.connect<MacroTile>(barrier2, otherTile);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    // barrier2 is not LDS-connected, so loop-carried dependency is not satisfied
    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("loop-carried"));
}

// =============================================================================
// LoadTileDirect2LDS tests
// =============================================================================

TEST_CASE("VerifyLDSBarriers - LoadTileDirect2LDS write and LoadLDSTile read needs barrier",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel      = graph.control.addElement(Kernel());
    auto loadDirect  = graph.control.addElement(LoadTileDirect2LDS());
    auto loadLDSTile = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {loadDirect});
    graph.control.addElement(Sequence(), {loadDirect}, {loadLDSTile});

    graph.mapper.connect<MacroTile>(loadDirect, tile);
    graph.mapper.connect<LDS>(loadDirect, lds);
    graph.mapper.connect<MacroTile>(loadLDSTile, tile);
    graph.mapper.connect<LDS>(loadLDSTile, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK_FALSE(result.satisfied);
    CHECK_THAT(result.explanation, ContainsSubstring("Missing LDS barrier"));
}

TEST_CASE("VerifyLDSBarriers - LoadTileDirect2LDS with LDS barrier passes",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel      = graph.control.addElement(Kernel());
    auto loadDirect  = graph.control.addElement(LoadTileDirect2LDS());
    auto barrier     = graph.control.addElement(Barrier());
    auto loadLDSTile = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {loadDirect});
    graph.control.addElement(Sequence(), {loadDirect}, {barrier});
    graph.control.addElement(Sequence(), {barrier}, {loadLDSTile});

    graph.mapper.connect<MacroTile>(loadDirect, tile);
    graph.mapper.connect<LDS>(loadDirect, lds);
    graph.mapper.connect<LDS>(barrier, lds);
    graph.mapper.connect<MacroTile>(loadLDSTile, tile);
    graph.mapper.connect<LDS>(loadLDSTile, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK(result.satisfied);
}

// =============================================================================
// Nested loops tests
// =============================================================================

TEST_CASE("VerifyLDSBarriers - Nested ForLoops need barriers in common ancestor",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile      = graph.coordinates.addElement(MacroTile());
    auto lds       = graph.coordinates.addElement(LDS());
    auto forCoord1 = graph.coordinates.addElement(ForLoop());
    auto forCoord2 = graph.coordinates.addElement(ForLoop());

    auto kernel = graph.control.addElement(Kernel());

    auto counter1  = dataFlowTag(forCoord1, Register::Type::Scalar, DataType::Int32);
    auto outerLoop = graph.control.addElement(ForLoopOp{counter1 < literal(10), "outer_loop"});

    auto counter2  = dataFlowTag(forCoord2, Register::Type::Scalar, DataType::Int32);
    auto innerLoop = graph.control.addElement(ForLoopOp{counter2 < literal(10), "inner_loop"});

    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier1 = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());
    auto barrier2 = graph.control.addElement(Barrier());

    graph.control.addElement(Body(), {kernel}, {outerLoop});
    graph.control.addElement(Body(), {outerLoop}, {innerLoop});
    graph.control.addElement(Body(), {innerLoop}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier1});
    graph.control.addElement(Sequence(), {barrier1}, {loadLDS});
    graph.control.addElement(Sequence(), {loadLDS}, {barrier2});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<LDS>(barrier1, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);
    graph.mapper.connect<LDS>(barrier2, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    // Both barriers are in innerLoop, so loop-carried is handled for innerLoop.
    // But the common ancestor is outerLoop, so loop-carried for outer loop
    // is also handled since barriers are inside it.
    CHECK(result.satisfied);
}

// =============================================================================
// Edge cases
// =============================================================================

TEST_CASE("VerifyLDSBarriers - Empty kernel passes", "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto kernel = graph.control.addElement(Kernel());

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - Only LDS write without read passes",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel   = graph.control.addElement(Kernel());
    auto storeLDS = graph.control.addElement(StoreLDSTile());

    graph.control.addElement(Body(), {kernel}, {storeLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - Only LDS read without write passes",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel  = graph.control.addElement(Kernel());
    auto loadLDS = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {loadLDS});

    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - Read before write in sequence (needs barrier)",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel   = graph.control.addElement(Kernel());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());
    auto storeLDS = graph.control.addElement(StoreLDSTile());

    graph.control.addElement(Body(), {kernel}, {loadLDS});
    graph.control.addElement(Sequence(), {loadLDS}, {storeLDS});

    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);
    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    // Read first, then write - still needs barrier between them
    CHECK_FALSE(result.satisfied);
}

TEST_CASE("VerifyLDSBarriers - Read before write in sequence with barrier passes",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel   = graph.control.addElement(Kernel());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());
    auto barrier  = graph.control.addElement(Barrier());
    auto storeLDS = graph.control.addElement(StoreLDSTile());

    graph.control.addElement(Body(), {kernel}, {loadLDS});
    graph.control.addElement(Sequence(), {loadLDS}, {barrier});
    graph.control.addElement(Sequence(), {barrier}, {storeLDS});

    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);
    graph.mapper.connect<LDS>(barrier, lds);
    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);

    auto constraints = getVerifyLDSBarriersConstraints();
    auto result      = graph.checkConstraints(constraints);

    CHECK(result.satisfied);
}

// =============================================================================
// AddLDSBarriers::apply tests
// =============================================================================

TEST_CASE("AddLDSBarriers::apply - No LDS operations returns unchanged graph",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile   = graph.coordinates.addElement(MacroTile());
    auto kernel = graph.control.addElement(Kernel());
    auto assign = graph.control.addElement(
        Assign{Register::Type::Vector, dataFlowTag(tile, Register::Type::Vector, DataType::Float)});

    graph.control.addElement(Body(), {kernel}, {assign});
    graph.mapper.connect(assign, tile, NaryArgument::DEST);

    KG::AddLDSBarriers transform;
    auto               result = transform.apply(graph);

    // Should return graph unchanged (no barriers added)
    auto constraints = transform.postConstraints();
    auto status      = result.checkConstraints(constraints);

    CHECK(status.satisfied);
}

TEST_CASE("AddLDSBarriers::apply - Inserts barrier between LDS write and read",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel   = graph.control.addElement(Kernel());
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    // Verify constraint fails before applying transform
    auto constraints  = KG::AddLDSBarriers().postConstraints();
    auto statusBefore = graph.checkConstraints(constraints);
    CHECK_FALSE(statusBefore.satisfied);

    // Apply the transform
    KG::AddLDSBarriers transform;
    auto               result = transform.apply(graph);

    // Verify constraint passes after applying transform
    auto statusAfter = result.checkConstraints(constraints);
    CHECK(statusAfter.satisfied);
}

TEST_CASE("AddLDSBarriers::apply - Do not reuse existing non-LDS barrier",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile      = graph.coordinates.addElement(MacroTile());
    auto lds       = graph.coordinates.addElement(LDS());
    auto otherTile = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier  = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier});
    graph.control.addElement(Sequence(), {barrier}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    // Barrier connected to a non-LDS coordinate
    graph.mapper.connect<MacroTile>(barrier, otherTile);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    // Verify constraint fails before applying transform
    auto constraints  = KG::AddLDSBarriers().postConstraints();
    auto statusBefore = graph.checkConstraints(constraints);
    CHECK_FALSE(statusBefore.satisfied);

    // Count barriers before
    int barrierCountBefore = 0;
    for(auto tag : graph.control.getNodes<Barrier>())
    {
        (void)tag;
        barrierCountBefore++;
    }

    // Apply the transform
    KG::AddLDSBarriers transform;
    auto               result = transform.apply(graph);

    // Count barriers after
    int barrierCountAfter = 0;
    for(auto tag : result.control.getNodes<Barrier>())
    {
        (void)tag;
        barrierCountAfter++;
    }

    // Should reuse the existing barrier (no new barriers added)
    CHECK(barrierCountAfter > barrierCountBefore);

    // Verify constraint passes after applying transform
    auto statusAfter = result.checkConstraints(constraints);
    CHECK(statusAfter.satisfied);
}

TEST_CASE("AddLDSBarriers::apply - Inserts two barriers for loop",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile     = graph.coordinates.addElement(MacroTile());
    auto lds      = graph.coordinates.addElement(LDS());
    auto forCoord = graph.coordinates.addElement(ForLoop());

    auto kernel   = graph.control.addElement(Kernel());
    auto counter  = dataFlowTag(forCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoop  = graph.control.addElement(ForLoopOp{counter < literal(10), "test_loop"});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {forLoop});
    graph.control.addElement(Body(), {forLoop}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {loadLDS});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    // Verify constraint fails before applying transform
    auto constraints  = KG::AddLDSBarriers().postConstraints();
    auto statusBefore = graph.checkConstraints(constraints);
    CHECK_FALSE(statusBefore.satisfied);

    // Apply the transform
    KG::AddLDSBarriers transform;
    auto               result = transform.apply(graph);

    // Verify constraint passes after applying transform
    auto statusAfter = result.checkConstraints(constraints);
    CHECK(statusAfter.satisfied);
}

TEST_CASE("AddLDSBarriers::apply - Do not reuse existing barriers in loop",
          "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile      = graph.coordinates.addElement(MacroTile());
    auto lds       = graph.coordinates.addElement(LDS());
    auto forCoord  = graph.coordinates.addElement(ForLoop());
    auto otherTile = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto counter  = dataFlowTag(forCoord, Register::Type::Scalar, DataType::Int32);
    auto forLoop  = graph.control.addElement(ForLoopOp{counter < literal(10), "test_loop"});
    auto storeLDS = graph.control.addElement(StoreLDSTile());
    auto barrier1 = graph.control.addElement(Barrier());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());
    auto barrier2 = graph.control.addElement(Barrier());

    graph.control.addElement(Body(), {kernel}, {forLoop});
    graph.control.addElement(Body(), {forLoop}, {storeLDS});
    graph.control.addElement(Sequence(), {storeLDS}, {barrier1});
    graph.control.addElement(Sequence(), {barrier1}, {loadLDS});
    graph.control.addElement(Sequence(), {loadLDS}, {barrier2});

    graph.mapper.connect<MacroTile>(storeLDS, tile);
    graph.mapper.connect<LDS>(storeLDS, lds);
    // Barriers connected to non-LDS
    graph.mapper.connect<MacroTile>(barrier1, otherTile);
    graph.mapper.connect<MacroTile>(barrier2, otherTile);
    graph.mapper.connect<MacroTile>(loadLDS, tile);
    graph.mapper.connect<LDS>(loadLDS, lds);

    // Verify constraint fails before applying transform
    auto constraints  = KG::AddLDSBarriers().postConstraints();
    auto statusBefore = graph.checkConstraints(constraints);
    CHECK_FALSE(statusBefore.satisfied);

    // Count barriers before
    int barrierCountBefore = 0;
    for(auto tag : graph.control.getNodes<Barrier>())
    {
        (void)tag;
        barrierCountBefore++;
    }

    // Apply the transform
    KG::AddLDSBarriers transform;
    auto               result = transform.apply(graph);

    // Count barriers after
    int barrierCountAfter = 0;
    for(auto tag : result.control.getNodes<Barrier>())
    {
        (void)tag;
        barrierCountAfter++;
    }

    // Should reuse the existing barriers (no new barriers added)
    CHECK(barrierCountAfter > barrierCountBefore);

    // Verify constraint passes after applying transform
    auto statusAfter = result.checkConstraints(constraints);
    CHECK(statusAfter.satisfied);
}

TEST_CASE("AddLDSBarriers::apply - Handles LoadTileDirect2LDS", "[kernel-graph][graph-transform]")
{
    KG::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel      = graph.control.addElement(Kernel());
    auto loadDirect  = graph.control.addElement(LoadTileDirect2LDS());
    auto loadLDSTile = graph.control.addElement(LoadLDSTile());

    graph.control.addElement(Body(), {kernel}, {loadDirect});
    graph.control.addElement(Sequence(), {loadDirect}, {loadLDSTile});

    graph.mapper.connect<MacroTile>(loadDirect, tile);
    graph.mapper.connect<LDS>(loadDirect, lds);
    graph.mapper.connect<MacroTile>(loadLDSTile, tile);
    graph.mapper.connect<LDS>(loadLDSTile, lds);

    // Verify constraint fails before applying transform
    auto constraints  = KG::AddLDSBarriers().postConstraints();
    auto statusBefore = graph.checkConstraints(constraints);
    CHECK_FALSE(statusBefore.satisfied);

    // Apply the transform
    KG::AddLDSBarriers transform;
    auto               result = transform.apply(graph);

    // Verify constraint passes after applying transform
    auto statusAfter = result.checkConstraints(constraints);
    CHECK(statusAfter.satisfied);
}
