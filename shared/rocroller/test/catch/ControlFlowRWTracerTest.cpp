// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>

using namespace rocRoller;
using namespace rocRoller::Expression;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;

using RW = KernelGraph::ControlFlowRWTracer::ReadWrite;

TEST_CASE("ControlFlowRWTracer - Basic SetCoordinate", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto unroll   = graph.coordinates.addElement(Unroll());
    auto kernel   = graph.control.addElement(Kernel());
    auto setCoord = graph.control.addElement(SetCoordinate(literal(5)));

    graph.control.addElement(Body(), {kernel}, {setCoord});
    graph.mapper.connect(setCoord, unroll, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);
    auto                             records = tracer.coordinatesReadWrite();

    // SetCoordinate doesn't read/write
    CHECK(records.size() == 0);
}

TEST_CASE("ControlFlowRWTracer - Assign Read and Write", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto tileC = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto exprB = dataFlowTag(tileB, Register::Type::Vector, DataType::Float);

    auto assign = graph.control.addElement(Assign{Register::Type::Vector, exprA + exprB});

    graph.control.addElement(Body(), {kernel}, {assign});
    graph.mapper.connect(assign, tileC, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // Check that tileA and tileB are READ
    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    REQUIRE(tileARecords.size() == 1);
    CHECK(tileARecords[0].control == assign);
    CHECK(tileARecords[0].coordinate == tileA);
    CHECK(tileARecords[0].rw == RW::READ);

    auto tileBRecords = tracer.coordinatesReadWrite(tileB);
    REQUIRE(tileBRecords.size() == 1);
    CHECK(tileBRecords[0].control == assign);
    CHECK(tileBRecords[0].coordinate == tileB);
    CHECK(tileBRecords[0].rw == RW::READ);

    // Check that tileC is WRITE
    auto tileCRecords = tracer.coordinatesReadWrite(tileC);
    REQUIRE(tileCRecords.size() == 1);
    CHECK(tileCRecords[0].control == assign);
    CHECK(tileCRecords[0].coordinate == tileC);
    CHECK(tileCRecords[0].rw == RW::WRITE);

    auto assignRecords = tracer.opReadWrite(assign);
    CHECK(assignRecords
          == std::vector<KernelGraph::ControlFlowRWTracer::ReadWriteRecord>(
              {{assign, tileA, RW::READ}, {assign, tileB, RW::READ}, {assign, tileC, RW::WRITE}}));
}

TEST_CASE("ControlFlowRWTracer - LoadTiled and StoreTiled", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto userA = graph.coordinates.addElement(User());
    auto userB = graph.coordinates.addElement(User());

    auto kernel = graph.control.addElement(Kernel());
    auto load   = graph.control.addElement(LoadTiled());
    auto store  = graph.control.addElement(StoreTiled());

    graph.control.addElement(Body(), {kernel}, {load});
    graph.control.addElement(Sequence(), {load}, {store});

    graph.mapper.connect<MacroTile>(load, tileA);
    graph.mapper.connect<User>(load, userA);
    graph.mapper.connect<MacroTile>(store, tileB);
    graph.mapper.connect<User>(store, userB);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // LoadTiled should write to tileA
    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    bool foundWrite   = std::ranges::any_of(
        tileARecords, [load](auto const& x) { return x.control == load && x.rw == RW::WRITE; });
    CHECK(foundWrite);

    // StoreTiled should read from tileB
    auto tileBRecords = tracer.coordinatesReadWrite(tileB);
    bool foundRead    = std::ranges::any_of(
        tileBRecords, [store](auto const& x) { return x.control == store && x.rw == RW::READ; });
    CHECK(foundRead);
}

TEST_CASE("ControlFlowRWTracer - ConditionalOp", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto tileC = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA     = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto condition = exprA > literal(0.0f);

    auto conditional = graph.control.addElement(ConditionalOp{condition});
    auto assignB     = graph.control.addElement(Assign{
        Register::Type::Vector, dataFlowTag(tileB, Register::Type::Vector, DataType::Float)});
    auto assignC     = graph.control.addElement(Assign{
        Register::Type::Vector, dataFlowTag(tileC, Register::Type::Vector, DataType::Float)});

    graph.control.addElement(Body(), {kernel}, {conditional});
    graph.control.addElement(Body(), {conditional}, {assignB}); // true body
    graph.control.addElement(Else(), {conditional}, {assignC}); // false body

    graph.mapper.connect(assignB, tileB, NaryArgument::DEST);
    graph.mapper.connect(assignC, tileC, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // ConditionalOp should read tileA (the condition)
    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    REQUIRE(tileARecords.size() >= 1);
    CHECK(tileARecords[0].control == conditional);
    CHECK(tileARecords[0].rw == RW::READ);

    // Both branches should be traced
    auto tileBRecords = tracer.coordinatesReadWrite(tileB);
    REQUIRE(tileBRecords.size() >= 1);

    auto tileCRecords = tracer.coordinatesReadWrite(tileC);
    REQUIRE(tileCRecords.size() >= 1);
}

TEST_CASE("ControlFlowRWTracer - ForLoopOp", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto unroll  = graph.coordinates.addElement(Unroll());
    auto forLoop = graph.coordinates.addElement(ForLoop());
    auto tile    = graph.coordinates.addElement(MacroTile());

    auto kernel    = graph.control.addElement(Kernel());
    auto counter   = dataFlowTag(forLoop, Register::Type::Scalar, DataType::Int32);
    auto forLoopOp = graph.control.addElement(ForLoopOp{counter < literal(10), "test_loop"});
    auto setCoord  = graph.control.addElement(SetCoordinate(literal(5)));

    graph.control.addElement(Body(), {kernel}, {forLoop});
    graph.control.addElement(Body(), {forLoop}, {setCoord});

    graph.mapper.connect(setCoord, unroll, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // ForLoop should read the loop coordinate (in condition)
    auto unrollRecords    = tracer.coordinatesReadWrite(forLoop);
    bool foundForLoopRead = std::ranges::any_of(unrollRecords, [forLoopOp](auto const& x) {
        return x.control == forLoopOp && x.rw == RW::READ;
    });

    CHECK(foundForLoopRead);
}

TEST_CASE("ControlFlowRWTracer - DoWhileOp", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA     = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto condition = exprA > literal(0.0f);

    auto doWhile = graph.control.addElement(DoWhileOp{condition});
    auto assign  = graph.control.addElement(Assign{
        Register::Type::Vector, dataFlowTag(tileB, Register::Type::Vector, DataType::Float)});

    graph.control.addElement(Body(), {kernel}, {doWhile});
    graph.control.addElement(Body(), {doWhile}, {assign});
    graph.mapper.connect(assign, tileB, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // DoWhile should read tileA (in the condition)
    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    REQUIRE(tileARecords.size() >= 1);

    bool foundRead = std::ranges::any_of(tileARecords, [doWhile](auto const& x) {
        return x.control == doWhile && x.rw == RW::READ;
    });
    CHECK(foundRead);

    // The body should be traced
    auto tileBRecords = tracer.coordinatesReadWrite(tileB);
    REQUIRE(tileBRecords.size() >= 1);
}

TEST_CASE("ControlFlowRWTracer - Exchange", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());

    auto kernel   = graph.control.addElement(Kernel());
    auto exchange = graph.control.addElement(Exchange{});

    graph.control.addElement(Body(), {kernel}, {exchange});
    graph.mapper.connect<MacroTile>(exchange, tileA);
    graph.mapper.connect(exchange, tileB, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // Exchange should READ from source (tileA)
    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    REQUIRE(tileARecords.size() >= 1);
    bool foundRead = std::ranges::any_of(tileARecords, [exchange](auto const& x) {
        return x.control == exchange && x.rw == RW::READ;
    });
    CHECK(foundRead);

    // Exchange should READWRITE to destination (tileB)
    auto tileBRecords = tracer.coordinatesReadWrite(tileB);
    REQUIRE(tileBRecords.size() >= 1);
    bool foundReadWrite = std::ranges::any_of(tileBRecords, [exchange](auto const& x) {
        return x.control == exchange && x.rw == RW::READWRITE;
    });
    CHECK(foundReadWrite);
}

TEST_CASE("ControlFlowRWTracer - Complex data flow", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto tileC = graph.coordinates.addElement(MacroTile());
    auto tileD = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto exprB = dataFlowTag(tileB, Register::Type::Vector, DataType::Float);
    auto exprC = dataFlowTag(tileC, Register::Type::Vector, DataType::Float);

    // C = A + B
    auto assign1 = graph.control.addElement(Assign{Register::Type::Vector, exprA + exprB});
    // D = C * 2
    auto assign2 = graph.control.addElement(Assign{Register::Type::Vector, exprC * literal(2.0f)});

    graph.control.addElement(Body(), {kernel}, {assign1});
    graph.control.addElement(Sequence(), {assign1}, {assign2});

    graph.mapper.connect(assign1, tileC, NaryArgument::DEST);
    graph.mapper.connect(assign2, tileD, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    auto allRecords = tracer.coordinatesReadWrite();

    // Verify the complete flow
    CHECK(allRecords.size() >= 4); // At least: read A, read B, write C, read C, write D

    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    CHECK(tileARecords.size() == 1);
    CHECK(tileARecords[0].rw == RW::READ);

    auto tileBRecords = tracer.coordinatesReadWrite(tileB);
    CHECK(tileBRecords.size() == 1);
    CHECK(tileBRecords[0].rw == RW::READ);

    auto tileCRecords = tracer.coordinatesReadWrite(tileC);
    CHECK(tileCRecords.size() == 2); // written by assign1, read by assign2

    auto tileDRecords = tracer.coordinatesReadWrite(tileD);
    CHECK(tileDRecords.size() == 1);
    CHECK(tileDRecords[0].rw == RW::WRITE);
}

TEST_CASE("ControlFlowRWTracer - buildDependencies and getCoordinateDependencies",
          "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto tileC = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto exprB = dataFlowTag(tileB, Register::Type::Vector, DataType::Float);

    // C = A + B
    auto assign = graph.control.addElement(Assign{Register::Type::Vector, exprA + exprB});

    graph.control.addElement(Body(), {kernel}, {assign});
    graph.mapper.connect(assign, tileC, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);
    tracer.buildDependencies();

    // tileC depends on tileA and tileB
    auto deps = tracer.getCoordinateDependencies(tileC);
    CHECK(deps.size() == 2);
    CHECK(deps.count(tileA) == 1);
    CHECK(deps.count(tileB) == 1);

    // tileA has no dependencies
    auto depsA = tracer.getCoordinateDependencies(tileA);
    CHECK(depsA.empty());

    // tileB has no dependencies
    auto depsB = tracer.getCoordinateDependencies(tileB);
    CHECK(depsB.empty());
}

TEST_CASE("ControlFlowRWTracer - Transitive dependencies", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto tileC = graph.coordinates.addElement(MacroTile());
    auto tileD = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto exprB = dataFlowTag(tileB, Register::Type::Vector, DataType::Float);
    auto exprC = dataFlowTag(tileC, Register::Type::Vector, DataType::Float);

    // C = A + B
    auto assign1 = graph.control.addElement(Assign{Register::Type::Vector, exprA + exprB});
    // D = C * 2
    auto assign2 = graph.control.addElement(Assign{Register::Type::Vector, exprC * literal(2.0f)});

    graph.control.addElement(Body(), {kernel}, {assign1});
    graph.control.addElement(Sequence(), {assign1}, {assign2});

    graph.mapper.connect(assign1, tileC, NaryArgument::DEST);
    graph.mapper.connect(assign2, tileD, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);
    tracer.buildDependencies();

    // tileC depends on tileA and tileB
    auto depsC = tracer.getCoordinateDependencies(tileC);
    CHECK(depsC.size() == 2);
    CHECK(depsC.count(tileA) == 1);
    CHECK(depsC.count(tileB) == 1);

    // tileD depends on tileC (and transitively on A and B)
    auto depsD = tracer.getCoordinateDependencies(tileD);
    CHECK(depsD.size() == 3);
    CHECK(depsD.count(tileA) == 1);
    CHECK(depsD.count(tileB) == 1);
    CHECK(depsD.count(tileC) == 1);
}

TEST_CASE("ControlFlowRWTracer - Error on getDependencies without build", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tile   = graph.coordinates.addElement(MacroTile());
    auto kernel = graph.control.addElement(Kernel());

    auto expr   = dataFlowTag(tile, Register::Type::Vector, DataType::Float);
    auto assign = graph.control.addElement(Assign{Register::Type::Vector, expr});

    graph.control.addElement(Body(), {kernel}, {assign});
    graph.mapper.connect(assign, tile, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // Should throw because buildDependencies() wasn't called
    CHECK_THROWS_AS(tracer.getCoordinateDependencies(tile), FatalError);
}

TEST_CASE("ControlFlowRWTracer - Empty graph", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;
    auto                     kernel = graph.control.addElement(Kernel());

    KernelGraph::ControlFlowRWTracer tracer(graph);
    auto                             records = tracer.coordinatesReadWrite();

    // Empty graph should have no records
    CHECK(records.empty());
}

TEST_CASE("ControlFlowRWTracer - Multiply operation", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto tileC = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());
    auto mult   = graph.control.addElement(rocRoller::KernelGraph::ControlGraph::Multiply());

    graph.control.addElement(Body(), {kernel}, {mult});
    graph.mapper.connect(
        mult, tileA, KernelGraph::Connections::typeArgument<MacroTile>(NaryArgument::LHS));
    graph.mapper.connect(
        mult, tileB, KernelGraph::Connections::typeArgument<MacroTile>(NaryArgument::RHS));
    graph.mapper.connect(
        mult, tileC, KernelGraph::Connections::typeArgument<MacroTile>(NaryArgument::DEST));

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // Multiply should read A and B
    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    REQUIRE(tileARecords.size() >= 1);
    CHECK(tileARecords[0].rw == RW::READ);

    auto tileBRecords = tracer.coordinatesReadWrite(tileB);
    REQUIRE(tileBRecords.size() >= 1);
    CHECK(tileBRecords[0].rw == RW::READ);

    // Multiply should readwrite C (accumulate)
    auto tileCRecords = tracer.coordinatesReadWrite(tileC);
    REQUIRE(tileCRecords.size() >= 1);
    CHECK(tileCRecords[0].rw == RW::READWRITE);
}

TEST_CASE("ControlFlowRWTracer - AssertOp", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tile   = graph.coordinates.addElement(MacroTile());
    auto kernel = graph.control.addElement(Kernel());

    auto expr      = dataFlowTag(tile, Register::Type::Vector, DataType::Float);
    auto condition = expr > literal(0.0f);

    auto assertOp = graph.control.addElement(AssertOp{"Test assertion", condition});

    graph.control.addElement(Body(), {kernel}, {assertOp});

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // AssertOp should read the tile (in the condition)
    auto tileRecords = tracer.coordinatesReadWrite(tile);
    REQUIRE(tileRecords.size() >= 1);
    CHECK(tileRecords[0].control == assertOp);
    CHECK(tileRecords[0].rw == RW::READ);
}

TEST_CASE("ControlFlowRWTracer - toString for ReadWrite enum", "[kernel-graph][utils]")
{
    CHECK(toString(RW::READ) == "READ");
    CHECK(toString(RW::WRITE) == "WRITE");
    CHECK(toString(RW::READWRITE) == "READWRITE");
}

TEST_CASE("ControlFlowRWTracer - toString for ReadWriteRecord", "[kernel-graph][utils]")
{
    KernelGraph::ControlFlowRWTracer::ReadWriteRecord record;
    record.control    = 42;
    record.coordinate = 7;
    record.rw         = RW::READ;

    auto str = toString(record);
    CHECK_THAT(str, Catch::Matchers::ContainsSubstring("42"));
    CHECK_THAT(str, Catch::Matchers::ContainsSubstring("7"));
    CHECK_THAT(str, Catch::Matchers::ContainsSubstring("READ"));
}

TEST_CASE("ControlFlowRWTracer - Barrier operation", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tile    = graph.coordinates.addElement(MacroTile());
    auto kernel  = graph.control.addElement(Kernel());
    auto barrier = graph.control.addElement(Barrier{});

    graph.control.addElement(Body(), {kernel}, {barrier});
    graph.mapper.connect<MacroTile>(barrier, tile);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // Barrier should read coordinates
    auto tileRecords = tracer.coordinatesReadWrite(tile);
    REQUIRE(tileRecords.size() >= 1);

    bool foundRead = std::ranges::any_of(
        tileRecords, [barrier](auto const& x) { return x.control == barrier && x.rw == RW::READ; });
    CHECK(foundRead);
}

TEST_CASE("ControlFlowRWTracer - Multiple operations same coordinate", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tile   = graph.coordinates.addElement(MacroTile());
    auto kernel = graph.control.addElement(Kernel());

    auto expr1 = dataFlowTag(tile, Register::Type::Vector, DataType::Float);
    auto expr2 = expr1 * literal(2.0f);
    auto expr3 = expr2 + literal(1.0f);

    auto assign1 = graph.control.addElement(Assign{Register::Type::Vector, expr2});
    auto assign2 = graph.control.addElement(Assign{Register::Type::Vector, expr3});

    graph.control.addElement(Body(), {kernel}, {assign1});
    graph.control.addElement(Sequence(), {assign1}, {assign2});

    graph.mapper.connect(assign1, tile, NaryArgument::DEST);
    graph.mapper.connect(assign2, tile, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);
    auto                             tileRecords = tracer.coordinatesReadWrite(tile);

    // Should have multiple records: read by assign1, write by assign1, read by assign2, write by assign2
    CHECK(tileRecords.size() >= 3);

    int readCount = 0, writeCount = 0;
    for(auto const& record : tileRecords)
    {
        if(record.rw == RW::READ)
            readCount++;
        else if(record.rw == RW::WRITE)
            writeCount++;
    }

    CHECK(readCount >= 2); // Read by both assigns
    CHECK(writeCount >= 2); // Written by both assigns
}

TEST_CASE("ControlFlowRWTracer - LoadLDSTile and StoreLDSTile", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto ldsA  = graph.coordinates.addElement(LDS());
    auto ldsB  = graph.coordinates.addElement(LDS());

    auto kernel   = graph.control.addElement(Kernel());
    auto loadLDS  = graph.control.addElement(LoadLDSTile());
    auto storeLDS = graph.control.addElement(StoreLDSTile());

    graph.control.addElement(Body(), {kernel}, {loadLDS});
    graph.control.addElement(Sequence(), {loadLDS}, {storeLDS});

    graph.mapper.connect<MacroTile>(loadLDS, tileA);
    graph.mapper.connect<LDS>(loadLDS, ldsA);
    graph.mapper.connect<MacroTile>(storeLDS, tileB);
    graph.mapper.connect<LDS>(storeLDS, ldsB);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // LoadLDSTile should write to MacroTile and read from LDSTile
    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    bool foundWrite   = std::ranges::any_of(tileARecords, [loadLDS](auto const& x) {
        return x.control == loadLDS && x.rw == RW::WRITE;
    });
    CHECK(foundWrite);

    auto ldsARecords = tracer.coordinatesReadWrite(ldsA);
    bool foundRead   = std::ranges::any_of(
        ldsARecords, [loadLDS](auto const& x) { return x.control == loadLDS && x.rw == RW::READ; });
    CHECK(foundRead);
}

TEST_CASE("ControlFlowRWTracer - SeedPRNG", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto vgpr     = graph.coordinates.addElement(VGPR());
    auto vgprRHS  = graph.coordinates.addElement(VGPR());
    auto kernel   = graph.control.addElement(Kernel());
    auto seedPRNG = graph.control.addElement(SeedPRNG{});

    graph.control.addElement(Body(), {kernel}, {seedPRNG});
    graph.mapper.connect(seedPRNG, vgpr, NaryArgument::DEST);
    graph.mapper.connect(seedPRNG, vgprRHS, NaryArgument::RHS);

    KernelGraph::ControlFlowRWTracer tracer(graph);
    auto                             vgprRecords = tracer.coordinatesReadWrite(vgpr);

    // SeedPRNG should write to VGPR coordinate (the seed destination)
    CHECK(vgprRecords.size() >= 1);
    bool foundWrite = std::ranges::any_of(vgprRecords, [seedPRNG](auto const& x) {
        return x.control == seedPRNG && x.rw == RW::WRITE;
    });
    CHECK(foundWrite);

    // SeedPRNG should also read from RHS
    auto vgprRHSRecords = tracer.coordinatesReadWrite(vgprRHS);
    CHECK(vgprRHSRecords.size() >= 1);
}

TEST_CASE("ControlFlowRWTracer - Dependencies with no writes", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA  = graph.coordinates.addElement(MacroTile());
    auto kernel = graph.control.addElement(Kernel());

    // Just create a tile, don't write to it
    KernelGraph::ControlFlowRWTracer tracer(graph);
    tracer.buildDependencies();

    // Should return empty set for a coordinate with no dependencies
    auto deps = tracer.getCoordinateDependencies(tileA);
    CHECK(deps.empty());
}

TEST_CASE("ControlFlowRWTracer - Complex control flow with branches", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());
    auto tileC = graph.coordinates.addElement(MacroTile());
    auto tileD = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA     = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto condition = exprA > literal(0.0f);

    auto conditional = graph.control.addElement(ConditionalOp{condition});

    auto exprB = dataFlowTag(tileB, Register::Type::Vector, DataType::Float);
    auto exprC = dataFlowTag(tileC, Register::Type::Vector, DataType::Float);

    // True branch: D = B
    auto assignTrue = graph.control.addElement(Assign{Register::Type::Vector, exprB});
    // False branch: D = C
    auto assignFalse = graph.control.addElement(Assign{Register::Type::Vector, exprC});

    graph.control.addElement(Body(), {kernel}, {conditional});
    graph.control.addElement(Body(), {conditional}, {assignTrue});
    graph.control.addElement(Else(), {conditional}, {assignFalse});

    graph.mapper.connect(assignTrue, tileD, NaryArgument::DEST);
    graph.mapper.connect(assignFalse, tileD, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);
    tracer.buildDependencies();

    // tileD depends on B (true branch), and C (false branch)
    auto depsD = tracer.getCoordinateDependencies(tileD);
    CHECK(depsD.count(tileB) == 1);
    CHECK(depsD.count(tileC) == 1);

    // tileD doesn't directly depend on A
    CHECK(depsD.count(tileA) == 0);
}

TEST_CASE("ControlFlowRWTracer - Block operation", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tile   = graph.coordinates.addElement(MacroTile());
    auto kernel = graph.control.addElement(Kernel());
    auto block  = graph.control.addElement(Block{});
    auto assign = graph.control.addElement(
        Assign{Register::Type::Vector, dataFlowTag(tile, Register::Type::Vector, DataType::Float)});

    graph.control.addElement(Body(), {kernel}, {block});
    graph.control.addElement(Body(), {block}, {assign});
    graph.mapper.connect(assign, tile, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);
    auto                             records = tracer.coordinatesReadWrite();

    // Block should not prevent tracing of its body
    auto tileRecords = tracer.coordinatesReadWrite(tile);
    CHECK(tileRecords.size() >= 1);
}

TEST_CASE("ControlFlowRWTracer - LoadVGPR and StoreVGPR", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto vgpr1 = graph.coordinates.addElement(VGPR());
    auto vgpr2 = graph.coordinates.addElement(VGPR());

    auto kernel    = graph.control.addElement(Kernel());
    auto loadVGPR  = graph.control.addElement(LoadVGPR{});
    auto storeVGPR = graph.control.addElement(StoreVGPR{});

    graph.control.addElement(Body(), {kernel}, {loadVGPR});
    graph.control.addElement(Sequence(), {loadVGPR}, {storeVGPR});

    graph.mapper.connect<VGPR>(loadVGPR, vgpr1);
    graph.mapper.connect<VGPR>(storeVGPR, vgpr2);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // LoadVGPR should write to vgpr1
    auto vgpr1Records = tracer.coordinatesReadWrite(vgpr1);
    bool foundWrite   = std::ranges::any_of(vgpr1Records, [loadVGPR](auto const& x) {
        return x.control == loadVGPR && x.rw == RW::WRITE;
    });
    CHECK(foundWrite);

    // StoreVGPR should read from vgpr2
    auto vgpr2Records = tracer.coordinatesReadWrite(vgpr2);
    bool foundRead    = std::ranges::any_of(vgpr2Records, [storeVGPR](auto const& x) {
        return x.control == storeVGPR && x.rw == RW::READ;
    });
    CHECK(foundRead);
}

TEST_CASE("ControlFlowRWTracer - LoadSGPR and StoreSGPR", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto vgpr1 = graph.coordinates.addElement(VGPR());
    auto vgpr2 = graph.coordinates.addElement(VGPR());

    auto kernel    = graph.control.addElement(Kernel());
    auto loadSGPR  = graph.control.addElement(LoadSGPR{});
    auto storeSGPR = graph.control.addElement(StoreSGPR{});

    graph.control.addElement(Body(), {kernel}, {loadSGPR});
    graph.control.addElement(Sequence(), {loadSGPR}, {storeSGPR});

    graph.mapper.connect<VGPR>(loadSGPR, vgpr1);
    graph.mapper.connect<VGPR>(storeSGPR, vgpr2);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // LoadSGPR should write to vgpr1
    auto vgpr1Records = tracer.coordinatesReadWrite(vgpr1);
    bool foundWrite   = std::ranges::any_of(vgpr1Records, [loadSGPR](auto const& x) {
        return x.control == loadSGPR && x.rw == RW::WRITE;
    });
    CHECK(foundWrite);

    // StoreSGPR should read from vgpr2
    auto vgpr2Records = tracer.coordinatesReadWrite(vgpr2);
    bool foundRead    = std::ranges::any_of(vgpr2Records, [storeSGPR](auto const& x) {
        return x.control == storeSGPR && x.rw == RW::READ;
    });
    CHECK(foundRead);
}

TEST_CASE("ControlFlowRWTracer - LoadLinear and StoreLinear", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto linear1 = graph.coordinates.addElement(Linear());
    auto linear2 = graph.coordinates.addElement(Linear());

    auto kernel      = graph.control.addElement(Kernel());
    auto loadLinear  = graph.control.addElement(LoadLinear{});
    auto storeLinear = graph.control.addElement(StoreLinear{});

    graph.control.addElement(Body(), {kernel}, {loadLinear});
    graph.control.addElement(Sequence(), {loadLinear}, {storeLinear});

    graph.mapper.connect<Linear>(loadLinear, linear1);
    graph.mapper.connect<Linear>(storeLinear, linear2);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // LoadLinear should write to linear1
    auto linear1Records = tracer.coordinatesReadWrite(linear1);
    bool foundWrite     = std::ranges::any_of(linear1Records, [loadLinear](auto const& x) {
        return x.control == loadLinear && x.rw == RW::WRITE;
    });
    CHECK(foundWrite);

    // StoreLinear should read from linear2
    auto linear2Records = tracer.coordinatesReadWrite(linear2);
    bool foundRead      = std::ranges::any_of(linear2Records, [storeLinear](auto const& x) {
        return x.control == storeLinear && x.rw == RW::READ;
    });
    CHECK(foundRead);
}

TEST_CASE("ControlFlowRWTracer - LoadTileDirect2LDS", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tile = graph.coordinates.addElement(MacroTile());
    auto lds  = graph.coordinates.addElement(LDS());

    auto kernel     = graph.control.addElement(Kernel());
    auto loadDirect = graph.control.addElement(LoadTileDirect2LDS{});

    graph.control.addElement(Body(), {kernel}, {loadDirect});
    graph.mapper.connect<MacroTile>(loadDirect, tile);
    graph.mapper.connect<LDS>(loadDirect, lds);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // LoadTileDirect2LDS should read from MacroTile
    auto tileRecords = tracer.coordinatesReadWrite(tile);
    bool foundRead   = std::ranges::any_of(tileRecords, [loadDirect](auto const& x) {
        return x.control == loadDirect && x.rw == RW::READ;
    });
    CHECK(foundRead);

    // LoadTileDirect2LDS should write to LDS
    auto ldsRecords = tracer.coordinatesReadWrite(lds);
    bool foundWrite = std::ranges::any_of(ldsRecords, [loadDirect](auto const& x) {
        return x.control == loadDirect && x.rw == RW::WRITE;
    });
    CHECK(foundWrite);
}

TEST_CASE("ControlFlowRWTracer - NOP operation", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tile   = graph.coordinates.addElement(MacroTile());
    auto kernel = graph.control.addElement(Kernel());
    auto nop    = graph.control.addElement(NOP{});
    auto assign = graph.control.addElement(
        Assign{Register::Type::Vector, dataFlowTag(tile, Register::Type::Vector, DataType::Float)});

    graph.control.addElement(Body(), {kernel}, {nop});
    graph.control.addElement(Body(), {nop}, {assign});
    graph.mapper.connect(assign, tile, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);
    auto                             records = tracer.coordinatesReadWrite();

    // NOP should not prevent tracing of its body
    auto tileRecords = tracer.coordinatesReadWrite(tile);
    CHECK(tileRecords.size() >= 1);
}

TEST_CASE("ControlFlowRWTracer - trackConnections enabled", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA  = graph.coordinates.addElement(MacroTile());
    auto tileB  = graph.coordinates.addElement(MacroTile());
    auto unroll = graph.coordinates.addElement(Unroll());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA  = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto assign = graph.control.addElement(Assign{Register::Type::Vector, exprA});

    graph.control.addElement(Body(), {kernel}, {assign});
    graph.mapper.connect(assign, tileB, NaryArgument::DEST);
    graph.mapper.connect<Unroll>(assign, unroll);

    // Create tracer with trackConnections enabled
    KernelGraph::ControlFlowRWTracer tracerWithConnections(graph, -1, true);
    auto recordsWithConnections = tracerWithConnections.coordinatesReadWrite();

    // Create tracer with trackConnections disabled
    KernelGraph::ControlFlowRWTracer tracerWithoutConnections(graph, -1, false);
    auto recordsWithoutConnections = tracerWithoutConnections.coordinatesReadWrite();

    // With trackConnections enabled, we should have more records
    // (including the unroll coordinate)
    CHECK(recordsWithConnections.size() >= recordsWithoutConnections.size());

    // Check that unroll coordinate is tracked when trackConnections is enabled
    auto unrollRecordsWithConnections = tracerWithConnections.coordinatesReadWrite(unroll);
    CHECK(unrollRecordsWithConnections.size() > 0);
}

TEST_CASE("ControlFlowRWTracer - Multiply with scaling", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA  = graph.coordinates.addElement(MacroTile());
    auto tileB  = graph.coordinates.addElement(MacroTile());
    auto tileC  = graph.coordinates.addElement(MacroTile());
    auto scaleA = graph.coordinates.addElement(MacroTile());
    auto scaleB = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());
    auto mult   = graph.control.addElement(rocRoller::KernelGraph::ControlGraph::Multiply{
        Operations::ScaleMode::Separate, Operations::ScaleMode::Separate});

    graph.control.addElement(Body(), {kernel}, {mult});
    graph.mapper.connect(
        mult, tileA, KernelGraph::Connections::typeArgument<MacroTile>(NaryArgument::LHS));
    graph.mapper.connect(
        mult, tileB, KernelGraph::Connections::typeArgument<MacroTile>(NaryArgument::RHS));
    graph.mapper.connect(
        mult, tileC, KernelGraph::Connections::typeArgument<MacroTile>(NaryArgument::DEST));
    graph.mapper.connect(
        mult, scaleA, KernelGraph::Connections::typeArgument<MacroTile>(NaryArgument::LHS_SCALE));
    graph.mapper.connect(
        mult, scaleB, KernelGraph::Connections::typeArgument<MacroTile>(NaryArgument::RHS_SCALE));

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // Multiply should read A, B, and their scales
    auto tileARecords = tracer.coordinatesReadWrite(tileA);
    CHECK(tileARecords.size() >= 1);
    CHECK(tileARecords[0].rw == RW::READ);

    auto tileBRecords = tracer.coordinatesReadWrite(tileB);
    CHECK(tileBRecords.size() >= 1);
    CHECK(tileBRecords[0].rw == RW::READ);

    auto scaleARecords = tracer.coordinatesReadWrite(scaleA);
    CHECK(scaleARecords.size() >= 1);
    CHECK(scaleARecords[0].rw == RW::READ);

    auto scaleBRecords = tracer.coordinatesReadWrite(scaleB);
    CHECK(scaleBRecords.size() >= 1);
    CHECK(scaleBRecords[0].rw == RW::READ);

    // Multiply should readwrite C (accumulate)
    auto tileCRecords = tracer.coordinatesReadWrite(tileC);
    CHECK(tileCRecords.size() >= 1);
    CHECK(tileCRecords[0].rw == RW::READWRITE);
}

TEST_CASE("ControlFlowRWTracer - buildDependencies called multiple times", "[kernel-graph][utils]")
{
    KernelGraph::KernelGraph graph;

    auto tileA = graph.coordinates.addElement(MacroTile());
    auto tileB = graph.coordinates.addElement(MacroTile());

    auto kernel = graph.control.addElement(Kernel());

    auto exprA  = dataFlowTag(tileA, Register::Type::Vector, DataType::Float);
    auto assign = graph.control.addElement(Assign{Register::Type::Vector, exprA});

    graph.control.addElement(Body(), {kernel}, {assign});
    graph.mapper.connect(assign, tileB, NaryArgument::DEST);

    KernelGraph::ControlFlowRWTracer tracer(graph);

    // Build dependencies multiple times - should not cause issues
    tracer.buildDependencies();
    auto deps1 = tracer.getCoordinateDependencies(tileB);

    tracer.buildDependencies();
    auto deps2 = tracer.getCoordinateDependencies(tileB);

    // Results should be the same
    CHECK(deps1 == deps2);
    CHECK(deps1.size() == 1);
    CHECK(deps1.count(tileA) == 1);
}
