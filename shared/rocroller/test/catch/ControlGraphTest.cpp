// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/KernelGraph/ControlGraph/ControlEdge_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include <common/SourceMatcher.hpp>

using namespace rocRoller;
using namespace KernelGraph::ControlGraph;

TEST_CASE("ControlGraph Basic", "[control-graph]")
{
    ControlGraph control = ControlGraph();

    int kernel_index = control.addElement(Kernel());
    int loadA_index  = control.addElement(LoadLinear(DataType::Float));
    int loadB_index  = control.addElement(LoadLinear(DataType::Float));
    int body1_index  = control.addElement(Body(), {kernel_index}, {loadA_index});
    int body2_index  = control.addElement(Body(), {kernel_index}, {loadB_index});

    int add_index       = control.addElement(Assign());
    int sequence1_index = control.addElement(Sequence(), {loadA_index}, {add_index});
    int sequence2_index = control.addElement(Sequence(), {loadB_index}, {add_index});

    int mul_index       = control.addElement(Assign());
    int sequence3_index = control.addElement(Sequence(), {add_index}, {mul_index});

    int storeC_index = control.addElement(StoreLinear());

    control.chain<Sequence>(loadB_index, mul_index, storeC_index);

    {
        std::vector<int> root = control.roots().to<std::vector>();
        REQUIRE(root.size() == 1);
        REQUIRE(root[0] == kernel_index);
    }

    {
        auto outputs = control.getOutputNodeIndices<Body>(kernel_index).to<std::vector>();
        REQUIRE(outputs.size() == 2);

        auto outputs2 = control.getOutputNodeIndices<Sequence>(kernel_index).to<std::vector>();
        REQUIRE(outputs2.size() == 0);
    }

    {
        std::vector<int> nodes1 = control.childNodes(kernel_index).to<std::vector>();
        REQUIRE(nodes1.size() == 2);

        std::vector<int> edges1 = control.getNeighbours<Graph::Direction::Downstream>(kernel_index);
        REQUIRE(edges1.size() == 2);
        REQUIRE(nodes1.size() == edges1.size());

        std::vector<int> nodes2 = control.getNeighbours<Graph::Direction::Upstream>(edges1[0]);
        REQUIRE(nodes2.size() == 1);
        CHECK(nodes2[0] == kernel_index);

        std::vector<int> nodes3 = control.getNeighbours<Graph::Direction::Upstream>(edges1[1]);
        REQUIRE(nodes3.size() == 1);
        CHECK(nodes3[0] == kernel_index);
    }

    {
        std::vector<int> nodes4 = control.parentNodes(loadA_index).to<std::vector>();
        REQUIRE(nodes4.size() == 1);
        CHECK(nodes4[0] == kernel_index);

        auto inputs = control.getInputNodeIndices<Body>(loadA_index).to<std::vector>();
        REQUIRE(inputs.size() == 1);
        CHECK(inputs.at(0) == kernel_index);

        CHECK(control.getInputNodeIndices<Sequence>(loadA_index).to<std::vector>().empty());
        CHECK(control.getInputNodeIndices<Initialize>(loadA_index).to<std::vector>().empty());
        CHECK(control.getInputNodeIndices<ForLoopIncrement>(loadA_index).to<std::vector>().empty());
    }

    {
        std::vector<int> edges2 = control.getNeighbours<Graph::Direction::Downstream>(loadA_index);
        REQUIRE(edges2.size() == 1);

        std::vector<int> nodes5 = control.parentNodes(loadB_index).to<std::vector>();
        REQUIRE(nodes5.size() == 1);
        CHECK(nodes5[0] == kernel_index);

        std::vector<int> edges3 = control.getNeighbours<Graph::Direction::Downstream>(loadB_index);
        CHECK(edges3.size() == 2);
    }

    {
        std::vector<int> nodes6 = control.childNodes(loadA_index).to<std::vector>();
        REQUIRE(nodes6.size() == 1);
        CHECK(nodes6[0] == add_index);

        std::vector<int> nodes7 = control.childNodes(add_index).to<std::vector>();
        REQUIRE(nodes7.size() == 1);
        CHECK(nodes7[0] == mul_index);
    }

    {
        std::vector<int> edges4 = control.getNeighbours<Graph::Direction::Upstream>(storeC_index);
        REQUIRE(edges4.size() == 1);

        std::vector<int> nodes8 = control.getNeighbours<Graph::Direction::Upstream>(edges4[0]);
        REQUIRE(nodes8.size() == 1);
        CHECK(nodes8[0] == mul_index);
    }

    CHECK(NormalizedSource(control.toDOT()) == NormalizedSource(R".(
        digraph {
                "1"[label="Kernel(1)"];
                "2"[label="LoadLinear Value: Float(2)"];
                "3"[label="LoadLinear Value: Float(3)"];
                "4"[label="Body(4)",shape=box];
                "5"[label="Body(5)",shape=box];
                "6"[label="Assign Count nullptr(6)"];
                "7"[label="Sequence(7)",shape=box];
                "8"[label="Sequence(8)",shape=box];
                "9"[label="Assign Count nullptr(9)"];
                "10"[label="Sequence(10)",shape=box];
                "11"[label="StoreLinear(11)"];
                "12"[label="Sequence(12)",shape=box];
                "13"[label="Sequence(13)",shape=box];
                "1" -> "4"
                "1" -> "5"
                "2" -> "7"
                "3" -> "8"
                "3" -> "12"
                "4" -> "2"
                "5" -> "3"
                "6" -> "10"
                "7" -> "6"
                "8" -> "6"
                "9" -> "13"
                "10" -> "9"
                "12" -> "9"
                "13" -> "11"
            }
        )."));

    SECTION("compareNodes throws on invalid inputs")
    {
        auto checkAllPolicies = [&](auto const policy) {
            CHECK_THROWS_AS(control.compareNodes(policy, loadA_index, loadA_index), FatalError);
            CHECK_THROWS_AS(control.compareNodes(policy, loadA_index, sequence1_index), FatalError);
            CHECK_THROWS_AS(control.compareNodes(policy, sequence2_index, loadB_index), FatalError);
            CHECK_THROWS(control.compareNodes(policy, loadA_index, 9000));
            CHECK_THROWS(control.compareNodes(policy, 9000, loadB_index));
        };
        checkAllPolicies(rocRoller::UpdateCache);
        checkAllPolicies(rocRoller::CacheOnly);
        checkAllPolicies(rocRoller::UseCacheIfAvailable);
        checkAllPolicies(rocRoller::IgnoreCache);
    }

    SECTION("nodesInBody / nodesAfter / nodesBefore / nodesContaining")
    {
        CHECK((std::set{loadA_index, loadB_index, add_index, mul_index, storeC_index})
              == control.nodesInBody(kernel_index).to<std::set>());

        CHECK((std::set{add_index, mul_index, storeC_index})
              == control.nodesAfter(loadA_index).to<std::set>());

        CHECK(control.nodesBefore(loadB_index).to<std::set>().empty());

        CHECK(std::set{kernel_index} == control.nodesContaining(loadA_index).to<std::set>());
        CHECK(std::set{kernel_index} == control.nodesContaining(storeC_index).to<std::set>());
    }

    SECTION("compareNodes ordering")
    {
        auto checkAllPolicies = [&](auto const policy) {
            CHECK(NodeOrdering::LeftFirst == control.compareNodes(policy, loadA_index, mul_index));
            CHECK(NodeOrdering::Undefined
                  == control.compareNodes(policy, loadA_index, loadB_index));
            CHECK(NodeOrdering::RightFirst == control.compareNodes(policy, mul_index, add_index));
            CHECK(NodeOrdering::LeftInBodyOfRight
                  == control.compareNodes(policy, mul_index, kernel_index));
            CHECK(NodeOrdering::RightInBodyOfLeft
                  == control.compareNodes(policy, kernel_index, loadB_index));
        };
        checkAllPolicies(rocRoller::UpdateCache);
        checkAllPolicies(rocRoller::CacheOnly);
        checkAllPolicies(rocRoller::UseCacheIfAvailable);
        checkAllPolicies(rocRoller::IgnoreCache);

        CHECK((std::set{loadA_index, loadB_index, add_index})
              == control.nodesBefore(mul_index).to<std::set>());
    }
}

TEST_CASE("ControlGraph BeforeAfter", "[control-graph]")
{
    ControlGraph control = ControlGraph();

    int kernel = control.addElement(Kernel());
    int loadA  = control.addElement(LoadLinear(DataType::Float));
    int loadB  = control.addElement(LoadLinear(DataType::Float));
    control.addElement(Body(), {kernel}, {loadA});
    control.addElement(Body(), {kernel}, {loadB});

    int add = control.addElement(Assign());
    control.addElement(Sequence(), {loadA}, {add});
    control.addElement(Sequence(), {loadB}, {add});

    int forOp = control.addElement(ForLoopOp());
    control.addElement(Sequence(), {loadA}, {forOp});

    int forInit = control.addElement(Assign());
    int forInc  = control.addElement(Assign());
    control.addElement(Initialize(), {forOp}, {forInit});
    control.addElement(ForLoopIncrement(), {forOp}, {forInc});

    int assign1 = control.addElement(Assign());
    control.addElement(Body(), {forOp}, {assign1});

    int loadC = control.addElement(LoadLinear(DataType::Float));
    control.addElement(Sequence(), {assign1}, {loadC});

    int assign2 = control.addElement(Assign());
    control.addElement(Body(), {forOp}, {assign2});

    int loadD = control.addElement(LoadLinear(DataType::Float));
    control.addElement(Sequence(), {assign2}, {loadD});

    int assign3 = control.addElement(Assign());
    control.addElement(Sequence(), {loadC}, {assign3});
    control.addElement(Sequence(), {loadD}, {assign3});

    int storeD = control.addElement(StoreLinear());
    control.addElement(Sequence(), {assign3}, {storeD});

    int scope3 = control.addElement(Scope());
    control.addElement(Sequence(), {forOp}, {scope3});

    int mul       = control.addElement(Assign());
    int sequence3 = control.addElement(Body(), {scope3}, {mul});

    int storeE    = control.addElement(StoreLinear());
    int sequence5 = control.addElement(Sequence(), {mul}, {storeE});

    CHECK((std::vector{kernel}) == control.roots().to<std::vector>());

    CHECK(control.nodesAfter(kernel).to<std::set>().empty());

    {
        auto expected = control.getNodes().to<std::set>();
        expected.erase(kernel);
        CHECK(expected == control.nodesInBody(kernel).to<std::set>());
    }

    CHECK((std::set{scope3, mul, storeE}) == control.nodesAfter(forOp).to<std::set>());
    // It doesn't walk up ForLoopIncrement edges yet.
    CHECK((std::set{scope3, mul, storeE}) == control.nodesAfter(forInc).to<std::set>());

    CHECK((std::set{scope3, kernel}) == control.nodesContaining(storeE).to<std::set>());

    SECTION("compareNodes ordering")
    {
        auto checkAllPolicies = [&](auto const policy) {
            CHECK(NodeOrdering::LeftFirst == control.compareNodes(policy, assign2, assign3));
            CHECK(NodeOrdering::LeftFirst == control.compareNodes(policy, assign2, mul));
            CHECK(NodeOrdering::Undefined == control.compareNodes(policy, assign1, assign2));

            CHECK(NodeOrdering::RightFirst == control.compareNodes(policy, forInc, forInit));
            CHECK(NodeOrdering::Undefined == control.compareNodes(policy, loadA, loadB));
            CHECK(NodeOrdering::RightInBodyOfLeft == control.compareNodes(policy, forOp, assign3));
            CHECK(NodeOrdering::LeftInBodyOfRight == control.compareNodes(policy, mul, scope3));

            CHECK(NodeOrdering::LeftFirst == control.compareNodes(policy, loadC, storeD));
            CHECK(NodeOrdering::LeftFirst == control.compareNodes(policy, loadD, storeD));
        };
        checkAllPolicies(rocRoller::UpdateCache);
        checkAllPolicies(rocRoller::CacheOnly);
        checkAllPolicies(rocRoller::UseCacheIfAvailable);
        checkAllPolicies(rocRoller::IgnoreCache);
    }

    SECTION("containingAncestors and controlStack")
    {
        {
            auto parents = KernelGraph::containingAncestors(forInit, control).to<std::vector>();
            std::vector<int> parentNodes;
            for(auto const& [node, edge] : parents)
                parentNodes.push_back(node);
            CHECK((std::vector{forOp, kernel}) == parentNodes);
            CHECK(std::holds_alternative<Initialize>(parents.at(0).second));
            CHECK(std::holds_alternative<Body>(parents.at(1).second));

            CHECK((std::deque{kernel, forOp, forInit})
                  == KernelGraph::controlStack(forInit, control));
        }

        {
            auto parents = KernelGraph::containingAncestors(storeD, control).to<std::vector>();
            std::vector<int> parentNodes;
            for(auto const& [node, edge] : parents)
                parentNodes.push_back(node);
            CHECK((std::vector{forOp, kernel}) == parentNodes);
            CHECK(std::holds_alternative<Body>(parents.at(0).second));
            CHECK(std::holds_alternative<Body>(parents.at(1).second));

            CHECK((std::deque{kernel, forOp, storeD})
                  == KernelGraph::controlStack(storeD, control));
        }

        {
            auto parents = KernelGraph::containingAncestors(storeE, control).to<std::vector>();
            std::vector<int> parentNodes;
            for(auto const& [node, edge] : parents)
                parentNodes.push_back(node);
            CHECK((std::vector{scope3, kernel}) == parentNodes);
            CHECK(std::holds_alternative<Body>(parents.at(0).second));
            CHECK(std::holds_alternative<Body>(parents.at(1).second));

            CHECK((std::deque{kernel, scope3, storeE})
                  == KernelGraph::controlStack(storeE, control));
        }
    }

    SECTION("nodeOrderTableString")
    {
        CHECK(NormalizedSource(control.nodeOrderTableString()) == NormalizedSource(R".(
               \   1   2   3   6   9  11  12  15  17  19  21  23  26  28  30  32
              1| --- RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB |   1
              2| LIB --- und  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF |   2
              3| LIB und ---  LF und und und und und und und und und und und und |   3
              6| LIB  RF  RF --- und und und und und und und und und und und und |   6
              9| LIB  RF und und --- RIB RIB RIB RIB RIB RIB RIB RIB  LF  LF  LF |   9
             11| LIB  RF und und LIB ---  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF |  11
             12| LIB  RF und und LIB  RF ---  RF  RF  RF  RF  RF  RF  LF  LF  LF |  12
             15| LIB  RF und und LIB  RF  LF ---  LF und und  LF  LF  LF  LF  LF |  15
             17| LIB  RF und und LIB  RF  LF  RF --- und und  LF  LF  LF  LF  LF |  17
             19| LIB  RF und und LIB  RF  LF und und ---  LF  LF  LF  LF  LF  LF |  19
             21| LIB  RF und und LIB  RF  LF und und  RF ---  LF  LF  LF  LF  LF |  21
             23| LIB  RF und und LIB  RF  LF  RF  RF  RF  RF ---  LF  LF  LF  LF |  23
             26| LIB  RF und und LIB  RF  LF  RF  RF  RF  RF  RF ---  LF  LF  LF |  26
             28| LIB  RF und und  RF  RF  RF  RF  RF  RF  RF  RF  RF --- RIB RIB |  28
             30| LIB  RF und und  RF  RF  RF  RF  RF  RF  RF  RF  RF LIB ---  LF |  30
             32| LIB  RF und und  RF  RF  RF  RF  RF  RF  RF  RF  RF LIB  RF --- |  32
               |   1   2   3   6   9  11  12  15  17  19  21  23  26  28  30  32
                    )."));
    }

    SECTION("toDOT")
    {
        CHECK(NormalizedSource(control.toDOT()) == NormalizedSource(R".(
        digraph {
            "1"[label="Kernel(1)"];
            "2"[label="LoadLinear Value: Float(2)"];
            "3"[label="LoadLinear Value: Float(3)"];
            "4"[label="Body(4)",shape=box];
            "5"[label="Body(5)",shape=box];
            "6"[label="Assign Count nullptr(6)"];
            "7"[label="Sequence(7)",shape=box];
            "8"[label="Sequence(8)",shape=box];
            "9"[label="ForLoopOp : nullptr(9)"];
            "10"[label="Sequence(10)",shape=box];
            "11"[label="Assign Count nullptr(11)"];
            "12"[label="Assign Count nullptr(12)"];
            "13"[label="Initialize(13)",shape=box];
            "14"[label="ForLoopIncrement(14)",shape=box];
            "15"[label="Assign Count nullptr(15)"];
            "16"[label="Body(16)",shape=box];
            "17"[label="LoadLinear Value: Float(17)"];
            "18"[label="Sequence(18)",shape=box];
            "19"[label="Assign Count nullptr(19)"];
            "20"[label="Body(20)",shape=box];
            "21"[label="LoadLinear Value: Float(21)"];
            "22"[label="Sequence(22)",shape=box];
            "23"[label="Assign Count nullptr(23)"];
            "24"[label="Sequence(24)",shape=box];
            "25"[label="Sequence(25)",shape=box];
            "26"[label="StoreLinear(26)"];
            "27"[label="Sequence(27)",shape=box];
            "28"[label="Scope(28)"];
            "29"[label="Sequence(29)",shape=box];
            "30"[label="Assign Count nullptr(30)"];
            "31"[label="Body(31)",shape=box];
            "32"[label="StoreLinear(32)"];
            "33"[label="Sequence(33)",shape=box];
            "1" -> "4"
            "1" -> "5"
            "2" -> "7"
            "2" -> "10"
            "3" -> "8"
            "4" -> "2"
            "5" -> "3"
            "7" -> "6"
            "8" -> "6"
            "9" -> "13"
            "9" -> "14"
            "9" -> "16"
            "9" -> "20"
            "9" -> "29"
            "10" -> "9"
            "13" -> "11"
            "14" -> "12"
            "15" -> "18"
            "16" -> "15"
            "17" -> "24"
            "18" -> "17"
            "19" -> "22"
            "20" -> "19"
            "21" -> "25"
            "22" -> "21"
            "23" -> "27"
            "24" -> "23"
            "25" -> "23"
            "27" -> "26"
            "28" -> "31"
            "29" -> "28"
            "30" -> "33"
            "31" -> "30"
            "33" -> "32"
        }
        )."));
    }
}

TEST_CASE("ControlGraph Conditional", "[control-graph]")
{
    ControlGraph control = ControlGraph();

    int kernelIndex = control.addElement(Kernel());
    int loadAIndex  = control.addElement(LoadLinear(DataType::Float));
    int body1Index  = control.addElement(Body(), {kernelIndex}, {loadAIndex});
    int condOp      = control.addElement(ConditionalOp());

    control.addElement(Sequence(), {loadAIndex}, {condOp});

    int addIndex   = control.addElement(Assign());
    int mulIndex   = control.addElement(Assign());
    int trueIndex  = control.addElement(Body(), {condOp}, {addIndex});
    int falseIndex = control.addElement(Else(), {condOp}, {mulIndex});

    int storeCIndex = control.addElement(StoreLinear());
    control.addElement(Sequence(), {condOp}, {storeCIndex});

    {
        std::vector<int> root = control.roots().to<std::vector>();
        REQUIRE(root.size() == 1);
        CHECK(root[0] == kernelIndex);
    }

    CHECK(control.getOutputNodeIndices<Body>(condOp).to<std::vector>().size() == 1);
    CHECK(control.getOutputNodeIndices<Else>(condOp).to<std::vector>().size() == 1);

    CHECK(NormalizedSource(control.toDOT()) == NormalizedSource(R".(
        digraph {
                "1"[label="Kernel(1)"];
                "2"[label="LoadLinear Value: Float(2)"];
                "3"[label="Body(3)",shape=box];
                "4"[label="ConditionalOp : nullptr(4)"];
                "5"[label="Sequence(5)",shape=box];
                "6"[label="Assign Count nullptr(6)"];
                "7"[label="Assign Count nullptr(7)"];
                "8"[label="Body(8)",shape=box];
                "9"[label="Else(9)",shape=box];
                "10"[label="StoreLinear(10)"];
                "11"[label="Sequence(11)",shape=box];
                "1" -> "3"
                "2" -> "5"
                "3" -> "2"
                "4" -> "8"
                "4" -> "9"
                "4" -> "11"
                "5" -> "4"
                "8" -> "6"
                "9" -> "7"
                "11" -> "10"
            }
        )."));
}

TEST_CASE("ControlGraph containingAncestors distinguishes Body vs Else edges", "[control-graph]")
{
    ControlGraph control = ControlGraph();

    int kernel = control.addElement(Kernel());
    int condOp = control.addElement(ConditionalOp());
    control.addElement(Body(), {kernel}, {condOp});

    int thenAssign = control.addElement(Assign());
    int elseAssign = control.addElement(Assign());
    control.addElement(Body(), {condOp}, {thenAssign});
    control.addElement(Else(), {condOp}, {elseAssign});

    // Both assigns have condOp as their immediate containing parent.
    auto thenParent = KernelGraph::containingAncestors(thenAssign, control).take(1).only();
    auto elseParent = KernelGraph::containingAncestors(elseAssign, control).take(1).only();

    REQUIRE(thenParent.has_value());
    REQUIRE(elseParent.has_value());

    CHECK(thenParent->first == condOp);
    CHECK(elseParent->first == condOp);

    // The edges must be distinct: Body for then-branch, Else for else-branch.
    CHECK(std::holds_alternative<Body>(thenParent->second));
    CHECK(std::holds_alternative<Else>(elseParent->second));
}

TEST_CASE("ControlGraph AssertOp", "[control-graph]")
{
    using GD             = rocRoller::Graph::Direction;
    ControlGraph control = ControlGraph();

    int kernelIndex = control.addElement(Kernel());
    int assertOp    = control.addElement(AssertOp());
    control.addElement(Body(), {kernelIndex}, {assertOp});

    int dummyIndex  = control.addElement(Assign());
    int passedIndex = control.addElement(Sequence(), {assertOp}, {dummyIndex});

    auto assertOps
        = control
              .findNodes(
                  kernelIndex,
                  [&](int tag) -> bool { return isOperation<AssertOp>(control.getElement(tag)); },
                  GD::Downstream)
              .to<std::vector>();
    CHECK(assertOps.size() == 1);
}

TEST_CASE("ControlGraph getSetCoordinates", "[control-graph]")
{
    KernelGraph::KernelGraph kg;
    using namespace KernelGraph;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    auto five = Expression::literal(5);
    auto four = Expression::literal(4);

    int topSet1 = kg.control.addElement(SetCoordinate{five});
    int topSet2 = kg.control.addElement(SetCoordinate{five});

    int notTopSet1 = kg.control.addElement(SetCoordinate{four});
    int topSet3    = kg.control.addElement(SetCoordinate{five});

    int load1 = kg.control.addElement(LoadLDSTile{DataType::Float});
    int load2 = kg.control.addElement(LoadLDSTile{DataType::Float});
    int load3 = kg.control.addElement(LoadLDSTile{DataType::Float});

    kg.control.addElement(Body{}, {topSet1}, {load1});

    CHECK_THROWS_AS(getTopSetCoordinate(kg, load1), FatalError);
    CHECK_THROWS_AS(getSetCoordinateForDim(kg, 1, load1), FatalError);
    CHECK_THROWS_AS(getUnrollValueForOp(kg, 1, load1), FatalError);

    kg.mapper.connect<CT::Unroll>(topSet1, 1);
    CHECK(topSet1 == getTopSetCoordinate(kg, load1));

    CHECK_THROWS_AS(getSetCoordinateForDim(kg, 2, load1), FatalError);
    CHECK_THROWS_AS(getUnrollValueForOp(kg, 2, load1), FatalError);

    CHECK(topSet1 == getSetCoordinateForDim(kg, 1, load1));
    CHECK(5 == getUnrollValueForOp(kg, 1, load1));

    auto loop = kg.control.addElement(ForLoopOp{});

    kg.control.chain<Body>(topSet2, notTopSet1, loop, topSet3, load3);
    kg.control.chain<Body>(notTopSet1, load2);

    CHECK_THROWS_AS(getTopSetCoordinate(kg, load3), FatalError);

    kg.mapper.connect<CT::Unroll>(topSet2, 2);
    kg.mapper.connect<CT::Unroll>(notTopSet1, 1);
    kg.mapper.connect<CT::Unroll>(topSet3, 3);
    CHECK(topSet2 == getTopSetCoordinate(kg, load2));
    CHECK(topSet3 == getTopSetCoordinate(kg, load3));

    CHECK_THROWS_AS(getSetCoordinateForDim(kg, 5, load3), FatalError);
    CHECK(topSet2 == getSetCoordinateForDim(kg, 2, load3));
    CHECK(notTopSet1 == getSetCoordinateForDim(kg, 1, load3));
    CHECK(topSet3 == getSetCoordinateForDim(kg, 3, load3));

    CHECK_THROWS_AS(getUnrollValueForOp(kg, 5, load3), FatalError);
    CHECK(5u == getUnrollValueForOp(kg, 2, load3));
    CHECK(4u == getUnrollValueForOp(kg, 1, load3));
    CHECK(5u == getUnrollValueForOp(kg, 3, load3));

    CHECK((std::set{topSet1, topSet3}) == getTopSetCoordinates(kg, {load1, load3}));
    CHECK((std::set{topSet1, topSet2, topSet3}) == getTopSetCoordinates(kg, {load1, load2, load3}));
}

TEST_CASE("ControlGraph hasExistingSetCoordinate", "[control-graph]")
{
    KernelGraph::KernelGraph kg;
    using namespace KernelGraph;
    namespace CT = rocRoller::KernelGraph::CoordinateGraph;

    int load1 = kg.control.addElement(LoadLDSTile{DataType::Float});
    int load2 = kg.control.addElement(LoadLDSTile{DataType::Float});

    auto coord1 = 1;
    auto coord2 = 2;
    auto coord3 = 3;

    auto unrollDim = 145;

    int set1 = kg.control.addElement(SetCoordinate{Expression::literal(coord1)});
    kg.mapper.connect<CT::Unroll>(set1, unrollDim);
    int set2 = kg.control.addElement(SetCoordinate{Expression::literal(coord2)});
    kg.mapper.connect<CT::Unroll>(set2, unrollDim);

    auto loop = kg.control.addElement(ForLoopOp{});

    kg.control.chain<Body>(loop, set1, set2, load1);
    kg.control.chain<Sequence>(set1, load2);

    CHECK(hasExistingSetCoordinate(kg, load1, coord1, unrollDim));
    CHECK_FALSE(hasExistingSetCoordinate(kg, load1, coord3, unrollDim));
    CHECK(hasExistingSetCoordinate(kg, load1, coord2, unrollDim));
    CHECK_FALSE(hasExistingSetCoordinate(kg, load2, coord1, unrollDim));
}

TEST_CASE("ControlGraph ModifyOrder", "[control-graph]")
{
    // This control graph is from the Basic test with two changes:
    //  - a new Assign node connected by an Initialize edge
    //  - a new Assign node connected by a Sequence edge
    ControlGraph control = ControlGraph();

    int kernel_index = control.addElement(Kernel());

    int assign1_index = control.addElement(Assign());
    int init_index    = control.addElement(Initialize(), {kernel_index}, {assign1_index});

    int assign2_index = control.addElement(Assign());
    int seq_index     = control.addElement(Sequence(), {kernel_index}, {assign2_index});

    int loadA_index = control.addElement(LoadLinear(DataType::Float));
    int loadB_index = control.addElement(LoadLinear(DataType::Float));
    int body1_index = control.addElement(Body(), {kernel_index}, {loadA_index});
    int body2_index = control.addElement(Body(), {kernel_index}, {loadB_index});

    int add_index       = control.addElement(Assign());
    int sequence1_index = control.addElement(Sequence(), {loadA_index}, {add_index});
    int sequence2_index = control.addElement(Sequence(), {loadB_index}, {add_index});

    int mul_index       = control.addElement(Assign());
    int sequence3_index = control.addElement(Sequence(), {add_index}, {mul_index});

    int storeC_index = control.addElement(StoreLinear());

    control.chain<Sequence>(loadB_index, mul_index, storeC_index);

    auto checkAllPolicies = [&](int nodeA, int nodeB, NodeOrdering expectedOrder) {
        CHECK(expectedOrder == control.compareNodes(rocRoller::UpdateCache, nodeA, nodeB));
        CHECK(expectedOrder == control.compareNodes(rocRoller::CacheOnly, nodeA, nodeB));
        CHECK(expectedOrder == control.compareNodes(rocRoller::UseCacheIfAvailable, nodeA, nodeB));
        CHECK(expectedOrder == control.compareNodes(rocRoller::IgnoreCache, nodeA, nodeB));
    };

    // assign1 is via an Initialize edge, so all other nodes (except kernel) are LeftFirst
    checkAllPolicies(assign1_index, loadA_index, NodeOrdering::LeftFirst);
    checkAllPolicies(assign1_index, loadB_index, NodeOrdering::LeftFirst);
    checkAllPolicies(assign1_index, storeC_index, NodeOrdering::LeftFirst);
    checkAllPolicies(assign1_index, assign2_index, NodeOrdering::LeftFirst);

    // assign2 is via a Sequence edge, so all other nodes (except kernel) are RightFirst
    checkAllPolicies(assign2_index, loadA_index, NodeOrdering::RightFirst);
    checkAllPolicies(assign2_index, loadB_index, NodeOrdering::RightFirst);
    checkAllPolicies(assign2_index, storeC_index, NodeOrdering::RightFirst);
    checkAllPolicies(assign2_index, assign1_index, NodeOrdering::RightFirst);

    // kernel is the only ancestor of loadA and loadB, both connected via Body
    checkAllPolicies(loadA_index, loadB_index, NodeOrdering::Undefined);

    // loadA is a parent of storeC with a path via a Sequence edge
    checkAllPolicies(storeC_index, loadA_index, NodeOrdering::RightFirst);
    // Delete the Sequence edge between loadA and add: no path between loadA and storeC
    control.deleteElement(sequence1_index);
    checkAllPolicies(storeC_index, loadA_index, NodeOrdering::Undefined);

    // Add the Sequence edge back: order should restore
    std::ignore = control.addElement(Sequence(), {loadA_index}, {add_index});
    checkAllPolicies(storeC_index, loadA_index, NodeOrdering::RightFirst);

    // add is a parent of mul (directly connected via a Sequence edge)
    checkAllPolicies(add_index, mul_index, NodeOrdering::LeftFirst);
    // Delete the Sequence edge between add and mul: order is Undefined
    control.deleteElement(sequence3_index);
    checkAllPolicies(add_index, mul_index, NodeOrdering::Undefined);

    // loadB is a parent of mul with two paths via two Sequence edges
    checkAllPolicies(loadB_index, mul_index, NodeOrdering::LeftFirst);
    // Delete one Sequence edge: order should still be the same
    control.deleteElement(sequence2_index);
    checkAllPolicies(loadB_index, mul_index, NodeOrdering::LeftFirst);
}
