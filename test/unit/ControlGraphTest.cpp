#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <rocRoller/KernelGraph/ControlGraph/ControlEdge_fwd.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlGraph/Operation_fwd.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include "DataTypes/DataTypes.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using namespace KernelGraph::ControlGraph;

namespace rocRollerTest
{
    TEST(ControlGraphTest, Basic)
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
        int sequence4_index = control.addElement(Sequence(), {loadB_index}, {mul_index});

        int storeC_index    = control.addElement(StoreLinear());
        int sequence5_index = control.addElement(Sequence(), {mul_index}, {storeC_index});

        std::vector<int> root = control.roots().to<std::vector>();
        EXPECT_EQ(1, root.size());
        EXPECT_EQ(root[0], kernel_index);

        auto outputs = control.getOutputNodeIndices<Body>(kernel_index).to<std::vector>();
        EXPECT_EQ(2, outputs.size());

        auto outputs2 = control.getOutputNodeIndices<Sequence>(kernel_index).to<std::vector>();
        EXPECT_EQ(0, outputs2.size());

        std::vector<int> nodes1 = control.childNodes(root[0]).to<std::vector>();
        EXPECT_EQ(2, nodes1.size());

        std::vector<int> edges1
            = control.getNeighbours<Graph::Direction::Downstream>(root[0]).to<std::vector>();
        EXPECT_EQ(2, edges1.size());

        EXPECT_EQ(nodes1.size(), edges1.size());

        std::vector<int> nodes2
            = control.getNeighbours<Graph::Direction::Upstream>(edges1[0]).to<std::vector>();
        EXPECT_EQ(1, nodes2.size());
        EXPECT_EQ(nodes2[0], root[0]);

        std::vector<int> nodes3
            = control.getNeighbours<Graph::Direction::Upstream>(edges1[1]).to<std::vector>();
        EXPECT_EQ(1, nodes3.size());
        EXPECT_EQ(nodes3[0], root[0]);

        std::vector<int> nodes4 = control.parentNodes(loadA_index).to<std::vector>();
        EXPECT_EQ(1, nodes4.size());
        EXPECT_EQ(nodes4[0], root[0]);

        auto inputs = control.getInputNodeIndices<Body>(loadA_index).to<std::vector>();
        EXPECT_EQ(1, inputs.size());
        EXPECT_EQ(inputs.at(0), kernel_index);

        auto inputs2 = control.getInputNodeIndices<Sequence>(loadA_index).to<std::vector>();
        EXPECT_EQ(0, inputs2.size());

        auto inputs3 = control.getInputNodeIndices<Initialize>(loadA_index).to<std::vector>();
        EXPECT_EQ(0, inputs3.size());

        auto inputs4 = control.getInputNodeIndices<ForLoopIncrement>(loadA_index).to<std::vector>();
        EXPECT_EQ(0, inputs4.size());

        std::vector<int> edges2
            = control.getNeighbours<Graph::Direction::Downstream>(loadA_index).to<std::vector>();
        EXPECT_EQ(1, edges2.size());

        std::vector<int> nodes5 = control.parentNodes(loadB_index).to<std::vector>();
        EXPECT_EQ(1, nodes5.size());
        EXPECT_EQ(nodes5[0], root[0]);

        std::vector<int> edges3
            = control.getNeighbours<Graph::Direction::Downstream>(loadB_index).to<std::vector>();
        EXPECT_EQ(2, edges3.size());

        std::vector<int> nodes6 = control.childNodes(loadA_index).to<std::vector>();
        EXPECT_EQ(1, nodes6.size());
        EXPECT_EQ(nodes6[0], add_index);

        std::vector<int> nodes7 = control.childNodes(add_index).to<std::vector>();
        EXPECT_EQ(1, nodes7.size());
        EXPECT_EQ(nodes7[0], mul_index);

        std::vector<int> edges4
            = control.getNeighbours<Graph::Direction::Upstream>(storeC_index).to<std::vector>();
        EXPECT_EQ(1, edges4.size());

        std::vector<int> nodes8
            = control.getNeighbours<Graph::Direction::Upstream>(edges4[0]).to<std::vector>();
        EXPECT_EQ(1, nodes8.size());
        EXPECT_EQ(nodes8[0], mul_index);

        std::string expected = R".(
        digraph {
                "1"[label="Kernel(1)"];
                "2"[label="LoadLinear(2)"];
                "3"[label="LoadLinear(3)"];
                "4"[label="Body(4)",shape=box];
                "5"[label="Body(5)",shape=box];
                "6"[label="Assign Literal nullptr(6)"];
                "7"[label="Sequence(7)",shape=box];
                "8"[label="Sequence(8)",shape=box];
                "9"[label="Assign Literal nullptr(9)"];
                "10"[label="Sequence(10)",shape=box];
                "11"[label="Sequence(11)",shape=box];
                "12"[label="StoreLinear(12)"];
                "13"[label="Sequence(13)",shape=box];
                "1" -> "4"
                "1" -> "5"
                "2" -> "7"
                "3" -> "8"
                "3" -> "11"
                "4" -> "2"
                "5" -> "3"
                "6" -> "10"
                "7" -> "6"
                "8" -> "6"
                "9" -> "13"
                "10" -> "9"
                "11" -> "9"
                "13" -> "12"
            }
        ).";

        // Can't compare a node to itself
        EXPECT_THROW(control.compareNodes(loadA_index, loadA_index), FatalError);
        // Can't compare a node to an edge
        EXPECT_THROW(control.compareNodes(loadA_index, sequence1_index), FatalError);
        EXPECT_THROW(control.compareNodes(sequence2_index, loadB_index), FatalError);
        // Not in the graph
        EXPECT_ANY_THROW(control.compareNodes(loadA_index, 9000));
        EXPECT_ANY_THROW(control.compareNodes(9000, loadB_index));

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(control.toDOT()));

        EXPECT_EQ((std::set{loadA_index, loadB_index, add_index, mul_index, storeC_index}),
                  control.nodesInBody(kernel_index).to<std::set>());

        EXPECT_EQ((std::set{add_index, mul_index, storeC_index}),
                  control.nodesAfter(loadA_index).to<std::set>());

        EXPECT_EQ(std::set<int>{}, control.nodesBefore(loadB_index).to<std::set>());

        EXPECT_EQ(NodeOrdering::LeftFirst, control.compareNodes(loadA_index, mul_index));

        EXPECT_EQ(NodeOrdering::Undefined, control.compareNodes(loadA_index, loadB_index));

        EXPECT_EQ(NodeOrdering::RightFirst, control.compareNodes(mul_index, add_index));

        EXPECT_EQ(NodeOrdering::LeftInBodyOfRight, control.compareNodes(mul_index, kernel_index));
        EXPECT_EQ(NodeOrdering::RightInBodyOfLeft, control.compareNodes(kernel_index, loadB_index));
    }

    TEST(ControlGraphTest, BeforeAfter)
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

        int scope1 = control.addElement(Scope());
        control.addElement(Body(), {forOp}, {scope1});
        int assign1 = control.addElement(Assign());
        control.addElement(Body(), {scope1}, {assign1});

        int loadC = control.addElement(LoadLinear(DataType::Float));
        control.addElement(Sequence(), {assign1}, {loadC});

        int scope2 = control.addElement(Scope());
        control.addElement(Body(), {forOp}, {scope2});

        int assign2 = control.addElement(Assign());
        control.addElement(Body(), {scope2}, {assign2});

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

        {
            std::vector<int> root = control.roots().to<std::vector>();
            EXPECT_EQ(std::vector{kernel}, root);
        }

        EXPECT_EQ(std::set<int>{}, control.nodesAfter(kernel).to<std::set>());

        {
            auto expected = control.getNodes().to<std::set>();
            expected.erase(kernel);
            EXPECT_EQ(expected, control.nodesInBody(kernel).to<std::set>());
        }

        EXPECT_EQ(std::set({scope3, mul, storeE}), control.nodesAfter(forOp).to<std::set>());

        // It doesn't walk up ForLoopIncrement edges yet.
        EXPECT_EQ(std::set({scope3, mul, storeE}), control.nodesAfter(forInc).to<std::set>());

        EXPECT_EQ(NodeOrdering::LeftFirst, control.compareNodes(assign2, assign3));
        EXPECT_EQ(NodeOrdering::LeftFirst, control.compareNodes(assign2, mul));
        EXPECT_EQ(NodeOrdering::Undefined, control.compareNodes(assign1, assign2));

        EXPECT_EQ(NodeOrdering::RightFirst, control.compareNodes(forInc, forInit));
        EXPECT_EQ(NodeOrdering::Undefined, control.compareNodes(loadA, loadB));
        EXPECT_EQ(NodeOrdering::RightInBodyOfLeft, control.compareNodes(forOp, assign3));
        EXPECT_EQ(NodeOrdering::LeftInBodyOfRight, control.compareNodes(mul, scope3));

        EXPECT_EQ(NodeOrdering::LeftFirst, control.compareNodes(loadC, storeD));
        EXPECT_EQ(NodeOrdering::LeftFirst, control.compareNodes(loadD, storeD));

        EXPECT_EQ(NodeOrdering::RightInBodyOfLeft, control.compareNodes(scope1, storeD));
        EXPECT_EQ(NodeOrdering::RightInBodyOfLeft, control.compareNodes(scope2, storeD));

        std::string expectedTable = R".(
               \   1   2   3   6   9  11  12  15  17  19  21  23  25  27  30  32  34  36
              1| --- RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB |   1
              2| LIB --- und  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF |   2
              3| LIB und ---  LF und und und und und und und und und und und und und und |   3
              6| LIB  RF  RF --- und und und und und und und und und und und und und und |   6
              9| LIB  RF und und --- RIB RIB RIB RIB RIB RIB RIB RIB RIB RIB  LF  LF  LF |   9
             11| LIB  RF und und LIB ---  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF  LF |  11
             12| LIB  RF und und LIB  RF ---  RF  RF  RF  RF  RF  RF  RF  RF  LF  LF  LF |  12
             15| LIB  RF und und LIB  RF  LF --- RIB RIB und und und RIB RIB  LF  LF  LF |  15
             17| LIB  RF und und LIB  RF  LF LIB ---  LF und und und  LF  LF  LF  LF  LF |  17
             19| LIB  RF und und LIB  RF  LF LIB  RF --- und und und  LF  LF  LF  LF  LF |  19
             21| LIB  RF und und LIB  RF  LF und und und --- RIB RIB RIB RIB  LF  LF  LF |  21
             23| LIB  RF und und LIB  RF  LF und und und LIB ---  LF  LF  LF  LF  LF  LF |  23
             25| LIB  RF und und LIB  RF  LF und und und LIB  RF ---  LF  LF  LF  LF  LF |  25
             27| LIB  RF und und LIB  RF  LF LIB  RF  RF LIB  RF  RF ---  LF  LF  LF  LF |  27
             30| LIB  RF und und LIB  RF  LF LIB  RF  RF LIB  RF  RF  RF ---  LF  LF  LF |  30
             32| LIB  RF und und  RF  RF  RF  RF  RF  RF  RF  RF  RF  RF  RF --- RIB RIB |  32
             34| LIB  RF und und  RF  RF  RF  RF  RF  RF  RF  RF  RF  RF  RF LIB ---  LF |  34
             36| LIB  RF und und  RF  RF  RF  RF  RF  RF  RF  RF  RF  RF  RF LIB  RF --- |  36
               |   1   2   3   6   9  11  12  15  17  19  21  23  25  27  30  32  34  36
        ).";

        EXPECT_EQ(NormalizedSource(expectedTable), NormalizedSource(control.nodeOrderTable()));

        // Include the graph here to make it easier to tell if there are any changes.
        std::string expected = R".(
        digraph {
            "1"[label="Kernel(1)"];
            "2"[label="LoadLinear(2)"];
            "3"[label="LoadLinear(3)"];
            "4"[label="Body(4)",shape=box];
            "5"[label="Body(5)",shape=box];
            "6"[label="Assign Literal nullptr(6)"];
            "7"[label="Sequence(7)",shape=box];
            "8"[label="Sequence(8)",shape=box];
            "9"[label="ForLoopOp : nullptr(9)"];
            "10"[label="Sequence(10)",shape=box];
            "11"[label="Assign Literal nullptr(11)"];
            "12"[label="Assign Literal nullptr(12)"];
            "13"[label="Initialize(13)",shape=box];
            "14"[label="ForLoopIncrement(14)",shape=box];
            "15"[label="Scope(15)"];
            "16"[label="Body(16)",shape=box];
            "17"[label="Assign Literal nullptr(17)"];
            "18"[label="Body(18)",shape=box];
            "19"[label="LoadLinear(19)"];
            "20"[label="Sequence(20)",shape=box];
            "21"[label="Scope(21)"];
            "22"[label="Body(22)",shape=box];
            "23"[label="Assign Literal nullptr(23)"];
            "24"[label="Body(24)",shape=box];
            "25"[label="LoadLinear(25)"];
            "26"[label="Sequence(26)",shape=box];
            "27"[label="Assign Literal nullptr(27)"];
            "28"[label="Sequence(28)",shape=box];
            "29"[label="Sequence(29)",shape=box];
            "30"[label="StoreLinear(30)"];
            "31"[label="Sequence(31)",shape=box];
            "32"[label="Scope(32)"];
            "33"[label="Sequence(33)",shape=box];
            "34"[label="Assign Literal nullptr(34)"];
            "35"[label="Body(35)",shape=box];
            "36"[label="StoreLinear(36)"];
            "37"[label="Sequence(37)",shape=box];
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
            "9" -> "22"
            "9" -> "33"
            "10" -> "9"
            "13" -> "11"
            "14" -> "12"
            "15" -> "18"
            "16" -> "15"
            "17" -> "20"
            "18" -> "17"
            "19" -> "28"
            "20" -> "19"
            "21" -> "24"
            "22" -> "21"
            "23" -> "26"
            "24" -> "23"
            "25" -> "29"
            "26" -> "25"
            "27" -> "31"
            "28" -> "27"
            "29" -> "27"
            "31" -> "30"
            "32" -> "35"
            "33" -> "32"
            "34" -> "37"
            "35" -> "34"
            "37" -> "36"
            }
        ).";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(control.toDOT()));
    }

    TEST(ControlGraphTest, getSetCoordinates)
    {
        KernelGraph::KernelGraph kg;
        using namespace KernelGraph;
        auto five = Expression::literal(5);
        auto four = Expression::literal(4);

        int topSet1 = kg.control.addElement(SetCoordinate{five});
        int topSet2 = kg.control.addElement(SetCoordinate{five});

        int notTopSet1 = kg.control.addElement(SetCoordinate{four});
        int notTopSet2 = kg.control.addElement(SetCoordinate{five});

        int load1 = kg.control.addElement(LoadLDSTile{DataType::Float});
        int load2 = kg.control.addElement(LoadLDSTile{DataType::Float});

        kg.control.addElement(Body{}, {topSet1}, {load1});

        EXPECT_THROW(getTopSetCoordinate(kg, load1), FatalError);
        EXPECT_THROW(getSetCoordinateForDim(kg, 1, load1), FatalError);

        kg.mapper.connect<CT::Unroll>(topSet1, 1);
        EXPECT_EQ(topSet1, getTopSetCoordinate(kg, load1));
        EXPECT_THROW(getSetCoordinateForDim(kg, 2, load1), FatalError);
        EXPECT_EQ(topSet1, getSetCoordinateForDim(kg, 1, load1));

        kg.control.addElement(Body{}, {topSet2}, {notTopSet1});
        kg.control.addElement(Body{}, {notTopSet1}, {notTopSet2});
        kg.control.addElement(Body{}, {notTopSet2}, {load2});

        EXPECT_THROW(getTopSetCoordinate(kg, load2), FatalError);

        kg.mapper.connect<CT::Unroll>(topSet2, 2);
        kg.mapper.connect<CT::Unroll>(notTopSet1, 1);
        kg.mapper.connect<CT::Unroll>(notTopSet2, 3);
        EXPECT_EQ(topSet2, getTopSetCoordinate(kg, load2));

        EXPECT_THROW(getSetCoordinateForDim(kg, 5, load2), FatalError);
        EXPECT_EQ(topSet2, getSetCoordinateForDim(kg, 2, load2));
        EXPECT_EQ(notTopSet1, getSetCoordinateForDim(kg, 1, load2));
        EXPECT_EQ(notTopSet2, getSetCoordinateForDim(kg, 3, load2));

        EXPECT_EQ((std::vector{load1}), getLoadsForUnroll(kg, 1, {load1, load2}, 5));
        EXPECT_EQ((std::vector{load2}), getLoadsForUnroll(kg, 1, {load1, load2}, 4));

        EXPECT_EQ((std::set{topSet1, topSet2}), getTopSetCoordinates(kg, {load1, load2}));
    }
}
