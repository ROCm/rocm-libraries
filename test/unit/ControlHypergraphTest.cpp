#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/KernelGraph/ControlHypergraph/ControlEdge_fwd.hpp>
#include <rocRoller/KernelGraph/ControlHypergraph/ControlHypergraph.hpp>
#include <rocRoller/KernelGraph/ControlHypergraph/Operation_fwd.hpp>

#include "SourceMatcher.hpp"

using namespace rocRoller;
using namespace KernelGraph::ControlHypergraph;

namespace rocRollerTest
{

    TEST(ControlHypergraphTest, Basic)
    {
        ControlHypergraph control = ControlHypergraph();

        int kernel_index = control.addElement(Kernel());
        int loadA_index  = control.addElement(LoadLinear());
        int loadB_index  = control.addElement(LoadLinear());
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

        auto outputs = control.getOutputIndices<Body>(kernel_index).to<std::vector>();
        EXPECT_EQ(2, outputs.size());

        auto outputs2 = control.getOutputIndices<Sequence>(kernel_index).to<std::vector>();
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

        auto inputs = control.getInputIndices<Body>(loadA_index).to<std::vector>();
        ASSERT_EQ(1, inputs.size());

        auto inputs2 = control.getInputIndices<Sequence>(loadA_index).to<std::vector>();
        EXPECT_EQ(0, inputs2.size());

        auto inputs3 = control.getInputIndices<Initialize>(loadA_index).to<std::vector>();
        EXPECT_EQ(0, inputs3.size());

        auto inputs4 = control.getInputIndices<ForLoopIncrement>(loadA_index).to<std::vector>();
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

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(control.toDOT()));
    }
}
