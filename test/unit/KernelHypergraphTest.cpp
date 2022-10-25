#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/KernelGraph/ControlHypergraph/ControlEdge_fwd.hpp>
#include <rocRoller/KernelGraph/ControlHypergraph/ControlHypergraph.hpp>
#include <rocRoller/KernelGraph/ControlHypergraph/Operation_fwd.hpp>
#include <rocRoller/KernelGraph/CoordGraph/CoordinateHypergraph.hpp>
#include <rocRoller/KernelGraph/CoordGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/KernelHypergraph.hpp>

#include "SourceMatcher.hpp"

using namespace rocRoller;
using namespace KernelGraph;
using namespace KernelGraph::ControlHypergraph;
using namespace KernelGraph::CoordGraph;

namespace rocRollerTest
{

    TEST(KernelHypergraphTest, Basic)
    {
        KernelHypergraph kgraph = KernelHypergraph();

        // Control Graph
        int kernel_index = kgraph.control.addElement(Kernel());
        int loadA_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int loadB_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int body1_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadA_index});
        int body2_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadB_index});

        int op1_index       = kgraph.control.addElement(Assign());
        int sequence1_index = kgraph.control.addElement(Sequence(), {loadA_index}, {op1_index});
        int sequence2_index = kgraph.control.addElement(Sequence(), {loadB_index}, {op1_index});

        int op2_index       = kgraph.control.addElement(Assign());
        int sequence3_index = kgraph.control.addElement(Sequence(), {op1_index}, {op2_index});

        int op3_index       = kgraph.control.addElement(Assign());
        int sequence4_index = kgraph.control.addElement(Sequence(), {op1_index}, {op3_index});
        int sequence5_index = kgraph.control.addElement(Sequence(), {op2_index}, {op3_index});

        int storeC_index    = kgraph.control.addElement(StoreLinear());
        int sequence6_index = kgraph.control.addElement(Sequence(), {op3_index}, {storeC_index});

        // Coordinate Graph
        int u1_index       = kgraph.coordinates.addElement(User());
        int sd1_index      = kgraph.coordinates.addElement(SubDimension());
        int split1_index   = kgraph.coordinates.addElement(Split(), {u1_index}, {sd1_index});
        int linear1_index  = kgraph.coordinates.addElement(Linear());
        int flatten1_index = kgraph.coordinates.addElement(Flatten(), {sd1_index}, {linear1_index});
        int dataflow1_index
            = kgraph.coordinates.addElement(DataFlow(), {u1_index}, {linear1_index});

        int u2_index       = kgraph.coordinates.addElement(User());
        int sd2_index      = kgraph.coordinates.addElement(SubDimension());
        int split2_index   = kgraph.coordinates.addElement(Split(), {u2_index}, {sd2_index});
        int linear2_index  = kgraph.coordinates.addElement(Linear());
        int flatten2_index = kgraph.coordinates.addElement(Flatten(), {sd2_index}, {linear2_index});
        int dataflow2_index
            = kgraph.coordinates.addElement(DataFlow(), {u2_index}, {linear2_index});

        int linear3_index   = kgraph.coordinates.addElement(Linear());
        int dataflow3_index = kgraph.coordinates.addElement(
            DataFlow(), {linear1_index, linear2_index}, {linear3_index});
        int linear4_index = kgraph.coordinates.addElement(Linear());
        int dataflow4_index
            = kgraph.coordinates.addElement(DataFlow(), {linear3_index}, {linear4_index});
        int linear5i_index  = kgraph.coordinates.addElement(Linear());
        int dataflow5_index = kgraph.coordinates.addElement(
            DataFlow(), {linear3_index, linear4_index}, {linear5i_index});

        int linear5o_index = kgraph.coordinates.addElement(Linear());
        int makeoutput1_index
            = kgraph.coordinates.addElement(MakeOutput(), {linear5i_index}, {linear5o_index});
        int sd5o_index   = kgraph.coordinates.addElement(SubDimension(0));
        int split3_index = kgraph.coordinates.addElement(Split(), {linear5o_index}, {sd5o_index});
        int u5o_index    = kgraph.coordinates.addElement(User(""));
        int join1_index  = kgraph.coordinates.addElement(Join(), {sd5o_index}, {u5o_index});
        int dataflow6_index
            = kgraph.coordinates.addElement(DataFlow(), {linear5i_index}, {u5o_index});

        std::string expected = R".(
	    digraph {
		"coord1"[label="User{NA}(1)"];
		"coord2"[label="SubDimension{0, NA}(2)"];
		"coord3"[label="Split(3)",shape=box];
		"coord4"[label="Linear{NA}(4)"];
		"coord5"[label="Flatten(5)",shape=box];
		"coord6"[label="DataFlow(6)",shape=box];
		"coord7"[label="User{NA}(7)"];
		"coord8"[label="SubDimension{0, NA}(8)"];
		"coord9"[label="Split(9)",shape=box];
		"coord10"[label="Linear{NA}(10)"];
		"coord11"[label="Flatten(11)",shape=box];
		"coord12"[label="DataFlow(12)",shape=box];
		"coord13"[label="Linear{NA}(13)"];
		"coord14"[label="DataFlow(14)",shape=box];
		"coord15"[label="Linear{NA}(15)"];
		"coord16"[label="DataFlow(16)",shape=box];
		"coord17"[label="Linear{NA}(17)"];
		"coord18"[label="DataFlow(18)",shape=box];
		"coord19"[label="Linear{NA}(19)"];
		"coord20"[label="MakeOutput(20)",shape=box];
		"coord21"[label="SubDimension{0, NA}(21)"];
		"coord22"[label="Split(22)",shape=box];
		"coord23"[label="User{NA}(23)"];
		"coord24"[label="Join(24)",shape=box];
		"coord25"[label="DataFlow(25)",shape=box];
		"coord1" -> "coord3"
		"coord1" -> "coord6"
		"coord2" -> "coord5"
		"coord3" -> "coord2"
		"coord4" -> "coord14"
		"coord5" -> "coord4"
		"coord6" -> "coord4"
		"coord7" -> "coord9"
		"coord7" -> "coord12"
		"coord8" -> "coord11"
		"coord9" -> "coord8"
		"coord10" -> "coord14"
		"coord11" -> "coord10"
		"coord12" -> "coord10"
		"coord13" -> "coord16"
		"coord13" -> "coord18"
		"coord14" -> "coord13"
		"coord15" -> "coord18"
		"coord16" -> "coord15"
		"coord17" -> "coord20"
		"coord17" -> "coord25"
		"coord18" -> "coord17"
		"coord19" -> "coord22"
		"coord20" -> "coord19"
		"coord21" -> "coord24"
		"coord22" -> "coord21"
		"coord24" -> "coord23"
		"coord25" -> "coord23"
		{
		    rank=same
		    "coord4"->"coord10"[style=invis]
		    rankdir=LR
		}
		{
		    rank=same
		    "coord13"->"coord15"[style=invis]
		    rankdir=LR
		}

		subgraph clusterCF {"cntrl1"[label="Kernel(1)"];
		"cntrl2"[label="LoadLinear(2)"];
		"cntrl3"[label="LoadLinear(3)"];
		"cntrl4"[label="Body(4)",shape=box];
		"cntrl5"[label="Body(5)",shape=box];
		"cntrl6"[label="Assign Literal nullptr(6)"];
		"cntrl7"[label="Sequence(7)",shape=box];
		"cntrl8"[label="Sequence(8)",shape=box];
		"cntrl9"[label="Assign Literal nullptr(9)"];
		"cntrl10"[label="Sequence(10)",shape=box];
		"cntrl11"[label="Assign Literal nullptr(11)"];
		"cntrl12"[label="Sequence(12)",shape=box];
		"cntrl13"[label="Sequence(13)",shape=box];
		"cntrl14"[label="StoreLinear(14)"];
		"cntrl15"[label="Sequence(15)",shape=box];
		"cntrl1" -> "cntrl4"
		"cntrl1" -> "cntrl5"
		"cntrl2" -> "cntrl7"
		"cntrl3" -> "cntrl8"
		"cntrl4" -> "cntrl2"
		"cntrl5" -> "cntrl3"
		"cntrl6" -> "cntrl10"
		"cntrl6" -> "cntrl12"
		"cntrl7" -> "cntrl6"
		"cntrl8" -> "cntrl6"
		"cntrl9" -> "cntrl13"
		"cntrl10" -> "cntrl9"
		"cntrl11" -> "cntrl15"
		"cntrl12" -> "cntrl11"
		"cntrl13" -> "cntrl11"
		"cntrl15" -> "cntrl14"
	    }
            }
        ).";

        EXPECT_EQ(NormalizedSource(expected), NormalizedSource(kgraph.toDOT()));
    }
}
