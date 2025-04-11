/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Reindexer.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Random.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Timer.hpp>

#include <common/CommonGraphs.hpp>

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"
#include "Utilities.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using ::testing::HasSubstr;

namespace KernelGraphTest
{

    class KernelGraphTestGPU : public CurrentGPUContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp() override
        {
            Settings::getInstance()->set(Settings::SaveAssembly, true);
            CurrentGPUContextFixture::SetUp();

            fastArith = Expression::FastArithmetic(m_context);
        }

        void GPU_SAXPBY(bool reload);
    };

    class KernelGraphTest : public GenericContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp() override
        {
            GenericContextFixture::SetUp();
            fastArith = Expression::FastArithmetic(m_context);
        }
    };

    // TODO: Add transforms and tests for VectorAdd with: 1. ForLoop
    // and 2. ForLoop+Unroll.
    //
    // The tests should also make sure the lowering fails if the
    // WorkitemCount is missing.
    //
    // See KernelGraphTestGPULoopSize :: MissingWorkitemCount
    //     KernelGraphTestGPULoopSize :: TestForLoop

    TEST_F(KernelGraphTest, BasicTranslateLinear)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto kgraph0 = example.getKernelGraph();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        auto visitor = rocRoller::KernelGraph::BaseGraphVisitor(m_context);
        auto kgraphC = rewrite(kgraph0, visitor);

        std::string expectedC = R".(
                digraph {
                "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
                "coord2"[label="User{CommandArgument(Tensor_2_extent)I64}(2)"];
                "coord3"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(3)"];
                "coord4"[label="Split(4)",shape=box];
                "coord5"[label="Linear{CommandArgument(Tensor_0_size_0)I64}(5)"];
                "coord6"[label="Flatten(6)",shape=box];
                "coord7"[label="DataFlow(7)",shape=box];
                "coord8"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(8)"];
                "coord9"[label="Split(9)",shape=box];
                "coord10"[label="Linear{CommandArgument(Tensor_2_size_0)I64}(10)"];
                "coord11"[label="Flatten(11)",shape=box];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord13"[label="Linear{NA}(13)"];
                "coord14"[label="DataFlow(14)",shape=box];
                "coord15"[label="Linear{NA}(15)"];
                "coord16"[label="DataFlow(16)",shape=box];
                "coord17"[label="Linear{NA}(17)"];
                "coord18"[label="DataFlow(18)",shape=box];
                "coord19"[label="SubDimension{0, NA}(19)"];
                "coord20"[label="Split(20)",shape=box];
                "coord21"[label="User{CommandArgument(Tensor_8_extent)I64}(21)"];
                "coord22"[label="Join(22)",shape=box];
                "coord23"[label="DataFlow(23)",shape=box];
                "coord1" -> "coord4"
                "coord1" -> "coord7"
                "coord2" -> "coord9"
                "coord2" -> "coord12"
                "coord3" -> "coord6"
                "coord4" -> "coord3"
                "coord5" -> "coord14"
                "coord6" -> "coord5"
                "coord7" -> "coord5"
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
                "coord17" -> "coord23"
                "coord18" -> "coord17"
                "coord19" -> "coord22"
                "coord20" -> "coord19"
                "coord22" -> "coord21"
                "coord23" -> "coord21"
                {
                rank=same
                "coord5"->"coord10"[style=invis]
                rankdir=LR
                }
                {
                rank=same
                "coord13"->"coord15"[style=invis]
                rankdir=LR
                }
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadLinear Value: Int32(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadLinear Value: Int32(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(5)NA, DataFlowTag(10)NA)NA(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(13)NA)NA(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(13)NA, DataFlowTag(15)NA)NA(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl14"[label="StoreLinear(14)"];
                "cntrl15"[label="Sequence(15)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
                "cntrl2" -> "cntrl7"
                "cntrl3" -> "cntrl2"
                "cntrl4" -> "cntrl8"
                "cntrl5" -> "cntrl4"
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
                "coord1" -> "to_cntrl_1_2"
                "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_1"
                "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
                "coord5" -> "to_cntrl_5_2"
                "to_cntrl_5_2"[label="2->5: Linear: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_5"
                "to_coord_2_5"[label="2->5: Linear: (0)", shape=cds]
                "coord2" -> "to_cntrl_2_4"
                "to_cntrl_2_4"[label="4->2: User: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_2"
                "to_coord_4_2"[label="4->2: User: (0)", shape=cds]
                "coord10" -> "to_cntrl_10_4"
                "to_cntrl_10_4"[label="4->10: Linear: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_10"
                "to_coord_4_10"[label="4->10: Linear: (0)", shape=cds]
                "coord13" -> "to_cntrl_13_6"
                "to_cntrl_13_6"[label="6->13: DEST", shape=cds]
                "cntrl6" -> "to_coord_6_13"
                "to_coord_6_13"[label="6->13: DEST", shape=cds]
                "coord15" -> "to_cntrl_15_9"
                "to_cntrl_15_9"[label="9->15: DEST", shape=cds]
                "cntrl9" -> "to_coord_9_15"
                "to_coord_9_15"[label="9->15: DEST", shape=cds]
                "coord17" -> "to_cntrl_17_11"
                "to_cntrl_17_11"[label="11->17: DEST", shape=cds]
                "cntrl11" -> "to_coord_11_17"
                "to_coord_11_17"[label="11->17: DEST", shape=cds]
                "coord17" -> "to_cntrl_17_14"
                "to_cntrl_17_14"[label="14->17: Linear: (0)", shape=cds]
                "cntrl14" -> "to_coord_14_17"
                "to_coord_14_17"[label="14->17: Linear: (0)", shape=cds]
                "coord21" -> "to_cntrl_21_14"
                "to_cntrl_21_14"[label="14->21: User: (0)", shape=cds]
                "cntrl14" -> "to_coord_14_21"
                "to_coord_14_21"[label="14->21: User: (0)", shape=cds]
                }).";

        EXPECT_EQ(NormalizedSource(expectedC), NormalizedSource(kgraphC.toDOT(true)));

        std::string expected0 = R".(
                digraph {
                "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
                "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
                "coord3"[label="Split(3)",shape=box];
                "coord4"[label="Linear{CommandArgument(Tensor_0_size_0)I64}(4)"];
                "coord5"[label="Flatten(5)",shape=box];
                "coord6"[label="DataFlow(6)",shape=box];
                "coord7"[label="User{CommandArgument(Tensor_2_extent)I64}(7)"];
                "coord8"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(8)"];
                "coord9"[label="Split(9)",shape=box];
                "coord10"[label="Linear{CommandArgument(Tensor_2_size_0)I64}(10)"];
                "coord11"[label="Flatten(11)",shape=box];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord13"[label="Linear{NA}(13)"];
                "coord14"[label="DataFlow(14)",shape=box];
                "coord15"[label="Linear{NA}(15)"];
                "coord16"[label="DataFlow(16)",shape=box];
                "coord17"[label="Linear{NA}(17)"];
                "coord18"[label="DataFlow(18)",shape=box];
                "coord19"[label="SubDimension{0, NA}(19)"];
                "coord20"[label="User{CommandArgument(Tensor_8_extent)I64}(20)"];
                "coord21"[label="Split(21)",shape=box];
                "coord22"[label="Join(22)",shape=box];
                "coord23"[label="DataFlow(23)",shape=box];
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
                "coord17" -> "coord21"
                "coord17" -> "coord23"
                "coord18" -> "coord17"
                "coord19" -> "coord22"
                "coord21" -> "coord19"
                "coord22" -> "coord20"
                "coord23" -> "coord20"
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
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadLinear Value: Int32(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadLinear Value: Int32(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(4)NA, DataFlowTag(10)NA)NA(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(13)NA)NA(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(13)NA, DataFlowTag(15)NA)NA(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl14"[label="StoreLinear(14)"];
                "cntrl15"[label="Sequence(15)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
                "cntrl2" -> "cntrl7"
                "cntrl3" -> "cntrl2"
                "cntrl4" -> "cntrl8"
                "cntrl5" -> "cntrl4"
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
                "coord1" -> "to_cntrl_1_2"
                "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_1"
                "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
                "coord4" -> "to_cntrl_4_2"
                "to_cntrl_4_2"[label="2->4: Linear: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_4"
                "to_coord_2_4"[label="2->4: Linear: (0)", shape=cds]
                "coord7" -> "to_cntrl_7_4"
                "to_cntrl_7_4"[label="4->7: User: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_7"
                "to_coord_4_7"[label="4->7: User: (0)", shape=cds]
                "coord10" -> "to_cntrl_10_4"
                "to_cntrl_10_4"[label="4->10: Linear: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_10"
                "to_coord_4_10"[label="4->10: Linear: (0)", shape=cds]
                "coord13" -> "to_cntrl_13_6"
                "to_cntrl_13_6"[label="6->13: DEST", shape=cds]
                "cntrl6" -> "to_coord_6_13"
                "to_coord_6_13"[label="6->13: DEST", shape=cds]
                "coord15" -> "to_cntrl_15_9"
                "to_cntrl_15_9"[label="9->15: DEST", shape=cds]
                "cntrl9" -> "to_coord_9_15"
                "to_coord_9_15"[label="9->15: DEST", shape=cds]
                "coord17" -> "to_cntrl_17_11"
                "to_cntrl_17_11"[label="11->17: DEST", shape=cds]
                "cntrl11" -> "to_coord_11_17"
                "to_coord_11_17"[label="11->17: DEST", shape=cds]
                "coord17" -> "to_cntrl_17_14"
                "to_cntrl_17_14"[label="14->17: Linear: (0)", shape=cds]
                "cntrl14" -> "to_coord_14_17"
                "to_coord_14_17"[label="14->17: Linear: (0)", shape=cds]
                "coord20" -> "to_cntrl_20_14"
                "to_cntrl_20_14"[label="14->20: User: (0)", shape=cds]
                "cntrl14" -> "to_coord_14_20"
                "to_coord_14_20"[label="14->20: User: (0)", shape=cds]
         }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));

        std::string expected1 = R".(
            digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="User{CommandArgument(Tensor_2_extent)I64}(2)"];
        "coord3"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(3)"];
        "coord4"[label="Split(4)",shape=box];
        "coord5"[label="Linear{CommandArgument(Tensor_0_size_0)I64}(5)"];
        "coord6"[label="Flatten(6)",shape=box];
        "coord7"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(7)"];
        "coord8"[label="Split(8)",shape=box];
        "coord9"[label="Linear{CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="Flatten(10)",shape=box];
        "coord11"[label="Linear{NA}(11)"];
        "coord12"[label="SubDimension{0, NA}(12)"];
        "coord13"[label="Split(13)",shape=box];
        "coord14"[label="User{CommandArgument(Tensor_8_extent)I64}(14)"];
        "coord15"[label="Join(15)",shape=box];
        "coord16"[label="VGPR{NA}(16)"];
        "coord17"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_0_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(17)"];
        "coord18"[label="Workitem{0, 64:U32}(18)"];
        "coord19"[label="Tile(19)",shape=box];
        "coord20"[label="Forget(20)",shape=box];
        "coord21"[label="DataFlow(21)",shape=box];
        "coord22"[label="VGPR{NA}(22)"];
        "coord23"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_2_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(23)"];
        "coord24"[label="Workitem{0, 64:U32}(24)"];
        "coord25"[label="Tile(25)",shape=box];
        "coord26"[label="Forget(26)",shape=box];
        "coord27"[label="DataFlow(27)",shape=box];
        "coord28"[label="VGPR{NA}(28)"];
        "coord29"[label="DataFlow(29)",shape=box];
        "coord30"[label="VGPR{NA}(30)"];
        "coord31"[label="DataFlow(31)",shape=box];
        "coord32"[label="VGPR{NA}(32)"];
        "coord33"[label="DataFlow(33)",shape=box];
        "coord34"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_8_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(34)"];
        "coord35"[label="Workitem{0, 64:U32}(35)"];
        "coord36"[label="Inherit(36)",shape=box];
        "coord37"[label="Flatten(37)",shape=box];
        "coord38"[label="DataFlow(38)",shape=box];
        "coord1" -> "coord4"
        "coord1" -> "coord21"
        "coord2" -> "coord8"
        "coord2" -> "coord27"
        "coord3" -> "coord6"
        "coord4" -> "coord3"
        "coord5" -> "coord19"
        "coord6" -> "coord5"
        "coord7" -> "coord10"
        "coord8" -> "coord7"
        "coord9" -> "coord25"
        "coord10" -> "coord9"
        "coord11" -> "coord13"
        "coord12" -> "coord15"
        "coord13" -> "coord12"
        "coord15" -> "coord14"
        "coord16" -> "coord29"
        "coord17" -> "coord20"
        "coord18" -> "coord20"
        "coord19" -> "coord17"
        "coord19" -> "coord18"
        "coord20" -> "coord16"
        "coord21" -> "coord16"
        "coord22" -> "coord29"
        "coord23" -> "coord26"
        "coord24" -> "coord26"
        "coord25" -> "coord23"
        "coord25" -> "coord24"
        "coord26" -> "coord22"
        "coord27" -> "coord22"
        "coord28" -> "coord31"
        "coord28" -> "coord33"
        "coord29" -> "coord28"
        "coord30" -> "coord33"
        "coord31" -> "coord30"
        "coord32" -> "coord36"
        "coord32" -> "coord38"
        "coord33" -> "coord32"
        "coord34" -> "coord37"
        "coord35" -> "coord37"
        "coord36" -> "coord34"
        "coord36" -> "coord35"
        "coord37" -> "coord11"
        "coord38" -> "coord14"
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord23"->"coord24"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord23"->"coord24"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord16"->"coord22"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord28"->"coord30"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord34"->"coord35"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord34"->"coord35"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadVGPR Value: Int32(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadVGPR Value: Int32(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="Assign VGPR Add(DataFlowTag(16)NA, DataFlowTag(22)NA)NA(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="Assign VGPR Negate(DataFlowTag(28)NA)NA(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(28)NA, DataFlowTag(30)NA)NA(11)"];
        "cntrl12"[label="Sequence(12)",shape=box];
        "cntrl13"[label="Sequence(13)",shape=box];
        "cntrl14"[label="StoreVGPR(14)"];
        "cntrl15"[label="Sequence(15)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl8"
        "cntrl5" -> "cntrl4"
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
        "coord1" -> "to_cntrl_1_2"
        "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_1"
        "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
        "coord16" -> "to_cntrl_16_2"
        "to_cntrl_16_2"[label="2->16: VGPR: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_16"
        "to_coord_2_16"[label="2->16: VGPR: (0)", shape=cds]
        "coord2" -> "to_cntrl_2_4"
        "to_cntrl_2_4"[label="4->2: User: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_2"
        "to_coord_4_2"[label="4->2: User: (0)", shape=cds]
        "coord22" -> "to_cntrl_22_4"
        "to_cntrl_22_4"[label="4->22: VGPR: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_22"
        "to_coord_4_22"[label="4->22: VGPR: (0)", shape=cds]
        "coord28" -> "to_cntrl_28_6"
        "to_cntrl_28_6"[label="6->28: DEST", shape=cds]
        "cntrl6" -> "to_coord_6_28"
        "to_coord_6_28"[label="6->28: DEST", shape=cds]
        "coord30" -> "to_cntrl_30_9"
        "to_cntrl_30_9"[label="9->30: DEST", shape=cds]
        "cntrl9" -> "to_coord_9_30"
        "to_coord_9_30"[label="9->30: DEST", shape=cds]
        "coord32" -> "to_cntrl_32_11"
        "to_cntrl_32_11"[label="11->32: DEST", shape=cds]
        "cntrl11" -> "to_coord_11_32"
        "to_coord_11_32"[label="11->32: DEST", shape=cds]
        "coord14" -> "to_cntrl_14_14"
        "to_cntrl_14_14"[label="14->14: User: (0)", shape=cds]
        "cntrl14" -> "to_coord_14_14"
        "to_coord_14_14"[label="14->14: User: (0)", shape=cds]
        "coord32" -> "to_cntrl_32_14"
        "to_cntrl_32_14"[label="14->32: VGPR: (0)", shape=cds]
        "cntrl14" -> "to_coord_14_32"
        "to_coord_14_32"[label="14->32: VGPR: (0)", shape=cds]
        }).";

        auto one = Expression::literal(1u);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto lowerLinearTransform = std::make_shared<LowerLinear>(m_context);

        auto kgraph1 = kgraph0.transform(lowerLinearTransform);
        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph1.toDOT(true)));

        std::string expected2 = R".(
        digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="User{CommandArgument(Tensor_2_extent)I64}(2)"];
        "coord3"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(3)"];
        "coord4"[label="Split(4)",shape=box];
        "coord5"[label="Linear{CommandArgument(Tensor_0_size_0)I64}(5)"];
        "coord6"[label="Flatten(6)",shape=box];
        "coord7"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_0_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(7)"];
        "coord8"[label="Workitem{0, 64:U32}(8)"];
        "coord9"[label="Tile(9)",shape=box];
        "coord10"[label="Linear{16:I}(10)"];
        "coord11"[label="ForLoop{16:I}(11)"];
        "coord12"[label="DataFlow(12)",shape=box];
        "coord13"[label="VGPR{NA}(13)"];
        "coord14"[label="Forget(14)",shape=box];
        "coord15"[label="DataFlow(15)",shape=box];
        "coord16"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(16)"];
        "coord17"[label="Split(17)",shape=box];
        "coord18"[label="Linear{CommandArgument(Tensor_2_size_0)I64}(18)"];
        "coord19"[label="Flatten(19)",shape=box];
        "coord20"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_2_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(20)"];
        "coord21"[label="Workitem{0, 64:U32}(21)"];
        "coord22"[label="Tile(22)",shape=box];
        "coord23"[label="ForLoop{16:I}(23)"];
        "coord24"[label="DataFlow(24)",shape=box];
        "coord25"[label="VGPR{NA}(25)"];
        "coord26"[label="Forget(26)",shape=box];
        "coord27"[label="DataFlow(27)",shape=box];
        "coord28"[label="VGPR{NA}(28)"];
        "coord29"[label="DataFlow(29)",shape=box];
        "coord30"[label="VGPR{NA}(30)"];
        "coord31"[label="DataFlow(31)",shape=box];
        "coord32"[label="VGPR{NA}(32)"];
        "coord33"[label="DataFlow(33)",shape=box];
        "coord34"[label="Workgroup{0, Divide(Subtract(Add(CommandArgument(Tensor_8_extent)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64}(34)"];
        "coord35"[label="Workitem{0, 64:U32}(35)"];
        "coord36"[label="Inherit(36)",shape=box];
        "coord37"[label="ForLoop{16:I}(37)"];
        "coord38"[label="DataFlow(38)",shape=box];
        "coord39"[label="Linear{NA}(39)"];
        "coord40"[label="Flatten(40)",shape=box];
        "coord41"[label="SubDimension{0, NA}(41)"];
        "coord42"[label="Split(42)",shape=box];
        "coord43"[label="User{CommandArgument(Tensor_8_extent)I64}(43)"];
        "coord44"[label="Join(44)",shape=box];
        "coord45"[label="DataFlow(45)",shape=box];
        "coord1" -> "coord4"
        "coord1" -> "coord15"
        "coord2" -> "coord17"
        "coord2" -> "coord27"
        "coord3" -> "coord6"
        "coord4" -> "coord3"
        "coord5" -> "coord9"
        "coord6" -> "coord5"
        "coord7" -> "coord14"
        "coord8" -> "coord14"
        "coord9" -> "coord11"
        "coord9" -> "coord7"
        "coord9" -> "coord8"
        "coord10" -> "coord12"
        "coord10" -> "coord24"
        "coord10" -> "coord38"
        "coord11" -> "coord14"
        "coord12" -> "coord11"
        "coord13" -> "coord29"
        "coord14" -> "coord13"
        "coord15" -> "coord13"
        "coord16" -> "coord19"
        "coord17" -> "coord16"
        "coord18" -> "coord22"
        "coord19" -> "coord18"
        "coord20" -> "coord26"
        "coord21" -> "coord26"
        "coord22" -> "coord23"
        "coord22" -> "coord20"
        "coord22" -> "coord21"
        "coord23" -> "coord26"
        "coord24" -> "coord23"
        "coord25" -> "coord29"
        "coord26" -> "coord25"
        "coord27" -> "coord25"
        "coord28" -> "coord31"
        "coord28" -> "coord33"
        "coord29" -> "coord28"
        "coord30" -> "coord33"
        "coord31" -> "coord30"
        "coord32" -> "coord36"
        "coord32" -> "coord45"
        "coord33" -> "coord32"
        "coord34" -> "coord40"
        "coord35" -> "coord40"
        "coord36" -> "coord37"
        "coord36" -> "coord34"
        "coord36" -> "coord35"
        "coord37" -> "coord40"
        "coord38" -> "coord37"
        "coord39" -> "coord42"
        "coord40" -> "coord39"
        "coord41" -> "coord44"
        "coord42" -> "coord41"
        "coord44" -> "coord43"
        "coord45" -> "coord43"
        {
        rank=same
        "coord11"->"coord7"->"coord8"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord11"->"coord7"->"coord8"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord23"->"coord20"->"coord21"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord23"->"coord20"->"coord21"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord13"->"coord25"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord28"->"coord30"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord37"->"coord34"->"coord35"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord37"->"coord34"->"coord35"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="ForLoopOp : LessThan(DataFlowTag(10)I, 16:I)BL(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="Assign SGPR 0:I(4)"];
        "cntrl5"[label="Initialize(5)",shape=box];
        "cntrl6"[label="Assign SGPR Add(DataFlowTag(10)I, 1:I)I(6)"];
        "cntrl7"[label="ForLoopIncrement(7)",shape=box];
        "cntrl8"[label="LoadVGPR Value: Int32(8)"];
        "cntrl9"[label="Body(9)",shape=box];
        "cntrl10"[label="LoadVGPR Value: Int32(10)"];
        "cntrl11"[label="Body(11)",shape=box];
        "cntrl12"[label="Assign VGPR Add(DataFlowTag(13)NA, DataFlowTag(25)NA)NA(12)"];
        "cntrl13"[label="Sequence(13)",shape=box];
        "cntrl14"[label="Sequence(14)",shape=box];
        "cntrl15"[label="Assign VGPR Negate(DataFlowTag(28)NA)NA(15)"];
        "cntrl16"[label="Sequence(16)",shape=box];
        "cntrl17"[label="Assign VGPR Multiply(DataFlowTag(28)NA, DataFlowTag(30)NA)NA(17)"];
        "cntrl18"[label="Sequence(18)",shape=box];
        "cntrl19"[label="Sequence(19)",shape=box];
        "cntrl20"[label="StoreVGPR(20)"];
        "cntrl21"[label="Sequence(21)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl2" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl2" -> "cntrl9"
        "cntrl2" -> "cntrl11"
        "cntrl3" -> "cntrl2"
        "cntrl5" -> "cntrl4"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl13"
        "cntrl9" -> "cntrl8"
        "cntrl10" -> "cntrl14"
        "cntrl11" -> "cntrl10"
        "cntrl12" -> "cntrl16"
        "cntrl12" -> "cntrl18"
        "cntrl13" -> "cntrl12"
        "cntrl14" -> "cntrl12"
        "cntrl15" -> "cntrl19"
        "cntrl16" -> "cntrl15"
        "cntrl17" -> "cntrl21"
        "cntrl18" -> "cntrl17"
        "cntrl19" -> "cntrl17"
        "cntrl21" -> "cntrl20"
        }
        "coord10" -> "to_cntrl_10_2"
        "to_cntrl_10_2"[label="2->10: ForLoop: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_10"
        "to_coord_2_10"[label="2->10: ForLoop: (0)", shape=cds]
        "coord10" -> "to_cntrl_10_4"
        "to_cntrl_10_4"[label="4->10: DEST", shape=cds]
        "cntrl4" -> "to_coord_4_10"
        "to_coord_4_10"[label="4->10: DEST", shape=cds]
        "coord10" -> "to_cntrl_10_6"
        "to_cntrl_10_6"[label="6->10: DEST", shape=cds]
        "cntrl6" -> "to_coord_6_10"
        "to_coord_6_10"[label="6->10: DEST", shape=cds]
        "coord1" -> "to_cntrl_1_8"
        "to_cntrl_1_8"[label="8->1: User: (0)", shape=cds]
        "cntrl8" -> "to_coord_8_1"
        "to_coord_8_1"[label="8->1: User: (0)", shape=cds]
        "coord13" -> "to_cntrl_13_8"
        "to_cntrl_13_8"[label="8->13: VGPR: (0)", shape=cds]
        "cntrl8" -> "to_coord_8_13"
        "to_coord_8_13"[label="8->13: VGPR: (0)", shape=cds]
        "coord2" -> "to_cntrl_2_10"
        "to_cntrl_2_10"[label="10->2: User: (0)", shape=cds]
        "cntrl10" -> "to_coord_10_2"
        "to_coord_10_2"[label="10->2: User: (0)", shape=cds]
        "coord25" -> "to_cntrl_25_10"
        "to_cntrl_25_10"[label="10->25: VGPR: (0)", shape=cds]
        "cntrl10" -> "to_coord_10_25"
        "to_coord_10_25"[label="10->25: VGPR: (0)", shape=cds]
        "coord28" -> "to_cntrl_28_12"
        "to_cntrl_28_12"[label="12->28: DEST", shape=cds]
        "cntrl12" -> "to_coord_12_28"
        "to_coord_12_28"[label="12->28: DEST", shape=cds]
        "coord30" -> "to_cntrl_30_15"
        "to_cntrl_30_15"[label="15->30: DEST", shape=cds]
        "cntrl15" -> "to_coord_15_30"
        "to_coord_15_30"[label="15->30: DEST", shape=cds]
        "coord32" -> "to_cntrl_32_17"
        "to_cntrl_32_17"[label="17->32: DEST", shape=cds]
        "cntrl17" -> "to_coord_17_32"
        "to_coord_17_32"[label="17->32: DEST", shape=cds]
        "coord32" -> "to_cntrl_32_20"
        "to_cntrl_32_20"[label="20->32: VGPR: (0)", shape=cds]
        "cntrl20" -> "to_coord_20_32"
        "to_coord_20_32"[label="20->32: VGPR: (0)", shape=cds]
        "coord43" -> "to_cntrl_43_20"
        "to_cntrl_43_20"[label="20->43: User: (0)", shape=cds]
        "cntrl20" -> "to_coord_20_43"
        "to_coord_20_43"[label="20->43: User: (0)", shape=cds]
        }).";

        int  loopSize     = 16;
        auto loopSizeExpr = Expression::literal(loopSize);

        auto lowerLinerLoopTransform = std::make_shared<LowerLinearLoop>(loopSizeExpr, m_context);

        auto kgraph2 = kgraph1.transform(lowerLinerLoopTransform);
        EXPECT_EQ(NormalizedSource(expected2), NormalizedSource(kgraph2.toDOT(true)));
    }

    TEST_F(KernelGraphTest, BasicTranslateScalar)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>(true);
        auto kgraph0 = example.getKernelGraph();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        std::string expected0 = R".(
                digraph {
                "coord1"[label="User{NA}(1)"];
                "coord2"[label="VGPR{NA}(2)"];
                "coord3"[label="DataFlow(3)",shape=box];
                "coord4"[label="User{NA}(4)"];
                "coord5"[label="VGPR{NA}(5)"];
                "coord6"[label="DataFlow(6)",shape=box];
                "coord7"[label="VGPR{NA}(7)"];
                "coord8"[label="DataFlow(8)",shape=box];
                "coord9"[label="VGPR{NA}(9)"];
                "coord10"[label="DataFlow(10)",shape=box];
                "coord11"[label="VGPR{NA}(11)"];
                "coord12"[label="DataFlow(12)",shape=box];
                "coord1" -> "coord3"
                "coord2" -> "coord8"
                "coord3" -> "coord2"
                "coord4" -> "coord6"
                "coord5" -> "coord8"
                "coord6" -> "coord5"
                "coord7" -> "coord10"
                "coord7" -> "coord12"
                "coord8" -> "coord7"
                "coord9" -> "coord12"
                "coord10" -> "coord9"
                "coord12" -> "coord11"
                {
                rank=same
                "coord2"->"coord5"[style=invis]
                rankdir=LR
                }
                {
                rank=same
                "coord7"->"coord9"[style=invis]
                rankdir=LR
                }
                subgraph clusterCF {label = "Control Graph";
                "cntrl1"[label="Kernel(1)"];
                "cntrl2"[label="LoadVGPR Value: Int32(2)"];
                "cntrl3"[label="Body(3)",shape=box];
                "cntrl4"[label="LoadVGPR Value: Int32(4)"];
                "cntrl5"[label="Body(5)",shape=box];
                "cntrl6"[label="Assign VGPR Add(DataFlowTag(2)NA, DataFlowTag(5)NA)NA(6)"];
                "cntrl7"[label="Sequence(7)",shape=box];
                "cntrl8"[label="Sequence(8)",shape=box];
                "cntrl9"[label="Assign VGPR Negate(DataFlowTag(7)NA)NA(9)"];
                "cntrl10"[label="Sequence(10)",shape=box];
                "cntrl11"[label="Assign VGPR Multiply(DataFlowTag(7)NA, DataFlowTag(9)NA)NA(11)"];
                "cntrl12"[label="Sequence(12)",shape=box];
                "cntrl13"[label="Sequence(13)",shape=box];
                "cntrl1" -> "cntrl3"
                "cntrl1" -> "cntrl5"
                "cntrl2" -> "cntrl7"
                "cntrl3" -> "cntrl2"
                "cntrl4" -> "cntrl8"
                "cntrl5" -> "cntrl4"
                "cntrl6" -> "cntrl10"
                "cntrl6" -> "cntrl12"
                "cntrl7" -> "cntrl6"
                "cntrl8" -> "cntrl6"
                "cntrl9" -> "cntrl13"
                "cntrl10" -> "cntrl9"
                "cntrl12" -> "cntrl11"
                "cntrl13" -> "cntrl11"
                }
                "coord1" -> "to_cntrl_1_2"
                "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_1"
                "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
                "coord2" -> "to_cntrl_2_2"
                "to_cntrl_2_2"[label="2->2: VGPR: (0)", shape=cds]
                "cntrl2" -> "to_coord_2_2"
                "to_coord_2_2"[label="2->2: VGPR: (0)", shape=cds]
                "coord4" -> "to_cntrl_4_4"
                "to_cntrl_4_4"[label="4->4: User: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_4"
                "to_coord_4_4"[label="4->4: User: (0)", shape=cds]
                "coord5" -> "to_cntrl_5_4"
                "to_cntrl_5_4"[label="4->5: VGPR: (0)", shape=cds]
                "cntrl4" -> "to_coord_4_5"
                "to_coord_4_5"[label="4->5: VGPR: (0)", shape=cds]
                "coord7" -> "to_cntrl_7_6"
                "to_cntrl_7_6"[label="6->7: DEST", shape=cds]
                "cntrl6" -> "to_coord_6_7"
                "to_coord_6_7"[label="6->7: DEST", shape=cds]
                "coord9" -> "to_cntrl_9_9"
                "to_cntrl_9_9"[label="9->9: DEST", shape=cds]
                "cntrl9" -> "to_coord_9_9"
                "to_coord_9_9"[label="9->9: DEST", shape=cds]
                "coord11" -> "to_cntrl_11_11"
                "to_cntrl_11_11"[label="11->11: DEST", shape=cds]
                "cntrl11" -> "to_coord_11_11"
                "to_coord_11_11"[label="11->11: DEST", shape=cds]
             }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));
    }

    TEST_F(KernelGraphTest, TranslateMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::Int32);
        auto kgraph0 = example.getKernelGraph();

        auto bottom = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(bottom.size(), 2);
        for(auto const& id : bottom)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        auto top = kgraph0.coordinates.leaves().to<std::vector>();
        EXPECT_EQ(top.size(), 1);
        for(auto const& id : top)
        {
            EXPECT_TRUE(kgraph0.coordinates.get<User>(id).has_value());
        }

        std::string expected0 = R".(
        digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
        "coord3"[label="SubDimension{1, CommandArgument(Tensor_0_size_1)I64}(3)"];
        "coord4"[label="MacroTile{NA}(2/None/None){}-()(4)"];
        "coord5"[label="Split(5)",shape=box];
        "coord6"[label="ConstructMacroTile(6)",shape=box];
        "coord7"[label="DataFlow(7)",shape=box];
        "coord8"[label="User{CommandArgument(Tensor_2_extent)I64}(8)"];
        "coord9"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="SubDimension{1, CommandArgument(Tensor_2_size_1)I64}(10)"];
        "coord11"[label="MacroTile{NA}(2/None/None){}-()(11)"];
        "coord12"[label="Split(12)",shape=box];
        "coord13"[label="ConstructMacroTile(13)",shape=box];
        "coord14"[label="DataFlow(14)",shape=box];
        "coord15"[label="MacroTile{NA}(0/None/None){}-()(15)"];
        "coord16"[label="DataFlow(16)",shape=box];
        "coord17"[label="SubDimension{0, NA}(17)"];
        "coord18"[label="SubDimension{1, NA}(18)"];
        "coord19"[label="User{CommandArgument(Tensor_5_extent)I64}(19)"];
        "coord20"[label="DestructMacroTile(20)",shape=box];
        "coord21"[label="Join(21)",shape=box];
        "coord22"[label="DataFlow(22)",shape=box];
        "coord1" -> "coord5"
        "coord1" -> "coord7"
        "coord2" -> "coord6"
        "coord3" -> "coord6"
        "coord4" -> "coord16"
        "coord5" -> "coord2"
        "coord5" -> "coord3"
        "coord6" -> "coord4"
        "coord7" -> "coord4"
        "coord8" -> "coord12"
        "coord8" -> "coord14"
        "coord9" -> "coord13"
        "coord10" -> "coord13"
        "coord11" -> "coord16"
        "coord12" -> "coord9"
        "coord12" -> "coord10"
        "coord13" -> "coord11"
        "coord14" -> "coord11"
        "coord15" -> "coord20"
        "coord15" -> "coord22"
        "coord16" -> "coord15"
        "coord17" -> "coord21"
        "coord18" -> "coord21"
        "coord20" -> "coord17"
        "coord20" -> "coord18"
        "coord21" -> "coord19"
        "coord22" -> "coord19"
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord4"->"coord11"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadTiled Value: Int32(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadTiled Value: Int32(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="TensorContraction(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="StoreTiled Value: Int32(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl8"
        "cntrl5" -> "cntrl4"
        "cntrl6" -> "cntrl10"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl6"
        "cntrl10" -> "cntrl9"
        }
        "coord1" -> "to_cntrl_1_2"
        "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_1"
        "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
        "coord4" -> "to_cntrl_4_2"
        "to_cntrl_4_2"[label="2->4: MacroTile: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_4"
        "to_coord_2_4"[label="2->4: MacroTile: (0)", shape=cds]
        "coord8" -> "to_cntrl_8_4"
        "to_cntrl_8_4"[label="4->8: User: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_8"
        "to_coord_4_8"[label="4->8: User: (0)", shape=cds]
        "coord11" -> "to_cntrl_11_4"
        "to_cntrl_11_4"[label="4->11: MacroTile: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_11"
        "to_coord_4_11"[label="4->11: MacroTile: (0)", shape=cds]
        "coord4" -> "to_cntrl_4_6"
        "to_cntrl_4_6"[label="6->4: LHS", shape=cds]
        "cntrl6" -> "to_coord_6_4"
        "to_coord_6_4"[label="6->4: LHS", shape=cds]
        "coord11" -> "to_cntrl_11_6"
        "to_cntrl_11_6"[label="6->11: RHS", shape=cds]
        "cntrl6" -> "to_coord_6_11"
        "to_coord_6_11"[label="6->11: RHS", shape=cds]
        "coord15" -> "to_cntrl_15_6"
        "to_cntrl_15_6"[label="6->15: DEST", shape=cds]
        "cntrl6" -> "to_coord_6_15"
        "to_coord_6_15"[label="6->15: DEST", shape=cds]
        "coord15" -> "to_cntrl_15_9"
        "to_cntrl_15_9"[label="9->15: MacroTile: (0)", shape=cds]
        "cntrl9" -> "to_coord_9_15"
        "to_coord_9_15"[label="9->15: MacroTile: (0)", shape=cds]
        "coord19" -> "to_cntrl_19_9"
        "to_cntrl_19_9"[label="9->19: User: (0)", shape=cds]
        "cntrl9" -> "to_coord_9_19"
        "to_coord_9_19"[label="9->19: User: (0)", shape=cds]
        }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));
    }

    TEST_F(KernelGraphTest, TranslateUnscaledMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::FP8,
                                                             DataType::FP6,
                                                             DataType::Float,
                                                             Operations::ScaleMode::None,
                                                             Operations::ScaleMode::None);

        auto expectedCommand = R".(
        Tensor.FP8.d2 0, (base=&0, lim=&8, sizes={&16 &24 }, strides={&32 &40 })
        T_LOAD_TILED 1 Source 0
        Tensor.FP6.d2 2, (base=&48, lim=&56, sizes={&64 &72 }, strides={&80 &88 })
        T_LOAD_TILED 3 Source 2
        T_Mul 1 3 Value: Float
        Tensor.Float.d2 5, (base=&96, lim=&104, sizes={&112 &120 }, strides={&128 &136 })
        T_STORE_TILED 6 Source 4 Dest 5
        ).";

        EXPECT_EQ(NormalizedSource(expectedCommand),
                  NormalizedSource(example.getCommand()->toString()));

        auto kgraph0 = example.getKernelGraph();

        auto roots = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(roots.size(), 2);
        for(auto const& id : roots)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto leaves = kgraph0.coordinates.leaves().to<std::vector>();
        ASSERT_EQ(leaves.size(), 1);
        for(auto const& id : leaves)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto reachableFromLeaf
            = kgraph0.coordinates.depthFirstVisit(leaves[0], Graph::Direction::Upstream)
                  .to<std::set>();

        for(auto id : roots)
        {
            EXPECT_TRUE(reachableFromLeaf.contains(id)) << ShowValue(id) << reachableFromLeaf;
        }

        // There should be 4 LoadTiled nodes with direct sequence edges to the single TensorContraction node.
        // These should be all of the LoadTiled nodes in the graph.
        auto is_tc = kgraph0.control.isElemType<TensorContraction>();
        auto tc_id = only(kgraph0.control.findElements(is_tc));
        ASSERT_TRUE(tc_id.has_value());

        auto is_loadTiled  = kgraph0.control.isElemType<LoadTiled>();
        auto loadTiled_ids = kgraph0.control.findElements(is_loadTiled).to<std::set>();

        auto tc_parent_ids = kgraph0.control.parentNodes(*tc_id).to<std::set>();
        EXPECT_EQ(loadTiled_ids, tc_parent_ids);
        EXPECT_EQ(2, loadTiled_ids.size());
    }

    TEST_F(KernelGraphTest, TranslateScaledMatrixMultiply)
    {
        auto example = rocRollerTest::Graphs::MatrixMultiply(DataType::FP8,
                                                             DataType::FP6,
                                                             DataType::Float,
                                                             Operations::ScaleMode::Separate,
                                                             Operations::ScaleMode::Separate);

        auto expectedCommand = R".(
        Tensor.FP8.d2 0, (base=&0, lim=&8, sizes={&16 &24 }, strides={&32 &40 })
        T_LOAD_TILED 1 Source 0
        Tensor.UInt8.d2 2, (base=&48, lim=&56, sizes={&64 &72 }, strides={&80 &88 })
        T_LOAD_TILED 3 Source 2
        BlockScale(Separate, {1, 32}): Data: 1, Scale: 3
        Tensor.FP6.d2 5, (base=&96, lim=&104, sizes={&112 &120 }, strides={&128 &136 })
        T_LOAD_TILED 6 Source 5
        Tensor.UInt8.d2 7, (base=&144, lim=&152, sizes={&160 &168 }, strides={&176 &184 })
        T_LOAD_TILED 8 Source 7
        BlockScale(Separate, {32, 1}): Data: 6, Scale: 8
        T_Mul 4 9 Value: Float
        Tensor.Float.d2 11, (base=&192, lim=&200, sizes={&208 &216 }, strides={&224 &232 })
        T_STORE_TILED 12 Source 10 Dest 11
        ).";

        EXPECT_EQ(NormalizedSource(expectedCommand),
                  NormalizedSource(example.getCommand()->toString()));

        auto kgraph0 = example.getKernelGraph();

        auto roots = kgraph0.coordinates.roots().to<std::vector>();
        EXPECT_EQ(roots.size(), 4);
        for(auto const& id : roots)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto leaves = kgraph0.coordinates.leaves().to<std::vector>();
        ASSERT_EQ(leaves.size(), 1);
        for(auto const& id : leaves)
        {
            EXPECT_TRUE(std::holds_alternative<User>(
                std::get<Dimension>(kgraph0.coordinates.getElement(id))));
        }

        auto reachableFromLeaf
            = kgraph0.coordinates.depthFirstVisit(leaves[0], Graph::Direction::Upstream)
                  .to<std::set>();

        for(auto id : roots)
        {
            EXPECT_TRUE(reachableFromLeaf.contains(id)) << ShowValue(id) << reachableFromLeaf;
        }

        // There should be 4 LoadTiled nodes with direct sequence edges to the single TensorContraction node.
        // These should be all of the LoadTiled nodes in the graph.
        auto is_tc = kgraph0.control.isElemType<TensorContraction>();
        auto tc_id = only(kgraph0.control.findElements(is_tc));
        ASSERT_TRUE(tc_id.has_value());

        auto is_loadTiled  = kgraph0.control.isElemType<LoadTiled>();
        auto loadTiled_ids = kgraph0.control.findElements(is_loadTiled).to<std::set>();

        auto tc_parent_ids = kgraph0.control.parentNodes(*tc_id).to<std::set>();
        EXPECT_EQ(loadTiled_ids, tc_parent_ids);
        EXPECT_EQ(4, loadTiled_ids.size());
    }

    TEST_F(KernelGraphTest, LowerTensor)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        int macK  = 16;
        int waveK = 8;

        example.setTileSize(128, 256, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, false, false);

        auto kgraph0 = example.getKernelGraph();
        auto params  = example.getCommandParameters();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);
        auto addLDSTransform           = std::make_shared<AddLDS>(params, m_context);
        auto lowerTileTransform        = std::make_shared<LowerTile>(params, m_context);
        auto lowerTensorContractionTransform
            = std::make_shared<LowerTensorContraction>(params, m_context);
        auto unrollLoopsTransform      = std::make_shared<UnrollLoops>(params, m_context);
        auto fuseLoopsTransform        = std::make_shared<FuseLoops>();
        auto removeDuplicatesTransform = std::make_shared<RemoveDuplicates>();

        auto cleanLoopsTransform      = std::make_shared<CleanLoops>();
        auto addComputeIndexTransform = std::make_shared<AddComputeIndex>();

        kgraph0      = kgraph0.transform(updateParametersTransform);
        auto kgraph1 = kgraph0.transform(addLDSTransform);
        kgraph1      = kgraph1.transform(lowerTileTransform);
        kgraph1      = kgraph1.transform(lowerTensorContractionTransform);

        // Verify the number of Multiply nodes in the graph after lowerTile
        auto multiplyNodes = kgraph1.control.getNodes<Multiply>().to<std::vector>();
        EXPECT_EQ(multiplyNodes.size(), macK / waveK);

        // Verify number of loads
        auto loads = kgraph0.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(loads.size(), 3); // A, B, C

        loads = kgraph1.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(loads.size(), 4); // 1 for A, 2 for B (no LDS), 1 for C

        loads = kgraph1.control.getNodes<LoadLDSTile>().to<std::vector>();
        EXPECT_EQ(loads.size(), 2); // 2 for A

        auto forLoops = kgraph1.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(forLoops.size(), 5); // main: X, Y, K; epilogue: X, Y

        auto kgraphUnrolled = kgraph1.transform(unrollLoopsTransform);

        // Verify that loops have been unrolled
        auto unrolledForLoops = kgraphUnrolled.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(unrolledForLoops.size(), 14);

        auto kgraphFused = kgraphUnrolled.transform(fuseLoopsTransform);
        kgraphFused      = kgraphFused.transform(removeDuplicatesTransform);

        // Verify that loops have been fused
        auto fusedForLoops = kgraphFused.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(fusedForLoops.size(), 5);

        auto fusedLoads = kgraphFused.control.getNodes<LoadTiled>().to<std::vector>();
        EXPECT_EQ(fusedLoads.size(), 17);

        // Verify that single iteration loops have been removed.
        auto kgraphClean     = kgraphFused.transform(cleanLoopsTransform);
        auto cleanedForLoops = kgraphClean.control.getNodes<ForLoopOp>().to<std::vector>();
        EXPECT_EQ(cleanedForLoops.size(), 1);

        // Verify that there is only a single StoreLDSTile node per K loop
        auto unrolled_kgraph_lds = kgraphUnrolled.transform(addLDSTransform);
        auto unrolledStoreLDS
            = unrolled_kgraph_lds.control.getNodes<StoreLDSTile>().to<std::vector>();
        auto kloops = kgraphUnrolled.control.getNodes<ForLoopOp>()
                          .filter([&](int node) {
                              auto loop = kgraphUnrolled.control.getNode<ForLoopOp>(node);
                              return loop.loopName == KLOOP;
                          })
                          .to<std::vector>();
        EXPECT_EQ(unrolledStoreLDS.size(), kloops.size());

        // Verify number of ComputeIndexes: A loads; A LDS loads; B loads; C load; D
        // store: 3 + (2+2) + 3 + 3 + 3 = 12
        kgraph1             = kgraph1.transform(addComputeIndexTransform);
        auto computeIndexes = kgraph1.control.getNodes<ComputeIndex>().to<std::vector>();
        EXPECT_EQ(computeIndexes.size(), 16);

        // Verify number of Deallocates
        auto addDeallocate  = std::make_shared<AddDeallocate>();
        auto kgraph2        = kgraph1.transform(addDeallocate);
        auto addDeallocates = kgraph2.control.getNodes<Deallocate>().to<std::vector>();
        EXPECT_EQ(addDeallocates.size(), 16);

        auto storeLDS = kgraphUnrolled.control.getNodes<StoreLDSTile>().to<std::vector>();
        EXPECT_EQ(storeLDS.size(), 8);

        auto fusedStoreLDS = kgraphFused.control.getNodes<StoreLDSTile>().to<std::vector>();
        EXPECT_EQ(fusedStoreLDS.size(), 1);

        // Verify number of ComputeIndexes after unroll/fuse/lds
        unrolled_kgraph_lds = unrolled_kgraph_lds.transform(addComputeIndexTransform);
        computeIndexes = unrolled_kgraph_lds.control.getNodes<ComputeIndex>().to<std::vector>();
        EXPECT_EQ(computeIndexes.size(), 112);

        // Verify number of Deallocates after unroll/fuse/lds
        unrolled_kgraph_lds = unrolled_kgraph_lds.transform(addDeallocate);
        addDeallocates      = unrolled_kgraph_lds.control.getNodes<Deallocate>().to<std::vector>();
        EXPECT_EQ(addDeallocates.size(), 86);
    }

    TEST_F(KernelGraphTest, InlineIncrement)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        example.setTileSize(128, 256, 8);
        example.setMFMA(32, 32, 2, 1);
        example.setUseLDS(true, true, true);

        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);
        auto addLDSTransform           = std::make_shared<AddLDS>(params, m_context);
        auto lowerLinearTransform      = std::make_shared<LowerLinear>(m_context);
        auto lowerTileTransform        = std::make_shared<LowerTile>(params, m_context);
        auto lowerTensorContractionTransform
            = std::make_shared<LowerTensorContraction>(params, m_context);
        auto unrollLoopsTransform        = std::make_shared<UnrollLoops>(params, m_context);
        auto cleanLoopsTransform         = std::make_shared<CleanLoops>();
        auto addComputeIndexTransform    = std::make_shared<AddComputeIndex>();
        auto inlineInrecrementsTransform = std::make_shared<InlineIncrements>();

        kgraph = kgraph.transform(updateParametersTransform);
        kgraph = kgraph.transform(addLDSTransform);
        kgraph = kgraph.transform(lowerLinearTransform);
        kgraph = kgraph.transform(lowerTileTransform);
        kgraph = kgraph.transform(lowerTensorContractionTransform);

        // Usual lowering, should be able to inline everything.
        auto kgraph1 = kgraph.transform(unrollLoopsTransform);
        kgraph1      = kgraph1.transform(cleanLoopsTransform);
        kgraph1      = kgraph1.transform(addComputeIndexTransform);

        auto pre1  = kgraph1.control.getEdges<ForLoopIncrement>().to<std::vector>();
        kgraph1    = kgraph1.transform(inlineInrecrementsTransform);
        auto post1 = kgraph1.control.getEdges<ForLoopIncrement>().to<std::vector>();

        EXPECT_TRUE(pre1.size() > 0);
        EXPECT_TRUE(post1.empty());
    }

    TEST_F(KernelGraphTest, TileAdd)
    {
        auto example = rocRollerTest::Graphs::TileDoubleAdd<int>();

        example.setTileSize(16, 8);
        example.setSubTileSize(4, 2);

        auto params  = example.getCommandParameters(512, 512);
        auto kgraph0 = example.getKernelGraph();

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);

        kgraph0 = kgraph0.transform(updateParametersTransform);

        std::string expected0 = R".(
            digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
        "coord3"[label="SubDimension{1, CommandArgument(Tensor_0_size_1)I64}(3)"];
        "coord4"[label="MacroTile{NA}(2/LDS/None){16,8}-(4,2)(4)"];
        "coord5"[label="Split(5)",shape=box];
        "coord6"[label="ConstructMacroTile(6)",shape=box];
        "coord7"[label="DataFlow(7)",shape=box];
        "coord8"[label="User{CommandArgument(Tensor_2_extent)I64}(8)"];
        "coord9"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="SubDimension{1, CommandArgument(Tensor_2_size_1)I64}(10)"];
        "coord11"[label="MacroTile{NA}(2/VGPR/None){16,8}-(4,2)(11)"];
        "coord12"[label="Split(12)",shape=box];
        "coord13"[label="ConstructMacroTile(13)",shape=box];
        "coord14"[label="DataFlow(14)",shape=box];
        "coord15"[label="MacroTile{NA}(2/VGPR/None){16,8}-(4,2)(15)"];
        "coord16"[label="DataFlow(16)",shape=box];
        "coord17"[label="MacroTile{NA}(2/VGPR/None){16,8}-(4,2)(17)"];
        "coord18"[label="DataFlow(18)",shape=box];
        "coord19"[label="MacroTile{NA}(2/VGPR/None){16,8}-(4,2)(19)"];
        "coord20"[label="DataFlow(20)",shape=box];
        "coord21"[label="SubDimension{0, NA}(21)"];
        "coord22"[label="SubDimension{1, NA}(22)"];
        "coord23"[label="User{CommandArgument(Tensor_8_extent)I64}(23)"];
        "coord24"[label="DestructMacroTile(24)",shape=box];
        "coord25"[label="Join(25)",shape=box];
        "coord26"[label="DataFlow(26)",shape=box];
        "coord1" -> "coord5"
        "coord1" -> "coord7"
        "coord2" -> "coord6"
        "coord3" -> "coord6"
        "coord4" -> "coord16"
        "coord5" -> "coord2"
        "coord5" -> "coord3"
        "coord6" -> "coord4"
        "coord7" -> "coord4"
        "coord8" -> "coord12"
        "coord8" -> "coord14"
        "coord9" -> "coord13"
        "coord10" -> "coord13"
        "coord11" -> "coord18"
        "coord12" -> "coord9"
        "coord12" -> "coord10"
        "coord13" -> "coord11"
        "coord14" -> "coord11"
        "coord15" -> "coord20"
        "coord16" -> "coord15"
        "coord17" -> "coord20"
        "coord18" -> "coord17"
        "coord19" -> "coord24"
        "coord19" -> "coord26"
        "coord20" -> "coord19"
        "coord21" -> "coord25"
        "coord22" -> "coord25"
        "coord24" -> "coord21"
        "coord24" -> "coord22"
        "coord25" -> "coord23"
        "coord26" -> "coord23"
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord15"->"coord17"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord21"->"coord22"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord21"->"coord22"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadTiled Value: Int32(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadTiled Value: Int32(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="Assign VGPR Add(DataFlowTag(4)NA, DataFlowTag(4)NA)NA(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="Assign VGPR Add(DataFlowTag(11)NA, DataFlowTag(11)NA)NA(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl11"[label="Sequence(11)",shape=box];
        "cntrl12"[label="Assign VGPR Add(DataFlowTag(15)NA, DataFlowTag(17)NA)NA(12)"];
        "cntrl13"[label="Sequence(13)",shape=box];
        "cntrl14"[label="Sequence(14)",shape=box];
        "cntrl15"[label="StoreTiled Value: Int32(15)"];
        "cntrl16"[label="Sequence(16)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl2" -> "cntrl8"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl10"
        "cntrl4" -> "cntrl11"
        "cntrl5" -> "cntrl4"
        "cntrl6" -> "cntrl13"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl6"
        "cntrl9" -> "cntrl14"
        "cntrl10" -> "cntrl9"
        "cntrl11" -> "cntrl9"
        "cntrl12" -> "cntrl16"
        "cntrl13" -> "cntrl12"
        "cntrl14" -> "cntrl12"
        "cntrl16" -> "cntrl15"
        }
        "coord1" -> "to_cntrl_1_2"
        "to_cntrl_1_2"[label="2->1: User: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_1"
        "to_coord_2_1"[label="2->1: User: (0)", shape=cds]
        "coord4" -> "to_cntrl_4_2"
        "to_cntrl_4_2"[label="2->4: MacroTile: (0)", shape=cds]
        "cntrl2" -> "to_coord_2_4"
        "to_coord_2_4"[label="2->4: MacroTile: (0)", shape=cds]
        "coord8" -> "to_cntrl_8_4"
        "to_cntrl_8_4"[label="4->8: User: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_8"
        "to_coord_4_8"[label="4->8: User: (0)", shape=cds]
        "coord11" -> "to_cntrl_11_4"
        "to_cntrl_11_4"[label="4->11: MacroTile: (0)", shape=cds]
        "cntrl4" -> "to_coord_4_11"
        "to_coord_4_11"[label="4->11: MacroTile: (0)", shape=cds]
        "coord15" -> "to_cntrl_15_6"
        "to_cntrl_15_6"[label="6->15: DEST", shape=cds]
        "cntrl6" -> "to_coord_6_15"
        "to_coord_6_15"[label="6->15: DEST", shape=cds]
        "coord17" -> "to_cntrl_17_9"
        "to_cntrl_17_9"[label="9->17: DEST", shape=cds]
        "cntrl9" -> "to_coord_9_17"
        "to_coord_9_17"[label="9->17: DEST", shape=cds]
        "coord19" -> "to_cntrl_19_12"
        "to_cntrl_19_12"[label="12->19: DEST", shape=cds]
        "cntrl12" -> "to_coord_12_19"
        "to_coord_12_19"[label="12->19: DEST", shape=cds]
        "coord19" -> "to_cntrl_19_15"
        "to_cntrl_19_15"[label="15->19: MacroTile: (0)", shape=cds]
        "cntrl15" -> "to_coord_15_19"
        "to_coord_15_19"[label="15->19: MacroTile: (0)", shape=cds]
        "coord23" -> "to_cntrl_23_15"
        "to_cntrl_23_15"[label="15->23: User: (0)", shape=cds]
        "cntrl15" -> "to_coord_15_23"
        "to_coord_15_23"[label="15->23: User: (0)", shape=cds]
        }).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT(true)));

        auto addLDSTransform          = std::make_shared<AddLDS>(params, m_context);
        auto lowerTileTransform       = std::make_shared<LowerTile>(params, m_context);
        auto addComputeIndexTransform = std::make_shared<AddComputeIndex>();

        auto kgraph1 = kgraph0.transform(addLDSTransform);
        kgraph1      = kgraph1.transform(lowerTileTransform);
        kgraph1      = kgraph1.transform(addComputeIndexTransform);

        namespace CG = rocRoller::KernelGraph::ControlGraph;
        ASSERT_EQ(kgraph1.control.getNodes<CG::LoadTiled>().to<std::vector>().size(), 2);
        ASSERT_EQ(kgraph1.control.getNodes<CG::LoadLDSTile>().to<std::vector>().size(), 1);
        ASSERT_EQ(kgraph1.control.getNodes<CG::StoreLDSTile>().to<std::vector>().size(), 1);
    }

    TEST_F(KernelGraphTest, Translate02)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto command = example.getCommand();

        auto one = Expression::literal(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});
        m_context->kernel()->setWorkitemCount({one, one, one});

        auto lowerLinearTransform = std::make_shared<LowerLinear>(m_context);

        auto kgraph0 = translate(command);
        auto kgraph1 = kgraph0.transform(lowerLinearTransform);

        auto user0   = 1;
        auto block0  = 17;
        auto thread0 = 18;

        // given block id and thread id, compute regular (user) index for first (0) dataflow array
        auto block_id  = Expression::literal(2);
        auto thread_id = Expression::literal(33);

        auto exprs = kgraph1.coordinates.reverse({block_id, thread_id}, {user0}, {block0, thread0});
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(2:I, 64:U32)U32, 33:I)U32}, "
                  "CommandArgument(Tensor_0_stride_0)I64)I64}");

        exprs = kgraph1.coordinates.reverse({block_id, thread_id}, {user0}, {block0, thread0});
        sexpr = Expression::toString(fastArith(exprs[0]));
        EXPECT_EQ(sexpr, "{Split: Multiply(161:U32, Tensor_0_stride_0_0:I64)I64}");
    }

#if 0
    TEST_F(KernelGraphTestGPU, GPU_Translate03)
    {
        TIMER(t_total, "Translate03");
        TIMER(t_gpu, "Translate03::GPU");
        TIMER(t_hip, "Translate03::HIP");

        TIC(t_total);

        auto command = std::make_shared<rocRoller::Command>();

        Operations::T_Load_Linear load_A(DataType::Int32, 1, 0);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));

        Operations::T_Load_Linear load_B(DataType::Int32, 1, 2);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(load_B)));

        Operations::T_Execute execute;
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Add(3, 2, 0)));
        execute.addXOp(std::make_shared<Operations::XOp>(Operations::E_Mul(5, 3, 0)));

        command->addOperation(std::make_shared<Operations::Operation>(std::move(execute)));

        Operations::T_Store_Linear store_C(1, 5);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_C)));

        CommandKernel commandKernel(command, "Translate03");

        auto kgraph2 = commandKernel.getKernelGraph();

        auto kernelNode = kgraph2.control.getRootOperation();

        {
            auto expected = getTag(kernelNode);
            auto outputs
                = kgraph2.control.getOutputs<Body>(getTag(kernelNode));
            EXPECT_EQ(2, outputs.size());

            auto outputs2 = kgraph2.control.getOutputs<Sequence>(
                getTag(kernelNode));
            EXPECT_EQ(0, outputs2.size());

            auto outputs3
                = kgraph2.control.getOutputs(getTag(kernelNode), Body{});

            auto outputTags3 = kgraph2.control.getOutputTags(getTag(kernelNode),
                                                             Body{});

            EXPECT_EQ(outputs3.size(), outputTags3.size());
            for(size_t i = 0; i < outputs3.size(); i++)
            {
                EXPECT_EQ(getTag(outputs3[i]), outputTags3[i]);
            }

            EXPECT_EQ(getTag(outputs[0]), getTag(outputs3[0]));

            auto inputs1
                = kgraph2.control.getInputs<Body>(getTag(outputs.at(0)));
            ASSERT_EQ(1, inputs1.size());

            auto actual1 = getTag(inputs1.at(0));
            EXPECT_EQ(actual1, expected);

            auto inputs2 = kgraph2.control.getInputs(getTag(outputs.at(0)),
                                                     Body{});
            ASSERT_EQ(1, inputs2.size());

            auto inputTags2 = kgraph2.control.getInputTags(getTag(outputs.at(0)),
                                                           Body{});

            EXPECT_EQ(inputs2.size(), inputTags2.size());
            for(size_t i = 0; i < inputs2.size(); i++)
            {
                EXPECT_EQ(getTag(inputs2[i]), inputTags2[i]);
            }

            auto actual2 = getTag(inputs2.at(0));
            EXPECT_EQ(actual1, actual2);

            auto inputs3 = kgraph2.control.getInputs<Sequence>(
                getTag(outputs.at(0)));
            EXPECT_EQ(0, inputs3.size());

            auto inputs4 = kgraph2.control.getInputs<Initialize>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs4.size());

            auto inputs5 = kgraph2.control.getInputs<ForLoopIncrement>(
                getTag(outputs.at(0)));
            ASSERT_EQ(0, inputs5.size());
        }

        {
            std::ostringstream msg;
            msg << kgraph2.control;

            std::ostringstream msg2;
            kgraph2.control.toDOT(msg2, "krn");

            EXPECT_EQ(msg.str(), msg2.str());
        }

        TIC(t_hip);
        size_t nx = 64;

        RandomGenerator random(17629u);
        auto            a = random.vector<int>(nx, -100, 100);
        auto            b = random.vector<int>(nx, -100, 100);

        auto user0 = make_shared_device(a);
        auto user2 = make_shared_device(b);
        auto user4 = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);
        TOC(t_hip);

        KernelArguments runtimeArgs;
        runtimeArgs.append("user0", user0.get());
        runtimeArgs.append("user1", nx);
        runtimeArgs.append("user2", nx);
        runtimeArgs.append("user3", (size_t)1);
        runtimeArgs.append("user4", user2.get());
        runtimeArgs.append("user5", nx);
        runtimeArgs.append("user6", nx);
        runtimeArgs.append("user7", (size_t)1);
        runtimeArgs.append("user8", user4.get());
        runtimeArgs.append("user9", nx);
        runtimeArgs.append("user10", (size_t)1);

        TIC(t_gpu);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        TOC(t_gpu);

        TIC(t_hip);
        ASSERT_THAT(hipMemcpy(r.data(), user4.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));
        TOC(t_hip);

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i] * (a[i] + b[i]);

        double rnorm = relativeNorm(r, x);

        ASSERT_LT(rnorm, 1.e-12);

        TIC(t_hip);
        user0.reset();
        user2.reset();
        user4.reset();
        TOC(t_hip);

        TOC(t_total);

        std::cout << TimerPool::summary() << std::endl;
        std::cout << TimerPool::CSV() << std::endl;
    }
#endif

    void KernelGraphTestGPU::GPU_SAXPBY(bool reload)
    {
        RandomGenerator random(1263u);

        size_t nx = 64;

        auto a = random.vector<int>(nx, -100, 100);
        auto b = random.vector<int>(nx, -100, 100);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device(b);
        auto d_c     = make_shared_device<int>(nx);
        auto d_alpha = make_shared_device<int>();

        int alpha = 22;
        int beta  = 33;

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto example = rocRollerTest::Graphs::VectorAdd<int>(true);
        auto command = example.getCommand();
        auto runtimeArgs
            = example.getRuntimeArguments(nx, d_alpha.get(), beta, d_a.get(), d_b.get(), d_c.get());

        CommandKernel commandKernel(command, testKernelName());
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        // launch again, using saved assembly
        auto assemblyFileName = m_context->assemblyFileName();

        if(reload)
        {
            commandKernel.loadKernelFromAssembly(assemblyFileName, testKernelName());
            commandKernel.launchKernel(runtimeArgs.runtimeArguments());
        }

        std::vector<int> r(nx);

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        double rnorm = relativeNormL2(r, example.referenceSolution(alpha, beta, a, b));

        ASSERT_LT(rnorm, 1.e-12);

        if(reload)
        {
            // load, using bad kernel name
            EXPECT_THROW(commandKernel.loadKernelFromAssembly(assemblyFileName, "SAXPBY_BAD"),
                         FatalError);

            // load, using non-existant file
            EXPECT_THROW(
                commandKernel.loadKernelFromAssembly(assemblyFileName + "_bad", testKernelName()),
                FatalError);

            std::filesystem::remove(assemblyFileName);
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBY)
    {
        GPU_SAXPBY(false);
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBYDebug)
    {
        // Make sure Debug mode doesn't introduce bad pointer
        // references in observers
        auto settings = Settings::getInstance();
        settings->set(Settings::LogLvl, LogLevel::Debug);
        GPU_SAXPBY(false);
        settings->reset();
    }

    TEST_F(KernelGraphTestGPU, GPU_SAXPBYLoadAssembly)
    {
        GPU_SAXPBY(true);
    }

    TEST_F(KernelGraphTestGPU, GPU_LeakyRelu)
    {
        auto command = std::make_shared<rocRoller::Command>();

        constexpr auto dataType = DataType::Float;

        auto xTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        auto xLoadTag   = command->addOperation(rocRoller::Operations::T_Load_Linear(xTensorTag));

        auto alphaScalarTag = command->addOperation(
            rocRoller::Operations::Scalar({dataType, PointerType::PointerGlobal}));
        auto alphaLoadTag
            = command->addOperation(rocRoller::Operations::T_Load_Scalar(alphaScalarTag));

        auto zeroLiteralTag = command->addOperation(rocRoller::Operations::Literal(0.0f));

        auto execute = rocRoller::Operations::T_Execute(command->getNextTag());
        auto condTag
            = execute.addXOp(rocRoller::Operations::E_GreaterThan(xLoadTag, zeroLiteralTag));
        auto productTag = execute.addXOp(rocRoller::Operations::E_Mul(xLoadTag, alphaLoadTag));
        auto reluTag
            = execute.addXOp(rocRoller::Operations::E_Conditional(condTag, xLoadTag, productTag));
        command->addOperation(std::move(execute));

        auto reluTensorTag = command->addOperation(rocRoller::Operations::Tensor(1, dataType));
        command->addOperation(rocRoller::Operations::T_Store_Linear(reluTag, reluTensorTag));

        CommandKernel commandKernel(command, "LeakyRelu");
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        size_t nx    = 64;
        float  alpha = 0.9;

        RandomGenerator random(135679u);
        auto            a = random.vector<float>(nx, -5, 5);

        auto d_a     = make_shared_device(a);
        auto d_b     = make_shared_device<float>(nx);
        auto d_alpha = make_shared_device<float>();

        std::vector<float> r(nx), x(nx);

        ASSERT_THAT(hipMemcpy(d_alpha.get(), &alpha, 1 * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(xTensorTag, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(xTensorTag, ArgumentType::Limit, nx);
        commandArgs.setArgument(xTensorTag, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(xTensorTag, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(alphaScalarTag, ArgumentType::Value, d_alpha.get());
        commandArgs.setArgument(reluTensorTag, ArgumentType::Value, d_b.get());
        commandArgs.setArgument(reluTensorTag, ArgumentType::Limit, nx);
        commandArgs.setArgument(reluTensorTag, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(reluTensorTag, ArgumentType::Stride, 0, (size_t)1);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * sizeof(float), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
        {
            x[i] = a[i] > 0 ? a[i] : alpha * a[i];
        }

        double rnorm = relativeNormL2(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTestGPU, GPU_LinearCopy)
    {
        auto command = std::make_shared<rocRoller::Command>();

        Operations::Tensor tensor_A(1, DataType::Int32);
        auto               tagTensorA
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(tensor_A)));

        Operations::Tensor tensor_B(1, DataType::Int32);
        auto               tagTensorB
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(tensor_B)));

        Operations::T_Load_Linear load_A(tagTensorA);
        auto                      tagLoadStore
            = command->addOperation(std::make_shared<Operations::Operation>(std::move(load_A)));
        Operations::T_Store_Linear store_B(tagLoadStore, tagTensorB);
        command->addOperation(std::make_shared<Operations::Operation>(std::move(store_B)));

        CommandKernel commandKernel(command, "LinearCopy");
        commandKernel.setContext(m_context);
        commandKernel.generateKernel();

        size_t nx = 64;

        RandomGenerator random(135679u);
        auto            a = random.vector<int>(nx, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<int>(nx);

        std::vector<int> r(nx), x(nx);

        CommandArguments commandArgs = command->createArguments();

        commandArgs.setArgument(tagTensorA, ArgumentType::Value, d_a.get());
        commandArgs.setArgument(tagTensorA, ArgumentType::Limit, nx);
        commandArgs.setArgument(tagTensorA, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(tagTensorA, ArgumentType::Stride, 0, (size_t)1);
        commandArgs.setArgument(tagTensorB, ArgumentType::Value, d_b.get());
        commandArgs.setArgument(tagTensorB, ArgumentType::Limit, nx);
        commandArgs.setArgument(tagTensorB, ArgumentType::Size, 0, nx);
        commandArgs.setArgument(tagTensorB, ArgumentType::Stride, 0, (size_t)1);

        commandKernel.launchKernel(commandArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        for(size_t i = 0; i < nx; ++i)
            x[i] = a[i];

        double rnorm = relativeNormL2(r, x);

        ASSERT_LT(rnorm, 1.e-12);
    }

    template <typename T>
    void CopyStrideOverride(CommandKernelPtr& commandKernel, bool override = false)
    {
        auto example = rocRollerTest::Graphs::TileCopy<T>();

        example.setTileSize(16, 4);
        example.setSubTileSize(4, 2);

        if(override)
        {
            example.setLiteralStrides({(size_t)0, (size_t)1});
        }

        size_t nx = 256;
        size_t ny = 128;

        RandomGenerator random(193674u);
        auto            ax = static_cast<T>(-100.);
        auto            ay = static_cast<T>(100.);
        auto            a  = random.vector<T>(nx * ny, ax, ay);

        std::vector<T> r(nx * ny, 0.);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device<T>(nx * ny);

        auto command     = example.getCommand();
        auto params      = example.getCommandParameters(nx, ny);
        auto launch      = example.getCommandLaunchParameters(nx, ny);
        auto runtimeArgs = example.getRuntimeArguments(nx, ny, d_a.get(), d_b.get());

        std::string colName    = (override) ? "ColOverride" : "";
        std::string kernelName = "TensorTileCopy" + colName + TypeInfo<T>::Name();

        commandKernel = std::make_shared<CommandKernel>(command, kernelName);
        commandKernel->setContext(Context::ForDefaultHipDevice(kernelName));
        commandKernel->setCommandParameters(params);
        commandKernel->generateKernel();

        commandKernel->setLaunchParameters(launch);
        commandKernel->launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_b.get(), nx * ny * sizeof(T), hipMemcpyDefault),
                    HasHipSuccess(0));

        auto   ref   = example.referenceSolution(a);
        double rnorm = relativeNormL2(r, ref);

        std::ostringstream msg;
        for(int i = 0; i < r.size(); i++)
        {
            msg << i << " " << r[i] << " | " << ref[i];
            if(r[i] != ref[i])
                msg << " * ";
            msg << std::endl;
        }

        ASSERT_LT(rnorm, 1.e-12) << msg.str();
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopy)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<int>(commandKernel);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideHalf)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<Half>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dword "))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dword "))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideFloat)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<float>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dwordx2"))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dwordx2"))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileCopyColStrideDouble)
    {
        CommandKernelPtr commandKernel;
        CopyStrideOverride<double>(commandKernel, true);

        auto instructions = NormalizedSourceLines(commandKernel->getInstructions(), false);

        int numRead  = 0;
        int numWrite = 0;
        for(auto const& instruction : instructions)
        {
            if(instruction.starts_with("buffer_load_dwordx4"))
            {
                numRead++;
            }
            else if(instruction.starts_with("buffer_store_dwordx4"))
            {
                numWrite++;
            }
        }

        EXPECT_EQ(numRead, 4);
        EXPECT_EQ(numWrite, 4);
    }

    TEST_F(KernelGraphTestGPU, GPU_TensorTileAdd)
    {
        size_t nx = 256; // tensor size x
        size_t ny = 512; // tensor size y

        RandomGenerator random(129674u);

        auto a = random.vector<int>(nx * ny, -100, 100);
        auto b = random.vector<int>(nx * ny, -100, 100);
        auto r = random.vector<int>(nx * ny, -100, 100);

        auto d_a = make_shared_device(a);
        auto d_b = make_shared_device(b);
        auto d_c = make_shared_device<int>(nx * ny);

        auto example = rocRollerTest::Graphs::TileDoubleAdd<int>();
        example.setTileSize(8, 64);
        example.setSubTileSize(2, 8);

        auto command     = example.getCommand();
        auto runtimeArgs = example.getRuntimeArguments(nx, ny, d_a.get(), d_b.get(), d_c.get());
        auto params      = example.getCommandParameters(nx, ny);
        auto launch      = example.getCommandLaunchParameters(nx, ny);

        CommandKernel commandKernel(command, "TensorTileAdd");
        commandKernel.setContext(m_context);
        commandKernel.setCommandParameters(params);
        commandKernel.generateKernel();

        commandKernel.setLaunchParameters(launch);
        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(hipMemcpy(r.data(), d_c.get(), nx * ny * sizeof(int), hipMemcpyDefault),
                    HasHipSuccess(0));

        // reference solution
        double rnorm = relativeNormL2(r, example.referenceSolution(a, b));

        ASSERT_LT(rnorm, 1.e-12);
    }

    TEST_F(KernelGraphTest, CleanExpression)
    {
        VariableType doubleVal{DataType::Double, PointerType::Value};
        auto         command = std::make_shared<Command>();

        auto aTag = command->allocateTag();
        auto a    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, aTag, ArgumentType::Value));
        auto bTag = command->allocateTag();
        auto b    = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::Int32, PointerType::Value}, bTag, ArgumentType::Value));

        m_context->kernel()->addCommandArguments(command->getArguments());

        auto expr1 = a + b;
        auto expr2 = b * expr1;

        auto clean_expr = cleanArguments(expr2, m_context->kernel());

        EXPECT_EQ(
            Expression::toString(clean_expr),
            "Multiply(user_Int32_Value_1:I, Add(user_Int32_Value_0:I, user_Int32_Value_1:I)I)I");
    }

    TEST_F(KernelGraphTest, CleanArguments)
    {
        auto example = rocRollerTest::Graphs::VectorAddNegSquare<int>();
        auto command = example.getCommand();

        int workGroupSize = 64;
        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto cleanArgumentsTransform = std::make_shared<CleanArguments>(m_context, command);

        auto kgraph = translate(command);

        auto beforePredicates = {HasSubstr("SubDimension{0, CommandArgument(Tensor_0_size_0)I64}"),
                                 HasSubstr("SubDimension{0, CommandArgument(Tensor_2_size_0)I64}"),
                                 HasSubstr("Linear{CommandArgument(Tensor_0_size_0)I64}"),
                                 HasSubstr("Linear{CommandArgument(Tensor_2_size_0)I64}")};

        // Note that these searches do not include the close braces ("}").  This is because the
        // argument name will have a number appended which is subject to change
        // (Load_Linear_0_size_0 might become Load_Linear_0_size_0_2).
        auto afterPredicates = {
            HasSubstr("SubDimension{0, Tensor_0_size_0"),
            HasSubstr("SubDimension{0, Tensor_2_size_0"),
            HasSubstr("Linear{Tensor_0_size_0"),
            HasSubstr("Linear{Tensor_2_size_0"),
        };

        {
            auto dot = kgraph.toDOT();

            for(auto const& pred : beforePredicates)
                EXPECT_THAT(dot, pred);

            for(auto const& pred : afterPredicates)
                EXPECT_THAT(dot, Not(pred));
        }

        kgraph = kgraph.transform(cleanArgumentsTransform);

        {
            auto dot = kgraph.toDOT();

            for(auto const& pred : beforePredicates)
                EXPECT_THAT(dot, Not(pred));

            for(auto const& pred : afterPredicates)
                EXPECT_THAT(dot, pred);
        }
    }

    TEST_F(KernelGraphTest, Basic)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        // Control Graph
        int kernel_index = kgraph.control.addElement(Kernel());
        int loadA_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int loadB_index  = kgraph.control.addElement(LoadLinear(DataType::Float));
        int body1_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadA_index});
        int body2_index  = kgraph.control.addElement(Body(), {kernel_index}, {loadB_index});

        int op1_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(5)});
        int sequence1_index = kgraph.control.addElement(Sequence(), {loadA_index}, {op1_index});
        int sequence2_index = kgraph.control.addElement(Sequence(), {loadB_index}, {op1_index});

        int op2_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(7)});
        int sequence3_index = kgraph.control.addElement(Sequence(), {op1_index}, {op2_index});

        int op3_index
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(9)});
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
        int buffer1_index = kgraph.coordinates.addElement(
            rocRoller::KernelGraph::CoordinateGraph::Buffer(), {u1_index}, {linear1_index});

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
        int u5o_index    = kgraph.coordinates.addElement(User({}, ""));
        int join1_index  = kgraph.coordinates.addElement(Join(), {sd5o_index}, {u5o_index});
        int dataflow6_index
            = kgraph.coordinates.addElement(DataFlow(), {linear5i_index}, {u5o_index});

        auto yamlData = toYAML(kgraph);
        auto graph2   = rocRoller::KernelGraph::fromYAML(yamlData);
        auto yaml2    = toYAML(graph2);
        EXPECT_EQ(yamlData, yaml2);

        std::string expected = R".(
        digraph {
        "coord1"[label="User{NA}(1)"];
        "coord2"[label="SubDimension{0, NA}(2)"];
        "coord3"[label="Split(3)",shape=box];
        "coord4"[label="Linear{NA}(4)"];
        "coord5"[label="Flatten(5)",shape=box];
        "coord6"[label="DataFlow(6)",shape=box];
        "coord7"[label="Buffer(7)",shape=box];
        "coord8"[label="User{NA}(8)"];
        "coord9"[label="SubDimension{0, NA}(9)"];
        "coord10"[label="Split(10)",shape=box];
        "coord11"[label="Linear{NA}(11)"];
        "coord12"[label="Flatten(12)",shape=box];
        "coord13"[label="DataFlow(13)",shape=box];
        "coord14"[label="Linear{NA}(14)"];
        "coord15"[label="DataFlow(15)",shape=box];
        "coord16"[label="Linear{NA}(16)"];
        "coord17"[label="DataFlow(17)",shape=box];
        "coord18"[label="Linear{NA}(18)"];
        "coord19"[label="DataFlow(19)",shape=box];
        "coord20"[label="Linear{NA}(20)"];
        "coord21"[label="MakeOutput(21)",shape=box];
        "coord22"[label="SubDimension{0, NA}(22)"];
        "coord23"[label="Split(23)",shape=box];
        "coord24"[label="User{NA}(24)"];
        "coord25"[label="Join(25)",shape=box];
        "coord26"[label="DataFlow(26)",shape=box];
        "coord1" -> "coord3"
        "coord1" -> "coord6"
        "coord1" -> "coord7"
        "coord2" -> "coord5"
        "coord3" -> "coord2"
        "coord4" -> "coord15"
        "coord5" -> "coord4"
        "coord6" -> "coord4"
        "coord7" -> "coord4"
        "coord8" -> "coord10"
        "coord8" -> "coord13"
        "coord9" -> "coord12"
        "coord10" -> "coord9"
        "coord11" -> "coord15"
        "coord12" -> "coord11"
        "coord13" -> "coord11"
        "coord14" -> "coord17"
        "coord14" -> "coord19"
        "coord15" -> "coord14"
        "coord16" -> "coord19"
        "coord17" -> "coord16"
        "coord18" -> "coord21"
        "coord18" -> "coord26"
        "coord19" -> "coord18"
        "coord20" -> "coord23"
        "coord21" -> "coord20"
        "coord22" -> "coord25"
        "coord23" -> "coord22"
        "coord25" -> "coord24"
        "coord26" -> "coord24"
        {
            rank=same
            "coord4"->"coord11"[style=invis]
            rankdir=LR
        }
        {
            rank=same
            "coord14"->"coord16"[style=invis]
            rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadLinear Value: Float(2)"];
        "cntrl3"[label="LoadLinear Value: Float(3)"];
        "cntrl4"[label="Body(4)",shape=box];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="Assign VGPR 5:I(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="Assign VGPR 7:I(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl11"[label="Assign VGPR 9:I(11)"];
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

    TEST_F(KernelGraphTest, UpdateParamsTMul)
    {
        auto command = std::make_shared<Command>();

        auto tagTensorA
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // A
        auto tagLoadA = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorA));

        auto tagTensorB
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // B
        auto tagLoadB = command->addOperation(rocRoller::Operations::T_Load_Tiled(tagTensorB));

        auto tagStoreD
            = command->addOperation(rocRoller::Operations::T_Mul(tagLoadA, tagLoadB)); // D = A * B

        auto tagTensorD
            = command->addOperation(rocRoller::Operations::Tensor(2, DataType::Float)); // D
        command->addOperation(rocRoller::Operations::T_Store_Tiled(tagStoreD, tagTensorD));

        auto kgraph0 = translate(command);

        std::string expected0 = R".(
        digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
        "coord3"[label="SubDimension{1, CommandArgument(Tensor_0_size_1)I64}(3)"];
        "coord4"[label="MacroTile{NA}(2/None/None){}-()(4)"];
        "coord5"[label="Split(5)",shape=box];
        "coord6"[label="ConstructMacroTile(6)",shape=box];
        "coord7"[label="DataFlow(7)",shape=box];
        "coord8"[label="User{CommandArgument(Tensor_2_extent)I64}(8)"];
        "coord9"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="SubDimension{1, CommandArgument(Tensor_2_size_1)I64}(10)"];
        "coord11"[label="MacroTile{NA}(2/None/None){}-()(11)"];
        "coord12"[label="Split(12)",shape=box];
        "coord13"[label="ConstructMacroTile(13)",shape=box];
        "coord14"[label="DataFlow(14)",shape=box];
        "coord15"[label="MacroTile{NA}(0/None/None){}-()(15)"];
        "coord16"[label="DataFlow(16)",shape=box];
        "coord17"[label="SubDimension{0, NA}(17)"];
        "coord18"[label="SubDimension{1, NA}(18)"];
        "coord19"[label="User{CommandArgument(Tensor_5_extent)I64}(19)"];
        "coord20"[label="DestructMacroTile(20)",shape=box];
        "coord21"[label="Join(21)",shape=box];
        "coord22"[label="DataFlow(22)",shape=box];
        "coord1" -> "coord5"
        "coord1" -> "coord7"
        "coord2" -> "coord6"
        "coord3" -> "coord6"
        "coord4" -> "coord16"
        "coord5" -> "coord2"
        "coord5" -> "coord3"
        "coord6" -> "coord4"
        "coord7" -> "coord4"
        "coord8" -> "coord12"
        "coord8" -> "coord14"
        "coord9" -> "coord13"
        "coord10" -> "coord13"
        "coord11" -> "coord16"
        "coord12" -> "coord9"
        "coord12" -> "coord10"
        "coord13" -> "coord11"
        "coord14" -> "coord11"
        "coord15" -> "coord20"
        "coord15" -> "coord22"
        "coord16" -> "coord15"
        "coord17" -> "coord21"
        "coord18" -> "coord21"
        "coord20" -> "coord17"
        "coord20" -> "coord18"
        "coord21" -> "coord19"
        "coord22" -> "coord19"
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord4"->"coord11"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadTiled Value: Float(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadTiled Value: Float(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="TensorContraction(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="StoreTiled Value: Float(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl8"
        "cntrl5" -> "cntrl4"
        "cntrl6" -> "cntrl10"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl6"
        "cntrl10" -> "cntrl9"
        }
        }
    ).";

        EXPECT_EQ(NormalizedSource(expected0), NormalizedSource(kgraph0.toDOT()));

        // macro tile sizes
        int mac_m = 64;
        int mac_n = 64;
        int mac_k = 64;

        auto macTileA = MacroTile({mac_m, mac_k}, MemoryType::VGPR); // A
        auto macTileB = MacroTile({mac_k, mac_n}, MemoryType::VGPR); // B

        auto params = std::make_shared<CommandParameters>();

        params->setDimensionInfo(tagLoadA, macTileA);
        params->setDimensionInfo(tagLoadB, macTileB);

        auto updateParametersTransform = std::make_shared<UpdateParameters>(params);

        kgraph0 = kgraph0.transform(updateParametersTransform);

        std::string expected1 = R".(
        digraph {
        "coord1"[label="User{CommandArgument(Tensor_0_extent)I64}(1)"];
        "coord2"[label="SubDimension{0, CommandArgument(Tensor_0_size_0)I64}(2)"];
        "coord3"[label="SubDimension{1, CommandArgument(Tensor_0_size_1)I64}(3)"];
        "coord4"[label="MacroTile{NA}(2/VGPR/None){64,64}-()(4)"];
        "coord5"[label="Split(5)",shape=box];
        "coord6"[label="ConstructMacroTile(6)",shape=box];
        "coord7"[label="DataFlow(7)",shape=box];
        "coord8"[label="User{CommandArgument(Tensor_2_extent)I64}(8)"];
        "coord9"[label="SubDimension{0, CommandArgument(Tensor_2_size_0)I64}(9)"];
        "coord10"[label="SubDimension{1, CommandArgument(Tensor_2_size_1)I64}(10)"];
        "coord11"[label="MacroTile{NA}(2/VGPR/None){64,64}-()(11)"];
        "coord12"[label="Split(12)",shape=box];
        "coord13"[label="ConstructMacroTile(13)",shape=box];
        "coord14"[label="DataFlow(14)",shape=box];
        "coord15"[label="MacroTile{NA}(2/WAVE/MATRIX_ACCUMULATOR){64,64}-()(15)"];
        "coord16"[label="DataFlow(16)",shape=box];
        "coord17"[label="SubDimension{0, NA}(17)"];
        "coord18"[label="SubDimension{1, NA}(18)"];
        "coord19"[label="User{CommandArgument(Tensor_5_extent)I64}(19)"];
        "coord20"[label="DestructMacroTile(20)",shape=box];
        "coord21"[label="Join(21)",shape=box];
        "coord22"[label="DataFlow(22)",shape=box];
        "coord1" -> "coord5"
        "coord1" -> "coord7"
        "coord2" -> "coord6"
        "coord3" -> "coord6"
        "coord4" -> "coord16"
        "coord5" -> "coord2"
        "coord5" -> "coord3"
        "coord6" -> "coord4"
        "coord7" -> "coord4"
        "coord8" -> "coord12"
        "coord8" -> "coord14"
        "coord9" -> "coord13"
        "coord10" -> "coord13"
        "coord11" -> "coord16"
        "coord12" -> "coord9"
        "coord12" -> "coord10"
        "coord13" -> "coord11"
        "coord14" -> "coord11"
        "coord15" -> "coord20"
        "coord15" -> "coord22"
        "coord16" -> "coord15"
        "coord17" -> "coord21"
        "coord18" -> "coord21"
        "coord20" -> "coord17"
        "coord20" -> "coord18"
        "coord21" -> "coord19"
        "coord22" -> "coord19"
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord2"->"coord3"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord9"->"coord10"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord4"->"coord11"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        {
        rank=same
        "coord17"->"coord18"[style=invis]
        rankdir=LR
        }
        subgraph clusterCF {label = "Control Graph";
        "cntrl1"[label="Kernel(1)"];
        "cntrl2"[label="LoadTiled Value: Float(2)"];
        "cntrl3"[label="Body(3)",shape=box];
        "cntrl4"[label="LoadTiled Value: Float(4)"];
        "cntrl5"[label="Body(5)",shape=box];
        "cntrl6"[label="TensorContraction(6)"];
        "cntrl7"[label="Sequence(7)",shape=box];
        "cntrl8"[label="Sequence(8)",shape=box];
        "cntrl9"[label="StoreTiled Value: Float(9)"];
        "cntrl10"[label="Sequence(10)",shape=box];
        "cntrl1" -> "cntrl3"
        "cntrl1" -> "cntrl5"
        "cntrl2" -> "cntrl7"
        "cntrl3" -> "cntrl2"
        "cntrl4" -> "cntrl8"
        "cntrl5" -> "cntrl4"
        "cntrl6" -> "cntrl10"
        "cntrl7" -> "cntrl6"
        "cntrl8" -> "cntrl6"
        "cntrl10" -> "cntrl9"
        }
        }
    ).";

        EXPECT_EQ(NormalizedSource(expected1), NormalizedSource(kgraph0.toDOT()));
    }

    TEST_F(KernelGraphTestGPU, GPU_Conditional)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        m_context->kernel()->setKernelDimensions(1);
        m_context->kernel()->setWorkgroupSize({64, 1, 1});

        auto kernel = kgraph.control.addElement(Kernel());
        auto unit   = Expression::literal(1);
        auto zero   = Expression::literal(0);
        auto test   = std::make_shared<Register::Value>(
            m_context, Register::Type::Scalar, DataType::Int32, 1);
        test->allocateNow();
        auto conditionalAssign = kgraph.control.addElement(
            Assign{Register::Type::Vector, test->expression() = Expression::literal(0)});
        kgraph.control.addElement(Body(), {kernel}, {conditionalAssign});

        auto conditional = kgraph.control.addElement(
            ConditionalOp{test->expression() < unit, "Test Conditional"});

        kgraph.control.addElement(Sequence(), {conditionalAssign}, {conditional});

        auto trueOp    = kgraph.control.addElement(Assign{Register::Type::Vector, unit});
        auto trueBody  = kgraph.control.addElement(Body(), {conditional}, {trueOp});
        auto falseOp   = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        auto falseBody = kgraph.control.addElement(Else(), {conditional}, {falseOp});

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        EXPECT_THAT(output(), testing::HasSubstr("s_cmp_lt_i32 s0, 1")); //Conditional Test
        EXPECT_THAT(output(), testing::HasSubstr("s_cbranch_scc0")); //Branch for False
        EXPECT_THAT(output(), testing::HasSubstr("s_branch")); //Branch after True
        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v0, 1")); //True Body
        EXPECT_THAT(output(), testing::HasSubstr("v_mov_b32 v0, 0")); //False Body
    }

    TEST_F(KernelGraphTestGPU, GPU_ConditionalExecute)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        std::vector<int> testValues = {22, 66};

        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto two   = Expression::literal(2u);
        auto three = Expression::literal(3u);

        auto k = m_context->kernel();

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(1);
        k->setWorkitemCount({three, one, one});
        k->setWorkgroupSize({1, 1, 1});
        k->setDynamicSharedMemBytes(zero);
        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        // global result
        auto user = kgraph.coordinates.addElement(User({}, "result"));
        auto wg   = kgraph.coordinates.addElement(Workgroup());
        kgraph.coordinates.addElement(PassThrough(), {wg}, {user});

        // result
        auto dstVGPR = kgraph.coordinates.addElement(VGPR());

        // set result to testValues[0]
        auto assignTrueBranch1 = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[0])});
        kgraph.mapper.connect(assignTrueBranch1, dstVGPR, NaryArgument::DEST);
        auto assignTrueBranch2 = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[0])});
        kgraph.mapper.connect(assignTrueBranch2, dstVGPR, NaryArgument::DEST);

        // set result to testValues[1]
        auto assignFalseBranch = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(testValues[1])});
        kgraph.mapper.connect(assignFalseBranch, dstVGPR, NaryArgument::DEST);

        auto workgroupExpr = k->workgroupIndex().at(0)->expression();
        auto firstConditional
            = kgraph.control.addElement(ConditionalOp{workgroupExpr < one, "First Conditional"});
        auto secondConditional = kgraph.control.addElement(
            ConditionalOp{(workgroupExpr > one) && (workgroupExpr <= two), "Second Conditional"});

        auto storeIndex = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeIndex, user);
        kgraph.mapper.connect<VGPR>(storeIndex, dstVGPR);

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {firstConditional});
        kgraph.control.addElement(Body(), {firstConditional}, {assignTrueBranch1});
        kgraph.control.addElement(Else(), {firstConditional}, {secondConditional});
        kgraph.control.addElement(Body(), {secondConditional}, {assignTrueBranch2});
        kgraph.control.addElement(Else(), {secondConditional}, {assignFalseBranch});
        kgraph.control.addElement(Sequence(), {firstConditional}, {storeIndex});

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(isLocalDevice())
        {
            auto d_result = make_shared_device<int>(3);

            KernelArguments kargs;
            kargs.append("result", d_result.get());

            KernelInvocation kinv;
            kinv.workitemCount = {3, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<int> result(3);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), result.size() * sizeof(int), hipMemcpyDefault),
                HasHipSuccess(0));
            EXPECT_EQ(result[0], testValues[0]);
            EXPECT_EQ(result[1], testValues[1]);
            EXPECT_EQ(result[2], testValues[0]);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_F(KernelGraphTestGPU, GPU_DoWhileExecute)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto three = Expression::literal(3u);

        auto k = m_context->kernel();

        k->addArgument(
            {"result", {DataType::Int32, PointerType::PointerGlobal}, DataDirection::WriteOnly});

        k->setKernelDimensions(1);
        k->setWorkitemCount({three, one, one});
        k->setWorkgroupSize({1, 1, 1});
        k->setDynamicSharedMemBytes(zero);
        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        // global result
        auto user = kgraph.coordinates.addElement(User({}, "result"));
        auto wg   = kgraph.coordinates.addElement(Workgroup());
        kgraph.coordinates.addElement(PassThrough(), {wg}, {user});

        // result
        auto dstVGPR = kgraph.coordinates.addElement(VGPR());

        auto dfa = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{dstVGPR, Register::Type::Vector, DataType::UInt32});
        auto assignVGPR = kgraph.control.addElement(Assign{Register::Type::Vector, zero});
        kgraph.mapper.connect(assignVGPR, dstVGPR, NaryArgument::DEST);
        auto assignBody = kgraph.control.addElement(Assign{Register::Type::Vector, dfa + one});
        kgraph.mapper.connect(assignBody, dstVGPR, NaryArgument::DEST);
        auto workgroupExpr = k->workgroupIndex().at(0)->expression();

        auto condVGPR = kgraph.coordinates.addElement(VGPR());
        auto condDFT  = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{condVGPR, Register::Type::Vector, DataType::UInt32});

        auto assignCond = kgraph.control.addElement(Assign{Register::Type::Vector, workgroupExpr});
        kgraph.mapper.connect(assignCond, condVGPR, NaryArgument::DEST);

        auto doWhile = kgraph.control.addElement(DoWhileOp{dfa < condDFT, "Test DoWhile"});

        auto storeIndex = kgraph.control.addElement(StoreVGPR());
        kgraph.mapper.connect<User>(storeIndex, user);
        kgraph.mapper.connect<VGPR>(storeIndex, dstVGPR);

        auto kernel = kgraph.control.addElement(Kernel());
        kgraph.control.addElement(Body(), {kernel}, {assignVGPR});
        kgraph.control.addElement(Body(), {kernel}, {assignCond});
        kgraph.control.addElement(Sequence(), {assignCond}, {doWhile});
        kgraph.control.addElement(Sequence(), {assignVGPR}, {doWhile});
        kgraph.control.addElement(Body(), {doWhile}, {assignBody});
        kgraph.control.addElement(Sequence(), {doWhile}, {storeIndex});

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());

        if(isLocalDevice())
        {
            auto d_result = make_shared_device<int>(3);

            KernelArguments kargs;
            kargs.append("result", d_result.get());

            KernelInvocation kinv;
            kinv.workitemCount = {3, 1, 1};
            kinv.workgroupSize = {1, 1, 1};

            auto executableKernel = m_context->instructions()->getExecutableKernel();
            executableKernel->executeKernel(kargs, kinv);

            std::vector<int> result(3);
            ASSERT_THAT(
                hipMemcpy(
                    result.data(), d_result.get(), result.size() * sizeof(int), hipMemcpyDefault),
                HasHipSuccess(0));
            EXPECT_EQ(result[0], 1);
            EXPECT_EQ(result[1], 1);
            EXPECT_EQ(result[2], 2);
        }
        else
        {
            std::vector<char> assembledKernel = m_context->instructions()->assemble();
            EXPECT_GT(assembledKernel.size(), 0);
        }
    }

    TEST_F(KernelGraphTest, LDSNoDeallocateInHotLoop)
    {
        using GD = Graph::Direction;

        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        example.setTileSize(128, 256, 8);
        example.setMFMA(32, 32, 2, 1);
        example.setUseLDS(true, true, true);

        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        params->unrollK           = 3;
        params->prefetch          = true;
        params->prefetchInFlight  = 3;
        params->prefetchLDSFactor = 3;
        params->prefetchMixMemOps = true;

        std::vector<GraphTransformPtr> transforms;
        transforms.push_back(std::make_shared<UpdateParameters>(params));
        transforms.push_back(std::make_shared<AddLDS>(params, m_context));
        transforms.push_back(std::make_shared<LowerLinear>(m_context));
        transforms.push_back(std::make_shared<LowerTile>(params, m_context));
        transforms.push_back(std::make_shared<LowerTensorContraction>(params, m_context));
        transforms.push_back(std::make_shared<ConnectWorkgroups>());
        transforms.push_back(std::make_shared<AddPrefetch>(params, m_context));
        transforms.push_back(std::make_shared<AddComputeIndex>());
        transforms.push_back(std::make_shared<AddDeallocate>());

        for(auto& t : transforms)
            kgraph = kgraph.transform(t);

        auto ldsDeallocatePredicate = [&](int tag) -> bool {
            auto maybeDeallocate = kgraph.control.get<Deallocate>(tag);
            if(!maybeDeallocate)
                return false;
            auto dimTag   = kgraph.mapper.get<Dimension>(tag);
            auto maybeLDS = kgraph.coordinates.get<LDS>(dimTag);
            return maybeLDS.has_value();
        };

        auto forKLoopPredicate = [&](int tag) -> bool {
            auto maybeForLoop = kgraph.control.get<ForLoopOp>(tag);
            if(!maybeForLoop)
                return false;
            return maybeForLoop->loopName == rocRoller::KLOOP;
        };

        auto kernel  = *only(kgraph.control.roots());
        auto forLoop = *only(kgraph.control.findNodes(kernel, forKLoopPredicate, GD::Downstream));

        auto ldsDeallocateFromKernel
            = kgraph.control.findNodes(kernel, ldsDeallocatePredicate, GD::Downstream)
                  .to<std::vector>();

        std::vector<int> ldsDeallocateInsideLoop;
        for(auto body : kgraph.control.getOutputNodeIndices<Body>(forLoop))
        {
            auto t = kgraph.control.findNodes(body, ldsDeallocatePredicate, GD::Downstream)
                         .to<std::vector>();
            std::copy(t.cbegin(), t.cend(), std::back_inserter(ldsDeallocateInsideLoop));
        }

        EXPECT_FALSE(ldsDeallocateFromKernel.empty());
        EXPECT_TRUE(ldsDeallocateInsideLoop.empty());
    }

    TEST_F(KernelGraphTest, WaitZero)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto kernel = kgraph.control.addElement(Kernel());
        auto wait   = kgraph.control.addElement(WaitZero());
        kgraph.control.addElement(Body(), {kernel}, {wait});

        m_context->schedule(rocRoller::KernelGraph::generate(kgraph, m_context->kernel()));

        EXPECT_THAT(output(), testing::HasSubstr("s_waitcnt"));

        EXPECT_THAT(output(), testing::HasSubstr("vmcnt(0)"));
        EXPECT_THAT(output(), testing::HasSubstr("lgkmcnt(0)"));
        if(m_context->targetArchitecture().HasCapability(GPUCapability::HasExpcnt))
        {
            EXPECT_THAT(output(), testing::HasSubstr("expcnt(0)"));
        }
    }

    TEST_F(KernelGraphTest, ReindexConditionalOpExpression)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto unit = Expression::literal(1);

        auto kernel = kgraph.control.addElement(Kernel());

        auto loadA = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadA});

        auto user0 = kgraph.coordinates.addElement(User({}, "user0"));
        auto vgprA = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprA});
        kgraph.mapper.connect<VGPR>(loadA, vgprA);

        auto exprA = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{vgprA, Register::Type::Scalar, DataType::Int32});
        auto conditional = kgraph.control.addElement(ConditionalOp{exprA > unit, "conditional"});
        kgraph.control.addElement(Sequence(), {loadA}, {conditional});

        auto loadB = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadB});
        auto vgprB = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprB});
        kgraph.mapper.connect<VGPR>(loadB, vgprB);

        kgraph.control.addElement(Sequence(), {loadB}, {conditional});

        GraphReindexer reindexer;
        reindexer.coordinates.emplace(vgprA, vgprB);
        reindexExpressions(kgraph, conditional, reindexer);

        auto condition = kgraph.control.get<ConditionalOp>(conditional)->condition;
        auto lhs       = std::get<Expression::GreaterThan>(*condition).lhs;
        auto tag       = std::get<Expression::DataFlowTag>(*lhs).tag;
        EXPECT_EQ(tag, vgprB);
    }

    TEST_F(KernelGraphTest, ReindexAssertOpExpression)
    {
        rocRoller::KernelGraph::KernelGraph kgraph;

        auto unit = Expression::literal(1);

        auto kernel = kgraph.control.addElement(Kernel());

        auto loadA = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadA});

        auto user0 = kgraph.coordinates.addElement(User(Operations::OperationTag(0), "user0"));
        auto vgprA = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprA});
        kgraph.mapper.connect<VGPR>(loadA, vgprA);

        auto exprA = std::make_shared<Expression::Expression>(
            Expression::DataFlowTag{vgprA, Register::Type::Scalar, DataType::Int32});
        auto assertOp = kgraph.control.addElement(AssertOp{"assert", exprA > unit});
        kgraph.control.addElement(Sequence(), {loadA}, {assertOp});

        auto loadB = kgraph.control.addElement(LoadVGPR(DataType::Int32, true));
        kgraph.control.addElement(Body(), {kernel}, {loadB});
        auto vgprB = kgraph.coordinates.addElement(VGPR());
        kgraph.coordinates.addElement(DataFlow(), {user0}, {vgprB});
        kgraph.mapper.connect<VGPR>(loadB, vgprB);

        kgraph.control.addElement(Sequence(), {loadB}, {assertOp});

        GraphReindexer reindexer;
        reindexer.coordinates.emplace(vgprA, vgprB);
        reindexExpressions(kgraph, assertOp, reindexer);

        auto condition = kgraph.control.get<AssertOp>(assertOp)->condition;
        auto lhs       = std::get<Expression::GreaterThan>(*condition).lhs;
        auto tag       = std::get<Expression::DataFlowTag>(*lhs).tag;
        EXPECT_EQ(tag, vgprB);
    }
}
