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

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Utils.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;

namespace ScopeTest
{
    class ScopeTest : public GenericContextFixture,
                      public ::testing::WithParamInterface<Scheduling::SchedulerProcedure>
    {
        void SetUp() override
        {
            Settings::getInstance()->set(Settings::AllowUnknownInstructions, true);
            GenericContextFixture::SetUp();
        }
    };

    TEST_P(ScopeTest, ScopeOperation)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        int dst1 = kgraph.coordinates.addElement(VGPR());
        int dst2 = kgraph.coordinates.addElement(VGPR());
        int dst3 = kgraph.coordinates.addElement(VGPR());

        int kernel = kgraph.control.addElement(Kernel());

        int scope1 = kgraph.control.addElement(Scope());
        int scope2 = kgraph.control.addElement(Scope());

        int assign1
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(11u)});
        int assign2
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(22u)});
        int assign3
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(33u)});
        int assign4
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(44u)});

        kgraph.mapper.connect(assign1, dst1, NaryArgument::DEST);
        kgraph.mapper.connect(assign2, dst2, NaryArgument::DEST);
        kgraph.mapper.connect(assign3, dst3, NaryArgument::DEST);
        kgraph.mapper.connect(assign4, dst1, NaryArgument::DEST);

        // kernel(base scope):
        //   - assign vector: 11u
        //   - scope1:
        //     - scope2:
        //         - assign vector: 33u
        //         - assign vector: 44u, using pre-existing vector
        //     - assign vector: 22u
        kgraph.control.addElement(Body(), {kernel}, {assign1});
        kgraph.control.addElement(Sequence(), {assign1}, {scope1});
        kgraph.control.addElement(Body(), {scope1}, {scope2});
        kgraph.control.addElement(Body(), {scope2}, {assign3});
        kgraph.control.addElement(Sequence(), {assign3}, {assign4});
        kgraph.control.addElement(Sequence(), {scope2}, {assign2});

        auto sched = GetParam();
        Settings::getInstance()->set(Settings::Scheduler, sched);

        m_context->schedule(generate(kgraph, m_context->kernel()));

        // assign1 should be to v0, which is deallocated before the end of the kernel
        // assign2 should be to v1, which is deallocated
        // assign3 should be to v1, which is deallocated

        // TODO: Rewrite ScopeTest test using the register allocation state
        // instead of comments so it's less fragile.

        auto kexpected = R"(
            // CodeGeneratorVisitor::generate() begin
            // generate({1})
            // Kernel(1) BEGIN
            // (op 1) generate({})
            // (op 1) end: generate({})
            // (op 1) generate({4})
            // (op 1) Assign VGPR 11:U32(4) BEGIN
            // (op 4) Assign dim(1) = 11:U32
            // tag 1: v**UNALLOCATED**
            // (op 4) Generate 11:U32 into DataFlowTag1: VGPR Value: UInt32 x 1: (unallocated)
            // Allocated DataFlowTag1: 1 VGPR (Value: UInt32) (op 4): v0
            v_mov_b32 v0, 11 // (op 4) call()
            // (op 1) Assign VGPR 11:U32(4) END
            // (op 1) Scope(2) BEGIN
            // (op 2) Lock Scope 2
            // (op 2) generate({3})
            // (op 2) Scope(3) BEGIN
            // (op 3) Lock Scope 3
            // (op 3) generate({6})
            // (op 3) Assign VGPR 33:U32(6) BEGIN
            // (op 6) Assign dim(3) = 33:U32
            // tag 3: v**UNALLOCATED**
            // (op 6) Generate 33:U32 into DataFlowTag3: VGPR Value: UInt32 x 1: (unallocated)
            // Allocated DataFlowTag3: 1 VGPR (Value: UInt32) (op 6): v1
            v_mov_b32 v1, 33 // (op 6) call()
            // (op 3) Assign VGPR 33:U32(6) END
            // (op 3) Assign VGPR 44:U32(7) BEGIN
            // (op 7) Assign dim(1) = 44:U32
            // (op 7) Generate 44:U32 into DataFlowTag1: VGPR Value: UInt32 x 1: v0
            v_mov_b32 v0, 44 // (op 7) call()
            // (op 3) Assign VGPR 44:U32(7) END
            // (op 3) end: generate({6})
            // Freeing DataFlowTag3: 1 VGPR (Value: UInt32) (op 6): v1
            // Deleting tag 3
            // (op 3) Unlock Scope 3
            // (op 2) Scope(3) END
            // (op 2) Assign VGPR 22:U32(5) BEGIN
            // (op 5) Assign dim(2) = 22:U32
            // tag 2: v**UNALLOCATED**
            // (op 5) Generate 22:U32 into DataFlowTag2: VGPR Value: UInt32 x 1: (unallocated)
            // Allocated DataFlowTag2: 1 VGPR (Value: UInt32) (op 5): v1
            v_mov_b32 v1, 22 // (op 5) call()
            // (op 2) Assign VGPR 22:U32(5) END
            // (op 2) end: generate({3})
            // Freeing DataFlowTag2: 1 VGPR (Value: UInt32) (op 5): v1
            // Deleting tag 2
            // (op 2) Unlock Scope 2
            // (op 1) Scope(2) END
            // (op 1) end: generate({4})
            // Freeing DataFlowTag1: 1 VGPR (Value: UInt32) (op 4): v0
            // Deleting tag 1
            // Kernel(1) END
            // end: generate({1})
            // CodeGeneratorVisitor::generate() end
        )";

        if(sched == Scheduling::SchedulerProcedure::Sequential)
            EXPECT_EQ(NormalizedSource(kexpected, true), NormalizedSource(output(), true));
    }

    /**
     * This is a regression test that will fail if codegen for the Scope operation doesn't
     * lock the scheduler.
     *
     * For the bug to appear, we need the following to happen:
     * Scope A and B are under separate locations of the graph.
     *
     * - Begin Scope A
     *                                         - Begin Scope B
     * - Allocate a DataFlow register (df1)
     *   (this goes into scope B)
     *                                         - End Scope B (will deallocate df1)
     * - Try to use df1 (throws an exception)
     */
    TEST_P(ScopeTest, ScopeOperationFalseNesting)
    {
        auto kgraph = rocRoller::KernelGraph::KernelGraph();

        int kernel = kgraph.control.addElement(Kernel());

        int dst1 = kgraph.coordinates.addElement(VGPR());
        int dst2 = kgraph.coordinates.addElement(VGPR());
        int dst3 = kgraph.coordinates.addElement(VGPR());
        int dst4 = kgraph.coordinates.addElement(VGPR());

        Expression::DataFlowTag dst3Tag{dst3, Register::Type::Vector, DataType::UInt32};
        auto                    dst3Expr = std::make_shared<Expression::Expression>(dst3Tag);

        Expression::DataFlowTag dst2Tag{dst2, Register::Type::Vector, DataType::UInt32};
        auto                    dst2Expr = std::make_shared<Expression::Expression>(dst2Tag);

        int scope1 = kgraph.control.addElement(Scope());
        int scope2 = kgraph.control.addElement(Scope());
        int scope3 = kgraph.control.addElement(Scope());

        int assign1
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(11u)});
        int assign2
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(22u)});
        int assign3
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(33u)});
        int assign4 = kgraph.control.addElement(
            Assign{Register::Type::Vector, (Expression::literal(44u) + dst3Expr) * dst3Expr});

        int assign5 = kgraph.control.addElement(
            Assign{Register::Type::Vector, Expression::literal(4u) + dst2Expr});

        kgraph.mapper.connect(assign1, dst1, NaryArgument::DEST);
        kgraph.mapper.connect(assign2, dst2, NaryArgument::DEST);
        kgraph.mapper.connect(assign3, dst3, NaryArgument::DEST);
        kgraph.mapper.connect(assign4, dst1, NaryArgument::DEST);
        kgraph.mapper.connect(assign5, dst4, NaryArgument::DEST);

        int barrier = kgraph.control.addElement(Barrier{});

        // kernel(base scope):
        //   - assign dst1 = 11u
        //   - scope1:
        //     - scope2:
        //         - barrier
        //         - assign dst3 = 33u
        //         - assign dst1 = (44u + dst3) * dst3
        //     - scope3:
        //         - assign dst2 = 22u
        //         - assign dst4 = dst2 + 4u

        kgraph.control.addElement(Body(), {kernel}, {assign1});
        kgraph.control.addElement(Sequence(), {assign1}, {scope1});
        kgraph.control.addElement(Body(), {scope1}, {scope2});
        kgraph.control.addElement(Body(), {scope2}, {barrier});
        kgraph.control.addElement(Body(), {barrier}, {assign3});
        kgraph.control.addElement(Sequence(), {assign3}, {assign4});
        kgraph.control.addElement(Body(), {scope1}, {scope3});
        kgraph.control.addElement(Body(), {scope3}, {assign2});
        kgraph.control.addElement(Sequence(), {assign2}, {assign5});

        auto sched = GetParam();
        Settings::getInstance()->set(Settings::Scheduler, sched);
        EXPECT_NO_THROW(m_context->schedule(generate(kgraph, m_context->kernel())));

        auto expected = R"(
            "coord1" -> "to_cntrl_1_5"
            "to_cntrl_1_5"[label="5->1: DEST", shape=cds]
            "cntrl5" -> "to_coord_5_1"
            "to_coord_5_1"[label="5->1: DEST", shape=cds]
            "coord2" -> "to_cntrl_2_6"
            "to_cntrl_2_6"[label="6->2: DEST", shape=cds]
            "cntrl6" -> "to_coord_6_2"
            "to_coord_6_2"[label="6->2: DEST", shape=cds]
            "coord3" -> "to_cntrl_3_7"
            "to_cntrl_3_7"[label="7->3: DEST", shape=cds]
            "cntrl7" -> "to_coord_7_3"
            "to_coord_7_3"[label="7->3: DEST", shape=cds]
            "coord1" -> "to_cntrl_1_8"
            "to_cntrl_1_8"[label="8->1: DEST", shape=cds]
            "cntrl8" -> "to_coord_8_1"
            "to_coord_8_1"[label="8->1: DEST", shape=cds]
            "coord4" -> "to_cntrl_4_9"
            "to_cntrl_4_9"[label="9->4: DEST", shape=cds]
            "cntrl9" -> "to_coord_9_4"
            "to_coord_9_4"[label="9->4: DEST", shape=cds]
        )";

        EXPECT_EQ(NormalizedSource(expected),
                  NormalizedSource(kgraph.mapper.toDOT("coord", "cntrl", true)));
    }

    auto schedulers()
    {
        std::vector<Scheduling::SchedulerProcedure> rv;
        for(int i = 0; i < static_cast<int>(Scheduling::SchedulerProcedure::Count); i++)
        {
            rv.push_back(static_cast<Scheduling::SchedulerProcedure>(i));
        }

        return rv;
    }

    INSTANTIATE_TEST_SUITE_P(ScopeTest, ScopeTest, ::testing::ValuesIn(schedulers()));
}
