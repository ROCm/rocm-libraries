
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

#include "Expression.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;

namespace ScopeTest
{
    class ScopeTest : public GenericContextFixture
    {
    };

    TEST_F(ScopeTest, ScopeOperation)
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

        m_context->schedule(generate(kgraph, m_context->kernel()));

        // assign1 should be to v0, which is deallocated before the end of the kernel
        // assign2 should be to v1, which is deallocated
        // assign3 should be to v1, which is deallocated

        auto kexpected = R"(
            // CodeGeneratorVisitor::generate() begin
            // generate(set{1})
            // Kernel(1) BEGIN
            // generate(set{4})
            // Assign VGPR 11j(4) BEGIN
            // 11j
            // Allocated : 1 VGPR (Value: UInt32): v0
            v_mov_b32 v0, 11
            // Assign VGPR 11j(4) END
            // Scope(2) BEGIN
            // generate(set{3})
            // Scope(3) BEGIN
            // generate(set{6})
            // Assign VGPR 33j(6) BEGIN
            // 33j
            // Allocated : 1 VGPR (Value: UInt32): v1
            v_mov_b32 v1, 33
            // Assign VGPR 33j(6) END
            // Assign VGPR 44j(7) BEGIN
            // 44j
            v_mov_b32 v0, 44
            // Assign VGPR 44j(7) END
            // Freeing : 1 VGPR (Value: UInt32): v1
            // Scope(3) END
            // Assign VGPR 22j(5) BEGIN
            // 22j
            // Allocated : 1 VGPR (Value: UInt32): v1
            v_mov_b32 v1, 22
            // Assign VGPR 22j(5) END
            // Freeing : 1 VGPR (Value: UInt32): v1
            // Scope(2) END
            // Freeing : 1 VGPR (Value: UInt32): v0
            // Kernel(1) END
            // CodeGeneratorVisitor::generate() end
        )";

        EXPECT_THAT(output(), MatchesSourceIncludingComments(kexpected));
    }
}
