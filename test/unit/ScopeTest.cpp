
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

#include "Expression.hpp"
#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::ControlHypergraph;
using namespace rocRoller::KernelGraph::CoordGraph;

namespace ScopeTest
{
    class ScopeTest : public GenericContextFixture
    {
    };

    TEST_F(ScopeTest, ScopeOperation)
    {
        KernelHypergraph kgraph = KernelHypergraph();

        int dst1 = kgraph.coordinates.addElement(CoordGraph::VGPR());
        int dst2 = kgraph.coordinates.addElement(CoordGraph::VGPR());

        int kernel = kgraph.control.addElement(Kernel());
        int scope  = kgraph.control.addElement(Scope());
        int assign1
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(11u)});
        int assign2
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(22u)});

        kgraph.mapper.connect<CoordGraph::VGPR>(assign1, dst1);
        kgraph.mapper.connect<CoordGraph::VGPR>(assign2, dst2);

        // kernel:
        // - assign vector: 11u
        // - scope:
        //   - assign vector: 22u
        kgraph.control.addElement(Body(), {kernel}, {assign1});
        kgraph.control.addElement(Sequence(), {assign1}, {scope});
        kgraph.control.addElement(Body(), {scope}, {assign2});

        m_context->schedule(generate(kgraph, nullptr, m_context->kernel()));

        // assign1 should be to v0, which is not deallocated
        // assign2 should be to v1, which is deallocated

        auto kexpected = R"(
            // CFCodeGeneratorVisitor::generate() begin
            // generate(set{1})
            // Kernel BEGIN
            // Begin Kernel
            // generate(set{3})
            // Assign VGPR 11j BEGIN
            // 11j
            // Allocated : 1 VGPR (Value: UInt32): v0
            v_mov_b32 v0, 11
            // Assign VGPR 11j END
            // Scope BEGIN
            // BEGIN SCOPE
            // generate(set{4})
            // Assign VGPR 22j BEGIN
            // 22j
            // Allocated : 1 VGPR (Value: UInt32): v1
            v_mov_b32 v1, 22
            // Assign VGPR 22j END
            // Freeing : 1 VGPR (Value: UInt32): v1
            // END SCOPE
            // Scope END
            // End Kernel
            // Kernel END
            // CFCodeGeneratorVisitor::generate() end
        )";

        EXPECT_THAT(output(), MatchesSourceIncludingComments(kexpected));
    }
}
