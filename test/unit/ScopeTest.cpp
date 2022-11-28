
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
        int dst3 = kgraph.coordinates.addElement(CoordGraph::VGPR());

        int kernel = kgraph.control.addElement(Kernel());

        int scope1 = kgraph.control.addElement(Scope());
        int scope2 = kgraph.control.addElement(Scope());

        int assign1
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(11u)});
        int assign2
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(22u)});
        int assign3
            = kgraph.control.addElement(Assign{Register::Type::Vector, Expression::literal(33u)});

        kgraph.mapper.connect<CoordGraph::VGPR>(assign1, dst1);
        kgraph.mapper.connect<CoordGraph::VGPR>(assign2, dst2);
        kgraph.mapper.connect<CoordGraph::VGPR>(assign3, dst3);

        // kernel:
        // - assign vector: 11u
        // - scope1:
        //   - scope2:
        //       - scope3:
        //         - assign vector: 33u
        //   - assign vector: 22u
        kgraph.control.addElement(Body(), {kernel}, {assign1});
        kgraph.control.addElement(Sequence(), {assign1}, {scope1});
        kgraph.control.addElement(Body(), {scope1}, {scope2});
        kgraph.control.addElement(Body(), {scope2}, {assign3});
        kgraph.control.addElement(Sequence(), {scope2}, {assign2});

        m_context->schedule(generate(kgraph, m_context->kernel()));

        // assign1 should be to v0, which is not deallocated
        // assign2 should be to v1, which is deallocated
        // assign3 should be to v1, which is deallocated

        auto kexpected = R"(
            // CFCodeGeneratorVisitor::generate() begin
            // generate(set{1})
            // Kernel BEGIN
            // Begin Kernel
            // generate(set{4})
            // Assign VGPR 11j BEGIN
            // 11j
            // Allocated : 1 VGPR (Value: UInt32): v0
            v_mov_b32 v0, 11
            // Assign VGPR 11j END
            // Scope BEGIN
            // generate(set{3})
            // Scope BEGIN
            // generate(set{6})
            // Assign VGPR 33j BEGIN
            // 33j
            // Allocated : 1 VGPR (Value: UInt32): v1
            v_mov_b32 v1, 33
            // Assign VGPR 33j END
            // Freeing : 1 VGPR (Value: UInt32): v1
            // Scope END
            // Assign VGPR 22j BEGIN
            // 22j
            // Allocated : 1 VGPR (Value: UInt32): v1
            v_mov_b32 v1, 22
            // Assign VGPR 22j END
            // Freeing : 1 VGPR (Value: UInt32): v1
            // Scope END
            // End Kernel
            // Kernel END
            // CFCodeGeneratorVisitor::generate() end
        )";

        EXPECT_THAT(output(), MatchesSourceIncludingComments(kexpected));
    }
}
