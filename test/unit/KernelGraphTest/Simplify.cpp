#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/Simplify.hpp>

#include "../GenericContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph::ControlGraph;

namespace KernelGraphTest
{
    class KernelGraphSimplifyTest : public GenericContextFixture
    {
    };

    TEST_F(KernelGraphSimplifyTest, BasicRedundantSequence)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        graph0.control.addElement(Sequence(), {A}, {B});
        graph0.control.addElement(Sequence(), {B}, {C});

        graph0.control.addElement(Sequence(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 3);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);
    }

    TEST_F(KernelGraphSimplifyTest, BasicRedundantBody)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        graph0.control.addElement(Body(), {A}, {B});
        graph0.control.addElement(Sequence(), {B}, {C});

        graph0.control.addElement(Body(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 3);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);
    }

    TEST_F(KernelGraphSimplifyTest, DoubleRedundantBody)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto A = graph0.control.addElement(NOP());
        auto B = graph0.control.addElement(NOP());
        auto C = graph0.control.addElement(NOP());

        graph0.control.addElement(Body(), {A}, {B});
        graph0.control.addElement(Body(), {A}, {B});
        graph0.control.addElement(Sequence(), {B}, {C});

        graph0.control.addElement(Body(), {A}, {C});

        auto graph1 = KernelGraph::Simplify().apply(graph0);

        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(), 4);
        EXPECT_EQ(graph1.control.getEdges().to<std::vector>().size(), 2);
    }
}
