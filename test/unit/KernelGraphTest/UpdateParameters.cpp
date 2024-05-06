#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/UpdateParameters.hpp>

#include "../GenericContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;

namespace KernelGraphTest
{

    class KernelGraphUpdateParametersTest : public GenericContextFixture
    {
    };

    TEST_F(KernelGraphUpdateParametersTest, TileInfoPropagateAdd)
    {
        auto DF = [](int tag) {
            return std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{tag, Register::Type::Vector, DataType::Float});
        };

        auto graph0 = KernelGraph::KernelGraph();

        auto tagA = graph0.coordinates.addElement(MacroTile(Operations::OperationTag(0)));
        auto tagB = graph0.coordinates.addElement(MacroTile(Operations::OperationTag(1)));
        auto tagD = graph0.coordinates.addElement(MacroTile(Operations::OperationTag(2)));

        auto tileA = MacroTile({4, 5}, MemoryType::VGPR);
        auto tileB = MacroTile({4, 5}, MemoryType::VGPR);

        auto expr = DF(tagA) + DF(tagB);

        auto kernel  = graph0.control.addElement(Kernel());
        auto assignD = graph0.control.addElement(Assign{Register::Type::Vector, expr});
        graph0.mapper.connect(assignD, tagD, NaryArgument::DEST);

        graph0.control.addElement(Sequence(), {kernel}, {assignD});

        auto params = std::make_shared<CommandParameters>();
        params->setDimensionInfo(Operations::OperationTag(0), tileA);
        params->setDimensionInfo(Operations::OperationTag(1), tileB);

        // Result of A + B should have same size as A (and B)
        auto graph1 = KernelGraph::UpdateParameters(params).apply(graph0);

        auto tileD = graph1.coordinates.getNode<MacroTile>(tagD);
        EXPECT_EQ(tileD.sizes[0], tileA.sizes[0]);
        EXPECT_EQ(tileD.sizes[1], tileA.sizes[1]);

        // If A and B are different sizes, propagating size to A + B should fail
        graph0.coordinates.setElement(tagD, MacroTile(Operations::OperationTag(2)));
        tileB = MacroTile({4, 7}, MemoryType::VGPR);
        params->setDimensionInfo(Operations::OperationTag(1), tileB);
        EXPECT_THROW(KernelGraph::UpdateParameters(params).apply(graph0), FatalError);
    }
}
