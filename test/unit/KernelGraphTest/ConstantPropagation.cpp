#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/ConstantPropagation.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

#include "../GenericContextFixture.hpp"

using namespace rocRoller;
using namespace rocRoller::KernelGraph::ControlGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;

namespace KernelGraphTest
{
    class KernelGraphConstantPropagationTest : public GenericContextFixture
    {
    };

    TEST_F(KernelGraphConstantPropagationTest, BetaIsZero)
    {
        auto graph0 = KernelGraph::KernelGraph();

        auto head  = graph0.control.addElement(NOP());
        auto alpha = graph0.control.addElement(LoadVGPR());
        auto beta  = graph0.control.addElement(LoadVGPR());
        auto c     = graph0.control.addElement(LoadTiled());
        auto lds = graph0.control.addElement(LoadLDSTile()); // assume the result for A*B is in LDS
        auto store = graph0.control.addElement(StoreTiled());

        auto alphaCoord         = graph0.coordinates.addElement(VGPR());
        auto betaCoord          = graph0.coordinates.addElement(VGPR());
        auto cCoord             = graph0.coordinates.addElement(MacroTile());
        auto ldsCoord           = graph0.coordinates.addElement(LDS());
        auto multiplyAlphaCoord = graph0.coordinates.addElement(VGPR());
        auto multiplyBetaCoord  = graph0.coordinates.addElement(VGPR());
        auto addCoord           = graph0.coordinates.addElement(VGPR());
        auto storeCoord         = graph0.coordinates.addElement(MacroTile());

        auto mulLHS1
            = graph0.coordinates.addElement(DataFlow(), {alphaCoord}, {multiplyAlphaCoord});
        auto mulRHS1 = graph0.coordinates.addElement(DataFlow(), {ldsCoord}, {multiplyAlphaCoord});
        auto mulLHS2 = graph0.coordinates.addElement(DataFlow(), {betaCoord}, {multiplyBetaCoord});
        auto mulRHS2 = graph0.coordinates.addElement(DataFlow(), {cCoord}, {multiplyBetaCoord});
        auto addLHS  = graph0.coordinates.addElement(DataFlow(), {multiplyAlphaCoord}, {addCoord});
        auto addRHS  = graph0.coordinates.addElement(DataFlow(), {multiplyBetaCoord}, {addCoord});

        graph0.mapper.connect<VGPR>(alpha, mulLHS1);
        graph0.mapper.connect<VGPR>(beta, mulLHS2);
        graph0.mapper.connect<MacroTile>(c, mulRHS2);
        graph0.mapper.connect<LDS>(lds, mulRHS1);
        graph0.mapper.connect<MacroTile>(store, storeCoord);

        auto DF = [](int tag) {
            return std::make_shared<Expression::Expression>(
                Expression::DataFlowTag{tag, Register::Type::Vector, DataType::Float});
        };

        auto multiplyAlpha
            = graph0.control.addElement(Assign{Register::Type::Vector, DF(mulLHS1) * DF(mulRHS1)});
        auto multiplyBeta
            = graph0.control.addElement(Assign{Register::Type::Vector, DF(mulLHS2) * DF(mulRHS2)});
        auto add
            = graph0.control.addElement(Assign{Register::Type::Vector, DF(addLHS) + DF(addRHS)});

        graph0.mapper.connect(multiplyAlpha, addLHS, NaryArgument::DEST);
        graph0.mapper.connect(multiplyBeta, addRHS, NaryArgument::DEST);
        graph0.mapper.connect(add, storeCoord, NaryArgument::DEST);

        graph0.control.addElement(Sequence(), {head}, {alpha});
        graph0.control.addElement(Sequence(), {alpha}, {lds});
        graph0.control.addElement(Sequence(), {lds}, {multiplyAlpha});
        graph0.control.addElement(Sequence(), {head}, {beta});
        graph0.control.addElement(Sequence(), {beta}, {c});
        graph0.control.addElement(Sequence(), {c}, {multiplyBeta});
        graph0.control.addElement(Sequence(), {multiplyAlpha}, {add});
        graph0.control.addElement(Sequence(), {multiplyBeta}, {add});
        graph0.control.addElement(Sequence(), {add}, {store});

        // check graph0 is doing alpha * lds + beta * c
        EXPECT_EQ(graph0.mapper.get<VGPR>(alpha), mulLHS1);
        EXPECT_EQ(graph0.mapper.get<LDS>(lds), mulRHS1);
        EXPECT_EQ(graph0.mapper.get<VGPR>(beta), mulLHS2);
        EXPECT_EQ(graph0.mapper.get<MacroTile>(c), mulRHS2);
        EXPECT_EQ(getDEST(graph0, multiplyAlpha), addLHS);
        EXPECT_EQ(getDEST(graph0, multiplyBeta), addRHS);

        auto graph1 = KernelGraph::ConstantPropagation().apply(graph0);

        EXPECT_EQ(graph0.control.getNodes().to<std::vector>().size(),
                  graph1.control.getNodes().to<std::vector>().size());
        EXPECT_EQ(graph0.control.getEdges().to<std::vector>().size(),
                  graph1.control.getEdges().to<std::vector>().size());

        EXPECT_EQ(graph1.control.getNodes<LoadTiled>().to<std::vector>().size(),
                  0); // check LoadTiled is removed

        auto graph1Assign = graph1.control.getNodes<Assign>().to<std::vector>();
        auto graph0Assign = graph0.control.getNodes<Assign>().to<std::vector>();

        // only 1 Assign node after transformation
        EXPECT_EQ(graph1Assign.size(), 1);

        // the transformation removes 2 Assign nodes
        EXPECT_EQ(graph0Assign.size() - graph1Assign.size(), 2);

        // only multiplyAlpha after transformation
        EXPECT_EQ(graph1Assign[0], multiplyAlpha);

        // check the destination of multiplyAlpha
        EXPECT_EQ(getDEST(graph0, add), getDEST(graph1, multiplyAlpha));
    }

}
