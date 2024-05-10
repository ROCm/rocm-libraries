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

        auto head    = graph0.control.addElement(NOP());
        auto alpha   = graph0.control.addElement(LoadVGPR());
        auto beta    = graph0.control.addElement(LoadVGPR());
        auto forLoop = graph0.control.addElement(ForLoopOp());
        auto c       = graph0.control.addElement(LoadTiled());
        auto lds = graph0.control.addElement(LoadLDSTile()); // assume the result for A*B is in LDS
        auto store = graph0.control.addElement(StoreTiled());
        auto nopOp = graph0.control.addElement(NOP());

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

        graph0.control.addElement(Body(), {head}, {alpha});
        graph0.control.addElement(Sequence(), {alpha}, {lds});
        graph0.control.addElement(Sequence(), {head}, {beta});
        graph0.control.addElement(Sequence(), {beta}, {forLoop});
        graph0.control.addElement(Body(), {forLoop}, {nopOp});
        graph0.control.addElement(Sequence(), {nopOp}, {c});
        graph0.control.addElement(Sequence(), {nopOp}, {multiplyAlpha});
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

        // double the number of for-loop
        auto numForLoop0 = graph0.control.getNodes<ForLoopOp>().to<std::vector>().size();
        auto numForLoop1 = graph1.control.getNodes<ForLoopOp>().to<std::vector>().size();
        EXPECT_EQ(numForLoop0 * 2, numForLoop1);

        // add conditional operation (1 node and 2 edges) and duplicate the for-loop
        auto nodes0 = graph0.control.depthFirstVisit(head, Graph::Direction::Downstream)
                          .to<std::vector>()
                          .size();
        auto forLoopNodes = graph0.control.depthFirstVisit(forLoop, Graph::Direction::Downstream)
                                .to<std::vector>()
                                .size();
        auto nodes1 = graph1.control.depthFirstVisit(head, Graph::Direction::Downstream)
                          .to<std::vector>()
                          .size();
        EXPECT_EQ(nodes0 + forLoopNodes + 3, nodes1);

        // not add LoadTiled
        auto graph0LoadTiled = graph0.control.getNodes<LoadTiled>().to<std::vector>().size();
        auto graph1LoadTiled = graph1.control.getNodes<LoadTiled>().to<std::vector>().size();
        EXPECT_EQ(graph0LoadTiled, graph1LoadTiled);

        // add conditional operation
        auto graph0CondOp = graph0.control.getNodes<ConditionalOp>().to<std::vector>().size();
        auto graph1CondOp = graph1.control.getNodes<ConditionalOp>().to<std::vector>().size();
        EXPECT_EQ(graph1CondOp - graph0CondOp, numForLoop0);
    }

}
