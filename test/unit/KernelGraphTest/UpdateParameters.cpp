#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>

#include "../GenericContextFixture.hpp"

#include "common/CommonGraphs.hpp"

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

    TEST_F(KernelGraphUpdateParametersTest, SetWorkitemCount)
    {
        using namespace rocRoller::KernelGraph;

        auto example = rocRollerTest::Graphs::GEMM<float>();

        int macK  = 16;
        int waveK = 8;

        example.setTileSize(128, 256, macK);
        example.setMFMA(32, 32, waveK, 1);
        example.setUseLDS(true, false, false);

        auto command = example.getCommand();
        auto kgraph  = example.getKernelGraph();
        auto params  = example.getCommandParameters();

        std::vector<GraphTransformPtr> transforms;
        transforms.push_back(std::make_shared<UpdateParameters>(params));
        transforms.push_back(std::make_shared<LowerTile>(params, m_context));
        transforms.push_back(std::make_shared<LowerTensorContraction>(params, m_context));
        transforms.push_back(std::make_shared<ConnectWorkgroups>());
        transforms.push_back(std::make_shared<UpdateWavefrontParameters>(params));
        for(auto& t : transforms)
            kgraph = kgraph.transform(t);

        // Without SetWorkitemCount, we should get a nullptr expression
        auto workitemCount = m_context->kernel()->workitemCount();
        EXPECT_TRUE(workitemCount[0] == nullptr);

        // Now apply SetWorkitemCount and try again
        kgraph = kgraph.transform(std::make_shared<SetWorkitemCount>(m_context));

        CommandArgumentPtr tensorDsizeX;
        {
            auto arguments = command->getArguments();
            for(auto argument : arguments)
            {
                std::cout << argument->name() << std::endl;
                if(argument->name() == "Tensor_4_size_0")
                    tensorDsizeX = argument;
            }
        }
        ASSERT_NE(tensorDsizeX, nullptr) << "D size not found";

        workitemCount = m_context->kernel()->workitemCount();

        auto one            = Expression::literal(1u);
        auto workgroupSizeX = Expression::literal(128u);

        auto expected
            = (((tensorDsizeX->expression() + workgroupSizeX) - one) / workgroupSizeX) * one;

        EXPECT_TRUE(Expression::identical(expected, workitemCount[0]));
    }
}
