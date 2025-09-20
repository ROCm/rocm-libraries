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

#include <rocRoller/AssemblyKernel.hpp>
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

        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

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
        transforms.push_back(std::make_shared<AddLDS>(params, m_context));
        transforms.push_back(std::make_shared<LowerTile>(params, m_context));
        transforms.push_back(std::make_shared<LowerTensorContraction>(params, m_context));
        transforms.push_back(std::make_shared<ConnectWorkgroups>(
            m_context, params->workgroupMappingDim, params->workgroupRemapXCC));
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
                if(argument->name() == "Tensor_4_size_0")
                    tensorDsizeX = argument;
            }
        }
        ASSERT_NE(tensorDsizeX, nullptr) << "D size not found";

        workitemCount = m_context->kernel()->workitemCount();

        auto one            = Expression::literal(1u);
        auto workgroupSizeX = Expression::literal(128u);

        auto expected = Expression::convert(DataType::UInt32,
                                            ((tensorDsizeX->expression() + workgroupSizeX) - one)
                                                / workgroupSizeX)
                        * one;

        EXPECT_TRUE(Expression::identical(expected, workitemCount[0]))
            << expected << "/" << workitemCount[0] << std::endl;
    }
}
