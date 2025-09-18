/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestContext.hpp"

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>

#include <common/CommonGraphs.hpp>

TEST_CASE("AssignComputeIndex", "[kernel-graph]")
{
    using namespace rocRoller;
    using namespace KernelGraph;
    using namespace ControlGraph;

    auto context = TestContext::ForDefaultTarget();

    auto example = rocRollerTest::Graphs::TileDoubleAdd<float>();

    example.setTileSize(16, 8);
    example.setSubTileSize(4, 2);

    auto params = example.getCommandParameters(512, 512);
    auto graph  = example.getKernelGraph();

    std::vector<GraphTransformPtr> transforms{
        std::make_shared<UpdateParameters>(params),
        std::make_shared<AddLDS>(params, context.get()),
        std::make_shared<LowerLinear>(context.get()),
        std::make_shared<LowerTile>(params, context.get()),
        std::make_shared<AddComputeIndex>(),
        std::make_shared<UpdateWavefrontParameters>(params),
        std::make_shared<AssignComputeIndex>(context.get()),
    };

    for(auto const& xform : transforms)
    {
        graph = graph.transform(xform);
    }

    auto verifyAssignComputeIndex = [&graph](int tag) {
        auto base = graph.mapper.get(
            tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::BASE});
        auto offset = graph.mapper.get(
            tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::OFFSET});
        auto stride = graph.mapper.get(
            tag, Connections::ComputeIndex{Connections::ComputeIndexArgument::STRIDE});
        auto opTag = tag;

        if(base < 0 && offset > 0)
        {
            auto candidate = graph.control.getOutputNodeIndices<Sequence>(opTag).to<std::vector>();
            opTag          = candidate[0];
            AssertFatal(candidate.size() == 1);
            auto dest = graph.mapper.get(candidate[0], NaryArgument::DEST);
            CHECK(dest == offset);
        }

        if(stride > 0)
        {
            auto candidate = graph.control.getOutputNodeIndices<Sequence>(opTag).to<std::vector>();
            AssertFatal(candidate.size() == 1);
            auto dest = graph.mapper.get(candidate[0], NaryArgument::DEST);
            CHECK(dest == stride);
        }
    };

    // search ComputeIndex operations
    auto isComputeIndexPredicate
        = [&graph](int x) { return graph.control.get<ComputeIndex>(x).has_value(); };
    auto candidates
        = graph.control.findNodes(*graph.control.roots().begin(), isComputeIndexPredicate)
              .to<std::vector>();

    // verify the assign operations connect to the correct offset/stride coordinate
    for(const auto& tag : candidates)
    {
        verifyAssignComputeIndex(tag);
    }
}
