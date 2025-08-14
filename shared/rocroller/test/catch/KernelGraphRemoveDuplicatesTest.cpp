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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>

#include <common/CommonGraphs.hpp>

#include "TestContext.hpp"

TEST_CASE("Remove duplicates", "[kernel-graph]")
{
    using namespace rocRoller;
    using namespace rocRoller::KernelGraph;
    using namespace rocRoller::KernelGraph::ControlGraph;

    auto ctx     = TestContext::ForDefaultTarget().get();
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    example.setTileSize(128, 96, 32);
    example.setMFMA(32, 32, 16, 1);
    example.setUseLDS(true, true, false);

    // Workgroup size: 128x4
    // K loops: 2
    // Jamming: 2x1

    auto graph0 = example.getKernelGraph();
    auto params = example.getCommandParameters();

    std::vector<GraphTransformPtr> transforms;
    transforms.push_back(std::make_shared<UpdateParameters>(params));
    transforms.push_back(std::make_shared<AddLDS>(params, ctx));
    transforms.push_back(std::make_shared<LowerLinear>(ctx));
    transforms.push_back(std::make_shared<LowerTile>(params, ctx));
    transforms.push_back(std::make_shared<LowerTensorContraction>(params, ctx));
    transforms.push_back(std::make_shared<ConnectWorkgroups>(
        ctx, params->workgroupMappingDim, params->workgroupRemapXCC));
    transforms.push_back(std::make_shared<UnrollLoops>(params, ctx));
    transforms.push_back(std::make_shared<FuseLoops>());

    for(auto& t : transforms)
        graph0 = graph0.transform(t);

    auto graph1 = graph0.transform(std::make_shared<RemoveDuplicates>());

    // LoadTiled: A A, B B, C C
    // After removing 2x1 jamming: A, B, C C
    CHECK(graph0.control.getElements<LoadTiled>().to<std::vector>().size() == 6);
    CHECK(graph1.control.getElements<LoadTiled>().to<std::vector>().size() == 4);

    // StoreLDSTile: A A, B B
    // After removing 2x1 jamming: A, B
    CHECK(graph0.control.getElements<StoreLDSTile>().to<std::vector>().size() == 4);
    CHECK(graph1.control.getElements<StoreLDSTile>().to<std::vector>().size() == 2);

    // LoadLDSTile: A A A A, B B B B
    // After removing 2x1 jamming: A A A A, B B
    CHECK(graph0.control.getElements<LoadLDSTile>().to<std::vector>().size() == 8);
    CHECK(graph1.control.getElements<LoadLDSTile>().to<std::vector>().size() == 6);
}
