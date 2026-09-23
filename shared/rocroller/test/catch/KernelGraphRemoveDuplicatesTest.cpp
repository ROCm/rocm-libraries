// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
    transforms.push_back(std::make_shared<ConnectWorkgroups>(ctx));
    transforms.push_back(std::make_shared<WorkgroupRemapXCC>(ctx, params->workgroupRemapXCC));
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
