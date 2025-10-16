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

TEST_CASE("StreamK multiple fix-ups", "[streamK][kernel-graph]")
{
    using namespace rocRoller;
    using namespace KernelGraph;
    using namespace ControlGraph;

    auto context = TestContext::ForTestDevice();
    auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

    example.setTileSize(128, 256, 8);
    example.setMFMA(32, 32, 2, 1);
    example.setUseLDS(false, false, false);
    example.setPrefetch(false, 0, 0, false);

    auto numWGs     = example.getTotalWorkgroupSize();
    auto numWGsExpr = std::make_shared<Expression::Expression>(numWGs);

    std::vector<GraphTransformPtr> transforms;
    transforms.push_back(std::make_shared<IdentifyParallelDimensions>());
    transforms.push_back(std::make_shared<OrderMemory>(false));
    transforms.push_back(std::make_shared<UpdateParameters>(params));
    transforms.push_back(std::make_shared<AddLDS>(params, context.get()));
    transforms.push_back(std::make_shared<LowerLinear>(context.get()));
    transforms.push_back(std::make_shared<LowerTile>(params, context.get()));
    transforms.push_back(std::make_shared<LowerTensorContraction>(params, context.get()));
    transforms.push_back(std::make_shared<Simplify>());
    transforms.push_back(std::make_shared<FuseExpressions>());

    SECTION("Standard StreamK Multiple Fixups")
    {
        example.setStreamK(StreamKMode::Standard);

        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        for(auto& t : transforms)
            kgraph = kgraph.transform(t);

        std::ofstream outFile0("graph0-standard.dot"); // Create and open a file

        outFile0 << kgraph.toDOT();

        kgraph = kgraph.transform(std::make_shared<AddStreamK>(
            context.get(), params, rocRoller::XLOOP, rocRoller::KLOOP, numWGsExpr));

        std::ofstream outFile1("graph1-standard.dot"); // Create and open a file
        outFile1 << kgraph.toDOT();

        std::ofstream outFile2("graph2-mapper-standard.dot"); // Create and open a file
        outFile2 << kgraph.toDOT(true);
    }

    SECTION("TwoTile StreamK Multiple Fixups")
    {
        example.setStreamK(StreamKMode::TwoTile);

        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        for(auto& t : transforms)
            kgraph = kgraph.transform(t);

        std::ofstream outFile0("graph0-2tile.dot"); // Create and open a file

        outFile0 << kgraph.toDOT();

        kgraph = kgraph.transform(std::make_shared<AddStreamK>(
            context.get(), params, rocRoller::XLOOP, rocRoller::KLOOP, numWGsExpr));

        std::ofstream outFile1("graph1-2tile.dot"); // Create and open a file
        outFile1 << kgraph.toDOT();

        std::ofstream outFile2("graph2-mapper-2tile.dot"); // Create and open a file
        outFile2 << kgraph.toDOT(true);
    }

    SECTION("TwoTileDPFirst StreamK Multiple Fixups")
    {
        example.setStreamK(StreamKMode::TwoTileDPFirst);

        auto kgraph = example.getKernelGraph();
        auto params = example.getCommandParameters();

        for(auto& t : transforms)
            kgraph = kgraph.transform(t);

        std::ofstream outFile0("graph0-dpfirst.dot"); // Create and open a file

        outFile0 << kgraph.toDOT();

        kgraph = kgraph.transform(std::make_shared<AddStreamK>(
            context.get(), params, rocRoller::XLOOP, rocRoller::KLOOP, numWGsExpr));

        std::ofstream outFile1("graph1-dpfirst.dot"); // Create and open a file
        outFile1 << kgraph.toDOT();

        std::ofstream outFile2("graph2-dpfirst-mapper.dot"); // Create and open a file
        outFile2 << kgraph.toDOT(true);
    }
}
