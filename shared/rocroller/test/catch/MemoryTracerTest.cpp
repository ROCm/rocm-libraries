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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/KernelGraph/Transforms/MemoryTracer.hpp>

#include <common/CommonGraphs.hpp>

#include "TestContext.hpp"

namespace MemoryTracerTest
{
    TEST_CASE("LDS bank conflicts", "[kernel-graph]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::ControlGraph;
        using namespace rocRoller::KernelGraph::CoordinateGraph;

        using GD = Graph::Direction;

        auto context = TestContext::ForTestDevice();
        auto example = rocRollerTest::Graphs::GEMM(DataType::Float);

        example.setTileSize(128, 256, 8);
        example.setMFMA(32, 32, 2, 1);
        example.setUseLDS(true, false, false); // For easier assertions

        auto kgraph  = example.getKernelGraph();
        auto params  = example.getCommandParameters();
        auto command = example.getCommand();

        params->unrollK           = 4;
        params->prefetch          = true;
        params->prefetchInFlight  = 2;
        params->prefetchLDSFactor = 2;

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
        transforms.push_back(std::make_shared<ConnectWorkgroups>(params, context.get()));
        transforms.push_back(std::make_shared<UnrollLoops>(params, context.get()));
        transforms.push_back(std::make_shared<FuseLoops>());
        transforms.push_back(std::make_shared<RemoveDuplicates>());
        transforms.push_back(std::make_shared<OrderEpilogueBlocks>());
        transforms.push_back(std::make_shared<CleanLoops>());
        transforms.push_back(std::make_shared<AddPrefetch>(params, context.get()));
        transforms.push_back(std::make_shared<AddComputeIndex>());
        transforms.push_back(std::make_shared<AddPRNG>(context.get()));
        transforms.push_back(std::make_shared<UpdateWavefrontParameters>(params));

        for(auto& t : transforms)
            kgraph = kgraph.transform(t);

        std::ofstream file("lds_bank_conflicts.dot");
        file << kgraph.toDOT() << std::endl;

        auto summary = memoryTrace(kgraph, {});
        std::cout << summary << std::endl;
    }
}
