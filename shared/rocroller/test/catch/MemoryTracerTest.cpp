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
#include <rocRoller/KernelGraph/Transforms/LDSBankModel.hpp>
#include <rocRoller/KernelGraph/Transforms/MemoryTracer.hpp>

#include <common/CommonGraphs.hpp>

#include "TestContext.hpp"

namespace MemoryTracerTest
{
    TEST_CASE("LDS bank conflicts", "[kernel-graph][lds-bank-model]")
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
        transforms.push_back(std::make_shared<ConnectWorkgroups>(
            context.get(), params->workgroupMappingDim, params->workgroupRemapXCC));
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

        KernelInvocation inv{.workgroupSize = {256, 1, 1}};

        auto summary = rocRoller::KernelGraph::MemoryTracer::memoryTrace(kgraph, inv);
        std::cout << summary << std::endl;

        // All visited nodes only access 4 banks in this graph
        for(const auto& [tag, access] : summary.accesses)
        {
            CHECK(access.accessedBanks.size() == 4);
        }
    }

    TEST_CASE("LDS threads per clock", "[kernel-graph][lds-bank-model]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph::MemoryTracer;

        SECTION("GFX950 read operations")
        {
            auto ldsRead = MemoryOpLDS{Direction::Load};

            CHECK(LDSBankModel::getThreadsPerClock(ldsRead, 1, GPUArchitectureGFX::GFX950) == 32);
            CHECK(LDSBankModel::getThreadsPerClock(ldsRead, 2, GPUArchitectureGFX::GFX950) == 32);
            CHECK(LDSBankModel::getThreadsPerClock(ldsRead, 3, GPUArchitectureGFX::GFX950) == 8);
            CHECK(LDSBankModel::getThreadsPerClock(ldsRead, 4, GPUArchitectureGFX::GFX950) == 16);
        }

        SECTION("GFX950 write operations")
        {
            auto ldsWrite = MemoryOpLDS{Direction::Store};

            CHECK(LDSBankModel::getThreadsPerClock(ldsWrite, 1, GPUArchitectureGFX::GFX950) == 32);
            CHECK(LDSBankModel::getThreadsPerClock(ldsWrite, 2, GPUArchitectureGFX::GFX950) == 16);
            CHECK(LDSBankModel::getThreadsPerClock(ldsWrite, 4, GPUArchitectureGFX::GFX950) == 8);
        }

        SECTION("Non-GFX950 operations")
        {
            auto ldsRead = MemoryOpLDS{Direction::Load};

            CHECK(LDSBankModel::getThreadsPerClock(ldsRead, 1, GPUArchitectureGFX::GFX942) == 32);
            CHECK(LDSBankModel::getThreadsPerClock(ldsRead, 2, GPUArchitectureGFX::GFX942) == 16);
            CHECK(LDSBankModel::getThreadsPerClock(ldsRead, 4, GPUArchitectureGFX::GFX942) == 8);
        }

        SECTION("Invalid dword count")
        {
            auto ldsRead = MemoryOpLDS{Direction::Load};

            CHECK_THROWS_AS(
                LDSBankModel::getThreadsPerClock(ldsRead, 5, GPUArchitectureGFX::GFX950),
                FatalError);
        }
    }

    TEST_CASE("LDS bank conflict calculation", "[kernel-graph][lds-bank-model]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph::MemoryTracer;

        SECTION("Empty addresses")
        {
            std::vector<uint32_t> addresses;
            CHECK(LDSBankModel::calculateBankConflicts(addresses, 4, 32) == 0);
        }

        SECTION("No conflicts - all different banks")
        {
            // Each address maps to a different bank: 0, 4, 8, 12 => banks 0, 1, 2, 3
            std::vector<uint32_t> addresses = {0, 4, 8, 12};
            CHECK(LDSBankModel::calculateBankConflicts(addresses, 4, 32) == 1);
        }

        SECTION("Full conflicts - all same bank")
        {
            // All addresses map to bank 0: 0, 128, 256, 384
            // (0/4)%32=0, (128/4)%32=0, (256/4)%32=0, (384/4)%32=0
            std::vector<uint32_t> addresses = {0, 128, 256, 384};
            CHECK(LDSBankModel::calculateBankConflicts(addresses, 4, 32) == 4);
        }

        SECTION("Partial conflicts")
        {
            // Bank 0: 0, 128 (2 addresses)
            // Bank 1: 4 (1 address)
            // Bank 2: 8 (1 address)
            std::vector<uint32_t> addresses = {0, 4, 8, 128};
            CHECK(LDSBankModel::calculateBankConflicts(addresses, 4, 32) == 2);
        }

        SECTION("Different bank configurations")
        {
            // Test with 16 banks instead of 32
            std::vector<uint32_t> addresses = {0, 64, 128}; // All map to bank 0
            CHECK(LDSBankModel::calculateBankConflicts(addresses, 4, 16) == 3);

            // Test with different entry width (8 bytes)
            addresses = {0, 256, 512}; // All map to bank 0
            CHECK(LDSBankModel::calculateBankConflicts(addresses, 8, 32) == 3);
        }
    }
}
