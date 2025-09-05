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

        KernelInvocation inv{.workgroupSize = {64, 1, 1}};

        auto summary = rocRoller::KernelGraph::MemoryTracer::memoryTrace(kgraph, inv);

        // All visited nodes only access 4 banks in this graph
        for(const auto& [tag, access] : summary.accesses)
        {
            CHECK(access.accessedBanks.size() == 4);
        }

        std::cout << "\nSummary:\n" << summary.toString() << std::endl;

        SECTION("Detailed summary of kernel graph")
        {
            auto tracer = MemoryTracer::MemoryTracer(kgraph);
            tracer.trace();

            auto model = MemoryTracer::LDSBankModel(4, 32, 512);

            auto workgroups            = 1;
            auto workitemsPerWorkgroup = product(inv.workgroupSize);
            tracer.simulateLaunch(model, workgroups, workitemsPerWorkgroup);

            auto detailed = model.detailedSummary(GPUArchitectureGFX::GFX942);

            std::cout << "\nDetailed Summary:\n" << detailed << std::endl;
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

    TEST_CASE("LDS detailed summary", "[kernel-graph][lds-bank-model]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::MemoryTracer;

        SECTION("Basic detailed summary")
        {
            LDSBankModel model(4, 32, 512);

            // Simulate some memory events
            MemoryEventSimulated event;
            event.bytesRequested = 4;
            event.operationTag   = 100;
            event.sourceTag      = 10;
            event.destinationTag = 20;
            event.memoryOp       = MemoryOpLDS{Direction::Load};

            // Simulate accesses from multiple workitems
            // With 32 threads per clock for b32 loads, thread groups will be:
            // Group 0: workitems 0-31
            // Group 1: workitems 32-63

            // Group 0 - create some bank conflicts
            event.workItem   = 0;
            event.byteOffset = 0; // Bank 0
            model.simulate(event);

            event.workItem   = 1;
            event.byteOffset = 128; // Bank 0 (conflict)
            model.simulate(event);

            event.workItem   = 2;
            event.byteOffset = 4; // Bank 1
            model.simulate(event);

            // Group 1 - different pattern
            event.workItem   = 32;
            event.byteOffset = 8; // Bank 2
            model.simulate(event);

            event.workItem   = 33;
            event.byteOffset = 12; // Bank 3
            model.simulate(event);

            auto detailed = model.detailedSummary(GPUArchitectureGFX::GFX942);

            // Verify operation exists
            CHECK(detailed.operationDetails.count(100) == 1);

            const auto& opDetail = detailed.operationDetails.at(100);
            CHECK(opDetail.threadsPerClock == 32);
            CHECK(opDetail.gfx == GPUArchitectureGFX::GFX942);
            CHECK(opDetail.conflictsPerClock.size() == 2); // Two thread groups

            // Check thread group 0
            const auto& group0 = opDetail.conflictsPerClock[0];
            CHECK(group0.threadGroupIndex == 0);
            CHECK(group0.workitemIds.size() == 3); // workitems 0, 1, 2
            CHECK(group0.maxConflictDegree == 2); // Bank 0 has 2 addresses
            CHECK(group0.bankToAddresses.at(0).size() == 2); // Bank 0: addresses 0, 128
            CHECK(group0.bankToAddresses.at(1).size() == 1); // Bank 1: address 4

            // Check thread group 1
            const auto& group1 = opDetail.conflictsPerClock[1];
            CHECK(group1.threadGroupIndex == 1);
            CHECK(group1.workitemIds.size() == 2); // workitems 32, 33
            CHECK(group1.maxConflictDegree == 1); // No conflicts
        }

        SECTION("Detailed summary with GFX950")
        {
            LDSBankModel model(4, 32, 512);

            MemoryEventSimulated event;
            event.bytesRequested = 4;
            event.operationTag   = 200;
            event.sourceTag      = 30;
            event.memoryOp       = MemoryOpLDS{Direction::Load};

            // For GFX950 b32 loads, still 32 threads per clock
            for(uint i = 0; i < 64; ++i)
            {
                event.workItem   = i;
                event.byteOffset = (i % 4) * 4; // Distribute across banks 0-3
                model.simulate(event);
            }

            auto        detailed = model.detailedSummary(GPUArchitectureGFX::GFX950);
            const auto& opDetail = detailed.operationDetails.at(200);

            CHECK(opDetail.threadsPerClock == 32);
            CHECK(opDetail.gfx == GPUArchitectureGFX::GFX950);
            CHECK(opDetail.conflictsPerClock.size() == 2); // 64 workitems / 32 per clock = 2 groups

            // Each group should have evenly distributed bank accesses
            for(const auto& conflict : opDetail.conflictsPerClock)
            {
                CHECK(conflict.bankToAddresses.size() == 4); // Using banks 0-3
                CHECK(conflict.maxConflictDegree == 8); // 32 threads / 4 banks = 8 per bank
            }

            std::cout << detailed.toString() << std::endl;
        }

        SECTION("DetailedSummary toString format")
        {
            LDSBankModel model(4, 32, 512);

            MemoryEventSimulated event;
            event.bytesRequested = 4;
            event.operationTag   = 300;
            event.sourceTag      = 40;
            event.memoryOp       = MemoryOpLDS{Direction::Store};

            event.workItem   = 0;
            event.byteOffset = 100;
            model.simulate(event);

            event.workItem   = 1;
            event.byteOffset = 200;
            model.simulate(event);

            auto detailed = model.detailedSummary(GPUArchitectureGFX::GFX942);
            auto str      = detailed.toString();

            // Verify the multi-level tabbed format
            CHECK(str.find("Operation tag 300:") != std::string::npos);
            CHECK(str.find("\tInstruction 0:") != std::string::npos);
            CHECK(str.find("\t\tThread group 0") != std::string::npos);
            CHECK(str.find("\t\t\tBank") != std::string::npos);
            CHECK(str.find("LDS Store") != std::string::npos);
            CHECK(str.find("32 threads/clock") != std::string::npos);
        }
    }
}
