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

        KernelInvocation inv{.workgroupSize = {64, 1, 1}};

        SECTION("Old Summary")
        {
            auto summary = rocRoller::KernelGraph::MemoryTracer::memoryTrace(kgraph, inv);

            // All visited nodes only access 4 banks in this graph
            for(const auto& [tag, access] : summary.accesses)
            {
                CHECK(access.accessedBanks.size() == 4);
            }

            if constexpr(true)
                std::cout << "\nSummary:\n" << summary.toString() << std::endl;
        }

        SECTION("Detailed summary of kernel graph")
        {
            auto tracer = MemoryTracer::MemoryTracer(kgraph);
            tracer.trace();

            auto model = MemoryTracer::LDSBankModel(4, 32, 512);

            auto workgroups            = 1;
            auto workitemsPerWorkgroup = product(inv.workgroupSize);
            tracer.simulateLaunch(model, workgroups, workitemsPerWorkgroup);

            auto detailed = model.detailedSummary(GPUArchitectureGFX::GFX942);

            if constexpr(true)
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

    TEST_CASE("LDS summary large", "[kernel-graph][lds-bank-model]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph;
        using namespace rocRoller::KernelGraph::MemoryTracer;

        auto model   = LDSBankModel(4, 32, 512);
        auto ldsRead = MemoryOpLDS{Direction::Load};

        const int  operationTag   = 1;
        const int  sourceTag      = 10;
        const int  destinationTag = 20;
        const uint workGroup      = 0;

        for(uint32_t threadId = 0; threadId < 64; ++threadId)
        {
            const auto baseAddr = threadId * 32;
            {
                const auto event = MemoryEventSimulated{
                    operationTag,
                    sourceTag,
                    destinationTag,
                    ldsRead,
                    baseAddr, // byteOffset
                    16, // bytesRequested
                    workGroup,
                    threadId // workItem
                };
                model.simulate(event);
            }
            {
                const auto event = MemoryEventSimulated{
                    operationTag,
                    sourceTag,
                    destinationTag,
                    ldsRead,
                    baseAddr + 16, // byteOffset
                    4, // bytesRequested
                    workGroup,
                    threadId // workItem
                };
                model.simulate(event);
            }
        }
        auto detailed = model.detailedSummary(GPUArchitectureGFX::GFX950);

        std::string detailedStr = detailed.toString();
        if constexpr(true)
            std::cout << detailedStr << std::endl;
    }

    TEST_CASE("createBankToAddressCounts", "[kernel-graph][lds-bank-model]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph::MemoryTracer;

        SECTION("Bank conflicts")
        {
            // Test multiple addresses accessing the same bank
            std::vector<uint32_t> addresses = {0, 128, 256}; // All map to bank 0
            uint                  dwords    = 1;

            auto bankCounts = LDSBankModel::createBankToAddressCounts(
                addresses, dwords, GPUArchitectureGFX::GFX942);

            // All 3 addresses should map to bank 0
            CHECK(bankCounts.size() == 1);
            CHECK(bankCounts[0] == 3);

            // Test conflicts with multi-dword accesses
            addresses = {0, 4, 124}; // With dwords=2, addresses 0 and 124 both touch bank 0
            dwords    = 2;

            bankCounts = LDSBankModel::createBankToAddressCounts(
                addresses, dwords, GPUArchitectureGFX::GFX942);

            // Address 0 touches banks 0-1
            // Address 4 touches banks 1-2
            // Address 124 touches banks 31-0 (wraparound)
            CHECK(bankCounts[0] == 2); // Accessed by addresses 0 and 124
            CHECK(bankCounts[1] == 2); // Accessed by addresses 0 and 4
            CHECK(bankCounts[2] == 1); // Accessed by address 4
            CHECK(bankCounts[31] == 1); // Accessed by address 124
        }

        SECTION("Wrap around")
        {
            // Test wraparound behavior when multi-dword access extends past last bank
            // For GFX942 with 32 banks, test a scenario that wraps around
            std::vector<uint32_t> addresses = {124}; // Banks 31, 0, 1, 2 for dwords=4
            uint                  dwords    = 4;

            auto bankCounts = LDSBankModel::createBankToAddressCounts(
                addresses, dwords, GPUArchitectureGFX::GFX942);

            CHECK(bankCounts.size() == 4);
            CHECK(bankCounts[31] == 1);
            CHECK(bankCounts[0] == 1);
            CHECK(bankCounts[1] == 1);
            CHECK(bankCounts[2] == 1);
        }
    }

    TEST_CASE("calculateBankConflictCycles", "[kernel-graph][lds-bank-model]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph::MemoryTracer;

        std::map<uint, uint> bankToAddressCounts = {};
        CHECK(LDSBankModel::calculateBankConflictCycles(bankToAddressCounts) == 0);

        bankToAddressCounts = {{0, 1}};
        CHECK(LDSBankModel::calculateBankConflictCycles(bankToAddressCounts) == 1);

        bankToAddressCounts = {
            {0, 2},
            {1, 3},
            {2, 1},
            {5, 3},
            {10, 2},
        };
        CHECK(LDSBankModel::calculateBankConflictCycles(bankToAddressCounts) == 3);
    }

    TEST_CASE("divideIntoThreadGroups", "[kernel-graph][lds-bank-model]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph::MemoryTracer;

        std::vector<uint32_t> addresses       = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};
        uint                  threadsPerClock = 4;

        auto groups = LDSBankModel::divideIntoThreadGroups(addresses, threadsPerClock);

        // Should have 3 groups of 4 addresses each
        CHECK(groups.size() == 3);
        CHECK(groups[0].size() == 4);
        CHECK(groups[1].size() == 4);
        CHECK(groups[2].size() == 4);

        CHECK(groups[0][0] == 0);
        CHECK(groups[0][1] == 4);
        CHECK(groups[0][2] == 8);
        CHECK(groups[0][3] == 12);

        CHECK(groups[1][0] == 16);
        CHECK(groups[1][1] == 20);
        CHECK(groups[1][2] == 24);
        CHECK(groups[1][3] == 28);

        CHECK(groups[2][0] == 32);
        CHECK(groups[2][1] == 36);
        CHECK(groups[2][2] == 40);
        CHECK(groups[2][3] == 44);
    }
}
