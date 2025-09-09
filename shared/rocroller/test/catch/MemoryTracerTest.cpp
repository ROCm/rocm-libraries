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

    TEST_CASE("LDS bank mapping and conflict calculation", "[kernel-graph][lds-bank-model]")
    {
        using namespace rocRoller;
        using namespace rocRoller::KernelGraph::MemoryTracer;

        SECTION("Empty addresses")
        {
            std::vector<uint32_t> addresses;
            auto                  bankMapping = LDSBankModel::makeBankMapping(addresses, 4, 32);
            CHECK(bankMapping.empty()); // No banks accessed
        }

        SECTION("No conflicts - all different banks")
        {
            // Each address maps to a different bank: 0, 4, 8, 12 => banks 0, 1, 2, 3
            std::vector<uint32_t> addresses   = {0, 4, 8, 12};
            auto                  bankMapping = LDSBankModel::makeBankMapping(addresses, 4, 32);
            CHECK(bankMapping.size() == 4); // 4 different banks
            // Each bank should have exactly 1 address (no conflicts)
            for(const auto& [bank, addrs] : bankMapping)
            {
                CHECK(addrs.size() == 1);
            }
        }

        SECTION("Full conflicts - all same bank")
        {
            // All addresses map to bank 0: 0, 128, 256, 384
            // (0/4)%32=0, (128/4)%32=0, (256/4)%32=0, (384/4)%32=0
            std::vector<uint32_t> addresses   = {0, 128, 256, 384};
            auto                  bankMapping = LDSBankModel::makeBankMapping(addresses, 4, 32);
            CHECK(bankMapping.size() == 1); // Only bank 0
            CHECK(bankMapping.at(0).size() == 4); // All 4 addresses in bank 0 (4-way conflict)
        }

        SECTION("Partial conflicts")
        {
            // Bank 0: 0, 128 (2 addresses)
            // Bank 1: 4 (1 address)
            // Bank 2: 8 (1 address)
            std::vector<uint32_t> addresses   = {0, 4, 8, 128};
            auto                  bankMapping = LDSBankModel::makeBankMapping(addresses, 4, 32);
            CHECK(bankMapping.size() == 3); // 3 different banks
            CHECK(bankMapping.at(0).size() == 2); // Bank 0 has 2 addresses (2-way conflict)
            CHECK(bankMapping.at(1).size() == 1); // Bank 1 has 1 address
            CHECK(bankMapping.at(2).size() == 1); // Bank 2 has 1 address
        }

        SECTION("Different bank configurations")
        {
            // Test with 16 banks instead of 32
            std::vector<uint32_t> addresses   = {0, 64, 128}; // All map to bank 0
            auto                  bankMapping = LDSBankModel::makeBankMapping(addresses, 4, 16);
            CHECK(bankMapping.size() == 1); // Only bank 0
            CHECK(bankMapping.at(0).size() == 3); // 3-way conflict

            // Test with different entry width (8 bytes)
            addresses   = {0, 256, 512}; // All map to bank 0
            bankMapping = LDSBankModel::makeBankMapping(addresses, 8, 32);
            CHECK(bankMapping.size() == 1); // Only bank 0
            CHECK(bankMapping.at(0).size() == 3); // 3-way conflict
        }

        SECTION("Verify bank mapping contents")
        {
            std::vector<uint32_t> addresses = {0, 4, 8, 128, 132, 136};
            // Bank 0: 0, 128
            // Bank 1: 4, 132
            // Bank 2: 8, 136
            auto bankMapping = LDSBankModel::makeBankMapping(addresses, 4, 32);

            // Check bank 0
            CHECK(bankMapping.at(0).size() == 2);
            CHECK(std::find(bankMapping.at(0).begin(), bankMapping.at(0).end(), 0)
                  != bankMapping.at(0).end());
            CHECK(std::find(bankMapping.at(0).begin(), bankMapping.at(0).end(), 128)
                  != bankMapping.at(0).end());

            // Check bank 1
            CHECK(bankMapping.at(1).size() == 2);
            CHECK(std::find(bankMapping.at(1).begin(), bankMapping.at(1).end(), 4)
                  != bankMapping.at(1).end());
            CHECK(std::find(bankMapping.at(1).begin(), bankMapping.at(1).end(), 132)
                  != bankMapping.at(1).end());

            // Check bank 2
            CHECK(bankMapping.at(2).size() == 2);
            CHECK(std::find(bankMapping.at(2).begin(), bankMapping.at(2).end(), 8)
                  != bankMapping.at(2).end());
            CHECK(std::find(bankMapping.at(2).begin(), bankMapping.at(2).end(), 136)
                  != bankMapping.at(2).end());
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
}
