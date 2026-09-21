// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/All.hpp>
#include <rocRoller/Scheduling/LDSModel.hpp>

#include <common/CommonGraphs.hpp>
#include <common/Scheduling.hpp>

#include "CustomSections.hpp"
#include "TestContext.hpp"

namespace LDSModelTest
{
    using namespace rocRoller;
    using namespace rocRoller::Scheduling::LDSModel;

    TEST_CASE("LDS model get threads per clock", "[lds-bank-model]")
    {
        SECTION("GFX950 read operations")
        {
            auto ldsRead = MemoryOpLDS{LdsDirection::Read};

            CHECK(getThreadsPerClock(ldsRead, 1, GPUArchitectureGFX::GFX950) == 16);
            CHECK(getThreadsPerClock(ldsRead, 2, GPUArchitectureGFX::GFX950) == 16);
            CHECK(getThreadsPerClock(ldsRead, 3, GPUArchitectureGFX::GFX950) == 4);
            CHECK(getThreadsPerClock(ldsRead, 4, GPUArchitectureGFX::GFX950) == 8);
        }

        SECTION("GFX950 write operations")
        {
            auto ldsWrite = MemoryOpLDS{LdsDirection::Write};

            CHECK(getThreadsPerClock(ldsWrite, 1, GPUArchitectureGFX::GFX950) == 16);
            CHECK(getThreadsPerClock(ldsWrite, 2, GPUArchitectureGFX::GFX950) == 8);
            CHECK(getThreadsPerClock(ldsWrite, 3, GPUArchitectureGFX::GFX950) == 4);
            CHECK(getThreadsPerClock(ldsWrite, 4, GPUArchitectureGFX::GFX950) == 4);
        }

        SECTION("Non-GFX950 operations")
        {
            auto ldsRead = MemoryOpLDS{LdsDirection::Read};

            CHECK(getThreadsPerClock(ldsRead, 1, GPUArchitectureGFX::GFX942) == 16);
            CHECK(getThreadsPerClock(ldsRead, 2, GPUArchitectureGFX::GFX942) == 8);
            CHECK(getThreadsPerClock(ldsRead, 3, GPUArchitectureGFX::GFX942) == 4);
            CHECK(getThreadsPerClock(ldsRead, 4, GPUArchitectureGFX::GFX942) == 4);
        }

        SECTION("Invalid dword count")
        {
            auto ldsRead = MemoryOpLDS{LdsDirection::Read};

            CHECK_THROWS_AS(getThreadsPerClock(ldsRead, 5, GPUArchitectureGFX::GFX950), FatalError);
        }
    }

    TEST_CASE("LDS model make bank to address counts", "[lds-bank-model]")
    {
        SECTION("Bank conflicts")
        {
            // Test multiple addresses accessing the same bank
            std::vector<size_t> addresses = {0, 128, 256}; // x / 4 mod 32 = 0 for all addresses
            uint                dwords    = 1;
            auto                ldsRead   = MemoryOpLDS{LdsDirection::Read};

            auto bankCounts
                = createBankToAddressCounts(addresses, dwords, GPUArchitectureGFX::GFX942, ldsRead);

            CHECK(bankCounts.size() == 1);
            CHECK(bankCounts[0] == 3);

            auto more = {4, 16};
            addresses.insert(addresses.end(), more.begin(), more.end());

            bankCounts
                = createBankToAddressCounts(addresses, dwords, GPUArchitectureGFX::GFX942, ldsRead);

            CHECK(bankCounts.size() == 3);
            CHECK(bankCounts[0] == 3);
            CHECK(bankCounts[1] == 1); // 4 / 4 mod 32 = 1
            CHECK(bankCounts[4] == 1); // 16 / 4 mod 32 = 4
        }

        SECTION("Wrap around")
        {
            std::vector<size_t> addresses = {124}; // 124 / 4 mod 32 = 31
            uint                dwords    = 4; // Accesses banks 31, 0, 1, 2
            auto                ldsRead   = MemoryOpLDS{LdsDirection::Read};

            auto bankCounts
                = createBankToAddressCounts(addresses, dwords, GPUArchitectureGFX::GFX942, ldsRead);

            CHECK(bankCounts.size() == 4);
            CHECK(bankCounts[31] == 1);
            CHECK(bankCounts[0] == 1);
            CHECK(bankCounts[1] == 1);
            CHECK(bankCounts[2] == 1);
        }
    }

    TEST_CASE("LDS model calculate bank conflict cycles", "[lds-bank-model]")
    {
        std::map<uint, uint> bankToAddressCounts = {};
        CHECK(calculateBankConflictCycles(bankToAddressCounts) == 0);

        bankToAddressCounts = {{0, 1}};
        CHECK(calculateBankConflictCycles(bankToAddressCounts) == 1);

        bankToAddressCounts = {
            {0, 2},
            {1, 3},
            {2, 1},
            {5, 3},
            {10, 2},
        };
        CHECK(calculateBankConflictCycles(bankToAddressCounts) == 3);
    }

    TEST_CASE("LDS model divide into thread groups", "[lds-bank-model]")
    {
        std::vector<size_t> addresses       = {0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44};
        uint                threadsPerClock = 4;

        auto groups = divideIntoThreadGroups(addresses, threadsPerClock);

        // 3 groups of 4 addresses each
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

    TEST_CASE("LDS model compute thread group bank mappings", "[lds-bank-model]")
    {
        RuntimeLDSInstruction instr;
        instr.memoryOp = MemoryOpLDS{LdsDirection::Read};
        instr.dwords   = 4;

        for(int i = 0; i < 64; ++i) // 64 threads
        {
            // No bank conflicts should occur
            instr.baseAddresses.push_back(i * 4 * instr.dwords);
        }

        auto mappings = computeThreadGroupBankMappings(instr, GPUArchitectureGFX::GFX950);

        CHECK(mappings.size() == 8); // 64 threads / 8 threads per clock = 8 groups

        for(size_t i = 0; i < mappings.size(); ++i)
        {
            CHECK(mappings[i].size() == 32); // 8 threads * 4 dwords each = 32 banks accessed

            for(const auto& [bank, count] : mappings[i])
            {
                CHECK(count == 1);
            }
        }
    }

    TEST_CASE("LDS model calculate total cycles from bank mappings", "[lds-bank-model]")
    {
        std::vector<std::map<uint, uint>> mappings = {
            {{0, 5}, {1, 2}, {2, 1}}, // Group 1: max 5 accesses
            {{3, 1}, {4, 1}, {5, 1}}, // Group 2: max 1 access (no conflicts)
            {{0, 3}, {1, 3}, {2, 3}} // Group 3: max 3 accesses
        };
        // 5 + 1 + 3 = 9 cycles
        CHECK(calculateTotalCyclesFromBankMappings(mappings) == 9);
    }

    TEST_CASE("LDS model get clock count", "[lds-bank-model]")
    {
        RuntimeLDSInstruction instr;
        instr.dwords        = 2;
        instr.baseAddresses = {0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120};

        const auto gfx = GPUArchitectureGFX::GFX942;

        const auto mappings   = computeThreadGroupBankMappings(instr, gfx);
        const auto baseCycles = calculateTotalCyclesFromBankMappings(mappings);

        SECTION("Read")
        {
            instr.memoryOp    = MemoryOpLDS{LdsDirection::Read};
            const auto cycles = getInstructionDataCycles(instr, gfx);
            CHECK(cycles == baseCycles);
        }

        SECTION("Write")
        {
            instr.memoryOp    = MemoryOpLDS{LdsDirection::Write};
            const auto cycles = getInstructionDataCycles(instr, gfx);
            CHECK(cycles == baseCycles);
        }
    }

    TEST_CASE("LDS model get instruction issue cycles", "[lds-bank-model]")
    {
        SECTION("Read operations")
        {
            auto ldsRead = MemoryOpLDS{LdsDirection::Read};

            // For reads, always 4 cycles (address transfer only)
            for(uint dwords = 1; dwords <= 4; ++dwords)
            {
                CHECK(getInstructionIssueCycles(ldsRead, dwords) == 4);
            }
        }

        SECTION("Write operations")
        {
            auto ldsWrite = MemoryOpLDS{LdsDirection::Write};

            // For writes, 4 cycles (address) + 4 cycles per dword (data)
            for(uint dwords = 1; dwords <= 4; ++dwords)
            {
                CHECK(getInstructionIssueCycles(ldsWrite, dwords) == 4 + 4 * dwords);
            }
        }
    }

    TEST_CASE("LDS model sample addresses", "[lds-bank-model]")
    {
        // Sample taken from a GEMM kernel through rocgdb
        // Note: stride is not always 64 even though the first few appear to be so
        std::vector<size_t> addresses
            = {0,  64,  128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960,
               16, 80,  144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976,
               32, 96,  160, 224, 288, 352, 416, 480, 544, 608, 672, 736, 800, 864, 928, 992,
               48, 112, 176, 240, 304, 368, 432, 496, 560, 624, 688, 752, 816, 880, 944, 1008};

        RuntimeLDSInstruction instr = {
            .memoryOp = MemoryOpLDS{LdsDirection::Read}, .dwords = 4, .baseAddresses = addresses};

        auto expectedCycles = getInstructionDataCycles(instr, GPUArchitectureGFX::GFX950);
        CHECK(expectedCycles == 16); // Compared to same kernel through rocprofv3
    }

    TEST_CASE("LDS model data cycle predictions", "[lds-bank-model]")
    {
        SECTION("ds_read_b32 stride 1")
        {
            const int dwords        = 1;
            const int stride        = 1;
            const int workgroupSize = 64;

            CHECK(getInstructionDataCycles(
                      {.memoryOp      = MemoryOpLDS{LdsDirection::Read},
                       .dwords        = dwords,
                       .baseAddresses = generateLDSAddresses(workgroupSize, stride, dwords)},
                      GPUArchitectureGFX::GFX950)
                  == 4);
        }

        SECTION("ds_read_b32 stride 2")
        {
            const int dwords        = 1;
            const int stride        = 2;
            const int workgroupSize = 64;

            CHECK(getInstructionDataCycles(
                      {.memoryOp      = MemoryOpLDS{LdsDirection::Read},
                       .dwords        = dwords,
                       .baseAddresses = generateLDSAddresses(workgroupSize, stride, dwords)},
                      GPUArchitectureGFX::GFX950)
                  == 4);
        }
    }

    TEST_CASE("hasNonOverlappingBankAccess", "[lds-bank-model]")
    {
        SECTION("ds_read_b64 stride 1 wgs 256")
        {
            const int             dwords = 2;
            RuntimeLDSInstruction instr;
            instr.memoryOp.direction = LdsDirection::Read;
            instr.dwords             = dwords;
            instr.baseAddresses      = generateLDSAddresses(256, 1, dwords);

            CHECK(hasNonOverlappingBankAccess(instr, GPUArchitectureGFX::GFX950));
        }

        SECTION("ds_read_b64 stride 2 wgs 256")
        {
            const int             dwords = 2;
            RuntimeLDSInstruction instr;
            instr.memoryOp.direction = LdsDirection::Read;
            instr.dwords             = dwords;
            instr.baseAddresses      = generateLDSAddresses(256, 2, dwords);

            CHECK_FALSE(hasNonOverlappingBankAccess(instr, GPUArchitectureGFX::GFX950));
        }
    }
}
