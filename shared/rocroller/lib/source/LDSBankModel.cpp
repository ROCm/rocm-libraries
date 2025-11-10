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

#include <algorithm>
#include <sstream>
#include <unordered_map>

#include <fmt/format.h>

#include <rocRoller/Expression.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Scheduling/LDSBankModel.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller::Scheduling::LDSBankModel
{

    uint getThreadsPerClock(const MemoryOpLDS& memoryOp, uint dwords, GPUArchitectureGFX gfx)
    {
        // Assumes aligned accesses (e.g. b128 is 16-byte aligned)
        // In future, when linked to codegen, update interface to check alignment of allocation
        if(gfx == GPUArchitectureGFX::GFX950 && memoryOp.direction == LdsDirection::Read)
        {
            switch(dwords)
            {
            case 1:
                return 16;
            case 2:
                return 16;
            case 3:
                // ds_read_b96 on gfx950 retains the throughput of gfx942
                return 4;
            case 4:
                return 8;
            }
        }
        else
        {
            switch(dwords)
            {
            case 1:
                return 16;
            case 2:
                return 8;
            case 3:
            case 4:
                return 4;
            }
        }
        Throw<FatalError>("Unsupported dword count: ", dwords);
    }

    uint getNumLDSBanks(GPUArchitectureGFX gfx, const MemoryOpLDS& memoryOp, uint dwords)
    {
        // Non ds_read_b128 and ds_read_b64 on gfx950 act as-if there are still only 32 banks for conflict resolution
        if(gfx == GPUArchitectureGFX::GFX950 && memoryOp.direction == LdsDirection::Read
           && (dwords == 2 || dwords == 4))
        {
            return 64;
        }
        return 32;
    }

    std::vector<std::vector<size_t>> divideIntoThreadGroups(const std::vector<size_t>& addresses,
                                                            uint threadsPerClock)
    {
        AssertFatal(addresses.size() % threadsPerClock == 0,
                    fmt::format("Number of addresses {} is not a multiple of threads per clock {}",
                                addresses.size(),
                                threadsPerClock));

        std::vector<std::vector<size_t>> threadGroups;

        for(size_t groupStart = 0; groupStart < addresses.size(); groupStart += threadsPerClock)
        {
            size_t              groupEnd = std::min(groupStart + threadsPerClock, addresses.size());
            std::vector<size_t> group(addresses.begin() + groupStart, addresses.begin() + groupEnd);
            threadGroups.push_back(group);
        }

        return threadGroups;
    }

    std::map<uint, uint> createBankToAddressCounts(const std::vector<size_t>& baseAddresses,
                                                   uint                       dwords,
                                                   GPUArchitectureGFX         gfx,
                                                   const MemoryOpLDS&         memoryOp)
    {
        std::map<uint, uint> bankToAddressCounts;
        uint                 numBanks = getNumLDSBanks(gfx, memoryOp, dwords);

        for(size_t i = 0; i < baseAddresses.size(); ++i)
        {
            AssertFatal(
                baseAddresses[i] % 4 == 0, "Base address is not dword aligned ", baseAddresses[i]);
            uint baseAddr = baseAddresses[i] / 4; // in dwords

            // Note: using dword as the unit here
            for(uint offset = 0; offset < dwords; offset += 1)
            {
                uint currentAddr = baseAddr + offset;
                uint bankIndex   = currentAddr % numBanks;
                bankToAddressCounts[bankIndex]++;
            }
        }

        return bankToAddressCounts;
    }

    uint calculateBankConflictCycles(const std::map<uint, uint>& bankToAddressCounts)
    {
        if(bankToAddressCounts.empty())
        {
            return 0;
        }

        // The number of clock cycles is determined by the bank accessed by the most addresses,
        // since only one address per bank can be serviced per cycle
        uint maxAddressesPerBank = 0;
        for(const auto& [bankIndex, count] : bankToAddressCounts)
        {
            maxAddressesPerBank = std::max(maxAddressesPerBank, count);
        }

        return maxAddressesPerBank;
    }

    std::vector<std::map<uint, uint>>
        computeThreadGroupBankMappings(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx)
    {
        std::vector<std::map<uint, uint>> threadGroupBankMappings;

        const auto threadGroupAddresses = divideIntoThreadGroups(
            instr.baseAddresses, getThreadsPerClock(instr.memoryOp, instr.dwords, gfx));

        for(const auto& groupAddresses : threadGroupAddresses)
        {
            threadGroupBankMappings.push_back(
                createBankToAddressCounts(groupAddresses, instr.dwords, gfx, instr.memoryOp));
        }

        return threadGroupBankMappings;
    }

    uint calculateTotalCyclesFromBankMappings(
        const std::vector<std::map<uint, uint>>& threadGroupBankMappings)
    {
        uint cycles = 0;
        for(const auto& bankMapping : threadGroupBankMappings)
        {
            cycles += calculateBankConflictCycles(bankMapping);
        }
        return cycles;
    }

    uint getInstructionDataCycles(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx)
    {
        AssertFatal(rocRoller::toString(gfx).starts_with("gfx9"),
                    "Unsupported GPU architecture: {}",
                    rocRoller::toString(gfx));

        const auto threadGroupBankMappings = computeThreadGroupBankMappings(instr, gfx);

        for(const auto& mapping : threadGroupBankMappings)
        {
            for(const auto& [bankIndex, count] : mapping)
            {
                Log::trace("Bank {} accessed {} times", bankIndex, count);
            }
        }

        uint cycles = calculateTotalCyclesFromBankMappings(threadGroupBankMappings);

        return cycles;
    }

    uint getInstructionIssueCycles(const MemoryOpLDS& memoryOp, uint dwords)
    {
        // 4 cycles for addresses, additional cycles for write
        uint cycles = 4;
        if(memoryOp.direction == LdsDirection::Write)
        {
            cycles += 4 * dwords;
        }
        return cycles;
    }

    uint getInstructionCycles(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx)
    {
        uint issueCycles = getInstructionIssueCycles(instr.memoryOp, instr.dwords);
        uint dataCycles  = getInstructionDataCycles(instr, gfx);
        return std::max(issueCycles, dataCycles);
    }

    std::string Summary::toString() const
    {
        std::stringstream ss;
        for(auto const& [tag, access] : this->tagToAccess)
        {
            auto const& [ldsTag, accessedBanks, banksToWorkitems] = access;
            ss << fmt::format("Operation tag {} accesses LDS {}:\n", tag, ldsTag);
            for(auto const& [bankIndex, workitemsAccessed, imbalanced] : accessedBanks)
            {
                ss << fmt::format("  Bank {}: {} workitems {}\n",
                                  bankIndex,
                                  workitemsAccessed,
                                  imbalanced ? "(imbalanced)" : "");
            }
        }
        ss << fmt::format("  Imbalanced tags: {}\n", this->imbalancedTags);
        return ss.str();
    }

    std::pair<std::string, uint> stringifyInstructionAnalysis(const RuntimeLDSInstruction& instr,
                                                              GPUArchitectureGFX           gfx)
    {
        std::stringstream ss;

        std::string instructionName;
        if(instr.memoryOp.direction == LdsDirection::Read)
        {
            instructionName = fmt::format("ds_read_b{}", instr.dwords * 32);
        }
        else
        {
            instructionName = fmt::format("ds_write_b{}", instr.dwords * 32);
        }
        ss << fmt::format("  Instruction: {}\n", instructionName);

        // Follows getClockCount (checked against it later for consistency)
        uint cycles = 0;
        {
            const auto threadsPerClock = getThreadsPerClock(instr.memoryOp, instr.dwords, gfx);
            uint       i               = 0;
            for(const auto& groupAddresses :
                divideIntoThreadGroups(instr.baseAddresses, threadsPerClock))
            {
                const auto bankToAddressCounts
                    = createBankToAddressCounts(groupAddresses, instr.dwords, gfx, instr.memoryOp);
                uint groupCycles = calculateBankConflictCycles(bankToAddressCounts);
                ss << fmt::format("    Group {}: threads {}-{}\n",
                                  i,
                                  i * threadsPerClock,
                                  (i + 1) * threadsPerClock - 1);

                uint maxCount = 0;
                for(const auto& [bankIndex, count] : bankToAddressCounts)
                {
                    maxCount = std::max(maxCount, count);
                }
                std::vector<uint> maxBanks;
                for(const auto& [bankIndex, count] : bankToAddressCounts)
                {
                    if(count == maxCount)
                    {
                        maxBanks.push_back(bankIndex);
                    }
                }

                if(!maxBanks.empty())
                {
                    ss << "      Max bank contention: " << maxCount
                       << " addresses/bank for bank(s) ";
                    rocRoller::streamJoin(ss, maxBanks, ", ");
                    ss << "\n";
                }
                ss << fmt::format("      Group cycles: {}\n", groupCycles);
                cycles += groupCycles;
                i++;
            }
        }

        const auto instructionTotalClocks = getInstructionDataCycles(instr, gfx);

        AssertFatal(
            cycles == instructionTotalClocks,
            "Cycle count mismatch between stringify and getInstructionDataCycles, {} and {}",
            cycles,
            instructionTotalClocks);

        ss << fmt::format("    Instruction cycles: {}\n", instructionTotalClocks);

        return std::make_pair(ss.str(), instructionTotalClocks);
    }

    std::string OperationsAnalysis::toString() const
    {
        std::stringstream ss;

        ss << rocRoller::toString(gfx) << "\n";

        for(const auto& [operationTag, opAccesses] : tagToAccess)
        {
            ss << fmt::format("Operation Tag: {}, LDS Tag: {}\n", operationTag, opAccesses.ldsTag);

            uint operationTotalClocks = 0;

            for(const auto& instr : opAccesses.instructions)
            {
                auto [instrStr, instructionClocks] = stringifyInstructionAnalysis(instr, gfx);
                ss << instrStr;
                operationTotalClocks += instructionClocks;
            }
            ss << fmt::format("  Operation cycles: {}\n", operationTotalClocks);
            ss << "\n";
        }

        return ss.str();
    }

    std::ostream& operator<<(std::ostream& stream, OperationsAnalysis const& operationAnalysis)
    {
        return stream << operationAnalysis.toString();
    }

    std::ostream& operator<<(std::ostream& stream, Summary const& summary)
    {
        return stream << summary.toString();
    }
}