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
#include <fmt/format.h>
#include <rocRoller/Expression.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/LDSBankModel.hpp>
#include <rocRoller/KernelGraph/Transforms/MemoryTracer.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Utils.hpp>
#include <sstream>
#include <unordered_map>

namespace rocRoller::KernelGraph::MemoryTracer
{
    bool isLDSOperation(const MemoryEventExpression& event)
    {
        return std::holds_alternative<MemoryOpLDS>(event.memoryOp);
    }

    OperationLevelAnalysis::OperationLevelAnalysis(uint entryWidthInBytes,
                                                   uint numBanks,
                                                   uint numEntriesPerBank)
        : m_entryWidthInBytes(entryWidthInBytes)
        , m_numBanks(numBanks)
        , m_numEntriesPerBank(numEntriesPerBank)
    {
    }

    bool OperationLevelAnalysis::filter(MemoryEventExpression event)
    {
        return isLDSOperation(event);
    }

    void OperationLevelAnalysis::simulate(MemoryEventSimulated event)
    {
        auto ldsOp = std::get_if<MemoryOpLDS>(&event.memoryOp);
        if(!ldsOp)
            return;

        std::set<uint> banksAccessed;
        for(uint offset = 0; offset < event.bytesRequested; offset += m_entryWidthInBytes)
        {
            uint currentAddr = event.byteOffset + offset;
            uint bankIndex   = (currentAddr / m_entryWidthInBytes) % m_numBanks;
            banksAccessed.insert(bankIndex);
        }
        for(uint bankIndex : banksAccessed)
        {
            LDSBankAccess bankAccess;
            bankAccess.operationTag = event.operationTag;
            bankAccess.ldsTag
                = ldsOp->direction == Direction::Load ? event.sourceTag : event.destinationTag;
            bankAccess.direction = ldsOp->direction;
            bankAccess.workitem  = event.workItem;
            bankAccess.bankIndex = bankIndex;
            m_bankAccesses[event.operationTag].push_back(bankAccess);
        }
    }

    Summary OperationLevelAnalysis::summary() const
    {
        Summary summary;

        for(const auto& [operationTag, bankAccesses] : m_bankAccesses)
        {
            Summary::Access access;

            if(!bankAccesses.empty())
            {
                access.ldsTag = bankAccesses[0].ldsTag;
            }

            std::map<uint, std::vector<int>> bankToWorkitems;
            for(const auto& bankAccess : bankAccesses)
            {
                bankToWorkitems[bankAccess.bankIndex].push_back(bankAccess.workitem);
            }

            // Initialize banksToWorkitems vector with 32 banks
            access.banksToWorkitems.resize(m_numBanks);

            // Populate accessedBanks and banksToWorkitems
            for(const auto& [bankIndex, workitems] : bankToWorkitems)
            {
                Summary::Banks bank;
                bank.bankIndex         = bankIndex;
                bank.workitemsAccessed = workitems.size();

                // Check if imbalanced - comparing against average access per bank
                size_t totalAccesses          = bankAccesses.size();
                size_t averageAccessesPerBank = (totalAccesses + m_numBanks - 1) / m_numBanks;
                bank.imbalanced               = bank.workitemsAccessed > averageAccessesPerBank * 2;

                access.accessedBanks.push_back(bank);

                // Copy workitems to the banksToWorkitems vector
                access.banksToWorkitems[bankIndex] = workitems;

                // Mark operation as imbalanced if any bank is imbalanced
                if(bank.imbalanced)
                {
                    summary.imbalancedTags.insert(operationTag);
                }
            }

            summary.accesses[operationTag] = access;
        }

        return summary;
    }

    LDSBankModel::LDSBankModel(uint entryWidthInBytes, uint numBanks, uint numEntriesPerBank) {}

    bool LDSBankModel::filter(MemoryEventExpression event)
    {
        return isLDSOperation(event);
    }

    void LDSBankModel::simulate(MemoryEventSimulated event)
    {
        auto ldsOp = std::get_if<MemoryOpLDS>(&event.memoryOp);
        if(!ldsOp)
            return;

        auto& opAccesses        = m_tagToOperationAccesses[event.operationTag];
        opAccesses.operationTag = event.operationTag;
        opAccesses.ldsTag
            = ldsOp->direction == Direction::Load ? event.sourceTag : event.destinationTag;

        uint instructionDwords = (event.bytesRequested + 3) / 4; // Round up to dwords

        RuntimeLDSInstruction* targetInstruction = nullptr;
        for(auto& instr : opAccesses.instructions)
        {
            if(instr.dwords == instructionDwords && instr.memoryOp.direction == ldsOp->direction)
            {
                targetInstruction = &instr;
                break;
            }
        }

        if(!targetInstruction)
        {
            RuntimeLDSInstruction newInstr;
            newInstr.memoryOp = *ldsOp;
            newInstr.dwords   = instructionDwords;
            opAccesses.instructions.push_back(newInstr);
            targetInstruction = &opAccesses.instructions.back();
        }

        targetInstruction->baseAddresses.push_back(event.byteOffset);
    }

    Summary LDSBankModel::summary() const
    {
        Throw<FatalError>("LDSBankModel::summary() is no longer supported. Use "
                          "OperationLevelAnalysis::summary() instead.");
    }

    std::string LDSBankModel::toString() const
    {
        return "LDS Bank Model (instruction-level analysis)";
    }

    uint LDSBankModel::getThreadsPerClock(const MemoryOpLDS& memoryOp,
                                          uint               dwords,
                                          GPUArchitectureGFX gfx)
    {
        // TODO: These numbers assume aligned accesses
        // (e.g. for 128-bit bottom 4 bits of base address are zero)
        // At codegen, can check allocation's alignment
        if(gfx == GPUArchitectureGFX::GFX950 && memoryOp.direction == Direction::Load)
        {
            switch(dwords)
            {
            case 1:
                return 32;
            case 2:
                return 32;
            case 3:
                // ds_read_b96 on gfx950 retains the same peak throughput as gfx942
                // and is thus slower than ds_read_b128
                return 8;
            case 4:
                return 16;
            }
        }
        else
        {
            switch(dwords)
            {
            case 1:
                return 32;
            case 2:
                return 16;
            case 3:
            case 4:
                return 8;
            }
        }

        Throw<FatalError>("Unsupported dword count: ", dwords);
    }

    uint LDSBankModel::getNumLDSBanks(GPUArchitectureGFX gfx)
    {
        return (gfx == GPUArchitectureGFX::GFX950) ? 64 : 32;
    }

    std::vector<std::vector<uint32_t>>
        LDSBankModel::divideIntoThreadGroups(const std::vector<uint32_t>& addresses,
                                             uint                         threadsPerClock)
    {
        AssertFatal(addresses.size() % threadsPerClock == 0,
                    "Number of addresses {} is not a multiple of threads per clock {}",
                    addresses.size(),
                    threadsPerClock);

        std::vector<std::vector<uint32_t>> threadGroups;

        for(size_t groupStart = 0; groupStart < addresses.size(); groupStart += threadsPerClock)
        {
            size_t groupEnd = std::min(groupStart + threadsPerClock, addresses.size());
            std::vector<uint32_t> group(addresses.begin() + groupStart,
                                        addresses.begin() + groupEnd);
            threadGroups.push_back(group);
        }

        return threadGroups;
    }

    std::map<uint, uint> LDSBankModel::createBankToAddressCounts(
        const std::vector<uint32_t>& baseAddresses, uint dwords, GPUArchitectureGFX gfx)
    {
        std::map<uint, uint> bankToAddressCounts;
        uint                 numBanks = getNumLDSBanks(gfx);

        for(size_t i = 0; i < baseAddresses.size(); ++i)
        {
            AssertFatal(baseAddresses[i] % 4 == 0,
                        "Base address {} is not dword aligned",
                        baseAddresses[i]);
            uint baseAddr = baseAddresses[i] / 4; // in dwords

            // Note: address arithmetic is operating on dword (instead of byte) units
            for(uint offset = 0; offset < dwords; offset += 1)
            {
                uint currentAddr = baseAddr + offset;
                uint bankIndex   = currentAddr % numBanks;
                bankToAddressCounts[bankIndex]++;
            }
        }

        return bankToAddressCounts;
    }

    uint LDSBankModel::calculateBankConflictCycles(const std::map<uint, uint>& bankToAddressCounts)
    {
        if(bankToAddressCounts.empty())
        {
            return 0;
        }

        // The number of clock cycles is determined by the bank with the maximum
        // number of addresses, since only one address per bank can be serviced per cycle
        uint maxAddressesPerBank = 0;
        for(const auto& [bankIndex, count] : bankToAddressCounts)
        {
            maxAddressesPerBank = std::max(maxAddressesPerBank, count);
        }

        return maxAddressesPerBank;
    }

    std::string LDSBankModel::instructionDetailedAnalysis(const RuntimeLDSInstruction& instr,
                                                          GPUArchitectureGFX           gfx,
                                                          uint&                        totalCycles)
    {
        std::stringstream ss;

        // Generate instruction name
        std::string instructionName;
        if(instr.memoryOp.direction == Direction::Load)
        {
            instructionName = fmt::format("ds_read_b{}", instr.dwords * 32);
        }
        else
        {
            instructionName = fmt::format("ds_write_b{}", instr.dwords * 32);
        }
        ss << fmt::format("  Instruction: {}\n", instructionName);

        // Follows getClockCount
        uint cycles = 0;
        {
            const auto threadsPerClock
                = LDSBankModel::getThreadsPerClock(instr.memoryOp, instr.dwords, gfx);
            uint i = 0;
            for(const auto& groupAddresses :
                LDSBankModel::divideIntoThreadGroups(instr.baseAddresses, threadsPerClock))
            {
                const auto bankToAddressCounts
                    = LDSBankModel::createBankToAddressCounts(groupAddresses, instr.dwords, gfx);
                uint groupCycles = LDSBankModel::calculateBankConflictCycles(bankToAddressCounts);
                ss << fmt::format("    Group {}: threads {}-{}\n",
                                  i,
                                  i * threadsPerClock,
                                  (i + 1) * threadsPerClock - 1);
                // Find banks with maximum address counts
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
        cycles += 4;

        const auto instructionTotalClocks = LDSBankModel::getClockCount(instr, gfx);

        AssertFatal(cycles == instructionTotalClocks, "Cycle count mismatch");
        ss << fmt::format("    Instruction cycles: {}\n", instructionTotalClocks);

        totalCycles = instructionTotalClocks;
        return ss.str();
    }

    uint LDSBankModel::getClockCount(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx)
    {
        uint cycles = 0;
        for(const auto& groupAddresses : divideIntoThreadGroups(
                instr.baseAddresses, getThreadsPerClock(instr.memoryOp, instr.dwords, gfx)))
        {
            cycles += calculateBankConflictCycles(
                createBankToAddressCounts(groupAddresses, instr.dwords, gfx));
        }
        return cycles + 4; // address transfer
    }

    OperationAnalysis LDSBankModel::doOperationAnalysis(GPUArchitectureGFX gfx) const
    {
        OperationAnalysis detailed;
        detailed.gfx = gfx;

        for(const auto& [operationTag, sourceOpAccesses] : m_tagToOperationAccesses)
        {
            OperationAccesses opAccesses;
            opAccesses.operationTag = sourceOpAccesses.operationTag;
            opAccesses.ldsTag       = sourceOpAccesses.ldsTag;

            for(const auto& sourceInstr : sourceOpAccesses.instructions)
            {
                RuntimeLDSInstruction instr;
                instr.memoryOp      = sourceInstr.memoryOp;
                instr.dwords        = sourceInstr.dwords;
                instr.baseAddresses = sourceInstr.baseAddresses;

                opAccesses.instructions.push_back(instr);
            }

            detailed.accesses[operationTag] = opAccesses;
        }

        return detailed;
    }

    std::ostream& operator<<(std::ostream& stream, LDSBankModel const& ldsBankModel)
    {
        return stream << ldsBankModel.toString();
    }

    std::string Summary::toString() const
    {
        std::stringstream ss;
        for(auto const& [tag, access] : this->accesses)
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
            if constexpr(echoBanks)
            {
                for(size_t bankIndex = 0; bankIndex < banksToWorkitems.size(); ++bankIndex)
                {
                    ss << fmt::format("  Bank {:2d}: ", bankIndex);
                    for(auto workitem : banksToWorkitems[bankIndex])
                    {
                        ss << fmt::format("{:2d} ", workitem);
                    }
                    ss << '\n';
                }
            }
        }
        ss << fmt::format("  Imbalanced tags: {}\n", this->imbalancedTags);
        return ss.str();
    }

    std::string OperationAnalysis::toString() const
    {
        std::stringstream ss;

        ss << rocRoller::toString(gfx) << "\n";

        for(const auto& [operationTag, opAccesses] : accesses)
        {
            ss << fmt::format("Operation Tag: {}, LDS Tag: {}\n", operationTag, opAccesses.ldsTag);

            uint operationTotalClocks = 0;

            for(const auto& instr : opAccesses.instructions)
            {
                uint instructionClocks = 0;
                ss << LDSBankModel::instructionDetailedAnalysis(instr, gfx, instructionClocks);
                operationTotalClocks += instructionClocks;
            }
            ss << fmt::format("  Operation cycles: {}\n", operationTotalClocks);
            ss << "\n";
        }

        return ss.str();
    }

    std::ostream& operator<<(std::ostream& stream, OperationAnalysis const& operationAnalysis)
    {
        return stream << operationAnalysis.toString();
    }

    std::ostream& operator<<(std::ostream& stream, Summary const& summary)
    {
        return stream << summary.toString();
    }

    Summary memoryTrace(KernelGraph const& original, KernelInvocation const& invocation)
    {
        Log::info("MemoryTracer::memoryTrace()");

        auto graph  = original;
        auto tracer = MemoryTracer(graph);
        tracer.trace();

        Log::info("MemoryTracer::OperationLevelAnalysis()");
        // 64KiB bank model: 4 bytes per bank entry, 32 banks, 512 entries per bank
        auto model = OperationLevelAnalysis(4, 32, 512);

        // For LDS, just simulate using 1 workgroup
        auto workgroups            = 1;
        auto workitemsPerWorkgroup = product(invocation.workgroupSize);
        tracer.simulateLaunch(model, workgroups, workitemsPerWorkgroup);

        return model.summary();
    }
}
