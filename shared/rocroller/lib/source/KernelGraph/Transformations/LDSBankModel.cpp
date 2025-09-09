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
#include <sstream>
#include <unordered_map>

namespace rocRoller::KernelGraph::MemoryTracer
{
    LDSBankModel::LDSBankModel(uint entryWidthInBytes, uint numBanks, uint numEntriesPerBank)
        : m_entryWidthInBytes(entryWidthInBytes)
        , m_numBanks(numBanks)
        , m_numEntriesPerBank(numEntriesPerBank)
    {
    }

    bool LDSBankModel::filter(MemoryEventExpression event)
    {
        // Check if the memory operation is for LDS
        return std::holds_alternative<MemoryOpLDS>(event.memoryOp);
    }

    void LDSBankModel::simulate(MemoryEventSimulated event)
    {
        // Only process LDS operations
        auto ldsOp = std::get_if<MemoryOpLDS>(&event.memoryOp);
        if(!ldsOp)
            return;

        // Get or create operation accesses for this operationTag
        auto& opAccesses        = m_hierarchicalAccesses[event.operationTag];
        opAccesses.operationTag = event.operationTag;
        opAccesses.ldsTag       = event.sourceTag; // Assuming sourceTag is the LDS tag

        // Break up bytesRequested into instructions
        uint remainingBytes = event.bytesRequested;
        uint currentOffset  = event.byteOffset;

        while(remainingBytes > 0)
        {
            // Determine instruction size (try to maximize width)
            uint instructionBytes;
            uint instructionDwords;

            if(remainingBytes >= 16)
            {
                instructionBytes  = 16;
                instructionDwords = 4;
            }
            else if(remainingBytes >= 8)
            {
                instructionBytes  = 8;
                instructionDwords = 2;
            }
            else if(remainingBytes >= 4)
            {
                instructionBytes  = 4;
                instructionDwords = 1;
            }
            else
            {
                // Less than 4 bytes - round up to 4 bytes (1 dword)
                instructionBytes  = 4;
                instructionDwords = 1;
            }

            // Find or create instruction for this dword size
            InstructionAccesses* targetInstruction = nullptr;
            for(auto& instr : opAccesses.instructions)
            {
                if(instr.dwords == instructionDwords
                   && std::holds_alternative<MemoryOpLDS>(instr.memoryOp))
                {
                    auto& instrLdsOp = std::get<MemoryOpLDS>(instr.memoryOp);
                    if(instrLdsOp.direction == ldsOp->direction)
                    {
                        targetInstruction = &instr;
                        break;
                    }
                }
            }

            if(!targetInstruction)
            {
                // Create new instruction
                InstructionAccesses newInstr;
                newInstr.memoryOp = event.memoryOp;
                newInstr.dwords   = instructionDwords;
                opAccesses.instructions.push_back(newInstr);
                targetInstruction = &opAccesses.instructions.back();
            }

            // Calculate bank index for this access
            uint bankIndex = (currentOffset / m_entryWidthInBytes) % m_numBanks;

            // Create thread access
            ThreadAccess threadAccess;
            threadAccess.workitem  = event.workItem;
            threadAccess.address   = currentOffset;
            threadAccess.bankIndex = bankIndex;

            // Add thread to the instruction
            // For now, we'll add each thread as a separate group
            // Later, we'll reorganize them based on threads per clock
            ThreadGroup threadGroup;
            threadGroup.groupIndex = event.workItem; // Temporary, will be reorganized
            threadGroup.threads.push_back(threadAccess);
            targetInstruction->threadGroups.push_back(threadGroup);

            // Also update legacy m_bankAccesses for backward compatibility
            LDSBankAccess bankAccess;
            bankAccess.operationTag = event.operationTag;
            bankAccess.ldsTag       = event.sourceTag;
            bankAccess.direction    = ldsOp->direction;
            bankAccess.workitem     = event.workItem;
            bankAccess.bankIndex    = bankIndex;
            m_bankAccesses[event.operationTag].push_back(bankAccess);

            // Move to next instruction
            remainingBytes -= instructionBytes;
            currentOffset += instructionBytes;
        }
    }

    Summary LDSBankModel::summary() const
    {
        Summary summary;

        // Process each operation
        for(const auto& [operationTag, bankAccesses] : m_bankAccesses)
        {
            Summary::Access access;

            // Find the LDS tag from the first access
            if(!bankAccesses.empty())
            {
                access.ldsTag = bankAccesses[0].ldsTag;
            }

            // Create a map from bank index to workitems
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

    std::string LDSBankModel::toString() const
    {
        std::stringstream ss;
        ss << "LDS Bank Model: "
           << this->m_entryWidthInBytes * this->m_numEntriesPerBank * this->m_numBanks << " bytes, "
           << this->m_entryWidthInBytes << "byte bank width, " << this->m_numBanks << " banks"
           << std::endl;
        return ss.str();
    }

    uint LDSBankModel::getThreadsPerClock(const MemoryOpLDS& memoryOp,
                                          uint               dwords,
                                          GPUArchitectureGFX gfx)
    {
        // TODO: These numbers assume aligned accesses
        // (e.g. for 128-bit bottom 4 bits of base address are zero)
        // Is there a way to check? Given kernel graph provides relative addresses
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

    std::map<uint, std::vector<uint32_t>> LDSBankModel::makeBankMapping(
        const std::vector<uint32_t>& addresses, uint entryWidthInBytes, uint numBanks)
    {
        std::map<uint, std::vector<uint32_t>> bankMapping;
        for(auto address : addresses)
        {
            auto bankIndex = (address / entryWidthInBytes) % numBanks;
            bankMapping[bankIndex].push_back(address);
        }
        return bankMapping;
    }

    uint LDSBankModel::calculateBankConflicts(
        const std::map<uint, std::vector<uint32_t>>& bankMapping)
    {
        if(bankMapping.empty())
            return 0;

        uint maxConflicts = 0;
        for(const auto& [bank, addresses] : bankMapping)
        {
            maxConflicts = std::max(maxConflicts, static_cast<uint>(addresses.size()));
        }
        return maxConflicts;
    }

    DetailedSummary LDSBankModel::detailedSummary(GPUArchitectureGFX gfx) const
    {
        DetailedSummary detailed;
        detailed.gfx = gfx;

        // Copy and reorganize hierarchical accesses
        for(const auto& [operationTag, sourceOpAccesses] : m_hierarchicalAccesses)
        {
            OperationAccesses opAccesses;
            opAccesses.operationTag = sourceOpAccesses.operationTag;
            opAccesses.ldsTag       = sourceOpAccesses.ldsTag;

            // Process each instruction type
            for(const auto& sourceInstr : sourceOpAccesses.instructions)
            {
                InstructionAccesses instr;
                instr.memoryOp = sourceInstr.memoryOp;
                instr.dwords   = sourceInstr.dwords;

                // Collect all threads from source thread groups
                std::vector<ThreadAccess> allThreads;
                for(const auto& threadGroup : sourceInstr.threadGroups)
                {
                    for(const auto& thread : threadGroup.threads)
                    {
                        allThreads.push_back(thread);
                    }
                }

                // Sort threads by workitem ID for consistent grouping
                std::sort(allThreads.begin(),
                          allThreads.end(),
                          [](const ThreadAccess& a, const ThreadAccess& b) {
                              return a.workitem < b.workitem;
                          });

                // Get threads per clock for this instruction
                const auto& ldsOp           = std::get<MemoryOpLDS>(sourceInstr.memoryOp);
                uint        threadsPerClock = getThreadsPerClock(ldsOp, sourceInstr.dwords, gfx);

                // Group threads based on threads per clock
                uint groupIndex = 0;
                for(size_t i = 0; i < allThreads.size(); i += threadsPerClock)
                {
                    ThreadGroup threadGroup;
                    threadGroup.groupIndex = groupIndex++;

                    // Add threads to this group (up to threadsPerClock)
                    for(size_t j = i; j < i + threadsPerClock && j < allThreads.size(); ++j)
                    {
                        threadGroup.threads.push_back(allThreads[j]);
                    }

                    instr.threadGroups.push_back(threadGroup);
                }

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

    std::string DetailedSummary::toString() const
    {
        std::stringstream ss;

        ss << rocRoller::toString(gfx) << "\n";

        for(const auto& [operationTag, opAccesses] : accesses)
        {
            ss << fmt::format("Operation Tag: {}, LDS Tag: {}\n", operationTag, opAccesses.ldsTag);

            uint operationTotalClocks = 0; // Track total clocks for the entire operation

            for(const auto& instr : opAccesses.instructions)
            {
                // Get instruction details
                std::string instructionName = "unknown";
                if(auto ldsOp = std::get_if<MemoryOpLDS>(&instr.memoryOp))
                {
                    if(ldsOp->direction == Direction::Load)
                    {
                        switch(instr.dwords)
                        {
                        case 1:
                            instructionName = "ds_read_b32";
                            break;
                        case 2:
                            instructionName = "ds_read_b64";
                            break;
                        case 3:
                            instructionName = "ds_read_b96";
                            break;
                        case 4:
                            instructionName = "ds_read_b128";
                            break;
                        default:
                            instructionName = fmt::format("ds_read_{}_dwords", instr.dwords);
                            break;
                        }
                    }
                    else // Store
                    {
                        switch(instr.dwords)
                        {
                        case 1:
                            instructionName = "ds_write_b32";
                            break;
                        case 2:
                            instructionName = "ds_write_b64";
                            break;
                        case 3:
                            instructionName = "ds_write_b96";
                            break;
                        case 4:
                            instructionName = "ds_write_b128";
                            break;
                        default:
                            instructionName = fmt::format("ds_write_{}_dwords", instr.dwords);
                            break;
                        }
                    }
                }

                ss << fmt::format("  Instruction: {}\n", instructionName);

                uint instructionTotalClocks = 0; // Track total clocks for this instruction

                // Process each thread group
                for(const auto& threadGroup : instr.threadGroups)
                {
                    ss << fmt::format("    Thread Group {}: {} threads\n",
                                      threadGroup.groupIndex,
                                      threadGroup.threads.size());

                    // Create a mapping of bank index to list of (address, thread) pairs
                    // For multi-dword accesses, we need to track all banks touched
                    std::map<uint, std::vector<std::pair<uint32_t, uint>>> bankToAccessInfo;

                    // TODO: these should not be hardcoded
                    const uint entryWidthInBytes = 4;
                    const uint numBanks          = 64;

                    for(const auto& thread : threadGroup.threads)
                    {
                        // Calculate all banks this access touches
                        uint bytesPerAccess = instr.dwords * 4;
                        for(uint offset = 0; offset < bytesPerAccess; offset += entryWidthInBytes)
                        {
                            uint currentAddr = thread.address + offset;
                            uint bankIndex   = (currentAddr / entryWidthInBytes) % numBanks;
                            bankToAccessInfo[bankIndex].push_back(
                                std::make_pair(currentAddr, thread.workitem));
                        }
                    }

                    // Calculate the maximum bank conflicts
                    uint maxConflicts = 0;
                    for(const auto& [bank, accessInfo] : bankToAccessInfo)
                    {
                        maxConflicts = std::max(maxConflicts, static_cast<uint>(accessInfo.size()));
                    }

                    // Simulate clock cycles needed to resolve bank conflicts
                    uint                                                   clockCycle = 0;
                    std::map<uint, std::vector<std::pair<uint32_t, uint>>> remainingAccesses
                        = bankToAccessInfo;

                    while(!remainingAccesses.empty())
                    {
                        ss << fmt::format("      Clock Cycle {}:\n", clockCycle);

                        std::map<uint, std::pair<uint32_t, uint>> cycleAccesses;
                        std::vector<uint>                         banksToRemove;

                        // Process one access per bank for this clock cycle
                        for(auto& [bankIndex, accessList] : remainingAccesses)
                        {
                            if(!accessList.empty())
                            {
                                // Take the first access for this bank
                                auto [address, workitem] = accessList.front();
                                cycleAccesses[bankIndex] = std::make_pair(address, workitem);
                                accessList.erase(accessList.begin());

                                if(accessList.empty())
                                {
                                    banksToRemove.push_back(bankIndex);
                                }
                            }
                        }

                        // Remove banks with no remaining accesses
                        for(auto bank : banksToRemove)
                        {
                            remainingAccesses.erase(bank);
                        }

                        // Group addresses by work-item
                        std::map<uint, std::vector<uint32_t>> workitemToAddresses;
                        for(const auto& [bank, accessInfo] : cycleAccesses)
                        {
                            workitemToAddresses[accessInfo.second].push_back(accessInfo.first);
                        }

                        // Print work-item index: [addresses, ...]
                        for(const auto& [workitem, addresses] : workitemToAddresses)
                        {
                            ss << fmt::format("        workitem {}: [", workitem);
                            bool first = true;
                            for(uint32_t addr : addresses)
                            {
                                if(!first)
                                    ss << ", ";
                                ss << addr;
                                first = false;
                            }
                            ss << "]\n";
                        }

                        // Print banks used and unused
                        std::vector<uint> usedBanks;
                        std::vector<uint> unusedBanks;

                        for(uint bankIdx = 0; bankIdx < numBanks; ++bankIdx)
                        {
                            if(cycleAccesses.find(bankIdx) != cycleAccesses.end())
                            {
                                usedBanks.push_back(bankIdx);
                            }
                            else
                            {
                                unusedBanks.push_back(bankIdx);
                            }
                        }

                        ss << "        banks used: [";
                        for(size_t i = 0; i < usedBanks.size(); ++i)
                        {
                            ss << usedBanks[i];
                            if(i < usedBanks.size() - 1)
                                ss << ", ";
                        }
                        ss << "]\n";

                        ss << "        banks unused: [";
                        for(size_t i = 0; i < unusedBanks.size(); ++i)
                        {
                            ss << unusedBanks[i];
                            if(i < unusedBanks.size() - 1)
                                ss << ", ";
                        }
                        ss << "]\n";

                        clockCycle++;
                    }

                    ss << fmt::format("      Total clock cycles: {}\n", clockCycle);
                    instructionTotalClocks += clockCycle; // Add to instruction total
                }

                // Print instruction total after all thread groups
                ss << fmt::format(
                    "    Total clock cycles for {}: {}\n", instructionName, instructionTotalClocks);
                operationTotalClocks += instructionTotalClocks; // Add to operation total
            }

            // Print operation total after all instructions
            ss << fmt::format("  Total clock cycles for operation: {}\n", operationTotalClocks);
            ss << "\n";
        }

        return ss.str();
    }

    std::ostream& operator<<(std::ostream& stream, DetailedSummary const& detailedSummary)
    {
        return stream << detailedSummary.toString();
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

        Log::info("MemoryTracer::LDSBankModel()");
        // 64KiB bank model: 4 bytes per bank entry, 32 banks, 512 entries per bank
        auto model = LDSBankModel(4, 32, 512);

        // For LDS, just simulate using 1 workgroup
        auto workgroups            = 1;
        auto workitemsPerWorkgroup = product(invocation.workgroupSize);
        tracer.simulateLaunch(model, workgroups, workitemsPerWorkgroup);

        return model.summary();
    }
}
