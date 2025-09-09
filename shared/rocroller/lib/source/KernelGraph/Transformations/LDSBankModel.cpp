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
        auto ldsOp = std::get_if<MemoryOpLDS>(&event.memoryOp);
        if(!ldsOp)
            return;

        auto& opAccesses        = m_hierarchicalAccesses[event.operationTag];
        opAccesses.operationTag = event.operationTag;
        opAccesses.ldsTag
            = ldsOp->direction == Direction::Load ? event.sourceTag : event.destinationTag;

        // Since simulateLaunch now provides instruction-level events,
        // we can directly work with the bytesRequested as it's already limited to 16 bytes max
        uint instructionDwords = (event.bytesRequested + 3) / 4; // Round up to dwords

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

        // Create thread access with the base address and workitem index
        ThreadAccess threadAccess;
        threadAccess.workitem  = event.workItem;
        threadAccess.address   = event.byteOffset;
        threadAccess.bankIndex = (event.byteOffset / m_entryWidthInBytes) % m_numBanks;

        // Simply add the thread access to the instruction's accesses vector
        targetInstruction->accesses.push_back(threadAccess);

        // Track unique banks accessed for legacy m_bankAccesses
        std::set<uint> banksAccessed;
        for(uint offset = 0; offset < event.bytesRequested; offset += m_entryWidthInBytes)
        {
            uint currentAddr = event.byteOffset + offset;
            uint bankIndex   = (currentAddr / m_entryWidthInBytes) % m_numBanks;
            banksAccessed.insert(bankIndex);
        }

        // Also update legacy m_bankAccesses for backward compatibility
        // Add entry for each bank accessed
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

    std::map<uint, std::vector<ThreadAccess>>
        LDSBankModel::makeBankMappingForThreads(const std::vector<ThreadAccess>& threads,
                                                uint                             dwords,
                                                uint                             entryWidthInBytes,
                                                uint                             numBanks)
    {
        std::map<uint, std::vector<ThreadAccess>> bankMapping;

        for(const auto& thread : threads)
        {
            // Calculate all banks this access touches
            uint bytesPerAccess = dwords * 4;
            for(uint offset = 0; offset < bytesPerAccess; offset += entryWidthInBytes)
            {
                uint currentAddr = thread.address + offset;
                uint bankIndex   = (currentAddr / entryWidthInBytes) % numBanks;
                bankMapping[bankIndex].push_back(thread);
            }
        }

        return bankMapping;
    }

    ThreadGroup LDSBankModel::buildThreadGroupWithClockCycles(
        const std::map<uint, std::vector<ThreadAccess>>& bankToThreads, uint groupIndex)
    {
        ThreadGroup threadGroup;
        threadGroup.groupIndex = groupIndex;

        // Simulate clock cycles needed to resolve bank conflicts
        std::map<uint, std::vector<ThreadAccess>> remainingAccesses = bankToThreads;

        while(!remainingAccesses.empty())
        {
            std::vector<ThreadAccess> clockCycleThreads;
            std::set<uint>    usedThreads; // Track which threads have been scheduled this cycle
            std::vector<uint> banksToRemove;

            // Process one thread per bank for this clock cycle
            for(auto& [bankIndex, threadList] : remainingAccesses)
            {
                if(!threadList.empty())
                {
                    // Find first thread that hasn't been scheduled yet
                    for(auto it = threadList.begin(); it != threadList.end(); ++it)
                    {
                        if(usedThreads.find(it->workitem) == usedThreads.end())
                        {
                            clockCycleThreads.push_back(*it);
                            usedThreads.insert(it->workitem);
                            threadList.erase(it);
                            break;
                        }
                    }

                    if(threadList.empty())
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

            // Add this clock cycle to the thread group
            if(!clockCycleThreads.empty())
            {
                threadGroup.clockCycles.push_back(clockCycleThreads);
            }
        }

        return threadGroup;
    }

    static uint
        calculateBankConflictCycles(const std::map<uint, std::vector<uint>>& bankToAddressIndices)
    {
        // Simulate clock cycles needed to resolve bank conflicts
        std::map<uint, std::vector<uint>> remainingAccesses = bankToAddressIndices;
        uint                              clockCycles       = 0;

        while(!remainingAccesses.empty())
        {
            std::set<uint>    usedAddresses; // Track which addresses have been scheduled this cycle
            std::vector<uint> banksToRemove;

            // Process one address per bank for this clock cycle
            for(auto& [bankIndex, addressIndices] : remainingAccesses)
            {
                if(!addressIndices.empty())
                {
                    // Find first address that hasn't been scheduled yet
                    for(auto it = addressIndices.begin(); it != addressIndices.end(); ++it)
                    {
                        if(usedAddresses.find(*it) == usedAddresses.end())
                        {
                            usedAddresses.insert(*it);
                            addressIndices.erase(it);
                            break;
                        }
                    }

                    if(addressIndices.empty())
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

            // Increment clock cycle count if we scheduled any addresses
            if(!usedAddresses.empty())
            {
                clockCycles++;
            }
        }

        return clockCycles;
    }

    uint LDSBankModel::immediateClockCount(GPUArchitectureGFX           gfx,
                                           const MemoryOpLDS&           memoryOp,
                                           uint                         dwords,
                                           const std::vector<uint32_t>& addresses,
                                           uint                         entryWidthInBytes,
                                           uint                         numBanks)
    {
        if(addresses.empty())
        {
            return 0;
        }

        // Get the maximum number of threads that can operate per clock for this instruction
        uint threadsPerClock = getThreadsPerClock(memoryOp, dwords, gfx);

        // Track total bank conflict cycles across all thread groups
        uint totalBankConflictCycles = 0;

        // Process addresses in groups based on threads-per-clock limit
        for(size_t groupStart = 0; groupStart < addresses.size(); groupStart += threadsPerClock)
        {
            // Get addresses for this thread group
            size_t groupEnd = std::min(groupStart + threadsPerClock, addresses.size());
            std::vector<uint32_t> groupAddresses(addresses.begin() + groupStart,
                                                 addresses.begin() + groupEnd);

            // Create a mapping from bank index to address indices for this group
            // For multi-dword accesses, we need to track all banks touched by each address
            std::map<uint, std::vector<uint>> bankToAddressIndices;

            for(size_t i = 0; i < groupAddresses.size(); ++i)
            {
                uint baseAddr       = groupAddresses[i];
                uint bytesPerAccess = dwords * 4;

                // Calculate all banks this access touches
                for(uint offset = 0; offset < bytesPerAccess; offset += entryWidthInBytes)
                {
                    uint currentAddr = baseAddr + offset;
                    uint bankIndex   = (currentAddr / entryWidthInBytes) % numBanks;
                    bankToAddressIndices[bankIndex].push_back(i);
                }
            }

            // Calculate bank conflict cycles for this group
            uint groupCycles = calculateBankConflictCycles(bankToAddressIndices);
            totalBankConflictCycles += groupCycles;
        }

        // Add 4 cycles for read/write address transfer
        const uint addressTransferCycles = 4;
        uint       totalCycles           = totalBankConflictCycles + addressTransferCycles;

        return totalCycles;
    }

    DetailedSummary LDSBankModel::detailedSummary(GPUArchitectureGFX gfx) const
    {
        DetailedSummary detailed;
        detailed.gfx = gfx;

        // TODO: these should not be hardcoded
        const uint entryWidthInBytes = 4;
        const uint numBanks          = 64;

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

                // Copy all threads from source accesses
                std::vector<ThreadAccess> allThreads = sourceInstr.accesses;

                // Sort threads by workitem ID for consistent grouping
                std::sort(allThreads.begin(),
                          allThreads.end(),
                          [](const ThreadAccess& a, const ThreadAccess& b) {
                              return a.workitem < b.workitem;
                          });

                // Keep the raw accesses for compatibility
                instr.accesses = allThreads;

                // Get threads per clock for this instruction
                const auto& ldsOp           = std::get<MemoryOpLDS>(sourceInstr.memoryOp);
                uint        threadsPerClock = getThreadsPerClock(ldsOp, sourceInstr.dwords, gfx);

                // Group threads based on threads per clock and organize into clock cycles
                uint groupIndex = 0;
                for(size_t i = 0; i < allThreads.size(); i += threadsPerClock)
                {
                    ThreadGroup threadGroup;
                    threadGroup.groupIndex = groupIndex++;

                    // Collect threads for this group (up to threadsPerClock)
                    std::vector<ThreadAccess> groupThreads;
                    for(size_t j = i; j < i + threadsPerClock && j < allThreads.size(); ++j)
                    {
                        groupThreads.push_back(allThreads[j]);
                    }

                    // Create a mapping of bank index to list of thread accesses
                    // For multi-dword accesses, we need to track all banks touched
                    auto bankToThreads = makeBankMappingForThreads(
                        groupThreads, instr.dwords, entryWidthInBytes, numBanks);

                    // Build thread group with clock cycles based on bank conflict resolution
                    auto threadGroupWithCycles
                        = buildThreadGroupWithClockCycles(bankToThreads, groupIndex - 1);

                    instr.threadGroups.push_back(threadGroupWithCycles);
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

                // Process each thread group (already organized by detailedSummary)
                for(const auto& threadGroup : instr.threadGroups)
                {
                    // Count total threads in all clock cycles
                    uint totalThreads = 0;
                    for(const auto& clockCycle : threadGroup.clockCycles)
                    {
                        totalThreads += clockCycle.size();
                    }

                    ss << fmt::format("    Thread Group {}: {} threads, {} clock cycles\n",
                                      threadGroup.groupIndex,
                                      totalThreads,
                                      threadGroup.clockCycles.size());

                    // Print each clock cycle
                    uint clockCycleIndex = 0;
                    for(const auto& clockCycleThreads : threadGroup.clockCycles)
                    {
                        ss << fmt::format("      Clock Cycle {}:\n", clockCycleIndex);

                        // Group addresses by work-item
                        std::map<uint, std::vector<uint32_t>> workitemToAddresses;
                        std::set<uint>                        banksUsed;

                        for(const auto& thread : clockCycleThreads)
                        {
                            // Calculate all addresses this thread accesses
                            uint bytesPerAccess = instr.dwords * 4;
                            for(uint offset = 0; offset < bytesPerAccess; offset += 4)
                            {
                                uint currentAddr = thread.address + offset;
                                uint bankIndex   = (currentAddr / 4) % 64;
                                workitemToAddresses[thread.workitem].push_back(currentAddr);
                                banksUsed.insert(bankIndex);
                            }
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
                        std::vector<uint> usedBanks(banksUsed.begin(), banksUsed.end());
                        std::vector<uint> unusedBanks;

                        for(uint bankIdx = 0; bankIdx < 64; ++bankIdx)
                        {
                            if(banksUsed.find(bankIdx) == banksUsed.end())
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

                        clockCycleIndex++;
                    }

                    ss << fmt::format("      Total clock cycles: {}\n",
                                      threadGroup.clockCycles.size());
                    instructionTotalClocks
                        += threadGroup.clockCycles.size(); // Add to instruction total
                }

                // Add 4 cycles for read/write address transfer
                uint bankConflictCycles    = instructionTotalClocks;
                uint addressTransferCycles = 4;
                instructionTotalClocks += addressTransferCycles;

                // Optional: Verify using the standalone immediateClockCount function
                // This demonstrates how the new function can be used independently
                if(!instr.accesses.empty())
                {
                    std::vector<uint32_t> addresses;
                    for(const auto& access : instr.accesses)
                    {
                        addresses.push_back(access.address);
                    }

                    const auto& ldsOp = std::get<MemoryOpLDS>(instr.memoryOp);
                    uint        verifyClocks
                        = LDSBankModel::immediateClockCount(gfx, ldsOp, instr.dwords, addresses);

                    // Note: The verification might differ slightly due to different ordering
                    // of threads in the detailed simulation vs the standalone calculation
                    ss << fmt::format("    Total clock cycles for {}: {} ({} bank conflict + {} "
                                      "address transfer)\n",
                                      instructionName,
                                      instructionTotalClocks,
                                      bankConflictCycles,
                                      addressTransferCycles);
                    ss << fmt::format("    [Verification using immediateClockCount: {} cycles]\n",
                                      verifyClocks);
                }
                else
                {
                    ss << fmt::format("    Total clock cycles for {}: {} ({} bank conflict + {} "
                                      "address transfer)\n",
                                      instructionName,
                                      instructionTotalClocks,
                                      bankConflictCycles,
                                      addressTransferCycles);
                }

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
