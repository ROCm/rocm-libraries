// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <sstream>
#include <unordered_map>

#include <fmt/format.h>

#include <rocRoller/Expression.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/Scheduling/LDSModel.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller::Scheduling::LDSModel
{
    std::optional<std::pair<LdsDirection, int>>
        getLdsInfoFromOpcodeIfSupported(const std::string& opCode)
    {
        // Model does not support sub-dword or special opcodes
        // e.g. ds_read_u8, ds_read2st64_b32

        LdsDirection direction;
        if(opCode.find("ds_write_") != std::string::npos)
            direction = LdsDirection::Write;
        else if(opCode.find("ds_read_") != std::string::npos)
            direction = LdsDirection::Read;
        else
            return std::nullopt;

        int dwords;
        if(opCode.find("_b32") != std::string::npos)
            dwords = 1;
        else if(opCode.find("_b64") != std::string::npos)
            dwords = 2;
        else if(opCode.find("_b96") != std::string::npos)
            dwords = 3;
        else if(opCode.find("_b128") != std::string::npos)
            dwords = 4;
        else
            return std::nullopt;

        return std::make_optional(std::make_pair(direction, dwords));
    }

    int getQueueSlotsRequired(LdsDirection direction, int dwords)
    {
        if(direction == LdsDirection::Write)
            return dwords + 1;
        return 1;
    }

    LDSModule::LDSModule(GPUArchitectureGFX gfx, int waveCount)
        : m_gfx(gfx)
        , m_programCycle(0)
        , m_multiplierWaveCount(waveCount)
        // With 3 or more waves, two SIMDs will be active on at least one SP
        // so two SIMDs share the same LDS queues
        , m_multiplierQueueSlots(waveCount > 2 ? 2 : 1)
    {
    }

    void LDSModule::incrementProgramCycleBy(int cycles)
    {
        m_programCycle += cycles;
    }

    void LDSModule::reset()
    {
        m_programCycle = 0;
        m_commandQueue.clear();
        m_waitcntQueue.clear();
        m_dataQueue.clear();
    }

    int LDSModule::getRemainingDataSlots() const
    {
        int usedSlots = 0;
        for(const auto& slotFreedCycle : m_dataQueue)
        {
            if(slotFreedCycle > static_cast<unsigned int>(m_programCycle))
            {
                usedSlots++;
            }
        }
        return dataQueueSize - usedSlots;
    }

    std::tuple<int, int> LDSModule::predictCycles(const RuntimeLDSInstruction& instr) const
    {
        int stallCycles = 0;

        const auto multiplier            = m_multiplierQueueSlots;
        const auto [direction, dwords]   = std::make_pair(instr.memoryOp.direction, instr.dwords);
        const auto requiredDataSlots     = getQueueSlotsRequired(direction, dwords) * multiplier;
        const auto requiredCommandSlots  = multiplier;
        const auto remainingDataSlots    = getRemainingDataSlots();
        const auto remainingCommandSlots = commandQueueSize - m_commandQueue.size();

        if(requiredCommandSlots > remainingCommandSlots)
        {
            const auto completionCycle
                = m_commandQueue[requiredCommandSlots - remainingCommandSlots - 1];
            stallCycles = std::max(stallCycles, static_cast<int>(completionCycle) - m_programCycle);
        }

        if(requiredDataSlots > remainingDataSlots)
        {
            const auto completionCycle = m_dataQueue[requiredDataSlots - remainingDataSlots - 1];
            stallCycles = std::max(stallCycles, static_cast<int>(completionCycle) - m_programCycle);
        }

        MemoryOpLDS memOp{direction};
        int         additionalCycles
            = (getInstructionIssueCycles(memOp, dwords)) * m_multiplierQueueSlots - 4;

        return std::make_tuple(stallCycles, additionalCycles);
    }

    int LDSModule::predictWaitcntStallCycles(int waitcnt) const
    {
        int stallCycles = 0;
        AssertFatal(m_waitcntQueue.size() <= std::numeric_limits<int>::max(),
                    "Waitcnt queue size exceeds int max");
        const auto size = static_cast<int>(m_waitcntQueue.size());
        if(waitcnt >= 0 && size > waitcnt)
        {
            // Wait until there are only 'waitcnt' commands in queue
            const auto commandsToWaitFor = size - waitcnt - 1;
            AssertFatal(commandsToWaitFor >= 0 && commandsToWaitFor < size,
                        ShowValue(commandsToWaitFor),
                        ShowValue(size),
                        ShowValue(waitcnt));
            const auto waitCompletionCycle = m_waitcntQueue[commandsToWaitFor];
            stallCycles                    = waitCompletionCycle - m_programCycle;
        }
        AssertFatal(stallCycles >= 0, ShowValue(stallCycles));
        return stallCycles;
    }

    void LDSModule::scheduleInstruction(const RuntimeLDSInstruction& origInstr)
    {
        auto instr = origInstr;
        updateQueues();

        const int requiredSlots = getQueueSlotsRequired(instr.memoryOp.direction, instr.dwords);

        // Each SP has own queue to LDS module, therefore at most only two waves' LDS instructions are in consideration
        // When workgroup size is 256, the model assumes the two groups of 128 threads have the same access pattern
        std::vector<size_t> truncatedAddresses(
            instr.baseAddresses.begin(),
            instr.baseAddresses.begin()
                + std::min(static_cast<size_t>(128), instr.baseAddresses.size()));
        instr.baseAddresses = std::move(truncatedAddresses);

        int dataCycles = getInstructionDataCycles(instr, m_gfx);

        // As soon as the waves on each SP go out-of-sync, this conflict no longer occurs
        // In the non-overlapping case, this is true even if in-sync
        if(m_multiplierWaveCount > 1 && hasNonOverlappingBankAccess(instr, m_gfx))
        {
            Log::debug("Non-overlapping bank access detected, reducing data cycles");
            dataCycles /= 2;
        }

        AssertFatal(getRemainingDataSlots() >= requiredSlots
                        && m_commandQueue.size() < static_cast<size_t>(commandQueueSize),
                    "Expected queue space to be accounted for in predict function and passed "
                    "through to total cycles calculation",
                    ShowValue(getRemainingDataSlots()),
                    ShowValue(requiredSlots));

        const auto roundtripLatencyCycles = 40; // Determined experimentally
        auto       waitcntBase            = m_programCycle + roundtripLatencyCycles + dataCycles;

        for(int i = 0; i < m_multiplierQueueSlots; ++i)
        {
            if(instr.memoryOp.direction == LdsDirection::Write)
            {
                auto cmdBase = m_programCycle + dataCycles;
                if(!m_commandQueue.empty())
                {
                    cmdBase = m_commandQueue.back() + dataCycles;
                }

                if(!m_waitcntQueue.empty())
                {
                    waitcntBase = std::max(waitcntBase, m_waitcntQueue.back() + dataCycles);
                }

                m_commandQueue.push_back(cmdBase);
                m_waitcntQueue.push_back(waitcntBase);
                {
                    auto const base = m_dataQueue.empty() ? (m_programCycle) : (m_dataQueue.back());

                    // For writes, 1 slot for addresses, rest for data
                    m_dataQueue.push_back(base);
                    for(int j = 0; j < (requiredSlots - 1); ++j)
                    {
                        m_dataQueue.push_back(base + dataCycles);
                    }
                }
            }
            else if(instr.memoryOp.direction == LdsDirection::Read)
            {
                auto cmdBase = m_programCycle + dataCycles;
                if(!m_commandQueue.empty())
                {
                    cmdBase = m_commandQueue.back() + dataCycles;
                }
                if(!m_waitcntQueue.empty())
                {
                    waitcntBase = std::max(waitcntBase, m_waitcntQueue.back() + dataCycles);
                }

                // read should only need 1 slot
                AssertFatal(requiredSlots == 1, ShowValue(requiredSlots));

                m_commandQueue.push_back(cmdBase);
                m_waitcntQueue.push_back(waitcntBase);
                m_dataQueue.push_back(cmdBase);
            }
            else
            {
                Throw<FatalError>("Unsupported LDS direction", ShowValue(instr.toString()));
            }
        }

        const auto cmdQueueStr
            = fmt::format("{}", fmt::join(m_commandQueue.begin(), m_commandQueue.end(), ", "));
        const auto waitcntQueueStr
            = fmt::format("{}", fmt::join(m_waitcntQueue.begin(), m_waitcntQueue.end(), ", "));
        const auto dataQueueStr
            = fmt::format("{}", fmt::join(m_dataQueue.begin(), m_dataQueue.end(), ", "));
        Log::debug("LdsScheduler {}: scheduled ds_{}_b{}, dataCycles {}, "
                   "issue cycles {}, "
                   "command queue cycles [{}], "
                   "data queue [{}], "
                   "waitcnt queue [{}]",
                   m_programCycle,
                   instr.memoryOp.direction == LdsDirection::Read ? "read" : "write",
                   instr.dwords * 32,
                   dataCycles,
                   getInstructionIssueCycles(instr.memoryOp, instr.dwords),
                   cmdQueueStr,
                   dataQueueStr,
                   waitcntQueueStr);
    }

    void LDSModule::updateQueues()
    {
        while(!m_commandQueue.empty() && static_cast<int>(m_commandQueue.front()) <= m_programCycle)
        {
            m_commandQueue.pop_front();
        }
        while(!m_waitcntQueue.empty() && static_cast<int>(m_waitcntQueue.front()) <= m_programCycle)
        {
            m_waitcntQueue.pop_front();
        }
        while(!m_dataQueue.empty() && static_cast<int>(m_dataQueue.front()) <= m_programCycle)
        {
            m_dataQueue.pop_front();
        }
        // Assert invariant that all queues are in increasing order
        AssertFatal(
            std::is_sorted(m_commandQueue.begin(), m_commandQueue.end()),
            "Command queue not sorted: ",
            fmt::format("{}", fmt::join(m_commandQueue.begin(), m_commandQueue.end(), ", ")));
        AssertFatal(
            std::is_sorted(m_waitcntQueue.begin(), m_waitcntQueue.end()),
            "Waitcnt queue not sorted: ",
            fmt::format("{}", fmt::join(m_waitcntQueue.begin(), m_waitcntQueue.end(), ", ")));
        AssertFatal(std::is_sorted(m_dataQueue.begin(), m_dataQueue.end()),
                    "Data queue not sorted: ",
                    fmt::format("{}", fmt::join(m_dataQueue.begin(), m_dataQueue.end(), ", ")));
    }

    std::string RuntimeLDSInstruction::toString() const
    {
        std::stringstream ss;
        ss << fmt::format("ds_{}_b{}, baseAddresses: [",
                          memoryOp.direction == LdsDirection::Read ? "read" : "write",
                          dwords * 32);
        rocRoller::streamJoin(ss, baseAddresses, ", ");
        ss << "]";
        return ss.str();
    }

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
            size_t              groupEnd = groupStart + threadsPerClock;
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

    template <std::integral T, class Compare = std::less<T>>
    bool disjoint_ordered(const std::set<T, Compare>& a, const std::set<T, Compare>& b)
    {
        auto       it1  = a.begin();
        auto       it2  = b.begin();
        const auto end1 = a.end();
        const auto end2 = b.end();
        Compare    comp;

        while(it1 != end1 && it2 != end2)
        {
            if(*it1 == *it2)
                return false; // found common element
            if(comp(*it1, *it2))
                ++it1;
            else
                ++it2; // advance the smaller
        }
        return true; // no common element
    }

    // Find a rotation k such that for all i, A[i] is disjoint from B[(i + k) % n]
    template <std::integral T, class Compare = std::less<T>>
    std::optional<std::size_t> find_disjoint_rotation(const std::vector<std::set<T, Compare>>& A,
                                                      const std::vector<std::set<T, Compare>>& B)
    {
        const std::size_t n = A.size();
        if(n != B.size())
            return std::nullopt;
        if(n == 0)
            return 0; // vacuously true: rotation 0 is fine

        for(std::size_t k = 0; k < n; ++k)
        {
            bool ok = true;
            for(std::size_t i = 0; i < n; ++i)
            {
                const std::size_t j = (i + k) % n;
                if(!disjoint_ordered<T, Compare>(A[i], B[j]))
                {
                    ok = false;
                    break;
                }
            }
            if(ok)
                return k; // found a valid rotation
        }
        return std::nullopt; // none exists
    }

    bool hasNonOverlappingBankAccess(const RuntimeLDSInstruction& instr, GPUArchitectureGFX gfx)
    {
        if(instr.baseAddresses.size() == 64)
        {
            return true;
        }

        /*
        Consider the case where at least 2 SIMDs are active (workgroup size >128).
        For each SP, at most 16 threads are processed per cycle (regardless of instr dwords).
        This totals to 32 threads per cycle. If these 32 threads access conflicting banks, then stalls will occur.
        This type of bank conflict does not occur with workgroup size of 64, where only one SP is active.

        On gfx950, a ds_read_b64 processes 2 dwords per thread, and ds_read_b128 processes 4 dwords per thread.

        The way to reproduce this behavior is to write a microkernel that uses ds_read_b64 with two cases:
            1) each thread accesses unique banks, e.g. thread 0 accesses bank 0 and 1, thread 1 accesses bank 2 and 3, etc.
            2) each thread accesses every other bank pairs, e.g. thread 0 accesses bank 0 and 1, thread 1 accesses bank 4, 5, etc.

            Compare the change in latency numbers from workgroup size of 64 to 128 for both cases.
            Notice only in case 2) do the latency numbers increase (gradually reach double).
        */
        const auto threadGroupBankMappings = computeThreadGroupBankMappings(instr, gfx);

        AssertFatal(threadGroupBankMappings.size() % 2 == 0,
                    "Odd {} number of thread groups not supported",
                    threadGroupBankMappings.size());

        std::vector<std::set<uint>> threadGroupBankSets;
        for(const auto& bankMapping : threadGroupBankMappings)
        {
            std::set<uint> bankSet;
            for(const auto& [bankIndex, count] : bankMapping)
            {
                bankSet.insert(bankIndex);
            }
            threadGroupBankSets.push_back(bankSet);
        }

        const auto maybeK = find_disjoint_rotation(threadGroupBankSets, threadGroupBankSets);
        return maybeK.has_value();
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

    std::ostream& operator<<(std::ostream& stream, RuntimeLDSInstruction const& instruction)
    {
        return stream << instruction.toString();
    }
}
