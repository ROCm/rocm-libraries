

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

#pragma once

#include <deque>
#include <map>
#include <ostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/KernelGraph/Transforms/MemoryTracer.hpp>

namespace rocRoller::KernelGraph::MemoryTracer
{
    struct Summary
    {
        static constexpr bool echoBanks = false;

        struct Banks
        {
            uint   bankIndex;
            size_t workitemsAccessed;
            bool   imbalanced;
        };
        struct Access
        {
            int                           ldsTag;
            std::vector<Banks>            accessedBanks;
            std::vector<std::vector<int>> banksToWorkitems;
        };

        std::map<int, Access> accesses;
        std::set<int>         imbalancedTags;

        std::string toString() const;
    };

    std::ostream& operator<<(std::ostream& stream, Summary const& summary);

    struct ThreadAccess
    {
        uint     workitem;
        uint32_t address;
        uint     bankIndex;
    };

    struct ThreadGroup
    {
        uint                                   groupIndex;
        std::vector<std::vector<ThreadAccess>> clockCycles; // Each inner vector is one clock cycle
    };

    struct InstructionAccesses
    {
        MemoryOpLDS           memoryOp;
        int                   dwords;
        std::vector<uint32_t> baseAddresses;
    };

    struct OperationAccesses
    {
        int                              operationTag;
        int                              ldsTag;
        std::vector<InstructionAccesses> instructions;
    };

    struct DetailedSummary
    {
        std::map<int, OperationAccesses> accesses;
        GPUArchitectureGFX               gfx;

        std::string toString() const;
    };

    std::ostream& operator<<(std::ostream& stream, DetailedSummary const& detailedSummary);

    /**
     * LDS bank model
     */
    struct LDSBankModel
    {
        struct LDSBankAccess
        {
            int       operationTag;
            int       ldsTag;
            Direction direction;
            uint      workitem;
            uint      bankIndex;
        };

        /**
         * @brief Construct a new LDSBankModel object.
         *
         * @param entryWidthInBytes Width of each bank entry in bytes.
         * @param numBanks Number of banks in the LDS.
         * @param numEntriesPerBank Number of entries per bank.
         */
        LDSBankModel(uint entryWidthInBytes, uint numBanks, uint numEntriesPerBank);

        bool filter(MemoryEventExpression event);

        void simulate(MemoryEventSimulated event);

        Summary summary() const;

        DetailedSummary detailedSummary(GPUArchitectureGFX gfx) const;

        std::string toString() const;

        /**
         * @brief Calculate how many threads per clock can operate on an LDS instruction
         * 
         * @param memoryOp The LDS memory operation
         * @param dwords Number of dwords (1 for b32, 2 for b64, 3 for b96, 4 for b128)
         * @param gfx The GPU architecture
         * @return Number of threads that can operate per clock
         */
        static uint
            getThreadsPerClock(const MemoryOpLDS& memoryOp, uint dwords, GPUArchitectureGFX gfx);

        /**
         * @brief Get the number of LDS banks for a given GPU architecture
         * 
         * @param gfx The GPU architecture
         * @return Number of LDS banks: 64 for GFX950, 32 for all other architectures
         */
        static uint getNumLDSBanks(GPUArchitectureGFX gfx);

        /**
         * @brief Calculate the immediate clock count for an LDS instruction
         * 
         * This function encapsulates all the logic for determining how many clock cycles
         * an LDS instruction will take, including:
         * - Dividing base addresses into thread groups based on architecture limits
         * - Maximizing threads without conflicts within groups
         * - Resolving bank conflicts to determine actual cycles
         * - Adding address transfer overhead
         * 
         * @param gfx The GPU architecture
         * @param memoryOp The LDS memory operation (load/store)
         * @param dwords Number of dwords accessed (1 for b32, 2 for b64, 3 for b96, 4 for b128)
         * @param baseAddresses Vector of base LDS addresses being accessed. For multi-dword accesses,
         *                      the actual addresses accessed are calculated from these base addresses.
         * @param numBanks Number of banks in the LDS
         * @param entryWidthInBytes Width of each bank entry in bytes (default: 4)
         * @return Total number of clock cycles for this instruction
         */
        static uint immediateClockCount(GPUArchitectureGFX           gfx,
                                        const MemoryOpLDS&           memoryOp,
                                        uint                         dwords,
                                        const std::vector<uint32_t>& baseAddresses,
                                        uint                         numBanks,
                                        uint                         entryWidthInBytes = 4);

        /**
         * @brief Divide addresses into thread groups based on threads-per-clock limit
         * 
         * @param addresses Vector of LDS addresses
         * @param threadsPerClock Maximum number of threads that can operate per clock
         * @return Vector of thread groups, each containing addresses for that group
         */
        static std::vector<std::vector<uint32_t>>
            divideIntoThreadGroups(const std::vector<uint32_t>& addresses, uint threadsPerClock);

        /**
         * @brief Create a mapping from bank indices to address counts
         * 
         * For multi-dword accesses, tracks all banks touched by each base address.
         * The resulting map has bank indices as keys and counts of addresses accessing that bank as values.
         * 
         * @param baseAddresses Vector of base LDS addresses. For multi-dword accesses, the actual
         *                      addresses accessed are calculated from these base addresses.
         * @param dwords Number of dwords accessed per address
         * @param entryWidthInBytes Width of each bank entry in bytes
         * @param numBanks Number of banks in the LDS
         * @return Map from bank index to count of addresses that access that bank
         */
        static std::map<uint, uint>
            createBankToAddressCounts(const std::vector<uint32_t>& baseAddresses,
                                      uint                         dwords,
                                      uint                         entryWidthInBytes,
                                      uint                         numBanks);

        /**
         * @brief Calculate the number of clock cycles needed to resolve bank conflicts
         * 
         * Simulates the bank conflict resolution process where only one address per bank
         * can be serviced per clock cycle. The calculation is based on the maximum
         * number of addresses that access any single bank.
         * 
         * @param bankToAddressCounts Map from bank index to count of addresses accessing that bank
         * @return Number of clock cycles needed to process all addresses
         */
        static uint calculateBankConflictCycles(const std::map<uint, uint>& bankToAddressCounts);

        /**
         * @brief Generate a detailed analysis string for an LDS instruction
         * 
         * This function generates the instruction name and computes cycle counts,
         * showing detailed bank contention information for each thread group.
         * 
         * @param instr The instruction access information
         * @param gfx The GPU architecture
         * @param[out] totalCycles Output parameter for the total instruction cycles
         * @return Formatted string containing the detailed analysis
         */
        static std::string instructionDetailedAnalysis(const InstructionAccesses& instr,
                                                       GPUArchitectureGFX         gfx,
                                                       uint&                      totalCycles);

    private:
        uint m_entryWidthInBytes;
        uint m_numBanks;
        uint m_numEntriesPerBank;

        std::map<int, std::vector<LDSBankAccess>> m_bankAccesses; // Keep for backward compatibility
        std::map<int, OperationAccesses>          m_hierarchicalAccesses;
    };

    std::ostream& operator<<(std::ostream& stream, LDSBankModel const& ldsBankModel);

    /**
     * @brief Trace memory accesses in a kernel graph and analyze LDS bank conflicts
     * 
     * @param original The original kernel graph to analyze
     * @param invocation The kernel invocation parameters
     * @return Summary of LDS bank access patterns and conflicts
     */
    Summary memoryTrace(KernelGraph const& original, KernelInvocation const& invocation);
}
