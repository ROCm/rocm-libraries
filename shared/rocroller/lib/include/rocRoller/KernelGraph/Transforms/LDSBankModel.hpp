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
        MemoryOp              memoryOp;
        int                   dwords;
        std::vector<uint32_t> addresses;
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
         *         - GFX950 reads: 32 for b32/b64, 16 for b128, 8 for b96
         *         - All other cases: 32 for b32, 16 for b64, 8 for b96/b128
         */
        static uint
            getThreadsPerClock(const MemoryOpLDS& memoryOp, uint dwords, GPUArchitectureGFX gfx);

        /**
         * @brief Create a mapping from bank indices to addresses
         * 
         * @param addresses Vector of LDS addresses
         * @param entryWidthInBytes Width of each bank entry in bytes
         * @param numBanks Number of banks in the LDS
         * @return Map from bank index to vector of addresses that map to that bank
         */
        static std::map<uint, std::vector<uint32_t>> makeBankMapping(
            const std::vector<uint32_t>& addresses, uint entryWidthInBytes, uint numBanks);

        /**
         * @brief Calculate the immediate clock count for an LDS instruction
         * 
         * This function encapsulates all the logic for determining how many clock cycles
         * an LDS instruction will take, including:
         * - Dividing addresses into thread groups based on architecture limits
         * - Maximizing threads without conflicts within groups
         * - Resolving bank conflicts to determine actual cycles
         * - Adding address transfer overhead
         * 
         * @param gfx The GPU architecture
         * @param memoryOp The LDS memory operation (load/store)
         * @param dwords Number of dwords accessed (1 for b32, 2 for b64, 3 for b96, 4 for b128)
         * @param addresses Vector of LDS addresses being accessed
         * @param entryWidthInBytes Width of each bank entry in bytes (default: 4)
         * @param numBanks Number of banks in the LDS (default: 64)
         * @return Total number of clock cycles for this instruction
         */
        static uint immediateClockCount(GPUArchitectureGFX           gfx,
                                        const MemoryOpLDS&           memoryOp,
                                        uint                         dwords,
                                        const std::vector<uint32_t>& addresses,
                                        uint                         entryWidthInBytes = 4,
                                        uint                         numBanks          = 64);

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
         * @brief Create a mapping from bank indices to address indices for conflict resolution
         * 
         * For multi-dword accesses, tracks all banks touched by each address.
         * The resulting map has bank indices as keys and vectors of address indices as values.
         * 
         * @param addresses Vector of LDS addresses
         * @param dwords Number of dwords accessed per address
         * @param entryWidthInBytes Width of each bank entry in bytes
         * @param numBanks Number of banks in the LDS
         * @return Map from bank index to vector of address indices that access that bank
         */
        static std::map<uint, std::vector<uint>>
            createBankToAddressIndices(const std::vector<uint32_t>& addresses,
                                       uint                         dwords,
                                       uint                         entryWidthInBytes,
                                       uint                         numBanks);

        /**
         * @brief Calculate the number of clock cycles needed to resolve bank conflicts
         * 
         * Simulates the bank conflict resolution process where only one address per bank
         * can be serviced per clock cycle. Addresses are scheduled to avoid conflicts,
         * with each address being processed exactly once.
         * 
         * @param bankToAddressIndices Map from bank index to vector of address indices
         * @return Number of clock cycles needed to process all addresses
         */
        static uint calculateBankConflictCycles(
            const std::map<uint, std::vector<uint>>& bankToAddressIndices);

    private:
        uint m_entryWidthInBytes;
        uint m_numBanks;
        uint m_numEntriesPerBank;

        std::map<int, std::vector<LDSBankAccess>> m_bankAccesses; // Keep for backward compatibility
        std::map<int, OperationAccesses>          m_hierarchicalAccesses;

        /**
         * @brief Create a mapping from bank indices to thread accesses
         * 
         * @param threads Vector of thread accesses
         * @param dwords Number of dwords accessed per thread
         * @param entryWidthInBytes Width of each bank entry in bytes
         * @param numBanks Number of banks in the LDS
         * @return Map from bank index to vector of thread accesses that map to that bank
         */
        static std::map<uint, std::vector<ThreadAccess>>
            makeBankMappingForThreads(const std::vector<ThreadAccess>& threads,
                                      uint                             dwords,
                                      uint                             entryWidthInBytes,
                                      uint                             numBanks);

        /**
         * @brief Build a ThreadGroup with clock cycles from bank mapping
         * 
         * @param bankToThreads Map from bank index to vector of thread accesses
         * @param groupIndex The index of this thread group
         * @return ThreadGroup with clock cycles populated based on bank conflict resolution
         */
        static ThreadGroup buildThreadGroupWithClockCycles(
            const std::map<uint, std::vector<ThreadAccess>>& bankToThreads, uint groupIndex);
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
