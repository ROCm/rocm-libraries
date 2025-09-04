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

    /**
     * LDS bank model
     */
    struct LDSBankModel
    {
        struct LDSBankAccess
        {
            int                            operationTag;
            int                            ldsTag;
            Direction                      direction;
            uint                           workitem;
            uint                           bankIndex;
            std::vector<MemoryInstruction> instructions;
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

        std::string toString() const;

    private:
        uint m_entryWidthInBytes;
        uint m_numBanks;
        uint m_numEntriesPerBank;

        std::map<int, std::vector<LDSBankAccess>> m_bankAccesses;
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
