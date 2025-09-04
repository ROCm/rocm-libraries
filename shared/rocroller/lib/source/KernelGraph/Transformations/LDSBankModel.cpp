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

#include <fmt/format.h>
#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/Transforms/LDSBankModel.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <sstream>

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
        return event.direction == Direction::LDSLoad || event.direction == Direction::LDSStore;
    }

    void LDSBankModel::simulate(MemoryEventSimulated event)
    {
        AssertFatal(event.bytesRequested == 4,
                    "MemoryTracer currently only supports 4-byte accesses");

        auto ldsAddressInBytes = event.byteOffset;
        auto bankIndex         = (ldsAddressInBytes / m_entryWidthInBytes) % m_numBanks;

        auto ldsTag
            = event.direction == Direction::LDSStore ? event.destinationTag : event.sourceTag;

        MemoryInstruction instruction{
            event.direction, 1, DataType::UInt32, {static_cast<uint32_t>(ldsAddressInBytes)}};

        m_bankAccesses[event.operationTag].push_back(LDSBankAccess{
            event.operationTag, ldsTag, event.direction, event.workItem, bankIndex, {instruction}});
    }

    Summary LDSBankModel::summary() const
    {
        Summary summary;

        for(auto const& [tag, accesses] : m_bankAccesses)
        {
            auto ldsTag = accesses[0].ldsTag;

            std::map<uint, std::unordered_set<uint>> bankWorkitems;
            for(auto access : accesses)
            {
                bankWorkitems[access.bankIndex].insert(access.workitem);
            }

            uint minWorkitemsPerBank = 0;
            for(int bankIndex = 0; bankIndex < m_numBanks; ++bankIndex)
            {
                if(bankWorkitems.contains(bankIndex))
                    minWorkitemsPerBank = std::min(
                        minWorkitemsPerBank, static_cast<uint>(bankWorkitems[bankIndex].size()));
            }

            bool anyImbalance = false;
            for(auto const& [bankIndex, workitems] : bankWorkitems)
                anyImbalance |= workitems.size() > minWorkitemsPerBank;

            if(anyImbalance)
                summary.imbalancedTags.insert(tag);

            const auto workitemsInfo = [&]() {
                decltype(Summary::Access::accessedBanks) workitemsInfo;
                for(auto const& [bankIndex, workitems] : bankWorkitems)
                {
                    auto imbalanced = workitems.size() > minWorkitemsPerBank;
                    workitemsInfo.emplace_back(bankIndex, workitems.size(), imbalanced);
                }
                return workitemsInfo;
            }();

            std::vector<std::vector<int>> banksToWorkitems;
            if constexpr(Summary::echoBanks)
            {
                const auto maxWorkitems = 256;
                for(int bankIndex = 0; bankIndex < m_numBanks; ++bankIndex)
                {
                    if(bankWorkitems.contains(bankIndex))
                    {
                        banksToWorkitems.emplace_back([&]() {
                            std::vector<int> workitems;
                            for(int workitem = 0; workitem < maxWorkitems; ++workitem)
                            {
                                if(bankWorkitems[bankIndex].contains(workitem))
                                {
                                    workitems.emplace_back(workitem);
                                }
                            }
                            return workitems;
                        }());
                    }
                    else
                    {
                        banksToWorkitems.emplace_back();
                    }
                }
            }

            summary.accesses.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(tag),
                std::forward_as_tuple(ldsTag, workitemsInfo, banksToWorkitems));
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

    std::ostream& operator<<(std::ostream& stream, LDSBankModel const& ldsBankModel)
    {
        return stream << ldsBankModel.toString();
    }
}
