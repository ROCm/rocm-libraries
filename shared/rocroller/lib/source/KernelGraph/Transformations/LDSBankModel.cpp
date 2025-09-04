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
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/LDSBankModel.hpp>
#include <rocRoller/KernelGraph/Transforms/MemoryTracer.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>
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
        // Check if the memory operation is for LDS
        return std::holds_alternative<MemoryOpLDS>(event.memoryOp);
    }

    void LDSBankModel::simulate(MemoryEventSimulated event)
    {
        AssertFatal(event.bytesRequested == 4,
                    "MemoryTracer currently only supports 4-byte accesses");

        auto ldsAddressInBytes = event.byteOffset;
        auto bankIndex         = (ldsAddressInBytes / m_entryWidthInBytes) % m_numBanks;

        // Get the LDS operation and check if it's a store
        const auto& ldsOp = std::get<MemoryOpLDS>(event.memoryOp);
        auto ldsTag = ldsOp.direction == Direction::Store ? event.destinationTag : event.sourceTag;

        MemoryInstruction instruction{
            event.memoryOp, 1, DataType::UInt32, {static_cast<uint32_t>(ldsAddressInBytes)}};

        m_bankAccesses[event.operationTag].push_back(LDSBankAccess{
            event.operationTag, ldsTag, ldsOp.direction, event.workItem, bankIndex, {instruction}});
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
