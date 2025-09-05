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

    uint LDSBankModel::getThreadsPerClock(const MemoryOpLDS& memoryOp,
                                          uint               dwords,
                                          GPUArchitectureGFX gfx)
    {
        if(gfx == GPUArchitectureGFX::GFX950 && memoryOp.direction == Direction::Load)
        {
            switch(dwords)
            {
            case 1:
                return 32;
            case 2:
                return 32;
            case 3:
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

    uint LDSBankModel::calculateBankConflicts(const std::vector<uint32_t>& addresses,
                                              uint                         entryWidthInBytes,
                                              uint                         numBanks)
    {
        if(addresses.empty())
            return 0;

        std::unordered_map<uint, uint> bankCount;
        for(auto address : addresses)
        {
            auto bankIndex = (address / entryWidthInBytes) % numBanks;
            bankCount[bankIndex]++;
        }

        uint maxConflicts = 0;
        for(const auto& [bank, count] : bankCount)
        {
            maxConflicts = std::max(maxConflicts, count);
        }

        return maxConflicts;
    }

    uint LDSBankModel::calculateBankConflicts(const std::vector<uint32_t>& addresses) const
    {
        return calculateBankConflicts(addresses, m_entryWidthInBytes, m_numBanks);
    }

    DetailedSummary LDSBankModel::detailedSummary(GPUArchitectureGFX gfx) const
    {
        DetailedSummary detailedSummary;

        for(auto const& [operationTag, accesses] : m_bankAccesses)
        {
            DetailedSummary::OperationDetail detail;

            // Collect all unique memory instructions for this operation
            std::unordered_set<uint32_t> uniqueAddresses;
            for(auto const& access : accesses)
            {
                for(auto const& instruction : access.instructions)
                {
                    for(auto addr : instruction.addresses)
                    {
                        uniqueAddresses.insert(addr);
                    }
                }
            }

            // Determine the memory operation and data size
            if(!accesses.empty() && !accesses[0].instructions.empty())
            {
                const auto& firstInstruction = accesses[0].instructions[0];
                detail.instructions          = accesses[0].instructions;

                // Get threads per clock based on operation type
                if(std::holds_alternative<MemoryOpLDS>(firstInstruction.memoryOp))
                {
                    const auto& ldsOp = std::get<MemoryOpLDS>(firstInstruction.memoryOp);
                    detail.threadsPerClock
                        = getThreadsPerClock(ldsOp, firstInstruction.dwords, gfx);
                    detail.gfx = gfx;
                }
            }

            // Group workitems by thread groups
            std::map<uint, std::vector<uint>>     threadGroups;
            std::map<uint, std::vector<uint32_t>> threadGroupAddresses;

            for(auto const& access : accesses)
            {
                uint groupIndex = access.workitem / detail.threadsPerClock;
                threadGroups[groupIndex].push_back(access.workitem);

                for(auto const& instruction : access.instructions)
                {
                    for(auto addr : instruction.addresses)
                    {
                        threadGroupAddresses[groupIndex].push_back(addr);
                    }
                }
            }

            // Calculate conflicts for each thread group
            for(auto const& [groupIndex, workitems] : threadGroups)
            {
                DetailedSummary::ThreadGroupConflict conflict;
                conflict.threadGroupIndex = groupIndex;
                conflict.workitemIds      = workitems;

                // Calculate bank mapping for this thread group
                std::map<uint, std::vector<uint32_t>> bankToAddr;
                for(auto addr : threadGroupAddresses[groupIndex])
                {
                    uint bankIndex = (addr / m_entryWidthInBytes) % m_numBanks;
                    bankToAddr[bankIndex].push_back(addr);
                }
                conflict.bankToAddresses = bankToAddr;

                // Use calculateBankConflicts to find max conflict degree
                conflict.maxConflictDegree
                    = calculateBankConflicts(threadGroupAddresses[groupIndex]);

                detail.conflictsPerClock.push_back(conflict);
            }

            detailedSummary.operationDetails[operationTag] = detail;
        }

        return detailedSummary;
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

        for(auto const& [operationTag, detail] : operationDetails)
        {
            ss << fmt::format("Operation tag {}:\n", operationTag);

            // Print instructions
            for(size_t i = 0; i < detail.instructions.size(); ++i)
            {
                const auto& instruction = detail.instructions[i];
                const auto& memOp       = instruction.memoryOp;

                ss << fmt::format("\tInstruction {}: ", i);

                if(std::holds_alternative<MemoryOpLDS>(memOp))
                {
                    const auto& ldsOp = std::get<MemoryOpLDS>(memOp);
                    ss << fmt::format("LDS {} ({} dwords, {} threads/clock on {})\n",
                                      ldsOp.direction == Direction::Load ? "Load" : "Store",
                                      instruction.dwords,
                                      detail.threadsPerClock,
                                      rocRoller::toString(detail.gfx));
                }
                else
                {
                    ss << "Non-LDS operation\n";
                }
            }

            // Print thread group conflicts
            for(const auto& conflict : detail.conflictsPerClock)
            {
                ss << fmt::format("\t\tThread group {} (workitems: ", conflict.threadGroupIndex);

                // Print workitem IDs
                for(size_t i = 0; i < conflict.workitemIds.size(); ++i)
                {
                    if(i > 0)
                        ss << ", ";
                    ss << conflict.workitemIds[i];
                }
                ss << fmt::format(") - Max conflict degree: {}\n", conflict.maxConflictDegree);

                // Print bank conflicts details
                for(const auto& [bankIndex, addresses] : conflict.bankToAddresses)
                {
                    ss << fmt::format("\t\t\tBank {}: {} addresses [", bankIndex, addresses.size());

                    // Print first few addresses
                    size_t numToPrint = std::min(addresses.size(), size_t(16));
                    for(size_t i = 0; i < numToPrint; ++i)
                    {
                        if(i > 0)
                            ss << ", ";
                        ss << fmt::format("{}", addresses[i]);
                    }
                    if(addresses.size() > 16)
                    {
                        ss << ", ...";
                    }
                    ss << "]\n";
                }
            }
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
