/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "code.hpp"
#include "instruction/common.hpp"
#include "instruction/mem.hpp"
#include "instruction/mfma.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>
#include <typeindex>

namespace nb = nanobind;

namespace rocisa
{
    // This function can be used for prototyping in Python, but it's slower
    // e.g. counType(item, Instruction)
    int countType(const std::shared_ptr<Item>& Item, nb::object& obj)
    {
        return Item->countType(obj);
    }

    // Use typeid for exact match, use dynamic_pointer_cast for inheritance match
    template <typename T>
    int countX(const std::shared_ptr<Item>& item,
                const std::unordered_map<std::type_index, int>& weights = {})
    {
        if(auto ptr = std::dynamic_pointer_cast<Module>(item))
        {
            int count = 0;
            for(const auto& i : ptr->itemList)
            {
                count += countX<T>(i, weights);
            }
            return count;
        }

        if (!std::dynamic_pointer_cast<const T>(item)) {
            return 0;
        }

        if (!weights.empty()) {
            const auto& itemRef = *item;
            auto it = weights.find(typeid(itemRef));
            if (it != weights.end())
                return it->second;
        }
        return 1;
    }

    // Scheduler-introspection seam (logical-IR migration, Phase 1):
    // the in-path scheduler counts below ask each leaf its SchedCategory /
    // SchedExactKind / schedWeight instead of dynamic_cast/typeid, so a logical
    // leaf (Phase 2) is counted the same way the rocisa instruction it replaces
    // is. rocisa reports the exact classification today's casts produce, so
    // these stay byte-identical while rocisa is still the only thing emitted.

    // Sum over schedulable leaves in the tree whose category == cat.
    int countCategory(const std::shared_ptr<Item>& item, SchedCategory cat)
    {
        if(auto ptr = std::dynamic_pointer_cast<Module>(item))
        {
            int count = 0;
            for(const auto& i : ptr->itemList)
                count += countCategory(i, cat);
            return count;
        }
        return item->schedCategory() == cat ? 1 : 0;
    }

    // Like countCategory, but each matching leaf contributes schedWeight()
    // (e.g. ds_*_b192 counts as two physical ops).
    int countWeightedCategory(const std::shared_ptr<Item>& item, SchedCategory cat)
    {
        if(auto ptr = std::dynamic_pointer_cast<Module>(item))
        {
            int count = 0;
            for(const auto& i : ptr->itemList)
                count += countWeightedCategory(i, cat);
            return count;
        }
        return item->schedCategory() == cat ? item->schedWeight() : 0;
    }

    // Sum over schedulable leaves whose exact kind == kind.
    int countExactKind(const std::shared_ptr<Item>& item, SchedExactKind kind)
    {
        if(auto ptr = std::dynamic_pointer_cast<Module>(item))
        {
            int count = 0;
            for(const auto& i : ptr->itemList)
                count += countExactKind(i, kind);
            return count;
        }
        return item->schedExactKind() == kind ? 1 : 0;
    }

    int countInstruction(const std::shared_ptr<Item>& item)
    {
        if(auto ptr = std::dynamic_pointer_cast<Module>(item))
        {
            int count = 0;
            for(const auto& i : ptr->itemList)
                count += countInstruction(i);
            return count;
        }
        // Every rocisa Instruction reports a non-None category (Other at least);
        // Module / TextBlock report None -> not counted, as before.
        return item->schedCategory() != SchedCategory::None ? 1 : 0;
    }

    int countGlobalRead(const std::shared_ptr<Item>& item)
    {
        return countCategory(item, SchedCategory::GlobalRead);
    }

    // countSMemLoad stays type-based: SMemLoad is a GlobalRead subtree, so it has
    // category GlobalRead and cannot be distinguished by category alone. It is
    // not called from the in-path scheduler, so it is left on countX for now.
    int countSMemLoad(const std::shared_ptr<Item>& item)
    {
        return countX<SMemLoadInstruction>(item);
    }

    int countLocalRead(const std::shared_ptr<Item>& item)
    {
        return countCategory(item, SchedCategory::LocalRead);
    }

    int countLocalWrite(const std::shared_ptr<Item>& item)
    {
        return countCategory(item, SchedCategory::LocalWrite);
    }

    // "countWeighted" functions apply per-op weights to count instructions accurately.
    int countWeightedLocalRead(const std::shared_ptr<Item>& item)
    {
        return countWeightedCategory(item, SchedCategory::LocalRead);
    }

    int countWeightedLocalWrite(const std::shared_ptr<Item>& item)
    {
        return countWeightedCategory(item, SchedCategory::LocalWrite);
    }

    // Exact kinds
    int countDSStoreB128(const std::shared_ptr<Item>& item)
    {
        return countExactKind(item, SchedExactKind::DSStoreB128);
    }

    int countDSStoreB192(const std::shared_ptr<Item>& item)
    {
        return countExactKind(item, SchedExactKind::DSStoreB192);
    }

    // countDSStoreB256 is not called from the in-path scheduler; left type-based.
    int countDSStoreB256(const std::shared_ptr<Item>& item)
    {
        return item->countExactType(typeid(DSStoreB256));
    }

    int countVMovB32(const std::shared_ptr<Item>& item)
    {
        return countExactKind(item, SchedExactKind::VMovB32);
    }

    // Counts all MFMA-family instructions (MFMA, SMFMA, MXMFMA) in the item tree.
    int countMFMA(const std::shared_ptr<Item>& item)
    {
        return countCategory(item, SchedCategory::MFMA);
    }

    // Helper functions
    std::vector<std::shared_ptr<Item>> getMFMAs(const std::shared_ptr<Item>& item)
    {
        std::vector<std::shared_ptr<Item>> mfmaList;
        if(auto module = std::dynamic_pointer_cast<Module>(item))
        {
            for(const auto& i : module->itemList)
            {
                auto mfm = getMFMAs(i);
                mfmaList.insert(mfmaList.end(), mfm.begin(), mfm.end());
            }
        }
        else if(item->schedCategory() == SchedCategory::MFMA)
        {
            mfmaList.push_back(item);
        }
        return std::move(mfmaList);
    }

    // Recursively find the index (count) of targetItem in the module tree.
    // Returns a pair: {count, found}
    std::pair<int, bool> findInstCount(const std::shared_ptr<Item>& module,
                                       const std::shared_ptr<Item>& targetItem,
                                       int                          count)
    {
        if(auto mod = std::dynamic_pointer_cast<Module>(module))
        {
            for(const auto& inst : mod->itemList)
            {
                if(auto submod = std::dynamic_pointer_cast<Module>(inst))
                {
                    auto [subCount, found] = findInstCount(submod, targetItem, count);
                    if(found)
                        return {subCount, true};
                    count = subCount;
                }
                else if(inst == targetItem)
                {
                    return {count, true};
                }
                else if(!std::dynamic_pointer_cast<TextBlock>(inst))
                {
                    count += 1;
                }
            }
        }
        return {count, false};
    }
}

void init_count(nb::module_ m)
{
    m.def("countType", &rocisa::countType, "A Python style API for fast prototyping.");

    m.def("countInstruction", &rocisa::countInstruction);
    m.def("countGlobalRead", &rocisa::countGlobalRead);
    m.def("countSMemLoad", &rocisa::countSMemLoad);
    m.def("countLocalRead", &rocisa::countLocalRead);
    m.def("countLocalWrite", &rocisa::countLocalWrite);
    m.def("countWeightedLocalRead", &rocisa::countWeightedLocalRead);
    m.def("countWeightedLocalWrite", &rocisa::countWeightedLocalWrite);

    m.def("countDSStoreB128", &rocisa::countDSStoreB128);
    m.def("countDSStoreB192", &rocisa::countDSStoreB192);
    m.def("countDSStoreB256", &rocisa::countDSStoreB256);
    m.def("countVMovB32", &rocisa::countVMovB32);
    m.def("countMFMA",
          &rocisa::countMFMA,
          "Count all MFMA-family instructions (MFMA, SMFMA, MXMFMA) in the item tree.");

    m.def("getMFMAs", &rocisa::getMFMAs, "Get all MFMA instructions in the item tree.");
    m.def(
        "findInstCount",
        &rocisa::findInstCount,
        "Find the index (count) of targetItem in the module tree. Returns a pair: {count, found}");
}
