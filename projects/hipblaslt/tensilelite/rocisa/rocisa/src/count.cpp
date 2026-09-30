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

    int countInstruction(const std::shared_ptr<Item>& item)
    {
        return countX<Instruction>(item);
    }

    int countGlobalRead(const std::shared_ptr<Item>& item)
    {
        return countX<GlobalReadInstruction>(item);
    }

    int countSMemLoad(const std::shared_ptr<Item>& item)
    {
        return countX<SMemLoadInstruction>(item);
    }

    int countLocalRead(const std::shared_ptr<Item>& item)
    {
        return countX<LocalReadInstruction>(item);
    }

    int countLocalWrite(const std::shared_ptr<Item>& item)
    {
        return countX<LocalWriteInstruction>(item);
    }

    // "countWeighted" functions apply type-based weights to count instructions accurately.
    int countWeightedLocalRead(const std::shared_ptr<Item>& item)
    {
        return countX<LocalReadInstruction>(item, {
            {typeid(DSLoadB192), 2}, // DSLoadB192 is composed by two instructions
        });
    }

    int countWeightedLocalWrite(const std::shared_ptr<Item>& item)
    {
        return countX<LocalWriteInstruction>(item, {
            {typeid(DSStoreB192), 2}, // DSStoreB192 is composed by two instructions
            {typeid(DSStoreB256), 2}, // DSStoreB256 is composed by two instructions
        });
    }

    // Exact types
    int countDSStoreB128(const std::shared_ptr<Item>& item)
    {
        return item->countExactType(typeid(DSStoreB128));
    }

    int countDSStoreB192(const std::shared_ptr<Item>& item)
    {
        return item->countExactType(typeid(DSStoreB192));
    }

    int countDSStoreB256(const std::shared_ptr<Item>& item)
    {
        return item->countExactType(typeid(DSStoreB256));
    }

    int countVMovB32(const std::shared_ptr<Item>& item)
    {
        return item->countExactType(typeid(VMovB32));
    }

    // Counts all MFMA-family instructions (MFMA, SMFMA, MXMFMA) in the item tree
    // using exact typeid matching.
    int countMFMA(const std::shared_ptr<Item>& item)
    {
        if(auto module = std::dynamic_pointer_cast<Module>(item))
        {
            int count = 0;
            for(const auto& i : module->itemList)
            {
                count += countMFMA(i);
            }
            return count;
        }

        const auto& tid = typeid(*item);
        if(tid == typeid(MFMAInstruction) || tid == typeid(SMFMAInstruction)
           || tid == typeid(MXMFMAInstruction))
        {
            return 1;
        }
        return 0;
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
        else if(std::dynamic_pointer_cast<MFMAInstruction>(item)
                || std::dynamic_pointer_cast<SMFMAInstruction>(item)
                || std::dynamic_pointer_cast<MXMFMAInstruction>(item))
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
