/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
    int countX(const std::shared_ptr<Item>& item)
    {
        if(auto ptr = std::dynamic_pointer_cast<Module>(item))
        {
            int count = 0;
            for(const auto& i : ptr->itemList)
            {
                count += countX<T>(i);
            }
            return count;
        }
        return static_cast<int>(std::dynamic_pointer_cast<const T>(item) != nullptr);
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

    // Exact types
    int countDSStoreB128(const std::shared_ptr<Item>& item)
    {
        return item->countExactType(typeid(DSStoreB128));
    }

    int countDSStoreB256(const std::shared_ptr<Item>& item)
    {
        return item->countExactType(typeid(DSStoreB256));
    }

    int countVMovB32(const std::shared_ptr<Item>& item)
    {
        return item->countExactType(typeid(VMovB32));
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
                || std::dynamic_pointer_cast<SMFMAInstruction>(item))
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

    m.def("countDSStoreB128", &rocisa::countDSStoreB128);
    m.def("countDSStoreB256", &rocisa::countDSStoreB256);
    m.def("countVMovB32", &rocisa::countVMovB32);

    m.def("getMFMAs", &rocisa::getMFMAs, "Get all MFMA instructions in the item tree.");
    m.def(
        "findInstCount",
        &rocisa::findInstCount,
        "Find the index (count) of targetItem in the module tree. Returns a pair: {count, found}");
}
