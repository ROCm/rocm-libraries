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

namespace rocisa
{
    std::vector<std::shared_ptr<Item>>
        cloneItemList(const std::vector<std::shared_ptr<Item>>& itemList)
    {
        std::vector<std::shared_ptr<Item>> clonedItemList;
        for(const auto& item : itemList)
        {
            std::shared_ptr<Item> cloned;
            if(typeid(item.get()) == typeid(Module))
            {
                cloned = std::make_shared<Module>(*dynamic_cast<Module*>(item.get()));
            }
            else
                cloned = item->clone();
            clonedItemList.push_back(cloned);
        }
        return std::move(clonedItemList);
    }

    std::shared_ptr<BitfieldUnion> SrdUpperValue(const IsaVersion& isa)
    {
        if(isa[0] == 12)
        {
            if(isa[1] == 5)
                return std::make_shared<SrdUpperValue125X>(SrdUpperValue125X::staticInit());
            else
                return std::make_shared<SrdUpperValue12XX>(SrdUpperValue12XX::staticInit());
        }
        else if(isa[0] == 11)
        {
            return std::make_shared<SrdUpperValue11XX>(SrdUpperValue11XX::staticInit());
        }
        else if(isa[0] == 10)
        {
            return std::make_shared<SrdUpperValue10XX>(SrdUpperValue10XX::staticInit());
        }
        else
        {
            return std::make_shared<SrdUpperValue9XX>(SrdUpperValue9XX::staticInit());
        }
    }

    const std::unordered_map<std::string, int> SignatureArgument::ValueTypeSizeDict
        = {{"i8", 1},
           {"i16", 2},
           {"i32", 4},
           {"i64", 8},
           {"u8", 1},
           {"u16", 2},
           {"u32", 4},
           {"u64", 8},
           {"bf16", 2},
           {"f16", 2},
           {"f32", 4},
           {"f32c", 8},
           {"f64", 8},
           {"f64c", 16},
           {"pkf16", 4},
           {"struct", 8}};
} // namespace rocisa
