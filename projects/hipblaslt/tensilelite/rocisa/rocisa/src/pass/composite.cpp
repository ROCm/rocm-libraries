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

#include "instruction/common.hpp"
#include "pass.hpp"

namespace rocisa
{
    template <typename T>
    inline void compositeToInstructionTemplate(std::shared_ptr<T>& module)
    {
        std::vector<std::shared_ptr<Item>> itemList;
        for(auto& item : module->itemList)
        {
            if(auto compositeInstruction = dynamic_cast<CompositeInstruction*>(item.get()))
            {
                auto items = compositeInstruction->getInstructions();
                itemList.insert(itemList.end(),
                                std::make_move_iterator(items.begin()),
                                std::make_move_iterator(items.end()));
                continue; // Skip appending composite instruction back to list
            }
            else if(auto subModule = std::dynamic_pointer_cast<Module>(item))
            {
                compositeToInstructionTemplate<Module>(subModule);
            }
            else if(auto macro = std::dynamic_pointer_cast<Macro>(item))
            {
                compositeToInstructionTemplate<Macro>(macro);
            }
            itemList.push_back(std::move(item));
        }
        module->itemList = std::move(itemList);
    }

    void compositeToInstruction(std::shared_ptr<Module>& module)
    {
        compositeToInstructionTemplate<Module>(module);
    }
} // namespace rocisa
