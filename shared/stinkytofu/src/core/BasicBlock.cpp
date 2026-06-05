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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "stinkytofu/core/BasicBlock.hpp"

#include <iostream>
#include <ostream>

#include "stinkytofu/core/Function.hpp"

namespace stinkytofu {
const Function* BasicBlock::getParentFunc() const {
    return getParent();
}

void BasicBlock::remove() {
    Function* p = getParent();
    if (p) p->removeBasicBlock(this);
}

void BasicBlock::erase() {
    Function* p = getParent();
    if (p)
        p->eraseBasicBlock(BasicBlockList::iterator(this));
    else
        delete this;
}

void BasicBlock::dump() const {
    if (!label.empty()) {
        std::cerr << label << ":\n";
    } else {
        std::cerr << "(unlabeled):";
    }

    for (const IRBase& irNode : ir) {
        std::cerr << "  ";
        irNode.dump(std::cerr);
    }

    if (!successors.empty()) {
        std::cerr << "  ; successors: ";
        for (size_t i = 0; i < successors.size(); ++i) {
            if (i > 0) std::cerr << ", ";
            if (!successors[i]->getLabel().empty())
                std::cerr << successors[i]->getLabel();
            else
                std::cerr << "<unlabeled>";
        }
        std::cerr << "\n";
    }
}
}  // namespace stinkytofu
