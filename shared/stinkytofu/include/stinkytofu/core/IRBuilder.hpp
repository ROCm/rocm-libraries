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
#pragma once

#include <cassert>
#include <iterator>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/IRBase.hpp"

namespace stinkytofu {
/// Base class for constructing IR. Holds a BasicBlock and an insertion point.
/// Derived builders (e.g. AsmIRBuilder) use createIR() and provide
/// type-specific create methods. IR is owned by the BasicBlock's IRList.
class IRBuilder {
   protected:
    BasicBlock* bb = nullptr;
    IRList::iterator insertPoint;

   public:
    explicit IRBuilder(BasicBlock& block) : bb(&block), insertPoint(block.end()) {}

    virtual ~IRBuilder() = default;

    /// Create IR and insert at current insertion point.
    template <class IRType, class... Args>
    IRType* createIR(Args&&... args) {
        IRType* ir = IRBase::createIR<IRType>(std::forward<Args>(args)...);
        bb->insertIR(insertPoint, ir);
        return ir;
    }

    /// Create IR and insert before insertBefore.
    /// If insertBefore is nullptr, use current insertion point.
    template <class IRType, class... Args>
    IRType* createIR(IRBase* insertBefore, Args&&... args) {
        assert(insertBefore != nullptr &&
               "Cannot create instruction with null insertBefore - insertBefore must be "
               "non-null");
        assert(insertBefore->getParent() == bb &&
               "Cannot create instruction with insertBefore from a different block");
        IRType* ir = IRBase::createIR<IRType>(std::forward<Args>(args)...);
        bb->insertIR(IRList::iterator(insertBefore), ir);
        return ir;
    }

    /// Set insertion point to the end of the block.
    void setInsertionPoint(BasicBlock& newBB) {
        bb = &newBB;
        insertPoint = newBB.end();
    }

    /// Set insertion point to a specific position.
    void setInsertionPoint(IRList::iterator it) {
        if (it != bb->end()) bb = it->getParent();
        insertPoint = it;
    }
};
}  // namespace stinkytofu
