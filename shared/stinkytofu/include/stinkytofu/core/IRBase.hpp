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

#include <iosfwd>

#include "stinkytofu/support/IntrusiveList.hpp"

namespace stinkytofu {
class BasicBlock;

/// IRBase is in an IntrusiveList<IRBase, BasicBlock>; getParent() comes from IntrusiveListNode and
/// returns the owning BasicBlock.
class IRBase : public IntrusiveListNode<IRBase, BasicBlock> {
   public:
    // Stinky framework could support multiple IR types in the future,
    // conceptually similar to MLIR but in much simpler framework.
    enum class IRType {
        StinkyTofu,          // Assembly-level IR (StinkyInstruction)
        StinkyAsmDirective,  // Assembly directives
        LogicalIR,           // High-level, architecture-independent IR (LogicalInstruction)
    };

    IRBase(const IRBase&) = delete;
    IRBase& operator=(const IRBase&) = delete;

    /// Clone this IR (caller must add to a BasicBlock via Function::cloneIR or similar).
    /// Default returns nullptr; override in derived types that support cloning.
    virtual IRBase* clone() const {
        return nullptr;
    }

    const BasicBlock* getParentBlock() const;

    virtual void dump(std::ostream& out) const = 0;

    void dump();

    IRType getType() const {
        return irType;
    }

    /// Create IR (caller owns; not in any BasicBlock). Add to a block with bb.appendIR(ir).
    template <class IRType, class... Args>
    static IRType* createIR(Args&&... args) {
        return new IRType(std::forward<Args>(args)...);
    }

    /// Parent block that owns this IR (from IntrusiveListNode); null if not in any block.
    using IntrusiveListNode<IRBase, BasicBlock>::getParent;

    /// Remove this IR from its parent block and delete it (list owns deletion via traits).
    void erase();

    /// Remove the IR from its parent block, but don't delete it.
    void remove();

   protected:
    virtual ~IRBase() = default;
    IRBase(IRType type) : irType(type) {}

   private:
    const IRType irType;

    friend struct IntrusiveListTraits<IRBase>;
    friend struct IntrusiveListAllocTraits<IRBase>;
};

using IRList = IntrusiveList<IRBase, BasicBlock>;
}  // namespace stinkytofu
