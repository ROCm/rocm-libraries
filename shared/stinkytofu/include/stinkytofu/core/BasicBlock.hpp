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

#include <algorithm>
#include <iosfwd>
#include <string>
#include <vector>

#include "stinkytofu/core/IRBase.hpp"

namespace stinkytofu {
// Forward declarations for BasicBlock and Function
class Function;

// BasicBlock is in an IntrusiveList<BasicBlock, Function>; getParent() comes from IntrusiveListNode
// and returns the owning Function. BasicBlock holds a list of IR and CFG edges.
class BasicBlock : public IntrusiveListNode<BasicBlock, Function> {
   private:
    std::string label;  // Optional label for the block
    IRList ir;          // List owns IR; parent is this block so IRBase::getParent() works
    std::vector<BasicBlock*> predecessors;
    std::vector<BasicBlock*> successors;

   public:
    explicit BasicBlock(const std::string& label = "")
        : label(label)
          // This is safe because IntrusiveList constructor just stores the "this" pointer.
          ,
          ir(this) {}

    BasicBlock(const BasicBlock&) = delete;
    BasicBlock& operator=(const BasicBlock&) = delete;

    /// Parent function that owns this block (from IntrusiveListNode); null if not in any function.
    using IntrusiveListNode<BasicBlock, Function>::getParent;

    const Function* getParentFunc() const;

    const std::string& getLabel() const {
        return label;
    }

    using iterator = IRList::iterator;
    using const_iterator = IRList::const_iterator;
    using reverse_iterator = IRList::reverse_iterator;
    using const_reverse_iterator = IRList::const_reverse_iterator;

    iterator begin() {
        return ir.begin();
    }
    iterator end() {
        return ir.end();
    }
    const_iterator begin() const {
        return ir.begin();
    }
    const_iterator end() const {
        return ir.end();
    }
    reverse_iterator rbegin() {
        return ir.rbegin();
    }
    reverse_iterator rend() {
        return ir.rend();
    }
    const_reverse_iterator rbegin() const {
        return ir.rbegin();
    }
    const_reverse_iterator rend() const {
        return ir.rend();
    }

    /// Append IR to this block. Ownership is with the Function; parent is set by the list.
    void appendIR(IRBase* node) {
        ir.push_back(node);
    }

    // CFG navigation
    void addSuccessor(BasicBlock* bb) {
        successors.push_back(bb);
    }

    void addPredecessor(BasicBlock* bb) {
        predecessors.push_back(bb);
    }

    void removeSuccessor(BasicBlock* bb) {
        successors.erase(std::remove(successors.begin(), successors.end(), bb), successors.end());
    }

    void removePredecessor(BasicBlock* bb) {
        predecessors.erase(std::remove(predecessors.begin(), predecessors.end(), bb),
                           predecessors.end());
    }

    const std::vector<BasicBlock*>& getSuccessors() const {
        return successors;
    }
    const std::vector<BasicBlock*>& getPredecessors() const {
        return predecessors;
    }

    std::vector<BasicBlock*>& getSuccessors() {
        return successors;
    }
    std::vector<BasicBlock*>& getPredecessors() {
        return predecessors;
    }

    friend struct IntrusiveListTraits<BasicBlock>;

    // Utilities
    IRBase* getTerminator() {
        if (ir.empty()) return nullptr;
        return &ir.back();
    }

    const IRBase* getTerminator() const {
        if (ir.empty()) return nullptr;
        return &ir.back();
    }

    bool empty() const {
        return ir.empty();
    }
    size_t size() const {
        return ir.size();
    }

    /// Insert IR at position (IR must be created via IRBase::createIR or similar).
    IRList::iterator insertIR(IRList::iterator pos, IRBase* node) {
        return ir.insert(pos, node);
    }

    /// Erase one IR node at position. Caller must not use the iterator after erase.
    IRList::iterator eraseIR(IRList::iterator pos) {
        return ir.erase(pos);
    }

    /// Remove IR node from this block (node must be in this block).
    void removeIR(IRBase* node) {
        ir.remove(node);
    }

    /// Remove this block from its parent function and delete it. Prefer this over raw delete.
    void erase();

    /// Remove the block from its parent function, but don't delete it.
    void remove();

    /// Clear all IR in this BasicBlock (removes and deletes via list traits).
    void clear() {
        ir.clear();
    }

    void dump() const;

    friend class Function;
};

using BasicBlockList = IntrusiveList<BasicBlock, Function>;
}  // namespace stinkytofu
