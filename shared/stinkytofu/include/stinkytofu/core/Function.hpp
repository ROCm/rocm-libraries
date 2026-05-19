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

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <unordered_map>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Types.hpp"

namespace stinkytofu {
// Function holds a list of BasicBlocks.
//
// This represents a function/kernel in the StinkyTofu IR.
// BasicBlocks are organized as an intrusive list and can be
// traversed in program order.
class STINKYTOFU_EXPORT Function {
   private:
    std::string name;
    BasicBlockList basicBlocks;  // List parent is this so BasicBlock::getParent() works
    GemmTileConfig gemmConfig;
    std::unordered_map<std::string, uint64_t> metadata_;

   public:
    explicit Function(const std::string& name = "") : name(name), basicBlocks(this) {}

    Function(const Function&) = delete;
    Function& operator=(const Function&) = delete;

    ~Function() = default;

    const std::string& getName() const {
        return name;
    }

    void setName(const std::string& name) {
        this->name = name;
    }

    // BasicBlock management
    BasicBlock* createBasicBlock(const std::string& label = "") {
        BasicBlock* bb = new BasicBlock(label);
        basicBlocks.push_back(bb);
        return bb;
    }

    BasicBlock* createBasicBlockBefore(BasicBlock* before, const std::string& label = "") {
        BasicBlock* bb = new BasicBlock(label);
        basicBlocks.insert(BasicBlockList::iterator(before), bb);
        return bb;
    }

    /// Clone IR and append to the given BasicBlock. Ownership is with the Function.
    /// Returns nullptr if the IR type does not support cloning.
    IRBase* cloneIR(IRBase* ir, BasicBlock& bb) {
        IRBase* c = ir->clone();
        if (c) bb.appendIR(c);
        return c;
    }

    /// Add CFG edge from \p from to \p to (updates both successor and predecessor).
    void addEdge(BasicBlock* from, BasicBlock* to) {
        from->addSuccessor(to);
        to->addPredecessor(from);
    }

    /// Remove BasicBlock from this function (block must be in this function). Does not delete the
    /// block.
    void removeBasicBlock(BasicBlock* bb) {
        basicBlocks.remove(bb);
    }

    /// Erase one BasicBlock at position (list traits delete the block). Caller must not use the
    /// iterator after erase.
    BasicBlockList::iterator eraseBasicBlock(BasicBlockList::iterator pos) {
        return basicBlocks.erase(pos);
    }

    /// Entry block is the first in the list (basicBlocks.front()).
    BasicBlock* getEntryBlock() {
        return basicBlocks.empty() ? nullptr : &basicBlocks.front();
    }
    const BasicBlock* getEntryBlock() const {
        return basicBlocks.empty() ? nullptr : &basicBlocks.front();
    }

    // GEMM tile configuration
    void setGemmTileConfig(const GemmTileConfig& config) {
        gemmConfig = config;
    }
    const GemmTileConfig& getGemmTileConfig() const {
        return gemmConfig;
    }

    // Function metadata
    void setMetaData(const std::string& key, uint64_t value) {
        metadata_[key] = value;
    }
    std::optional<uint64_t> getMetaData(const std::string& key) const {
        auto it = metadata_.find(key);
        if (it == metadata_.end()) return std::nullopt;
        return it->second;
    }
    bool hasMetaData(const std::string& key) const {
        return metadata_.find(key) != metadata_.end();
    }

    // Iteration over basic blocks
    auto begin() {
        return basicBlocks.begin();
    }
    auto end() {
        return basicBlocks.end();
    }
    auto begin() const {
        return basicBlocks.begin();
    }
    auto end() const {
        return basicBlocks.end();
    }

    bool empty() const {
        return basicBlocks.empty();
    }
    size_t size() const {
        return basicBlocks.size();
    }

    /// Delete all BasicBlocks and their IR (list traits delete each block and its IR).
    void clear() {
        basicBlocks.clear();
    }

    void dump(std::ostream& out) const;

    void dump() const;
};
}  // namespace stinkytofu
