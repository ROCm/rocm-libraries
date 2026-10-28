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
#include "stinkytofu/core/Function.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>
#include <unordered_map>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmPrinter.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
Function::Function(const std::string& name)
    : name(name), ssaArena_(std::make_unique<SSAArena>(this)), basicBlocks(this) {}

Function::~Function() {
    clearAttachedSSA();
}

void Function::clear() {
    clearAttachedSSA();
    basicBlocks.clear();
}

bool Function::copyFrom(const Function& other) {
    if (&other == this) return true;
    clear();
    name = other.name;
    isCallable = other.isCallable;
    gemmConfig = other.gemmConfig;
    metadata_ = other.metadata_;

    bool complete = true;
    std::unordered_map<const BasicBlock*, BasicBlock*> copies;
    for (const BasicBlock& block : other) {
        BasicBlock* copy = createBasicBlock(block.getLabel());
        copies.emplace(&block, copy);
        for (const IRBase& ir : block) {
            if (IRBase* cloned = ir.clone())
                copy->appendIR(cloned);
            else
                complete = false;
        }
    }
    for (const BasicBlock& block : other) {
        BasicBlock* copy = copies.at(&block);
        for (const BasicBlock* succ : block.getSuccessors()) copy->addSuccessor(copies.at(succ));
        for (const BasicBlock* pred : block.getPredecessors())
            copy->addPredecessor(copies.at(pred));
    }
    return complete;
}

void Function::eraseBlock(BasicBlock& block) {
    clearAttachedSSA();
    const auto after = std::next(BasicBlockList::iterator(&block));
    BasicBlock* next = after == basicBlocks.end() ? nullptr : &*after;
    for (BasicBlock* pred : block.getPredecessors()) {
        if (next == nullptr || pred == &block || pred == next) continue;
        std::vector<BasicBlock*>& succs = pred->getSuccessors();
        if (std::find(succs.begin(), succs.end(), next) != succs.end()) continue;
        const auto edge = std::find(succs.begin(), succs.end(), &block);
        if (edge == succs.end()) continue;
        *edge = next;
        next->addPredecessor(pred);
    }
    removePredecessorEdges(block);
    removeSuccessorEdges(block);
    block.erase();
}

SSAArena& Function::ssaArena() {
    return *ssaArena_;
}

const SSAArena& Function::ssaArena() const {
    return *ssaArena_;
}

bool Function::hasAttachedSSA() const {
    for (const BasicBlock& bb : *this) {
        if (bb.hasSSAArguments()) return true;
        for (const IRBase& ir : bb) {
            const auto* inst = dyn_cast<StinkyInstruction>(&ir);
            if (inst != nullptr && inst->hasAttachedSSA()) return true;
        }
    }
    return false;
}

void Function::clearAttachedSSA() {
    for (BasicBlock& bb : *this) {
        bb.clearSSAArguments();
        for (IRBase& ir : bb) {
            if (auto* inst = dyn_cast<StinkyInstruction>(&ir)) inst->clearAttachedSSA();
        }
    }
    if (ssaArena_) ssaArena_->clear();
}

void Function::dump(std::ostream& out) const {
    AsmPrinter printer(out, AsmPrinterOptions());
    printer.print(*this);
}

void Function::dump() const {
    dump(std::cerr);
}
}  // namespace stinkytofu
