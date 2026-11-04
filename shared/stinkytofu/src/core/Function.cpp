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

#include <iostream>
#include <ostream>

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
