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

#include "stinkytofu/bindings/python/LogicalModule.hpp"

#include <algorithm>
#include <iostream>

#include "stinkytofu/ir/logical/LogicalInstructions.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
struct PyLogicalModule::Impl {
    std::string name;
    std::vector<std::shared_ptr<LogicalInstruction>> instructions;

    Impl(const std::string& name) : name(name) {}

    ~Impl() {
        // shared_ptr will automatically handle cleanup
    }
};

PyLogicalModule::PyLogicalModule(const std::string& name) : pImpl(std::make_unique<Impl>(name)) {}

PyLogicalModule::~PyLogicalModule() = default;

PyLogicalModule::PyLogicalModule(PyLogicalModule&&) noexcept = default;
PyLogicalModule& PyLogicalModule::operator=(PyLogicalModule&&) noexcept = default;

std::string PyLogicalModule::getName() const {
    return pImpl->name;
}

std::shared_ptr<LogicalInstruction> PyLogicalModule::add(std::shared_ptr<LogicalInstruction> inst) {
    if (inst) {
        pImpl->instructions.push_back(inst);
    }
    return inst;
}

const std::vector<std::shared_ptr<LogicalInstruction>>& PyLogicalModule::getInstructions() const {
    return pImpl->instructions;
}

std::vector<std::shared_ptr<LogicalInstruction>>& PyLogicalModule::getMutableInstructions() {
    return pImpl->instructions;
}

bool PyLogicalModule::removeInstruction(LogicalInstruction* inst) {
    auto it = std::find_if(pImpl->instructions.begin(), pImpl->instructions.end(),
                           [inst](const auto& ptr) { return ptr.get() == inst; });
    if (it != pImpl->instructions.end()) {
        pImpl->instructions.erase(it);
        return true;
    }
    return false;
}

size_t PyLogicalModule::size() const {
    return pImpl->instructions.size();
}

void PyLogicalModule::dump(std::ostream& out) const {
    out << "LogicalModule: " << pImpl->name << "\n";
    out << "Instructions: " << pImpl->instructions.size() << "\n";
    for (size_t i = 0; i < pImpl->instructions.size(); ++i) {
        out << "  [" << i << "] ";
        pImpl->instructions[i]->dump(out);
        out << "\n";
    }
}

// -------------------------------------------------------------------------
// PyLogicalFunction
// -------------------------------------------------------------------------

PyLogicalFunction::PyLogicalFunction(Function* func) : func(func) {}

PyLogicalFunction::~PyLogicalFunction() {
    detachExternallyOwnedIRs();
}

void PyLogicalFunction::detachExternallyOwnedIRs() {
    if (!func) return;
    for (BasicBlock& bb : *func) {
        for (auto it = bb.begin(); it != bb.end(); /* no increment */) {
            IRBase* ir = it.getNodePtr();
            ++it;
            if (LogicalInstruction* li = dyn_cast<LogicalInstruction>(ir)) {
                if (li->ownedExternally) li->remove();
            }
        }
    }
}

}  // namespace stinkytofu
