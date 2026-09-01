/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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
#include "stinkytofu/transforms/asm/ra/PhysRegMatrix.hpp"

#include <algorithm>
#include <cassert>

namespace stinkytofu {

PhysRegMatrix::PhysRegMatrix(const AsmTargetRegisters& target) : target_(&target) {
    // The target owns the class set; restating it here would silently skip a
    // class this architecture allows.
    for (RegType regClass : target.allocatableClasses()) {
        ClassUnits units;
        units.regClass = regClass;
        units.byIndex.resize(target.indexCount(regClass));
        classes_.push_back(std::move(units));
    }
}

const PhysRegMatrix::ClassUnits* PhysRegMatrix::unitsFor(RegType regClass) const {
    for (const ClassUnits& units : classes_) {
        if (units.regClass == regClass) return &units;
    }
    return nullptr;
}

PhysRegMatrix::ClassUnits* PhysRegMatrix::unitsFor(RegType regClass) {
    for (ClassUnits& units : classes_) {
        if (units.regClass == regClass) return &units;
    }
    return nullptr;
}

bool PhysRegMatrix::available(RegType regClass, uint32_t idx, const LiveRange& range) const {
    if (!target_->isAllocatable(regClass, idx)) return false;

    const ClassUnits* units = unitsFor(regClass);
    if (units == nullptr || idx >= units->byIndex.size()) return false;

    for (const Binding& bound : units->byIndex[idx]) {
        if (bound.range != nullptr && bound.range->overlaps(range)) return false;
    }
    return true;
}

bool PhysRegMatrix::runAvailable(RegType regClass, uint32_t base, uint32_t width,
                                 const LiveRange& range) const {
    if (width == 0) return false;

    const uint32_t count = target_->indexCount(regClass);
    // Guards the run leaving the class, and the addition overflowing.
    if (base >= count || width > count - base) return false;

    for (uint32_t offset = 0; offset < width; ++offset) {
        if (!available(regClass, base + offset, range)) return false;
    }
    return true;
}

std::optional<uint32_t> PhysRegMatrix::findFreeRun(RegType regClass, uint32_t width,
                                                   const LiveRange& range) const {
    if (width == 0) return std::nullopt;

    const uint32_t count = target_->indexCount(regClass);
    if (width > count) return std::nullopt;

    for (uint32_t base = 0; base + width <= count; ++base) {
        if (runAvailable(regClass, base, width, range)) return base;
    }
    return std::nullopt;
}

void PhysRegMatrix::collectConflicts(RegType regClass, uint32_t idx, const LiveRange& range,
                                     std::vector<SSAValueID>& out) const {
    const ClassUnits* units = unitsFor(regClass);
    if (units == nullptr || idx >= units->byIndex.size()) return;

    for (const Binding& bound : units->byIndex[idx]) {
        if (bound.range != nullptr && bound.range->overlaps(range)) out.push_back(bound.value);
    }
}

void PhysRegMatrix::bind(RegType regClass, uint32_t idx, SSAValueID value, const LiveRange& range) {
    assert(target_->isAllocatable(regClass, idx) && "binding a register that is not allocatable");
    assert(value != kInvalidSSAValueID && "binding the invalid value ID");

    ClassUnits* units = unitsFor(regClass);
    if (units == nullptr || idx >= units->byIndex.size()) return;
    units->byIndex[idx].push_back({value, &range});
}

void PhysRegMatrix::unbind(RegType regClass, uint32_t idx, SSAValueID value) {
    ClassUnits* units = unitsFor(regClass);
    if (units == nullptr || idx >= units->byIndex.size()) return;

    std::vector<Binding>& bindings = units->byIndex[idx];
    bindings.erase(std::remove_if(bindings.begin(), bindings.end(),
                                  [value](const Binding& b) { return b.value == value; }),
                   bindings.end());
}

size_t PhysRegMatrix::bindingCount() const {
    size_t total = 0;
    for (const ClassUnits& units : classes_) {
        for (const std::vector<Binding>& bindings : units.byIndex) total += bindings.size();
    }
    return total;
}

std::optional<uint32_t> PhysRegMatrix::highestBound(RegType regClass) const {
    const ClassUnits* units = unitsFor(regClass);
    if (units == nullptr) return std::nullopt;

    for (size_t idx = units->byIndex.size(); idx > 0; --idx) {
        if (!units->byIndex[idx - 1].empty()) return static_cast<uint32_t>(idx - 1);
    }
    return std::nullopt;
}

std::string PhysRegMatrix::toString() const {
    std::string text;
    for (const ClassUnits& units : classes_) {
        const std::string prefix = regTypeToString(units.regClass);
        for (size_t idx = 0; idx < units.byIndex.size(); ++idx) {
            const std::vector<Binding>& bindings = units.byIndex[idx];
            if (bindings.empty()) continue;
            text += prefix + std::to_string(idx) + ":";
            for (const Binding& bound : bindings) {
                text += " %" + std::to_string(bound.value);
                if (bound.range != nullptr) text += bound.range->toString();
            }
            text += "\n";
        }
    }
    return text.empty() ? "empty\n" : text;
}

}  // namespace stinkytofu
