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
#include "stinkytofu/analysis/ssa/CanonicalSSAAllocation.hpp"

#include <cassert>

namespace stinkytofu {
namespace {

const RegKey kUnassigned{RegType::UNKNOWN, 0, RegHalf::NONE};

}  // namespace

AllocationResult::AllocationResult(const CanonicalSSA& ssa)
    : byValue_(ssa.valueCount() + 1, kUnassigned), shape_(ssa.shape()) {}

void AllocationResult::assign(SSAValueID id, RegKey physical) {
    assert(id != kInvalidSSAValueID && id < byValue_.size() && "invalid SSA value ID");
    byValue_[id] = physical;
}

bool AllocationResult::isAssigned(SSAValueID id) const {
    if (id == kInvalidSSAValueID || id >= byValue_.size()) return false;
    return byValue_[id].type != RegType::UNKNOWN;
}

RegKey AllocationResult::assignmentOf(SSAValueID id) const {
    assert(isAssigned(id) && "value has no physical register");
    return byValue_[id];
}

size_t AllocationResult::valueCount() const {
    return byValue_.empty() ? 0 : byValue_.size() - 1;
}

uint64_t AllocationResult::shape() const {
    return shape_;
}

size_t AllocationResult::unassignedCount() const {
    size_t unassigned = 0;
    for (size_t id = 1; id < byValue_.size(); ++id) {
        if (byValue_[id].type == RegType::UNKNOWN) ++unassigned;
    }
    return unassigned;
}

AllocationResult createLegacyColoring(const CanonicalSSA& ssa) {
    AllocationResult result(ssa);
    for (const SSAValue& value : ssa.values()) result.assign(value.id, value.origin);
    return result;
}

}  // namespace stinkytofu
