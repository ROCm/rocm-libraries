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
#pragma once

// One pass for every colouring policy.
//
// The driver owns everything policy-independent: fetch analyses, build
// constraints, refuse a policy whose capabilities lowering cannot honour,
// call allocate, verify, then optionally apply. Injection at construction
// follows createStinkyWmmaVgprReorderPass.

#include <memory>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/ssa/AllocationResult.hpp"
#include "stinkytofu/support/ErrorHandling.hpp"
#include "stinkytofu/transforms/asm/ra/RegisterAllocator.hpp"

namespace stinkytofu {

class Pass;

struct RegisterAllocationOptions {
    /// Registry name; "greedy" and "legacy" are both registered.
    std::string allocator = "greedy";

    /// Register classes the policy may move; see AllocationScope::classes(). The
    /// VGPR-only default is a safety statement, not a preference: no ABI range
    /// is reserved, so widen it knowingly. An architecture may still forbid
    /// some bases as a placement rule (see AllocationRules).
    RegClassSet allocate = RegClassSet::only(RegType::V);

    /// When non-empty, only values whose live range lies entirely before this
    /// block's end slot may move. The label matches BasicBlock::getLabel(); a
    /// leading '^' is ignored. Empty means the whole function.
    std::string regionEnd;

    /// Registers to leave exactly as found: each keeps the value lifted into it
    /// and takes no other. Bounds are inclusive, so {{RegType::S, 0, 19}} holds
    /// s0 through s19. See AllocationScope::pinRegisters.
    std::vector<AllocationScope::HeldRange> pinRegisters;

    bool applyToOperands = false;  // false = shadow
    bool verify = true;

    /// Emit a per-kernel comparison of this colouring against the producer's:
    /// value count, peak pressure, highest index, and projected occupancy per
    /// class. This is what a shadow run is for, since it changes nothing itself.
    bool report = false;

    /// After apply, emit a register-map TEXTBLOCK (see
    /// docs/developer/register-allocation.md §11.3).
    bool emitRegisterMap = false;

    /// After apply, append per-instruction breadcrumbs when a symbolic name is stripped.
    bool emitSymbolBreadcrumbs = false;

    /// Module capabilities, which parameterize an architecture's allocation
    /// rules (see AllocationRulesRegistry). Carried here rather than as another
    /// allocateRegisters parameter because the pass already owns the options and
    /// can fill this from PassContext; the free driver leaves it defaulted, which
    /// means "no capability is set".
    AsmCapsConfig caps;

    /// Force rule statuses regardless of the architecture's own gate. Empty in
    /// production; see RuleOverrides for why the hatch exists.
    RuleOverrides rules;
};

/// allocator == nullptr looks the policy up by name.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRegisterAllocationPass(
    RegisterAllocationOptions options = {}, std::unique_ptr<RegisterAllocator> allocator = nullptr);

/// Driver without a PassManager, for tests.
///
/// \p report receives the per-kernel comparison when `options.report` is set. It
/// is an out-parameter because the report reads attached SSA, which destruction
/// clears, so it has to be built before the colouring is applied.
Expected<AllocationResult> allocateRegisters(Function& function, RegisterAllocator& allocator,
                                             const RegisterAllocationOptions& options = {},
                                             std::string* report = nullptr);

}  // namespace stinkytofu
