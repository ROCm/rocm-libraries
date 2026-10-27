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
    /// Registry name, such as "greedy" or "legacy". Several names joined with
    /// '+' are tried in turn, and the pass applies the first that colours the
    /// function; allocateRegisters() itself takes one allocator.
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

    /// What to do about operands whose encoding field cannot select a VGPR
    /// bank, and which therefore reach one bank only.
    ///
    /// Allocate places them like anything else, under a ceiling, in a phase
    /// before the free blocks. It owes the producer nothing, so it moves an
    /// out-of-reach operand back into the bank instead of inheriting it.
    ///
    /// Hold keeps the register the producer chose. That register is reachable
    /// by construction, since the producer emitted working code, so the answer
    /// is known good before allocation starts -- but only while the producer
    /// keeps choosing reachable ones. Asked to freeze an out-of-reach register
    /// it refuses, having no other register to offer.
    ///
    /// Auto tries Allocate and, when that colouring is refused, Hold. Each
    /// fails where the other does not: Allocate can find no room under the
    /// ceiling once the matrix operands are pinned into the bottom bank, and
    /// Hold has nothing to offer when the producer placed an operand out of
    /// reach. Only the colouring that succeeds is applied.
    enum class UnbankableOperands { Hold, Allocate, Auto };
    UnbankableOperands unbankableOperands = UnbankableOperands::Auto;

    /// Where in the register file a free block may start.
    ///
    /// Any takes the lowest legal base. Prefer first searches the 256-register
    /// VGPR bank that holds the block's producer register, lowest base first,
    /// and searches the whole file only when that bank has no room. Values that
    /// stay in their producer's bank need no s_set_vgpr_msb switch the producer
    /// did not already pay for, at the cost of a less compact colouring.
    enum class Quadrant { Any, Prefer };
    Quadrant quadrant = Quadrant::Any;

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

    /// When non-empty, every attempt writes the allocator's placement table --
    /// one row per block with its width, live span, placement rank and base --
    /// to `<blockDumpPrefix><allocator>.<unbankable>.tsv`. Only allocators built
    /// on the greedy engine write one. A debugging aid; it changes no colouring.
    std::string blockDumpPrefix;

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

/// A pass that, when it runs, reports that register allocation did not, and
/// why: the same NotAllocated missed remark the allocation pass emits. For a
/// pipeline that decides while it is being built not to allocate, before any
/// PassContext exists to report through.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRegisterAllocationNotRunPass(std::string reason);

/// Driver without a PassManager, for tests.
///
/// Under UnbankableOperands::Auto this may colour twice, once allocating the
/// operands that cannot name a bank and once holding them. Only the colouring
/// returned is applied, and a refusal of both names each attempt's reason.
///
/// \p report receives the per-kernel comparison when `options.report` is set. It
/// is an out-parameter because the report reads attached SSA, which destruction
/// clears, so it has to be built before the colouring is applied.
Expected<AllocationResult> allocateRegisters(Function& function, RegisterAllocator& allocator,
                                             const RegisterAllocationOptions& options = {},
                                             std::string* report = nullptr);

}  // namespace stinkytofu
