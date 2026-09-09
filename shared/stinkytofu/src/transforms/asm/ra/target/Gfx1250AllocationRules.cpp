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
// What gfx1250 requires of a register allocation. Two rules; see
// docs/developer/register-allocation.md §14.

#include <array>
#include <vector>

#include "stinkytofu/core/Types.hpp"
#include "stinkytofu/ir/asm/RegisterKey.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRules.hpp"
#include "stinkytofu/transforms/asm/ra/AllocationRulesRegistry.hpp"

namespace stinkytofu {
namespace {

/// Scalar memory, as Gfx1250HazardPass classifies it in getMemoryGroupKind().
bool isSMemGroup(const StinkyInstruction& inst) {
    return isSMemLoad(inst) || isSMemStore(inst) || inst.is(InstFlag::IF_SMemAtomic);
}

/// An access that can return some DWORDs and XNACK on others; one DWORD returns
/// all or nothing, hence the width test.
///
/// Mirrors `returnsMultipleDwords` in Gfx1250HazardPass, which reports this
/// hazard after the fact while this rule prevents it. Keep the two in step.
bool smemCanPartiallyComplete(const StinkyInstruction& inst) {
    if (!isSMemGroup(inst)) return false;
    unsigned dwords = 0;
    for (const StinkyRegister& dest : inst.getDestRegs())
        forEachRegUnit(dest, [&](RegKey) { ++dwords; });
    return dwords > 1;
}

/// Index a \p width -DWORD scalar tuple must start on: pairs even, quads and
/// wider 4-aligned, singles anywhere.
///
/// An encoding requirement, not a preference -- the assembler rejects anything
/// else with "invalid register alignment". Widths that actually occur are
/// 1, 2, 4, 8 and 16, giving 1, 2, 4, 4, 4.
uint32_t scalarTupleAlign(uint32_t width) {
    if (width < 2) return 1;
    return width < 4 ? 2 : 4;
}

// A literal triple, not getArchTriple(GfxArchID::Gfx1250): this TU is compiled
// into a Gfx1250v0-only build where that enumerator does not exist, and keying
// on {12,5,0} is what gives v0 the same rules as v1.
constexpr std::array<int, 3> GFX1250_ARCH{12, 5, 0};

AllocationRules buildGfx1250Rules(const AsmCapsConfig& caps) {
    /// Replay re-reads the address, so an access whose destination covers its
    /// own address register has nothing left to replay from.
    ///
    /// clobbersEarly rather than forbidsBase because the hardware fact is about
    /// when the access reads versus writes; "destination and address must
    /// differ" is derived from it. That makes the destination overlap the
    /// sources dying there, which PhysRegMatrix refuses to share.
    ///
    /// Gated on a capability, not the triple: the same gfx1250 can be built
    /// either way.
    AllocationRule smemSelfOverlap;
    smemSelfOverlap.name = "SmemSelfOverlapUnderXnackReplay";
    smemSelfOverlap.description =
        "a multi-DWORD scalar memory access must not write any register its address occupies";
    // Active: the audit was silent on the producer's colouring, so this refuses
    // nothing that used to colour. What it prevents is greedy-compact reusing a
    // dead address register for the destination (register-allocation.md §14.5).
    smemSelfOverlap.status = caps.enableXnackReplay ? RuleStatus::Active : RuleStatus::Off;
    smemSelfOverlap.clobbersEarly = smemCanPartiallyComplete;

    /// An encoding requirement of every scalar tuple, not just SMEM operands,
    /// which is why it needs no instruction and no capability: the assembler
    /// rejects a misaligned SGPR tuple wherever it appears. Ungated for the same
    /// reason -- it is true of every gfx1250 module.
    AllocationRule scalarAlignment;
    scalarAlignment.name = "ScalarTupleAlignment";
    scalarAlignment.description =
        "a multi-DWORD scalar tuple must start on an index its width is aligned to";
    scalarAlignment.status = RuleStatus::Active;
    scalarAlignment.forbidsBase = [](RegType regClass, uint32_t base, uint32_t width) {
        return regClass == RegType::S && base % scalarTupleAlign(width) != 0;
    };

    return AllocationRules({smemSelfOverlap, scalarAlignment});
}

struct Gfx1250RulesRegistrar {
    Gfx1250RulesRegistrar() {
        AllocationRulesRegistry::setArch(GFX1250_ARCH, buildGfx1250Rules);
    }
};
static Gfx1250RulesRegistrar s_gfx1250RulesRegistrar;

}  // namespace

void anchorGfx1250AllocationRules() {}  // NOLINT(misc-use-internal-linkage)

}  // namespace stinkytofu
