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
// What gfx1250 requires of a register allocation. Three rules; see
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

/// Pair a matrix destination with the accumulator it adds into.
///
/// Every WMMA is D = A*B + C, and the producer names one register range for D
/// and C so an accumulator chain occupies one tuple however long it runs. The
/// lift cannot see that: src2 is not read-write, correctly, since D and C are
/// genuinely separate values and must differ whenever C outlives the
/// instruction. So the chain arrives as a run of unrelated 8-wide values, and
/// without this the allocator has no reason to keep them in one place.
///
/// A preference rather than an AffinitySet precisely because of that "whenever".
/// Where C does outlive the instruction the two interfere, the shared register
/// is not available, and this simply goes unmet -- no test for the case is
/// needed here, because the one in placement already covers it.
void pairMatrixAccumulator(const StinkyInstruction& inst, const OperandValues& values,
                           std::vector<Preference>& preferences) {
    if (!isMatrixInstruction(inst)) return;
    const HwInstDesc* desc = inst.getHwInstDesc();
    if (desc == nullptr) return;

    // By encoding field rather than by operand number or mnemonic: the position
    // of src2 is a property of the format, and a new matrix opcode should not
    // need this rule edited.
    size_t srcIdx = 0;
    size_t accumulator = 0;
    bool found = false;
    for (const HwInstDesc::OperandFieldDesc& field : desc->operandFields) {
        if (field.isDest || field.isReadWrite) continue;
        if (field.encodeField == EncodeField::src2) {
            accumulator = srcIdx;
            found = true;
            break;
        }
        ++srcIdx;
    }
    if (!found) return;

    const std::span<const SSAValueID> dest = values(0, /*isDest=*/true);
    const std::span<const SSAValueID> from = values(accumulator, /*isDest=*/false);
    // Unit by unit, so DWORD i of the destination wants DWORD i of the
    // accumulator. Pairing only the bases would leave the rest of the tuple
    // free to drift and satisfy nothing.
    for (size_t unit = 0; unit < dest.size() && unit < from.size(); ++unit) {
        if (dest[unit] == kInvalidSSAValueID || from[unit] == kInvalidSSAValueID) continue;
        preferences.push_back({dest[unit], from[unit], 0, 1.0});
    }
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

    /// A different requirement from the scalar one, not just a different class:
    /// a vector tuple needs 64-bit alignment at every width where a scalar tuple
    /// needs its own, so `v[2:5]` is legal and `s[2:5]` is not. Hence a separate
    /// row, which also gets it named in the shadow report. The assembler rejects
    /// an odd base with "vgpr tuples must be 64 bit aligned".
    AllocationRule vectorAlignment;
    vectorAlignment.name = "VectorTupleAlignment";
    vectorAlignment.description = "a multi-DWORD vector tuple must start on an even index";
    vectorAlignment.status = RuleStatus::Active;
    vectorAlignment.forbidsBase = [](RegType regClass, uint32_t base, uint32_t width) {
        return regClass == RegType::V && width > 1 && base % 2 != 0;
    };

    /// Keeps an accumulator chain in one register range rather than one range
    /// per link. That lowers how many registers a WMMA-heavy kernel needs at
    /// once.
    ///
    /// Soft, so it cannot stop a kernel colouring. It can raise the high-water
    /// mark, because a destination may move up to reach its accumulator. Read
    /// pref[...] and highest= together, and pass noRule=WmmaAccumulatorReuse
    /// to measure without it.
    AllocationRule wmmaAccumulator;
    wmmaAccumulator.name = "WmmaAccumulatorReuse";
    wmmaAccumulator.description = "a matrix destination should reuse its accumulator's registers";
    wmmaAccumulator.status = RuleStatus::Active;
    wmmaAccumulator.satisfiedBy = [](RegKey d, RegKey c) { return d == c; };
    wmmaAccumulator.addPreferences = pairMatrixAccumulator;

    return AllocationRules({smemSelfOverlap, scalarAlignment, vectorAlignment, wmmaAccumulator});
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
