// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "stinkytofu/transforms/asm/ra/RegisterBudget.hpp"

#include <algorithm>
#include <limits>
#include <optional>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/ir/asm/StinkySignature.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu {
namespace {

/// One past the last index this operand covers, or 0 when it is not a physical
/// register of \p regClass. A multi-DWORD operand names only its base, so the
/// width is what decides the count. A virtual register carries kVirtualBit in
/// its index and would swamp the maximum, so it is skipped rather than counted.
uint32_t endOf(const StinkyRegister& reg, RegType regClass) {
    if (reg.dataType != StinkyRegister::Type::Register) return 0;
    if (reg.isVirtualReg()) return 0;
    if (reg.reg.type != regClass) return 0;
    const uint32_t width = std::max<uint16_t>(1, reg.reg.num);
    return reg.reg.idx + width;
}

/// Where the dispatch stopped writing scalars, as the signature published it on
/// the function. Nothing when the descriptor left it unsettled, which must not
/// read as zero: that would hand out the registers the dispatch did fill.
std::optional<uint32_t> publishedDispatchFilledSgprs(const Function& function) {
    const uint64_t filled = function.getMetaData(kSigDispatchFilledSgprsMetaKey).value_or(0);
    if (filled == 0 || filled > std::numeric_limits<uint32_t>::max()) return std::nullopt;
    return static_cast<uint32_t>(filled);
}

}  // namespace

uint32_t highestRegisterCount(const Function& function, RegType regClass) {
    uint32_t count = 0;
    for (const BasicBlock& block : function) {
        for (const IRBase& ir : block) {
            const auto* instruction = dyn_cast<StinkyInstruction>(&ir);
            if (instruction == nullptr) continue;
            for (const StinkyRegister& reg : instruction->getDestRegs())
                count = std::max(count, endOf(reg, regClass));
            for (const StinkyRegister& reg : instruction->getSrcRegs())
                count = std::max(count, endOf(reg, regClass));
        }
    }
    return count;
}

uint32_t nextEvenRegisterBase(const Function& function, RegType regClass) {
    const uint32_t base = highestRegisterCount(function, regClass);
    return base + (base & 1u);
}

std::optional<uint32_t> reusableEvenSgprBase(const Function& function, uint32_t width,
                                             uint32_t limit) {
    const std::optional<uint32_t> dispatchFilled = publishedDispatchFilledSgprs(function);
    if (!dispatchFilled) return std::nullopt;

    const uint32_t top = std::min(limit, highestRegisterCount(function, RegType::S));
    if (width == 0 || top < width) return std::nullopt;

    // Highest even base whose whole block ends at or below `top`, so the block
    // stays inside what the kernel already declares.
    const uint32_t base = (top - width) & ~1u;
    if (base < *dispatchFilled) return std::nullopt;
    return base;
}

uint32_t dispatchFilledSgprCount(int numSgprPreload, const std::array<int, 3>& workgroupIds) {
    // The kernarg segment pointer occupies two, matching the `numSgprPreload + 2`
    // the descriptor emits as .amdhsa_user_sgpr_count.
    uint32_t filled = numSgprPreload > 0 ? static_cast<uint32_t>(numSgprPreload) + 2u : 0u;
    for (int enabled : workgroupIds) {
        if (enabled > 0) ++filled;
    }
    return filled;
}

std::optional<uint32_t> settledDispatchFilledSgprCount(int numSgprPreload,
                                                       const std::array<int, 3>& workgroupIds) {
    if (numSgprPreload <= 0) return std::nullopt;
    return dispatchFilledSgprCount(numSgprPreload, workgroupIds);
}

uint32_t requiredSgprCount(const Function& function, int numSgprPreload,
                           const std::array<int, 3>& workgroupIds) {
    return std::max(highestRegisterCount(function, RegType::S),
                    dispatchFilledSgprCount(numSgprPreload, workgroupIds));
}

uint32_t dispatchFilledVgprCount(int vgprWorkItem, bool packedWorkitemId) {
    // Packed: x, y and z share v0's bits 0:9, 10:19 and 20:29, so the enabled
    // dimensions cost one register between them and the field says nothing.
    if (packedWorkitemId) return 1u;
    // Unpacked: one register per dimension from v0 up. The field counts extra
    // dimensions, so x alone is 0 and reaches v0.
    return vgprWorkItem < 0 ? 1u : static_cast<uint32_t>(vgprWorkItem) + 1u;
}

std::optional<uint32_t> settledDispatchFilledVgprCount(GfxArchID arch) {
    if (!ArchHelper::getInstance().getArchInfo(arch)) return std::nullopt;
    if (!hasPackedWorkitemId(arch)) return std::nullopt;
    return 1u;
}

uint32_t requiredVgprCount(const Function& function, int vgprWorkItem, bool packedWorkitemId) {
    return std::max(highestRegisterCount(function, RegType::V),
                    dispatchFilledVgprCount(vgprWorkItem, packedWorkitemId));
}

}  // namespace stinkytofu
