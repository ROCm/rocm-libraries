// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cctype>
#include <cstdint>
#include <string>
#include <string_view>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

/// gfx1250 XNACK-replay group model shared by Gfx1250HazardPass and the SW instruction prefetch
/// passes (MI400-SPG §6.9.1, §6.9.4).
namespace stinkytofu::gfx1250 {

enum class ReplayGroupKind {
    None,
    SMEM,
    VMEM,
    Other,
};

enum class ReplayMode {
    SingleGroup,
    MultiGroup,
};

// A scalar prefetch is SMEM-family (opcodes 36-40) but is not classified here: it carries no
// replay sources and leaves XCNT unchanged, so it neither opens nor extends a tracked group.
inline ReplayGroupKind getReplayGroupKind(const StinkyInstruction& inst, ReplayMode replayMode) {
    if (isSMemLoad(inst) || isSMemStore(inst) || inst.is(InstFlag::IF_SMemAtomic))
        return ReplayGroupKind::SMEM;
    if (isMUBUFLoad(inst) || isMUBUFStore(inst) || isMUBUFAtomic(inst) || isFLATLoad(inst) ||
        isFLATStore(inst) || isFLATAtomic(inst) || isGLOBALLoad(inst) || isGLOBALStore(inst) ||
        isGLOBALAtomic(inst))
        return ReplayGroupKind::VMEM;
    if (isTensorLoad(inst))
        return replayMode == ReplayMode::SingleGroup ? ReplayGroupKind::Other
                                                     : ReplayGroupKind::VMEM;
    return ReplayGroupKind::None;
}

inline bool isScalarPrefetch(const StinkyInstruction& inst) {
    return inst.is(InstFlag::IF_ScalarInstPrefetch);
}

inline bool hasZeroWaitImmediate(const StinkyInstruction& inst) {
    const auto& srcs = inst.getSrcRegs();
    return srcs.size() == 1 && srcs.front().dataType == StinkyRegister::Type::LiteralInt &&
           srcs.front().getLiteralInt() == 0;
}

// A zero-count data wait guarantees translation for the corresponding memory
// operations. Nonzero waits leave work in flight and are not full drains.
inline bool isFullTranslationDrain(const StinkyInstruction& inst) {
    switch (inst.getUnifiedOpcode()) {
        case GFX::s_wait_xcnt:
        case GFX::s_wait_loadcnt:
        case GFX::s_wait_kmcnt:
        case GFX::s_wait_dscnt:
        case GFX::s_wait_loadcnt_dscnt:
        case GFX::s_wait_tensorcnt:
        case GFX::s_wait_storecnt:
        case GFX::s_wait_storecnt_dscnt:
        case GFX::s_wait_asynccnt:
            return hasZeroWaitImmediate(inst);
        default:
            return false;
    }
}

// Internal-control instructions can co-issue with a neighbour, and s_set_vgpr_msb is known to
// issue with XCNT != 0 (DEMI400-9920, ROCM-25414), so none of them is trusted to end a group.
inline bool isReplayGroupTransparent(const StinkyInstruction& inst) {
    if (isPseudoInst(&inst) || isScalarPrefetch(inst)) return true;
    switch (inst.getUnifiedOpcode()) {
        case GFX::s_set_vgpr_msb:
        case GFX::s_delay_alu:
        case GFX::s_wait_alu:
            return true;
        default:
            return false;
    }
}

inline bool isCommentOrBlankLine(std::string_view line) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string_view::npos) return true;
    line.remove_prefix(first);
    return line.starts_with("//") || line.starts_with("/*") || line.starts_with("*") ||
           line.starts_with(";") || line.starts_with("#");
}

// A TEXTBLOCK (raw text or a macro call) whose content the passes cannot classify. It may hold
// any instruction, including SMEM.
inline bool isOpaqueCode(const IRBase& node) {
    const auto* directive = dyn_cast<AsmDirective>(&node);
    if (directive == nullptr || directive->kind != AsmDirectiveKind::TEXTBLOCK) return false;
    std::string_view text = directive->value;
    while (!text.empty()) {
        const auto eol = text.find('\n');
        if (!isCommentOrBlankLine(text.substr(0, eol))) return true;
        if (eol == std::string_view::npos) break;
        text.remove_prefix(eol + 1);
    }
    return false;
}

inline constexpr uint16_t kModeReplayModeBit = 25;

// MODE.REPLAY_MODE (MODE bit 25) = 1 selects multi-group XNACK replay. A setreg whose field covers
// the bit counts unless it provably writes 0 there.
inline bool mayWriteMultiGroupReplayMode(const StinkyInstruction& inst) {
    const uint16_t op = inst.getUnifiedOpcode();
    if (op != GFX::s_setreg_IMM32_b32 && op != GFX::s_setreg_b32) return false;
    for (const auto& reg : inst.getDestRegs()) {
        if (reg.dataType != StinkyRegister::Type::HwReg) continue;
        if (reg.hwreg.id != HwReg::ID_MODE) return false;
        if (reg.hwreg.offset > kModeReplayModeBit ||
            reg.hwreg.offset + reg.hwreg.size <= kModeReplayModeBit)
            return false;
        const auto& srcs = inst.getSrcRegs();
        if (op != GFX::s_setreg_IMM32_b32 || srcs.empty() ||
            srcs.front().dataType != StinkyRegister::Type::LiteralInt)
            return true;
        const auto value = static_cast<uint64_t>(srcs.front().getLiteralInt());
        return ((value >> (kModeReplayModeBit - reg.hwreg.offset)) & 1U) != 0;
    }
    return true;
}

inline bool textMayWriteMode(std::string_view text) {
    std::string lower(text);
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (lower.find("setreg") == std::string::npos) return false;
    if (lower.find("hw_reg_mode") != std::string::npos ||
        lower.find("hw_reg_wave_mode") != std::string::npos)
        return true;
    for (auto pos = lower.find("hwreg("); pos != std::string::npos;
         pos = lower.find("hwreg(", pos + 1)) {
        auto i = lower.find_first_not_of(" \t", pos + 6);
        if (i == std::string::npos || lower[i] != '1') continue;
        i = lower.find_first_not_of(" \t", i + 1);
        if (i == std::string::npos || lower[i] == ',' || lower[i] == ')') return true;
    }
    return false;
}

/// True if \p func may switch the wave to multi-group XNACK replay. The single-group model in
/// this header and in Gfx1250HazardPass is unsound for such a kernel.
inline bool mayEnableMultiGroupReplay(const Function& func) {
    for (const BasicBlock& bb : func) {
        for (const IRBase& node : bb) {
            if (const auto* inst = dyn_cast<StinkyInstruction>(&node)) {
                if (mayWriteMultiGroupReplayMode(*inst)) return true;
            } else if (const auto* directive = dyn_cast<AsmDirective>(&node)) {
                if (directive->kind == AsmDirectiveKind::TEXTBLOCK &&
                    textMayWriteMode(directive->value))
                    return true;
            }
        }
    }
    return false;
}

/// True iff, in single-group mode, an SMEM replay group can still be open with XCNT != 0 when an
/// instruction placed before \p insertBefore issues. The closest preceding instruction that is not
/// replay-group transparent decides: SMEM keeps the group open; anything else ends it, because the
/// hardware drains XCNT before leaving a group. Opaque text may hold SMEM, so it answers true. The
/// walk follows physical layout and so crosses fall-through block boundaries; a block entered by a
/// branch has the branch as its predecessor.
inline bool smemReplayGroupOpenBefore(BasicBlock& bb, IRList::iterator insertBefore) {
    BasicBlock* block = &bb;
    IRBase* node = nullptr;
    if (insertBefore != bb.end())
        node = insertBefore.getNodePtr()->getPrev();
    else if (bb.rbegin() != bb.rend())
        node = &*bb.rbegin();

    while (block != nullptr) {
        for (; node != nullptr; node = node->getPrev()) {
            if (isOpaqueCode(*node)) return true;
            if (node->getType() != IRBase::IRType::StinkyTofu) continue;
            const auto& inst = *cast<StinkyInstruction>(node);
            if (isReplayGroupTransparent(inst)) continue;
            return getReplayGroupKind(inst, ReplayMode::SingleGroup) == ReplayGroupKind::SMEM;
        }
        block = block->getPrev();
        node = (block != nullptr && block->rbegin() != block->rend()) ? &*block->rbegin() : nullptr;
    }
    return false;
}

}  // namespace stinkytofu::gfx1250
