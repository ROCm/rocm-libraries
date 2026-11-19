// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <utility>

#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"

namespace stinkytofu {

// Conservative facts from scalar eq/ne-u32 branches. Retain both equality and
// inequality so a nonzero entry guard rules out a later zero-trip exit. This is
// not value propagation: writes invalidate facts, and joins intersect them.
// Callers must resolve physical register indices before using this analysis.
struct ScalarBranchFacts {
    struct Predicate {
        uint32_t reg;
        uint32_t value;
        bool equal;
        bool operator==(const Predicate&) const = default;
    };
    std::map<std::pair<uint32_t, uint32_t>, bool> comparisons;
    std::optional<Predicate> predicate;
    std::optional<bool> scc;

    bool merge(const ScalarBranchFacts& other) {
        bool changed = false;
        for (auto it = comparisons.begin(); it != comparisons.end();) {
            auto found = other.comparisons.find(it->first);
            if (found == other.comparisons.end() || found->second != it->second) {
                it = comparisons.erase(it);
                changed = true;
            } else {
                ++it;
            }
        }
        if (predicate && predicate != other.predicate) {
            predicate.reset();
            changed = true;
        }
        if (scc && scc != other.scc) {
            scc.reset();
            changed = true;
        }
        return changed;
    }

    void transfer(const StinkyInstruction& inst) {
        if (isCall(inst)) {
            *this = {};
            return;
        }
        bool writesScc = inst.is(InstFlag::IF_ImplicitWriteSCC);
        for (const auto& dst : inst.getDestRegs()) {
            if (!dst.isRegister()) continue;
            writesScc |= dst.reg.type == RegType::SCC;
            if (dst.reg.type != RegType::S) continue;
            if (dst.reg.offset != 0 || dst.isVirtualReg()) {
                comparisons.clear();
                predicate.reset();
                continue;
            }
            auto overlaps = [&](uint32_t reg) {
                return reg >= dst.reg.idx && reg - dst.reg.idx < dst.reg.num;
            };
            for (auto it = comparisons.begin(); it != comparisons.end();) {
                if (overlaps(it->first.first))
                    it = comparisons.erase(it);
                else
                    ++it;
            }
            if (predicate && overlaps(predicate->reg)) predicate.reset();
        }
        if (writesScc) {
            predicate.reset();
            scc.reset();
        }
        const auto opcode = inst.getUnifiedOpcode();
        if (opcode != GFX::s_cmp_eq_u32 && opcode != GFX::s_cmp_lg_u32) return;
        predicate.reset();
        scc.reset();
        const auto& srcs = inst.getSrcRegs();
        if (srcs.size() < 2) return;
        const StinkyRegister* reg = &srcs[0];
        const StinkyRegister* value = &srcs[1];
        if (reg->dataType == StinkyRegister::Type::LiteralInt) std::swap(reg, value);
        if (!reg->isRegister() || reg->isVirtualReg() || reg->reg.type != RegType::S ||
            reg->reg.num != 1 || reg->reg.offset != 0 || reg->reg.isMinus || reg->reg.isAbs ||
            value->dataType != StinkyRegister::Type::LiteralInt)
            return;
        predicate = Predicate{reg->reg.idx, static_cast<uint32_t>(value->getLiteralInt()),
                              opcode == GFX::s_cmp_eq_u32};
        for (const auto& [key, equal] : comparisons) {
            if (key.first != predicate->reg) continue;
            if (key.second == predicate->value) {
                scc = equal == predicate->equal;
                break;
            }
            if (equal) {
                scc = !predicate->equal;
                break;
            }
        }
    }

    bool assumeScc(bool value) {
        if (scc && *scc != value) return false;
        if (predicate) comparisons[{predicate->reg, predicate->value}] = value == predicate->equal;
        scc = value;
        return true;
    }
};

}  // namespace stinkytofu
