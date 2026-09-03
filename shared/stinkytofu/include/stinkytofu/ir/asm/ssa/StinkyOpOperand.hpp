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

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"

namespace stinkytofu {

class BasicBlock;
struct StinkyInstruction;

struct HwRegPayload {
    uint16_t id = 0;
    uint16_t offset = 0;
    uint16_t size = 0;
};

using LegacyImmPayload = std::variant<std::monostate, int32_t, double, std::string, HwRegPayload>;

/// One operand slot and, for value operands, the use-list node.
class STINKYTOFU_EXPORT StinkyOpOperand {
   public:
    enum class Kind : uint8_t { Value, LiteralInt, LiteralDouble, LiteralString, HwReg, Invalid };

    StinkyOpOperand() = default;
    ~StinkyOpOperand();

    StinkyOpOperand(const StinkyOpOperand&) = delete;
    StinkyOpOperand& operator=(const StinkyOpOperand&) = delete;

    StinkyInstruction* owner() const {
        return instOwner_;
    }
    BasicBlock* ownerBlock() const {
        return blockOwner_;
    }
    uint16_t operandIndex() const {
        return index_;
    }
    Kind kind() const {
        return kind_;
    }

    StinkySSAValue* value() const;
    void setValue(StinkySSAValue* v);

    const LegacyImmPayload& imm() const {
        return imm_;
    }
    void setImm(LegacyImmPayload payload);

   private:
    friend struct StinkyInstruction;
    friend class BasicBlock;

    void bindOwner(StinkyInstruction* inst, uint16_t index);
    void bindBlockOwner(BasicBlock* block, uint16_t index);

    Kind kind_ = Kind::Invalid;
    StinkyInstruction* instOwner_ = nullptr;
    BasicBlock* blockOwner_ = nullptr;
    uint16_t index_ = 0;
    StinkySSAValue* value_ = nullptr;
    LegacyImmPayload imm_{};
};

inline std::unique_ptr<StinkyOpOperand> makeSSAValueOperand(StinkySSAValue* value) {
    auto operand = std::make_unique<StinkyOpOperand>();
    operand->setValue(value);
    return operand;
}

inline std::unique_ptr<StinkyOpOperand> makeSSAImmOperand(LegacyImmPayload payload) {
    auto operand = std::make_unique<StinkyOpOperand>();
    operand->setImm(std::move(payload));
    return operand;
}

/// Post-lift SSA payload stored on one instruction.
struct STINKYTOFU_EXPORT AttachedSSA {
    std::vector<StinkySSAValue*> results;
    std::vector<std::unique_ptr<StinkyOpOperand>> operands;

    // Copies must be deleted, not just unused: dllexport defines every declared,
    // non-deleted special member, and copying a unique_ptr is ill-formed.
    AttachedSSA() = default;
    AttachedSSA(AttachedSSA&&) = default;
    AttachedSSA& operator=(AttachedSSA&&) = default;
    AttachedSSA(const AttachedSSA&) = delete;
    AttachedSSA& operator=(const AttachedSSA&) = delete;
};

static_assert(!std::is_copy_constructible_v<AttachedSSA>,
              "AttachedSSA owns unique_ptrs; dllexport would force an ill-formed copy");

}  // namespace stinkytofu
