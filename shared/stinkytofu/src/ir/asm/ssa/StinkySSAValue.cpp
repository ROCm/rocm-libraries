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

#include "stinkytofu/ir/asm/ssa/StinkySSAValue.hpp"

#include <algorithm>
#include <cassert>
#include <utility>

#include "stinkytofu/ir/asm/ssa/StinkyOpOperand.hpp"

namespace stinkytofu {

StinkySSAValue::StinkySSAValue(Kind kind, TypeInfo type) : kind_(kind), type_(type) {}

StinkySSAValue::~StinkySSAValue() {
    assert(uses_.empty() && "destroying a StinkySSAValue that still has uses");
}

std::span<StinkyOpOperand* const> StinkySSAValue::uses() const {
    return uses_;
}

void StinkySSAValue::addUse(StinkyOpOperand* use) {
    assert(use != nullptr);
    uses_.push_back(use);
}

void StinkySSAValue::removeUse(StinkyOpOperand* use) {
    auto it = std::find(uses_.begin(), uses_.end(), use);
    if (it == uses_.end()) return;
    *it = uses_.back();
    uses_.pop_back();
}

void StinkySSAValue::bindDef(StinkyInstruction* defOp, uint16_t resultIndex) {
    defOp_ = defOp;
    resultIndex_ = resultIndex;
}

void StinkySSAValue::unbindDef() {
    defOp_ = nullptr;
    resultIndex_ = 0;
}

void StinkySSAValue::replaceAllUsesWith(StinkySSAValue* newValue) {
    assert(newValue != nullptr && "replaceAllUsesWith requires a non-null value");
    assert(newValue != this && "replaceAllUsesWith rejects self replacement");
    assert(type_ == newValue->type_ && "replaceAllUsesWith requires matching type/width");

    std::vector<StinkyOpOperand*> snapshot = uses_;
    for (StinkyOpOperand* use : snapshot) use->setValue(newValue);
    assert(uses_.empty());
}

const StinkySSAValue::PhysicalBinding& StinkySSAValue::physical() const {
    assert(hasBinding_);
    return binding_;
}

void StinkySSAValue::setPhysicalBinding(const PhysicalBinding& binding) {
    binding_ = binding;
    hasBinding_ = true;
}

void StinkySSAValue::clearPhysicalBinding() {
    binding_ = PhysicalBinding{};
    hasBinding_ = false;
}

void StinkySSAValue::setSymbol(std::string symbol) {
    symbol_ = std::move(symbol);
}

SSAArena::SSAArena(Function* owner) : owner_(owner) {
    byId_.push_back(nullptr);
}

SSAArena::~SSAArena() {
    clear();
}

StinkySSAValue* SSAArena::create(StinkySSAValue::Kind kind, RegType type, uint16_t dwordWidth) {
    auto value = std::unique_ptr<StinkySSAValue>(
        new StinkySSAValue(kind, StinkySSAValue::TypeInfo{type, dwordWidth}));
    const uint32_t id = static_cast<uint32_t>(byId_.size());
    value->setValueId(id);
    StinkySSAValue* raw = value.get();
    storage_.push_back(std::move(value));
    byId_.push_back(raw);
    return raw;
}

StinkySSAValue* SSAArena::createRegister(RegType type, uint16_t dwordWidth) {
    return create(StinkySSAValue::Kind::Register, type, dwordWidth);
}

StinkySSAValue* SSAArena::createBlockArgument(RegType type, uint16_t dwordWidth) {
    return create(StinkySSAValue::Kind::BlockArgument, type, dwordWidth);
}

StinkySSAValue* SSAArena::get(SSAValueID valueId) const {
    if (valueId == kInvalidSSAValueID || valueId >= byId_.size()) return nullptr;
    return byId_[valueId];
}

std::span<StinkySSAValue* const> SSAArena::values() const {
    if (byId_.size() <= 1) return {};
    return std::span<StinkySSAValue* const>(byId_.data() + 1, byId_.size() - 1);
}

void SSAArena::clear() {
    storage_.clear();
    byId_.clear();
    byId_.push_back(nullptr);
    shape_ = kUnstampedShape;
    liftedClasses_ = RegClassSet::all();
}

StinkyOpOperand::~StinkyOpOperand() {
    if (value_ != nullptr) value_->removeUse(this);
}

StinkySSAValue* StinkyOpOperand::value() const {
    return kind_ == Kind::Value ? value_ : nullptr;
}

void StinkyOpOperand::setValue(StinkySSAValue* v) {
    if (kind_ == Kind::Value && value_ == v) return;
    if (kind_ == Kind::Value && value_ != nullptr) value_->removeUse(this);
    value_ = v;
    kind_ = Kind::Value;
    imm_ = std::monostate{};
    if (value_ != nullptr) value_->addUse(this);
}

void StinkyOpOperand::setImm(LegacyImmPayload payload) {
    if (kind_ == Kind::Value && value_ != nullptr) {
        value_->removeUse(this);
        value_ = nullptr;
    }
    imm_ = std::move(payload);
    if (std::holds_alternative<int32_t>(imm_))
        kind_ = Kind::LiteralInt;
    else if (std::holds_alternative<double>(imm_))
        kind_ = Kind::LiteralDouble;
    else if (std::holds_alternative<std::string>(imm_))
        kind_ = Kind::LiteralString;
    else if (std::holds_alternative<HwRegPayload>(imm_))
        kind_ = Kind::HwReg;
    else
        kind_ = Kind::Invalid;
}

void StinkyOpOperand::bindOwner(StinkyInstruction* inst, uint16_t index) {
    instOwner_ = inst;
    blockOwner_ = nullptr;
    index_ = index;
}

void StinkyOpOperand::bindBlockOwner(BasicBlock* block, uint16_t index) {
    blockOwner_ = block;
    instOwner_ = nullptr;
    index_ = index;
}

}  // namespace stinkytofu
