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
#include <span>
#include <string>
#include <vector>

#include "stinkytofu/ir/asm/StinkyRegister.hpp"
#include "stinkytofu/ir/asm/ssa/SSAOperandUnits.hpp"

namespace stinkytofu {

class BasicBlock;
class Function;
struct StinkyInstruction;
class StinkyOpOperand;
class SSAArena;

/// Index of a value within its SSAArena. Zero is reserved, so a default-built
/// ID names nothing and the arena can use index 0 as a hole.
using SSAValueID = uint32_t;
inline constexpr SSAValueID kInvalidSSAValueID = 0;

/// Shape of SSA that was built by hand rather than lifted, and whose agreement
/// with a function therefore cannot be checked. Lives with the field it
/// initialises, SSAArena::shape(); the stamp itself is computed by
/// computeFunctionShape() in analysis/asm/ssa/SSAFunctionShape.hpp.
inline constexpr uint64_t kUnstampedShape = 0;

/// SSA identity object. Allocated only in a Function-owned SSAArena.
///
/// Distinct from copyable StinkyRegister, which is the physical spelling in
/// srcRegs / destRegs. Kind distinguishes instruction results from block
/// arguments; both may carry a PhysicalBinding.
class StinkySSAValue {
   public:
    enum class Kind : uint8_t {
        Register,
        BlockArgument,
    };

    struct TypeInfo {
        RegType regType = RegType::UNKNOWN;
        uint16_t dwordWidth = 1;

        bool operator==(const TypeInfo& other) const {
            return regType == other.regType && dwordWidth == other.dwordWidth;
        }
        bool operator!=(const TypeInfo& other) const {
            return !(*this == other);
        }
    };

    struct PhysicalBinding {
        RegType type = RegType::UNKNOWN;
        uint32_t idx = 0;
        uint16_t num = 1;
        int16_t offset = 0;
        bool isVirtual = false;
        bool isMinus = false;
        bool isAbs = false;
    };

    ~StinkySSAValue();

    StinkySSAValue(const StinkySSAValue&) = delete;
    StinkySSAValue& operator=(const StinkySSAValue&) = delete;

    Kind kind() const {
        return kind_;
    }
    const TypeInfo& type() const {
        return type_;
    }
    SSAValueID valueId() const {
        return valueId_;
    }
    StinkyInstruction* defOp() const {
        return defOp_;
    }
    uint16_t resultIndex() const {
        return resultIndex_;
    }

    std::span<StinkyOpOperand* const> uses() const;
    bool hasOneUse() const {
        return uses_.size() == 1;
    }
    bool useEmpty() const {
        return uses_.empty();
    }
    size_t useCount() const {
        return uses_.size();
    }

    /// Rewrites every current use through StinkyOpOperand::setValue.
    /// Rejects null, self, and incompatible type/width.
    void replaceAllUsesWith(StinkySSAValue* newValue);

    bool hasPhysicalBinding() const {
        return hasBinding_;
    }
    const PhysicalBinding& physical() const;
    void setPhysicalBinding(const PhysicalBinding& binding);
    void clearPhysicalBinding();

    const std::string& symbol() const {
        return symbol_;
    }
    void setSymbol(std::string symbol);

   private:
    friend class StinkyOpOperand;
    friend struct StinkyInstruction;
    friend class SSAArena;

    StinkySSAValue(Kind kind, TypeInfo type);

    void addUse(StinkyOpOperand* use);
    void removeUse(StinkyOpOperand* use);
    void bindDef(StinkyInstruction* defOp, uint16_t resultIndex);
    void unbindDef();
    void setValueId(SSAValueID id) {
        valueId_ = id;
    }

    Kind kind_;
    TypeInfo type_;
    SSAValueID valueId_ = kInvalidSSAValueID;
    StinkyInstruction* defOp_ = nullptr;
    uint16_t resultIndex_ = 0;
    std::vector<StinkyOpOperand*> uses_;
    PhysicalBinding binding_{};
    bool hasBinding_ = false;
    std::string symbol_;
};

/// Function-owned storage for SSA values. Pointers remain stable until
/// `clear()` or Function destruction.
class SSAArena {
   public:
    explicit SSAArena(Function* owner);
    ~SSAArena();

    SSAArena(const SSAArena&) = delete;
    SSAArena& operator=(const SSAArena&) = delete;

    Function* owner() const {
        return owner_;
    }

    StinkySSAValue* createRegister(RegType type, uint16_t dwordWidth = 1);
    StinkySSAValue* createBlockArgument(RegType type, uint16_t dwordWidth = 1);

    size_t valueCount() const {
        return storage_.size();
    }

    StinkySSAValue* get(SSAValueID valueId) const;
    std::span<StinkySSAValue* const> values() const;

    /// Fingerprint of the function this arena was lifted from, or
    /// kUnstampedShape when the SSA was built by hand and cannot be checked
    /// against a program shape. computeFunctionShape() produces the stamp.
    uint64_t shape() const {
        return shape_;
    }
    void setShape(uint64_t shape) {
        shape_ = shape;
    }

    /// Register classes this arena's SSA was lifted from.
    ///
    /// Every walker that steps through srcRegs/destRegs alongside AttachedSSA
    /// must use this, because the slot layout depends on it. The shape
    /// fingerprint cannot stand in: it hashes the physical program, which is
    /// identical whichever classes were lifted.
    const RegClassSet& liftedClasses() const {
        return liftedClasses_;
    }
    void setLiftedClasses(const RegClassSet& classes) {
        liftedClasses_ = classes;
    }

    void clear();

   private:
    StinkySSAValue* create(StinkySSAValue::Kind kind, RegType type, uint16_t dwordWidth);

    Function* owner_ = nullptr;
    std::vector<std::unique_ptr<StinkySSAValue>> storage_;
    std::vector<StinkySSAValue*> byId_;
    uint64_t shape_ = kUnstampedShape;
    RegClassSet liftedClasses_ = RegClassSet::all();
};

}  // namespace stinkytofu
