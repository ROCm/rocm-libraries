/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
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

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
// Register type enumeration for different register classes
enum class RegType {
#define REGISTER_TYPE(ENUM, STR, DESC) ENUM,
#include "stinkytofu/ir/asm/RegisterType.def"
};

// Helper function to convert string to RegType
/// Convert string to RegType. Returns RegType::UNKNOWN for invalid strings.
/// Use isValidRegType() to check if the result is valid.
///
/// Example usage:
///   RegType type = stringToRegType("v");
///   if (!isValidRegType(type)) {
///       // Handle error: unknown register type
///   }
///
/// For new code, prefer tryParseRegType() which returns std::optional.
inline RegType stringToRegType(const std::string& str) {
#define REGISTER_TYPE(ENUM, STR, DESC) /* NOLINT(bugprone-macro-parentheses) */ \
    if (str == STR) return RegType::ENUM;
#include "stinkytofu/ir/asm/RegisterType.def"
    // Don't assert on invalid input - return UNKNOWN for error handling
    // Callers should check with isValidRegType() or compare to UNKNOWN
    return RegType::UNKNOWN;
}

/// Check if a register type is valid (not UNKNOWN)
inline bool isValidRegType(RegType type) {
    return type != RegType::UNKNOWN;
}

/// Check if a register type is an allocatable register (VGPR, SGPR, AGPR).
/// Excludes special registers (SCC, VCC, EXEC, M0, LDS).
inline bool isAllocatableReg(RegType type) {
    return type == RegType::V || type == RegType::S || type == RegType::A || type == RegType::ACC ||
           type == RegType::AGPR;
}

/// Validate if a string represents a valid register type
inline bool isValidRegTypeString(const std::string& str) {
    return stringToRegType(str) != RegType::UNKNOWN;
}

/// Convert string to RegType with explicit validation.
/// Returns std::nullopt if the string is not a valid register type.
/// Prefer this over stringToRegType() for new code.
///
/// Example usage:
///   auto type = tryParseRegType("v");
///   if (!type) {
///       // Handle error: unknown register type
///       return error;
///   }
///   // Use *type safely
///
/// This is the recommended API for C++ code that needs validation.
inline std::optional<RegType> tryParseRegType(const std::string& str) {
    RegType result = stringToRegType(str);
    if (result == RegType::UNKNOWN) {
        return std::nullopt;
    }
    return result;
}

// Helper function to convert RegType to string
inline std::string regTypeToString(RegType type) {
    switch (type) {
#define REGISTER_TYPE(ENUM, STR, DESC) \
    case RegType::ENUM:                \
        return STR;
#include "stinkytofu/ir/asm/RegisterType.def"
    }
    assert(false && "Invalid register type");
    return "UNKNOWN";
}

// Represents a register or a literal value in the StinkyTofu IR.
struct STINKYTOFU_EXPORT StinkyRegister {
    // Bit 31 of reg.idx marks a virtual register. Virtual registers are
    // placeholders used in instruction templates; they must be resolved to
    // physical registers via resolveVirtualToPhysical() before the IR
    // reaches the emitter or scheduler. If a virtual register leaks into
    // code that reads reg.idx directly, the huge value (>= 2^31) will
    // cause obvious failures.
    static constexpr uint32_t kVirtualBit = 1u << 31;

    enum class Type { Register, LiteralInt, LiteralDouble, LiteralString, Invalid };

    Type dataType;

    // Unnamed union to save memory - different data types use different fields
    union {
        // For Register type
        // registers = [idx : idx + num)
        struct {
            RegType type;

            // Index of the first register in the range.
            // Bit 31 (kVirtualBit) is reserved: when set the register is
            // virtual and the lower 31 bits hold the virtual index.
            uint32_t idx;
            // number of consecutive registers
            uint16_t num;

            // offset of the register for use case such as msb, etc.
            int16_t offset;

            uint32_t isMinus : 1;
            uint32_t isAbs : 1;
        } reg;

        int32_t literalInt;
        double literalDouble;
    };

    // For LiteralString type - kept separate as std::string is non-trivial
    // Also used to store symbolic register name (e.g., "vgprLocalWriteAddrA") for Register type
    std::string literalValue;

    // define ordering for std::map key
    bool operator<(const StinkyRegister& other) const noexcept {
        if (dataType != other.dataType) return dataType < other.dataType;

        switch (dataType) {
            case Type::Register: {
                if (reg.type != other.reg.type) return reg.type < other.reg.type;
                if (reg.idx != other.reg.idx) return reg.idx < other.reg.idx;
                if (reg.num != other.reg.num) return reg.num < other.reg.num;
                return literalValue < other.literalValue;
            }
            case Type::LiteralInt:
                return literalInt < other.literalInt;
            case Type::LiteralDouble:
                return literalDouble < other.literalDouble;
            case Type::LiteralString:
                return literalValue < other.literalValue;
            case Type::Invalid:
                return false;
        }

        return false;
    }

    // define equality for comparisons
    bool operator==(const StinkyRegister& other) const noexcept {
        if (dataType != other.dataType) return false;

        switch (dataType) {
            case Type::Register:
                return reg.type == other.reg.type && reg.idx == other.reg.idx &&
                       reg.num == other.reg.num;
            case Type::LiteralInt:
                return literalInt == other.literalInt;
            case Type::LiteralDouble:
                return literalDouble == other.literalDouble;
            case Type::LiteralString:
                return literalValue == other.literalValue;
            case Type::Invalid:
                return true;  // Two invalid registers are equal
        }
        return false;
    }

    bool operator!=(const StinkyRegister& other) const noexcept {
        return !(*this == other);
    }

    StinkyRegister(RegType type, uint32_t regIdx, uint16_t regNum, int16_t offset = 0)
        : dataType(Type::Register), reg{type, regIdx, regNum, offset, false, false} {}

    // Constructor accepting string for backward compatibility
    StinkyRegister(const std::string& typeStr, uint32_t regIdx, uint16_t regNum, int16_t offset = 0)
        : dataType(Type::Register),
          reg{stringToRegType(typeStr), regIdx, regNum, offset, false, false} {}

    StinkyRegister(const std::string& str) : dataType(Type::LiteralString), literalValue(str) {}

    StinkyRegister(int literalInt) : dataType(Type::LiteralInt), literalInt(literalInt) {}

    StinkyRegister(double literalDouble)
        : dataType(Type::LiteralDouble), literalDouble(literalDouble) {}

    StinkyRegister() : dataType(Type::Invalid), reg{RegType::UNKNOWN, 0, 0, 0, false, false} {}

    int getLiteralInt() const {
        assert(dataType == Type::LiteralInt);
        return literalInt;
    }

    double getLiteralDouble() const {
        assert(dataType == Type::LiteralDouble);
        return literalDouble;
    }

    std::string getLiteralString() const {
        assert(dataType == Type::LiteralString);
        return literalValue;
    }

    bool isValid() const {
        return dataType != Type::Invalid;
    }

    bool isRegister() const {
        return dataType == Type::Register;
    }

    // Get symbolic register name (e.g., "vgprLocalWriteAddrA")
    // Returns empty string if no symbolic name is set
    std::string getSymbolicName() const {
        if (dataType == Type::Register) return literalValue;
        return "";
    }

    // Set symbolic register name for Register type
    void setSymbolicName(const std::string& name) {
        if (dataType == Type::Register) literalValue = name;
    }

    // Check if register has a symbolic name
    bool hasSymbolicName() const {
        return dataType == Type::Register && !literalValue.empty();
    }

    bool isOverlap(const StinkyRegister& other) const {
        if (dataType != Type::Register || other.dataType != Type::Register) return false;
        if (reg.type != other.reg.type) return false;
        if (reg.idx + reg.num <= other.reg.idx || other.reg.idx + other.reg.num <= reg.idx)
            return false;
        return true;
    }

    void dump(std::ostream& out, const std::string& prefix = "") const;

    void dump() const;

    static StinkyRegister getSCCRegister() {
        return StinkyRegister(RegType::SCC, 0, 1);
    }

    static StinkyRegister getVCCRegister(uint32_t wavefrontSize) {
        return wavefrontSize == 32 ? StinkyRegister(RegType::VCC_LO, 0, 1)
                                   : StinkyRegister(RegType::VCC, 0, 1);
    }

    static StinkyRegister getEXECRegister(uint32_t wavefrontSize) {
        return wavefrontSize == 32 ? StinkyRegister(RegType::EXEC_LO, 0, 1)
                                   : StinkyRegister(RegType::EXEC, 0, 1);
    }

    /// Create a virtual VGPR register for template-based code generation.
    static StinkyRegister Virtual(uint32_t idx, uint16_t num = 1) {
        return VirtualReg(RegType::V, idx, num);
    }

    /// Create a virtual SGPR register for template-based code generation.
    static StinkyRegister VirtualSGPR(uint32_t idx, uint16_t num = 1) {
        return VirtualReg(RegType::S, idx, num);
    }

    /// Create a virtual register for template-based code generation.
    ///
    /// Virtual registers are placeholders used when generating reusable
    /// instruction templates (e.g., activation functions). They carry
    /// kVirtualBit in reg.idx and must be resolved to physical registers
    /// via resolveVirtualToPhysical() before the IR reaches the emitter.
    ///
    /// @param type Register class (e.g., RegType::V for VGPR, RegType::S for SGPR)
    /// @param idx  Virtual register index
    /// @param num  Number of consecutive registers
    static StinkyRegister VirtualReg(RegType type, uint32_t idx, uint16_t num = 1) {
        StinkyRegister reg(type, idx | kVirtualBit, num);
        return reg;
    }

    /// DANGEROUS -- Resolve a virtual register to a physical register.
    ///
    /// Clears the virtual bit and adds @p offset to the base index,
    /// producing a physical register. The caller is responsible for
    /// ensuring:
    ///   - The resulting physical index is within the architectural limit.
    ///   - The index range [result .. result+num) does not collide with
    ///     other live registers.
    ///
    /// Non-virtual registers and literals are returned unchanged.
    ///
    /// @param offset  Offset added to the virtual index to produce the
    ///                physical index.
    StinkyRegister resolveVirtualToPhysical(int offset) const {
        if (dataType != Type::Register || !(reg.idx & kVirtualBit)) return *this;

        StinkyRegister result = *this;
        result.reg.idx = (reg.idx & ~kVirtualBit) + offset;
        return result;
    }

    bool isVirtualReg() const {
        return dataType == Type::Register && (reg.idx & kVirtualBit);
    }

    /// Compute hash value for this register.
    /// Used by std::unordered_map and other hash-based containers.
    size_t hash() const noexcept {
        size_t h = std::hash<int>{}(static_cast<int>(dataType));
        if (dataType == Type::Register) {
            h ^= std::hash<int>{}(static_cast<int>(reg.type)) << 1;
            h ^= std::hash<uint32_t>{}(reg.idx) << 2;
            h ^= std::hash<uint16_t>{}(reg.num) << 3;
        } else if (dataType == Type::LiteralInt) {
            h ^= std::hash<int>{}(literalInt) << 1;
        } else if (dataType == Type::LiteralDouble) {
            h ^= std::hash<double>{}(literalDouble) << 1;
        } else if (dataType == Type::LiteralString) {
            h ^= std::hash<std::string>{}(literalValue) << 1;
        }
        return h;
    }
};

/// Check if register is a pseudo register (BARRIER, DS_WRITE, TENSOR_LOAD, etc.).
/// Pseudo registers are used internally for dependency tracking but should not
/// appear in assembly output. All pseudo registers are defined after PSEUDO_START
/// in RegisterType.def.
inline bool isPseudoReg(const StinkyRegister& reg) {
    if (reg.dataType != StinkyRegister::Type::Register) return false;

    return reg.reg.type >= RegType::PSEUDO_START;
}

}  // namespace stinkytofu

// Hash specialization for StinkyRegister (required for std::unordered_map)
namespace std {
template <>
struct hash<stinkytofu::StinkyRegister> {
    size_t operator()(const stinkytofu::StinkyRegister& sreg) const noexcept {
        return sreg.hash();
    }
};
}  // namespace std
