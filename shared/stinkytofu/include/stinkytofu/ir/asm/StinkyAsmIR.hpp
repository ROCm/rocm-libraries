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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "stinkytofu/core/BasicBlock.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/core/IRBase.hpp"
#include "stinkytofu/core/IRBuilder.hpp"
#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"
#include "stinkytofu/ir/asm/StinkyModifiers.hpp"
#include "stinkytofu/support/Casting.hpp"

namespace stinkytofu
{
#define GET_ISAINFO_UNIFIED_OPCODES
#include "hardware/gfxIsa.inc"

    // Register type enumeration for different register classes
    enum class RegType
    {
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
    inline RegType stringToRegType(const std::string& str)
    {
#define REGISTER_TYPE(ENUM, STR, DESC) \
    if(str == STR)                     \
        return RegType::ENUM;
#include "stinkytofu/ir/asm/RegisterType.def"
        // Don't assert on invalid input - return UNKNOWN for error handling
        // Callers should check with isValidRegType() or compare to UNKNOWN
        return RegType::UNKNOWN;
    }

    /// Check if a register type is valid (not UNKNOWN)
    inline bool isValidRegType(RegType type)
    {
        return type != RegType::UNKNOWN;
    }

    /// Validate if a string represents a valid register type
    inline bool isValidRegTypeString(const std::string& str)
    {
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
    inline std::optional<RegType> tryParseRegType(const std::string& str)
    {
        RegType result = stringToRegType(str);
        if(result == RegType::UNKNOWN)
        {
            return std::nullopt;
        }
        return result;
    }

    // Helper function to convert RegType to string
    inline std::string regTypeToString(RegType type)
    {
        switch(type)
        {
#define REGISTER_TYPE(ENUM, STR, DESC) \
    case RegType::ENUM:                \
        return STR;
#include "stinkytofu/ir/asm/RegisterType.def"
        }
        assert(false && "Invalid register type");
        return "UNKNOWN";
    }

    // Represents a register or a literal value in the StinkyTofu IR.
    struct StinkyRegister
    {
        enum class Type
        {
            Register,
            LiteralInt,
            LiteralDouble,
            LiteralString,
            Invalid
        };

        Type dataType;

        // Unnamed union to save memory - different data types use different fields
        union
        {
            // For Register type
            // registers = [idx : idx + num)
            struct
            {
                RegType type;

                // index of the first register in the range
                uint32_t idx;
                // number of consecutive registers
                uint16_t num;

                // offset of the register for use case such as msb, etc.
                int16_t offset;

                // Virtual register flag for template-based code generation
                // Only meaningful when dataType == Type::Register
                // When true, this register needs offset remapping before use
                uint32_t isVirtual : 1;

                uint32_t isMinus : 1;
                uint32_t isAbs : 1;
            } reg;

            int32_t literalInt;
            double  literalDouble;
        };

        // For LiteralString type - kept separate as std::string is non-trivial
        // Also used to store symbolic register name (e.g., "vgprLocalWriteAddrA") for Register type
        std::string literalValue;

        // define ordering for std::map key
        bool operator<(const StinkyRegister& other) const noexcept
        {
            if(dataType != other.dataType)
                return dataType < other.dataType;

            switch(dataType)
            {
            case Type::Register:
            {
                if(reg.type != other.reg.type)
                    return reg.type < other.reg.type;
                if(reg.idx != other.reg.idx)
                    return reg.idx < other.reg.idx;
                if(reg.num != other.reg.num)
                    return reg.num < other.reg.num;
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
        bool operator==(const StinkyRegister& other) const noexcept
        {
            if(dataType != other.dataType)
                return false;

            switch(dataType)
            {
            case Type::Register:
                return reg.type == other.reg.type && reg.idx == other.reg.idx
                       && reg.num == other.reg.num;
            case Type::LiteralInt:
                return literalInt == other.literalInt;
            case Type::LiteralDouble:
                return literalDouble == other.literalDouble;
            case Type::LiteralString:
                return literalValue == other.literalValue;
            case Type::Invalid:
                return true; // Two invalid registers are equal
            }
            return false;
        }

        bool operator!=(const StinkyRegister& other) const noexcept
        {
            return !(*this == other);
        }

        StinkyRegister(RegType type, uint32_t regIdx, uint16_t regNum, int16_t offset = 0)
            : dataType(Type::Register)
            , reg{type, regIdx, regNum, offset, false, false, false}
        {
        }

        // Constructor accepting string for backward compatibility
        StinkyRegister(const std::string& typeStr,
                       uint32_t           regIdx,
                       uint16_t           regNum,
                       int16_t            offset = 0)
            : dataType(Type::Register)
            , reg{stringToRegType(typeStr), regIdx, regNum, offset, false, false, false}
        {
        }

        StinkyRegister(const std::string& str)
            : dataType(Type::LiteralString)
            , literalValue(str)
        {
        }

        StinkyRegister(int literalInt)
            : dataType(Type::LiteralInt)
            , literalInt(literalInt)
        {
        }

        StinkyRegister(double literalDouble)
            : dataType(Type::LiteralDouble)
            , literalDouble(literalDouble)
        {
        }

        StinkyRegister()
            : dataType(Type::Invalid)
            , reg{RegType::UNKNOWN, 0, 0, 0, false, false, false}
        {
        }

        int getLiteralInt() const
        {
            assert(dataType == Type::LiteralInt);
            return literalInt;
        }

        double getLiteralDouble() const
        {
            assert(dataType == Type::LiteralDouble);
            return literalDouble;
        }

        std::string getLiteralString() const
        {
            assert(dataType == Type::LiteralString);
            return literalValue;
        }

        bool isValid() const
        {
            return dataType != Type::Invalid;
        }

        bool isRegister() const
        {
            return dataType == Type::Register;
        }

        // Get symbolic register name (e.g., "vgprLocalWriteAddrA")
        // Returns empty string if no symbolic name is set
        std::string getSymbolicName() const
        {
            if(dataType == Type::Register)
                return literalValue;
            return "";
        }

        // Set symbolic register name for Register type
        void setSymbolicName(const std::string& name)
        {
            if(dataType == Type::Register)
                literalValue = name;
        }

        // Check if register has a symbolic name
        bool hasSymbolicName() const
        {
            return dataType == Type::Register && !literalValue.empty();
        }

        bool isOverlap(const StinkyRegister& other) const
        {
            if(dataType != Type::Register || other.dataType != Type::Register)
                return false;
            if(reg.type != other.reg.type)
                return false;
            if(reg.idx + reg.num <= other.reg.idx || other.reg.idx + other.reg.num <= reg.idx)
                return false;
            return true;
        }

        void dump(std::ostream& out, const std::string& prefix = "") const;

        void dump() const;

        // Implicit register: SCC, VCC, EXEC, etc.
        static StinkyRegister getSCCRegister()
        {
            return StinkyRegister(RegType::SCC, 0, 1);
        }

        // Pseudo register: BARRIER, DS_WRITE, TENSOR_LOAD, etc.
        static StinkyRegister getBarrierRegister()
        {
            return StinkyRegister(RegType::BARRIER, 0, 1);
        }

        static StinkyRegister getDSWriteRegister()
        {
            return StinkyRegister(RegType::DS_WRITE, 0, 1);
        }

        static StinkyRegister getTensorLoadRegister()
        {
            return StinkyRegister(RegType::TENSOR_LOAD, 0, 1);
        }

        /**
         * @brief Create a virtual VGPR register for template-based code generation
         *
         * Virtual registers are used when generating reusable instruction templates
         * (e.g., activation functions). They need offset remapping when instantiated.
         *
         * @param idx Virtual register index (e.g., 0 for first temp register)
         * @param num Number of consecutive registers (default 1)
         * @return Virtual register that can be remapped with withOffset()
         *
         * Example:
         *   auto vTemp0 = StinkyRegister::Virtual(0);  // Virtual v0
         *   auto vTemp1 = StinkyRegister::Virtual(1);  // Virtual v1
         *   // Later when inlining:
         *   auto v10 = vTemp0.withOffset(10);  // Maps to physical v10
         */
        static StinkyRegister Virtual(int idx, int num = 1)
        {
            StinkyRegister reg(RegType::V, idx, num);
            reg.reg.isVirtual = true;
            return reg;
        }

        /**
         * @brief Create a virtual SGPR register for template-based code generation
         *
         * Similar to Virtual() but for scalar registers.
         *
         * @param idx Virtual register index
         * @param num Number of consecutive registers (default 1)
         * @return Virtual SGPR that can be remapped with withOffset()
         */
        static StinkyRegister VirtualSGPR(int idx, int num = 1)
        {
            StinkyRegister reg(RegType::S, idx, num);
            reg.reg.isVirtual = true;
            return reg;
        }

        /**
         * @brief Apply register offset to remap virtual register to physical
         *
         * Used when instantiating instruction templates with actual register allocations.
         * Only applies to virtual registers - non-virtual registers are returned unchanged.
         * Literals are also returned unchanged.
         *
         * @param offset Offset to add to register index
         * @return New register with offset applied (or original if not virtual/register)
         *
         * Example:
         *   auto vTemp = StinkyRegister::Virtual(0);  // Virtual v0
         *   auto v10 = vTemp.withOffset(10);          // Physical v10
         *   auto v20 = vTemp.withOffset(20);          // Physical v20
         */
        StinkyRegister withOffset(int offset) const
        {
            // Only apply offset to virtual registers
            if(dataType != Type::Register || !reg.isVirtual)
                return *this;

            StinkyRegister result = *this;
            result.reg.idx += offset;
            result.reg.isVirtual = false; // No longer virtual after remapping
            return result;
        }

        /**
         * @brief Check if this is a virtual register needing remapping
         *
         * Note: Only registers (Type::Register) can be virtual. Literals always return false.
         */
        bool isVirtualRegister() const
        {
            return dataType == Type::Register && reg.isVirtual;
        }

        /**
         * @brief Compute hash value for this register
         *
         * Used by std::unordered_map and other hash-based containers.
         */
        size_t hash() const noexcept
        {
            size_t h = std::hash<int>{}(static_cast<int>(dataType));
            if(dataType == Type::Register)
            {
                h ^= std::hash<int>{}(static_cast<int>(reg.type)) << 1;
                h ^= std::hash<uint32_t>{}(reg.idx) << 2;
                h ^= std::hash<uint16_t>{}(reg.num) << 3;
            }
            else if(dataType == Type::LiteralInt)
            {
                h ^= std::hash<int>{}(literalInt) << 1;
            }
            else if(dataType == Type::LiteralDouble)
            {
                h ^= std::hash<double>{}(literalDouble) << 1;
            }
            else if(dataType == Type::LiteralString)
            {
                h ^= std::hash<std::string>{}(literalValue) << 1;
            }
            return h;
        }
    };

    /// Check if register is a pseudo register (BARRIER, DS_WRITE, TENSOR_LOAD, etc.).
    /// Pseudo registers are used internally for dependency tracking but should not
    /// appear in assembly output. All pseudo registers are defined after PSEUDO_START
    /// in RegisterType.def.
    inline bool isPseudoReg(const StinkyRegister& reg)
    {
        if(reg.dataType != StinkyRegister::Type::Register)
            return false;

        return reg.reg.type >= RegType::PSEUDO_START;
    }

    /// Check if register is an implicit register (SCC, VCC, EXEC, etc.).
    /// Implicit registers are set implicitly by instructions and should not be
    /// printed in assembly output.
    inline bool isImplicitRegister(const StinkyRegister& reg)
    {
        if(reg.dataType != StinkyRegister::Type::Register)
            return false;

        return reg.reg.type == RegType::SCC;
    }

    // Represents a single assembly instruction.
    struct StinkyInstruction : public IRBase
    {
        friend class IRBase;
        friend class AsmIRBuilder;

        int issueCycles;
        int latencyCycles;

    private:
        // Def-use chain:
        // instructions that USE the value DEFINED by this instruction
        // (consumers of this instruction's output).
        std::vector<StinkyInstruction*> users;

        // instructions that DEFINE the values USED by this instruction
        // (producers of this instruction's operands).
        std::vector<StinkyInstruction*> sources;

        const HwInstDesc* hwInstDesc;

        // Modifiers are extra bits/fields in the instruction encoding that
        // tweak how the operation is performed, without needing a completely
        // different opcode.
        std::vector<std::unique_ptr<Modifier>> modifiers;

        // Register operands - private to enforce use-def chain maintenance
        // Access through getters, mutate through setSrcRegs/setDestRegs
        // TODO: CMP instructions imply modify SCC register, but it's not tracked in
        //       the frontend (e.g. SCmpKEQU32 struct), a workaround is to track it
        //       here.
        std::vector<StinkyRegister> destRegs;
        std::vector<StinkyRegister> srcRegs;

        StinkyInstruction(const HwInstDesc* mcid)
            : IRBase(IRType::StinkyTofu)
            , hwInstDesc(mcid)
            , issueCycles(mcid->issue)
            , latencyCycles(mcid->latency)
        {
        }

        ~StinkyInstruction()
        {
            // Clean up use-def chains
            unlinkFromSources();
            unlinkFromUsers();

            // Verify that this instruction is no longer referenced by any sources
            // After unlinkFromSources(), no operand def should have this in their users list
            for(StinkyInstruction* source : sources)
            {
                if(source != nullptr)
                {
                    // Check that this instruction is not in source's users list
                    auto it = std::find(source->users.begin(), source->users.end(), this);
                    assert(it == source->users.end()
                           && "Destructor: source still references this instruction in users list");
                }
            }
        }

    public:
        void addSrcReg(const StinkyRegister& srcReg)
        {
            srcRegs.push_back(srcReg);
        }

        void addDestReg(const StinkyRegister& destReg)
        {
            destRegs.push_back(destReg);
        }

        /// Get destination registers (read-only)
        const std::vector<StinkyRegister>& getDestRegs() const
        {
            return destRegs;
        }

        /// Get source registers (read-only)
        const std::vector<StinkyRegister>& getSrcRegs() const
        {
            return srcRegs;
        }

        /// Get number of destination registers
        size_t getNumDestRegs() const
        {
            return destRegs.size();
        }

        /// Get number of source registers
        size_t getNumSrcRegs() const
        {
            return srcRegs.size();
        }

        /// Get instructions that use the value defined by this instruction (def-use chain)
        const std::vector<StinkyInstruction*>& getUsers() const
        {
            return users;
        }
        std::vector<StinkyInstruction*>& getUsers()
        {
            return users;
        }

        /// Get instructions that define the operands used by this instruction (def-use chain)
        const std::vector<StinkyInstruction*>& getSources() const
        {
            return sources;
        }
        std::vector<StinkyInstruction*>& getSources()
        {
            return sources;
        }

        /// Link this instruction (user) to the instruction that defines an operand.
        /// Adds def to sources and adds this to def's users. No-op if def is null.
        void addOperandDef(StinkyInstruction* def)
        {
            if(def == nullptr)
                return;
            sources.push_back(def);
            def->getUsers().push_back(this);
        }

        /// Set sources[i] to def.
        /// removes this from old def's users (if any), adds this to def's users if def != nullptr.
        void setOperandDef(size_t i, StinkyInstruction* def)
        {
            assert(i < sources.size() && "sources index out of bounds");
            StinkyInstruction* oldDef = sources[i];
            if(oldDef != nullptr)
            {
                auto& u = oldDef->getUsers();
                u.erase(std::remove(u.begin(), u.end(), this), u.end());
            }
            sources[i] = def;
            if(def != nullptr)
                def->getUsers().push_back(this);
        }

        /// Get destination register by index
        const StinkyRegister& getDestReg(size_t idx) const
        {
            return destRegs.at(idx);
        }

        /// Get source register by index
        const StinkyRegister& getSrcReg(size_t idx) const
        {
            return srcRegs.at(idx);
        }

        uint16_t getISAOpcode() const
        {
            return hwInstDesc->isaOpcode;
        }

        uint16_t getUnifiedOpcode() const
        {
            return hwInstDesc->unifiedOpcode;
        }

        const HwInstDesc* getHwInstDesc() const
        {
            return hwInstDesc;
        }

        void updateHwInstDesc(const HwInstDesc* newDesc)
        {
            if(newDesc)
            {
                hwInstDesc    = newDesc;
                issueCycles   = newDesc->issue;
                latencyCycles = newDesc->latency;
            }
        }

        bool is(InstFlag flag) const
        {
            return hwInstDesc->has(flag);
        }

        template <class ModType>
        void addModifier(const ModType& mod)
        {
            static_assert(std::is_base_of_v<Modifier, ModType>,
                          "ModType must derive from Modifier");
            modifiers.push_back(std::make_unique<ModType>(mod));
        }

        template <class ModType>
        const ModType* getModifier() const
        {
            for(const std::unique_ptr<Modifier>& mod : modifiers)
                if(mod->getType() == ModType::Type)
                    return static_cast<const ModType*>(mod.get());
            return nullptr;
        }

        template <class ModType>
        ModType* getModifier()
        {
            for(std::unique_ptr<Modifier>& mod : modifiers)
                if(mod->getType() == ModType::Type)
                    return static_cast<ModType*>(mod.get());
            return nullptr;
        }

        const std::vector<std::unique_ptr<Modifier>>& getModifiers() const
        {
            return modifiers;
        }

        void dump(std::ostream& out) const override
        {
            dump(out, true);
        }

        void dump(bool printDetails = false, const std::string& prefix = "") const;
        void dump(std::ostream&      out,
                  bool               printDetails = false,
                  const std::string& prefix       = "") const;
        void dump() const;

        // TODO: Review the algorithm and usage.
        //
        // These methods automatically maintain use-def chains when operands change.
        // Following LLVM/MLIR design: mutation methods are on the Instruction itself,
        // not on the builder.
        //
        // Usage:
        //   builder.setInsertionPoint(bb);  // or construct with desired block
        //   auto* inst = builder.create(mcid);
        //   inst->setSrcRegs({v1, v2});   // Automatically links to v1/v2 definitions
        //   inst->setDestRegs({v0});      // Automatically tracks v0 users
        //
        // Performance: O(R) where R = number of registers (typically 2-4)
        //              Much faster than O(N) full buildUseDefChain() scan

        /// Set source registers and automatically update use-def chains.
        /// Removes old use-def links and establishes new ones.
        void setSrcRegs(const std::vector<StinkyRegister>& srcRegs);

        /// Set destination registers and automatically update use-def chains.
        /// Removes old use-def links and establishes new ones for users.
        void setDestRegs(const std::vector<StinkyRegister>& destRegs);

        /// Add a single source register and update use-def chain.
        /// WARNING: O(N) backward scan. TODO: Review usage and re-design if needed.
        void addSrcRegAndUpdateUD(const StinkyRegister& srcReg);

        /// Add a single destination register and update use-def chain.
        /// WARNING: O(N) forward scan. TODO: Review usage and re-design if needed.
        void addDestRegAndUpdateUD(const StinkyRegister& destReg);

        /// Set a specific source register by index (for pattern rewriting).
        /// Note: Does NOT update use-def chains automatically - caller must rebuild if needed.
        void setSrcReg(size_t idx, const StinkyRegister& srcReg)
        {
            srcRegs.at(idx) = srcReg;
        }

        /// Set a specific destination register by index (for pattern rewriting).
        /// Note: Does NOT update use-def chains automatically - caller must rebuild if needed.
        void setDestReg(size_t idx, const StinkyRegister& destReg)
        {
            destRegs.at(idx) = destReg;
        }

        /// Resize source registers array (for pattern rewriting).
        void resizeSrcRegs(size_t size)
        {
            srcRegs.resize(size);
        }

        /// Resize destination registers array (for pattern rewriting).
        void resizeDestRegs(size_t size)
        {
            destRegs.resize(size);
        }

        /**
         * @brief Clone this instruction (deep copy)
         *
         * Creates a new instruction with the same descriptor, registers, and modifiers.
         *
         * Notes:
         * - Modifiers are deep copied using copy constructors (works for POD modifiers)
         * - users/sources are NOT copied (dependency tracking should be rebuilt)
         *
         * This is needed because StinkyInstruction inherits from IntrusiveListNode
         * which has a deleted copy constructor.
         *
         * @return New instruction (caller owns the pointer)
         */
        StinkyInstruction* clone() const override
        {
            StinkyInstruction* cloned = IRBase::createIR<StinkyInstruction>(hwInstDesc);

            // Copy register lists
            cloned->destRegs = destRegs;
            cloned->srcRegs  = srcRegs;

            // Copy issue/latency cycles
            cloned->issueCycles   = issueCycles;
            cloned->latencyCycles = latencyCycles;

            // Deep copy modifiers via virtual clone() (TypedModifier implements it per type).
            for(const auto& mod : modifiers)
            {
                cloned->modifiers.push_back(mod->clone());
            }

            // Note: users/sources are intentionally NOT copied
            // These are dependency tracking and should be rebuilt if needed

            return cloned;
        }

        //----------------------------------------------------------------------
        // Use-Def Chain Maintenance API (Public for passes that delete instructions)
        //----------------------------------------------------------------------
        /// Remove this instruction from all its operand-defs' user lists.
        /// Must be called before deleting an instruction to prevent dangling pointers.
        void unlinkFromSources();

        /// Remove this instruction from all its users' operand-def lists.
        /// Must be called before deleting an instruction to prevent dangling pointers.
        void unlinkFromUsers();

    private:
        //----------------------------------------------------------------------
        // Use-Def Chain Maintenance Helpers (Private)
        //----------------------------------------------------------------------
        /// Update this->sources by scanning backwards for register definitions.
        /// Only updates for the registers currently in srcRegs.
        void updateSourcesForInst();

        /// Update use-def chains for all instructions that use the registers
        /// defined by this instruction. Scans forward to find users.
        void updateUsersForInst();

        // Temporary: Allow legacy IR creation code to access private fields
        // TODO: Migrate these to use setSrcRegs/setDestRegs, then remove
        friend class ToStinkyAsmPass;

    public:
        static bool classof(const IRBase* ir)
        {
            return ir->getType() == IRType::StinkyTofu;
        }
    };

    /// Builder for assembly-level IR. Use for creation only; mutation and
    /// deletion are on the instruction (setSrcRegs, setDestRegs, erase).
    /// In passes, create StinkyInstruction only through this builder.
    class AsmIRBuilder : public IRBuilder
    {
    public:
        const GfxArchID arch;

        StinkyInstruction* create(const HwInstDesc* mcid, IRBase* insertBefore = nullptr)
        {
            assert(mcid != nullptr
                   && "Cannot create instruction with null descriptor - instruction not supported "
                      "on this architecture");

            if(insertBefore == nullptr)
                return createIR<StinkyInstruction>(mcid);

            return createIR<StinkyInstruction>(insertBefore, mcid);
        }

        /// Creates a LABEL instruction. TODO: remove when basic-block labels are supported.
        StinkyInstruction* createLabel(const std::string& label, uint16_t alignment = 1);

        /// PHI instruction properties:
        /// - Must appear at the beginning of a basic block
        /// - Defines a single register (DWORD only)
        /// - One incoming value per predecessor
        ///   * For a basic block B:
        ///     . If B has N predecessors, each PHI in B must have exactly N incoming
        ///       pairs, and each predecessor appears exactly once
        /// - Operand is nullptr if undefined on that edge
        ///
        /// PHI instruction format:
        /// def = PHI(pred0_def, pred1_def, ..., predN_def)

        /// Creates and inserts a PHI instruction at the beginning of the block.
        /// The PHI defines one DWORD register and has one placeholder srcReg per
        /// predecessor. sources and users are NOT initialized — the caller
        /// (PhiPlacement / BuildDefUseChain) is responsible for linking chains.
        ///
        /// If \p insertPt is non-null, the PHI is inserted before \p insertPt
        /// instead of before bb->begin(). This allows callers to build PHIs in
        /// a stable order by passing a fixed anchor point.
        StinkyInstruction* createPhi(RegType type, unsigned regIdx, IRBase* insertPt = nullptr);

        AsmIRBuilder(BasicBlock& bb, GfxArchID arch)
            : IRBuilder(bb)
            , arch(arch)
        {
        }
    };

    const HwInstDesc* getMCIDByUOp(GFX unifiedOpcode, GfxArchID arch);
    const HwInstDesc* getMCIDByIsaOp(IsaOpcode isaOpcode, GfxArchID arch);
    uint16_t          getMnemonicToIsaOpcode(const std::string& mnemonic, GfxArchID arch);

    // Processing StinkyTofu IR.
    class StinkyInstPass : public Pass
    {
    };

    inline StinkyInstruction& getStinkyInst(IRList::iterator it)
    {
        return *cast<StinkyInstruction>(it.getNodePtr());
    }

    inline StinkyInstruction& getStinkyInst(IRList::reverse_iterator it)
    {
        return *cast<StinkyInstruction>(it.getNodePtr());
    }

    //----------------------------------------------------------------------
    // StinkyInstruction utilities
    //----------------------------------------------------------------------
    uint32_t getBytesPerGlobalLoad(const StinkyInstruction& inst);

    inline bool isMUBUFLoad(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_MUBUFLoad);
    }

    inline bool isMUBUFStore(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_MUBUFStore);
    }

    inline bool isFLATLoad(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_FLATLoad);
    }

    inline bool isFLATStore(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_FLATStore);
    }

    inline bool isGLOBALLoad(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_GLOBALLoad);
    }

    inline bool isGLOBALStore(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_GLOBALStore);
    }

    inline bool isSMemLoad(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_SMemLoad);
    }

    inline bool isSMemStore(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_SMemStore);
    }

    /// Check if instruction is a pseudo instruction (LABEL or PHI) that should be
    /// skipped for def-use chain processing of "real" instructions.
    inline bool isPseudoInst(const StinkyInstruction* inst)
    {
        return inst->getUnifiedOpcode() == GFX::LABEL || inst->getUnifiedOpcode() == GFX::PHI;
    }

    inline bool isGlobalMemLoad(const StinkyInstruction& inst)
    {
        return isMUBUFLoad(inst) || isFLATLoad(inst) || isGLOBALLoad(inst) || isSMemLoad(inst);
    }

    inline bool isGlobalMemAtomic(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_SMemAtomic) || inst.is(InstFlag::IF_MUBUFAtomic)
               || inst.is(InstFlag::IF_FLATAtomic);
    }

    inline bool isGlobalMemStore(const StinkyInstruction& inst)
    {
        return isSMemStore(inst) || isFLATStore(inst) || isMUBUFStore(inst) || isGLOBALStore(inst);
    }

    inline bool isTensorLoad(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_TENSORLoadToLds);
    }

    inline bool isDSRead(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_DSRead);
    }

    inline bool isDSWrite(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_DSStore);
    }

    inline bool isBarrier(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_Barrier);
    }

    inline bool isBranch(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_Branch);
    }

    inline bool isConditionalBranch(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_ConditionalBranch);
    }

    inline bool isUnconditionalBranch(const StinkyInstruction& inst)
    {
        return isBranch(inst) && !isConditionalBranch(inst);
    }

    // Get the branch target label name from a branch instruction.
    // Branch instructions store their target as the first source register (LiteralString type).
    inline std::string getBranchTarget(const StinkyInstruction& inst)
    {
        assert(isBranch(inst) && "Instruction must be a branch");
        assert(!inst.getSrcRegs().empty()
               && "Branch instruction must have at least one source register");

        const StinkyRegister& targetReg = inst.getSrcRegs()[0];
        assert(targetReg.dataType == StinkyRegister::Type::LiteralString
               && "Branch target must be a LiteralString");

        return targetReg.getLiteralString();
    }

    inline bool isWaitCnt(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_WaitCnt);
    }

    inline bool isMFMA(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_MFMA);
    }

    inline bool isSMFMA(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_SMFMA);
    }

    inline bool isWMMA(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_WMMA);
    }

    inline bool isSWMMA(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_SWMMA);
    }

    inline bool isMXWMMA(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_MXWMMA);
    }

    inline bool isHasSideEffect(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_HasSideEffect);
    }

    /// Determines if an instruction must be preserved and cannot be eliminated.
    ///
    /// This is a comprehensive check that covers all instructions with observable effects,
    /// including memory operations, control flow, barriers, and instructions explicitly
    /// marked with side effects.
    ///
    /// This function is distinct from isHasSideEffect() which only checks the
    /// IF_HasSideEffect flag. This function performs a broader classification suitable
    /// for optimization passes like dead code elimination.
    ///
    /// @param inst The instruction to check
    /// @return true if the instruction must be preserved, false if it can be eliminated
    inline bool mustPreserveInstruction(const StinkyInstruction& inst)
    {
        // Memory operations (loads/stores)
        if(isGlobalMemLoad(inst) || isGlobalMemStore(inst))
            return true;

        if(isDSRead(inst) || isDSWrite(inst))
            return true;

        if(isGlobalMemAtomic(inst))
            return true;

        if(isTensorLoad(inst))
            return true;

        // Control flow
        if(isBranch(inst))
            return true;

        // Barriers and synchronization
        if(isBarrier(inst))
            return true;

        // Instructions explicitly marked with side effects
        if(isHasSideEffect(inst))
            return true;

        return false;
    }

    /// Check if instruction is a vector ALU instruction (v_*)
    /// Excludes transcendental instructions which are classified separately
    inline bool isVectorALU(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_VALU) && !inst.is(InstFlag::IF_Transcendental);
    }

    /// Check if instruction is a transcendental instruction
    /// Includes: v_s_*, v_exp_*, v_log_*, v_rcp_*, v_rsq_*, v_sqrt_*
    inline bool isTranscendental(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_Transcendental);
    }

    /// Check if instruction is a scalar ALU instruction (s_*)
    /// Excludes: control flow, memory operations, waitcnt, barrier, delay_alu
    inline bool isScalarALU(const StinkyInstruction& inst)
    {
        return inst.is(InstFlag::IF_SALU);
    }

} // namespace stinkytofu

// Hash specialization for StinkyRegister (required for std::unordered_map)
namespace std
{
    template <>
    struct hash<stinkytofu::StinkyRegister>
    {
        size_t operator()(const stinkytofu::StinkyRegister& sreg) const noexcept
        {
            return sreg.hash();
        }
    };
} // namespace std
