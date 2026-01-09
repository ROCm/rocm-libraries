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
#include <memory>
#include <vector>

#include "ir/asm/StinkyModifiers.hpp"
#include "isa/gfx/GfxIsa.hpp"
#include "stinkytofu.hpp"
#include "support/Casting.hpp"

namespace stinkytofu
{
#define GET_ISAINFO_UNIFIED_OPCODES
#include "hardware/gfxIsa.inc"

    // Register type enumeration for different register classes
    enum class RegType
    {
#define REGISTER_TYPE(ENUM, STR, DESC) ENUM,
#include "ir/asm/RegisterType.def"
    };

    // Helper function to convert string to RegType
    inline RegType stringToRegType(const std::string& str)
    {
#define REGISTER_TYPE(ENUM, STR, DESC) \
    if(str == STR)                     \
        return RegType::ENUM;
#include "ir/asm/RegisterType.def"
        assert(false && "Invalid register type");
        return RegType::UNKNOWN;
    }

    // Helper function to convert RegType to string
    inline std::string regTypeToString(RegType type)
    {
        switch(type)
        {
#define REGISTER_TYPE(ENUM, STR, DESC) \
    case RegType::ENUM:                \
        return STR;
#include "ir/asm/RegisterType.def"
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
            struct
            {
                RegType  type;
                unsigned idx;
                unsigned num;
                // Virtual register flag for template-based code generation
                // Only meaningful when dataType == Type::Register
                // When true, this register needs offset remapping before use
                bool isVirtual;
            } reg;

            // For LiteralInt type
            int literalInt;

            // For LiteralDouble type
            double literalDouble;
        };

        // For LiteralString type - kept separate as std::string is non-trivial
        std::string literalValue;

        // define ordering for std::map key
        bool operator<(const StinkyRegister& other) const noexcept
        {
            if(dataType != other.dataType)
                return dataType < other.dataType;
            if(reg.type != other.reg.type)
                return reg.type < other.reg.type;
            if(reg.idx != other.reg.idx)
                return reg.idx < other.reg.idx;
            return reg.num < other.reg.num;
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

        StinkyRegister(RegType type, int regIdx, int regNum)
            : dataType(Type::Register)
            , reg{type, static_cast<unsigned>(regIdx), static_cast<unsigned>(regNum), false}
        {
        }

        // Constructor accepting string for backward compatibility
        StinkyRegister(const std::string& typeStr, int regIdx, int regNum)
            : dataType(Type::Register)
            , reg{stringToRegType(typeStr),
                  static_cast<unsigned>(regIdx),
                  static_cast<unsigned>(regNum),
                  false}
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
            , reg{RegType::UNKNOWN, 0, 0, false}
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

        static StinkyRegister getSCCRegister()
        {
            // SCC register is a special register, it is not a physical register.
            return StinkyRegister(RegType::SCC, 0, 1);
        }

        static StinkyRegister getBarrierRegister()
        {
            // Barrier register is a special register, it is not a physical register.
            return StinkyRegister(RegType::BARRIER, 0, 1);
        }

        static StinkyRegister getDSWriteRegister()
        {
            // DS write register is a special register, it is not a physical register.
            return StinkyRegister(RegType::DS_WRITE, 0, 1);
        }

        static StinkyRegister getTensorLoadRegister()
        {
            // Tensor load register is a special register, it is not a physical register.
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
                h ^= std::hash<unsigned>{}(reg.idx) << 2;
                h ^= std::hash<unsigned>{}(reg.num) << 3;
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

    // Represents a single assembly instruction.
    struct StinkyInstruction : public IRBase
    {
        // Instructions that use this instruction's output.
        std::vector<StinkyInstruction*> users;
        std::vector<StinkyInstruction*> sources;

        int issueCycles;
        int latencyCycles;

    private:
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

    public:
        StinkyInstruction(const HwInstDesc* mcid)
            : IRBase(IRType::StinkyTofu)
            , hwInstDesc(mcid)
            , issueCycles(mcid->issue)
            , latencyCycles(mcid->latency)
        {
        }

        //----------------------------------------------------------------------
        // Register Operand Access (LLVM-style getters)
        //----------------------------------------------------------------------
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
            constexpr Modifier::Type modType = getModifierType<ModType>();
            for(const std::unique_ptr<Modifier>& mod : modifiers)
                if(mod->getType() == modType)
                    return static_cast<const ModType*>(mod.get());
            return nullptr;
        }

        template <class ModType>
        ModType* getModifier()
        {
            constexpr Modifier::Type modType = getModifierType<ModType>();
            for(std::unique_ptr<Modifier>& mod : modifiers)
                if(mod->getType() == modType)
                    return static_cast<ModType*>(mod.get());
            return nullptr;
        }

        void dump(std::ostream& out) const override
        {
            dump(out, true);
        }

        void dump(bool printDetails = false, const std::string& prefix = "") const;
        void dump(std::ostream&      out,
                  bool               printDetails = false,
                  const std::string& prefix       = "") const;

        //----------------------------------------------------------------------
        // Automatic Use-Def Chain Maintenance (LLVM-style)
        //----------------------------------------------------------------------
        // These methods automatically maintain use-def chains when operands change.
        // Following LLVM/MLIR design: mutation methods are on the Instruction itself,
        // not on the builder.
        //
        // Usage:
        //   auto* inst = builder.createStinkyInstBefore(pos, mcid);
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
        void addSrcReg(const StinkyRegister& srcReg);

        /// Add a single destination register and update use-def chain.
        void addDestReg(const StinkyRegister& destReg);

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
        StinkyInstruction* clone() const
        {
            StinkyInstruction* cloned = new StinkyInstruction(hwInstDesc);

            // Copy register lists
            cloned->destRegs = destRegs;
            cloned->srcRegs  = srcRegs;

            // Copy issue/latency cycles
            cloned->issueCycles   = issueCycles;
            cloned->latencyCycles = latencyCycles;

            // Deep copy modifiers - this works because all current modifier structs
            // are copyable (POD types, std::vector, std::string)
            for(const auto& mod : modifiers)
            {
                // We can't use mod->clone() without adding virtual clone() to Modifier base
                // Instead, we rely on the fact that all modifiers are copyable
                // This requires type-specific handling
                switch(mod->getType())
                {
                case Modifier::Type::DS:
                    cloned->modifiers.push_back(
                        std::make_unique<DSModifiers>(*static_cast<DSModifiers*>(mod.get())));
                    break;
                case Modifier::Type::FLAT:
                    cloned->modifiers.push_back(
                        std::make_unique<FLATModifiers>(*static_cast<FLATModifiers*>(mod.get())));
                    break;
                case Modifier::Type::GLOBAL:
                    cloned->modifiers.push_back(std::make_unique<GLOBALModifiers>(
                        *static_cast<GLOBALModifiers*>(mod.get())));
                    break;
                case Modifier::Type::MUBUF:
                    cloned->modifiers.push_back(
                        std::make_unique<MUBUFModifiers>(*static_cast<MUBUFModifiers*>(mod.get())));
                    break;
                case Modifier::Type::SMEM:
                    cloned->modifiers.push_back(
                        std::make_unique<SMEMModifiers>(*static_cast<SMEMModifiers*>(mod.get())));
                    break;
                case Modifier::Type::SDWA:
                    cloned->modifiers.push_back(
                        std::make_unique<SDWAModifiers>(*static_cast<SDWAModifiers*>(mod.get())));
                    break;
                case Modifier::Type::DPP:
                    cloned->modifiers.push_back(
                        std::make_unique<DPPModifiers>(*static_cast<DPPModifiers*>(mod.get())));
                    break;
                case Modifier::Type::VOP3:
                    cloned->modifiers.push_back(
                        std::make_unique<VOP3Modifiers>(*static_cast<VOP3Modifiers*>(mod.get())));
                    break;
                case Modifier::Type::VOP3P:
                    cloned->modifiers.push_back(
                        std::make_unique<VOP3PModifiers>(*static_cast<VOP3PModifiers*>(mod.get())));
                    break;
                case Modifier::Type::TRUE16:
                    cloned->modifiers.push_back(std::make_unique<True16Modifiers>(
                        *static_cast<True16Modifiers*>(mod.get())));
                    break;
                case Modifier::Type::EXEC:
                    cloned->modifiers.push_back(
                        std::make_unique<EXEC>(*static_cast<EXEC*>(mod.get())));
                    break;
                case Modifier::Type::VCC:
                    cloned->modifiers.push_back(
                        std::make_unique<VCC>(*static_cast<VCC*>(mod.get())));
                    break;
                case Modifier::Type::SWAITCNT_DATA:
                    cloned->modifiers.push_back(
                        std::make_unique<SWaitCntData>(*static_cast<SWaitCntData*>(mod.get())));
                    break;
                case Modifier::Type::SWAITTENSORCNT_DATA:
                    cloned->modifiers.push_back(std::make_unique<SWaitTensorCntData>(
                        *static_cast<SWaitTensorCntData*>(mod.get())));
                    break;
                case Modifier::Type::SDELAYALU_DATA:
                    cloned->modifiers.push_back(
                        std::make_unique<SDelayAluData>(*static_cast<SDelayAluData*>(mod.get())));
                    break;
                case Modifier::Type::LABEL_NAME:
                    cloned->modifiers.push_back(
                        std::make_unique<LabelData>(*static_cast<LabelData*>(mod.get())));
                    break;
                case Modifier::Type::COMMENT:
                    cloned->modifiers.push_back(
                        std::make_unique<CommentData>(*static_cast<CommentData*>(mod.get())));
                    break;
                }
            }

            // Note: users/sources are intentionally NOT copied
            // These are dependency tracking and should be rebuilt if needed

            return cloned;
        }

        /**
         * @brief Remap virtual registers to physical registers
         *
         * This method applies register offsets to all virtual registers in the instruction.
         * Used when instantiating instruction templates with actual register allocations.
         *
         * @param vgprOffset Offset to add to VGPR virtual registers
         * @param sgprOffset Offset to add to SGPR virtual registers
         *
         * Example:
         *   // Template instruction: v_add_u32 v0, v1, v2 (using virtual registers)
         *   StinkyInstruction* inst = ...;
         *   inst->remapRegisters(10, 0);  // Remaps to: v_add_u32 v10, v11, v12
         */
        void remapRegisters(int vgprOffset, int sgprOffset)
        {
            // Remap destination registers
            for(auto& reg : destRegs)
            {
                if(!reg.isVirtualRegister())
                    continue;

                if(reg.reg.type == RegType::V)
                    reg = reg.withOffset(vgprOffset);
                else if(reg.reg.type == RegType::S)
                    reg = reg.withOffset(sgprOffset);
            }

            // Remap source registers
            for(auto& reg : srcRegs)
            {
                if(!reg.isVirtualRegister())
                    continue;

                if(reg.reg.type == RegType::V)
                    reg = reg.withOffset(vgprOffset);
                else if(reg.reg.type == RegType::S)
                    reg = reg.withOffset(sgprOffset);
            }
        }

    private:
        //----------------------------------------------------------------------
        // Use-Def Chain Maintenance Helpers
        //----------------------------------------------------------------------
        /// Update this->sources by scanning backwards for register definitions.
        /// Only updates for the registers currently in srcRegs.
        void updateSourcesForInst();

        /// Update use-def chains for all instructions that use the registers
        /// defined by this instruction. Scans forward to find users.
        void updateUsersForInst();

        /// Remove this instruction from all its sources' user lists.
        void unlinkFromSources();

        /// Remove this instruction from all its users' source lists.
        void unlinkFromUsers();

        // Allow builder to call private unlink methods during erase()
        friend class StinkyInstIRBuilder;

        // Temporary: Allow legacy IR creation code to access private fields
        // TODO: Migrate these to use setSrcRegs/setDestRegs, then remove
        friend class ToStinkyAsmPass;

    public:
        static bool classof(const IRBase* ir)
        {
            return ir->getType() == IRType::StinkyTofu;
        }
    };

    // Builder for StinkyTofu IR (LLVM-style).
    //
    // Following LLVM/MLIR design principles:
    // - Builder is for CREATION and DELETION only
    // - Mutation methods (setSrcRegs, setDestRegs) are on StinkyInstruction itself
    //
    // All creation and deletion of StinkyTofu IR in **Passes** should be
    // handled **exclusively** through this class.
    //
    // Note that PassContext owns the IRList, PassContext will delete IRBase
    // when PassContext is destructed.
    //
    // Usage:
    //   auto builder = StinkyInstIRBuilder(bb.getIR(), arch);
    //   auto* inst = builder.createStinkyInstBefore(pos, mcid);
    //   inst->setSrcRegs({v1, v2});   // Mutation is on the instruction
    //   inst->setDestRegs({v0});
    //   ...
    //   builder.erase(inst);  // Deletion through builder
    class StinkyInstIRBuilder : public IRBuilder
    {
    public:
        static IRBuilder::ID ID;

        const GfxArchID arch;

        StinkyInstruction* createStinkyInstBefore(IRList::iterator pos, const HwInstDesc* mcid)
        {
            assert(irlist != nullptr && "StinkyInstIRBuilder not initialized");
            assert(mcid != nullptr
                   && "Cannot create instruction with null descriptor - instruction not supported "
                      "on this architecture");

            StinkyInstruction* stinkyInst = new StinkyInstruction(mcid);
            irlist->insert(pos, stinkyInst);
            return stinkyInst;
        }

        // Create a StinkyInstruction with GFX::LABEL opcode.
        // Note this is a temporary workaround for the fact that stinkytofu
        // doesn't support basicblocks concept.
        //
        // TODO: Remove this once basicblocks concept is supported.
        StinkyInstruction* createStinkyLabel(IRList::iterator pos, const std::string& label);

        void erase(StinkyInstruction* stinkyInst);

        StinkyInstIRBuilder(IRList& irlist, const GfxArchID& arch)
            : IRBuilder(irlist)
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

    // Helper function to get the IR list from a Function's entry BasicBlock.
    // This is useful for passes that work on flat IR lists.
    // Note: Assumes the Function has at least one BasicBlock.
    inline IRList& getEntryIR(Function& func)
    {
        assert(func.getEntryBlock() && "Function must have an entry BasicBlock");
        return func.getEntryBlock()->getIR();
    }

    inline const IRList& getEntryIR(const Function& func)
    {
        assert(func.getEntryBlock() && "Function must have an entry BasicBlock");
        return func.getEntryBlock()->getIR();
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
