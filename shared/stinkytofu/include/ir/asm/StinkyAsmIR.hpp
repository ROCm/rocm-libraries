/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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

        StinkyRegister(RegType type, int regIdx, int regNum)
            : dataType(Type::Register)
            , reg{type, static_cast<unsigned>(regIdx), static_cast<unsigned>(regNum)}
        {
        }

        // Constructor accepting string for backward compatibility
        StinkyRegister(const std::string& typeStr, int regIdx, int regNum)
            : dataType(Type::Register)
            , reg{stringToRegType(typeStr),
                  static_cast<unsigned>(regIdx),
                  static_cast<unsigned>(regNum)}
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
            , reg{RegType::UNKNOWN, 0, 0}
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
    };

    // Represents a single assembly instruction.
    struct StinkyInstruction : public IRBase
    {
        // Instructions that use this instruction's output.
        std::vector<StinkyInstruction*> users;
        std::vector<StinkyInstruction*> sources;

        // TODO: CMP instructions imply modify SCC register, but it's not tracked in
        //       the frontend (e.g. SCmpKEQU32 struct), a workaround is to track it
        //       here.
        std::vector<StinkyRegister> destRegs;
        std::vector<StinkyRegister> srcRegs;

        int issueCycles;
        int latencyCycles;

    private:
        const HwInstDesc* hwInstDesc;

        // Modifiers are extra bits/fields in the instruction encoding that
        // tweak how the operation is performed, without needing a completely
        // different opcode.
        std::vector<std::unique_ptr<Modifier>> modifiers;

    public:
        StinkyInstruction(const HwInstDesc* mcid)
            : IRBase(IRType::StinkyTofu)
            , hwInstDesc(mcid)
            , issueCycles(mcid->issue)
            , latencyCycles(mcid->latency)
        {
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

    public:
        static bool classof(const IRBase* ir)
        {
            return ir->getType() == IRType::StinkyTofu;
        }
    };

    // Builder for StinkyTofu IR.
    //
    // All creation and deletion of StinkyTofu IR in **Passes** should be
    // handled **exclusively** through this class.
    //
    // Note that PassContext owns the IRList, PassContext will delete IRBase
    // when PassContext is destructed.
    //
    // It provides methods to:
    //   1. create StinkyInstructions.
    //   2. erase StinkyInstructions from the IRList, and delete that
    //      StinkyInst because it is no longer needed.
    class StinkyInstIRBuilder : public IRBuilder
    {
    public:
        static IRBuilder::ID ID;

        const GfxArchID arch;

        StinkyInstruction* createStinkyInstBefore(IRList::iterator pos, const HwInstDesc* mcid)
        {
            assert(irlist != nullptr && "StinkyInstIRBuilder not initialized");

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
        assert(!inst.srcRegs.empty()
               && "Branch instruction must have at least one source register");

        const StinkyRegister& targetReg = inst.srcRegs[0];
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
} // namespace stinkytofu
