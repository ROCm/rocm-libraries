/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include <memory>
#include <string>

#include "WaitCount.hpp"

#include "../Context_fwd.hpp"
#include "../InstructionValues/Register_fwd.hpp"
#include "../Scheduling/Scheduler_fwd.hpp"
#include "../Utilities/Settings_fwd.hpp"

namespace rocRoller
{
    struct Instruction
    {
        enum
        {
            MaxDstRegisters = 2,
            MaxSrcRegisters = 4,
            MaxModifiers    = 10,
            MaxAllocations  = 4
        };

        Instruction(std::string const&                        opcode,
                    std::initializer_list<Register::ValuePtr> dst,
                    std::initializer_list<Register::ValuePtr> src,
                    std::initializer_list<std::string>        modifiers,
                    std::string const&                        comment);

        Instruction();

        static Generator<std::string> EscapeComment(std::string comment, int indent = 0);

        static Instruction Allocate(Register::ValuePtr reg);
        static Instruction Allocate(std::shared_ptr<Register::Allocation> reg);
        static Instruction
            Allocate(std::initializer_list<std::shared_ptr<Register::Allocation>> regs);

        static Instruction Directive(std::string const& directive);
        static Instruction Directive(std::string const& directive, std::string const& comment);

        static Instruction Comment(std::string const& comment);

        static Instruction Warning(std::string const& warning);

        static Instruction Nop();
        static Instruction Nop(int nopCount);
        static Instruction Nop(std::string const& comment);
        static Instruction Nop(int nopCount, std::string const& comment);

        static Instruction Label(std::string const& name);
        static Instruction Label(std::string&& name);

        static Instruction Label(std::string const& name, std::string const& comment);
        static Instruction Label(std::string&& name, std::string const& comment);

        static Instruction Label(Register::ValuePtr label);
        static Instruction Label(Register::ValuePtr label, std::string const& comment);

        static Instruction InoutInstruction(std::string const&                        opcode,
                                            std::initializer_list<Register::ValuePtr> inout,
                                            std::initializer_list<std::string>        modifiers,
                                            std::string const&                        comment);

        static Instruction Wait(WaitCount const& wait);
        static Instruction Wait(WaitCount&& wait);

        static Instruction Lock(Scheduling::Dependency const& dependency, std::string comment);
        static Instruction Unlock(std::string comment);

        std::array<Register::ValuePtr, MaxDstRegisters> const& getDsts() const;
        std::array<Register::ValuePtr, MaxSrcRegisters> const& getSrcs() const;

        bool           hasRegisters() const;
        constexpr bool readsSpecialRegisters() const;
        WaitCount      getWaitCount() const;

        /**
         * @brief Returns |currentSrc n previousDest| > 0 or |currentDest n previousDest| > 0
         *
         * @param previousDest Destination registers of previous instruction.
         * @return Whether this instructions registers intersect with a past instructions destination registers.
         */
        bool isAfterWriteDependency(
            std::array<Register::ValuePtr, Instruction::MaxDstRegisters> const& previousDest) const;

        void        toStream(std::ostream&, LogLevel level) const;
        std::string toString(LogLevel level) const;

        constexpr int          getLockValue() const;
        Scheduling::Dependency getDependency() const;

        std::string const&                                 getOpCode() const;
        std::array<std::string, Instruction::MaxModifiers> getModifiers() const;

        constexpr int getNopCount() const
        {
            return m_nopCount;
        }

        Instruction lock(Scheduling::Dependency const& depedency, std::string comment);
        Instruction unlock(std::string comment);

        void addAllocation(std::shared_ptr<Register::Allocation> alloc);
        void addWaitCount(WaitCount const& wait);
        void addComment(std::string const& comment);
        void addWarning(std::string const& warning);
        void addNop();
        void addNop(int count);
        void setNopMin(int count);

        constexpr int nopCount() const;

        constexpr bool isCommentOnly() const;
        constexpr bool isLabel() const;
        std::string    getLabel() const;

        /**
         * How many instructions actually executed by the GPU are included?
         * This includes the main instruction, as well as any s_nop or s_waitcnt
         * instructions attached to it.
         */
        int numExecutedInstructions() const;

        void allocateNow();

        using AllocationArray = std::array<std::shared_ptr<Register::Allocation>, MaxAllocations>;
        AllocationArray allocations() const;

    private:
        /**
         * toString = preamble + functional + coda
         */
        void preambleString(std::ostream& oss, LogLevel level) const;
        void functionalString(std::ostream& oss, LogLevel level) const;
        void codaString(std::ostream& oss, LogLevel level) const;

        /**
         * A comment detailing allocations that happened when scheduling this instruction.
         */
        void allocationString(std::ostream& oss, LogLevel level) const;

        /**
         * Assembler directive(s), if this instruction contains an any.
         */
        void directiveString(std::ostream& oss, LogLevel level) const;

        /**
         * Just the main instruction.
         */
        void coreInstructionString(std::ostream& oss) const;

        /**
         * When this instruction is scheduled, perform this register allocation.
         */
        AllocationArray m_allocations;

        int m_nopCount = 0;

        std::string m_directive;

        std::vector<std::string> m_warnings;

        std::vector<std::string> m_comments;

        std::string m_label;

        WaitCount m_waitCount;

        std::string m_opcode;

        Scheduling::Dependency m_dependency = Scheduling::Dependency::None;

        std::array<Register::ValuePtr, MaxDstRegisters> m_inoutDsts;
        std::array<Register::ValuePtr, MaxDstRegisters> m_dst;
        std::array<Register::ValuePtr, MaxSrcRegisters> m_src;

        std::array<std::string, MaxModifiers> m_modifiers;

        bool m_operandsAreInout = false;
    };
}

#include "Instruction_impl.hpp"
